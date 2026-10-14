"""
routes/gmail_push.py

Gmail push notifications and the AI auto-responder.

- POST /api/v1/gmail/push: Pub/Sub push endpoint. Authenticated by Pub/Sub
  (OIDC token and/or verification token), not by a user JWT.
- Watch, settings and status endpoints for the logged-in user.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response, status
from fastapi.concurrency import run_in_threadpool

from config import settings
from initializer import db
from schemas.gmail import (
    AutoresponderSettingsRequest,
    GmailActivityItem,
    GmailAutomationStatus,
    GmailWatchStatus,
)
from services import gmail_push_service
from services.app_auth_service import get_current_user
from services.gmail_auth_service import AuthError
from services.gmail_push_service import (
    InvalidNotificationError,
    PushAuthError,
    PushNotConfiguredError,
)
from services.gmail_service import GmailApiError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/gmail", tags=["gmail"])


@router.post(
    "/push",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Gmail Pub/Sub push endpoint",
    description=(
        "Receives Gmail change notifications from Cloud Pub/Sub. New messages "
        "are discovered through the Gmail History API and processed in the "
        "background. Not for direct use."
    ),
)
async def gmail_push(
    request: Request,
    background_tasks: BackgroundTasks,
    token: Optional[str] = Query(default=None, description="Pub/Sub verification token"),
):
    """Authenticate the push, then sync the mailbox in the background."""
    try:
        body = await request.json()
    except ValueError:
        body = None
    subscription = body.get("subscription") if isinstance(body, dict) else None

    try:
        # OIDC verification may fetch Google's signing certificates.
        await run_in_threadpool(
            gmail_push_service.verify_push_request,
            request.headers.get("authorization"),
            token,
            subscription,
        )
    except PushNotConfiguredError as exc:
        logger.error("Rejected Gmail push: %s", exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Push notifications are not configured.") from exc
    except PushAuthError as exc:
        logger.warning("Rejected unauthenticated Gmail push: %s", exc)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid push credentials.") from exc

    try:
        email_address, history_id = gmail_push_service.parse_push_message(body)
    except InvalidNotificationError as exc:
        # Acknowledge: Pub/Sub would redeliver a malformed message forever.
        logger.warning("Dropped malformed Gmail notification: %s", exc)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    user_id = await run_in_threadpool(
        gmail_push_service.handle_notification, email_address, history_id
    )
    if user_id is not None:
        # Acknowledge quickly; the work (and its retries) does not depend on
        # this request, because the history pointer only moves after it.
        background_tasks.add_task(gmail_push_service.sync_mailbox, user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _automation_status(user_id: int) -> GmailAutomationStatus:
    state = db.find_gmail_sync_state(user_id)
    watch = None
    if state is not None:
        watch = GmailWatchStatus(
            status=state.watch_status,
            email_address=state.email_address,
            expires_at=state.watch_expiration,
            last_notification_at=state.last_notification_at,
            last_sync_at=state.last_sync_at,
            last_error=state.last_error,
        )
    activity = [
        GmailActivityItem(
            sender=row.sender,
            subject=row.subject,
            status=row.status,
            reply_sent=row.reply_message_id is not None,
            indexed=row.incoming_indexed,
            error=row.error,
            updated_at=row.updated_at,
        )
        for row in db.list_gmail_reply_states(user_id, limit=10)
    ]
    return GmailAutomationStatus(
        google_connected=db.find_google_credentials(user_id) is not None,
        push_configured=settings.gmail_push_configured,
        autoresponder_enabled=db.get_autoresponder_enabled(user_id),
        watch=watch,
        recent_activity=activity,
    )


@router.get(
    "/automation",
    response_model=GmailAutomationStatus,
    summary="Gmail notification and auto-responder status",
)
def automation_status(user=Depends(get_current_user)):
    """Return watch, sync, auto-responder setting and recent activity."""
    return _automation_status(user.id)


@router.put(
    "/autoresponder",
    response_model=GmailAutomationStatus,
    summary="Enable or disable automatic replies",
)
def set_autoresponder(req: AutoresponderSettingsRequest, user=Depends(get_current_user)):
    """Persist the auto-responder opt-in for the logged-in user."""
    db.set_autoresponder_enabled(user.id, req.enabled)
    return _automation_status(user.id)


@router.post(
    "/watch",
    response_model=GmailAutomationStatus,
    summary="Start or renew Gmail push notifications",
)
def start_watch(user=Depends(get_current_user)):
    """Register (or renew) the Gmail watch for the logged-in user's mailbox."""
    if db.find_google_credentials(user.id) is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Google account not linked. Connect Google first.")
    if not settings.gmail_push_configured:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Gmail push notifications are not configured on the server (see README).",
        )
    try:
        gmail_push_service.ensure_watch(user.id)
    except (GmailApiError, AuthError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Gmail watch failed: {exc}") from exc
    return _automation_status(user.id)


@router.delete(
    "/watch",
    response_model=GmailAutomationStatus,
    summary="Stop Gmail push notifications",
)
def stop_watch(user=Depends(get_current_user)):
    """Stop the Gmail watch for the logged-in user's mailbox."""
    if db.find_gmail_sync_state(user.id) is None:
        return _automation_status(user.id)
    try:
        gmail_push_service.stop_watch(user.id)
    except (GmailApiError, AuthError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Stopping Gmail watch failed: {exc}") from exc
    return _automation_status(user.id)
