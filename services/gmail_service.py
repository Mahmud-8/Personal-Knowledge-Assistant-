"""
services/gmail_service.py

Gmail API access for a user's connected Gmail account: sending replies,
push-notification watches, history sync, threads, and fetching messages.

The caller only supplies the application `user_id`; this module looks up
the stored Google credentials, refreshes the access token through the
existing `AuthService` when needed, and calls the Gmail API. Nothing in
here is exposed to the LLM.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import parseaddr
from typing import Any, Optional

from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import settings
from initializer import db
from loaders.base_email_loader import EmailLoaderError, EmailLoaderParseError
from loaders.gmail_loader import GmailLoader
from models import Email
from services.gmail_auth_service import AuthError, AuthService, TokenRefreshError

logger = logging.getLogger(__name__)

# Refresh a little before the stored expiry to avoid racing it.
EXPIRY_SKEW = timedelta(seconds=60)


class GmailSendError(Exception):
    """The Gmail API rejected or failed to send a message."""


class GmailApiError(Exception):
    """A Gmail API read/watch call failed."""


class HistoryExpiredError(GmailApiError):
    """The stored historyId is too old for history.list (HTTP 404)."""


class MessageNotFoundError(GmailApiError):
    """The message no longer exists (e.g. deleted before it was fetched)."""


class InvalidEmailError(GmailApiError):
    """The message was fetched but is not a valid email (e.g. no sender)."""


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _get_credentials(user_id: int) -> Credentials:
    """Build valid Google credentials for `user_id`, refreshing if expired.

    Raises `AuthError` (or its `TokenRefreshError` subclass) when the user
    has no linked Google account or the token cannot be refreshed.
    """
    row = db.find_google_credentials(user_id)
    if row is None:
        raise AuthError("No Google account is connected to this user.")

    access_token = row.access_token
    expiry = _utc(row.token_expiry)

    if expiry is not None and expiry - EXPIRY_SKEW <= datetime.now(timezone.utc):
        if not row.refresh_token:
            raise AuthError("Gmail access token expired and no refresh token is available.")

        logger.info("Refreshing expired Gmail token for user %d.", user_id)
        access_token, new_expiry = AuthService(db=db).refresh_access_token(row.refresh_token)
        expiry = _utc(new_expiry)
        db.save_google_credentials(
            user_id,
            google_id=row.google_id,
            access_token=access_token,
            refresh_token=row.refresh_token,
            token_expiry=expiry,
        )

    # No `scopes` here: refreshing keeps whatever scopes the user granted.
    return Credentials(
        token=access_token,
        refresh_token=row.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET.get_secret_value(),
    )


def send_email(
    user_id: int,
    *,
    to: str,
    subject: str,
    body: str,
    thread_id: Optional[str] = None,
    in_reply_to: Optional[str] = None,
) -> dict[str, Any]:
    """Send a plain-text email from `user_id`'s Gmail account.

    `thread_id` / `in_reply_to` (the original RFC Message-ID) make Gmail
    and other clients thread the message as a reply.

    Returns {"message_id", "thread_id", "sender"}.
    Raises `AuthError` on credential problems, `GmailSendError` on API failures.
    """
    creds = _get_credentials(user_id)

    try:
        service = build("gmail", "v1", credentials=creds, cache_discovery=False)

        # Resolve the sender before sending, so a failure here can never
        # leave an already-sent email reported as failed.
        sender = service.users().getProfile(userId="me").execute().get("emailAddress")

        message = EmailMessage()
        message["To"] = to
        message["Subject"] = subject
        if in_reply_to:
            message["In-Reply-To"] = in_reply_to
            message["References"] = in_reply_to
        message.set_content(body)

        payload: dict[str, Any] = {
            "raw": base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8"),
        }
        if thread_id:
            payload["threadId"] = thread_id

        result = service.users().messages().send(userId="me", body=payload).execute()
    except RefreshError as exc:
        # The client library retries once with a refreshed token on 401.
        raise TokenRefreshError(f"Gmail token refresh failed: {exc}") from exc
    except HttpError as exc:
        logger.error("Gmail send failed for user %d: %s", user_id, exc)
        raise GmailSendError(f"Gmail API error: {exc}") from exc

    logger.info("Sent Gmail message %s for user %d.", result.get("id"), user_id)
    return {
        "message_id": result.get("id"),
        "thread_id": result.get("threadId"),
        "sender": sender,
    }


# ----------------------------------------------------------------------
# Read / watch API
# ----------------------------------------------------------------------


def _gmail(user_id: int):
    """Build a Gmail API client for `user_id` with valid credentials."""
    return build("gmail", "v1", credentials=_get_credentials(user_id), cache_discovery=False)


def _execute(request, what: str):
    try:
        return request.execute()
    except RefreshError as exc:
        raise TokenRefreshError(f"Gmail token refresh failed: {exc}") from exc
    except HttpError as exc:
        raise GmailApiError(f"{what} failed: {exc}") from exc


def _ms_to_datetime(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)


def get_mailbox_profile(user_id: int) -> dict[str, Any]:
    """Return {"email_address", "history_id"} of the user's mailbox."""
    profile = _execute(_gmail(user_id).users().getProfile(userId="me"), "getProfile")
    return {
        "email_address": profile.get("emailAddress"),
        "history_id": int(profile["historyId"]) if profile.get("historyId") else None,
    }


def start_watch(user_id: int, *, topic_name: str, label_ids: list[str]) -> dict[str, Any]:
    """Call users.watch so Gmail publishes mailbox changes to `topic_name`.

    Calling it again on an active watch simply renews it. Returns
    {"email_address", "history_id", "expiration"}.
    """
    service = _gmail(user_id)
    body: dict[str, Any] = {"topicName": topic_name}
    if label_ids:
        body["labelIds"] = label_ids
        body["labelFilterBehavior"] = "INCLUDE"
    response = _execute(service.users().watch(userId="me", body=body), "users.watch")
    profile = _execute(service.users().getProfile(userId="me"), "getProfile")
    return {
        "email_address": profile.get("emailAddress"),
        "history_id": int(response["historyId"]),
        "expiration": _ms_to_datetime(response.get("expiration")),
    }


def stop_watch(user_id: int) -> None:
    """Stop push notifications for the user's mailbox."""
    _execute(_gmail(user_id).users().stop(userId="me"), "users.stop")


def list_history(user_id: int, start_history_id: int) -> tuple[list[str], Optional[int]]:
    """Return (IDs of messages added since `start_history_id`, latest historyId).

    Raises `HistoryExpiredError` when Gmail no longer has history that old.
    """
    service = _gmail(user_id)
    message_ids: list[str] = []
    seen: set[str] = set()
    latest: Optional[int] = None
    page_token: Optional[str] = None

    while True:
        request = service.users().history().list(
            userId="me",
            startHistoryId=str(start_history_id),
            historyTypes=["messageAdded"],
            pageToken=page_token,
        )
        try:
            response = request.execute()
        except RefreshError as exc:
            raise TokenRefreshError(f"Gmail token refresh failed: {exc}") from exc
        except HttpError as exc:
            if getattr(exc.resp, "status", None) == 404:
                raise HistoryExpiredError(
                    f"History {start_history_id} is no longer available."
                ) from exc
            raise GmailApiError(f"history.list failed: {exc}") from exc

        for record in response.get("history", []) or []:
            for added in record.get("messagesAdded", []) or []:
                message_id = (added.get("message") or {}).get("id")
                if message_id and message_id not in seen:
                    seen.add(message_id)
                    message_ids.append(message_id)

        if response.get("historyId"):
            latest = int(response["historyId"])
        page_token = response.get("nextPageToken")
        if not page_token:
            return message_ids, latest


def list_message_ids_since(user_id: int, after: datetime, max_results: int = 100) -> list[str]:
    """IDs of messages received after `after` (used to recover from expired history)."""
    request = _gmail(user_id).users().messages().list(
        userId="me",
        q=f"after:{int(after.timestamp())}",
        maxResults=max_results,
    )
    response = _execute(request, "messages.list")
    return [m["id"] for m in response.get("messages", []) or []]


def get_thread_messages(user_id: int, thread_id: str) -> list[dict[str, Any]]:
    """Return the thread's messages as {"id", "label_ids", "internal_date", "sender"}."""
    request = _gmail(user_id).users().threads().get(
        userId="me", id=thread_id, format="metadata", metadataHeaders=["From"],
    )
    thread = _execute(request, "threads.get")
    messages = []
    for message in thread.get("messages", []) or []:
        headers = {
            h["name"].lower(): h["value"]
            for h in (message.get("payload") or {}).get("headers", []) or []
        }
        messages.append({
            "id": message.get("id"),
            "label_ids": list(message.get("labelIds") or []),
            "internal_date": _ms_to_datetime(message.get("internalDate")),
            "sender": parseaddr(headers.get("from", ""))[1].lower(),
        })
    return messages


def fetch_email(user_id: int, message_id: str) -> Email:
    """Fetch one message, parsed by the existing `GmailLoader`."""
    creds = _get_credentials(user_id)
    loader = GmailLoader(
        access_token=creds.token,
        refresh_token=creds.refresh_token,
        credentials=creds,
    )
    try:
        with loader:
            email = loader.fetch_email(message_id)
    except EmailLoaderParseError as exc:
        raise InvalidEmailError(str(exc)) from exc
    except EmailLoaderError as exc:
        cause = exc.__cause__
        if isinstance(cause, HttpError) and getattr(cause.resp, "status", None) == 404:
            raise MessageNotFoundError(str(exc)) from exc
        if isinstance(cause, RefreshError):
            raise TokenRefreshError(f"Gmail token refresh failed: {cause}") from exc
        raise GmailApiError(str(exc)) from exc
    return email.model_copy(update={"owner_user_id": user_id})
