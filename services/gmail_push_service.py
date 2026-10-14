"""
services/gmail_push_service.py

Gmail push notifications through Google Cloud Pub/Sub.

- `ensure_watch` / `stop_watch`: manage the users.watch subscription.
- `verify_push_request` / `parse_push_message`: authenticate and decode a
  Pub/Sub push request. A notification only carries the mailbox address
  and a historyId, never the email itself.
- `sync_mailbox`: read Gmail history from the stored historyId, hand each
  new message to the auto-responder, then advance the pointer.
- `run_maintenance`: renew watches before they expire and retry unfinished
  work. This is the only periodic task; new mail is never polled for.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from config import settings
from database.models import GmailSyncStateRow
from initializer import db
from services import gmail_autoresponder, gmail_service
from services.gmail_auth_service import AuthError
from services.gmail_service import GmailApiError, HistoryExpiredError

logger = logging.getLogger(__name__)

# How far back to look when Gmail no longer has the stored history.
RESYNC_MAX_AGE = timedelta(days=7)
RESYNC_MAX_MESSAGES = 100


class PushNotConfiguredError(Exception):
    """Gmail push notifications are not configured on this server."""


class PushAuthError(Exception):
    """A push request failed authentication."""


class InvalidNotificationError(Exception):
    """A push request body is not a valid Gmail notification."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


# ----------------------------------------------------------------------
# Watch lifecycle
# ----------------------------------------------------------------------


def ensure_watch(user_id: int) -> GmailSyncStateRow:
    """Start or renew the Gmail watch for `user_id` and store its state."""
    if not settings.GMAIL_PUBSUB_TOPIC:
        raise PushNotConfiguredError("GMAIL_PUBSUB_TOPIC is not set.")

    existing = db.find_gmail_sync_state(user_id)
    try:
        info = gmail_service.start_watch(
            user_id,
            topic_name=settings.GMAIL_PUBSUB_TOPIC,
            label_ids=settings.gmail_watch_label_ids,
        )
    except (GmailApiError, AuthError) as exc:
        if existing:
            db.update_gmail_sync_state(user_id, watch_status="error", last_error=str(exc)[:1000])
        raise

    # After an explicit stop, start fresh instead of replaying (and
    # answering) everything that arrived while notifications were off.
    reset = existing is not None and existing.watch_status == "stopped"
    state = db.save_gmail_watch(
        user_id,
        email_address=info["email_address"],
        history_id=info["history_id"],
        expiration=info["expiration"],
        reset_history=reset,
    )
    logger.info("Gmail watch active for user %d until %s.", user_id, state.watch_expiration)
    return state


def stop_watch(user_id: int) -> None:
    """Stop notifications for `user_id`; the sync state is kept."""
    gmail_service.stop_watch(user_id)
    db.update_gmail_sync_state(user_id, watch_status="stopped", watch_expiration=None)


# ----------------------------------------------------------------------
# Push request validation
# ----------------------------------------------------------------------


def verify_push_request(
    authorization: Optional[str],
    token: Optional[str],
    subscription: Optional[str],
) -> None:
    """Authenticate a Pub/Sub push request.

    Supports Pub/Sub OIDC push authentication (recommended) and/or a shared
    verification token in the push URL. Every configured check must pass;
    with neither configured, all requests are rejected.
    """
    expected_token = settings.GMAIL_PUBSUB_VERIFICATION_TOKEN.get_secret_value()
    audience = settings.GMAIL_PUBSUB_AUDIENCE
    if not expected_token and not audience:
        raise PushNotConfiguredError("No Pub/Sub push authentication is configured.")

    if expected_token and not hmac.compare_digest(
        (token or "").encode("utf-8"), expected_token.encode("utf-8")
    ):
        raise PushAuthError("Invalid verification token.")

    if audience:
        service_account = settings.GMAIL_PUBSUB_SERVICE_ACCOUNT
        if not service_account:
            raise PushNotConfiguredError("GMAIL_PUBSUB_SERVICE_ACCOUNT is not set.")
        scheme, _, bearer = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not bearer:
            raise PushAuthError("Missing bearer token.")
        try:
            claims = id_token.verify_oauth2_token(
                bearer, google_requests.Request(), audience=audience
            )
        except ValueError as exc:
            raise PushAuthError(f"Invalid push token: {exc}") from exc
        if claims.get("email") != service_account or not claims.get("email_verified"):
            raise PushAuthError("Push token was not issued for the expected service account.")

    expected_subscription = settings.GMAIL_PUBSUB_SUBSCRIPTION
    if expected_subscription and subscription != expected_subscription:
        raise PushAuthError("Unexpected Pub/Sub subscription.")


def parse_push_message(body: Any) -> tuple[str, int]:
    """Decode a Pub/Sub push body into (email_address, history_id)."""
    if not isinstance(body, dict) or not isinstance(body.get("message"), dict):
        raise InvalidNotificationError("Body is not a Pub/Sub push message.")
    data = body["message"].get("data")
    if not isinstance(data, str) or not data:
        raise InvalidNotificationError("Pub/Sub message has no data.")
    try:
        payload = json.loads(base64.b64decode(data + "=" * (-len(data) % 4), altchars=b"-_"))
    except (binascii.Error, ValueError) as exc:
        raise InvalidNotificationError(f"Undecodable notification data: {exc}") from exc
    if not isinstance(payload, dict):
        raise InvalidNotificationError("Notification data is not an object.")

    email_address = payload.get("emailAddress")
    history_id = payload.get("historyId")
    if not isinstance(email_address, str) or "@" not in email_address:
        raise InvalidNotificationError("Notification has no valid emailAddress.")
    try:
        history_id = int(history_id)
    except (TypeError, ValueError) as exc:
        raise InvalidNotificationError("Notification has no valid historyId.") from exc
    if history_id <= 0:
        raise InvalidNotificationError("Notification has no valid historyId.")
    return email_address.strip().lower(), history_id


def handle_notification(email_address: str, history_id: int) -> Optional[int]:
    """Map a notification to its user. Returns the user ID to sync, or None.

    Only mailboxes with a watch this server created are accepted; the
    notification's historyId is not trusted as a sync position.
    """
    state = db.find_gmail_sync_state_by_email(email_address)
    if state is None or state.watch_status == "stopped":
        logger.info("Ignoring Gmail notification for an unwatched mailbox.")
        return None
    db.update_gmail_sync_state(state.user_id, last_notification_at=_utcnow())
    logger.info("Gmail notification for user %d (historyId %d).", state.user_id, history_id)
    return state.user_id


# ----------------------------------------------------------------------
# History sync
# ----------------------------------------------------------------------

_guard = threading.Lock()
_running: set[int] = set()
_pending: set[int] = set()


def sync_mailbox(user_id: int) -> None:
    """Process new mail for `user_id`.

    Notifications that arrive while a sync for the same user is running in
    this process are coalesced into one more pass. Across processes, the
    per-message claim in `gmail_reply_states` prevents duplicate replies.
    """
    with _guard:
        _pending.add(user_id)
        if user_id in _running:
            return
        _running.add(user_id)
    try:
        while True:
            with _guard:
                if user_id not in _pending:
                    _running.discard(user_id)
                    return
                _pending.discard(user_id)
            try:
                _sync_once(user_id)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Gmail sync failed for user %d.", user_id)
    except BaseException:
        with _guard:
            _running.discard(user_id)
        raise


def _sync_once(user_id: int) -> None:
    state = db.find_gmail_sync_state(user_id)
    if state is None or state.watch_status == "stopped":
        return

    try:
        if state.history_id is None:
            profile = gmail_service.get_mailbox_profile(user_id)
            db.advance_gmail_history_id(user_id, profile["history_id"])
            return

        try:
            message_ids, latest = gmail_service.list_history(user_id, state.history_id)
            allow_reply = True
        except HistoryExpiredError:
            # Gmail keeps roughly a week of history. Index what arrived since
            # the last good sync, but do not auto-answer possibly stale mail.
            logger.warning("Gmail history expired for user %d; resyncing.", user_id)
            since = max(
                _utc(state.last_sync_at) or _utcnow() - timedelta(days=1),
                _utcnow() - RESYNC_MAX_AGE,
            )
            latest = gmail_service.get_mailbox_profile(user_id)["history_id"]
            message_ids = gmail_service.list_message_ids_since(
                user_id, since, max_results=RESYNC_MAX_MESSAGES
            )
            allow_reply = False

        for message_id in message_ids:
            gmail_autoresponder.process_message(
                user_id,
                message_id,
                owner_address=state.email_address,
                allow_reply=allow_reply,
            )

        if latest is not None:
            if allow_reply:
                db.advance_gmail_history_id(user_id, latest)
            else:
                db.update_gmail_sync_state(user_id, history_id=latest)

        gmail_autoresponder.retry_pending(user_id, state.email_address)
    except (GmailApiError, AuthError) as exc:
        # The pointer was not advanced, so the next notification retries.
        logger.error("Gmail sync failed for user %d: %s", user_id, exc)
        db.update_gmail_sync_state(user_id, last_error=str(exc)[:1000])
        return

    db.update_gmail_sync_state(user_id, last_sync_at=_utcnow(), last_error=None)


# ----------------------------------------------------------------------
# Maintenance: watch renewal and retries
# ----------------------------------------------------------------------


def run_maintenance() -> None:
    """Renew watches close to expiry, then catch up and retry for each user."""
    now = _utcnow()
    renew_before = now + timedelta(seconds=settings.GMAIL_WATCH_RENEW_BEFORE_SECONDS)

    for state in db.list_gmail_sync_states():
        if state.watch_status == "stopped":
            continue
        expiration = _utc(state.watch_expiration)
        if settings.GMAIL_PUBSUB_TOPIC and (expiration is None or expiration <= renew_before):
            try:
                ensure_watch(state.user_id)
            except (GmailApiError, AuthError, PushNotConfiguredError) as exc:
                logger.error("Renewing Gmail watch for user %d failed: %s", state.user_id, exc)
                continue
        # Catches up on anything missed while a watch was lapsed or the
        # server was down, and retries unfinished replies/indexing.
        try:
            sync_mailbox(state.user_id)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Gmail maintenance sync failed for user %d.", state.user_id)
