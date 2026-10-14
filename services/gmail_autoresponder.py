"""
services/gmail_autoresponder.py

Handles one Gmail message discovered by the push-notification sync:

    claim -> fetch -> index incoming -> check reply conditions
          -> agent reply (search email history, send via Gmail) -> index reply

The incoming email is indexed before any reply decision, so email history
is preserved whether the reply is sent, skipped, or fails. Every step's
outcome is recorded in `gmail_reply_states`; a row with unfinished work
is picked up again by `retry_pending` until it runs out of attempts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Optional

from agent.controller import build_incoming_email_prompt, run
from agent.state import AgentState
from config import settings
from database.models import GmailReplyStateRow
from initializer import chunker, db, dispatcher, embedder, store
from models import Email
from rag.pipeline import Pipeline
from services import gmail_service
from services.gmail_auth_service import AuthError
from services.gmail_service import GmailApiError, InvalidEmailError, MessageNotFoundError

logger = logging.getLogger(__name__)

# Messages in these labels are neither indexed nor answered.
IGNORED_LABELS = {"DRAFT", "SPAM", "TRASH"}

MAX_ERROR_CHARS = 1000


def _lock_timeout() -> timedelta:
    return timedelta(seconds=settings.GMAIL_REPLY_LOCK_TIMEOUT_SECONDS)


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _error(exc: object) -> str:
    return str(exc)[:MAX_ERROR_CHARS]


def index_email(email: Email) -> int:
    """Index one email through the RAG pipeline unless it is already stored.

    Chunk IDs are derived from the owner and Gmail message ID and stored
    with upsert, so a retry after a partial failure never duplicates chunks.
    """
    if store.has_email(email.id, owner_user_id=email.owner_user_id):
        return 0
    pipeline = Pipeline(store=store, embedder=embedder, dispatcher=dispatcher, chunker=chunker)
    return pipeline.run([email])


def process_message(
    user_id: int,
    message_id: str,
    *,
    owner_address: str,
    allow_reply: bool = True,
) -> Optional[str]:
    """Process one Gmail message for `user_id`.

    Returns the final reply status, or None when the message was not
    claimed (already finished, or being processed by another worker).
    `allow_reply=False` indexes the message without ever answering it.
    """
    row = db.claim_gmail_message(
        user_id,
        message_id,
        lock_timeout=_lock_timeout(),
        max_attempts=settings.GMAIL_REPLY_MAX_ATTEMPTS,
    )
    if row is None:
        logger.info("Gmail message %s for user %d already handled or in progress.", message_id, user_id)
        return None

    update = partial(db.update_gmail_reply_state, user_id, message_id, row.lock_token)
    if not allow_reply and row.status == "new":
        update(status="skipped_resync", retryable=False)
        row.status, row.retryable = "skipped_resync", False

    try:
        status = _process_claimed(user_id, row, owner_address, update)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Unexpected failure. The status is left as last recorded: it may
        # already say "replied", and must never be reset to a retryable
        # state. Unfinished work stays visible to retry_pending.
        logger.exception("Processing Gmail message %s for user %d failed.", message_id, user_id)
        status = row.status
        update(error=_error(exc))
    finally:
        update(release=True)

    logger.info("Gmail message %s for user %d: %s", message_id, user_id, status)
    return status


def _decision_pending(row: GmailReplyStateRow) -> bool:
    """True while it is still undecided whether to reply."""
    return row.status == "new" or (row.status == "failed" and row.retryable)


def _process_claimed(user_id: int, row: GmailReplyStateRow, owner_address: str, update) -> str:
    try:
        email = gmail_service.fetch_email(user_id, row.gmail_message_id)
    except (MessageNotFoundError, InvalidEmailError) as exc:
        # Permanent: nothing can be indexed or answered.
        update(status="invalid", retryable=False, incoming_indexed=True, error=_error(exc))
        return "invalid"
    except (GmailApiError, AuthError) as exc:
        status = "failed" if _decision_pending(row) else row.status
        update(status=status, error=_error(exc))
        return status

    update(thread_id=email.thread_id, sender=email.sender, subject=(email.subject or "")[:500])

    if IGNORED_LABELS & set(email.label_ids):
        update(status="skipped_ignored", retryable=False, incoming_indexed=True, error=None)
        return "skipped_ignored"

    # 1. Preserve history first: indexing never depends on the reply.
    index_error = None
    if not row.incoming_indexed:
        try:
            index_email(email)
            update(incoming_indexed=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Indexing Gmail message %s failed.", email.id)
            index_error = f"Indexing failed: {_error(exc)}"

    status = row.status
    error = None
    if _decision_pending(row):
        status, error = _decide_and_reply(user_id, email, owner_address, update)
    elif row.reply_message_id and not row.reply_indexed:
        # The reply was sent on an earlier attempt but not indexed yet.
        error = _index_sent_reply(user_id, email, row.reply_message_id, update)

    update(error=index_error or error)
    return status


def _decide_and_reply(user_id: int, email: Email, owner_address: str, update) -> tuple[str, Optional[str]]:
    """Check reply conditions, then let the agent answer. Returns (status, error)."""
    owner = owner_address.lower()
    labels = set(email.label_ids)

    def skip(status: str) -> tuple[str, None]:
        update(status=status, retryable=False)
        return status, None

    # Never answer the mailbox owner's own outgoing mail.
    if "SENT" in labels or email.sender.lower() == owner:
        return skip("skipped_owner")
    if "INBOX" not in labels:
        return skip("skipped_not_inbox")
    # Auto-replies, mailing lists and no-reply senders (avoids reply loops).
    if email.is_automated:
        return skip("skipped_automated")
    if not db.get_autoresponder_enabled(user_id):
        return skip("skipped_disabled")

    try:
        owner_replies = _owner_replies_after(user_id, email, owner)
    except (GmailApiError, AuthError) as exc:
        update(status="failed")
        return "failed", f"Thread lookup failed: {_error(exc)}"

    if owner_replies:
        # The owner already answered by hand: index that exchange instead.
        errors = []
        for reply_id in owner_replies:
            try:
                reply = gmail_service.fetch_email(user_id, reply_id)
                index_email(reply.model_copy(update={"in_reply_to": email.id}))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.exception("Indexing manual reply %s failed.", reply_id)
                errors.append(_error(exc))
        update(status="skipped_manual_reply", retryable=False)
        return "skipped_manual_reply", ("Indexing manual reply failed: " + "; ".join(errors)) if errors else None

    return _reply_with_agent(user_id, email, update)


def _owner_replies_after(user_id: int, email: Email, owner: str) -> list[str]:
    """IDs of the owner's messages in the thread sent after `email`."""
    if not email.thread_id:
        return []
    messages = gmail_service.get_thread_messages(user_id, email.thread_id)
    received_at = next(
        (m["internal_date"] for m in messages if m["id"] == email.id and m["internal_date"]),
        _utc(email.date),
    )
    return [
        m["id"]
        for m in messages
        if m["id"] != email.id
        and "DRAFT" not in m["label_ids"]
        and ("SENT" in m["label_ids"] or m["sender"] == owner)
        and m["internal_date"] is not None
        and m["internal_date"] > received_at
    ]


def _reply_with_agent(user_id: int, email: Email, update) -> tuple[str, Optional[str]]:
    # The agent sees the cleaned body (HTML, quoted history and signature stripped).
    cleaned = dispatcher.clean(email)
    state = AgentState(
        user_id=user_id,
        user_input=build_incoming_email_prompt(cleaned),
        incoming_email=email,
    )
    agent_error = None
    try:
        agent_response = run(state)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Agent failed on Gmail message %s.", email.id)
        agent_response = ""
        agent_error = f"Agent failed: {_error(exc)}"

    if not state.sent_emails:
        error = state.send_email_error or agent_error or (
            f"Agent did not send a reply: {agent_response[:300]}"
        )
        update(status="failed", retryable=state.send_email_retryable)
        return "failed", error

    sent = state.sent_emails[0]
    # Record the send before anything else can fail.
    update(status="replied", reply_message_id=sent["message_id"], retryable=False)

    reply = Email(
        id=sent["message_id"],
        subject=sent["subject"],
        sender=sent["sender"],
        recipients=[sent["to"]],
        body=sent["body"],
        date=datetime.now(timezone.utc),
        thread_id=sent.get("thread_id"),
        in_reply_to=email.id,
        label_ids=["SENT"],
        owner_user_id=user_id,
        auto_reply=True,
    )
    try:
        index_email(reply)
        update(reply_indexed=True)
        return "replied", None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Indexing reply %s failed.", reply.id)
        return "replied", f"Indexing reply failed: {_error(exc)}"


def _index_sent_reply(user_id: int, email: Email, reply_message_id: str, update) -> Optional[str]:
    try:
        reply = gmail_service.fetch_email(user_id, reply_message_id)
        index_email(reply.model_copy(update={"in_reply_to": email.id, "auto_reply": True}))
        update(reply_indexed=True)
        return None
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.exception("Indexing reply %s failed.", reply_message_id)
        return f"Indexing reply failed: {_error(exc)}"


def retry_pending(user_id: int, owner_address: str) -> int:
    """Retry this user's messages with unfinished reply or indexing work."""
    message_ids = db.list_gmail_reply_work(
        user_id,
        lock_timeout=_lock_timeout(),
        max_attempts=settings.GMAIL_REPLY_MAX_ATTEMPTS,
    )
    for message_id in message_ids:
        process_message(user_id, message_id, owner_address=owner_address)
    return len(message_ids)
