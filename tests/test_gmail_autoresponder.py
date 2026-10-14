"""Auto-responder workflow: reply conditions, grounding, failures, indexing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from conftest import OWNER_A, email_chunks
from models import Email
from services import gmail_autoresponder
from services.gmail_service import GmailSendError


def seed_history(user_id: int, body: str, *, subject: str = "Earlier thread", email_id: str | None = None):
    """Index a previously ingested email for `user_id`."""
    email = Email(
        id=email_id or f"old-{user_id}-{abs(hash(body)) % 10_000}",
        subject=subject,
        sender="finance@example.com",
        recipients=["someone@example.com"],
        body=body,
        date=datetime(2026, 9, 1, tzinfo=timezone.utc),
        owner_user_id=user_id,
    )
    gmail_autoresponder.index_email(email)
    return email


def process(user_id, message_id, **kwargs):
    return gmail_autoresponder.process_message(user_id, message_id, owner_address=OWNER_A, **kwargs)


def reply_state(db, user_id, message_id):
    return next(r for r in db.list_gmail_reply_states(user_id, limit=100) if r.gmail_message_id == message_id)


# ----------------------------------------------------------------------
# Grounded reply through the Gmail API
# ----------------------------------------------------------------------


def test_valid_email_gets_grounded_reply_via_gmail_api(gmail, llm, db):
    seed_history(1, "The October invoice is due on the 15th of the month.")
    incoming = gmail.add_message(1, sender="carol@example.com", subject="Invoice due date",
                                 body="Hi, when is the October invoice due?")

    assert process(1, incoming.id) == "replied"

    assert len(gmail.sent) == 1
    sent = gmail.sent[0]
    assert sent["user_id"] == 1
    assert sent["to"] == "carol@example.com"
    assert sent["subject"] == "Re: Invoice due date"
    assert sent["thread_id"] == incoming.thread_id
    assert sent["in_reply_to"] == incoming.rfc_message_id
    assert "15th" in sent["body"]  # grounded in the retrieved history
    state = reply_state(db, 1, incoming.id)
    assert state.status == "replied" and state.reply_message_id == sent["message_id"]
    assert state.locked_at is None and state.lock_token is None


def test_reply_subject_is_not_doubled():
    from agent.tools import reply_subject
    assert reply_subject("Invoice") == "Re: Invoice"
    assert reply_subject("RE: Invoice") == "RE: Invoice"
    assert reply_subject("") == "Re:"


def test_history_search_uses_email_source_scoped_to_user(gmail, llm, store, monkeypatch):
    seed_history(1, "Project Falcon launches on March 3.")
    # A non-email record that matches the query just as well.
    store.add(fake_embedded("slack-1", "Project Falcon launches on March 9 per slack.", "slack"))
    incoming = gmail.add_message(1, sender="dan@example.com", subject="Project Falcon launch",
                                 body="When does Project Falcon launch?")

    calls = []
    original = store.similarity_search

    def spy(query, **kwargs):
        calls.append(kwargs)
        return original(query, **kwargs)

    monkeypatch.setattr(store, "similarity_search", spy)
    assert process(1, incoming.id) == "replied"

    assert calls, "the agent must search before replying"
    for kwargs in calls:
        assert kwargs["source_type"] == "email"
        assert kwargs["owner_user_id"] == 1
        assert kwargs["exclude_email_id"] == incoming.id
    assert "March 3" in gmail.sent[0]["body"]
    assert "slack" not in gmail.sent[0]["body"]


def test_search_tool_forces_email_source_even_if_model_omits_it(gmail, monkeypatch, store):
    from agent.state import AgentState
    from agent.tool_executer import execute_tool

    incoming = gmail.add_message(1, sender="x@example.com")
    state = AgentState(user_id=1, user_input="New email", incoming_email=incoming)
    seen = {}
    monkeypatch.setattr(store, "similarity_search", lambda q, **kw: seen.update(kw) or [])

    execute_tool("search", {"query": "anything", "source": "slack"}, state)
    assert seen["source_type"] == "email"


def fake_embedded(chunk_id, text, source_type, **meta):
    from conftest import FakeEmbedder
    chunk = {"id": chunk_id, "text": text, "chunk_index": 0,
             "metadata": {"source_type": source_type, **meta}}
    return [{"chunk": chunk, "vector": FakeEmbedder().encode([text])[0]}]


# ----------------------------------------------------------------------
# Reply conditions
# ----------------------------------------------------------------------


def test_disabled_autoresponder_indexes_without_sending(gmail, llm, db, store):
    db.set_autoresponder_enabled(1, False)
    incoming = gmail.add_message(1, sender="erin@example.com", subject="Hello", body="Are you there?")

    assert process(1, incoming.id) == "skipped_disabled"

    assert gmail.sent == []
    assert llm.offered_tools == []  # the agent was never run
    assert email_chunks(store, email_id=incoming.id, owner_user_id=1)
    state = reply_state(db, 1, incoming.id)
    assert state.incoming_indexed and not state.retryable


@pytest.mark.parametrize("labels,sender", [
    (("SENT",), OWNER_A),            # owner's outgoing mail
    (("INBOX", "SENT"), OWNER_A),    # owner mailed themself
    (("INBOX",), "Alice@Example.com"),  # owner address, no SENT label
])
def test_owner_outgoing_email_never_triggers_reply(gmail, llm, db, store, labels, sender):
    message = gmail.add_message(1, sender=sender, labels=labels, body="Note to self about budgets.")

    assert process(1, message.id) == "skipped_owner"

    assert gmail.sent == []
    assert email_chunks(store, email_id=message.id, owner_user_id=1)


def test_existing_manual_reply_skips_ai_and_indexes_exchange(gmail, llm, db, store):
    incoming = gmail.add_message(1, sender="frank@example.com", subject="Contract", body="Can you sign?",
                                 thread_id="t-contract", date=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc))
    manual = gmail.add_message(1, sender=OWNER_A, subject="Re: Contract", body="Signed and returned today.",
                               labels=("SENT",), thread_id="t-contract", record_history=False,
                               date=datetime(2026, 10, 1, 10, 0, tzinfo=timezone.utc))

    assert process(1, incoming.id) == "skipped_manual_reply"

    assert gmail.sent == []
    assert email_chunks(store, email_id=incoming.id, owner_user_id=1)
    owner_reply = email_chunks(store, email_id=manual.id, owner_user_id=1)
    assert owner_reply and owner_reply[0]["in_reply_to"] == incoming.id


def test_owner_message_earlier_in_thread_does_not_block_reply(gmail, llm):
    gmail.add_message(1, sender=OWNER_A, labels=("SENT",), thread_id="t-1", record_history=False,
                      date=datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc))
    incoming = gmail.add_message(1, sender="gina@example.com", thread_id="t-1",
                                 date=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc))

    assert process(1, incoming.id) == "replied"
    assert len(gmail.sent) == 1


def test_automated_sender_is_indexed_not_answered(gmail, llm, store):
    message = gmail.add_message(1, sender="no-reply@service.example.com", is_automated=True)

    assert process(1, message.id) == "skipped_automated"
    assert gmail.sent == []
    assert email_chunks(store, email_id=message.id, owner_user_id=1)


def test_spam_and_drafts_are_ignored(gmail, llm, store):
    spam = gmail.add_message(1, sender="spam@example.com", labels=("SPAM",))
    assert process(1, spam.id) == "skipped_ignored"
    assert gmail.sent == [] and not email_chunks(store, email_id=spam.id)


def test_deleted_message_is_marked_invalid_and_not_retried(gmail, llm, db):
    assert process(1, "missing-message") == "invalid"
    assert db.list_gmail_reply_work(1, lock_timeout=gmail_autoresponder._lock_timeout(), max_attempts=3) == []


# ----------------------------------------------------------------------
# Duplicates and concurrency
# ----------------------------------------------------------------------


def test_duplicate_processing_sends_one_reply(gmail, llm):
    incoming = gmail.add_message(1, sender="hank@example.com")

    assert process(1, incoming.id) == "replied"
    assert process(1, incoming.id) is None
    assert process(1, incoming.id) is None
    assert len(gmail.sent) == 1


def test_concurrent_workers_send_one_reply(gmail, llm):
    import threading
    incoming = gmail.add_message(1, sender="ivy@example.com")
    barrier = threading.Barrier(6)
    results = []

    def worker():
        barrier.wait()
        results.append(process(1, incoming.id))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(gmail.sent) == 1
    assert results.count("replied") == 1
    assert results.count(None) == 5


def test_claim_is_exclusive_until_lock_times_out(db, users):
    from datetime import timedelta
    from database.models import GmailReplyStateRow
    from sqlalchemy import update

    timeout = timedelta(minutes=15)
    first = db.claim_gmail_message(1, "m1", lock_timeout=timeout, max_attempts=3)
    assert first is not None
    assert db.claim_gmail_message(1, "m1", lock_timeout=timeout, max_attempts=3) is None

    # Abandoned for longer than the timeout (worker crashed): reclaimable.
    with db._session() as session:  # pylint: disable=protected-access
        session.execute(update(GmailReplyStateRow).values(
            locked_at=datetime.now(timezone.utc) - timedelta(minutes=16)))
        session.commit()
    second = db.claim_gmail_message(1, "m1", lock_timeout=timeout, max_attempts=3)
    assert second is not None and second.lock_token != first.lock_token
    # The stale worker can no longer write.
    assert db.update_gmail_reply_state(1, "m1", first.lock_token, status="replied") is False


def test_reclaimed_message_after_crash_does_not_double_reply(gmail, llm, db):
    """A worker crashed after sending: the thread now holds our reply."""
    from datetime import timedelta
    from database.models import GmailReplyStateRow
    from sqlalchemy import update

    incoming = gmail.add_message(1, sender="jay@example.com", thread_id="t-crash")
    db.claim_gmail_message(1, incoming.id, lock_timeout=timedelta(minutes=15), max_attempts=3)
    gmail.send_email(1, to="jay@example.com", subject="Re: Question", body="Sent before crash",
                     thread_id="t-crash")
    with db._session() as session:  # pylint: disable=protected-access
        session.execute(update(GmailReplyStateRow).values(
            locked_at=datetime.now(timezone.utc) - timedelta(minutes=20)))
        session.commit()

    gmail_autoresponder.retry_pending(1, OWNER_A)

    assert len(gmail.sent) == 1
    assert reply_state(db, 1, incoming.id).status == "skipped_manual_reply"


# ----------------------------------------------------------------------
# Failures keep history and stay retryable
# ----------------------------------------------------------------------


def test_agent_failure_indexes_email_and_retries_later(gmail, llm, db, store):
    incoming = gmail.add_message(1, sender="kim@example.com")
    llm.fail = True

    assert process(1, incoming.id) == "failed"
    assert gmail.sent == []
    assert email_chunks(store, email_id=incoming.id, owner_user_id=1)
    state = reply_state(db, 1, incoming.id)
    assert state.retryable and state.error

    llm.fail = False
    assert gmail_autoresponder.retry_pending(1, OWNER_A) == 1
    assert reply_state(db, 1, incoming.id).status == "replied"
    assert len(gmail.sent) == 1
    # Indexed once, not duplicated by the retry.
    assert len(email_chunks(store, email_id=incoming.id, owner_user_id=1)) == 1


def test_gmail_send_error_indexes_email_and_is_retryable(gmail, llm, db, store):
    incoming = gmail.add_message(1, sender="lee@example.com")
    gmail.send_error = GmailSendError("Gmail API error: 500")

    assert process(1, incoming.id) == "failed"
    assert email_chunks(store, email_id=incoming.id, owner_user_id=1)
    assert reply_state(db, 1, incoming.id).retryable

    gmail.send_error = None
    gmail_autoresponder.retry_pending(1, OWNER_A)
    assert reply_state(db, 1, incoming.id).status == "replied"
    assert len(gmail.sent) == 1


def test_oauth_error_on_send_is_retryable(gmail, llm, db):
    from services.gmail_auth_service import TokenRefreshError
    incoming = gmail.add_message(1, sender="max@example.com")
    gmail.send_error = TokenRefreshError("refresh token revoked")

    assert process(1, incoming.id) == "failed"
    state = reply_state(db, 1, incoming.id)
    assert state.retryable and "authentication" in state.error.lower()


def test_ambiguous_send_failure_is_not_retried(gmail, llm, db, store):
    """A transport error mid-send may have delivered the email: never resend."""
    incoming = gmail.add_message(1, sender="ned@example.com")
    gmail.send_error = ConnectionResetError("connection reset during send")

    assert process(1, incoming.id) == "failed"
    assert email_chunks(store, email_id=incoming.id, owner_user_id=1)
    assert not reply_state(db, 1, incoming.id).retryable

    gmail.send_error = None
    assert gmail_autoresponder.retry_pending(1, OWNER_A) == 0
    assert gmail.sent == []


def test_index_failure_does_not_block_reply_and_is_retried(gmail, llm, db, store, monkeypatch):
    incoming = gmail.add_message(1, sender="olga@example.com")
    real_index = gmail_autoresponder.index_email
    failures = {"left": 1}

    def flaky_index(email):
        if email.id == incoming.id and failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("chroma unavailable")
        return real_index(email)

    monkeypatch.setattr(gmail_autoresponder, "index_email", flaky_index)

    assert process(1, incoming.id) == "replied"
    state = reply_state(db, 1, incoming.id)
    assert not state.incoming_indexed and "Indexing failed" in state.error

    gmail_autoresponder.retry_pending(1, OWNER_A)
    state = reply_state(db, 1, incoming.id)
    assert state.incoming_indexed and state.error is None
    assert email_chunks(store, email_id=incoming.id, owner_user_id=1)
    assert len(gmail.sent) == 1  # the retry only indexed


def test_successful_reply_is_indexed_with_relationship(gmail, llm, db, store):
    incoming = gmail.add_message(1, sender="pat@example.com", subject="Opening hours")

    assert process(1, incoming.id) == "replied"

    sent = gmail.sent[0]
    chunks = email_chunks(store, email_id=sent["message_id"], owner_user_id=1)
    assert chunks
    meta = chunks[0]
    assert meta["in_reply_to"] == incoming.id
    assert meta["auto_reply"] is True
    assert meta["thread_id"] == incoming.thread_id
    assert meta["sender"] == OWNER_A
    assert meta["subject"] == "Re: Opening hours"
    assert reply_state(db, 1, incoming.id).reply_indexed


def test_indexing_is_idempotent(gmail, store):
    email = seed_history(1, "Idempotent body text.", email_id="dup-1")
    before = store.count()
    assert gmail_autoresponder.index_email(email) == 0
    # Even a forced re-run of the pipeline overwrites instead of duplicating.
    from initializer import chunker, dispatcher, embedder
    from rag.pipeline import Pipeline
    Pipeline(store=store, embedder=embedder, dispatcher=dispatcher, chunker=chunker).run([email])
    assert store.count() == before
