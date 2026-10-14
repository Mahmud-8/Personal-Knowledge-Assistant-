"""
Shared test fixtures.

The real `initializer` builds PostgreSQL, a sentence-transformers model and
an on-disk Chroma store at import time. Tests replace it before any app
module is imported with:

- `Database` on a temporary SQLite file (same data-access code, no server),
- a real `ChromaVectorStore` in a temporary directory with a deterministic
  bag-of-words embedder (real metadata filtering, no model download),
- the real chunker and cleaners.

Gmail, Pub/Sub and the LLM are always mocked: no test touches the network
or sends a real email.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import sys
import tempfile
import types
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

_TMP = tempfile.mkdtemp(prefix="pka_tests_")

for key, value in {
    "SECRET_KEY": "test-secret",
    "GOOGLE_CLIENT_ID": "test-client-id",
    "GOOGLE_CLIENT_SECRET": "test-client-secret",
    "GOOGLE_REDIRECT_URI": "http://localhost/callback",
    "GROQ_API_KEY": "test-groq-key",
    "VECTOR_STORE_DIR": os.path.join(_TMP, "chroma"),
    "POSTGRES_DB": "test",
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "ANONYMIZED_TELEMETRY": "False",
}.items():
    os.environ.setdefault(key, value)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine  # noqa: E402

from config import settings  # noqa: E402
from database.models import Base, GoogleCredentialRow, User  # noqa: E402
from database.postgresql import Database  # noqa: E402
from rag.chunker import LangChainChunker  # noqa: E402
from rag.preprocessing.cleaner_dispatcher import CleanerDispatcher  # noqa: E402
from rag.vector_store.chroma import ChromaVectorStore  # noqa: E402


class FakeEmbedder:
    """Deterministic hashed bag-of-words vectors (similar text -> similar vectors)."""

    DIM = 128

    def encode(self, texts):
        vectors = []
        for text in texts:
            vec = [0.0] * self.DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    def embed_many(self, chunks):
        vectors = self.encode([c["text"] for c in chunks])
        return [{"chunk": c, "vector": v} for c, v in zip(chunks, vectors)]


_engine = create_engine(
    "sqlite:///" + os.path.join(_TMP, "test.db"),
    connect_args={"check_same_thread": False, "timeout": 30},
)

fake_initializer = types.ModuleType("initializer")
fake_initializer.db = Database(settings, engine=_engine)
fake_initializer.embedder = FakeEmbedder()
fake_initializer.store = ChromaVectorStore(
    fake_initializer.embedder, os.path.join(_TMP, "chroma"), "knowledge"
)
fake_initializer.chunker = LangChainChunker()
fake_initializer.dispatcher = CleanerDispatcher()
sys.modules["initializer"] = fake_initializer

OWNER_A = "alice@example.com"
OWNER_B = "bob@example.com"


def pytest_sessionfinish(session, exitstatus):  # pragma: no cover - cleanup
    _engine.dispose()
    shutil.rmtree(_TMP, ignore_errors=True)


@pytest.fixture
def db():
    return fake_initializer.db


@pytest.fixture
def store():
    return fake_initializer.store


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    """Fresh tables, empty vector store, and test push settings per test."""
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)

    store = fake_initializer.store
    store._client.delete_collection("knowledge")  # pylint: disable=protected-access
    store._collection = store._client.get_or_create_collection("knowledge")  # pylint: disable=protected-access

    from pydantic import SecretStr
    monkeypatch.setattr(settings, "GMAIL_PUBSUB_TOPIC", "projects/test/topics/gmail")
    monkeypatch.setattr(settings, "GMAIL_PUBSUB_VERIFICATION_TOKEN", SecretStr("push-secret"))
    monkeypatch.setattr(settings, "GMAIL_PUBSUB_AUDIENCE", "")
    monkeypatch.setattr(settings, "GMAIL_PUBSUB_SERVICE_ACCOUNT", "")
    monkeypatch.setattr(settings, "GMAIL_PUBSUB_SUBSCRIPTION", "")
    yield


def add_user(db, user_id: int, email: str, *, autoresponder: bool = True, google_token: str | None = None):
    with db._session() as session:  # pylint: disable=protected-access
        session.add(User(
            id=user_id, name=email.split("@")[0], email=email,
            password_hash="x", autoresponder_enabled=autoresponder,
        ))
        session.add(GoogleCredentialRow(
            id=user_id, user_id=user_id, google_id=f"google-{user_id}",
            access_token=google_token or f"token-{user_id}", refresh_token=f"refresh-{user_id}",
        ))
        session.commit()


@pytest.fixture
def users(db):
    """Two users with linked Google accounts and active watches."""
    add_user(db, 1, OWNER_A)
    add_user(db, 2, OWNER_B)
    for user_id, address in ((1, OWNER_A), (2, OWNER_B)):
        db.save_gmail_watch(
            user_id, email_address=address, history_id=100,
            expiration=datetime(2099, 1, 1, tzinfo=timezone.utc),
        )
    return {1: OWNER_A, 2: OWNER_B}


# ----------------------------------------------------------------------
# Fake Gmail
# ----------------------------------------------------------------------


class FakeGmail:
    """In-memory Gmail for several mailboxes, replacing services.gmail_service calls."""

    def __init__(self, owners: dict[int, str]):
        self.owners = owners
        self.messages: dict[int, dict] = {uid: {} for uid in owners}
        self.history: dict[int, list[tuple[int, str]]] = {uid: [] for uid in owners}
        self.current_history: dict[int, int] = {uid: 100 for uid in owners}
        self.sent: list[dict] = []
        self.calls: list[tuple[str, int]] = []
        self.expired_history: set[int] = set()
        self.send_error: Exception | None = None
        self._next = 0

    def add_message(self, user_id, *, sender, subject="Question", body="Hello", labels=("INBOX",),
                    thread_id=None, date=None, message_id=None, is_automated=False, record_history=True):
        from models import Email
        self._next += 1
        message_id = message_id or f"m{user_id}-{self._next}"
        email = Email(
            id=message_id, subject=subject, sender=sender,
            recipients=[self.owners[user_id] if sender != self.owners[user_id] else "someone@example.com"],
            body=body, date=date or datetime(2026, 10, 1, 9, self._next % 60, tzinfo=timezone.utc),
            thread_id=thread_id or f"t-{message_id}", rfc_message_id=f"<{message_id}@mail>",
            label_ids=list(labels), is_automated=is_automated,
        )
        self.messages[user_id][message_id] = email
        if record_history:
            self.current_history[user_id] += 1
            self.history[user_id].append((self.current_history[user_id], message_id))
        return email

    # --- replacements for services.gmail_service ---

    def fetch_email(self, user_id, message_id):
        from services.gmail_service import MessageNotFoundError
        self.calls.append(("fetch_email", user_id))
        email = self.messages[user_id].get(message_id)
        if email is None:
            raise MessageNotFoundError(message_id)
        return email.model_copy(update={"owner_user_id": user_id})

    def list_history(self, user_id, start_history_id):
        from services.gmail_service import HistoryExpiredError
        self.calls.append(("list_history", user_id))
        if user_id in self.expired_history:
            raise HistoryExpiredError("expired")
        ids = [mid for hid, mid in self.history[user_id] if hid > start_history_id]
        return ids, self.current_history[user_id]

    def get_mailbox_profile(self, user_id):
        return {"email_address": self.owners[user_id], "history_id": self.current_history[user_id]}

    def list_message_ids_since(self, user_id, after, max_results=100):
        return [m.id for m in self.messages[user_id].values() if m.date > after][:max_results]

    def get_thread_messages(self, user_id, thread_id):
        return [
            {
                "id": m.id, "label_ids": list(m.label_ids),
                "internal_date": m.date, "sender": m.sender.lower(),
            }
            for m in self.messages[user_id].values() if m.thread_id == thread_id
        ]

    def send_email(self, user_id, *, to, subject, body, thread_id=None, in_reply_to=None):
        self.calls.append(("send_email", user_id))
        if self.send_error is not None:
            raise self.send_error
        self._next += 1
        message_id = f"sent{user_id}-{self._next}"
        self.sent.append({
            "user_id": user_id, "to": to, "subject": subject, "body": body,
            "thread_id": thread_id, "in_reply_to": in_reply_to, "message_id": message_id,
        })
        # The reply lands in the owner's mailbox as a SENT message.
        from models import Email
        self.messages[user_id][message_id] = Email(
            id=message_id, subject=subject, sender=self.owners[user_id], recipients=[to],
            body=body, date=datetime(2026, 10, 1, 23, 59, tzinfo=timezone.utc),
            thread_id=thread_id, label_ids=["SENT"],
        )
        return {"message_id": message_id, "thread_id": thread_id, "sender": self.owners[user_id]}

    def start_watch(self, user_id, *, topic_name, label_ids):
        self.calls.append(("start_watch", user_id))
        return {
            "email_address": self.owners[user_id],
            "history_id": self.current_history[user_id],
            "expiration": datetime(2099, 1, 8, tzinfo=timezone.utc),
        }

    def stop_watch(self, user_id):
        self.calls.append(("stop_watch", user_id))


@pytest.fixture
def gmail(monkeypatch, users):
    from services import gmail_service
    fake = FakeGmail(users)
    for name in ("fetch_email", "list_history", "get_mailbox_profile", "list_message_ids_since",
                 "get_thread_messages", "send_email", "start_watch", "stop_watch"):
        monkeypatch.setattr(gmail_service, name, getattr(fake, name))
    return fake


# ----------------------------------------------------------------------
# Scripted LLM
# ----------------------------------------------------------------------


def _tool_call(call_id, name, arguments):
    import json
    return SimpleNamespace(
        id=call_id, type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class FakeLLM:
    """Plays the agent: search email history, then reply with what it found.

    Records the tool names offered on every call, so tests can assert what
    the model was (not) allowed to do.
    """

    def __init__(self):
        self.offered_tools: list[list[str] | None] = []
        self.fail = False
        self.chat_tries_to_send = False

    def __call__(self, messages, system_prompt, tool_names=None):
        self.offered_tools.append(tool_names)
        if self.fail:
            from agent.llm_client import LLMResponseError
            raise LLMResponseError("model unavailable")

        first = messages[0]["content"] if messages else ""
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        is_email = first.startswith("New email")

        if is_email:
            sender = re.search(r"^Sender: (.+)$", first, re.M).group(1)
            subject = re.search(r"^Subject: (.*)$", first, re.M).group(1)
            if not tool_msgs:
                return SimpleNamespace(content="", tool_calls=[
                    _tool_call("c1", "search", {"query": subject, "source": "email"})])
            if len(tool_msgs) == 1:
                found = tool_msgs[0]["content"]
                facts = re.findall(r"'text': '([^']*)'", found)
                body = "Thanks for your email. " + (f"From our records: {facts[0]}" if facts
                                                     else "We will follow up shortly.")
                return SimpleNamespace(content="", tool_calls=[
                    _tool_call("c2", "send_email_reply",
                               {"to": sender, "subject": f"Re: {subject}", "body": body})])
            return SimpleNamespace(content="Reply sent.", tool_calls=None)

        # Ordinary chat.
        if not tool_msgs:
            calls = [_tool_call("c1", "search", {"query": first})]
            if self.chat_tries_to_send:
                calls.append(_tool_call("c2", "send_email_reply",
                                        {"to": "x@example.com", "subject": "hi", "body": "hi"}))
            return SimpleNamespace(content="", tool_calls=calls)
        return SimpleNamespace(content="Here is what I found.", tool_calls=None)


@pytest.fixture
def llm(monkeypatch):
    import agent.controller
    fake = FakeLLM()
    monkeypatch.setattr(agent.controller, "call_llm", fake)
    return fake


def email_chunks(store, **where):
    """Metadata of stored email chunks matching `where` (exact match)."""
    conditions = [{"source_type": "email"}] + [{k: v} for k, v in where.items()]
    filt = conditions[0] if len(conditions) == 1 else {"$and": conditions}
    result = store._collection.get(where=filt)  # pylint: disable=protected-access
    return result["metadatas"]
