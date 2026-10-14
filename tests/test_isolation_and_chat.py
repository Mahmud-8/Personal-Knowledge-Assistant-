"""Multi-user isolation, ordinary chat, loader parsing, and settings endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.controller import run
from agent.state import AgentState
from agent.tool_executer import execute_tool
from conftest import OWNER_A, add_user
from models import Email
from services import gmail_autoresponder


def index(user_id, email_id, body):
    gmail_autoresponder.index_email(Email(
        id=email_id, subject="Secret plans", sender="partner@example.com",
        recipients=["x@example.com"], body=body,
        date=datetime(2026, 9, 1, tzinfo=timezone.utc), owner_user_id=user_id,
    ))


def found_ids(result):
    return {r["metadata"].get("email_id") for r in result.data["results"]}


# ----------------------------------------------------------------------
# Ordinary chat never sends email
# ----------------------------------------------------------------------


def test_chat_does_not_offer_or_execute_send_tool(gmail, llm):
    llm.chat_tries_to_send = True
    state = AgentState(user_id=1, user_input="What is the invoice due date?")

    answer = run(state)

    assert answer == "Here is what I found."
    assert all("send_email_reply" not in tools for tools in llm.offered_tools)
    assert gmail.sent == [] and ("send_email", 1) not in gmail.calls
    tool_outputs = [m["content"] for m in state.messages if m["role"] == "tool"]
    assert any("not available" in out for out in tool_outputs)


def test_send_tool_refuses_without_incoming_email(gmail):
    state = AgentState(user_id=1, user_input="hi")
    result = execute_tool("send_email_reply", {"to": "a@example.com", "subject": "s", "body": "b"}, state)
    assert not result.success
    assert gmail.sent == []


def test_send_tool_only_replies_to_original_sender(gmail):
    incoming = gmail.add_message(1, sender="real@example.com")
    state = AgentState(user_id=1, user_input="New email", incoming_email=incoming)
    result = execute_tool("send_email_reply",
                          {"to": "attacker@evil.com", "subject": "x", "body": "data"}, state)
    assert not result.success and gmail.sent == []


def test_email_handling_offers_send_tool(gmail, llm):
    incoming = gmail.add_message(1, sender="ann@example.com")
    gmail_autoresponder.process_message(1, incoming.id, owner_address=OWNER_A)
    assert "send_email_reply" in llm.offered_tools[0]


# ----------------------------------------------------------------------
# Isolation between users
# ----------------------------------------------------------------------


def test_email_search_is_scoped_to_the_user(users, store):
    index(1, "a-1", "Alice merger plans for Q4.")
    index(2, "b-1", "Bob merger plans for Q4.")
    store.add([{"chunk": {"id": "pdf-1", "text": "merger plans handbook", "chunk_index": 0,
                          "metadata": {"source_type": "pdf"}},
                "vector": __import__("conftest").FakeEmbedder().encode(["merger plans handbook"])[0]}])

    state = AgentState(user_id=1, user_input="merger plans")
    emails = execute_tool("search", {"query": "merger plans", "source": "email", "num_results": 10}, state)
    everything = execute_tool("search", {"query": "merger plans", "num_results": 10}, state)

    assert found_ids(emails) == {"a-1"}
    assert "b-1" not in found_ids(everything)
    assert {r["metadata"]["source_type"] for r in everything.data["results"]} == {"email", "pdf"}


def test_model_cannot_override_user_id(users, store):
    index(2, "b-1", "Bob payroll numbers.")
    state = AgentState(user_id=1, user_input="payroll")
    result = execute_tool("search", {"query": "payroll", "source": "email", "_user_id": 2, "user_id": 2}, state)
    assert "b-1" not in found_ids(result)


def test_search_without_user_context_returns_no_email(users, store):
    from agent.tools import search
    index(1, "a-1", "Alice notes.")
    result = search({"query": "Alice notes", "source": "email"})
    assert result.data["results"] == []


def test_same_gmail_message_id_in_two_mailboxes_is_kept_apart(users, store):
    index(1, "same-id", "Alice version.")
    index(2, "same-id", "Bob version.")
    assert store.has_email("same-id", owner_user_id=1)
    assert store.has_email("same-id", owner_user_id=2)
    texts = {r["text"] for r in store.similarity_search("version", k=10, source_type="email", owner_user_id=2)}
    assert texts == {"Bob version."}


def test_tool_schemas_do_not_expose_user_id():
    from agent.tools import search, send_email_reply
    for tool in (search, send_email_reply):
        props = tool.schema["parameters"]["properties"]
        assert not any("user" in name for name in props)


def test_credentials_never_reach_the_llm(gmail, llm, monkeypatch):
    seen = []
    original = llm.__call__

    def recording(messages, system_prompt, tool_names=None):
        seen.append(repr(messages) + system_prompt)
        return original(messages, system_prompt, tool_names)

    import agent.controller
    monkeypatch.setattr(agent.controller, "call_llm", recording)
    incoming = gmail.add_message(1, sender="zed@example.com")
    gmail_autoresponder.process_message(1, incoming.id, owner_address=OWNER_A)

    text = "".join(seen)
    for secret in ("token-1", "refresh-1", "test-client-secret", "push-secret"):
        assert secret not in text


def test_gmail_client_uses_each_users_own_credentials(users, monkeypatch):
    from unittest.mock import MagicMock
    from services import gmail_service

    used_tokens = []

    def fake_build(api, version, credentials, cache_discovery):
        used_tokens.append(credentials.token)
        service = MagicMock()
        service.users.return_value.getProfile.return_value.execute.return_value = {"emailAddress": "me@example.com"}
        service.users.return_value.messages.return_value.send.return_value.execute.return_value = {"id": "s1"}
        return service

    monkeypatch.setattr(gmail_service, "build", fake_build)
    gmail_service.send_email(2, to="a@example.com", subject="s", body="b")
    gmail_service.send_email(1, to="a@example.com", subject="s", body="b")

    assert used_tokens == ["token-2", "token-1"]


def test_user_without_google_account_cannot_send(db, users):
    from services import gmail_service
    from services.gmail_auth_service import AuthError
    add_user_without_google = 3
    from database.models import User
    with db._session() as session:  # pylint: disable=protected-access
        session.add(User(id=add_user_without_google, name="c", email="c@example.com", password_hash="x"))
        session.commit()
    with pytest.raises(AuthError):
        gmail_service.send_email(3, to="a@example.com", subject="s", body="b")


def test_mailbox_relinked_to_new_user_moves_notifications(db, users):
    add_user(db, 3, "carl@example.com")
    db.save_gmail_watch(3, email_address=OWNER_A, history_id=500, expiration=None)
    assert db.find_gmail_sync_state_by_email(OWNER_A).user_id == 3
    assert db.find_gmail_sync_state(1) is None


# ----------------------------------------------------------------------
# Loader parsing
# ----------------------------------------------------------------------


def _raw(headers, labels=("INBOX",)):
    import base64
    return {
        "threadId": "t1", "labelIds": list(labels), "internalDate": "1790000000000",
        "payload": {"mimeType": "text/plain",
                    "headers": [{"name": k, "value": v} for k, v in headers.items()],
                    "body": {"data": base64.urlsafe_b64encode(b"Hello there").decode()}},
    }


@pytest.mark.parametrize("headers,automated", [
    ({"From": "Carol <carol@example.com>", "To": "me@example.com"}, False),
    ({"From": "carol@example.com", "To": "me@example.com", "Auto-Submitted": "auto-replied"}, True),
    ({"From": "carol@example.com", "To": "me@example.com", "Auto-Submitted": "no"}, False),
    ({"From": "news@example.com", "To": "me@example.com", "List-Unsubscribe": "<mailto:u@x>"}, True),
    ({"From": "noreply@example.com", "To": "me@example.com"}, True),
    ({"From": "carol@example.com", "To": "me@example.com", "Precedence": "bulk"}, True),
])
def test_loader_flags_automated_mail(headers, automated):
    from loaders.gmail_loader import GmailLoader
    email = GmailLoader("token")._parse_email("m1", _raw(headers))  # pylint: disable=protected-access
    assert email.is_automated is automated
    assert email.label_ids == ["INBOX"]
    assert email.thread_id == "t1"


def test_loader_falls_back_to_delivered_to_for_bcc_mail():
    from loaders.gmail_loader import GmailLoader
    email = GmailLoader("token")._parse_email(  # pylint: disable=protected-access
        "m1", _raw({"From": "carol@example.com", "Delivered-To": "me@example.com"}))
    assert email.recipients == ["me@example.com"]


# ----------------------------------------------------------------------
# Settings / status endpoints
# ----------------------------------------------------------------------


@pytest.fixture
def api(users):
    from routers import gmail_push
    from services.app_auth_service import get_current_user

    app = FastAPI()
    app.include_router(gmail_push.router)
    current = {"user": SimpleNamespace(id=1)}
    app.dependency_overrides[get_current_user] = lambda: current["user"]
    client = TestClient(app)
    client.current = current
    return client


def test_autoresponder_setting_persists_per_user(api, db):
    assert api.get("/api/v1/gmail/automation").json()["autoresponder_enabled"] is True
    response = api.put("/api/v1/gmail/autoresponder", json={"enabled": False})
    assert response.status_code == 200 and response.json()["autoresponder_enabled"] is False
    assert db.get_autoresponder_enabled(1) is False
    assert db.get_autoresponder_enabled(2) is True


def test_status_reports_watch_and_activity_without_message_ids(api, gmail, llm):
    incoming = gmail.add_message(1, sender="amy@example.com", subject="Hours?")
    gmail_autoresponder.process_message(1, incoming.id, owner_address=OWNER_A)

    data = api.get("/api/v1/gmail/automation").json()

    assert data["google_connected"] and data["push_configured"]
    assert data["watch"]["status"] == "active" and data["watch"]["email_address"] == OWNER_A
    item = data["recent_activity"][0]
    assert item["status"] == "replied" and item["reply_sent"] and item["indexed"]
    assert item["subject"] == "Hours?"
    assert incoming.id not in repr(data)


def test_watch_endpoints(api, gmail, db, monkeypatch):
    from config import settings
    assert api.delete("/api/v1/gmail/watch").json()["watch"]["status"] == "stopped"
    assert api.post("/api/v1/gmail/watch").json()["watch"]["status"] == "active"

    monkeypatch.setattr(settings, "GMAIL_PUBSUB_TOPIC", "")
    assert api.post("/api/v1/gmail/watch").status_code == 503


def test_app_registers_push_routes_and_drops_manual_message_id_endpoint():
    import main
    paths = set(main.app.openapi()["paths"])
    assert {"/api/v1/gmail/push", "/api/v1/gmail/automation", "/api/v1/gmail/autoresponder",
            "/api/v1/gmail/watch", "/api/v1/ingest", "/api/v1/chat"} <= paths
    assert "/api/v1/gmail/incoming" not in paths
