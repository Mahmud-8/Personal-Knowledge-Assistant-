"""Pub/Sub push endpoint, history sync, and watch lifecycle."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from config import settings
from conftest import OWNER_A, OWNER_B, email_chunks
from routers import gmail_push
from services import gmail_push_service
from services.gmail_push_service import InvalidNotificationError


def push_body(email_address: str, history_id, subscription: str = "projects/test/subscriptions/gmail"):
    data = base64.b64encode(json.dumps({"emailAddress": email_address, "historyId": history_id}).encode()).decode()
    return {"message": {"data": data, "messageId": "1", "publishTime": "2026-10-01T00:00:00Z"},
            "subscription": subscription}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(gmail_push.router)
    return TestClient(app)


def push(client, body, token="push-secret", headers=None):
    url = "/api/v1/gmail/push" + (f"?token={token}" if token is not None else "")
    return client.post(url, json=body, headers=headers or {})


def history_calls(gmail):
    return [c for c in gmail.calls if c[0] == "list_history"]


# ----------------------------------------------------------------------
# Notification -> discovery -> processing
# ----------------------------------------------------------------------


def test_notification_discovers_new_email_and_replies(client, gmail, llm, db, store):
    incoming = gmail.add_message(1, sender="carol@example.com", subject="Opening hours",
                                 body="What are your opening hours?")

    response = push(client, push_body(OWNER_A, 101))

    assert response.status_code == 204
    assert history_calls(gmail) == [("list_history", 1)]
    assert [s["to"] for s in gmail.sent] == ["carol@example.com"]
    assert email_chunks(store, email_id=incoming.id, owner_user_id=1)
    state = db.find_gmail_sync_state(1)
    assert state.history_id == gmail.current_history[1]
    assert state.last_notification_at is not None and state.last_sync_at is not None


def test_duplicate_notifications_send_one_reply(client, gmail, llm):
    gmail.add_message(1, sender="dave@example.com")

    for _ in range(3):
        assert push(client, push_body(OWNER_A, 101)).status_code == 204

    assert len(gmail.sent) == 1


def test_notification_history_id_is_not_trusted_as_sync_position(client, gmail, llm, db):
    """A forged/huge historyId must not skip unprocessed messages."""
    gmail.add_message(1, sender="eve@example.com")
    push(client, push_body(OWNER_A, 999_999))
    assert len(gmail.sent) == 1
    assert db.find_gmail_sync_state(1).history_id == gmail.current_history[1]


def test_notification_only_touches_the_addressed_mailbox(client, gmail, llm):
    gmail.add_message(1, sender="x@example.com")
    gmail.add_message(2, sender="y@example.com")

    push(client, push_body(OWNER_B, 101))

    assert {user for _, user in gmail.calls} == {2}
    assert [(s["user_id"], s["to"]) for s in gmail.sent] == [(2, "y@example.com")]


# ----------------------------------------------------------------------
# Request validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("token", ["wrong", "", None])
def test_push_with_bad_token_is_rejected(client, gmail, token):
    assert push(client, push_body(OWNER_A, 101), token=token).status_code == 403
    assert gmail.calls == []


def test_push_rejected_when_no_authentication_configured(client, gmail, monkeypatch):
    from pydantic import SecretStr
    monkeypatch.setattr(settings, "GMAIL_PUBSUB_VERIFICATION_TOKEN", SecretStr(""))
    assert push(client, push_body(OWNER_A, 101)).status_code == 503
    assert gmail.calls == []


def test_push_from_unexpected_subscription_is_rejected(client, gmail, monkeypatch):
    monkeypatch.setattr(settings, "GMAIL_PUBSUB_SUBSCRIPTION", "projects/test/subscriptions/gmail")
    body = push_body(OWNER_A, 101, subscription="projects/evil/subscriptions/x")
    assert push(client, body).status_code == 403
    assert push(client, push_body(OWNER_A, 101)).status_code == 204


class TestOidc:
    @pytest.fixture(autouse=True)
    def oidc(self, monkeypatch):
        from pydantic import SecretStr
        monkeypatch.setattr(settings, "GMAIL_PUBSUB_VERIFICATION_TOKEN", SecretStr(""))
        monkeypatch.setattr(settings, "GMAIL_PUBSUB_AUDIENCE", "https://app.example.com/api/v1/gmail/push")
        monkeypatch.setattr(settings, "GMAIL_PUBSUB_SERVICE_ACCOUNT", "push@proj.iam.gserviceaccount.com")
        self.claims = {"email": "push@proj.iam.gserviceaccount.com", "email_verified": True}
        self.audiences = []

        def verify(token, request, audience):
            self.audiences.append(audience)
            if token != "good-jwt":
                raise ValueError("bad signature")
            return self.claims

        monkeypatch.setattr(gmail_push_service.id_token, "verify_oauth2_token", verify)

    def test_valid_oidc_token_is_accepted(self, client, gmail, llm):
        response = push(client, push_body(OWNER_A, 101), token=None,
                        headers={"Authorization": "Bearer good-jwt"})
        assert response.status_code == 204
        assert self.audiences == ["https://app.example.com/api/v1/gmail/push"]

    def test_missing_or_invalid_oidc_token_is_rejected(self, client, gmail):
        assert push(client, push_body(OWNER_A, 101), token=None).status_code == 403
        assert push(client, push_body(OWNER_A, 101), token=None,
                    headers={"Authorization": "Bearer forged"}).status_code == 403
        assert gmail.calls == []

    def test_token_for_other_service_account_is_rejected(self, client, gmail):
        self.claims = {"email": "attacker@evil.iam.gserviceaccount.com", "email_verified": True}
        assert push(client, push_body(OWNER_A, 101), token=None,
                    headers={"Authorization": "Bearer good-jwt"}).status_code == 403


@pytest.mark.parametrize("body", [
    None,
    {},
    {"message": "nope"},
    {"message": {"data": "!!!not-base64!!!"}},
    {"message": {"data": base64.b64encode(b"not json").decode()}},
    {"message": {"data": base64.b64encode(b'{"historyId": 5}').decode()}},
    {"message": {"data": base64.b64encode(b'{"emailAddress": "a@example.com"}').decode()}},
    {"message": {"data": base64.b64encode(b'{"emailAddress": "a@example.com", "historyId": "x"}').decode()}},
])
def test_malformed_notifications_are_acknowledged_and_ignored(client, gmail, body):
    response = client.post("/api/v1/gmail/push?token=push-secret", json=body)
    assert response.status_code == 204  # ack so Pub/Sub does not redeliver forever
    assert gmail.calls == []


def test_notification_for_unknown_mailbox_is_ignored(client, gmail):
    assert push(client, push_body("stranger@example.com", 101)).status_code == 204
    assert gmail.calls == []


def test_notification_for_stopped_watch_is_ignored(client, gmail, db):
    gmail_push_service.stop_watch(1)
    gmail.add_message(1, sender="x@example.com")
    assert push(client, push_body(OWNER_A, 101)).status_code == 204
    assert history_calls(gmail) == []


def test_parse_push_message_accepts_urlsafe_and_unpadded_data():
    raw = json.dumps({"emailAddress": "Alice@Example.com", "historyId": "12345"}).encode()
    data = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    assert gmail_push_service.parse_push_message({"message": {"data": data}}) == ("alice@example.com", 12345)
    with pytest.raises(InvalidNotificationError):
        gmail_push_service.parse_push_message({"message": {"data": base64.b64encode(
            b'{"emailAddress": "a@example.com", "historyId": 0}').decode()}})


# ----------------------------------------------------------------------
# Sync state: expired history, API errors
# ----------------------------------------------------------------------


def test_expired_history_resyncs_by_indexing_without_replying(gmail, llm, db, store):
    now = datetime.now(timezone.utc)
    db.update_gmail_sync_state(1, last_sync_at=now - timedelta(days=2))
    gmail.add_message(1, sender="ancient@example.com", date=now - timedelta(days=3))
    missed = gmail.add_message(1, sender="quinn@example.com", date=now - timedelta(hours=1))
    gmail.expired_history.add(1)

    gmail_push_service.sync_mailbox(1)

    assert gmail.sent == []
    assert email_chunks(store, email_id=missed.id, owner_user_id=1)
    state = db.find_gmail_sync_state(1)
    assert state.history_id == gmail.current_history[1]
    statuses = {r.gmail_message_id: r.status for r in db.list_gmail_reply_states(1)}
    assert statuses == {missed.id: "skipped_resync"}  # older than the last good sync: untouched

    # Later mail is handled normally again.
    gmail.expired_history.clear()
    gmail.add_message(1, sender="rita@example.com")
    gmail_push_service.sync_mailbox(1)
    assert [s["to"] for s in gmail.sent] == ["rita@example.com"]


def test_gmail_error_during_sync_keeps_pointer_and_records_error(gmail, llm, db, monkeypatch):
    from services import gmail_service
    from services.gmail_auth_service import TokenRefreshError

    gmail.add_message(1, sender="sam@example.com")

    def broken(user_id, start):
        raise TokenRefreshError("refresh token revoked")

    monkeypatch.setattr(gmail_service, "list_history", broken)
    gmail_push_service.sync_mailbox(1)

    state = db.find_gmail_sync_state(1)
    assert state.history_id == 100
    assert "revoked" in state.last_error
    assert gmail.sent == []

    # Once credentials work again the same message is still picked up.
    monkeypatch.setattr(gmail_service, "list_history", gmail.list_history)
    gmail_push_service.sync_mailbox(1)
    assert len(gmail.sent) == 1
    assert db.find_gmail_sync_state(1).last_error is None


def test_history_pointer_only_moves_forward(db, users):
    assert db.advance_gmail_history_id(1, 150)
    assert not db.advance_gmail_history_id(1, 120)
    assert db.find_gmail_sync_state(1).history_id == 150


def test_list_history_maps_404_to_history_expired(monkeypatch, users):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from googleapiclient.errors import HttpError
    from services import gmail_service

    api = MagicMock()
    api.users.return_value.history.return_value.list.return_value.execute.side_effect = HttpError(
        SimpleNamespace(status=404, reason="Not Found"), b"{}")
    monkeypatch.setattr(gmail_service, "_gmail", lambda user_id: api)

    with pytest.raises(gmail_service.HistoryExpiredError):
        gmail_service.list_history(1, 5)


def test_list_history_collects_added_messages_across_pages(monkeypatch, users):
    from unittest.mock import MagicMock
    from services import gmail_service

    pages = [
        {"history": [{"messagesAdded": [{"message": {"id": "a"}}, {"message": {"id": "b"}}]}],
         "historyId": "110", "nextPageToken": "p2"},
        {"history": [{"messagesAdded": [{"message": {"id": "b"}}]}, {"messagesAdded": [{"message": {"id": "c"}}]}],
         "historyId": "120"},
    ]
    api = MagicMock()
    api.users.return_value.history.return_value.list.return_value.execute.side_effect = pages
    monkeypatch.setattr(gmail_service, "_gmail", lambda user_id: api)

    assert gmail_service.list_history(1, 100) == (["a", "b", "c"], 120)


# ----------------------------------------------------------------------
# Watch lifecycle and renewal
# ----------------------------------------------------------------------


def test_expiring_watch_is_renewed_and_keeps_history(gmail, db):
    soon = datetime.now(timezone.utc) + timedelta(hours=2)
    db.update_gmail_sync_state(1, watch_expiration=soon, history_id=140)

    gmail_push_service.run_maintenance()

    assert ("start_watch", 1) in gmail.calls
    assert ("start_watch", 2) not in gmail.calls  # user 2's watch is far from expiry
    state = db.find_gmail_sync_state(1)
    assert state.watch_expiration.replace(tzinfo=timezone.utc) > soon
    assert state.history_id == 140  # renewal never skips unprocessed history


def test_expired_watch_is_renewed_and_missed_mail_processed(gmail, llm, db):
    db.update_gmail_sync_state(1, watch_expiration=datetime.now(timezone.utc) - timedelta(days=1))
    gmail.add_message(1, sender="tom@example.com")

    gmail_push_service.run_maintenance()

    assert ("start_watch", 1) in gmail.calls
    assert [s["to"] for s in gmail.sent] == ["tom@example.com"]


def test_failed_renewal_marks_watch_error(gmail, db, monkeypatch):
    from services import gmail_service
    from services.gmail_service import GmailApiError

    db.update_gmail_sync_state(1, watch_expiration=datetime.now(timezone.utc))

    def broken(user_id, **kwargs):
        raise GmailApiError("topic permission denied")

    monkeypatch.setattr(gmail_service, "start_watch", broken)
    gmail_push_service.run_maintenance()

    state = db.find_gmail_sync_state(1)
    assert state.watch_status == "error" and "permission" in state.last_error


def test_restarting_a_stopped_watch_does_not_replay_old_mail(gmail, llm, db):
    gmail_push_service.stop_watch(1)
    gmail.add_message(1, sender="old@example.com")  # arrives while stopped

    gmail_push_service.ensure_watch(1)
    gmail_push_service.sync_mailbox(1)

    assert gmail.sent == []
    assert db.find_gmail_sync_state(1).watch_status == "active"


def test_watch_requires_topic(gmail, monkeypatch):
    monkeypatch.setattr(settings, "GMAIL_PUBSUB_TOPIC", "")
    with pytest.raises(gmail_push_service.PushNotConfiguredError):
        gmail_push_service.ensure_watch(1)
