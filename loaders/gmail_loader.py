"""
gmail_loader.py

Concrete `EmailLoader` implementation that connects to the Gmail API and
returns emails as validated `Email` model objects.

Data flow:
    Gmail Server  --(GmailLoader)-->  List[Email]

This module only reads and structures emails. It performs no cleaning,
chunking, embedding, storage, or retrieval — those belong to later
stages of the RAG pipeline.

Authentication
--------------
Uses OAuth tokens obtained by the auth service (stored in PostgreSQL).
The loader receives an access_token and optional refresh_token — it does
NOT run its own OAuth consent flow. If the access token is expired, it
refreshes using the client credentials from config.py.
"""

from __future__ import annotations

import base64
import logging
import re
from datetime import datetime, timezone
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import Resource, build
from googleapiclient.errors import HttpError

from config import settings
from models import Attachment, Email
from .base_email_loader import (
    EmailLoader,
    EmailLoaderConnectionError,
    EmailLoaderFetchError,
    EmailLoaderParseError,
)

logger = logging.getLogger(__name__)

DEFAULT_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Senders that never read replies (RFC 3834 recommends not auto-replying).
_NO_REPLY_SENDER = re.compile(
    r"^(no-?reply|do-?not-?reply|mailer-daemon|postmaster|bounces?)([+.@-]|$)",
    re.IGNORECASE,
)


class GmailLoader(EmailLoader):
    """
    Loads emails from a Gmail account via the Gmail API.

    Parameters
    ----------
    access_token:
        OAuth access token from the auth service.
    refresh_token:
        OAuth refresh token for renewing expired access tokens.
    max_emails:
        Maximum number of emails `fetch_emails()` will return.
    label:
        Gmail label or folder to read from (e.g. "INBOX", "SENT",
        a custom label name, or a label ID).
    credentials:
        Ready-to-use Google credentials. When given, they are used as-is
        instead of building credentials from the tokens.
    """

    def __init__(
        self,
        access_token: str,
        refresh_token: Optional[str] = None,
        max_emails: int = 50,
        label: str = "INBOX",
        credentials: Optional[Credentials] = None,
    ) -> None:
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.max_emails = max_emails
        self.label = label
        self._provided_creds = credentials

        self._service: Optional[Resource] = None
        self._creds: Optional[Credentials] = None

    @property
    def is_connected(self) -> bool:
        """Check if the loader is currently connected to the Gmail API."""
        return self._service is not None

    # ------------------------------------------------------------------
    # EmailLoader interface
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Build credentials from stored tokens and connect to Gmail API."""
        logger.info("Connecting to Gmail API...")
        try:
            creds = self._build_credentials()
            self._creds = creds
            self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
            logger.info("Gmail API connection established.")
        except EmailLoaderConnectionError:
            raise
        except Exception as exc:
            raise EmailLoaderConnectionError(f"Failed to connect to Gmail: {exc}") from exc

    def disconnect(self) -> None:
        """Release the Gmail API service and cached credentials."""
        if self._service is not None:
            try:
                self._service.close()
            except Exception as exc:
                logger.debug("Error while closing Gmail service: %s", exc)
        self._service = None
        self._creds = None
        logger.info("Disconnected from Gmail API.")

    def fetch_emails(self) -> List[Email]:
        """
        Fetch up to `max_emails` emails from `label` and return them as
        validated `Email` objects. Individual messages that fail to
        download or parse are logged and skipped rather than aborting
        the whole batch.
        """
        if not self.is_connected:
            raise EmailLoaderConnectionError("Not connected. Call connect() first.")

        emails: List[Email] = []
        message_ids = self._get_message_ids()

        for message_id in message_ids:
            try:
                raw_message = self._download_message(message_id)
                email_obj = self._parse_email(message_id, raw_message)
                emails.append(email_obj)
            except Exception as exc:
                logger.warning("Skipping message %s due to error: %s", message_id, exc)
                continue

        logger.info(
            "Fetched %d/%d emails from Gmail (label=%s).",
            len(emails),
            len(message_ids),
            self.label,
        )
        return emails

    def fetch_email(self, message_id: str) -> Email:
        """Fetch and parse a single email by its Gmail message ID."""
        if not self.is_connected:
            raise EmailLoaderConnectionError("Not connected. Call connect() first.")

        raw_message = self._download_message(message_id)
        try:
            return self._parse_email(message_id, raw_message)
        except Exception as exc:
            raise EmailLoaderParseError(f"Failed to parse message {message_id}: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_credentials(self) -> Credentials:
        """Build Google credentials from stored tokens, refreshing if needed."""
        if self._provided_creds is not None:
            return self._provided_creds

        creds = Credentials(
            token=self.access_token,
            refresh_token=self.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET.get_secret_value(),
            scopes=DEFAULT_SCOPES,
        )

        if not creds.valid:
            if creds.expired and creds.refresh_token:
                logger.info("Refreshing expired Gmail token...")
                creds.refresh(Request())
                self.access_token = creds.token
            else:
                raise EmailLoaderConnectionError(
                    "Access token is invalid and no refresh token is available."
                )

        return creds

    def _get_message_ids(self) -> List[str]:
        """Retrieve up to `max_emails` message IDs for `label`, paginating as needed."""
        assert self._service is not None
        message_ids: List[str] = []
        page_token: Optional[str] = None

        try:
            while len(message_ids) < self.max_emails:
                remaining = self.max_emails - len(message_ids)
                response = (
                    self._service.users()
                    .messages()
                    .list(
                        userId="me",
                        labelIds=[self.label] if self.label else None,
                        maxResults=min(remaining, 500),
                        pageToken=page_token,
                    )
                    .execute()
                )
                message_ids.extend(m["id"] for m in response.get("messages", []))
                page_token = response.get("nextPageToken")
                if not page_token:
                    break
        except HttpError as exc:
            raise EmailLoaderFetchError(f"Failed to list Gmail messages: {exc}") from exc

        return message_ids[: self.max_emails]

    def _download_message(self, message_id: str) -> Dict[str, Any]:
        """Download the full raw message payload for a single message ID."""
        assert self._service is not None
        try:
            return (
                self._service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
        except HttpError as exc:
            raise EmailLoaderFetchError(f"Failed to fetch message {message_id}: {exc}") from exc

    def _extract_headers(self, payload: Dict[str, Any]) -> Dict[str, str]:
        """Flatten Gmail's header list into a lowercase-keyed dict."""
        return {h["name"].lower(): h["value"] for h in payload.get("headers", [])}

    def _extract_body(self, payload: Dict[str, Any]) -> str:
        """Extract the message body, preferring text/plain over text/html."""
        plain_text = self._find_body_by_mime_type(payload, "text/plain")
        if plain_text:
            return plain_text

        html_text = self._find_body_by_mime_type(payload, "text/html")
        return html_text or ""

    def _find_body_by_mime_type(self, payload: Dict[str, Any], mime_type: str) -> Optional[str]:
        """Recursively search MIME parts for the first match of `mime_type`."""
        if payload.get("mimeType") == mime_type:
            data = payload.get("body", {}).get("data")
            if data:
                return self._decode_base64(data)

        for part in payload.get("parts", []) or []:
            result = self._find_body_by_mime_type(part, mime_type)
            if result:
                return result

        return None

    def _extract_attachments(self, payload: Dict[str, Any]) -> List[Attachment]:
        """Recursively collect attachment metadata from the MIME tree."""
        attachments: List[Attachment] = []
        self._collect_attachments(payload, attachments)
        return attachments

    def _collect_attachments(
        self, payload: Dict[str, Any], attachments: List[Attachment]
    ) -> None:
        filename = payload.get("filename")
        body = payload.get("body", {})

        if filename and body.get("attachmentId"):
            attachments.append(
                Attachment(
                    filename=filename,
                    content_type=payload.get("mimeType"),
                    size=body.get("size"),
                )
            )

        for part in payload.get("parts", []) or []:
            self._collect_attachments(part, attachments)

    def _parse_email(self, message_id: str, raw_message: Dict[str, Any]) -> Email:
        """Convert a raw Gmail API message into a validated `Email` object."""
        payload = raw_message.get("payload", {})
        headers = self._extract_headers(payload)

        subject = headers.get("subject", "")
        _, sender_addr = parseaddr(headers.get("from", ""))
        recipients = self._parse_recipients(headers.get("to", ""))
        if not recipients:
            # e.g. Bcc-only mail: fall back to the receiving mailbox.
            recipients = self._parse_recipients(headers.get("delivered-to", ""))
        date = self._parse_date(headers.get("date"), raw_message.get("internalDate"))
        body = self._extract_body(payload)
        attachments = self._extract_attachments(payload)

        return Email(
            id=message_id,
            subject=subject,
            sender=sender_addr,
            recipients=recipients,
            body=body,
            date=date,
            attachments=attachments,
            thread_id=raw_message.get("threadId"),
            rfc_message_id=headers.get("message-id"),
            label_ids=list(raw_message.get("labelIds") or []),
            is_automated=self._is_automated(headers, sender_addr),
        )

    @staticmethod
    def _is_automated(headers: Dict[str, str], sender_addr: str) -> bool:
        """True for auto-replies, bulk/list mail and no-reply senders."""
        auto_submitted = headers.get("auto-submitted", "").strip().lower()
        if auto_submitted and auto_submitted != "no":
            return True
        if headers.get("precedence", "").strip().lower() in {"bulk", "junk", "list", "auto_reply"}:
            return True
        if "list-id" in headers or "list-unsubscribe" in headers:
            return True
        if "x-autoreply" in headers or "x-autorespond" in headers:
            return True
        return bool(_NO_REPLY_SENDER.match(sender_addr or ""))

    @staticmethod
    def _parse_recipients(to_header: str) -> List[str]:
        if not to_header:
            return []
        parsed = [parseaddr(addr)[1] for addr in to_header.split(",")]
        return [addr for addr in parsed if addr]

    @staticmethod
    def _parse_date(date_header: Optional[str], internal_date_ms: Optional[str]) -> datetime:
        if date_header:
            try:
                return parsedate_to_datetime(date_header)
            except (TypeError, ValueError):
                pass

        if internal_date_ms:
            return datetime.fromtimestamp(int(internal_date_ms) / 1000, tz=timezone.utc)

        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _decode_base64(data: str) -> str:
        try:
            decoded_bytes = base64.urlsafe_b64decode(data.encode("utf-8"))
            return decoded_bytes.decode("utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Failed to decode message body: %s", exc)
            return ""
