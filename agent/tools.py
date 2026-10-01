"""
Tool implementations for the agentic LLM.

Every function must:
  1. Accept a single `arguments: dict`.
  2. Return a `ToolResult` — never raise for expected failures.
  3. Be pure with respect to AgentState.
  4. Get registered in `tool_executer.py`'s TOOL_REGISTRY.
"""

from __future__ import annotations

import base64
from email.message import EmailMessage

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from agent.tool_result import ToolResult
from config import settings
from initializer import db, store

VALID_SOURCES = {"email", "slack", "discord", "telegram", "pdf"}


def search(arguments: dict) -> ToolResult:
    """Search the vector store, optionally filtered by source type."""
    query = arguments.get("query")

    if not query:
        return ToolResult(
            tool_name="search",
            success=False,
            error="Missing required argument: query",
        )

    source = arguments.get("source")

    if source and source not in VALID_SOURCES:
        return ToolResult(
            tool_name="search",
            success=False,
            error=(
                f"Invalid source: '{source}'. "
                f"Must be one of: {', '.join(sorted(VALID_SOURCES))}"
            ),
        )

    # Number of results to return, defaulting to 2 if not specified.
    k = arguments.get("num_results", 2)

    results = store.similarity_search(
        query,
        k=k,
        source_type=source,
    )

    if not results:
        return ToolResult(
            tool_name="search",
            success=True,
            data={
                "results": [],
                "message": "No matching documents found.",
            },
        )

    return ToolResult(
        tool_name="search",
        success=True,
        data={"results": results},
    )


search.schema = {
    "description": (
        "Search ingested content by semantic similarity. "
        "Set 'source' to filter by platform."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "source": {
                "type": "string",
                "enum": [
                    "email",
                    "slack",
                    "discord",
                    "telegram",
                    "pdf",
                ],
                "description": "Filter by source type. Omit to search all.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results (default 5).",
            },
        },
        "required": ["query"],
    },
}


# ------------------------------------------------------------------
# Gmail send email reply tool
# ------------------------------------------------------------------


def send_email_reply(arguments: dict) -> ToolResult:
    """
    Send an email using the authenticated user's connected Gmail account.

    `_user_id` is injected internally by the controller and is not part
    of the LLM-facing tool schema.
    """
    user_id = arguments.get("_user_id")
    to = arguments.get("to")
    subject = arguments.get("subject", "")
    body = arguments.get("body", "")

    # --------------------------------------------------------------
    # Validate arguments
    # --------------------------------------------------------------

    if user_id is None:
        return ToolResult(
            tool_name="send_email_reply",
            success=False,
            error="Missing authenticated user context.",
        )

    if not to:
        return ToolResult(
            tool_name="send_email_reply",
            success=False,
            error="Missing required argument: to",
        )

    if not body:
        return ToolResult(
            tool_name="send_email_reply",
            success=False,
            error="Missing required argument: body",
        )

    # --------------------------------------------------------------
    # Get connected Gmail credentials
    # --------------------------------------------------------------

    credentials_row = db.find_google_credentials(user_id)

    if credentials_row is None:
        return ToolResult(
            tool_name="send_email_reply",
            success=False,
            error="No Google account is connected to this user.",
        )

    try:
        # Reconstruct Google OAuth credentials from the tokens stored
        # in PostgreSQL.
        creds = Credentials(
            token=credentials_row.access_token,
            refresh_token=credentials_row.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.GOOGLE_CLIENT_ID,
            client_secret=settings.GOOGLE_CLIENT_SECRET.get_secret_value(),
            scopes=settings.google_scopes,
        )

        # ----------------------------------------------------------
        # Refresh expired access token when possible
        # ----------------------------------------------------------

        if not creds.valid:
            if not creds.refresh_token:
                return ToolResult(
                    tool_name="send_email_reply",
                    success=False,
                    error=(
                        "Gmail access token is invalid and "
                        "no refresh token is available."
                    ),
                )

            creds.refresh(Request())

            # Persist refreshed credentials so the next request can
            # use the new access token.
            db.save_google_credentials(
                user_id,
                google_id=credentials_row.google_id,
                access_token=creds.token,
                refresh_token=(
                    creds.refresh_token
                    or credentials_row.refresh_token
                ),
                token_expiry=creds.expiry,
            )

        # ----------------------------------------------------------
        # Build MIME email
        # ----------------------------------------------------------

        message = EmailMessage()
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)

        raw_message = base64.urlsafe_b64encode(
            message.as_bytes()
        ).decode("utf-8")

        # ----------------------------------------------------------
        # Send through Gmail API
        # ----------------------------------------------------------

        service = build(
            "gmail",
            "v1",
            credentials=creds,
            cache_discovery=False,
        )

        result = (
            service.users()
            .messages()
            .send(
                userId="me",
                body={"raw": raw_message},
            )
            .execute()
        )

        profile = (
            service.users()
            .getProfile(userId="me")
            .execute()
        )

        sender_email = profile.get("emailAddress")

        return ToolResult(
            tool_name="send_email_reply",
            success=True,
            data={
                "message_id": result.get("id"),
                "sender": sender_email,
                "to": to,
                "subject": subject,
                "body": body,
                "message": "Email sent successfully.",
            },
        )

    except Exception as exc:
        # Do not expose tokens or credentials in the response.
        logger = __import__("logging").getLogger(__name__)
        logger.exception("Failed to send Gmail message.")

        return ToolResult(
            tool_name="send_email_reply",
            success=False,
            error=f"Failed to send email: {exc}",
        )


send_email_reply.schema = {
    "description": (
        "Send a reply email from the authenticated user's "
        "connected Gmail account."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Recipient email address.",
            },
            "subject": {
                "type": "string",
                "description": "Email subject line.",
            },
            "body": {
                "type": "string",
                "description": "Plain-text email body.",
            },
        },
        "required": ["to", "subject", "body"],
    },
}