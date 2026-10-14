"""
Tool implementations for the agentic LLM.

Every function must:
  1. Accept a single `arguments: dict`.
  2. Return a `ToolResult` — never raise for expected failures.
  3. Be pure with respect to AgentState.
  4. Get registered in `tool_executer.py`'s TOOL_REGISTRY.
"""

from __future__ import annotations

from agent.tool_result import ToolResult
from initializer import store
from services import gmail_service
from services.gmail_service import GmailSendError
from services.gmail_auth_service import AuthError

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
        owner_user_id=arguments.get("_user_id"),
        exclude_email_id=arguments.get("_exclude_email_id"),
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


def reply_subject(original_subject: str | None) -> str:
    """'Re: <original subject>', without stacking a second 'Re:'."""
    subject = (original_subject or "").strip()
    if subject.lower().startswith("re:"):
        return subject
    return f"Re: {subject}".strip()


def _send_failed(error: str) -> ToolResult:
    """A failure raised before anything was handed to Gmail (safe to retry)."""
    return ToolResult(
        tool_name="send_email_reply",
        success=False,
        data={"retryable": True},
        error=error,
    )


def send_email_reply(arguments: dict) -> ToolResult:
    """
    Send a reply to the incoming email being handled, from the user's
    connected Gmail account.

    `_user_id` and `_incoming_email` are injected by the tool execution
    layer and are not part of the LLM-facing tool schema.
    """
    user_id = arguments.get("_user_id")
    incoming_email = arguments.get("_incoming_email")
    to = (arguments.get("to") or "").strip()
    body = arguments.get("body") or ""

    if user_id is None:
        return _send_failed("Missing authenticated user context.")

    if incoming_email is None:
        return _send_failed("send_email_reply can only be used while handling an incoming email.")

    if not to:
        return _send_failed("Missing required argument: to")

    # A reply always goes back to the original sender; this also stops
    # instructions inside an email body from redirecting the reply.
    if to.lower() != incoming_email.sender.lower():
        return _send_failed(f"Recipient must be the original sender: {incoming_email.sender}")

    if not body.strip():
        return _send_failed("Missing required argument: body")

    # The subject is derived from the original so the reply always threads
    # correctly, whatever the model put in the `subject` argument.
    subject = reply_subject(incoming_email.subject)

    try:
        sent = gmail_service.send_email(
            user_id,
            to=to,
            subject=subject,
            body=body,
            thread_id=incoming_email.thread_id,
            in_reply_to=incoming_email.rfc_message_id,
        )
    # Both errors mean Gmail did not accept the message, so a later
    # retry cannot produce a duplicate.
    except AuthError as exc:
        return _send_failed(f"Gmail authentication failed: {exc}")
    except GmailSendError as exc:
        return _send_failed(f"Failed to send email: {exc}")

    return ToolResult(
        tool_name="send_email_reply",
        success=True,
        data={
            **sent,
            "to": to,
            "subject": subject,
            "body": body,
            "message": "Email sent successfully.",
        },
    )


send_email_reply.schema = {
    "description": (
        "Send a reply to the incoming email from the user's connected "
        "Gmail account. Only use this when handling a 'New email' message."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Recipient email address (the original sender).",
            },
            "subject": {
                "type": "string",
                "description": "Reply subject line, e.g. 'Re: <original subject>'.",
            },
            "body": {
                "type": "string",
                "description": "Plain-text reply body.",
            },
        },
        "required": ["to", "subject", "body"],
    },
}
