"""
Conversation state for the agentic LLM loop.

Tracks the message history in the OpenAI messages format
(list of dicts with "role" and "content" keys).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import Email


class AgentState:
    """Holds the authenticated user context and conversation history."""

    def __init__(
        self,
        user_id: int,
        user_input: str,
        messages: list[dict] | None = None,
        incoming_email: Email | None = None,
    ) -> None:
        self.user_id = user_id
        self.user_input = user_input
        # Set when the agent is handling a new incoming email (not chat).
        self.incoming_email = incoming_email
        # Results of successful send_email_reply calls / last send failure.
        self.sent_emails: list[dict] = []
        self.send_email_error: str | None = None
        # False once a send failed in a way that may have delivered the email.
        self.send_email_retryable: bool = True
        self.messages: list[dict] = list(messages) if messages else []
        self.messages.append({
            "role": "user",
            "content": user_input,
        })

    def add_message(self, role: str, content: str) -> None:
        """Append a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
        })

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Append a tool result to the conversation history."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })