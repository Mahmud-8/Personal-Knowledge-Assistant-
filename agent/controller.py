"""
Agentic orchestrator loop.

Takes a conversation state, repeatedly calls the LLM, executes any
requested tools, and returns the final text answer. The controller
never talks to the provider SDK directly — that's llm_client.py's job.
"""

from __future__ import annotations

import json
import logging

from agent.state import AgentState
from agent.llm_client import call_llm, LLMResponseError
from agent.tool_executer import TOOL_REGISTRY, execute_tool
from agent.tool_result import ToolResult
from agent.system_prompt import SYSTEM_PROMPT


logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10

# Keep very long email bodies from blowing up the prompt.
MAX_EMAIL_BODY_CHARS = 8000

# Tools that act on an incoming email; never offered during ordinary chat.
EMAIL_ONLY_TOOLS = {"send_email_reply"}


def _available_tools(state: AgentState) -> list[str]:
    if state.incoming_email is not None:
        return list(TOOL_REGISTRY)
    return [name for name in TOOL_REGISTRY if name not in EMAIL_ONLY_TOOLS]


def build_incoming_email_prompt(email) -> str:
    """Build the structured "New email" message the agent receives."""
    body = email.body or ""
    if len(body) > MAX_EMAIL_BODY_CHARS:
        body = body[:MAX_EMAIL_BODY_CHARS] + "\n[... truncated]"
    return (
        "New email\n"
        f"Sender: {email.sender}\n"
        f"Subject: {email.subject}\n"
        f"Date: {email.date.isoformat()}\n"
        f"Body:\n{body}"
    )


def run(state: AgentState) -> str:
    """
    Run the agent loop until the model produces a final answer.
    Returns the final text response to the user.
    """
    tool_names = _available_tools(state)
    for round_num in range(MAX_TOOL_ROUNDS):
        try:
            message = call_llm(state.messages, SYSTEM_PROMPT, tool_names)
        except LLMResponseError as e:
            logger.error("LLM call failed: %s", e)
            return f"Sorry, something went wrong: {e}"

        # No tool calls — model is done
        if not message.tool_calls:
            return message.content or "I have no response."

        # Append the assistant message (with tool_calls) to history
        # Only include fields Groq accepts — exclude SDK extras like "annotations"
        assistant_msg = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        }
        state.messages.append(assistant_msg)

        # Execute each tool call
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = None

            logger.info("Tool call: %s(%s)", name, args)

            if not isinstance(args, dict):
                result = ToolResult(
                    tool_name=name,
                    success=False,
                    error="Tool arguments must be a JSON object.",
                )
            elif name not in tool_names:
                result = ToolResult(
                    tool_name=name,
                    success=False,
                    error=f"Tool '{name}' is not available here.",
                )
            elif name == "send_email_reply" and state.sent_emails:
                result = ToolResult(
                    tool_name=name,
                    success=False,
                    error="A reply has already been sent for this email.",
                )
            else:
                # Internal context (user_id, incoming email) is injected
                # by the tool execution layer, never by the LLM.
                result = execute_tool(name, args, state)
                if name == "send_email_reply":
                    if result.success:
                        state.sent_emails.append(result.data)
                    else:
                        state.send_email_error = result.error
                        # Only failures that provably sent nothing may be retried.
                        if not (result.data or {}).get("retryable"):
                            state.send_email_retryable = False
            logger.info("Tool result: %s", result.to_observation_text())

            state.add_tool_result(
                tool_call_id=tool_call.id,
                content=result.to_observation_text(),
            )

    return "I exceeded the maximum number of tool calls. Please try again."