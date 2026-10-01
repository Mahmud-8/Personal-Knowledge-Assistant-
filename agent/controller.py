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
from agent.tool_executer import execute_tool
from agent.system_prompt import SYSTEM_PROMPT


def build_incoming_email_prompt(email) -> str:
    """Build structured context for an incoming email."""
    return (
        "Incoming email:\n"
        f"Sender: {email.sender}\n"
        f"Subject: {email.subject}\n"
        f"Body:\n{email.body}\n\n"
        "Handle this email according to the system instructions."
    )


logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10


def run(state: AgentState) -> str:
    """
    Run the agent loop until the model produces a final answer.
    Returns the final text response to the user.
    """
    for round_num in range(MAX_TOOL_ROUNDS):
        try:
            message = call_llm(state.messages, SYSTEM_PROMPT)
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
            args = json.loads(tool_call.function.arguments)

            # Pass the authenticated user context to tools that need it.
            if name == "send_email_reply":
                args["_user_id"] = state.user_id

            logger.info("Tool call: %s(%s)", name, args)

            result = execute_tool(name, args)
            logger.info("Tool result: %s", result.to_observation_text())

            state.add_tool_result(
                tool_call_id=tool_call.id,
                content=result.to_observation_text(),
            )

    return "I exceeded the maximum number of tool calls. Please try again."