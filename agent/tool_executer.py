"""
This is the piece that implements your core idea: the LLM only ever
names a tool, the code decides what running that tool means.

TOOL_REGISTRY maps a tool name (string) to a callable. Adding a tool is
a one-line registration here — no if/elif chain to extend, no touching
controller.py. Each tool function's signature is the contract teammate B
builds every tool against:

    def some_tool(arguments: dict) -> ToolResult:
        ...

Register new tools at the bottom of this file, or import them from
testing_tools.py / a future tools/weather.py etc. and add them to the dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from agent.tool_result import ToolResult
from agent.tools import search, send_email_reply

if TYPE_CHECKING:
    from agent.state import AgentState


ToolFn = Callable[[dict], ToolResult]


TOOL_REGISTRY: dict[str, ToolFn] = {
    "search": search,
    "send_email_reply": send_email_reply,
}


def _prepare_arguments(name: str, arguments: dict, state: AgentState | None) -> dict:
    """Inject application context the LLM must never supply itself.

    Keys starting with "_" are internal: anything the model sent under
    such a name is dropped before the real values are injected.
    """
    args = {k: v for k, v in arguments.items() if not k.startswith("_")}

    if state is None:
        return args

    if name == "search":
        # Email results are scoped to the authenticated user's mailbox.
        args["_user_id"] = state.user_id
        # Email history for an incoming email must come from emails only,
        # and not be the email that is being answered.
        if state.incoming_email is not None:
            args["source"] = "email"
            args["_exclude_email_id"] = state.incoming_email.id

    if name == "send_email_reply":
        args["_user_id"] = state.user_id
        args["_incoming_email"] = state.incoming_email

    return args


def execute_tool(name: str, arguments: dict, state: AgentState | None = None) -> ToolResult:
    """Execute a registered tool by name, injecting internal context from `state`."""

    tool_fn = TOOL_REGISTRY.get(name)

    if tool_fn is None:
        return ToolResult(
            tool_name=name,
            success=False,
            error=f"Unknown tool: '{name}'. Available: {list(TOOL_REGISTRY)}",
        )

    try:
        return tool_fn(_prepare_arguments(name, arguments, state))

    except Exception as e:
        # A broken tool must not crash the whole agent loop.
        return ToolResult(
            tool_name=name,
            success=False,
            error=str(e),
        )
