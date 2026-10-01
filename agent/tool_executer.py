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
 
from typing import Callable 
 
from agent.tool_result import ToolResult 
from agent.tools import search, send_email_reply 
 
 
ToolFn = Callable[[dict], ToolResult] 
 
 
TOOL_REGISTRY: dict[str, ToolFn] = { 
    "search": search, 
    "send_email_reply": send_email_reply, 
} 
 
 
def execute_tool(name: str, arguments: dict) -> ToolResult: 
    """Execute a registered tool by name.""" 
 
    tool_fn = TOOL_REGISTRY.get(name) 
 
    if tool_fn is None: 
        return ToolResult( 
            tool_name=name, 
            success=False, 
            error=f"Unknown tool: '{name}'. Available: {list(TOOL_REGISTRY)}", 
        ) 
 
    try: 
        return tool_fn(arguments) 
 
    except Exception as e: 
        # A broken tool must not crash the whole agent loop. 
        return ToolResult( 
            tool_name=name, 
            success=False, 
            error=str(e), 
        ) 