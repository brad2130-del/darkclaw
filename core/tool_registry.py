"""
darkclaw/core/tool_registry.py
Tool definitions and dispatch — s02 from Claude Code harness analysis.

Every tool registers here. The agent loop calls dispatch() when
the model returns a tool_use stop_reason.

Tool interface:
    {
        "name": "tool_name",
        "description": "what it does",
        "input_schema": { ... JSON schema ... },
        "handler": async_fn(input) -> str
    }

Built-in tools (implement in Sprint 1):
    - memory_query: search Darkclaw for a fact
    - memory_teach: add a fact to Darkclaw
    - web_search: search the web (via LiteLLM web_search tool)
    - read_file: read a local file
    - write_file: write/append to a local file
    - send_email: send feedback email via mailto
    - check_compliance: look up a business requirement

Adding a new tool:
    registry = ToolRegistry()
    registry.register({
        "name": "my_tool",
        "description": "Does X given Y",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
        "handler": my_async_handler
    })
"""
import inspect
from typing import Any, Callable, Dict, List, Optional
from core.event_bus import emit, EventType

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, dict] = {}

    def register(self, tool: dict):
        assert "name" in tool and "handler" in tool
        self._tools[tool["name"]] = tool

    def get_schemas(self) -> List[dict]:
        """Return tool schemas for LiteLLM tools= parameter."""
        return [
            {"type": "function", "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t.get("input_schema", {})
            }}
            for t in self._tools.values()
        ]

    async def dispatch(self, agent_id: str, tool_name: str, tool_input: dict) -> str:
        """Call a tool and return its string result."""
        if tool_name not in self._tools:
            return f"Error: unknown tool '{tool_name}'"

        # Dangerous-command gate: no tool may run interpreter/shell-escape
        # inputs without human approval, including tools added later.
        from core.guardrails import guard_tool_input
        refusal = guard_tool_input(tool_name, tool_input)
        if refusal:
            emit(EventType.HEALTH_WARN, agent_id,
                 issue="dangerous tool input blocked",
                 tool=tool_name, input_preview=str(tool_input)[:80])
            return refusal

        emit(EventType.AGENT_TOOL_CALL, agent_id,
             tool=tool_name, input_preview=str(tool_input)[:80])
        try:
            handler = self._tools[tool_name]["handler"]
            # Handlers that gate on *who* is calling (the approval queue needs
            # it to attribute a request) declare an agent_id parameter. Keep
            # the plain handler(input) contract for everyone else rather than
            # smuggling agent_id inside tool_input, where a model could forge it.
            if "agent_id" in inspect.signature(handler).parameters:
                result = await handler(tool_input, agent_id=agent_id)
            else:
                result = await handler(tool_input)
            emit(EventType.AGENT_TOOL_RESULT, agent_id,
                 tool=tool_name, result_preview=str(result)[:80])
            return str(result)
        except Exception as e:
            return f"Tool error ({tool_name}): {e}"
