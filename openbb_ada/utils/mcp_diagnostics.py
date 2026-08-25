from typing import Any, Callable, Sequence

from openbb_ai.models import (
    AgentTool,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
)


def summarize_mcp_tool(tool: AgentTool) -> dict[str, Any]:
    input_schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
    properties = input_schema.get("properties")
    required_raw = input_schema.get("required")

    parameter_names = (
        sorted(str(name) for name in properties.keys())
        if isinstance(properties, dict)
        else []
    )
    required_args = (
        [str(name) for name in required_raw if isinstance(name, str)]
        if isinstance(required_raw, list)
        else []
    )

    return {
        "server_id": tool.server_id,
        "tool_name": tool.name,
        "schema_available": bool(parameter_names or required_args),
        "required_args": required_args,
        "parameter_names": parameter_names,
    }


def summarize_mcp_tools(
    tools: Sequence[AgentTool],
    max_tools_to_log: int = 50,
) -> tuple[list[dict[str, Any]], bool]:
    tool_summaries = [summarize_mcp_tool(tool) for tool in tools[:max_tools_to_log]]
    return tool_summaries, len(tools) > max_tools_to_log


def restore_last_failed_mcp_call(
    messages: Sequence[LlmClientFunctionCallResultMessage | LlmClientMessage],
    *,
    normalize_content: Callable[[Any], Any],
    is_mcp_error: Callable[[Any], bool],
    get_mcp_error_content: Callable[[Any], str | None],
) -> tuple[dict[str, Any] | None, str | None]:
    if not messages:
        return None, None

    last_message = messages[-1]
    if not isinstance(last_message, LlmClientFunctionCallResultMessage):
        return None, None
    if last_message.function != "execute_agent_tool":
        return None, None
    if not isinstance(last_message.extra_state, dict):
        return None, None

    diagnostics = last_message.extra_state.get("mcp_tool_diagnostics")
    if not isinstance(diagnostics, dict):
        return None, None

    all_raw_content = []
    for data in last_message.data:
        for item in getattr(data, "items", []):
            item_content = getattr(item, "content", None)
            if item_content:
                all_raw_content.append(normalize_content(item_content))

    raw_content = all_raw_content[-1] if all_raw_content else None
    if raw_content is None or not is_mcp_error(raw_content):
        return None, None

    error_details = get_mcp_error_content(raw_content)
    restored_diagnostics = dict(diagnostics)
    restored_diagnostics["error_details"] = error_details or "Unknown MCP error"
    return restored_diagnostics, restored_diagnostics["error_details"]


def build_emitted_mcp_tool_call(
    *,
    function_name: str,
    response_arguments: dict[str, Any],
    get_original_tool_info: Callable[[str], tuple[str, str] | None],
    make_stable_hash: Callable[[Any], str],
) -> tuple[dict[str, Any], str]:
    original_info = get_original_tool_info(function_name)
    if original_info:
        server_id, tool_name = original_info
        normalized_tool_args = {
            key: value
            for key, value in response_arguments.items()
            if key != "display_summary" and value is not None
        }
    else:
        tool_name = str(response_arguments.get("tool_name", ""))
        server_id = str(response_arguments.get("server_id", ""))
        tool_args = response_arguments.get("tool_args")
        normalized_tool_args = tool_args if isinstance(tool_args, dict) else {}

    return (
        {
            "tool_name": tool_name,
            "server_id": server_id,
            "provided_arg_keys": sorted(normalized_tool_args.keys()),
        },
        make_stable_hash(normalized_tool_args),
    )
