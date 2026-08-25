import json

from openbb_ai.models import (
    AgentTool,
    DataContent,
    LlmClientFunctionCallResultMessage,
    SingleDataContent,
)

from openbb_ada.services import LoggingService
from openbb_ada.utils.mcp_diagnostics import (
    build_emitted_mcp_tool_call,
    restore_last_failed_mcp_call,
    summarize_mcp_tools,
)


def _make_agent_tool(
    name: str,
    server_id: str,
    required: list[str] | None = None,
    properties: dict | None = None,
) -> AgentTool:
    if properties is None:
        properties = {}
        for arg in required or []:
            properties[arg] = {"type": "string", "description": f"The {arg}"}
    return AgentTool(
        name=name,
        server_id=server_id,
        url="",
        description="Test tool",
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    )


def test_summarize_mcp_tools_truncates_and_marks_truncated():
    tools = [
        _make_agent_tool(name=f"tool_{idx}", server_id="server_1", required=["table"])
        for idx in range(3)
    ]

    summaries, truncated = summarize_mcp_tools(tools, max_tools_to_log=2)

    assert len(summaries) == 2
    assert truncated is True
    assert summaries[0]["required_args"] == ["table"]
    assert summaries[0]["parameter_names"] == ["table"]


def test_restore_last_failed_mcp_call_returns_error_details():
    message = LlmClientFunctionCallResultMessage(
        function="execute_agent_tool",
        data=[
            DataContent(
                items=[
                    SingleDataContent(
                        content=json.dumps(
                            {"error": "Missing required argument: table"}
                        )
                    )
                ]
            )
        ],
        extra_state={
            "mcp_tool_diagnostics": {
                "tool_name": "mcp_list_columns",
                "server_id": "server_1",
                "tool_args_hash": "abc123",
            }
        },
    )

    diagnostics, error_summary = restore_last_failed_mcp_call(
        [message],
        normalize_content=json.loads,
        is_mcp_error=lambda value: isinstance(value, dict) and "error" in value,
        get_mcp_error_content=lambda value: value.get("error"),
    )

    assert diagnostics is not None
    assert diagnostics["tool_name"] == "mcp_list_columns"
    assert diagnostics["error_details"] == "Missing required argument: table"
    assert error_summary == "Missing required argument: table"


def test_build_emitted_mcp_tool_call_for_flat_tool_ignores_display_summary():
    emitted_tool_call, tool_args_hash = build_emitted_mcp_tool_call(
        function_name="mcp_server_1_list_columns",
        response_arguments={
            "table": "prices",
            "display_summary": "Listing columns",
            "limit": 10,
        },
        get_original_tool_info=lambda name: ("server_1", "mcp_list_columns"),
        make_stable_hash=LoggingService.make_stable_hash,
    )

    assert emitted_tool_call == {
        "tool_name": "mcp_list_columns",
        "server_id": "server_1",
        "provided_arg_keys": ["limit", "table"],
    }
    assert isinstance(tool_args_hash, str)
    assert tool_args_hash
