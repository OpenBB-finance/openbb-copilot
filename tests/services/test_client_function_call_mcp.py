"""
Tests for MCP tool call validation and flat MCP tool factory in
ClientFunctionCallService.

Tests the short-circuit behavior when required arguments are missing,
preventing unnecessary round-trips to MCP servers, and the dynamic
flat-function generation that replaces the meta-tool pattern.
"""

import inspect
from unittest.mock import MagicMock, Mock

import pytest
from openbb_ai.models import (
    AgentTool,
    FunctionCallResponse,
    LlmClientFunctionCallResultMessage,
    StatusUpdateSSE,
)

from openbb_ada.errors import FunctionCallError
from openbb_ada.services import ClientFunctionCallService, LoggingService
from openbb_ada.utils.utils import sanitize_tool_name


@pytest.fixture
def client_function_call_service() -> ClientFunctionCallService:
    """Create a ClientFunctionCallService instance for testing."""
    return ClientFunctionCallService(
        copilot_data_service=Mock(),
        logging_service=LoggingService(),
    )


def _make_agent_tool(
    name: str = "qtap duckdb mcp_list_columns",
    server_id: str = "1771593417468",
    required: list[str] | None = None,
    properties: dict | None = None,
) -> AgentTool:
    """Create an AgentTool with a given input schema."""
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


class TestMcpToolCallMissingRequiredArgs:
    """Tests for short-circuiting MCP calls when required args are missing."""

    @pytest.mark.asyncio
    async def test_raises_function_call_error_when_required_arg_missing(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Calling mcp_list_columns without 'table' should raise immediately."""
        tool = _make_agent_tool(required=["table"])

        with pytest.raises(FunctionCallError) as exc_info:
            async for _ in client_function_call_service.llm_execute_agent_tool(
                server_id="1771593417468",
                tool_name="qtap duckdb mcp_list_columns",
                tool_args={},
                summary="Listing columns for some table",
                available_tools=[tool],
            ):
                pass

        error_msg = str(exc_info.value)
        assert "table" in error_msg


class TestGenerativeWidgetFunctionCalls:
    @pytest.mark.asyncio
    async def test_llm_generate_widget_in_dashboard_errors_for_unresolved_artifact(
        self, client_function_call_service: ClientFunctionCallService
    ):
        events = []
        async for (
            event
        ) in client_function_call_service.llm_generate_widget_in_dashboard(
            widget_type="table",
            artifact_id="query_artifact_dsge",
            summary="Adding widget to dashboard",
        ):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], StatusUpdateSSE)
        assert events[0].data.eventType == "ERROR"
        assert "artifact directly" in events[0].data.message

    @pytest.mark.asyncio
    async def test_raises_when_required_arg_is_empty_string(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """A required arg with empty string value should also be rejected."""
        tool = _make_agent_tool(required=["table"])

        with pytest.raises(FunctionCallError) as exc_info:
            async for _ in client_function_call_service.llm_execute_agent_tool(
                server_id="1771593417468",
                tool_name="qtap duckdb mcp_list_columns",
                tool_args={"table": ""},
                summary="Listing columns",
                available_tools=[tool],
            ):
                pass

        assert "table" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_when_required_arg_is_none(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """A required arg with None value should also be rejected."""
        tool = _make_agent_tool(required=["table"])

        with pytest.raises(FunctionCallError) as exc_info:
            async for _ in client_function_call_service.llm_execute_agent_tool(
                server_id="1771593417468",
                tool_name="qtap duckdb mcp_list_columns",
                tool_args={"table": None},
                summary="Listing columns",
                available_tools=[tool],
            ):
                pass

        assert "table" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_succeeds_when_required_arg_provided(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Providing required args should not raise — yields events normally."""
        tool = _make_agent_tool(required=["table"])

        events = []
        async for event in client_function_call_service.llm_execute_agent_tool(
            server_id="1771593417468",
            tool_name="qtap duckdb mcp_list_columns",
            tool_args={"table": "fct_btc_transfer_flows"},
            summary="Listing columns for fct_btc_transfer_flows",
            available_tools=[tool],
        ):
            events.append(event)

        # Should yield StatusUpdateSSE + FunctionCallResponse
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_succeeds_when_no_required_args(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Tools with no required args should work with empty tool_args."""
        tool = _make_agent_tool(required=[])

        events = []
        async for event in client_function_call_service.llm_execute_agent_tool(
            server_id="1771593417468",
            tool_name="qtap duckdb mcp_list_tables",
            tool_args={},
            summary="Listing tables",
            available_tools=[tool],
        ):
            events.append(event)

        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_succeeds_when_no_available_tools_for_validation(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """When available_tools is None, skip validation (no schema to check)."""
        events = []
        async for event in client_function_call_service.llm_execute_agent_tool(
            server_id="1771593417468",
            tool_name="qtap duckdb mcp_list_columns",
            tool_args={},
            summary="Listing columns",
            available_tools=None,
        ):
            events.append(event)

        # No schema to validate against — should proceed
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_raises_lists_all_missing_args(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Error message should list all missing required args."""
        tool = _make_agent_tool(required=["database", "schema", "table"])

        with pytest.raises(FunctionCallError) as exc_info:
            async for _ in client_function_call_service.llm_execute_agent_tool(
                server_id="1771593417468",
                tool_name="qtap duckdb mcp_list_columns",
                tool_args={"database": "api"},
                summary="Listing columns",
                available_tools=[tool],
            ):
                pass

        error_msg = str(exc_info.value)
        assert "schema" in error_msg
        assert "table" in error_msg
        # database was provided, should not be listed as missing
        assert (
            "database" not in error_msg or "database" in error_msg.split("requires")[0]
        )


class TestHandleFunctionCallsCatchesMissingArgsError:
    """Tests that handle_function_calls converts FunctionCallError to LLM messages."""

    @pytest.mark.asyncio
    async def test_handle_function_calls_yields_error_messages(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """When llm_execute_agent_tool raises FunctionCallError,
        handle_function_calls should yield [AssistantMessage, FunctionResultMessage]."""
        from magentic import FunctionCall

        def _dummy(**kwargs):
            return kwargs

        # Use a real FunctionCall so magentic's _unique_id is valid
        real_fc = FunctionCall(_dummy)

        async def _fc_gen():
            raise FunctionCallError("Missing required argument: table")
            yield  # noqa: E501 — makes this an async generator

        mock_response = MagicMock(wraps=real_fc)
        mock_response.function = MagicMock()
        mock_response.function.__name__ = "llm_execute_agent_tool"
        mock_response.function.__qualname__ = "llm_execute_agent_tool"
        mock_response.arguments = {}
        mock_response.return_value = _fc_gen()
        mock_response._unique_id = real_fc._unique_id

        events = []
        async for event in client_function_call_service.handle_function_calls(
            mock_response
        ):
            events.append(event)

        # Should yield a list of [AssistantMessage, FunctionResultMessage]
        assert len(events) == 1
        assert isinstance(events[0], list)
        assert len(events[0]) == 2
        assert "Missing required argument: table" in events[0][1].content


# =====================================================================
# Tests for sanitize_tool_name
# =====================================================================


class TestSanitizeToolName:
    def test_basic_sanitization(self):
        result = sanitize_tool_name("qtap duckdb mcp_list_columns")
        assert result == "mcp_qtap_duckdb_mcp_list_columns"

    def test_replaces_dashes_and_dots(self):
        result = sanitize_tool_name("my-server.tool-name")
        assert result == "mcp_my_server_tool_name"

    def test_truncates_long_names(self):
        long_name = "a" * 100
        result = sanitize_tool_name(long_name, max_length=30)
        assert len(result) <= 30

    def test_collision_handling(self):
        existing = {"mcp_my_tool"}
        result = sanitize_tool_name("my_tool", existing_names=existing)
        assert result != "mcp_my_tool"
        assert result.startswith("mcp_my_tool")
        assert result == "mcp_my_tool_2"

    def test_prefix_added(self):
        result = sanitize_tool_name("list_tables")
        assert result.startswith("mcp_")

    def test_no_double_underscores(self):
        result = sanitize_tool_name("my  tool--name")
        assert "__" not in result


# =====================================================================
# Tests for build_flat_mcp_tools
# =====================================================================


class TestBuildFlatMcpTools:
    def test_creates_correct_number_of_functions(
        self, client_function_call_service: ClientFunctionCallService
    ):
        tools = [
            _make_agent_tool(name="mcp_list_tables", server_id="s1", required=[]),
            _make_agent_tool(
                name="mcp_list_columns",
                server_id="s1",
                required=["table"],
            ),
            _make_agent_tool(name="mcp_query", server_id="s1", required=["query"]),
        ]
        flat_fns = client_function_call_service.build_flat_mcp_tools(tools)
        assert len(flat_fns) == 3

    def test_function_names_are_sanitized(
        self, client_function_call_service: ClientFunctionCallService
    ):
        tools = [
            _make_agent_tool(
                name="qtap duckdb mcp_list_columns",
                server_id="s1",
                required=["table"],
            ),
        ]
        flat_fns = client_function_call_service.build_flat_mcp_tools(tools)
        assert flat_fns[0].__name__ == "mcp_qtap_duckdb_mcp_list_columns"

    def test_function_has_correct_signature(
        self, client_function_call_service: ClientFunctionCallService
    ):
        tools = [
            _make_agent_tool(
                name="mcp_query",
                server_id="s1",
                required=["query"],
                properties={
                    "query": {
                        "type": "string",
                        "description": "SQL query",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows",
                    },
                },
            ),
        ]
        flat_fns = client_function_call_service.build_flat_mcp_tools(tools)
        sig = inspect.signature(flat_fns[0])
        param_names = list(sig.parameters.keys())

        # display_summary should be first, then tool params
        assert param_names[0] == "display_summary"
        assert "query" in param_names
        assert "limit" in param_names

        # display_summary is optional (has default)
        assert sig.parameters["display_summary"].default == "Executing MCP tool"
        # query is required (no default), limit is optional (has default)
        assert sig.parameters["query"].default is inspect.Parameter.empty
        assert sig.parameters["limit"].default is None

    def test_populates_dynamic_registry(
        self, client_function_call_service: ClientFunctionCallService
    ):
        tools = [
            _make_agent_tool(name="mcp_list_tables", server_id="s1", required=[]),
        ]
        client_function_call_service.build_flat_mcp_tools(tools)

        sanitized = "mcp_mcp_list_tables"
        assert sanitized in client_function_call_service._dynamic_mcp_tool_names
        info = client_function_call_service.get_original_tool_info(sanitized)
        assert info is not None
        assert info == ("s1", "mcp_list_tables")

    def test_is_client_function_call_recognizes_dynamic_tools(
        self, client_function_call_service: ClientFunctionCallService
    ):
        tools = [
            _make_agent_tool(name="mcp_list_tables", server_id="s1", required=[]),
        ]
        flat_fns = client_function_call_service.build_flat_mcp_tools(tools)

        # Create a mock FunctionCall with the dynamic function name
        mock_response = MagicMock()
        mock_response.function = flat_fns[0]
        assert client_function_call_service.is_client_function_call(mock_response)


class TestFlatMcpToolExecution:
    @pytest.mark.asyncio
    async def test_flat_tool_yields_status_and_function_call(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Flat MCP tool should yield StatusUpdateSSE + FunctionCallResponse."""
        tools = [
            _make_agent_tool(
                name="mcp_list_columns",
                server_id="s1",
                required=["table"],
            ),
        ]
        flat_fns = client_function_call_service.build_flat_mcp_tools(tools)
        fn = flat_fns[0]

        events = []
        async for event in fn(
            display_summary="Listing columns for my_table",
            table="my_table",
        ):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], StatusUpdateSSE)
        assert isinstance(events[1], FunctionCallResponse)
        # FunctionCallResponse should have execute_agent_tool as function
        assert events[1].function == "execute_agent_tool"
        assert events[1].input_arguments["tool_name"] == "mcp_list_columns"
        assert events[1].input_arguments["server_id"] == "s1"
        assert events[1].input_arguments["parameters"] == {"table": "my_table"}

    @pytest.mark.asyncio
    async def test_flat_tool_raises_on_missing_required_arg(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Flat tool with missing required arg should raise FunctionCallError."""
        tools = [
            _make_agent_tool(
                name="mcp_list_columns",
                server_id="s1",
                required=["table"],
            ),
        ]
        flat_fns = client_function_call_service.build_flat_mcp_tools(tools)
        fn = flat_fns[0]

        with pytest.raises(FunctionCallError) as exc_info:
            async for _ in fn(
                display_summary="Listing columns",
            ):
                pass

        assert "table" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_flat_tool_with_no_params(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Tool with no parameters should work with just display_summary."""
        tools = [
            _make_agent_tool(name="mcp_list_tables", server_id="s1", required=[]),
        ]
        flat_fns = client_function_call_service.build_flat_mcp_tools(tools)
        fn = flat_fns[0]

        events = []
        async for event in fn(display_summary="Listing tables"):
            events.append(event)

        assert len(events) == 2
        assert events[1].input_arguments["parameters"] == {}


class TestGetFunctionCallSpecFlatMcpRemapping:
    """Verify that get_function_call_spec remaps execute_agent_tool
    history entries to the corresponding flat MCP function."""

    def _make_history_message(
        self,
        server_id: str,
        tool_name: str,
        tool_args: dict | None = None,
        summary: str = "Executing MCP tool",
    ) -> LlmClientFunctionCallResultMessage:
        return LlmClientFunctionCallResultMessage(
            function="execute_agent_tool",
            data=[],
            extra_state={
                "copilot_function_call_arguments": {
                    "server_id": server_id,
                    "summary": summary,
                    "tool_name": tool_name,
                    "tool_args": tool_args or {},
                },
            },
        )

    def test_exact_key_match(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Remapping by exact (server_id, tool_name) key."""
        tools = [
            _make_agent_tool(name="mcp_list_tables", server_id="s1", required=[]),
        ]
        flat_fns = client_function_call_service.build_flat_mcp_tools(tools)

        msg = self._make_history_message(server_id="s1", tool_name="mcp_list_tables")
        fn, args = client_function_call_service.get_function_call_spec(msg)

        assert fn is flat_fns[0]
        assert args["display_summary"] == "Executing MCP tool"

    def test_fallback_by_tool_name_alone(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """When server_id differs, fallback matching by tool_name should work."""
        tools = [
            _make_agent_tool(name="mcp_list_tables", server_id="s1", required=[]),
        ]
        flat_fns = client_function_call_service.build_flat_mcp_tools(tools)

        # Historical message has a different server_id
        msg = self._make_history_message(
            server_id="old_server_999", tool_name="mcp_list_tables"
        )
        fn, args = client_function_call_service.get_function_call_spec(msg)

        assert fn is flat_fns[0]

    def test_preserves_tool_args(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Remapped args should include both display_summary and tool_args."""
        tools = [
            _make_agent_tool(
                name="mcp_list_columns", server_id="s1", required=["table"]
            ),
        ]
        flat_fns = client_function_call_service.build_flat_mcp_tools(tools)

        msg = self._make_history_message(
            server_id="s1",
            tool_name="mcp_list_columns",
            tool_args={"table": "my_table"},
            summary="Listing columns",
        )
        fn, args = client_function_call_service.get_function_call_spec(msg)

        assert fn is flat_fns[0]
        assert args["display_summary"] == "Listing columns"
        assert args["table"] == "my_table"

    def test_falls_back_to_legacy_when_no_flat_match(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """When no flat function matches, falls through to legacy get_client_tool."""
        # Don't register any flat tools — registry is empty
        msg = self._make_history_message(server_id="s1", tool_name="unknown_tool")
        fn, args = client_function_call_service.get_function_call_spec(msg)

        # Should fall through to the legacy execute_agent_tool handler
        assert fn.__name__ == "llm_execute_agent_tool"
