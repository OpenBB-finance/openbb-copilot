from unittest.mock import AsyncMock

import pytest

from openbb_ada.services import (
    ContextService,
    LoggingService,
    McpDataService,
    SqlAgentService,
    TemplateService,
)


@pytest.fixture
def test_context_service(
    test_template_service: TemplateService, test_sql_agent_service: SqlAgentService
) -> ContextService:
    """Provide a ContextService instance configured for MCP tests."""
    return ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
    )


@pytest.fixture
def test_mcp_data_service(
    test_context_service: ContextService, test_template_service: TemplateService
) -> McpDataService:
    """Provide an McpDataService instance for testing."""
    return McpDataService(
        context_service=test_context_service,
        logging_service=LoggingService(),
        template_service=test_template_service,
        openai_api_key=None,
    )


@pytest.mark.asyncio
async def test_process_mcp_tool_response_registers_table_artifact(
    test_mcp_data_service: McpDataService,
    test_context_service: ContextService,
):
    """Ensure tabular MCP content is turned into an artifact and stored as context."""
    raw_content = [
        {"symbol": "AAPL", "price": 150.0},
        {"symbol": "MSFT", "price": 300.0},
    ]

    events = [
        event
        async for event in test_mcp_data_service.process_mcp_tool_response(
            tool_name="finance_quotes",
            server_id="my_server",
            raw_content=raw_content,
        )
    ]

    assert len(events) == 1
    assert events[0].data.eventType == "INFO"
    assert events[0].data.artifacts[0].type == "table"
    assert len(test_context_service.structured_context) == 1
    assert test_context_service.structured_context[
        0
    ].sql_table_info.table_name.startswith("mcp_finance_quotes")


@pytest.mark.asyncio
async def test_process_mcp_tool_response_handles_error(
    test_mcp_data_service: McpDataService,
    test_context_service: ContextService,
):
    """MCP errors should surface as error SSE events without registering context."""
    raw_content = {"error_type": "RuntimeError", "content": "boom"}

    events = [
        event
        async for event in test_mcp_data_service.process_mcp_tool_response(
            tool_name="server_tool",
            server_id="server",
            raw_content=raw_content,
        )
    ]

    assert len(events) == 1
    error_event = events[0]
    assert error_event.data.eventType == "ERROR"
    assert error_event.data.message == "Unexpected error"
    assert error_event.data.details == ["boom"]
    assert test_context_service.structured_context == []


@pytest.mark.asyncio
async def test_process_mcp_tool_response_handles_mcp_spec_error(
    test_mcp_data_service: McpDataService,
    test_context_service: ContextService,
):
    """MCP spec errors with isError/content blocks should map to ERROR SSE."""
    raw_content = {
        "isError": True,
        "content": [
            {"type": "text", "text": "boom"},
            {"type": "text", "text": "missing required parameter: symbol"},
        ],
    }

    events = [
        event
        async for event in test_mcp_data_service.process_mcp_tool_response(
            tool_name="server_tool",
            server_id="server",
            raw_content=raw_content,
        )
    ]

    assert len(events) == 1
    error_event = events[0]
    assert error_event.data.eventType == "ERROR"
    assert error_event.data.message == "Unexpected error"
    assert error_event.data.details == ["boom\nmissing required parameter: symbol"]
    assert test_context_service.structured_context == []


@pytest.mark.asyncio
async def test_process_mcp_tool_response_converts_text_to_table(
    test_mcp_data_service: McpDataService,
    test_context_service: ContextService,
    monkeypatch: pytest.MonkeyPatch,
):
    """Textual MCP output should be converted to a table via AI structuring."""
    raw_content = "symbol,price\nAAPL,150\nMSFT,300"
    converted_rows = [
        {"symbol": "AAPL", "price": 150},
        {"symbol": "MSFT", "price": 300},
    ]

    monkeypatch.setattr(
        test_mcp_data_service,
        "_determine_mcp_artifact_type_with_ai",
        AsyncMock(return_value=("table", converted_rows)),
    )

    events = [
        event
        async for event in test_mcp_data_service.process_mcp_tool_response(
            tool_name="server_tool",
            server_id="server",
            raw_content=raw_content,
        )
    ]

    assert [event.data.message for event in events] == [
        "Artifact generated",
        "Converting artifact to table",
        "Table artifact generated",
    ]
    assert events[-1].data.artifacts[0].type == "table"
    assert len(test_context_service.structured_context) == 1
    assert test_context_service.structured_context[
        0
    ].sql_table_info.table_name.startswith("mcp_server_tool")


def test_determine_mcp_artifact_type_parses_json_string(
    test_mcp_data_service: McpDataService,
):
    """A JSON string containing row objects should be treated as a table."""
    artifact_type, artifact_content = (
        test_mcp_data_service._determine_mcp_artifact_type(
            '[{"symbol": "AAPL", "price": 150}]'
        )
    )

    assert artifact_type == "table"
    assert artifact_content == [{"symbol": "AAPL", "price": 150}]


def test_detect_tabular_patterns_from_repeated_keywords(
    test_mcp_data_service: McpDataService,
):
    """Repeated field names in text should flag the content as tabular."""
    repeated_fields_text = (
        '{"price": 1, "volume": 10, "price": 2, "volume": 11, "price": 3, "volume": 12}'
    )

    assert test_mcp_data_service._detect_tabular_patterns(repeated_fields_text)
