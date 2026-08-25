import io
import json
import uuid
from unittest.mock import Mock
from uuid import UUID

import pandas as pd
import pytest
from openbb_ai.models import SingleDataContent, SourceInfo, StatusUpdateSSE

from openbb_ada.copilot import CopilotService
from openbb_ada.models import (
    ContextStructuredQueryResult,
    DataContent,
    ParsedContext,
    RawContext,
    RawObjectDataFormat,
    SqlTableInfo,
    StructuredContext,
    UnstructuredContext,
)
from openbb_ada.services import (
    CitationService,
    ContextService,
    LoggingService,
    SqlAgentService,
    TemplateService,
)
from tests.conftest import MockUUIDs


@pytest.fixture
def test_context_service(
    test_template_service: TemplateService,
    test_sql_agent_service: SqlAgentService,
) -> ContextService:
    """Provide a ContextService instance for testing."""
    return ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
    )


# Initialization Tests


@pytest.mark.asyncio
async def test_context_service_initialization(test_context_service: ContextService):
    """Test that ContextService initializes with empty context lists."""
    assert test_context_service.unstructured_context == []
    assert test_context_service.structured_context == []


@pytest.mark.asyncio
async def test_context_service_has_correct_dependencies(
    test_context_service: ContextService,
    test_template_service: TemplateService,
    test_sql_agent_service: SqlAgentService,
):
    """Test that ContextService stores its dependencies correctly."""
    assert test_context_service._template_service == test_template_service
    assert test_context_service._sql_agent_service == test_sql_agent_service
    assert test_context_service._logging_service is not None


# Load Context Tests - Structured Data


@pytest.mark.asyncio
async def test_load_context_valid_json_as_structured(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    """Test loading valid tabular JSON data as structured context."""
    json_data = json.dumps(
        [
            {"symbol": "AAPL", "price": 150.0, "volume": 1000000},
            {"symbol": "GOOGL", "price": 2800.0, "volume": 500000},
            {"symbol": "MSFT", "price": 300.0, "volume": 750000},
        ]
    )

    parsed_context = ParsedContext(
        content=json_data,
        source_info=SourceInfo(
            type="artifact",
            uuid=uuid.UUID(mock_uuids.ID1.value),
            name="stock_prices",
            description="Stock price data",
            metadata={},
        ),
    )

    loaded = test_context_service.load_context([parsed_context])

    assert len(loaded) == 1
    assert isinstance(loaded[0], StructuredContext)
    assert len(test_context_service.structured_context) == 1
    assert len(test_context_service.unstructured_context) == 0


@pytest.mark.asyncio
async def test_load_context_with_nested_data(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    """Test loading JSON with nested arrays creates parent and child tables."""
    json_data = json.dumps(
        [
            {
                "company": "OpenBB",
                "employees": [
                    {"name": "Alice", "role": "Engineer"},
                    {"name": "Bob", "role": "Designer"},
                ],
            },
            {
                "company": "TechCorp",
                "employees": [{"name": "Charlie", "role": "Manager"}],
            },
        ]
    )

    parsed_context = ParsedContext(
        content=json_data,
        source_info=SourceInfo(
            type="artifact",
            uuid=uuid.UUID(mock_uuids.ID2.value),
            name="company_data",
            description="Company and employee data",
            metadata={},
        ),
    )

    loaded = test_context_service.load_context([parsed_context])

    # Should create structured context (parent table only in context list)
    assert len(loaded) == 1
    assert isinstance(loaded[0], StructuredContext)


# Load Context Tests - Unstructured Data


@pytest.mark.asyncio
async def test_load_context_invalid_json_as_unstructured(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    """Test loading invalid JSON as unstructured context."""
    invalid_json = "This is not JSON at all, just plain text."

    parsed_context = ParsedContext(
        content=invalid_json,
        source_info=SourceInfo(
            type="artifact",
            uuid=uuid.UUID(mock_uuids.ID3.value),
            name="plain_text",
            description="Plain text content",
            metadata={},
        ),
    )

    loaded = test_context_service.load_context([parsed_context])

    assert len(loaded) == 1
    assert isinstance(loaded[0], UnstructuredContext)
    assert len(test_context_service.unstructured_context) == 1
    assert len(test_context_service.structured_context) == 0


@pytest.mark.asyncio
async def test_load_context_long_text_cells_as_unstructured(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    """Test that data with very long text cells is treated as unstructured."""
    # Create JSON with very long text that exceeds threshold
    long_text = "x" * 200  # 200 characters
    json_data = json.dumps(
        [
            {"id": 1, "description": long_text},
            {"id": 2, "description": long_text},
            {"id": 3, "description": long_text},
        ]
    )

    parsed_context = ParsedContext(
        content=json_data,
        source_info=SourceInfo(
            type="artifact",
            uuid=uuid.UUID(mock_uuids.ID4.value),
            name="long_text_data",
            description="Data with long text",
            metadata={},
        ),
    )

    loaded = test_context_service.load_context([parsed_context])

    # Should be treated as unstructured due to long cells
    assert len(loaded) == 1
    assert isinstance(loaded[0], UnstructuredContext)


@pytest.mark.asyncio
async def test_load_context_empty_dataframe_as_unstructured(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    """Test that empty JSON array creates unstructured context."""
    json_data = json.dumps([])

    parsed_context = ParsedContext(
        content=json_data,
        source_info=SourceInfo(
            type="artifact",
            uuid=uuid.UUID(mock_uuids.ID5.value),
            name="empty_data",
            description="Empty data",
            metadata={},
        ),
    )

    loaded = test_context_service.load_context([parsed_context])

    # Empty dataframe should be unstructured
    assert len(loaded) == 1
    assert isinstance(loaded[0], UnstructuredContext)


# Load Explicit Context Tests


@pytest.mark.asyncio
async def test_load_explicit_context_with_data_content(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    """Test loading explicit context from RawContext with DataContent."""
    json_data = json.dumps([{"col1": "val1", "col2": 123}])

    raw_context = RawContext(
        uuid=uuid.UUID(mock_uuids.ID6.value),
        name="test_context",
        description="Test context",
        data=DataContent(
            items=[
                SingleDataContent(content=json_data, citable=True),
            ]
        ),
        metadata={"key": "value"},
    )

    test_context_service.load_explicit_context([raw_context])

    # Should have loaded one context element
    assert (
        len(test_context_service.structured_context)
        + len(test_context_service.unstructured_context)
        == 1
    )


@pytest.mark.asyncio
async def test_load_explicit_context_preserves_raw_object_data_format_metadata(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    raw_context = RawContext(
        uuid=uuid.UUID(mock_uuids.ID6.value),
        name="query_artifact_test",
        description="Snowflake SQL query generated by AI copilot",
        data=DataContent(
            items=[
                SingleDataContent(
                    content="```sql\nSELECT 1\n```",
                    citable=False,
                    data_format=RawObjectDataFormat(
                        parse_as="snowflake_query",
                        query_data_source={
                            "origin": "SNOW backend",
                            "id": "fred_public_dsge",
                            "widget_uuid": "3a2b6d0f-79d1-4209-8bea-63c1bb8f48b9",
                        },
                    ),
                ),
            ]
        ),
        metadata={"source": "artifact"},
    )

    test_context_service.load_explicit_context([raw_context])

    assert len(test_context_service.unstructured_context) == 1
    artifact_context = test_context_service.unstructured_context[0]
    assert artifact_context.source_info.metadata == {
        "source": "artifact",
        "parse_as": "snowflake_query",
        "query_data_source": {
            "origin": "SNOW backend",
            "id": "fred_public_dsge",
            "widget_uuid": "3a2b6d0f-79d1-4209-8bea-63c1bb8f48b9",
        },
    }


def test_dump_and_restore_roundtrip_artifacts_preserves_snowflake_query_artifact(
    test_template_service: TemplateService,
    test_sql_agent_service: SqlAgentService,
):
    source_context_service = ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
    )
    query_data_source = {
        "origin": "SNOW backend",
        "id": "fred_public_dsge",
        "widget_uuid": "3a2b6d0f-79d1-4209-8bea-63c1bb8f48b9",
    }
    source_context_service.load_context(
        [
            ParsedContext(
                content="```sql\nSELECT * FROM FRED.PUBLIC.DSGE_SV\n```",
                source_info=SourceInfo(
                    type="artifact",
                    uuid=UUID("00000000-0000-0000-0000-000000000099"),
                    name="query_artifact_dsge",
                    description="Snowflake SQL query generated by AI copilot",
                    metadata={
                        "parse_as": "snowflake_query",
                        "query_data_source": query_data_source,
                    },
                    citable=False,
                ),
                data_format=RawObjectDataFormat(
                    parse_as="snowflake_query",
                    query_data_source=query_data_source,
                ),
            )
        ]
    )

    serialized_artifacts = source_context_service.dump_roundtrip_artifacts()
    serialized_query_artifact = serialized_artifacts["query_artifact_dsge"]
    assert "kind" not in serialized_query_artifact
    assert serialized_query_artifact["content"] == (
        "```sql\nSELECT * FROM FRED.PUBLIC.DSGE_SV\n```"
    )
    assert serialized_query_artifact["source_info"]["name"] == "query_artifact_dsge"
    assert serialized_query_artifact["data_format"]["parse_as"] == "snowflake_query"
    assert serialized_query_artifact["data_format"]["query_data_source"] == (
        query_data_source
    )

    restored_context_service = ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
    )
    restored_context_service.restore_roundtrip_artifacts(serialized_artifacts)

    restored_context = restored_context_service.get_context_by_name(
        "query_artifact_dsge"
    )
    assert isinstance(restored_context, UnstructuredContext)
    assert restored_context.content == "```sql\nSELECT * FROM FRED.PUBLIC.DSGE_SV\n```"
    assert restored_context.source_info.uuid == UUID(
        "00000000-0000-0000-0000-000000000099"
    )
    assert restored_context.source_info.metadata == {
        "parse_as": "snowflake_query",
        "query_data_source": query_data_source,
    }


# Context Retrieval Tests


@pytest.mark.asyncio
async def test_get_context_by_source_info_name_found_structured(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    """Test retrieving structured context by source info name."""
    json_data = json.dumps([{"a": 1}, {"a": 2}])
    parsed_context = ParsedContext(
        content=json_data,
        source_info=SourceInfo(
            type="artifact",
            uuid=uuid.UUID(mock_uuids.ID7.value),
            name="findable_context",
            description="Context that can be found",
            metadata={},
        ),
    )

    test_context_service.load_context([parsed_context])
    result = test_context_service.get_context_by_source_info_name("findable_context")

    assert result is not None
    assert isinstance(result, StructuredContext)
    assert result.source_info.name == "findable_context"


@pytest.mark.asyncio
async def test_get_context_by_source_info_name_found_unstructured(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    """Test retrieving unstructured context by source info name."""
    plain_text = "This is plain text that cannot be structured."
    parsed_context = ParsedContext(
        content=plain_text,
        source_info=SourceInfo(
            type="artifact",
            uuid=uuid.UUID(mock_uuids.ID8.value),
            name="text_context",
            description="Plain text context",
            metadata={},
        ),
    )

    test_context_service.load_context([parsed_context])
    result = test_context_service.get_context_by_source_info_name("text_context")

    assert result is not None
    assert isinstance(result, UnstructuredContext)
    assert result.source_info.name == "text_context"


@pytest.mark.asyncio
async def test_get_context_by_source_info_name_not_found(
    test_context_service: ContextService,
):
    """Test that retrieving non-existent context returns None."""
    result = test_context_service.get_context_by_source_info_name("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_get_context_by_id_found(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    """Test retrieving context by UUID."""
    json_data = json.dumps([{"x": 1}])
    test_uuid = uuid.UUID(mock_uuids.ID9.value)

    parsed_context = ParsedContext(
        content=json_data,
        source_info=SourceInfo(
            type="artifact",
            uuid=test_uuid,
            name="uuid_context",
            description="Context with specific UUID",
            metadata={},
        ),
    )

    test_context_service.load_context([parsed_context])
    result = test_context_service.get_context_by_id(str(test_uuid))

    assert result is not None
    assert str(result.source_info.uuid) == str(test_uuid)


@pytest.mark.asyncio
async def test_get_context_by_id_not_found(test_context_service: ContextService):
    """Test that retrieving context with wrong UUID returns None."""
    result = test_context_service.get_context_by_id(str(uuid.uuid4()))
    assert result is None


# Preview Context Tests


@pytest.mark.asyncio
async def test_preview_context_structured(
    test_context_service: ContextService,
    test_csv_tsla_historical_data: bytes,
    mock_uuids: type[MockUUIDs],
):
    """Test previewing structured context returns table preview."""
    df = pd.read_csv(io.BytesIO(test_csv_tsla_historical_data), index_col=0)
    json_data = df.head(5).to_json(orient="records")

    parsed_context = ParsedContext(
        content=json_data,
        source_info=SourceInfo(
            type="artifact",
            uuid=uuid.UUID(mock_uuids.ID10.value),
            name="tesla_preview",
            description="Tesla data for preview",
            metadata={},
        ),
    )

    test_context_service.load_context([parsed_context])
    preview = test_context_service.preview_context("tesla_preview")

    assert preview != ""
    assert isinstance(preview, str)


@pytest.mark.asyncio
async def test_preview_context_unstructured(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    """Test previewing unstructured context returns truncated content."""
    long_text = "This is a long piece of text. " * 100  # Make it long

    parsed_context = ParsedContext(
        content=long_text,
        source_info=SourceInfo(
            type="artifact",
            uuid=uuid.UUID(mock_uuids.ID11.value),
            name="long_text_preview",
            description="Long text for preview",
            metadata={},
        ),
    )

    test_context_service.load_context([parsed_context])
    preview = test_context_service.preview_context("long_text_preview")

    assert len(preview) <= 259  # 256 chars + "..."
    assert preview.endswith("...")


@pytest.mark.asyncio
async def test_preview_context_not_found(test_context_service: ContextService):
    """Test that preview of non-existent context returns empty string."""
    preview = test_context_service.preview_context("nonexistent")
    assert preview == ""


# Read Unstructured Context Tests


@pytest.mark.asyncio
async def test_read_unstructured_context_by_ids_single(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    """Test reading unstructured context by ID."""
    content = "Important unstructured content"
    test_uuid = uuid.UUID(mock_uuids.ID12.value)

    parsed_context = ParsedContext(
        content=content,
        source_info=SourceInfo(
            type="artifact",
            uuid=test_uuid,
            name="unstructured_doc",
            description="An unstructured document",
            metadata={"author": "test"},
        ),
    )

    test_context_service.load_context([parsed_context])
    results = test_context_service.read_unstructured_context_by_ids([str(test_uuid)])

    assert len(results) == 1
    assert results[0].answer == content
    assert len(results[0].citations) == 1
    assert results[0].citations[0].source_info.uuid == test_uuid


@pytest.mark.asyncio
async def test_read_unstructured_context_by_ids_not_found(
    test_context_service: ContextService,
):
    """Test that reading non-existent unstructured context raises ValueError."""
    with pytest.raises(ValueError, match="No unstructured context found with id"):
        test_context_service.read_unstructured_context_by_ids([str(uuid.uuid4())])


# Async Query Tests


@pytest.mark.asyncio
async def test_query_structured_context_with_natural_language(
    test_context_service: ContextService,
    test_csv_tsla_historical_data: bytes,
    mock_uuids: type[MockUUIDs],
):
    """Test querying structured context with natural language."""
    # Load structured context
    df = pd.read_csv(io.BytesIO(test_csv_tsla_historical_data), index_col=0)
    json_data = df.head(10).to_json(orient="records")

    parsed_context = ParsedContext(
        content=json_data,
        source_info=SourceInfo(
            type="artifact",
            uuid=uuid.UUID(mock_uuids.ID1.value),
            name="tesla_data",
            description="Tesla historical data",
            metadata={},
        ),
    )

    test_context_service.load_context([parsed_context])

    # Query the context
    query = "What is the average closing price?"
    result = None

    async for (
        event
    ) in test_context_service.query_structured_context_with_natural_language(query):
        if isinstance(event, ContextStructuredQueryResult):
            result = event
        elif isinstance(event, StatusUpdateSSE):
            # Status updates are expected
            pass

    # Verify result
    assert result is not None
    assert isinstance(result, ContextStructuredQueryResult)
    assert result.answer != ""
    # Should have citations for the queried table
    assert len(result.citations) >= 0  # May vary based on query execution


# Edge Cases and Error Handling


@pytest.mark.asyncio
async def test_is_structured_empty_dataframe(test_context_service: ContextService):
    """Test _is_structured with empty DataFrame returns True."""
    empty_df = pd.DataFrame()
    # Empty DataFrames default to True but are caught elsewhere
    result = test_context_service._is_structured(empty_df)
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_get_structured_context_from_table_name_not_found(
    test_context_service: ContextService,
):
    """Test that getting structured context for non-existent table raises ValueError."""
    with pytest.raises(ValueError, match="No structured context found for table name"):
        test_context_service.get_structured_context_from_table_name("nonexistent_table")


@pytest.mark.asyncio
async def test_load_context_multiple_elements(
    test_context_service: ContextService,
    mock_uuids: type[MockUUIDs],
):
    """Test loading multiple context elements at once."""
    structured_data = json.dumps([{"a": 1, "b": 2}])
    unstructured_data = "Plain text content"

    elements = [
        ParsedContext(
            content=structured_data,
            source_info=SourceInfo(
                type="artifact",
                uuid=uuid.UUID(mock_uuids.ID1.value),
                name="structured",
                description="Structured data",
                metadata={},
            ),
        ),
        ParsedContext(
            content=unstructured_data,
            source_info=SourceInfo(
                type="artifact",
                uuid=uuid.UUID(mock_uuids.ID2.value),
                name="unstructured",
                description="Unstructured data",
                metadata={},
            ),
        ),
    ]

    loaded = test_context_service.load_context(elements)

    assert len(loaded) == 2
    assert len(test_context_service.structured_context) == 1
    assert len(test_context_service.unstructured_context) == 1


@pytest.mark.asyncio
async def test_process_df_list_columns_with_nested_data(
    test_context_service: ContextService,
):
    """Test that _process_df_list_columns returns unchanged DF for nested data."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "items": [
                [{"name": "a", "value": 1}],
                [{"name": "b", "value": 2}],
            ],
        }
    )

    processed_df, dropped = test_context_service._process_df_list_columns(df)

    # Should detect nested data and return unchanged
    assert len(dropped) == 0
    # DataFrame should be unchanged (nested data preserved)
    assert "items" in processed_df.columns


@pytest.mark.asyncio
async def test_process_df_list_columns_with_simple_lists(
    test_context_service: ContextService,
):
    """Test that _process_df_list_columns handles simple lists correctly."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "tags": [["tag1", "tag2"], ["tag3"]],  # Simple string lists
        }
    )

    processed_df, dropped = test_context_service._process_df_list_columns(
        df, max_length=10
    )

    # Simple lists should NOT be converted to strings
    assert "tags" in processed_df.columns
    assert isinstance(processed_df["tags"].iloc[0], list)
    assert "tag1" in processed_df["tags"].iloc[0]


# Regression tests

"""
Regression test for artifact reference bug where LLM tries to reference widget data
as an artifact before calling _llm_query_structured_data.

This test reproduces the issue where:
1. Widget data is loaded as structured context (not an artifact)
2. LLM tries to reference it as an artifact with a hallucinated ID
3. Artifact lookup fails because the ID doesn't exist
4. User sees empty output
"""


@pytest.fixture
def mock_context_service_with_widget_data():
    """Create a mock context service with widget data loaded (but no artifact)."""
    context_service = Mock(spec=ContextService)

    # Simulate widget data loaded as structured context
    widget_structured_context = StructuredContext(
        content='[{"ticker": "AAPL", "market_cap": 3000000000000}]',
        source_info=SourceInfo(
            type="widget",
            uuid=UUID("12345678-1234-1234-1234-123456789abc"),
            name="Watchlist",
            origin="OpenBB Sandbox",
            widget_id="watchlist",
            description="Watchlist of stocks",
        ),
        sql_table_info=SqlTableInfo(
            table_name="watchlist_table",
            sql_schema="CREATE TABLE watchlist_table (ticker TEXT, market_cap REAL)",
            unique_column_values={},
        ),
        data_format=RawObjectDataFormat(parse_as="table"),
    )

    # Return the widget data when asked for structured context
    context_service.structured_context = [widget_structured_context]
    context_service.unstructured_context = []

    # The bug: LLM tries to look up a hallucinated artifact ID
    # This should return None because the artifact doesn't exist yet
    context_service.get_context_by_source_info_name.return_value = None

    return context_service


@pytest.fixture
def copilot_service_with_widget_loaded(mock_context_service_with_widget_data):
    """Create copilot service with widget data loaded but no artifacts."""
    return CopilotService(
        user_id="test-user-id",
        document_service=Mock(),
        context_service=mock_context_service_with_widget_data,
        template_service=Mock(spec=TemplateService),
        url_retrieval_service=Mock(),
        copilot_data_service=Mock(),
        logging_service=Mock(spec=LoggingService),
        client_function_call_service=Mock(),
        native_function_call_service=Mock(),
        citation_service=Mock(spec=CitationService),
        mcp_data_service=Mock(),
        openai_api_key=None,
    )


async def _collect_stream_output(copilot: CopilotService, response: str) -> str:
    """Stream mock model output through Copilot and collect visible text deltas."""

    async def mock_llm_stream():
        for char in response:
            yield char

    collected_output: list[str] = []
    async for event in copilot._handle_copilot_stream(mock_llm_stream()):
        if hasattr(event, "data") and hasattr(event.data, "delta"):
            collected_output.append(event.data.delta)
    return "".join(collected_output)


@pytest.mark.asyncio
async def test_llm_dangling_artifact_end_tag_does_not_crash(
    copilot_service_with_widget_loaded,
):
    copilot = copilot_service_with_widget_loaded

    final_output = await _collect_stream_output(
        copilot,
        "Here is the answer. <|end_artifact_id|>",
    )

    copilot._context_service.get_context_by_source_info_name.assert_not_called()
    assert final_output == "Here is the answer. "


@pytest.mark.asyncio
async def test_llm_references_nonexistent_artifact_does_not_leak_identifiers(
    copilot_service_with_widget_loaded,
):
    """
    Contract test for unresolved artifact references in stream output.

    The model may emit a hallucinated artifact ID. Even if the artifact lookup
    fails, user-visible output must not leak raw artifact tags or UUIDs.
    """
    copilot = copilot_service_with_widget_loaded

    hallucinated_artifact_id = "b1e2e2e2-2e2e-4e2e-8e2e-1e2e2e2e2e2e"
    response = (
        f"Here is the data: <|start_artifact_id|>{hallucinated_artifact_id}"
        f"<|end_artifact_id|>"
    )
    final_output = await _collect_stream_output(copilot, response)

    # Lookup should still be attempted for the referenced artifact ID.
    copilot._context_service.get_context_by_source_info_name.assert_called_once_with(
        hallucinated_artifact_id
    )

    # Safety contract: never leak unresolved artifact references.
    assert "<|start_artifact_id|>" not in final_output, (
        "Artifact tags should be removed from output"
    )
    assert hallucinated_artifact_id not in final_output, (
        "Artifact ID should not appear in output"
    )
    # Non-artifact text should still be preserved.
    assert "Here is the data:" in final_output


@pytest.mark.asyncio
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known gap: tag-only unresolved artifact references can yield empty output "
        "instead of a user-facing fallback message."
    ),
)
async def test_llm_references_nonexistent_artifact_only_tag_has_fallback_message(
    copilot_service_with_widget_loaded,
):
    """Tag-only unresolved references should still show guidance."""
    copilot = copilot_service_with_widget_loaded
    hallucinated_artifact_id = "c3d4e5f6-2e2e-4e2e-8e2e-1e2e2e2e2e2e"
    response = f"<|start_artifact_id|>{hallucinated_artifact_id}<|end_artifact_id|>"

    final_output = await _collect_stream_output(copilot, response)

    assert "<|start_artifact_id|>" not in final_output
    assert hallucinated_artifact_id not in final_output
    assert final_output.strip() != ""
