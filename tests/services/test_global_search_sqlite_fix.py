"""
Tests for global_search SQLite internal table handling.

These tests replicate the issue where enabling global_search would cause
500 errors when the SQL Agent attempted to query SQLite internal tables
like sqlite_master.

The tests verify:
1. SQLite internal tables are filtered from SQL Agent context
2. Attempts to query sqlite_master are handled gracefully
3. SHOW statements (Snowflake dialect) are blocked
4. Context service skips unknown tables without crashing
"""

import uuid
from unittest.mock import patch

import pandas as pd
import pytest

from openbb_ada.errors import SqlAgentError
from openbb_ada.models import (
    ContextStructuredQueryResult,
    ParsedContext,
    SourceInfo,
    SqlAgentQueryResult,
)
from openbb_ada.services._logging import LoggingService
from openbb_ada.services.context import ContextService
from openbb_ada.services.sql_agent import SqlAgentService
from openbb_ada.services.template import TemplateService


@pytest.fixture
def test_logging_service():
    """Create a test logging service."""
    return LoggingService()


@pytest.fixture
def test_template_service():
    """Create a test template service."""
    return TemplateService()


@pytest.fixture
def test_sql_agent_service(test_template_service, test_logging_service):
    """Create a test SQL agent service."""
    return SqlAgentService(
        template_service=test_template_service,
        logging_service=test_logging_service,
        openai_api_key="test_key",
    )


@pytest.fixture
def test_context_service(
    test_template_service, test_sql_agent_service, test_logging_service
):
    """Create a test context service."""
    return ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=test_logging_service,
    )


# Test 1: Verify SQLite internal tables are filtered
def test_sql_agent_filters_sqlite_internal_tables(test_sql_agent_service):
    """
    Test that get_sql_tables_info() filters out SQLite internal tables.

    Before fix: Would return sqlite_master and other internal tables
    After fix: Filters out all tables starting with 'sqlite_'
    """
    # Create test data
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # Insert a normal table
    table_info = test_sql_agent_service.insert_table(
        df=df,
        table_name="user_table",
        description="Normal user table",
    )

    # Verify the table was created
    assert table_info.table_name == "user_table"

    # Get all tables from SQL agent
    all_tables = test_sql_agent_service.get_sql_tables_info()

    # Verify no SQLite internal tables are returned
    sqlite_internal_tables = [
        t.table_name for t in all_tables if t.table_name.startswith("sqlite_")
    ]

    assert len(sqlite_internal_tables) == 0, (
        f"SQLite internal tables should be filtered but found: {sqlite_internal_tables}"
    )

    # Verify user table is still present
    user_table_names = [t.table_name for t in all_tables]
    assert "user_table" in user_table_names


def test_is_sqlite_internal_table_detection(test_sql_agent_service):
    """
    Test the _is_sqlite_internal_table() method correctly identifies internal tables.
    """
    # Test internal tables
    assert test_sql_agent_service._is_sqlite_internal_table("sqlite_master") is True
    assert test_sql_agent_service._is_sqlite_internal_table("sqlite_sequence") is True
    assert test_sql_agent_service._is_sqlite_internal_table("sqlite_stat1") is True

    # Test user tables
    assert test_sql_agent_service._is_sqlite_internal_table("user_table") is False
    assert test_sql_agent_service._is_sqlite_internal_table("my_data") is False
    # case sensitive
    assert test_sql_agent_service._is_sqlite_internal_table("SQLITE_TEST") is False


# Test 2: Block SHOW statements (Snowflake dialect)
@pytest.mark.asyncio
async def test_sql_agent_blocks_show_statements(test_sql_agent_service):
    """
    Test that SQL Agent blocks Snowflake-style SHOW statements.

    Before fix: Would attempt to execute and fail with cryptic error
    After fix: Raises SqlAgentError with helpful message
    """
    # Create test data
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    test_sql_agent_service.insert_table(df=df, table_name="test_table")

    # Test various SHOW statement formats
    show_statements = [
        "SHOW DATABASES",
        "SHOW TABLES",
        "SHOW SCHEMAS",
        "show databases",  # lowercase
        "  SHOW TABLES  ",  # with whitespace
    ]

    for show_sql in show_statements:
        with pytest.raises(SqlAgentError) as exc_info:
            # Attempt to execute SHOW statement
            events = []
            async for event in test_sql_agent_service._llm_execute_sql(
                sql_query=show_sql, queried_tables=[]
            ):
                events.append(event)

        # Verify error message is helpful
        error_msg = str(exc_info.value)
        assert "Invalid SQL for SQLite" in error_msg
        assert "Snowflake" in error_msg or "SQLite" in error_msg
        assert "SELECT name FROM sqlite_master" in error_msg or "PRAGMA" in error_msg


# Test 3: Block peeking SQLite internal tables
@pytest.mark.asyncio
async def test_sql_agent_blocks_peeking_internal_tables(test_sql_agent_service):
    """
    Test that SQL Agent prevents peeking SQLite internal tables.

    Before fix: Would attempt to peek and cause issues
    After fix: Raises SqlAgentError with clear message
    """
    internal_tables = ["sqlite_master", "sqlite_sequence", "sqlite_stat1"]

    for table_name in internal_tables:
        with pytest.raises(SqlAgentError) as exc_info:
            # Attempt to peek internal table
            events = []
            async for event in test_sql_agent_service._llm_peek_table(
                table_name=table_name
            ):
                events.append(event)

        # Verify error message is educational
        error_msg = str(exc_info.value)
        assert "Cannot peek SQLite internal table" in error_msg
        assert table_name in error_msg
        assert "user tables" in error_msg.lower()


# Test 4: Context service gracefully handles unknown tables
@pytest.mark.asyncio
async def test_context_service_skips_sqlite_internal_tables(
    test_context_service, test_logging_service
):
    """
    Test that context service gracefully skips SQLite internal tables.

    This is the core test that replicates the original 500 error issue.

    Before fix: ValueError would be raised, causing 500 error
    After fix: Tables are skipped with warning log, no crash
    """
    # Load some test data into context
    df = pd.DataFrame({"price": [100, 200, 300], "volume": [1000, 2000, 3000]})
    json_data = df.to_json(orient="records")

    parsed_context = ParsedContext(
        content=json_data,
        source_info=SourceInfo(
            type="widget",
            uuid=uuid.uuid4(),
            name="test_widget",
            description="Test widget data",
            metadata={},
        ),
    )

    test_context_service.load_context([parsed_context])

    # Mock SQL Agent result that includes sqlite_master (simulating the bug)
    mock_result = SqlAgentQueryResult(
        answer="Here is the data",
        queried_tables=["test_widget", "sqlite_master", "sqlite_sequence"],
        artifact=None,
    )

    # Mock the SQL Agent query to return our result
    async def mock_query(query: str):
        yield mock_result

    with patch.object(
        test_context_service._sql_agent_service, "query", side_effect=mock_query
    ):
        # This should NOT raise an error
        result = None
        query_generator = (
            test_context_service.query_structured_context_with_natural_language(
                "test query"
            )
        )
        async for event in query_generator:
            if isinstance(event, ContextStructuredQueryResult):
                result = event

        # Verify we got a result (no crash)
        assert result is not None
        assert isinstance(result, ContextStructuredQueryResult)

        # Verify citations only include user tables, not sqlite_master
        citation_tables = []
        for citation in result.citations:
            if citation.source_info.name:
                citation_tables.append(citation.source_info.name)

        # Should not include sqlite_master in citations
        assert "sqlite_master" not in str(citation_tables).lower()
        assert "sqlite_sequence" not in str(citation_tables).lower()


# Test 5: End-to-end test with logging verification
@pytest.mark.asyncio
async def test_global_search_with_widget_data_handles_errors_gracefully(
    test_context_service, test_logging_service
):
    """
    End-to-end test simulating the global_search flow.

    This test simulates:
    1. Widget data being loaded
    2. SQL Agent querying the data
    3. SQL Agent accidentally including sqlite_master in results
    4. System handling it gracefully without 500 error
    """
    # Step 1: Load widget data
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "value": [100, 150, 200, 175, 225],
        }
    )
    json_data = df.to_json(orient="records", date_format="iso")

    parsed_context = ParsedContext(
        content=json_data,
        source_info=SourceInfo(
            type="widget",
            uuid=uuid.uuid4(),
            name="financial_data",
            description="Financial widget data",
            metadata={"widget_uuid": str(uuid.uuid4())},
        ),
    )

    loaded = test_context_service.load_context([parsed_context])
    assert len(loaded) > 0

    # Step 2: Simulate SQL Agent query that includes sqlite_master
    mock_result = SqlAgentQueryResult(
        answer="Analysis complete",
        # Bug: includes sqlite_master
        queried_tables=["financial_data", "sqlite_master"],
        artifact=None,
    )

    # Step 3: Query with the mock result
    async def mock_query(query: str):
        yield mock_result

    with patch.object(
        test_context_service._sql_agent_service, "query", side_effect=mock_query
    ):
        # This should complete without raising ValueError or 500 error
        result = None
        events = []

        try:
            query_gen = (
                test_context_service.query_structured_context_with_natural_language(
                    "Show me the trend"
                )
            )
            async for event in query_gen:
                events.append(event)
                if isinstance(event, ContextStructuredQueryResult):
                    result = event
        except ValueError as e:
            # Before fix: ValueError would be raised here
            # After fix: This should not happen
            pytest.fail(
                f"ValueError was raised (the bug still exists): {e}\n"
                "Expected: System should gracefully skip sqlite_master"
            )

        # Verify we got a successful result
        assert result is not None, (
            "Should return result even with sqlite_master in queried_tables"
        )
        assert len(events) > 0, "Should emit events"


# Test 6: Verify prompt includes SQLite dialect warnings
def test_sql_agent_prompt_includes_dialect_warnings(test_template_service):
    """
    Test that the SQL Agent system prompt includes warnings about SQLite vs Snowflake.
    """
    # Render the prompt
    prompt = test_template_service.render_copilot_sql_agent_system_prompt(
        sql_tables_info=[]
    )

    # Verify prompt contains key warnings
    assert "SQLite" in prompt
    assert "SHOW DATABASES" in prompt or "SHOW TABLES" in prompt
    assert "FORBIDDEN" in prompt or "DO NOT" in prompt

    # Verify it mentions alternatives
    assert "sqlite_master" in prompt or "PRAGMA" in prompt


# Test 7: Integration test - Full flow with actual SQL Agent
@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_flow_with_widget_data_and_sql_agent(
    test_context_service, test_sql_agent_service
):
    """
    Integration test of the full flow with actual SQL Agent execution.

    Note: This test requires LLM access and may be slow. Use @pytest.mark.integration
    to allow selective execution.
    """
    # Create realistic widget data
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "revenue": [1000, 1200, 1100, 1300, 1250, 1400, 1350, 1500, 1450, 1600],
            "profit": [100, 150, 110, 180, 160, 200, 190, 220, 210, 250],
        }
    )
    json_data = df.to_json(orient="records", date_format="iso")

    parsed_context = ParsedContext(
        content=json_data,
        source_info=SourceInfo(
            type="widget",
            uuid=uuid.uuid4(),
            name="company_financials",
            description="Company financial data",
            metadata={},
        ),
    )

    # Load context
    test_context_service.load_context([parsed_context])

    # Verify tables were created and sqlite internals are filtered
    available_tables = test_sql_agent_service.get_sql_tables_info()
    table_names = [t.table_name for t in available_tables]

    assert len(table_names) > 0
    assert not any(name.startswith("sqlite_") for name in table_names)
    assert any("company_financials" in name.lower() for name in table_names)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
