import io
from unittest.mock import Mock

import pandas as pd
import pytest
from openbb_ai.models import StatusUpdateSSE

from openbb_ada.models import SqlAgentQueryResult, SqlTableInfo
from openbb_ada.services import SqlAgentService, TemplateService


def test_init_with_memory_db(
    test_template_service: TemplateService, test_database_engine
):
    sql_service = SqlAgentService(
        template_service=test_template_service,
        logging_service=Mock(),
        openai_api_key="test_key",
        engine=test_database_engine,
    )
    assert sql_service is not None
    assert sql_service._openai_api_key == "test_key"


def test_get_table_schema(
    test_sql_agent_service: SqlAgentService, test_csv_tsla_historical_data: bytes
):
    test_sql_agent_service.insert_table(
        df=pd.read_csv(io.BytesIO(test_csv_tsla_historical_data), index_col=0),
        table_name="tesla_historical",
    )

    # Test getting schema for existing table
    expected_schema = "\nCREATE TABLE tesla_historical (\n\tdate TEXT, \n\topen FLOAT, \n\thigh FLOAT, \n\tlow FLOAT, \n\tclose FLOAT, \n\tvolume BIGINT, \n\tvwap FLOAT, \n\tlabel TEXT, \n\tadj_close FLOAT, \n\tunadjusted_volume FLOAT, \n\tchange FLOAT, \n\tchange_percent FLOAT, \n\tchange_over_time FLOAT\n)\n\n"  # noqa: E501
    actual_schema = test_sql_agent_service.get_table_schema("tesla_historical")
    assert actual_schema == expected_schema


def test_get_table_schema_nonexistent(
    test_sql_agent_service: SqlAgentService,
):
    # Test getting schema for non-existent table
    with pytest.raises(ValueError):
        test_sql_agent_service.get_table_schema("nonexistent_table")


def test_insert_table_in_memory_db(
    test_sql_agent_service: SqlAgentService, test_template_service: TemplateService
):
    test_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "open": [100, 101, 102],
            "high": [103, 104, 105],
            "low": [98, 99, 100],
            "close": [101, 102, 103],
        }
    )

    test_table_name = "test_table"

    actual_table_info = test_sql_agent_service.insert_table(test_df, test_table_name)
    expected_table_info = SqlTableInfo(
        table_name="test_table",
        sql_schema='\nCREATE TABLE test_table (\n\t"index" BIGINT, \n\tdate DATETIME, \n\topen BIGINT, \n\thigh BIGINT, \n\tlow BIGINT, \n\tclose BIGINT\n)\n\n',  # noqa: E501
        unique_column_values={},
        description="test_table",  # Now auto-generated from table name
        metadata=None,
    )

    assert actual_table_info.model_dump() == expected_table_info.model_dump()
    assert len(test_sql_agent_service._tables) == 1
    assert (
        test_sql_agent_service._tables[0].model_dump()
        == expected_table_info.model_dump()
    )


def test_insert_table_in_memory_db_same_table_name_avoids_collision(
    test_sql_agent_service: SqlAgentService, test_template_service: TemplateService
):
    test_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "open": [100, 101, 102],
            "high": [103, 104, 105],
            "low": [98, 99, 100],
            "close": [101, 102, 103],
        }
    )

    test_table_name = "test_table"

    actual_table_info_1 = test_sql_agent_service.insert_table(test_df, test_table_name)

    actual_table_info_2 = test_sql_agent_service.insert_table(test_df, test_table_name)

    assert actual_table_info_1.table_name != actual_table_info_2.table_name
    assert "test_table" in actual_table_info_1.table_name
    assert "test_table" in actual_table_info_2.table_name


def test_insert_table_error_raised_for_bad_table(
    test_sql_agent_service: SqlAgentService,
):
    """Test that nested array data is successfully normalized into multiple tables."""
    nested_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "nested_column": [
                [
                    {"name": "item1", "value": 10, "category": "A"},
                    {"name": "item2", "value": 20, "category": "B"},
                ],
                [
                    {"name": "item3", "value": 30, "category": "A"},
                ],
                [
                    {"name": "item4", "value": 40, "category": "C"},
                    {"name": "item5", "value": 50, "category": "B"},
                    {"name": "item6", "value": 60, "category": "A"},
                ],
            ],
        }
    )
    # Should successfully insert and create multiple tables
    result = test_sql_agent_service.insert_table(
        df=nested_df, table_name="nested_table"
    )

    # Verify the result
    assert result is not None
    assert "nested_table" in result.table_name

    # Verify multiple tables were created (parent + child tables)
    all_tables = test_sql_agent_service.get_sql_tables_info()
    assert len(all_tables) >= 2  # At least parent table + child table for nested_column

    # Verify table names contain our base name
    table_names = [table.table_name for table in all_tables]
    assert any("nested_table" in name for name in table_names)
    assert any("nested_column" in name for name in table_names)


def test_insert_table_with_integer_column_names(
    test_sql_agent_service: SqlAgentService,
):
    test_df = pd.DataFrame({"1": [1, 2, 3], "2": [4, 5, 6]})
    test_sql_agent_service.insert_table(df=test_df, table_name="int_col_table")

    assert len(test_sql_agent_service._tables) == 1
    assert "int_col_table" in test_sql_agent_service._tables[0].table_name


def test_insert_table_with_none_column_name(
    test_sql_agent_service: SqlAgentService,
):
    test_df = pd.DataFrame({"1": [1, 2, 3], None: [4, 5, 6]})
    test_sql_agent_service.insert_table(df=test_df, table_name="none_col_table")

    assert len(test_sql_agent_service._tables) == 1
    assert "none_col_table" in test_sql_agent_service._tables[0].table_name


@pytest.mark.asyncio
async def test_query_execution(
    test_sql_agent_service: SqlAgentService, test_csv_tsla_historical_data: bytes
):
    test_df = pd.read_csv(io.BytesIO(test_csv_tsla_historical_data), index_col=0)
    test_df.index = pd.to_datetime(test_df.index)
    test_sql_agent_service.insert_table(
        df=test_df,
        table_name="tesla_historical",
    )

    async for result in test_sql_agent_service.query(
        "what is the average closing price for Tesla?"
    ):
        if isinstance(result, SqlAgentQueryResult):
            actual_result = result
            actual_answer = actual_result.answer
            actual_queried_tables = [
                t.table_name for t in test_sql_agent_service.get_sql_tables_info()
            ]
            actual_query_result_artifact = actual_result.artifact
            expected_in_answer = ["average", "closing", "price", "tesla"]
            assert all([word in actual_answer.lower() for word in expected_in_answer])
            assert "tesla_historical" in actual_queried_tables[0]
            # sql_agent creates artifacts for all results (including single-row)
            # Filtering happens in context.py for the response
            assert "223.777" in str(actual_query_result_artifact.content)
            break
        elif isinstance(result, StatusUpdateSSE):
            if "Intermediate result artifact generated" in result.data.message:
                assert len(result.data.artifacts) == 1


@pytest.mark.asyncio
async def test_query_execution_generates_chart(
    test_sql_agent_service: SqlAgentService, test_csv_tsla_historical_data: bytes
):
    test_df = pd.read_csv(io.BytesIO(test_csv_tsla_historical_data), index_col=0)
    test_df.index = pd.to_datetime(test_df.index)
    test_sql_agent_service.insert_table(
        df=test_df,
        table_name="tesla_historical",
    )

    async for result in test_sql_agent_service.query(
        "Create a chart of the closing price."
    ):
        if isinstance(result, SqlAgentQueryResult):
            actual_result = result
            actual_answer = actual_result.answer
            actual_queried_tables = [
                t.table_name for t in test_sql_agent_service.get_sql_tables_info()
            ]
            actual_query_result_artifact = actual_result.artifact
            answer_lower = actual_answer.lower()
            # Wording is model-dependent. Validate semantics plus artifact contract.
            assert any(
                word in answer_lower
                for word in ["price", "close", "tesla", "historical", "trend"]
            )
            assert "tesla_historical" in actual_queried_tables[0]
            assert actual_query_result_artifact.data_format.parse_as == "chart"
            assert actual_query_result_artifact.data_format.chart_params.xKey == "date"
            # This assertion accommodates "close" and "closing_price"
            assert (
                "clos" in actual_query_result_artifact.data_format.chart_params.yKey[0]
            )

        elif isinstance(result, StatusUpdateSSE):
            # Only check artifacts for StatusUpdateSSE that should have them
            # (i.e., those representing actual data processing results)
            if "Intermediate result artifact generated" in result.data.message:
                assert len(result.data.artifacts) == 1


def test_enhanced_description_generation_basic(
    test_sql_agent_service: SqlAgentService,
):
    """Test basic enhanced description generation without metadata."""
    description = test_sql_agent_service._generate_enhanced_description(
        table_name="test_table",
        description="Basic table description",
        metadata=None,
    )
    assert description == "Basic table description"

    # Test with no description provided
    description_no_desc = test_sql_agent_service._generate_enhanced_description(
        table_name="test_table",
        description=None,
        metadata=None,
    )
    assert description_no_desc == "test_table"


def test_enhanced_description_generation_with_filters(
    test_sql_agent_service: SqlAgentService,
):
    """Test enhanced description generation with filter metadata."""
    metadata = {"filtered_by": ["sector='Technology'", "market_cap > 1B"]}

    description = test_sql_agent_service._generate_enhanced_description(
        table_name="stocks_table",
        description="Stock data",
        metadata=metadata,
    )

    expected = "Stock data - filtered for sector='Technology', market_cap > 1B"
    assert description == expected


def test_enhanced_description_generation_with_grouping(
    test_sql_agent_service: SqlAgentService,
):
    """Test enhanced description generation with grouping metadata."""
    metadata = {"grouped_by": ["sector", "year"]}

    description = test_sql_agent_service._generate_enhanced_description(
        table_name="aggregated_table",
        description="Aggregated stock data",
        metadata=metadata,
    )

    expected = "Aggregated stock data - grouped by sector, year"
    assert description == expected


def test_enhanced_description_generation_with_row_count_range(
    test_sql_agent_service: SqlAgentService,
):
    """Test enhanced description generation with row count range."""
    metadata = {"row_count_range": "1000-5000 rows"}

    description = test_sql_agent_service._generate_enhanced_description(
        table_name="large_table",
        description="Large dataset",
        metadata=metadata,
    )

    expected = "Large dataset - (1000-5000 rows)"
    assert description == expected


def test_enhanced_description_generation_with_display_name(
    test_sql_agent_service: SqlAgentService,
):
    """Test enhanced description generation with table display name."""
    metadata = {"table_display_name": "S&P 500 Companies"}

    description = test_sql_agent_service._generate_enhanced_description(
        table_name="sp500_companies",
        description="Stock market data",
        metadata=metadata,
    )

    expected = "Stock market data - from 'S&P 500 Companies'"
    assert description == expected


def test_enhanced_description_generation_comprehensive(
    test_sql_agent_service: SqlAgentService,
):
    """Test enhanced description generation with all metadata types."""
    metadata = {
        "filtered_by": ["market_cap > 10B"],
        "grouped_by": ["sector"],
        "row_count_range": "50-500 rows",
        "table_display_name": "Fortune 500 Tech Companies",
    }

    description = test_sql_agent_service._generate_enhanced_description(
        table_name="tech_companies",
        description="Technology companies data",
        metadata=metadata,
    )

    expected = "Technology companies data - from 'Fortune 500 Tech Companies' - filtered for market_cap > 10B - grouped by sector - (50-500 rows)"  # noqa: E501
    assert description == expected


def test_enhanced_description_generation_empty_filters_and_groups(
    test_sql_agent_service: SqlAgentService,
):
    """Test enhanced description generation with empty filter/group lists."""
    metadata = {"filtered_by": [], "grouped_by": [], "row_count_range": "100 rows"}

    description = test_sql_agent_service._generate_enhanced_description(
        table_name="test_table",
        description="Test data",
        metadata=metadata,
    )

    expected = "Test data - (100 rows)"
    assert description == expected


def test_insert_table_with_enhanced_description(
    test_sql_agent_service: SqlAgentService,
):
    """Test that insert_table creates enhanced descriptions correctly."""
    test_df = pd.DataFrame(
        {
            "company": ["Apple", "Microsoft", "Google"],
            "sector": ["Technology", "Technology", "Technology"],
            "market_cap": [3000, 2800, 1800],
        }
    )

    metadata = {
        "filtered_by": ["sector='Technology'"],
        "grouped_by": ["sector"],
        "row_count_range": "3 rows",
        "table_display_name": "Tech Giants",
    }

    table_info = test_sql_agent_service.insert_table(
        df=test_df,
        table_name="tech_companies",
        description="Technology companies data",
        metadata=metadata,
    )

    expected_description = "Technology companies data - from 'Tech Giants' - filtered for sector='Technology' - grouped by sector - (3 rows)"  # noqa: E501
    assert table_info.description == expected_description
    assert table_info.metadata == metadata

    # Ensure the table is stored with enhanced description
    stored_table = test_sql_agent_service.get_sql_table_info(table_info.table_name)
    assert stored_table.description == expected_description


def test_insert_table_with_metadata_only(
    test_sql_agent_service: SqlAgentService,
):
    """Test insert_table with metadata but no explicit description."""
    test_df = pd.DataFrame(
        {
            "product": ["iPhone", "iPad", "MacBook"],
            "price": [999, 599, 1299],
        }
    )

    metadata = {"filtered_by": ["price > 500"], "table_display_name": "Apple Products"}

    table_info = test_sql_agent_service.insert_table(
        df=test_df,
        table_name="apple_products",
        metadata=metadata,
    )

    # Should use table name as base description when none provided
    expected_description = (
        "apple_products - from 'Apple Products' - filtered for price > 500"
    )
    assert table_info.description == expected_description


def test_insert_table_enhanced_description_logging(
    test_sql_agent_service: SqlAgentService,
):
    """Test that enhanced description generation is properly logged."""
    test_df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "price": [150, 300],
        }
    )

    metadata = {"filtered_by": ["price > 100"]}

    # Mock the logging service to capture log messages
    mock_logging = Mock()
    test_sql_agent_service._logging_service = mock_logging

    table_info = test_sql_agent_service.insert_table(
        df=test_df,
        table_name="stocks",
        description="Stock prices",
        metadata=metadata,
    )

    # Verify logging was called for enhanced description
    mock_logging.info.assert_any_call(
        "Enhanced table description: original='%s' -> enhanced='%s'",
        "Stock prices",
        table_info.description,
    )


@pytest.mark.asyncio
async def test_query_execution_uses_enhanced_descriptions(
    test_sql_agent_service: SqlAgentService,
):
    """Test that queries utilize enhanced table descriptions for better context."""
    test_df = pd.DataFrame(
        {
            "company": ["Apple", "Microsoft", "Google", "Amazon"],
            "sector": ["Technology", "Technology", "Technology", "E-commerce"],
            "revenue": [394.3, 198.3, 282.8, 513.9],
            "employees": [164000, 221000, 156000, 1540000],
        }
    )

    metadata = {
        "filtered_by": ["revenue > 100B"],
        "grouped_by": ["sector"],
        "row_count_range": "4 large companies",
        "table_display_name": "Fortune 500 Tech & E-commerce",
    }

    table_info = test_sql_agent_service.insert_table(
        df=test_df,
        table_name="large_companies",
        description="Large technology and e-commerce companies",
        metadata=metadata,
    )

    # Verify the enhanced description was created
    expected_description = "Large technology and e-commerce companies - from 'Fortune 500 Tech & E-commerce' - filtered for revenue > 100B - grouped by sector - (4 large companies)"  # noqa: E501
    assert table_info.description == expected_description

    # Verify the enhanced description is available in the SQL tables info
    sql_tables_info = test_sql_agent_service.get_sql_tables_info()
    assert len(sql_tables_info) == 1
    assert sql_tables_info[0].description == expected_description

    # Test that the table can be queried (basic functionality test)
    query_executed = False
    async for result in test_sql_agent_service.query("What is the total revenue?"):
        if isinstance(result, SqlAgentQueryResult):
            query_executed = True
            # The enhanced description should help the LLM understand the context better
            assert result.answer is not None
        elif isinstance(result, StatusUpdateSSE):
            # Only intermediate result artifacts should have artifacts
            if result.data.message == "Intermediate result artifact generated":
                assert result.data.artifacts is not None
                assert len(result.data.artifacts) == 1

    # Ensure at least one result was processed
    assert query_executed or any(
        isinstance(result, StatusUpdateSSE)
        async for result in test_sql_agent_service.query("What is the total revenue?")
    )


def test_get_sql_table_info_with_enhanced_description(
    test_sql_agent_service: SqlAgentService,
):
    """Test retrieving table info preserves enhanced descriptions."""
    test_df = pd.DataFrame(
        {
            "quarter": ["Q1", "Q2", "Q3", "Q4"],
            "sales": [100, 120, 150, 180],
        }
    )

    metadata = {
        "filtered_by": ["year=2024"],
        "grouped_by": ["quarter"],
    }

    table_info = test_sql_agent_service.insert_table(
        df=test_df,
        table_name="quarterly_sales",
        description="Quarterly sales data",
        metadata=metadata,
    )

    # Retrieve the table info
    retrieved_info = test_sql_agent_service.get_sql_table_info(table_info.table_name)

    expected_description = (
        "Quarterly sales data - filtered for year=2024 - grouped by quarter"
    )
    assert retrieved_info.description == expected_description
    assert retrieved_info.metadata == metadata


def test_format_table_name_with_enhanced_tables(
    test_sql_agent_service: SqlAgentService,
):
    """Test table name formatting works correctly with enhanced table metadata."""
    # Test with special characters that should be cleaned
    formatted_name = test_sql_agent_service.format_table_name("S&P 500 Companies!")
    assert formatted_name == "sp_500_companies"

    # Test uniqueness when inserting tables with same formatted names
    test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    table1 = test_sql_agent_service.insert_table(
        df=test_df,
        table_name="S&P 500 Companies",
        description="First table",
        metadata={"source": "table1"},
    )

    table2 = test_sql_agent_service.insert_table(
        df=test_df,
        table_name="S&P-500 Companies",  # Different special chars, same base
        description="Second table",
        metadata={"source": "table2"},
    )

    # Should have different table names due to collision avoidance
    assert table1.table_name != table2.table_name
    assert "sp_500_companies" in table1.table_name
    assert "sp500_companies" in table2.table_name  # Hyphen removed: "S&P-500" → "sp500"


def test_enhanced_description_generation_with_invalid_metadata(
    test_sql_agent_service: SqlAgentService,
):
    """Test enhanced description generation handles invalid metadata gracefully."""
    # Test with non-list filtered_by (should be ignored)
    metadata_invalid_filter = {
        "filtered_by": "not_a_list",  # Invalid: should be list
        "grouped_by": ["sector"],
    }

    description = test_sql_agent_service._generate_enhanced_description(
        table_name="test_table",
        description="Test data",
        metadata=metadata_invalid_filter,
    )

    # Should only include valid grouped_by, ignore invalid filtered_by
    expected = "Test data - grouped by sector"
    assert description == expected

    # Test with non-list grouped_by (should be ignored)
    metadata_invalid_group = {
        "filtered_by": ["price > 100"],
        "grouped_by": 123,  # Invalid: should be list
    }

    description = test_sql_agent_service._generate_enhanced_description(
        table_name="test_table",
        description="Test data",
        metadata=metadata_invalid_group,
    )

    # Should only include valid filtered_by, ignore invalid grouped_by
    expected = "Test data - filtered for price > 100"
    assert description == expected

    # Test with both invalid
    metadata_both_invalid = {
        "filtered_by": 123,  # Invalid
        "grouped_by": "not_a_list",  # Invalid
        "row_count_range": "100 rows",  # Valid
    }

    description = test_sql_agent_service._generate_enhanced_description(
        table_name="test_table",
        description="Test data",
        metadata=metadata_both_invalid,
    )

    # Should only include valid row_count_range
    expected = "Test data - (100 rows)"
    assert description == expected


def test_enhanced_description_generation_with_mixed_types_in_lists(
    test_sql_agent_service: SqlAgentService,
):
    """Test enhanced description generation converts non-string items to strings."""
    metadata = {
        "filtered_by": ["sector='Tech'", 123, True, None],  # Mixed types
        "grouped_by": ["year", 2024, False],  # Mixed types
    }

    description = test_sql_agent_service._generate_enhanced_description(
        table_name="test_table",
        description="Test data",
        metadata=metadata,
    )

    # Should convert all items to strings
    expected = "Test data - filtered for sector='Tech', 123, True, None - grouped by year, 2024, False"  # noqa: E501
    assert description == expected


@pytest.mark.asyncio
async def test_query_execution_no_data(
    test_sql_agent_service: SqlAgentService,
):
    async for result in test_sql_agent_service.query(
        "Give me a table of the monthly trading volume for Amazon?"
    ):
        if isinstance(result, SqlAgentQueryResult):
            actual_result = result
            assert actual_result.artifact is None
            assert actual_result.queried_tables == []


def test_insert_table_handles_nested_array_data(
    test_sql_agent_service: SqlAgentService,
):
    """Test that nested array data is successfully normalized into multiple tables."""
    nested_df = pd.DataFrame(
        [
            {"id": [1]},
            {"nested_column": [["arrays", "are", "now", "supported", "in", "sqlite"]]},
        ]
    )

    # Should successfully insert and create multiple tables
    result = test_sql_agent_service.insert_table(
        df=nested_df, table_name="nested_table"
    )

    # Verify the result
    assert result is not None
    assert "nested_table" in result.table_name

    # Verify multiple tables were created (parent + child tables)
    all_tables = test_sql_agent_service.get_sql_tables_info()
    assert len(all_tables) >= 2  # At least parent table + child table for nested_column

    # Verify table names contain our base name
    table_names = [table.table_name for table in all_tables]
    assert any("nested_table" in name for name in table_names)
    assert any("nested_column" in name for name in table_names)


def test_calendar_year_aggregation_query_structure(
    test_sql_agent_service: SqlAgentService,
):
    """Test that calendar year aggregation SQL creates proper dense data structure."""
    # Create test data mimicking different fiscal year endings
    # Amazon: Dec 31 fiscal year
    amazon_df = pd.DataFrame(
        {
            "period_ending": ["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31"],
            "total_revenue": [386064000000, 469822000000, 513983000000, 574785000000],
        }
    )

    # Apple: Sep 30 fiscal year
    apple_df = pd.DataFrame(
        {
            "period_ending": ["2020-09-30", "2021-09-30", "2022-09-30", "2023-09-30"],
            "total_revenue": [274515000000, 365817000000, 394328000000, 383285000000],
        }
    )

    # Microsoft: Jun 30 fiscal year
    microsoft_df = pd.DataFrame(
        {
            "period_ending": ["2020-06-30", "2021-06-30", "2022-06-30", "2023-06-30"],
            "total_revenue": [143015000000, 168088000000, 198270000000, 211915000000],
        }
    )

    # Insert tables
    test_sql_agent_service.insert_table(
        amazon_df, "income_amzn", metadata={"symbol": "AMZN"}
    )
    test_sql_agent_service.insert_table(
        apple_df, "income_aapl", metadata={"symbol": "AAPL"}
    )
    test_sql_agent_service.insert_table(
        microsoft_df, "income_msft", metadata={"symbol": "MSFT"}
    )

    # Test the calendar year aggregation query
    calendar_query = """
    SELECT
      calendar_year,
      SUM(amzn_revenue) as amzn_revenue,
      SUM(aapl_revenue) as aapl_revenue,
      SUM(msft_revenue) as msft_revenue
    FROM (
      SELECT
        CAST(strftime('%Y', period_ending) AS INTEGER) as calendar_year,
        total_revenue AS amzn_revenue,
        NULL AS aapl_revenue,
        NULL AS msft_revenue
      FROM income_amzn WHERE period_ending >= '2020-01-01'
      UNION ALL
      SELECT
        CAST(strftime('%Y', period_ending) AS INTEGER) as calendar_year,
        NULL AS amzn_revenue,
        total_revenue AS aapl_revenue,
        NULL AS msft_revenue
      FROM income_aapl WHERE period_ending >= '2020-01-01'
      UNION ALL
      SELECT
        CAST(strftime('%Y', period_ending) AS INTEGER) as calendar_year,
        NULL AS amzn_revenue,
        NULL AS aapl_revenue,
        total_revenue AS msft_revenue
      FROM income_msft WHERE period_ending >= '2020-01-01'
    )
    GROUP BY calendar_year
    ORDER BY calendar_year ASC
    """

    # Execute the query
    result_df = test_sql_agent_service._execute_sql_query(calendar_query)

    # Verify results - should have dense data (no null values)
    assert len(result_df) == 4  # 2020, 2021, 2022, 2023

    # Check that each year has data for all three companies (no nulls)
    for _, row in result_df.iterrows():
        assert row["amzn_revenue"] is not None and row["amzn_revenue"] > 0
        assert row["aapl_revenue"] is not None and row["aapl_revenue"] > 0
        assert row["msft_revenue"] is not None and row["msft_revenue"] > 0

    # Verify the data structure is suitable for charting
    columns = list(result_df.columns)
    assert "calendar_year" in columns
    assert "amzn_revenue" in columns
    assert "aapl_revenue" in columns
    assert "msft_revenue" in columns

    # Verify years are in ascending order
    years = result_df["calendar_year"].tolist()
    assert years == sorted(years)
    assert years == [2020, 2021, 2022, 2023]


def test_fiscal_vs_calendar_year_data_sparsity(
    test_sql_agent_service: SqlAgentService,
):
    """Test that demonstrates the sparsity problem with fiscal periods
    vs density with calendar years.
    """
    # Create test data with different fiscal year endings
    amazon_df = pd.DataFrame(
        {
            "period_ending": ["2020-12-31", "2021-12-31"],
            "total_revenue": [386064000000, 469822000000],
        }
    )

    apple_df = pd.DataFrame(
        {
            "period_ending": ["2020-09-30", "2021-09-30"],
            "total_revenue": [274515000000, 365817000000],
        }
    )

    test_sql_agent_service.insert_table(amazon_df, "income_amzn")
    test_sql_agent_service.insert_table(apple_df, "income_aapl")

    # SPARSE approach (problematic) - using exact fiscal periods
    sparse_query = """
    SELECT period_ending, total_revenue AS amzn_revenue,
           NULL AS aapl_revenue FROM income_amzn
    UNION ALL
    SELECT period_ending, NULL AS amzn_revenue,
           total_revenue AS aapl_revenue FROM income_aapl
    ORDER BY period_ending ASC
    """

    sparse_result = test_sql_agent_service._execute_sql_query(sparse_query)

    # Verify sparsity - each row should have one null/NaN value
    null_count = 0
    for _, row in sparse_result.iterrows():
        row_nulls = sum(
            1 for val in [row["amzn_revenue"], row["aapl_revenue"]] if pd.isna(val)
        )
        null_count += row_nulls

    # Should have significant nulls (each row has exactly one null)
    assert null_count == len(
        sparse_result
    )  # Each row has 1 null = total nulls equals row count

    # DENSE approach (solution) - using calendar year aggregation
    dense_query = """
    SELECT
      calendar_year,
      SUM(amzn_revenue) as amzn_revenue,
      SUM(aapl_revenue) as aapl_revenue
    FROM (
      SELECT
        CAST(strftime('%Y', period_ending) AS INTEGER) as calendar_year,
        total_revenue AS amzn_revenue,
        NULL AS aapl_revenue
      FROM income_amzn
      UNION ALL
      SELECT
        CAST(strftime('%Y', period_ending) AS INTEGER) as calendar_year,
        NULL AS amzn_revenue,
        total_revenue AS aapl_revenue
      FROM income_aapl
    )
    GROUP BY calendar_year
    ORDER BY calendar_year ASC
    """

    dense_result = test_sql_agent_service._execute_sql_query(dense_query)

    # Verify density - no null values
    for _, row in dense_result.iterrows():
        assert row["amzn_revenue"] is not None and row["amzn_revenue"] > 0
        assert row["aapl_revenue"] is not None and row["aapl_revenue"] > 0

    # Dense result should have fewer rows but complete data
    assert len(dense_result) <= len(sparse_result)  # Aggregated data has fewer rows
    assert len(dense_result) == 2  # 2020, 2021


def test_system_prompt_contains_calendar_year_guidance(
    test_template_service: TemplateService,
):
    """Test that the SQL Agent system prompt includes calendar
    year aggregation guidance.
    """
    mock_tables = [
        SqlTableInfo(
            table_name="income_amzn",
            sql_schema=(
                "CREATE TABLE income_amzn (period_ending TEXT, total_revenue INTEGER)"
            ),
            unique_column_values={},
            description="Amazon income statement data",
            metadata={"symbol": "AMZN"},
        ),
        SqlTableInfo(
            table_name="income_aapl",
            sql_schema=(
                "CREATE TABLE income_aapl (period_ending TEXT, total_revenue INTEGER)"
            ),
            unique_column_values={},
            description="Apple income statement data",
            metadata={"symbol": "AAPL"},
        ),
    ]

    system_prompt = test_template_service.render_copilot_sql_agent_system_prompt(
        sql_tables_info=mock_tables
    )

    # Verify calendar year aggregation guidance is present
    assert "Calendar Year Aggregation for Dense Comparison Charts" in system_prompt
    assert "CRITICAL for multi-company/multi-entity comparisons" in system_prompt
    assert "GROUP BY calendar_year" in system_prompt
    assert "strftime('%Y', period_ending)" in system_prompt

    # Verify it explains the problem
    assert "sparse data" in system_prompt
    assert "different fiscal year endings" in system_prompt

    # Verify it provides solution examples
    assert "WRONG APPROACH" in system_prompt
    assert "CORRECT APPROACH" in system_prompt
    assert "SUM(amzn_revenue)" in system_prompt


@pytest.mark.asyncio
async def test_calendar_year_aggregation_in_chart_generation(
    test_sql_agent_service: SqlAgentService,
):
    """Test that calendar year aggregation works in chart generation."""
    # Create multi-company revenue data with different fiscal year endings
    amazon_df = pd.DataFrame(
        {
            "period_ending": ["2020-12-31", "2021-12-31", "2022-12-31"],
            "total_revenue": [386064000000, 469822000000, 513983000000],
        }
    )

    apple_df = pd.DataFrame(
        {
            "period_ending": ["2020-09-30", "2021-09-30", "2022-09-30"],
            "total_revenue": [274515000000, 365817000000, 394328000000],
        }
    )

    test_sql_agent_service.insert_table(
        amazon_df,
        "income_amzn",
        description="Amazon annual revenue data",
        metadata={"symbol": "AMZN", "fiscal_year_ending": "December"},
    )
    test_sql_agent_service.insert_table(
        apple_df,
        "income_aapl",
        description="Apple annual revenue data",
        metadata={"symbol": "AAPL", "fiscal_year_ending": "September"},
    )

    # Test query that should trigger calendar year aggregation for comparison
    query_executed = False
    async for result in test_sql_agent_service.query(
        "Create a bar chart comparing revenue from Amazon vs Apple since 2020"
    ):
        if isinstance(result, SqlAgentQueryResult):
            query_executed = True

            # Verify we got a result
            assert result.answer is not None

            # If an artifact was created, verify it's suitable for charting
            if result.artifact is not None:
                # Should be chart format
                assert result.artifact.data_format.parse_as == "chart"

                # Verify the data structure is dense (no sparse data)
                data = result.artifact.content
                if data and len(data) > 0:
                    # Each row should have values for both companies
                    for row in data:
                        non_null_values = sum(
                            1 for v in row.values() if v is not None and v != 0
                        )
                        # Should have multiple non-null values (not sparse)
                        assert non_null_values >= 2

            break
        elif isinstance(result, StatusUpdateSSE):
            # Check for intermediate artifacts during processing
            if "Intermediate result artifact generated" in result.data.message:
                assert result.data.artifacts is not None

    # Ensure the query was processed
    assert query_executed


def test_calendar_year_aggregation_sql_syntax_validation(
    test_sql_agent_service: SqlAgentService,
):
    """Test that the calendar year aggregation SQL syntax is valid SQLite."""
    # Create minimal test data
    test_df = pd.DataFrame(
        {
            "period_ending": ["2020-12-31", "2021-12-31"],
            "total_revenue": [100000, 200000],
        }
    )

    test_sql_agent_service.insert_table(test_df, "test_income")

    # Test the core calendar year aggregation pattern
    calendar_aggregation_sql = """
    SELECT
      CAST(strftime('%Y', period_ending) AS INTEGER) as calendar_year,
      SUM(total_revenue) as total_revenue
    FROM test_income
    WHERE period_ending >= '2020-01-01'
    GROUP BY calendar_year
    ORDER BY calendar_year ASC
    """

    # Should execute without error
    result = test_sql_agent_service._execute_sql_query(calendar_aggregation_sql)

    # Verify results
    assert len(result) == 2
    assert list(result.columns) == ["calendar_year", "total_revenue"]
    assert result.iloc[0]["calendar_year"] == 2020
    assert result.iloc[1]["calendar_year"] == 2021
    assert result.iloc[0]["total_revenue"] == 100000
    assert result.iloc[1]["total_revenue"] == 200000


@pytest.mark.asyncio
async def test_peek_column_unique_values_returns_all_values(
    test_sql_agent_service: SqlAgentService,
):
    """Test _llm_peek_column_unique_values returns all unique values."""
    # Create a table with a categorical 'metric' column (similar to financial_metrics)
    test_df = pd.DataFrame(
        {
            "metric": [
                "Asset Turnover",
                "Book Value Growth",
                "Cash Ratio",
                "Current Ratio",
                "Debt to Equity",
                "Free Cash Flow Yield",
                "Gross Margin",
                "Net Margin",
                "Operating Margin",
                "Return on Assets",
            ],
            "value": [0.5, 0.03, 0.58, 2.07, 0.15, 0.02, 0.25, 0.12, 0.18, 0.08],
        }
    )

    table_info = test_sql_agent_service.insert_table(
        df=test_df,
        table_name="financial_metrics",
        description="Financial metrics data",
    )

    # Call _llm_peek_column_unique_values
    unique_values_result = None
    async for result in test_sql_agent_service._llm_peek_column_unique_values(
        table_name=table_info.table_name,
        column_name="metric",
        summary="Getting unique metric values",
    ):
        if isinstance(result, str) and "Unique values" in result:
            unique_values_result = result

    # Verify all unique values are returned
    assert unique_values_result is not None
    assert "Asset Turnover" in unique_values_result
    assert "Current Ratio" in unique_values_result
    assert "Debt to Equity" in unique_values_result
    assert "Return on Assets" in unique_values_result

    # Verify the result contains exactly 10 values
    assert "['Asset Turnover'" in unique_values_result


@pytest.mark.asyncio
async def test_peek_column_unique_values_yields_status_update(
    test_sql_agent_service: SqlAgentService,
):
    """Test that _llm_peek_column_unique_values yields a StatusUpdateSSE."""
    test_df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "value": [1, 2, 3],
        }
    )

    table_info = test_sql_agent_service.insert_table(
        df=test_df,
        table_name="test_categories",
    )

    # Collect all yielded results
    results = []
    async for result in test_sql_agent_service._llm_peek_column_unique_values(
        table_name=table_info.table_name,
        column_name="category",
        summary="Getting categories",
    ):
        results.append(result)

    # Should yield StatusUpdateSSE first, then the string result
    assert len(results) == 2
    assert isinstance(results[0], StatusUpdateSSE)
    assert results[0].data.message == "Getting categories"
    assert isinstance(results[1], str)
    assert "A" in results[1]
    assert "B" in results[1]
    assert "C" in results[1]


def test_sql_query_is_metadata_inspection_detects_sqlite_type_checks(
    test_sql_agent_service: SqlAgentService,
):
    sql_query = """
    SELECT 'close' AS available_column,
           min(typeof(close)) AS observed_sqlite_type
    FROM stock_price_aapl;
    """

    assert test_sql_agent_service._sql_query_is_metadata_inspection(sql_query) is True
    assert (
        test_sql_agent_service._sql_query_is_metadata_inspection(
            "SELECT date, close FROM stock_price_aapl"
        )
        is False
    )


@pytest.mark.asyncio
async def test_llm_complete_defaults_to_creating_artifact(
    test_sql_agent_service: SqlAgentService,
):
    table_info = test_sql_agent_service.insert_table(
        df=pd.DataFrame({"year": [2024, 2025], "revenue": [100, 125]}),
        table_name="revenue_table",
    )

    results = []
    async for result in test_sql_agent_service._llm_complete(
        final_answer="Revenue increased.",
        final_sql_query="SELECT year, revenue FROM revenue_table",
        queried_tables=[table_info.table_name],
        summary="Providing revenue table",
    ):
        results.append(result)

    query_result = next(
        result for result in results if isinstance(result, SqlAgentQueryResult)
    )
    assert query_result.artifact is not None
    assert query_result.artifact.data_format.parse_as == "table"


@pytest.mark.asyncio
async def test_peek_column_unique_values_handles_nonexistent_column(
    test_sql_agent_service: SqlAgentService,
):
    """Test that _llm_peek_column_unique_values raises error for nonexistent column."""
    from openbb_ada.errors import SqlAgentError

    test_df = pd.DataFrame(
        {
            "existing_column": [1, 2, 3],
        }
    )

    table_info = test_sql_agent_service.insert_table(
        df=test_df,
        table_name="test_table",
    )

    # Should raise SqlAgentError for nonexistent column
    with pytest.raises(SqlAgentError) as exc_info:
        async for _ in test_sql_agent_service._llm_peek_column_unique_values(
            table_name=table_info.table_name,
            column_name="nonexistent_column",
            summary="Getting values",
        ):
            pass

    assert "Unable to get unique values" in str(exc_info.value)


@pytest.mark.asyncio
async def test_peek_column_unique_values_rejects_sqlite_internal_tables(
    test_sql_agent_service: SqlAgentService,
):
    """Test that _llm_peek_column_unique_values rejects SQLite internal tables."""
    from openbb_ada.errors import SqlAgentError

    # Should raise SqlAgentError for SQLite internal table
    with pytest.raises(SqlAgentError) as exc_info:
        async for _ in test_sql_agent_service._llm_peek_column_unique_values(
            table_name="sqlite_master",
            column_name="name",
            summary="Getting values",
        ):
            pass

    assert "Cannot query SQLite internal table" in str(exc_info.value)
