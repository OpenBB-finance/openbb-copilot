from unittest.mock import patch
from uuid import UUID

import pytest
from magentic.prompt_chain import MaxFunctionCallsError
from openbb_ai.models import (
    OptionsEndpointParam,
    Undefined,
    Widget,
    WidgetCollection,
    WidgetParam,
    WidgetParamOption,
)

from openbb_ada.errors import FunctionCallError
from openbb_ada.models import (
    DataSource,
    DataSourceInputField,
    InputArgGenerationResult,
    QueryWidgetRequest,
    WidgetParamOptions,
)
from openbb_ada.services import CopilotDataService, LoggingService, TemplateService
from tests.conftest import MockUUIDs


@pytest.mark.asyncio
async def test_copilot_data_service_init(mock_uuids: type[MockUUIDs]):
    copilot_data_service = CopilotDataService(
        template_service=TemplateService(),
        logging_service=LoggingService(),
        widget_collection=WidgetCollection(
            primary=[
                Widget(
                    uuid=UUID(mock_uuids.ID1.value),
                    origin="origin_1",
                    widget_id="widget_a",
                    name="widget_a",
                    description="desc widget_a",
                    params=[
                        WidgetParam(
                            name="ticker",
                            type="string",
                            description="ticker symbol",
                            current_value="TICKER_A",
                        ),
                    ],
                    metadata={},
                )
            ],
            secondary=[
                Widget(
                    uuid=UUID(mock_uuids.ID2.value),
                    origin="origin_2",
                    widget_id="widget_b",
                    name="widget_b",
                    description="desc widget_b",
                    params=[
                        WidgetParam(
                            name="ticker",
                            type="string",
                            description="ticker symbol",
                            current_value="TICKER_B",
                            default_value="ABC",
                        ),
                    ],
                    metadata={},
                )
            ],
            extra=[],
        ),
        openai_api_key=None,
    )

    assert copilot_data_service is not None
    assert copilot_data_service._data_sources_map is not None
    assert copilot_data_service._widget_collection is not None
    assert mock_uuids.ID1.value in copilot_data_service._data_sources_map
    assert mock_uuids.ID2.value in copilot_data_service._data_sources_map
    assert copilot_data_service._data_sources_map[mock_uuids.ID1.value] == DataSource(
        origin="origin_1",
        id="widget_a",
        name="widget_a",
        description="desc widget_a",
        input_fields={
            "ticker": DataSourceInputField(
                title="ticker",
                description="ticker symbol",
                type="string",
                enum=None,
                default=Undefined.UNDEFINED,
                current_value="TICKER_A",
                get_options=False,
                options_params=[],
            )
        },
        widget=Widget(
            uuid=UUID(mock_uuids.ID1.value),
            origin="origin_1",
            widget_id="widget_a",
            name="widget_a",
            description="desc widget_a",
            params=[
                WidgetParam(
                    name="ticker",
                    type="string",
                    description="ticker symbol",
                    default_value=Undefined.UNDEFINED,
                    current_value="TICKER_A",
                    options=None,
                    get_options=False,
                    options_params=[],
                )
            ],
            metadata={},
        ),
    )
    assert copilot_data_service._data_sources_map[mock_uuids.ID2.value] == DataSource(
        origin="origin_2",
        id="widget_b",
        name="widget_b",
        description="desc widget_b",
        input_fields={
            "ticker": DataSourceInputField(
                title="ticker",
                description="ticker symbol",
                type="string",
                enum=None,
                default="ABC",
                current_value="TICKER_B",
                get_options=False,
                options_params=[],
            )
        },
        widget=Widget(
            uuid=UUID(mock_uuids.ID2.value),
            origin="origin_2",
            widget_id="widget_b",
            name="widget_b",
            description="desc widget_b",
            params=[
                WidgetParam(
                    name="ticker",
                    type="string",
                    description="ticker symbol",
                    default_value="ABC",
                    current_value="TICKER_B",
                    options=None,
                    get_options=False,
                    options_params=[],
                )
            ],
            metadata={},
        ),
    )


@pytest.mark.asyncio
async def test_content_search_increases_limit_parameter(mock_uuids: type[MockUUIDs]):
    """Test that searching for specific content increases the limit parameter."""
    # Create a news widget with a limit parameter
    news_widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        origin="ODP",
        widget_id="news_world_fmp_obb",
        name="World",
        description="World News. Global news data.",
        params=[
            WidgetParam(
                name="start_date",
                type="date",
                description="Start date of the data, in YYYY-MM-DD format.",
                current_value="2025-10-19",
            ),
            WidgetParam(
                name="end_date",
                type="date",
                description="End date of the data, in YYYY-MM-DD format.",
                current_value="2025-11-02",
            ),
            WidgetParam(
                name="limit",
                type="number",
                description="The number of data entries to return.",
                current_value=5,  # Default small limit
            ),
            WidgetParam(
                name="topic",
                type="text",
                description="The topic of the news to be fetched.",
                current_value="general",
                default_value="general",
            ),
        ],
        metadata={},
    )

    copilot_data_service = CopilotDataService(
        template_service=TemplateService(),
        logging_service=LoggingService(),
        widget_collection=WidgetCollection(
            primary=[news_widget],
            secondary=[],
            extra=[],
        ),
        openai_api_key="test_key",
    )

    # Mock the LLM response to return a large limit for search queries
    with patch.object(copilot_data_service, "_generate_input_args") as mock_generate:
        # Configure mock to return result with increased limit
        mock_generate.return_value = InputArgGenerationResult(
            input_args={
                "start_date": "2025-10-19",
                "end_date": "2025-11-02",
                "limit": 100,  # Should be increased from 5 to 100+ for search
                "topic": "general",
            },
            data_source=copilot_data_service._data_sources_map[mock_uuids.ID1.value],
            used_extra_param_options=False,
        )

        # Test search query that should trigger increased limit
        query_request = QueryWidgetRequest(
            widget_uuid=UUID(mock_uuids.ID1.value),
            widget_query="Find news about Argentina",  # Search query
            use_current_inputs=False,
        )

        result = await copilot_data_service.query_widgets([query_request])

        # Verify the mock was called
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args

        # Check that the query passed contains search intent
        assert "Argentina" in call_args[1]["query"]

        # Verify result has increased limit
        assert result.content[0].payload.input_args["limit"] == 100


@pytest.mark.asyncio
async def test_normal_query_keeps_default_limit(mock_uuids: type[MockUUIDs]):
    """Test that normal queries keep the default limit parameter."""
    # Create a news widget with a limit parameter
    news_widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        origin="ODP",
        widget_id="news_world_fmp_obb",
        name="World",
        description="World News. Global news data.",
        params=[
            WidgetParam(
                name="limit",
                type="number",
                description="The number of data entries to return.",
                current_value=5,  # Default small limit
            ),
        ],
        metadata={},
    )

    copilot_data_service = CopilotDataService(
        template_service=TemplateService(),
        logging_service=LoggingService(),
        widget_collection=WidgetCollection(
            primary=[news_widget],
            secondary=[],
            extra=[],
        ),
        openai_api_key="test_key",
    )

    # Mock the LLM response to keep default limit for normal queries
    with patch.object(copilot_data_service, "_generate_input_args") as mock_generate:
        # Configure mock to return result with default limit
        mock_generate.return_value = InputArgGenerationResult(
            input_args={
                "limit": 5,  # Should keep default for non-search queries
            },
            data_source=copilot_data_service._data_sources_map[mock_uuids.ID1.value],
            used_extra_param_options=False,
        )

        # Test normal query that should NOT trigger increased limit
        query_request = QueryWidgetRequest(
            widget_uuid=UUID(mock_uuids.ID1.value),
            widget_query="Show me the latest news",  # Normal query, not searching
            use_current_inputs=False,
        )

        result = await copilot_data_service.query_widgets([query_request])

        # Verify result keeps default limit
        assert result.content[0].payload.input_args["limit"] == 5


@pytest.mark.asyncio
async def test_filter_input_arg_options_fallback_to_semantic_search(
    mock_uuids: type[MockUUIDs],
):
    """Test that semantic search fallback works when pattern-based filtering fails."""
    # Setup service with basic configuration
    copilot_data_service = CopilotDataService(
        template_service=TemplateService(),
        logging_service=LoggingService(),
        widget_collection=None,
        openai_api_key="test_key",
    )

    # Create test data - stock-related options that won't match simple patterns
    # but should match semantically
    param_options = [
        WidgetParamOption(label="Equity Research Reports", value="equity_research"),
        WidgetParamOption(label="Bond Analysis Documents", value="bond_analysis"),
        WidgetParamOption(label="Market Commentary", value="market_commentary"),
        WidgetParamOption(label="Cryptocurrency Data", value="crypto_data"),
    ]

    extra_param_options = [
        WidgetParamOptions(
            param_name="report_type",
            widget_id="test_widget",
            widget_origin="test_origin",
            options=param_options,
        )
    ]

    # Create a mock data source
    mock_data_source = DataSource(
        origin="test_origin",
        id="test_widget",
        name="Test Widget",
        description="Test widget for reports",
        input_fields={},
        widget=Widget(
            uuid=UUID(mock_uuids.ID1.value),
            origin="test_origin",
            widget_id="test_widget",
            name="Test Widget",
            description="Test widget for reports",
            params=[],
            metadata={},
        ),
    )

    # Mock the pattern-based LLM to fail with MaxFunctionCallsError
    with patch("openbb_ada.services.copilot_data.prompt_chain") as mock_prompt_chain:
        # Setup the prompt_chain decorator to raise MaxFunctionCallsError
        def failing_decorator(*_, **__):
            def decorator(_):
                async def wrapper(*_, **__):
                    raise MaxFunctionCallsError("Max function calls reached")

                return wrapper

            return decorator

        mock_prompt_chain.side_effect = failing_decorator

        # Mock semantic search to return expected results
        with patch.object(
            copilot_data_service, "_semantic_search_options"
        ) as mock_semantic:
            mock_semantic.return_value = ["equity_research", "market_commentary"]

            # Test query for stock analysis reports
            result = await copilot_data_service._filter_input_arg_options(
                query="stock analysis reports",
                data_source=mock_data_source,
                extra_param_options=extra_param_options,
            )

            # Verify semantic search was called as fallback
            mock_semantic.assert_called_once_with(
                user_query="stock analysis reports",
                param_options=param_options,
                top_k=4,  # min(20, len(param_options)) = min(20, 4) = 4
            )

            # Verify correct filtered results returned
            assert "report_type" in result
            assert len(result["report_type"]) == 2
            assert result["report_type"][0].value == "equity_research"
            assert result["report_type"][1].value == "market_commentary"


@pytest.mark.asyncio
async def test_get_param_options_normalizes_label_style_values(
    mock_uuids: type[MockUUIDs],
):
    """When the LLM generates a label-style value (e.g. 'north america') for a
    field that has get_options=True, dependent parameters should receive the
    normalized snake_case value ('north_america') in options_endpoint_input_args.
    Values that are already in API format (no spaces) must not be altered."""

    widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        origin="Portfolio Risk",
        widget_id="load_factors_custom_obb",
        name="F-F Factors Data",
        description="Get dataset.",
        params=[
            WidgetParam(
                name="region",
                type="string",
                description="Select the region.",
                current_value="america",
                get_options=True,
                options_params=[],
            ),
            WidgetParam(
                name="factor",
                type="string",
                description="Select the factor.",
                current_value="5_Factors",
                get_options=True,
                options_params=[
                    OptionsEndpointParam(
                        type="text",
                        name="region",
                        description="region",
                        inherit_value_from="region",
                    ),
                ],
            ),
            WidgetParam(
                name="frequency",
                type="string",
                description="Select the frequency.",
                current_value="Monthly",
                get_options=True,
                options_params=[
                    OptionsEndpointParam(
                        type="text",
                        name="region",
                        description="region",
                        inherit_value_from="region",
                    ),
                    OptionsEndpointParam(
                        type="text",
                        name="factor",
                        description="factor",
                        inherit_value_from="factor",
                    ),
                ],
            ),
            WidgetParam(
                name="start_date",
                type="date",
                description="Start date.",
                current_value="2021-01-01",
            ),
            WidgetParam(
                name="end_date",
                type="date",
                description="End date.",
                current_value="2025-03-27",
            ),
        ],
        metadata={},
    )

    copilot_data_service = CopilotDataService(
        template_service=TemplateService(),
        logging_service=LoggingService(),
        widget_collection=WidgetCollection(
            primary=[widget],
            secondary=[],
            extra=[],
        ),
        openai_api_key=None,
    )

    data_source = copilot_data_service._data_sources_map[mock_uuids.ID1.value]

    # Simulate LLM generating label-style "north america" (with space)
    # and already-correct "5_Factors" (no space)
    input_arg_result = InputArgGenerationResult(
        input_args={
            "region": "north america",
            "factor": "5_Factors",
            "frequency": "Monthly",
            "start_date": "2021-01-01",
            "end_date": "2025-03-27",
        },
        data_source=data_source,
        used_extra_param_options=False,
    )

    payloads = await copilot_data_service._get_param_options_request_payloads(
        input_arg_result
    )

    # Build a lookup: param -> options_endpoint_input_args
    payload_map = {p.param: p.options_endpoint_input_args for p in payloads}

    # region has no dependencies — empty args
    assert payload_map["region"] == {}

    # factor depends on region — label "north america" should be normalized
    assert payload_map["factor"]["region"] == "north_america"

    # frequency depends on region and factor
    assert payload_map["frequency"]["region"] == "north_america"
    # "5_Factors" has no spaces — must NOT be lowercased
    assert payload_map["frequency"]["factor"] == "5_Factors"


@pytest.mark.asyncio
async def test_query_widgets_skips_missing_widgets(mock_uuids: type[MockUUIDs]):
    """When the LLM references widgets not on the dashboard (e.g. extras from
    vector search), query_widgets should skip them and proceed with the valid
    ones instead of failing the entire batch."""

    widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        origin="ODP",
        widget_id="ticker_information",
        name="Ticker Information",
        description="Information about a particular asset.",
        params=[
            WidgetParam(
                name="symbol",
                type="ticker",
                description="Ticker symbol.",
                current_value="AAPL",
            ),
        ],
        metadata={},
    )

    copilot_data_service = CopilotDataService(
        template_service=TemplateService(),
        logging_service=LoggingService(),
        widget_collection=WidgetCollection(
            primary=[widget],
            secondary=[],
            extra=[],
        ),
        openai_api_key="test_key",
    )

    valid_uuid = UUID(mock_uuids.ID1.value)
    extra_uuid = UUID(mock_uuids.ID2.value)  # not in _data_sources_map

    with patch.object(copilot_data_service, "_generate_input_args") as mock_generate:
        mock_generate.return_value = InputArgGenerationResult(
            input_args={"symbol": "MSFT"},
            data_source=copilot_data_service._data_sources_map[mock_uuids.ID1.value],
            used_extra_param_options=False,
        )

        # Mix of valid dashboard widget + extra widget not in map
        result = await copilot_data_service.query_widgets(
            query_widget_requests=[
                QueryWidgetRequest(
                    widget_uuid=valid_uuid,
                    widget_query="Update ticker to MSFT",
                ),
                QueryWidgetRequest(
                    widget_uuid=extra_uuid,
                    widget_query="Update ticker to MSFT",
                ),
            ]
        )

        # Should succeed with only the valid widget
        assert result is not None
        assert len(result.content) == 1
        # _generate_input_args should only be called for the valid widget
        assert mock_generate.call_count == 1


@pytest.mark.asyncio
async def test_query_widgets_errors_when_all_widgets_missing(
    mock_uuids: type[MockUUIDs],
):
    """When ALL requested widgets are missing from the dashboard,
    query_widgets should raise FunctionCallError."""

    widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        origin="ODP",
        widget_id="ticker_information",
        name="Ticker Information",
        description="Information about a particular asset.",
        params=[
            WidgetParam(
                name="symbol",
                type="ticker",
                description="Ticker symbol.",
                current_value="AAPL",
            ),
        ],
        metadata={},
    )

    copilot_data_service = CopilotDataService(
        template_service=TemplateService(),
        logging_service=LoggingService(),
        widget_collection=WidgetCollection(
            primary=[widget],
            secondary=[],
            extra=[],
        ),
        openai_api_key="test_key",
    )

    # Both UUIDs are not in the map
    with pytest.raises(FunctionCallError, match="None of the requested widgets"):
        await copilot_data_service.query_widgets(
            query_widget_requests=[
                QueryWidgetRequest(
                    widget_uuid=UUID(mock_uuids.ID2.value),
                    widget_query="Update ticker to MSFT",
                ),
                QueryWidgetRequest(
                    widget_uuid=UUID(mock_uuids.ID3.value),
                    widget_query="Update ticker to MSFT",
                ),
            ]
        )
