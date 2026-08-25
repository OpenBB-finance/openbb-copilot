"""Unit tests for NativeFunctionCallService."""

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from openbb_ai.models import (
    StatusUpdateSSE,
    Widget,
    WidgetCollection,
    WidgetParam,
)

from openbb_ada.errors import FunctionCallError
from openbb_ada.models import (
    AppArtifactSSE,
    AppArtifactTabInput,
    AvailableSemanticView,
    CodeGenerationResponse,
    ContextStructuredQueryResult,
    CopilotArtifact,
    RawObjectDataFormat,
    SourceInfo,
    SqlQueryFunctionCallResult,
    SqlQueryGenerationResult,
    SqlWidgetContext,
)
from openbb_ada.services import ContextService, LoggingService
from openbb_ada.services.native_function_call import NativeFunctionCallService


class TestNativeFunctionCallService:
    """Test suite for NativeFunctionCallService."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return {
            "document_service": MagicMock(),
            "logging_service": MagicMock(),
            "web_search_llm_service": MagicMock(),
            "context_service": MagicMock(),
            "template_service": MagicMock(),
            "citation_service": MagicMock(),
        }

    @pytest.fixture
    def service(self, mock_services):
        """Create NativeFunctionCallService instance with mocked dependencies."""
        return NativeFunctionCallService(
            document_service=mock_services["document_service"],
            logging_service=mock_services["logging_service"],
            web_search_llm_service=mock_services["web_search_llm_service"],
            context_service=mock_services["context_service"],
            template_service=mock_services["template_service"],
            citation_service=mock_services["citation_service"],
        )

    @pytest.mark.asyncio
    async def test_llm_think_string_plan(self, service):
        """Test _llm_think with string plan."""
        plan = "This is my plan to solve the problem"
        summary = "Planning approach"

        events = []
        async for event in service._llm_think(plan, summary):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], StatusUpdateSSE)
        assert events[0].data.message == summary
        assert events[0].data.details == [plan]
        assert events[1] == plan

    @pytest.mark.asyncio
    async def test_llm_think_list_plan(self, service):
        """Test _llm_think with list plan."""
        plan = [
            "Step 1: Analyze data",
            "Step 2: Create table",
            "Step 3: Generate chart",
        ]
        summary = "Planning workflow"

        events = []
        async for event in service._llm_think(plan, summary):
            events.append(event)

        expected_full_plan = "\n".join(plan)
        assert len(events) == 2
        assert isinstance(events[0], StatusUpdateSSE)
        assert events[0].data.message == summary
        assert events[0].data.details == [expected_full_plan]
        assert events[1] == expected_full_plan

    @pytest.mark.asyncio
    async def test_llm_complete(self, service):
        """Test llm_complete returns immediately."""
        events = []
        async for event in service.llm_complete():
            events.append(event)

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_llm_create_table_from_text_success(
        self, test_template_service, test_sql_agent_service
    ):
        """Test table artifacts remain usable when the result is handled."""
        context_service = ContextService(
            template_service=test_template_service,
            sql_agent_service=test_sql_agent_service,
            logging_service=LoggingService(),
        )
        service = NativeFunctionCallService(
            document_service=MagicMock(),
            logging_service=LoggingService(),
            web_search_llm_service=MagicMock(),
            context_service=context_service,
            template_service=test_template_service,
            citation_service=MagicMock(),
        )
        text_data = "Week 1: $25k, Week 2: $30k, Week 3: $28k"
        mock_table_data = [
            {"week": "Week 1", "amount": 25000},
            {"week": "Week 2", "amount": 30000},
            {"week": "Week 3", "amount": 28000},
        ]
        expected_materialized_rows = [
            {"index": 0, "week": "Week 1", "amount": 25000},
            {"index": 1, "week": "Week 2", "amount": 30000},
            {"index": 2, "week": "Week 3", "amount": 28000},
        ]

        with patch(
            "openbb_ada.services.native_function_call.extract_table_data",
            return_value=mock_table_data,
        ):
            events = []
            async for event in service.llm_create_table_from_text(text_data):
                events.append(event)

            # Should have status update with artifact followed by result
            assert len(events) == 2
            assert isinstance(events[0], StatusUpdateSSE)
            assert events[0].data.message == "Creating table"
            assert events[0].data.artifacts is not None
            assert len(events[0].data.artifacts) == 1
            assert events[0].data.artifacts[0].type == "table"

            assert isinstance(events[1], ContextStructuredQueryResult)
            result = events[1]
            assert "Created structured table with 3 data points" in result.answer
            assert len(result.artifacts) == 1

            artifact = result.artifacts[0]
            assert artifact.data_format.parse_as == "table"
            assert artifact.source_info.citable is False

            context_service = service._context_service
            assert (
                context_service.get_context_by_name(artifact.source_info.name) is None
            )

            rendered_result = service._handle_native_function_call_result(  # noqa: SLF001
                result
            )

            stored_context = context_service.get_context_by_name(
                artifact.source_info.name
            )
            assert stored_context is not None
            assert stored_context.data_format.parse_as == "table"
            assert (
                context_service.get_full_table_data(artifact.source_info.name)
                == expected_materialized_rows
            )
            assert "Created structured table with 3 data points" in rendered_result

    @pytest.mark.asyncio
    async def test_llm_create_table_from_text_failure(self, service, mock_services):
        """Test table creation failure handling."""
        text_data = "Invalid data"
        error_msg = "Failed to extract data"

        with patch(
            "openbb_ada.services.native_function_call.extract_table_data",
            side_effect=Exception(error_msg),
        ):
            events = []
            async for event in service.llm_create_table_from_text(text_data):
                events.append(event)

            # Should have error status
            assert len(events) == 1
            assert isinstance(events[0], StatusUpdateSSE)
            assert events[0].data.eventType == "ERROR"
            assert "Failed to extract data from text" in events[0].data.message

    @pytest.mark.asyncio
    async def test_llm_create_chart_from_table_success(self, service, mock_services):
        """Test successful chart creation from table."""
        table_artifact_id = "table_12345678"
        chart_type = "line"
        chart_title = "Sales Data"

        # Mock table data
        mock_table_data = [
            {"week": "Week 1", "amount": 25000},
            {"week": "Week 2", "amount": 30000},
        ]

        # Mock context object
        mock_context = MagicMock()
        mock_context.content = json.dumps(mock_table_data)
        mock_services["context_service"].get_context_by_name.return_value = mock_context

        # Mock valid chart parameters that match LineChartParameters schema
        mock_chart_params = {"chartType": "line", "xKey": "week", "yKey": ["amount"]}

        with patch(
            "openbb_ada.services.native_function_call.generate_chart_parameters",
            return_value=mock_chart_params,
        ):
            events = []
            async for event in service.llm_create_chart_from_table(
                table_artifact_id, chart_type, chart_title
            ):
                events.append(event)

            # Should have status update with artifact followed by result
            assert len(events) == 2
            assert isinstance(events[0], StatusUpdateSSE)
            assert events[0].data.message == "Creating chart"
            assert events[0].data.artifacts is not None
            assert len(events[0].data.artifacts) == 1
            assert events[0].data.artifacts[0].type == "chart"

            assert isinstance(events[1], ContextStructuredQueryResult)
            result = events[1]
            assert "Created line chart visualization" in result.answer
            assert len(result.artifacts) == 1

            artifact = result.artifacts[0]
            assert artifact.data_format.parse_as == "chart"
            assert artifact.source_info.citable is False

    @pytest.mark.asyncio
    async def test_llm_create_chart_from_table_not_found(self, service, mock_services):
        """Test chart creation when table not found."""
        table_artifact_id = "nonexistent_table"
        chart_type = "line"

        # Mock context service to return None for all lookups
        mock_services["context_service"].get_context_by_name.return_value = None
        mock_services["context_service"].get_context_by_id.return_value = None

        with pytest.raises(FunctionCallError):
            events = []
            async for event in service.llm_create_chart_from_table(
                table_artifact_id, chart_type
            ):
                events.append(event)

    @pytest.mark.asyncio
    async def test_llm_create_chart_from_table_with_axis_labels(
        self, service, mock_services
    ):
        """Test chart creation with custom axis labels."""
        table_artifact_id = "table_12345678"
        chart_type = "line"
        chart_title = "Sales Data"
        x_axis = "Time Period"
        y_axis = "Revenue ($)"

        # Mock table data
        mock_table_data = [{"week": "Week 1", "amount": 25000}]

        # Mock context object
        mock_context = MagicMock()
        mock_context.content = json.dumps(mock_table_data)
        mock_services["context_service"].get_context_by_name.return_value = mock_context

        # Mock valid chart parameters
        mock_chart_params = {"chartType": "line", "xKey": "week", "yKey": ["amount"]}

        with patch(
            "openbb_ada.services.native_function_call.generate_chart_parameters",
            return_value=mock_chart_params,
        ):
            events = []
            async for event in service.llm_create_chart_from_table(
                table_artifact_id, chart_type, chart_title, x_axis, y_axis
            ):
                events.append(event)

            # Verify axis labels are handled (chart generation logic uses them
            # internally)
            assert len(events) == 2
            assert isinstance(events[1], ContextStructuredQueryResult)

    @pytest.mark.asyncio
    async def test_llm_web_search(self, service, mock_services):
        """Test web search functionality."""
        query = "latest AI developments"
        summary = "Searching web for AI news"

        # Create async generator for web search response
        async def mock_web_search_generator():
            yield "mock event 1"
            yield "mock event 2"

        mock_services[
            "web_search_llm_service"
        ].query.return_value = mock_web_search_generator()

        events = []
        async for event in service.llm_web_search(query, summary):
            events.append(event)

        assert events == ["mock event 1", "mock event 2"]
        mock_services["web_search_llm_service"].query.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_query_unstructured_data(self, service, mock_services):
        """Test querying unstructured data."""
        data_ids = ["id1", "id2"]
        summary = "Querying documents"

        # Mock context service response
        mock_result = MagicMock()
        mock_result.citations = []
        mock_services[
            "context_service"
        ].read_unstructured_context_by_ids.return_value = [mock_result]

        events = []
        async for event in service.llm_query_unstructured_data(data_ids, summary):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], StatusUpdateSSE)
        assert events[0].data.message == summary
        assert events[1] == mock_result

    @pytest.mark.asyncio
    async def test_llm_query_uploaded_files(self, service, mock_services):
        """Test querying uploaded files."""
        query = "summarize financial data"
        summary = "Querying files"

        # Create async generator for document service response
        async def mock_document_query_generator():
            yield "mock file result"

        mock_services[
            "document_service"
        ].query_all_user_files.return_value = mock_document_query_generator()

        events = []
        async for event in service.llm_query_uploaded_files(query, summary):
            events.append(event)

        assert len(events) == 2  # Status update + mock result
        assert isinstance(events[0], StatusUpdateSSE)
        assert events[0].data.message == summary
        assert events[1] == "mock file result"

    def test_is_function_call_result(self, service):
        """Test _is_function_call_result method."""
        # Test with ContextStructuredQueryResult
        result1 = ContextStructuredQueryResult(
            answer="test", citations=[], artifacts=[]
        )
        assert service._is_function_call_result(result1) is True

        sql_result = SqlQueryFunctionCallResult(
            sql_query="SELECT 1",
            widget_uuid=str(uuid4()),
            widget_id="fred_public_dsge",
            widget_origin="SNOW backend",
            artifacts=[
                CopilotArtifact(
                    content="```sql\nSELECT 1\n```",
                    source_info=SourceInfo(
                        type="artifact",
                        uuid=uuid4(),
                        name="query_artifact_test",
                        description="Snowflake SQL query generated by AI copilot",
                        metadata={
                            "parse_as": "snowflake_query",
                            "query_data_source": {
                                "origin": "SNOW backend",
                                "id": "fred_public_dsge",
                                "widget_uuid": str(uuid4()),
                            },
                        },
                        citable=False,
                    ),
                    data_format=RawObjectDataFormat(
                        parse_as="snowflake_query",
                        query_data_source={
                            "origin": "SNOW backend",
                            "id": "fred_public_dsge",
                            "widget_uuid": str(uuid4()),
                        },
                    ),
                )
            ],
        )
        assert service._is_function_call_result(sql_result) is False

        sql_generation_result = SqlQueryGenerationResult(
            sql_query="SELECT 1",
            widget_uuid=str(uuid4()),
            widget_id="fred_public_dsge",
            widget_origin="SNOW backend",
            artifacts=[],
        )
        assert service._is_function_call_result(sql_generation_result) is True

        # Test with string (should be False)
        assert service._is_function_call_result("test string") is False

        # Test with other types
        assert service._is_function_call_result(123) is False
        assert service._is_function_call_result(None) is False

    @pytest.mark.asyncio
    async def test_llm_get_widget_input_state_single_widget(self, service):
        """Reads current widget query from widget context as JSON."""
        widget_collection = WidgetCollection(
            primary=[
                Widget(
                    uuid=uuid4(),
                    origin="SNOW backend",
                    widget_id="sec_widget",
                    name="SEC_CORPORATE_REPORT_ATTRIBUTES",
                    description="SEC widget",
                    params=[
                        WidgetParam(
                            name="query",
                            type="string",
                            description="SQL query",
                            current_value="SELECT * FROM table LIMIT 10",
                            executed_value="SELECT * FROM table LIMIT 100",
                        )
                    ],
                    metadata={},
                )
            ],
            secondary=[],
            extra=[],
        )

        events = []
        async for event in service.llm_get_widget_input_state(
            widget_collection=widget_collection
        ):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], StatusUpdateSSE)
        assert events[0].data.message == "Reading widget input state"
        assert isinstance(events[1], str)
        payload = json.loads(events[1])
        assert payload["widget"]["name"] == "SEC_CORPORATE_REPORT_ATTRIBUTES"
        assert payload["input_arguments"]["query"] == "SELECT * FROM table LIMIT 10"
        executed_query = payload["param_sync"]["executed_params"]["query"]
        assert executed_query == "SELECT * FROM table LIMIT 100"
        assert payload["param_sync"]["has_unapplied_changes"] is True
        assert payload["param_sync"]["unapplied_param_names"] == ["query"]

    @pytest.mark.asyncio
    async def test_llm_get_widget_input_state_ambiguous_widgets(self, service):
        """Returns a machine-readable ambiguity response when multiple widgets exist."""
        widget_a = Widget(
            uuid=uuid4(),
            origin="SNOW backend",
            widget_id="w1",
            name="Widget A",
            description="A",
            params=[
                WidgetParam(
                    name="query",
                    type="string",
                    description="SQL query",
                    current_value="SELECT 1",
                )
            ],
            metadata={},
        )
        widget_b = Widget(
            uuid=uuid4(),
            origin="SNOW backend",
            widget_id="w2",
            name="Widget B",
            description="B",
            params=[
                WidgetParam(
                    name="query",
                    type="string",
                    description="SQL query",
                    current_value="SELECT 2",
                )
            ],
            metadata={},
        )
        widget_collection = WidgetCollection(
            primary=[widget_a, widget_b],
            secondary=[],
            extra=[],
        )

        events = []
        async for event in service.llm_get_widget_input_state(
            widget_collection=widget_collection
        ):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[1], str)
        payload = json.loads(events[1])
        assert payload["error"] == "ambiguous_widget"
        available_uuids = {w["widget_uuid"] for w in payload["available_widgets"]}
        assert str(widget_a.uuid) in available_uuids
        assert str(widget_b.uuid) in available_uuids

    @pytest.mark.asyncio
    async def test_llm_get_widget_input_state_no_widgets_in_context(self, service):
        """Returns error when widget_collection is None."""
        events = []
        async for event in service.llm_get_widget_input_state(widget_collection=None):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], StatusUpdateSSE)
        assert isinstance(events[1], str)
        payload = json.loads(events[1])
        assert payload["error"] == "no_widgets_in_context"

    @pytest.mark.asyncio
    async def test_llm_get_widget_input_state_widget_not_found(self, service):
        """Returns error when widget_uuid doesn't match any widget."""
        widget_collection = WidgetCollection(
            primary=[
                Widget(
                    uuid=uuid4(),
                    origin="SNOW backend",
                    widget_id="w1",
                    name="Widget A",
                    description="A",
                    params=[],
                    metadata={},
                )
            ],
            secondary=[],
            extra=[],
        )

        events = []
        async for event in service.llm_get_widget_input_state(
            widget_uuid="non-existent-uuid",
            widget_collection=widget_collection,
        ):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[1], str)
        payload = json.loads(events[1])
        assert payload["error"] == "widget_not_found"
        assert payload["widget_uuid"] == "non-existent-uuid"

    @pytest.mark.asyncio
    async def test_llm_get_widget_input_state_filter_by_uuid(self, service):
        """Filters to specific widget when widget_uuid is provided."""
        target_uuid = uuid4()
        widget_a = Widget(
            uuid=target_uuid,
            origin="SNOW backend",
            widget_id="w1",
            name="Target Widget",
            description="A",
            params=[
                WidgetParam(
                    name="query",
                    type="string",
                    description="SQL query",
                    current_value="SELECT target",
                )
            ],
            metadata={},
        )
        widget_b = Widget(
            uuid=uuid4(),
            origin="SNOW backend",
            widget_id="w2",
            name="Other Widget",
            description="B",
            params=[
                WidgetParam(
                    name="query",
                    type="string",
                    description="SQL query",
                    current_value="SELECT other",
                )
            ],
            metadata={},
        )
        widget_collection = WidgetCollection(
            primary=[widget_a, widget_b],
            secondary=[],
            extra=[],
        )

        events = []
        async for event in service.llm_get_widget_input_state(
            widget_uuid=str(target_uuid),
            widget_collection=widget_collection,
        ):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[1], str)
        payload = json.loads(events[1])
        assert payload["widget"]["name"] == "Target Widget"
        assert payload["input_arguments"]["query"] == "SELECT target"

    @pytest.mark.asyncio
    async def test_llm_search_widgets_returns_table_artifact_with_widget_details(
        self, service
    ):
        """Widget search exposes match metadata as a table artifact and JSON."""
        target_widget = Widget(
            uuid=uuid4(),
            origin="Analytics Backend",
            widget_id="performance_returns",
            name="Performance Returns",
            description="Performance returns over time",
            params=[
                WidgetParam(
                    name="symbol",
                    type="ticker",
                    description="Ticker symbol",
                    default_value="AAPL",
                    current_value="MSFT",
                    options=["AAPL", "MSFT"],
                )
            ],
            category="Performance",
            sub_category="Returns",
            metadata={"schema": {"tableName": "PERFORMANCE_RETURNS"}},
        )
        other_widget = Widget(
            uuid=uuid4(),
            origin="Analytics Backend",
            widget_id="macro_calendar",
            name="Macro Calendar",
            description="Upcoming macroeconomic events",
            params=[],
            category="Economics",
            sub_category="Calendar",
            metadata={"schema": {"tableName": "MACRO_CALENDAR"}},
        )
        widget_collection = WidgetCollection(
            primary=[target_widget],
            secondary=[],
            extra=[other_widget],
        )
        service.set_app_widget_collection(widget_collection)

        events = []
        async for event in service.llm_search_widgets(
            query="performance returns",
        ):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], StatusUpdateSSE)
        assert len(events[0].data.artifacts) == 1

        artifact = events[0].data.artifacts[0]
        assert artifact.type == "table"
        assert artifact.content == [
            {
                "Name": "Performance Returns",
                "Description": "Performance returns over time",
                "Source": "Analytics Backend",
                "Category": "Performance",
                "Subcategory": "Returns",
            }
        ]

        payload = json.loads(events[1])
        assert payload["returned"] == 1
        assert payload["total"] == 1
        assert payload["matches"][0]["origin"] == "Analytics Backend"
        assert payload["matches"][0]["widget_id"] == "performance_returns"
        assert payload["query_terms"] == ["performance", "returns"]
        assert payload["matches"][0]["params"] == [
            {
                "name": "symbol",
                "type": "ticker",
                "description": "Ticker symbol",
                "current_value": "MSFT",
                "default_value": "AAPL",
                "options": ["AAPL", "MSFT"],
            }
        ]

    @pytest.mark.asyncio
    async def test_llm_search_widgets_returns_guidance_when_no_widgets_match(
        self, service
    ):
        """No-match searches tell the model to retry broadly or stop."""
        widget_collection = WidgetCollection(
            primary=[
                Widget(
                    uuid=uuid4(),
                    origin="OpenBB",
                    widget_id="sector_exposure",
                    name="Exposure by Sector",
                    description="Portfolio exposure by sector",
                    params=[],
                    metadata={},
                )
            ],
            secondary=[],
            extra=[],
        )
        service.set_app_widget_collection(widget_collection)

        events = []
        async for event in service.llm_search_widgets(query="healthcare"):
            events.append(event)

        payload = json.loads(events[1])
        assert payload["returned"] == 0
        assert "Retry once with broader adjacent terms" in payload["guidance"]

    @pytest.mark.asyncio
    async def test_llm_create_app_emits_app_artifact(self, service):
        """Creates an app artifact with tabs and widget refs."""
        price_widget = Widget(
            uuid=uuid4(),
            origin="OpenBB",
            widget_id="price_chart",
            name="Price Chart",
            description="Price history",
            params=[
                WidgetParam(
                    name="symbol",
                    type="ticker",
                    description="Ticker symbol",
                )
            ],
            metadata={},
        )
        news_widget = Widget(
            uuid=uuid4(),
            origin="OpenBB",
            widget_id="news",
            name="News",
            description="Company news",
            params=[],
            metadata={},
        )
        widget_collection = WidgetCollection(
            primary=[price_widget],
            secondary=[news_widget],
            extra=[],
        )
        service.set_app_widget_collection(widget_collection)

        events = []
        async for event in service.llm_create_app(
            name="AAPL Dashboard",
            description="AAPL price and news dashboard",
            tabs=[
                AppArtifactTabInput(
                    id="overview",
                    name="Overview",
                    layout=[
                        {
                            "origin": "OpenBB",
                            "widget_id": "price_chart",
                            "x": 0,
                            "y": 0,
                            "w": 40,
                            "h": 12,
                            "state": {"params": {"symbol": "AAPL"}},
                        },
                        {
                            "origin": "OpenBB",
                            "widget_id": "news",
                            "x": 0,
                            "y": 12,
                            "w": 40,
                            "h": 10,
                        },
                    ],
                )
            ],
        ):
            events.append(event)

        assert isinstance(events[0], StatusUpdateSSE)
        assert isinstance(events[1], AppArtifactSSE)
        assert isinstance(events[2], str)

        artifact = events[1].data
        assert artifact.type == "app"
        assert artifact.name == "AAPL Dashboard"
        assert artifact.app.allowCustomization is True
        assert list(artifact.app.tabs) == ["overview"]
        assert artifact.app.tabs["overview"].layout[0].i == "price_chart"
        assert artifact.app.tabs["overview"].layout[0].state == {
            "params": {"symbol": "AAPL"}
        }
        assert [ref.widget_id for ref in artifact.widget_refs] == [
            "price_chart",
            "news",
        ]
        assert 'Created app "AAPL Dashboard" with 2 widgets across 1 tab' in events[2]

    @pytest.mark.asyncio
    async def test_llm_create_app_rejects_unknown_widget_params(self, service):
        """Rejects dashboard app params that do not exist on the selected widget."""
        exposure_widget = Widget(
            uuid=uuid4(),
            origin="OpenBB",
            widget_id="sector_exposure",
            name="Exposure by Sector",
            description="Portfolio exposure by sector",
            params=[
                WidgetParam(
                    name="portfolio_id",
                    type="text",
                    description="Portfolio identifier",
                )
            ],
            metadata={},
        )
        widget_collection = WidgetCollection(
            primary=[exposure_widget],
            secondary=[],
            extra=[],
        )
        service.set_app_widget_collection(widget_collection)

        events = []
        async for event in service.llm_create_app(
            name="Healthcare Dashboard",
            description="Healthcare dashboard",
            tabs=[
                AppArtifactTabInput(
                    id="overview",
                    name="Overview",
                    layout=[
                        {
                            "origin": "OpenBB",
                            "widget_id": "sector_exposure",
                            "x": 0,
                            "y": 0,
                            "w": 40,
                            "h": 12,
                            "state": {"params": {"Healthcare": "Healthcare"}},
                        }
                    ],
                )
            ],
        ):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], str)
        assert "does not support state.params ['Healthcare']" in events[0]
        assert "portfolio_id (text): Portfolio identifier" in events[0]

    @pytest.mark.asyncio
    async def test_llm_create_app_rejects_invalid_widget_param_options(self, service):
        """Rejects dashboard app param values outside the widget's static options."""
        exposure_widget = Widget(
            uuid=uuid4(),
            origin="OpenBB",
            widget_id="sector_exposure",
            name="Exposure by Sector",
            description="Portfolio exposure by sector",
            params=[
                WidgetParam(
                    name="portfolio_id",
                    type="text",
                    description="Portfolio identifier",
                    options=["Portfolio 1"],
                )
            ],
            metadata={},
        )
        widget_collection = WidgetCollection(
            primary=[exposure_widget],
            secondary=[],
            extra=[],
        )
        service.set_app_widget_collection(widget_collection)

        events = []
        async for event in service.llm_create_app(
            name="Healthcare Dashboard",
            description="Healthcare dashboard",
            tabs=[
                AppArtifactTabInput(
                    id="overview",
                    name="Overview",
                    layout=[
                        {
                            "origin": "OpenBB",
                            "widget_id": "sector_exposure",
                            "x": 0,
                            "y": 0,
                            "w": 40,
                            "h": 12,
                            "state": {"params": {"portfolio_id": "Healthcare"}},
                        }
                    ],
                )
            ],
        ):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], str)
        assert "does not support these state.params values" in events[0]
        assert "portfolio_id=['Healthcare'] (allowed: Portfolio 1)" in events[0]

    @pytest.mark.asyncio
    async def test_llm_create_app_rejects_parameterized_widget_without_params(
        self, service
    ):
        """Rejects configurable widgets that would otherwise keep defaults."""
        price_widget = Widget(
            uuid=uuid4(),
            origin="OpenBB",
            widget_id="price_chart",
            name="Price Chart",
            description="Price history",
            params=[
                WidgetParam(
                    name="symbol",
                    type="ticker",
                    description="Ticker symbol",
                    current_value="AAPL",
                )
            ],
            metadata={},
        )
        widget_collection = WidgetCollection(
            primary=[price_widget],
            secondary=[],
            extra=[],
        )
        service.set_app_widget_collection(widget_collection)

        events = []
        async for event in service.llm_create_app(
            name="Healthcare Dashboard",
            description="Healthcare dashboard",
            tabs=[
                AppArtifactTabInput(
                    id="overview",
                    name="Overview",
                    layout=[
                        {
                            "origin": "OpenBB",
                            "widget_id": "price_chart",
                            "x": 0,
                            "y": 0,
                            "w": 40,
                            "h": 12,
                        }
                    ],
                )
            ],
        ):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], str)
        assert "must set explicit state.params" in events[0]
        assert "symbol (ticker): Ticker symbol" in events[0]

    @pytest.mark.asyncio
    async def test_llm_create_app_rejects_empty_widget_selection(self, service):
        """Dashboard apps need at least one selected widget."""
        widget_collection = WidgetCollection(
            primary=[
                Widget(
                    uuid=uuid4(),
                    origin="OpenBB",
                    widget_id="sector_exposure",
                    name="Exposure by Sector",
                    description="Portfolio exposure by sector",
                    params=[],
                    metadata={},
                )
            ],
            secondary=[],
            extra=[],
        )
        service.set_app_widget_collection(widget_collection)

        events = []
        async for event in service.llm_create_app(
            name="Empty Dashboard",
            description="No widgets",
            tabs=[],
        ):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], str)
        assert "no widgets were selected" in events[0]

    @pytest.mark.asyncio
    async def test_llm_create_app_rejects_unknown_widget(self, service):
        """Validation failures return an LLM-visible correction message only."""
        widget_collection = WidgetCollection(
            primary=[
                Widget(
                    uuid=uuid4(),
                    origin="OpenBB",
                    widget_id="known",
                    name="Known",
                    description="Known widget",
                    params=[],
                    metadata={},
                )
            ],
            secondary=[],
            extra=[],
        )
        service.set_app_widget_collection(widget_collection)

        events = []
        async for event in service.llm_create_app(
            name="Broken Dashboard",
            description="Uses an unknown widget",
            tabs=[
                AppArtifactTabInput(
                    id="overview",
                    name="Overview",
                    layout=[
                        {
                            "origin": "OpenBB",
                            "widget_id": "missing",
                            "x": 0,
                            "y": 0,
                            "w": 40,
                            "h": 10,
                        }
                    ],
                )
            ],
        ):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], str)
        assert "unknown widget" in events[0]
        assert "missing" in events[0]

    @pytest.mark.asyncio
    async def test_llm_generate_sql_query_reports_explicit_semantic_view_usage(self):
        class StubSnowflakeCortexAnalystService:
            async def generate_code(self, request):
                return CodeGenerationResponse(
                    generated_code="SELECT 1",
                    generation_source="cortex_analyst",
                )

        service = NativeFunctionCallService(
            document_service=MagicMock(),
            logging_service=MagicMock(),
            web_search_llm_service=MagicMock(),
            context_service=MagicMock(),
            template_service=MagicMock(),
            citation_service=MagicMock(),
            snowflake_cortex_analyst_service=StubSnowflakeCortexAnalystService(),
            semantic_views=["DB.SCHEMA.EXPLICIT_VIEW"],
        )
        sql_widget = SqlWidgetContext(
            widget_uuid="widget-1",
            widget_id="revenue_widget",
            widget_origin="Snowflake",
            sql_schema={
                "database": "DB",
                "schema": "SCHEMA",
                "tableName": "REVENUE",
            },
            current_sql="SELECT * FROM REVENUE",
        )

        events = []
        async for event in service.llm_generate_sql_query_snowflake(
            user_request="show revenue",
            widget_uuid="widget-1",
            generate_query_only=True,
            sql_widgets=[sql_widget],
        ):
            events.append(event)

        assert len(events) == 4
        assert isinstance(events[1], StatusUpdateSSE)
        assert events[1].data.message == "Using semantic view context"
        assert events[1].data.details == [
            {
                "Reason": "Explicit selection",
                "Semantic Views": "DB.SCHEMA.EXPLICIT_VIEW",
            }
        ]
        assert isinstance(events[2], StatusUpdateSSE)
        assert events[2].data.message == "Cortex Analyst generated SQL"
        assert isinstance(events[3], SqlQueryGenerationResult)

    @pytest.mark.asyncio
    async def test_llm_generate_sql_query_reports_model_selected_semantic_view_usage(
        self,
    ):
        class StubSnowflakeCortexAnalystService:
            async def generate_code(self, request):
                return CodeGenerationResponse(
                    generated_code="SELECT 1",
                    generation_source="cortex_analyst",
                )

        service = NativeFunctionCallService(
            document_service=MagicMock(),
            logging_service=MagicMock(),
            web_search_llm_service=MagicMock(),
            context_service=MagicMock(),
            template_service=MagicMock(),
            citation_service=MagicMock(),
            snowflake_cortex_analyst_service=StubSnowflakeCortexAnalystService(),
            available_semantic_views=[
                AvailableSemanticView(
                    fqn="DB.SCHEMA.REVENUE_VIEW",
                    database="DB",
                    schema="SCHEMA",
                    view_name="REVENUE_VIEW",
                    base_table="REVENUE",
                )
            ],
        )
        sql_widget = SqlWidgetContext(
            widget_uuid="widget-1",
            widget_id="revenue_widget",
            widget_origin="Snowflake",
            sql_schema={
                "database": "DB",
                "schema": "SCHEMA",
                "tableName": "REVENUE",
            },
            current_sql="SELECT * FROM REVENUE",
        )

        events = []
        async for event in service.llm_generate_sql_query_snowflake(
            user_request="show revenue",
            widget_uuid="widget-1",
            generate_query_only=True,
            semantic_view="DB.SCHEMA.REVENUE_VIEW",
            sql_widgets=[sql_widget],
        ):
            events.append(event)

        assert len(events) == 4
        assert isinstance(events[1], StatusUpdateSSE)
        assert events[1].data.message == "Using semantic view context"
        assert events[1].data.details == [
            {
                "Reason": "Model selected available semantic view",
                "Semantic Views": "DB.SCHEMA.REVENUE_VIEW",
            }
        ]
        assert isinstance(events[2], StatusUpdateSSE)
        assert events[2].data.message == "Cortex Analyst generated SQL"
        assert isinstance(events[3], SqlQueryGenerationResult)

    def test_select_semantic_views_prefers_explicit_selection(self, service):
        service._semantic_views = ["DB.SCHEMA.EXPLICIT_VIEW"]
        service._available_semantic_views = [
            AvailableSemanticView(
                fqn="DB.SCHEMA.REVENUE_VIEW",
                database="DB",
                schema="SCHEMA",
                view_name="REVENUE_VIEW",
                base_table="REVENUE",
            )
        ]

        result, _ = service._select_semantic_views_with_reason(
            requested_semantic_view="DB.SCHEMA.REVENUE_VIEW"
        )

        assert result == ["DB.SCHEMA.EXPLICIT_VIEW"]

    def test_select_semantic_views_uses_model_selected_available_semantic_view(
        self, service
    ):
        service._available_semantic_views = [
            AvailableSemanticView(
                fqn="DB.SCHEMA.REVENUE_VIEW",
                database="DB",
                schema="SCHEMA",
                view_name="REVENUE_VIEW",
                base_table="REVENUE",
            ),
            AvailableSemanticView(
                fqn="OTHER.SCHEMA.COST_VIEW",
                database="OTHER",
                schema="SCHEMA",
                view_name="COST_VIEW",
                base_table="COSTS",
            ),
        ]

        result, _ = service._select_semantic_views_with_reason(
            requested_semantic_view="DB.SCHEMA.REVENUE_VIEW"
        )

        assert result == ["DB.SCHEMA.REVENUE_VIEW"]

    def test_select_semantic_views_returns_none_for_unavailable_model_selection(
        self, service
    ):
        service._available_semantic_views = [
            AvailableSemanticView(
                fqn="DB.SCHEMA.REVENUE_VIEW",
                database="DB",
                schema="SCHEMA",
                view_name="REVENUE_VIEW",
                base_table="REVENUE",
                comment="Curated revenue metrics by product and region",
            ),
            AvailableSemanticView(
                fqn="DB.SCHEMA.COST_VIEW",
                database="DB",
                schema="SCHEMA",
                view_name="COST_VIEW",
                base_table="COSTS",
                comment="Curated operating cost metrics",
            ),
        ]

        result, _ = service._select_semantic_views_with_reason(
            requested_semantic_view="DB.SCHEMA.UNKNOWN_VIEW"
        )

        assert result is None
