import json
from inspect import Parameter, Signature, signature
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Literal,
    Sequence,
    Tuple,
    cast,
)

from fastapi import HTTPException
from magentic import AssistantMessage, FunctionCall, FunctionResultMessage
from magentic.chat_model.message import Message
from openbb_ai.models import (
    AgentTool,
    ClientArtifact,
    ClientFunctionCallError,
    FunctionCallResponse,
    FunctionCallSSE,
    FunctionCallSSEData,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    StatusUpdateSSE,
    StatusUpdateSSEData,
)
from pydantic import ValidationError, create_model

from ..errors import FunctionCallError
from ..models import (
    DataSourceRequestPayload,
    DataSourceSearchFailure,
    DataSourceSearchQuery,
    NavigationBarTab,
    QueryExtraWidgetsRequest,
    QueryExtraWidgetsResult,
    QueryWidgetRequest,
    QueryWidgetsResult,
    TaskRequest,
    WidgetDataSourceParamOptionsRequest,
    WidgetDataSourceRequest,
    WidgetParamOptions,
    WidgetQueryRequest,
)
from ._logging import LoggingService
from .copilot_data import CopilotDataService


class ClientFunctionCallService:
    def __init__(
        self,
        copilot_data_service: CopilotDataService,
        logging_service: LoggingService,
    ):
        self._copilot_data_service = copilot_data_service
        self._logging_service = logging_service
        # Registry of dynamically-generated flat MCP tool function names.
        # Populated by build_flat_mcp_tools(), checked by is_client_function_call().
        self._dynamic_mcp_tool_names: set[str] = set()
        # Reverse mapping: sanitized_name -> (server_id, original_tool_name)
        self._dynamic_mcp_tool_lookup: dict[str, tuple[str, str]] = {}
        # Reverse mapping: (server_id, original_tool_name) -> flat function
        self._dynamic_mcp_tool_by_origin: dict[tuple[str, str], Callable] = {}

    # ------------------------------------------------------------------
    # Flat MCP tool factory
    # ------------------------------------------------------------------

    def build_flat_mcp_tools(
        self,
        tools: Sequence[AgentTool],
    ) -> list[Callable]:
        """Create one flat function per MCP tool for direct LLM invocation.

        Each generated function has flat parameters matching the tool's
        ``input_schema.properties`` so that proxy layers (e.g. OpenRouter)
        never need to serialise nested dicts.  Internally the function
        delegates to ``llm_execute_agent_tool``.

        Returns a list of callables ready to be passed to magentic.
        Also populates ``self._dynamic_mcp_tool_names`` and
        ``self._dynamic_mcp_tool_lookup``.
        """
        from ..utils.utils import sanitize_tool_name

        # Clear previous registrations (idempotent across calls).
        self._dynamic_mcp_tool_names.clear()
        self._dynamic_mcp_tool_lookup.clear()
        self._dynamic_mcp_tool_by_origin.clear()

        flat_functions: list[Callable] = []
        used_names: set[str] = set()

        for tool in tools:
            server_id = tool.server_id or ""
            sanitized = sanitize_tool_name(
                tool_name=tool.name,
                existing_names=used_names,
            )
            used_names.add(sanitized)

            fn = self._make_mcp_tool_function(tool, sanitized, server_id)
            flat_functions.append(fn)

            self._dynamic_mcp_tool_names.add(sanitized)
            self._dynamic_mcp_tool_lookup[sanitized] = (server_id, tool.name)
            self._dynamic_mcp_tool_by_origin[(server_id, tool.name)] = fn

        self._logging_service.debug(
            "Built %d flat MCP tool functions: %s",
            len(flat_functions),
            sorted(used_names),
        )
        return flat_functions

    def _make_mcp_tool_function(
        self,
        tool: AgentTool,
        sanitized_name: str,
        server_id: str,
    ) -> Callable:
        """Build a single flat async-generator function for one MCP tool."""
        # Extract schema info
        input_schema = tool.input_schema or {}
        properties = input_schema.get("properties", {})
        required_set = set(input_schema.get("required", []))

        # Build inspect.Parameter list using KEYWORD_ONLY so that
        # display_summary (optional, with default) can appear first —
        # the LLM sees it first and fills it in before tool params,
        # which helps it reason about what it's about to do.
        params: list[Parameter] = []

        # display_summary comes first so the LLM fills it before tool params.
        params.append(
            Parameter(
                "display_summary",
                Parameter.KEYWORD_ONLY,
                default="Executing MCP tool",
                annotation=str,
            ),
        )

        tool_param_names: list[str] = []
        # Sort so required params come before optional ones (Python
        # Signature forbids non-default args after default args).
        sorted_props = sorted(
            properties.items(),
            key=lambda item: item[0] not in required_set,
        )
        for pname, pschema in sorted_props:
            if pname == "display_summary":
                continue
            tool_param_names.append(pname)
            annotation = self._json_type_to_python(pschema.get("type", "string"))
            if pname in required_set:
                params.append(
                    Parameter(
                        pname,
                        Parameter.KEYWORD_ONLY,
                        annotation=annotation,
                    )
                )
            else:
                params.append(
                    Parameter(
                        pname,
                        Parameter.KEYWORD_ONLY,
                        default=None,
                        annotation=annotation | None,
                    )
                )

        sig = Signature(params)

        # Build docstring from tool description
        doc_parts = [tool.description or f"Execute MCP tool: {tool.name}"]
        doc_parts.append("")
        doc_parts.append("Parameters")
        doc_parts.append("----------")
        doc_parts.append(
            "display_summary : str\n"
            "    A short, human-readable description of what you are about to do,\n"
            "    displayed to the user while the tool executes.\n"
            "    This is DISPLAY-ONLY — values here are NOT forwarded to the tool."
        )
        for pname, pschema in properties.items():
            pdesc = pschema.get("description", "")
            ptype = pschema.get("type", "string")
            req_marker = " (required)" if pname in required_set else ""
            params_line = f"{pname} : {ptype}{req_marker}"
            if pdesc:
                params_line += f"\n    {pdesc}"
            doc_parts.append(params_line)

        docstring = "\n".join(doc_parts)

        # Capture variables for closure
        _self = self
        _server_id = server_id
        _tool_name = tool.name
        _tool_param_names = tool_param_names
        _required_set = required_set
        _available_tools = [tool]

        async def _flat_mcp_tool(**kwargs):
            summary = kwargs.pop("display_summary", "Executing MCP tool")
            # Only strip None for optional params; required params are
            # passed through as-is so the validation layer can flag them.
            tool_args = {
                k: v
                for k, v in kwargs.items()
                if k in _tool_param_names and (k in _required_set or v is not None)
            }
            async for event in _self.llm_execute_agent_tool(
                server_id=_server_id,
                summary=summary,
                tool_name=_tool_name,
                tool_args=tool_args,
                available_tools=_available_tools,
            ):
                yield event

        # Override function metadata for magentic introspection
        _flat_mcp_tool.__name__ = sanitized_name
        _flat_mcp_tool.__qualname__ = sanitized_name
        _flat_mcp_tool.__doc__ = docstring
        _flat_mcp_tool.__signature__ = sig  # type: ignore[attr-defined]

        return _flat_mcp_tool

    @staticmethod
    def _json_type_to_python(json_type: str) -> type:
        """Map JSON Schema type strings to Python types."""
        mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
        }
        return mapping.get(json_type, str)

    def get_original_tool_info(self, sanitized_name: str) -> tuple[str, str] | None:
        """Return (server_id, original_tool_name) for a dynamic MCP tool name."""
        return self._dynamic_mcp_tool_lookup.get(sanitized_name)

    @staticmethod
    def _is_missing_required_value(value: Any) -> bool:
        return value is None or (isinstance(value, str) and not value.strip())

    @staticmethod
    def _extract_tool_schema_fields(
        tool: AgentTool | None,
    ) -> tuple[list[str], list[str], bool]:
        if tool is None or not isinstance(tool.input_schema, dict):
            return [], [], False

        input_schema = tool.input_schema
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
        return required_args, parameter_names, bool(parameter_names or required_args)

    def _find_agent_tool(
        self,
        tool_name: str,
        server_id: str,
        available_tools: Sequence[AgentTool] | None,
    ) -> AgentTool | None:
        if not available_tools:
            return None

        for tool in available_tools:
            if tool.name == tool_name and (
                tool.server_id is None or tool.server_id == server_id
            ):
                return tool

        actual_tool_name = tool_name.split("_", 1)[1] if "_" in tool_name else tool_name
        for tool in available_tools:
            if tool.server_id != server_id:
                continue
            if tool.name == actual_tool_name or tool.name.endswith(
                f"_{actual_tool_name}"
            ):
                return tool

        return None

    def _build_agent_tool_diagnostics(
        self,
        server_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        available_tools: Sequence[AgentTool] | None,
    ) -> dict[str, Any]:
        matched_tool = self._find_agent_tool(
            tool_name=tool_name,
            server_id=server_id,
            available_tools=available_tools,
        )
        required_args, parameter_names, schema_available = (
            self._extract_tool_schema_fields(matched_tool)
        )
        provided_arg_keys = sorted(tool_args.keys())
        missing_required_args = [
            arg
            for arg in required_args
            if self._is_missing_required_value(tool_args.get(arg))
        ]
        optional_args = [arg for arg in parameter_names if arg not in required_args]

        return {
            "server_id": server_id,
            "tool_name": tool_name,
            "tool_found": matched_tool is not None,
            "schema_available": schema_available,
            "tool_server_id": matched_tool.server_id if matched_tool else None,
            "required_args": required_args,
            "optional_args": optional_args,
            "parameter_names": parameter_names,
            "provided_arg_keys": provided_arg_keys,
            "missing_required_args": missing_required_args,
            "tool_args_empty": not bool(tool_args),
            "tool_args_hash": self._logging_service.make_stable_hash(tool_args),
            "input_schema_hash": (
                self._logging_service.make_stable_hash(matched_tool.input_schema)
                if matched_tool and matched_tool.input_schema is not None
                else None
            ),
        }

    async def llm_assign_tasks_to_agents(
        self,
        task_requests: list[TaskRequest],
        summary: str = "Assigning tasks to agents",
        expect_direct_response: bool = True,
    ) -> AsyncGenerator[FunctionCallResponse | StatusUpdateSSE, None]:
        """Use this tool to assign tasks to agents.

        Parameters
        ----------
        task_requests : list[TaskRequest]
            List of task requests to assign to agents.
            Each task request should have a description and a query string.
            The description is used to identify the task to assign.
            The query string is the actual task to assign.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.

        Examples
        --------
        "Assign tasks to agents for data retrieval and analysis."
        {
            "summary": "Assigning prompt optimization to agent XYZ",
            "task_requests": [
                {
                    "id": <uuid>,
                    "description": "@XYZ please optimize the following prompt: ...",
                    "assigned_holder_url": "https://example.com",
                    "assigned_agent_id": "agent_xyz",
                }
            ]
        }
        """
        self._logging_service.info("Assigning tasks to agents: %s", summary)
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
            )
        )
        yield FunctionCallResponse(
            function="assign_tasks_to_agents",
            input_arguments={
                "task_requests": task_requests,
            },
            extra_state={
                "copilot_function_call_arguments": {
                    "summary": summary,
                    "task_requests": task_requests,
                },
                # Signal that we expect a direct response from the agent
                # This can be used by the frontend to suppress the tool result display
                "expect_direct_response": expect_direct_response,
            },
        )

    async def llm_query_extra_widgets(
        self,
        search_queries: list[DataSourceSearchQuery],
        summary: str = "Searching widgets",
        extra_state: dict[str, Any] | None = None,
    ) -> AsyncGenerator[FunctionCallResponse | StatusUpdateSSE, None]:
        """Use natural language to search for data sources.

        Important: You must mention the type of data source you want to search
        for as part of your query.

        Parameters
        ----------
        search_queries : list[DataSourceSearchQuery]
            List of search queries to execute.
            Each query should have a description and a query string.
            The description is used to identify the type of data source to search for.
            The query string is the actual search query to execute.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.

        Examples
        --------
        "Show me the news on TSLA between 2023-10-10 and 2024-10-10, and the
        historical stock price of AAPL for 2023."
        {
            "summary": "Searching for TSLA news and AAPL stock price",
            "search_queries": [
                {
                    "description": "earnings transcript",
                    "query": "Show me the news on TSLA between 2023-10-10 and 2024-10-10."
                },
                {
                    "description": "historical stock price",
                    "query": "Historical stock price for AAPL in 2023."
                }
            ]
        }
        """  # noqa: E501
        async for event in self._search_extra_widgets(
            search_queries=search_queries,
            summary=summary,
            extra_state=extra_state,
        ):
            if isinstance(event, StatusUpdateSSE):
                yield event
            elif isinstance(event, QueryExtraWidgetsResult):
                async for fc_event in self._send_client_widget_function_call(
                    function="get_extra_widget_data",
                    result=event,
                    copilot_function_call_arguments={
                        "summary": summary,
                        "search_queries": search_queries,
                    },
                    queries_length=len(search_queries),
                    extra_state=extra_state,
                ):
                    yield fc_event

    async def llm_add_widget_to_dashboard(
        self,
        search_queries: list[DataSourceSearchQuery],
        summary: str = "Searching widgets",
        extra_state: dict[str, Any] | None = None,
    ) -> AsyncGenerator[FunctionCallResponse | StatusUpdateSSE, None]:
        """Use natural language to first search for and then add the data sources.

        Important: You must mention the type of data source you want to search
        for as part of your query.

        Parameters
        ----------
        search_queries : list[DataSourceSearchQuery]
            List of search queries to execute.
            Each query should have a description and a query string.
            The description is used to identify the type of data source to search for.
            The query string is the actual search query to execute.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.

        Each search query can optionally include an `inner_tab` field with the
        tab name where the widget should be placed (requires a navigation bar
        on the dashboard). If not provided, the widget will be placed in the
        currently active tab.

        Examples
        --------
        "Show me the news on TSLA between 2023-10-10 and 2024-10-10, and the
        historical stock price of AAPL for 2023."
        {
            "summary": "Searching for TSLA news and AAPL stock price",
            "search_queries": [
                {
                    "description": "earnings transcript",
                    "query": "Show me the news on TSLA between 2023-10-10 and 2024-10-10.",
                    "inner_tab": "News"
                },
                {
                    "description": "historical stock price",
                    "query": "Historical stock price for AAPL in 2023.",
                    "inner_tab": "Stocks"
                }
            ]
        }
        """  # noqa: E501
        # Convert to pydantic model if needed (e.g. when resuming from a
        # continued function call, search_queries may be plain dicts after
        # the JSON round-trip through SSE).
        if search_queries and not isinstance(search_queries[0], DataSourceSearchQuery):
            search_queries = [
                DataSourceSearchQuery(**query) for query in search_queries
            ]

        async for event in self._search_extra_widgets(
            search_queries=search_queries,
            summary=summary,
            extra_state=extra_state,
        ):
            if isinstance(event, StatusUpdateSSE):
                yield event
            elif isinstance(event, QueryExtraWidgetsResult):
                async for fc_event in self._send_client_widget_function_call(
                    function="add_widget_to_dashboard",
                    result=event,
                    copilot_function_call_arguments={
                        "search_queries": search_queries,
                        "summary": summary,
                    },
                    queries_length=len(search_queries),
                    extra_state=extra_state,
                    search_queries=search_queries,
                ):
                    yield fc_event

    async def llm_query_widgets(
        self,
        widget_queries: list[WidgetQueryRequest],
        summary: str = "Querying widgets",
        extra_state: dict[str, Any] | None = None,
    ) -> AsyncGenerator[FunctionCallResponse | StatusUpdateSSE, None]:
        """Query widget data sources.

        Each widget query must specify the UUID of the widget, and a query
        of the data that the user has asked for.

        IMPORTANT:
        - If the user is asking for the current prompt/query/input arguments
          selected in a widget, do NOT use this tool just to inspect those
          values. Use `llm_get_widget_input_state` to read the current widget
          input state.
        - Use this tool when you actually need to fetch data from the widget.

        Be very specific on which data you want to fetch from the widget.

        Parameters
        ----------
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        widget_queries : list[WidgetQueryRequest]
            List of widget queries to execute. Each query must specify the
            widget UUID, the query to execute, and whether to use the current
            input arguments or not.

        Example
        --------
        {
            "summary": "Querying widgets for AAPL data",
            "widget_queries": [
                {
                    "widget_uuid": "defgh-67890",
                    "query": "Summarize the currently-selected news for AAPL.",
                    "does_query_match_input_args": "Yes, the query matches the current input arguments since we want to summarize the currently-selected news.",
                    "use_current_inputs": True,
                },
                {
                    "widget_uuid": "alkjdg-98765",  // you can query the same widget multiple times
                    "query": "Show me the balance sheet for AAPL.",
                    "does_query_match_input_args": "No, we need to set AAPL as the input argument to get its balance sheet data.",
                    "use_current_inputs": False
                },
                {
                    "widget_uuid": "alkjdg-98765",
                    "query": "Show me the balance sheet for AMZN.",
                    "does_query_match_input_args": "No, we need to set AMZN as the input argument to get its balance sheet data.",
                    "use_current_inputs": False
                },
            ]
        }
        """  # noqa: E501

        async for event in self._search_primary_and_secondary_widgets(
            widget_queries=widget_queries,
            summary=summary,
            extra_state=extra_state,
        ):
            if isinstance(event, StatusUpdateSSE):
                yield event
            elif isinstance(event, QueryWidgetsResult):
                async for fc_event in self._send_client_widget_function_call(
                    function="get_widget_data",
                    result=event,
                    copilot_function_call_arguments={
                        "summary": summary,
                        "widget_queries": widget_queries,
                    },
                    queries_length=len(widget_queries),
                    extra_state=extra_state,
                ):
                    yield fc_event

    async def llm_update_widget_in_dashboard(
        self,
        widget_queries: list[WidgetQueryRequest],
        summary: str = "Querying widgets",
        extra_state: dict[str, Any] | None = None,
    ) -> AsyncGenerator[FunctionCallResponse | StatusUpdateSSE, None]:
        """Query widget data sources.

        Each widget query must specify the UUID of the widget, and a query
        of the data that the user has asked for.

        Be very specific on which data you want to fetch from the widget.

        Parameters
        ----------
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        widget_queries : list[WidgetQueryRequest]
            List of widget queries to execute. Each query must specify the
            widget UUID, the query to execute, and whether to use the current
            input arguments or not.

        Example
        --------
        {
            "summary": "Querying widgets for AAPL data",
            "widget_queries": [
                {
                    "widget_uuid": "08d47a2f-bd35-4f53-a0e6-a45b4c7252f0", // make sure you use widget uuid, NOT the widget_id
                    "query": "Summarize the currently-selected news for AAPL.",
                    "does_query_match_input_args": "Yes, the query matches the current input arguments since we want to summarize the currently-selected news.",
                    "use_current_inputs": True,
                },
                {
                    "widget_uuid": "1fc69f4b-1fbc-472a-852a-cc96adff0701",
                    "query": "Show me the balance sheet for AAPL.",
                    "does_query_match_input_args": "No, we need to set AAPL as the input argument to get its balance sheet data.",
                    "use_current_inputs": False
                },
                {
                    "widget_uuid": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                    "query": "Show me the balance sheet for AMZN.",
                    "does_query_match_input_args": "No, we need to set AMZN as the input argument to get its balance sheet data.",
                    "use_current_inputs": False
                },
            ]
        }


        IMPORTANT: This function can ONLY update widgets that have configurable
        input parameters (e.g. ticker, date range, region). It CANNOT update
        HTML widgets (widget IDs starting with "html-") because they have no
        input parameters — calling this on an HTML widget will do nothing.
        To change an HTML widget's content, use `llm_create_html_artifact`
        to recreate it with the new content instead.
        """  # noqa: E501
        async for event in self._search_primary_and_secondary_widgets(
            widget_queries=widget_queries,
            summary=summary,
            extra_state=extra_state,
        ):
            if isinstance(event, StatusUpdateSSE):
                yield event
            elif isinstance(event, QueryWidgetsResult):
                async for fc_event in self._send_client_widget_function_call(
                    function="update_widget_in_dashboard",
                    result=event,
                    copilot_function_call_arguments={
                        "summary": summary,
                        "widget_queries": widget_queries,
                    },
                    queries_length=len(widget_queries),
                    extra_state=extra_state,
                ):
                    yield fc_event

    async def _search_extra_widgets(
        self,
        search_queries: list[DataSourceSearchQuery],
        summary: str = "Searching widgets",
        extra_state: dict[str, Any] | None = None,
    ) -> AsyncGenerator[StatusUpdateSSE | QueryExtraWidgetsResult, None]:
        # Convert to pydantic model
        if search_queries and not isinstance(search_queries[0], DataSourceSearchQuery):
            search_queries = [
                DataSourceSearchQuery(**query) for query in search_queries
            ]

        # Having extra state implies we are continuing with the function call
        # so we don't need to send this again.
        if extra_state:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message="Continuing from previous state",  # noqa: E501
                )
            )
        else:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message=summary,
                    details=[
                        f"Searching all widgets: {', '.join(set([query.description for query in search_queries]))}"  # noqa: E501
                    ],  # noqa: E501
                )
            )
        if extra_state and (
            failed_param_queries := self._count_failed_param_option_queries(extra_state)
        ):
            total_param_queries = len(extra_state["intermediate_tool_call_result"].data)
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="WARNING",
                    message=(
                        f"Some widget parameter lookups failed "
                        f"({failed_param_queries} of {total_param_queries}); "
                        "continuing with the rest."
                    ),
                )
            )
        try:
            query_extra_widget_requests = self._prepare_query_extra_widget_requests(
                search_queries=search_queries,
                extra_state=extra_state,
            )
            result = await self._copilot_data_service.query_extra_widgets(
                query_extra_widget_requests=query_extra_widget_requests,
            )
            yield result
        except FunctionCallError as err:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Widget query failed",
                    details=[],
                )
            )
            raise err

    async def _send_client_widget_function_call(
        self,
        function: Literal[
            "get_widget_data",
            "get_extra_widget_data",
            "add_widget_to_dashboard",
            "update_widget_in_dashboard",
        ],
        result: QueryExtraWidgetsResult | QueryWidgetsResult,
        copilot_function_call_arguments: dict[str, Any],
        queries_length: int,
        extra_state: dict[str, Any] | None = None,
        search_queries: list[DataSourceSearchQuery] | None = None,
    ) -> AsyncGenerator[FunctionCallResponse | StatusUpdateSSE, None]:
        if result.must_fetch_param_options:  # type: ignore
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message="Fetching parameter options",
                )
            )

            # Encode extra state so we understand which param options belong to
            # which query.
            options_params_lookup_table = []
            completed_data_source_requests_lookup_table = []
            param_options_request_payloads = []
            for request, widget_query_index in zip(
                result.content,
                range(queries_length),
                strict=True,
            ):
                if isinstance(request, WidgetDataSourceParamOptionsRequest):
                    for param_options_request_payload in request.payload:
                        param_options_request_payloads.append(
                            param_options_request_payload
                        )
                        options_params_lookup_table.append(
                            {
                                "widget_query_index": widget_query_index,
                                "partial_input_args": request.partial_input_args.input_args,  # noqa: E501
                            }
                        )
                elif isinstance(request, WidgetDataSourceRequest):
                    completed_data_source_requests_lookup_table.append(
                        {
                            "widget_query_index": widget_query_index,
                            "data_source_request": request.payload,
                        }
                    )

            extra_state = {
                "continue_from": function,
                "param_options_widget_query_mapping": options_params_lookup_table,
                "completed_data_source_request_query_mapping": completed_data_source_requests_lookup_table,  # noqa: E501
                "copilot_function_call_arguments": copilot_function_call_arguments,
            }

            yield FunctionCallResponse(
                function="get_params_options",
                input_arguments={
                    "param_options_queries": param_options_request_payloads
                },
                extra_state=extra_state,
            )
            return
        # Otherwise, we send back the function call to fetch the data
        # from the widgets with the generated input args.
        else:
            # TODO: Should be doing this after receiving any result -- not just here.
            result_errors: list[DataSourceSearchFailure] | None = getattr(
                result, "errors", None
            )
            not_found_data_sources = [
                error.data_source_query for error in (result_errors or [])
            ]
            extra_state = {
                "copilot_function_call_arguments": copilot_function_call_arguments,
            }
            if not_found_data_sources:
                extra_state["not_found_data_sources"] = not_found_data_sources
            for error in result_errors or []:
                error = cast(DataSourceSearchFailure, error)
                yield StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType="WARNING",
                        details=[
                            {
                                "Searched for description": error.data_source_description,  # noqa: E501
                                "Searched with query": error.data_source_query,
                            }
                        ],
                        message=error.reason,
                    )
                )
            data_source_request_payloads: list[DataSourceRequestPayload] = []

            # Determine the message prefix based on function type
            message_prefix = ""
            if function == "get_extra_widget_data":
                message_prefix = "Requesting extra widget data"
            elif function == "add_widget_to_dashboard":
                message_prefix = "Adding widget to dashboard"
            elif function == "get_widget_data":
                message_prefix = "Requesting widget data"
            elif function == "update_widget_in_dashboard":
                message_prefix = "Updating widget in dashboard"

            # Yield a separate status update for each widget
            for idx, widget_data_source_request in enumerate(result.content):
                payload = cast(
                    DataSourceRequestPayload, widget_data_source_request.payload
                )
                # Inject inner_tab from search query into input_args
                if (
                    search_queries
                    and idx < len(search_queries)
                    and search_queries[idx].inner_tab
                ):
                    payload.input_args["inner_tab"] = search_queries[idx].inner_tab
                data_source_request_payloads.append(payload)

                is_python_widget = (
                    widget_data_source_request.widget.widget_id == "run_code"
                )

                if is_python_widget:
                    filtered_input_args = {
                        k: v for k, v in payload.input_args.items() if k != "prompt"
                    }
                else:
                    filtered_input_args = payload.input_args

                detail = {
                    "Origin": widget_data_source_request.widget.origin,
                    "Widget Id": widget_data_source_request.widget.widget_id,  # noqa: E501
                    **filtered_input_args,
                }

                if message_prefix:
                    yield StatusUpdateSSE(
                        data=StatusUpdateSSEData(
                            eventType="INFO",
                            message=message_prefix,
                            details=[detail],
                        )
                    )

                # Send python code as a separate artifact SSE
                if is_python_widget and "prompt" in payload.input_args:
                    python_code = payload.input_args["prompt"]
                    yield StatusUpdateSSE(
                        data=StatusUpdateSSEData(
                            eventType="INFO",
                            message="Python code",
                            artifacts=[
                                ClientArtifact(
                                    name=f"python_code_{widget_data_source_request.widget.widget_id}",  # noqa: E501
                                    description="Python code for execution",
                                    type="snowflake_python",
                                    content=f"```python\n{python_code}\n```",
                                    query_data_source={
                                        "origin": payload.origin,
                                        "id": payload.id,
                                        "widget_uuid": payload.widget_uuid,
                                    },
                                )
                            ],
                        )
                    )

            if not data_source_request_payloads:
                self._logging_service.warning(
                    "Client widget function call payload is empty "
                    "for function=%s with error_count=%s and not_found_data_sources=%s",
                    function,
                    len(result_errors or []),
                    not_found_data_sources,
                )
            yield FunctionCallResponse(
                function=function,
                input_arguments={
                    "data_sources": data_source_request_payloads,
                },
                extra_state=extra_state,
            )
            return

    async def llm_generate_widget_in_dashboard(
        self,
        widget_type: Literal["chart", "table", "note", "html"],
        data: list[dict] | str | None = None,
        name: str = "",
        description: str = "",
        chart_params: dict | None = None,
        artifact_id: str | None = None,
        inner_tab: str | None = None,
        summary: str = "Adding widget to dashboard",
        extra_state: dict[str, Any] | None = None,
    ) -> AsyncGenerator[FunctionCallResponse | StatusUpdateSSE, None]:
        """Create a chart, table, or note widget directly on the dashboard.

        Use this tool when you want to create a fully functional widget
        on the user's dashboard using data you have generated or processed.

        Parameters
        ----------
        widget_type : Literal["chart", "table", "note", "html"]
            The type of widget to create:
            - "chart": For visualizations (bar, line, scatter, pie, donut)
            - "table": For tabular data with multiple columns
            - "note": For markdown text notes, insights, or summaries
            - "html": For rich HTML content such as branded reports, investment
              memos, or any multi-section styled document. Pass the complete
              HTML string as `data`.
        data : list[dict] | str | None
            For charts and tables: A list of dictionaries where each represents a row.
            For notes: A markdown string containing the note content.
            Optional when artifact_id is provided (the full data will be resolved
            automatically from the artifact).
        name : str
            A descriptive name for the widget that will appear in the widget header.
        description : str, optional
            A brief description of what the widget displays.
        chart_params : dict, optional
            Required for chart widgets. Contains chart configuration:
            - chartType: "bar" | "line" | "scatter" | "pie" | "donut"
            - xKey: The key in data to use for the x-axis (e.g., "quarter")
            - yKey: List of keys to use for y-axis values (e.g., ["revenue", "profit"])
        artifact_id : str, optional
            The name of an existing artifact (e.g., "chart_artifact_d2251") whose
            full data should be used for the widget. When provided, the complete
            dataset is resolved automatically — you do NOT need to pass inline data.
            Prefer this over inline data for large datasets from SQL queries.
        inner_tab : str, optional
            The tab name to place the widget in when a navigation bar exists on
            the dashboard. If not provided, the widget will be placed in the
            currently active tab.
        summary : str
            Very short summary of the action (3 to 5 words, starts with gerund).

        Examples
        --------
        Creating a chart from an existing artifact (PREFERRED for queried data):
        {
            "widget_type": "chart",
            "artifact_id": "chart_artifact_d2251",
            "name": "AAPL Daily Close Price",
            "chart_params": {
                "chartType": "line", "xKey": "date", "yKey": ["close"]
            },
            "summary": "Creating AAPL price chart"
        }

        Creating a bar chart with inline data:
        {
            "widget_type": "chart",
            "name": "Quarterly Revenue",
            "data": [
                {"quarter": "Q1", "revenue": 100000},
                {"quarter": "Q2", "revenue": 150000}
            ],
            "chart_params": {
                "chartType": "bar", "xKey": "quarter", "yKey": ["revenue"]
            },
            "summary": "Creating quarterly revenue chart"
        }

        Creating a table from an existing artifact:
        {
            "widget_type": "table",
            "artifact_id": "chart_artifact_a1b2c",
            "name": "Financial Summary",
            "summary": "Creating financial summary table"
        }

        Creating a table with inline data:
        {
            "widget_type": "table",
            "name": "Team Members",
            "data": [
                {"name": "Alice", "role": "Engineer", "years": 5},
                {"name": "Bob", "role": "Designer", "years": 3}
            ],
            "summary": "Creating team members table"
        }

        Creating a note (for insights, summaries, or explanatory text):
        {
            "widget_type": "note",
            "name": "2026 Macro Outlook",
            "data": "## Key Insights\\n\\n- Recovery is underway\\n"
                    "- PCE and RPI returning to positive growth\\n"
                    "- Cautiously optimistic outlook for 2026",
            "summary": "Adding macro outlook note"
        }
        """
        if data is None and artifact_id is None:
            self._logging_service.warning(
                "Neither data nor artifact_id provided for widget_type=%s",
                widget_type,
            )
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Cannot create widget: either data "
                    "or artifact_id must be provided.",
                )
            )
            return

        if (
            widget_type in ("chart", "table")
            and data is not None
            and not isinstance(data, list)
        ):
            self._logging_service.warning(
                "Invalid data type for widget_type=%s: expected list[dict], got %s",
                widget_type,
                type(data).__name__,
            )
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message=f"Cannot create {widget_type} widget:"
                    " data must be a list of records,"
                    " not text.",
                )
            )
            return

        self._logging_service.info(
            "Adding generative widget: type=%s, name=%s, artifact_id=%s",
            widget_type,
            name,
            artifact_id,
        )
        # The frontend only receives concrete widget payloads. Any artifact-only
        # widget request must be resolved inside Ada before we cross the client
        # boundary.
        if data is None and artifact_id is not None:
            self._logging_service.warning(
                "Unresolved artifact-backed widget request for widget_type=%s",
                widget_type,
            )
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Cannot create widget from artifact directly.",
                    details=[
                        "Artifact-backed widget requests must be resolved on "
                        "the Ada side before sending them to the client."
                    ],
                )
            )
            return

        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
            )
        )
        yield FunctionCallResponse(
            function="add_generative_widget",
            input_arguments={
                "widget_type": widget_type,
                "data": data,
                "name": name,
                "description": description,
                "chart_params": chart_params,
                "inner_tab": inner_tab,
            },
            extra_state={
                "copilot_function_call_arguments": {
                    "widget_type": widget_type,
                    "data": data,
                    "name": name,
                    "description": description,
                    "chart_params": chart_params,
                    "summary": summary,
                }
            },
        )

    async def llm_execute_agent_tool(
        self,
        server_id: str,
        summary: str,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        extra_state: dict[str, Any] | None = None,
        available_tools: Sequence[AgentTool] | None = None,
    ) -> AsyncGenerator[FunctionCallResponse | StatusUpdateSSE, None]:
        """Execute an agent tool via MCP.

        Parameters
        ----------
        server_id : str
            The ID of the server where the tool is located
        summary : str
            A short, human-readable description of what you are about to do,
            displayed to the user while the tool executes (e.g.
            "Fetching AAPL stock prices for the last year").
            Starts with a gerund. No period at the end.
            This is DISPLAY-ONLY — values here are NOT forwarded to the tool.
        tool_name : str
            The name of the tool to execute
            (format: "serverName_toolname", e.g., "context7_get-library-docs")
        tool_args : dict[str, Any]
            A dictionary mapping parameter names to their values.
            Every required parameter listed in the tool's Parameters section
            MUST appear as a key here with a real value — for example,
            if the tool requires `table`, you must pass
            `"tool_args": {"table": "my_table_name"}`.
            Values placed in `summary` are NOT forwarded to the tool.
            For tools with no parameters, pass an empty dictionary {}.
        """
        # Validate and normalize tool_args
        # Handle cases where LLM sends a string instead of dict for parameter-less tools
        if tool_args is None:
            tool_args = {}
        elif isinstance(tool_args, str):
            # If it's an empty or descriptive string, convert to empty dict
            # Otherwise, log a warning and convert to empty dict
            self._logging_service.warning(
                "Received string for tool_args instead of dict: %s. "
                "Converting to empty dict.",
                tool_args,
            )
            tool_args = {}
        elif not isinstance(tool_args, dict):
            # Handle any other unexpected types
            self._logging_service.warning(
                "Received unexpected type for tool_args: %s. Converting to empty dict.",
                type(tool_args),
            )
            tool_args = {}

        tool_call_diagnostics = self._build_agent_tool_diagnostics(
            server_id=server_id,
            tool_name=tool_name,
            tool_args=tool_args,
            available_tools=available_tools,
        )
        emitted_tool_call_diagnostics = {
            key: value
            for key, value in tool_call_diagnostics.items()
            if key not in {"tool_args_hash", "input_schema_hash"}
        }
        self._logging_service.debug(
            "tool_call_emitted",
            extra={
                "event": "tool_call_emitted",
                "summary": summary,
                **emitted_tool_call_diagnostics,
            },
        )
        if tool_call_diagnostics["missing_required_args"]:
            self._logging_service.warning(
                "tool_call_missing_required_args",
                extra={
                    "event": "tool_call_missing_required_args",
                    "summary": summary,
                    **emitted_tool_call_diagnostics,
                },
            )
            missing = tool_call_diagnostics["missing_required_args"]
            example_args = ", ".join(f'"{arg}": "<value>"' for arg in missing)
            raise FunctionCallError(
                f"MCP tool `{tool_name}` requires arguments "
                f"{missing} but they were not provided in "
                f"`tool_args`. You MUST pass ALL required "
                f"parameters inside the `tool_args` dict. "
                f"The `summary` field is display-only and its "
                f"values are NOT forwarded to the tool. "
                f"Retry with: "
                f'"tool_args": {{{example_args}}}.'
            )

        # Use the tool name as the main message
        # Only show parameters in the details
        details = []

        # Add parameters as a dictionary in the same format as widget data
        if tool_args:
            details.append(tool_args)

        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=f"MCP tool: {tool_name}",
                details=details if details else None,
            )
        )

        # Send the function call to the frontend for execution
        yield FunctionCallResponse(
            function="execute_agent_tool",
            input_arguments={
                "server_id": server_id,
                "tool_name": tool_name,
                "parameters": tool_args,
            },
            extra_state={
                **(extra_state or {}),
                "copilot_function_call_arguments": {
                    "server_id": server_id,
                    "summary": summary,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                },
                "mcp_tool_diagnostics": tool_call_diagnostics,
            },
        )

    async def llm_manage_navigation_bar(
        self,
        operation: Literal["create", "add_tabs", "remove_tabs", "rename_tabs"],
        tabs: list[NavigationBarTab] | None = None,
        rename_map: dict[str, str] | None = None,
        summary: str = "Managing navigation bar",
    ) -> AsyncGenerator[FunctionCallResponse | StatusUpdateSSE, None]:
        """Manage the navigation bar on the current dashboard.

        The navigation bar organizes dashboard widgets into named tabs.
        Use this tool to create a navigation bar, add or remove tabs,
        or rename existing tabs.

        Parameters
        ----------
        operation : str
            The operation to perform on the navigation bar:
            - "create": Create a new navigation bar with initial tabs.
              Requires 'tabs'. The dashboard must NOT already have one.
            - "add_tabs": Add new tabs to an existing navigation bar.
              Requires 'tabs'.
            - "remove_tabs": Remove tabs by name from the navigation bar.
              Requires 'tabs'. Widgets in removed tabs will be deleted.
            - "rename_tabs": Rename existing tabs. Requires 'rename_map'.
        tabs : list[NavigationBarTab], optional
            Tabs for create/add_tabs/remove_tabs operations.
            Each tab has a 'name' field (display name).
        rename_map : dict[str, str], optional
            For 'rename_tabs' only. Maps current tab name to new tab name.
        summary : str
            Very short summary of the action (gerund, 3-5 words, no period).

        Examples
        --------
        Create a navigation bar with tabs:
        {
            "operation": "create",
            "tabs": [{"name": "Overview"}, {"name": "Fundamentals"}, {"name": "Technicals"}],
            "summary": "Creating navigation bar"
        }

        Add a new tab:
        {
            "operation": "add_tabs",
            "tabs": [{"name": "News"}],
            "summary": "Adding News tab"
        }

        Remove a tab:
        {
            "operation": "remove_tabs",
            "tabs": [{"name": "News"}],
            "summary": "Removing News tab"
        }

        Rename a tab:
        {
            "operation": "rename_tabs",
            "rename_map": {"Overview": "Summary"},
            "summary": "Renaming Overview tab"
        }
        """  # noqa: E501
        self._logging_service.info(
            "manage_navigation_bar: operation=%s, tabs=%s, rename_map=%s",
            operation,
            tabs,
            rename_map,
        )

        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
            )
        )

        # Normalize tabs to dicts
        tabs_data = None
        if tabs:
            tabs_data = [
                t.model_dump() if isinstance(t, NavigationBarTab) else t for t in tabs
            ]

        input_arguments: dict[str, Any] = {"operation": operation}
        if tabs_data:
            input_arguments["tabs"] = tabs_data
        if rename_map:
            input_arguments["rename_map"] = rename_map

        yield FunctionCallResponse(
            function="manage_navigation_bar",
            input_arguments=input_arguments,
            extra_state={
                "copilot_function_call_arguments": {
                    "summary": summary,
                    "operation": operation,
                    "tabs": tabs_data,
                    "rename_map": rename_map,
                },
            },
        )

    async def llm_get_skill_content(
        self,
        slug: str,
        reason: str | None = None,
        summary: str = "Loading skill instructions",
        extra_state: dict[str, Any] | None = None,
    ) -> AsyncGenerator[FunctionCallResponse | StatusUpdateSSE, None]:
        """Request full skill instructions from the user's skill library.

        Use when you need detailed instructions from a skill to handle
        a task. Only request skills relevant to the current query.

        Parameters
        ----------
        slug : str
            The slug identifier of the skill to retrieve (e.g., "financial-analysis").
            This should match one of the slugs from the skills catalog.
        reason : str, optional
            A brief explanation of why you need this skill's instructions.
            This helps with logging and debugging.
        summary : str
            Very short summary of the action (gerund, 3-5 words, no period).

        Examples
        --------
        {
            "slug": "financial-analysis",
            "reason": "User asked for earnings analysis, need specialized instructions",
            "summary": "Loading financial analysis skill"
        }
        """
        self._logging_service.info(
            "get_skill_content: slug=%s, reason=%s",
            slug,
            reason,
        )

        yield FunctionCallResponse(
            function="get_skill_content",
            input_arguments={
                "slug": slug,
                "reason": reason,
            },
            extra_state={
                "copilot_function_call_arguments": {
                    "summary": summary,
                    "slug": slug,
                    "reason": reason,
                }
            },
        )

    async def llm_save_skill(
        self,
        name: str | None = None,
        instructions: str | None = None,
        summary: str = "Saving skill",
        extra_state: dict[str, Any] | None = None,
    ) -> AsyncGenerator[FunctionCallResponse | StatusUpdateSSE, None]:
        """Save the current conversation's workflow as a reusable skill.

        Use when the user asks to save this conversation, workflow or
        process as a skill (e.g. "save this as a skill"). The client
        parses the conversation, generates the skill and stores it in
        the user's skill library. Do NOT write the skill content
        yourself and do NOT call this more than once per request.

        Parameters
        ----------
        name : str, optional
            A name for the skill, only if the user suggested one.
        instructions : str, optional
            Specific user guidance about what the skill should capture
            or emphasize, only if the user provided any.
        summary : str
            Very short summary of the action (gerund, 3-5 words, no period).

        Examples
        --------
        {
            "name": "earnings analysis",
            "instructions": "focus on the comparison table format",
            "summary": "Saving skill"
        }
        """
        self._logging_service.info(
            "save_skill: name=%s, instructions=%s",
            name,
            instructions,
        )

        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
            )
        )

        yield FunctionCallResponse(
            function="save_skill",
            input_arguments={
                "name": name,
                "instructions": instructions,
            },
            extra_state={
                "copilot_function_call_arguments": {
                    "summary": summary,
                    "name": name,
                    "instructions": instructions,
                }
            },
        )

    async def _search_primary_and_secondary_widgets(
        self,
        widget_queries: list[WidgetQueryRequest],
        summary: str = "Querying widgets",
        extra_state: dict[str, Any] | None = None,
    ) -> AsyncGenerator[StatusUpdateSSE | QueryWidgetsResult, None]:
        # Quick and dirty parse of inputs into pydantic models, which the code
        # relies on (We could do the below in a generic function in `utils.py`,
        # but probably not necessary at the moment. (We also flatten objects)

        if isinstance(widget_queries[0], dict):
            widget_queries = [
                WidgetQueryRequest(**widget_query) for widget_query in widget_queries
            ]
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message="Continuing from previous state" if extra_state else summary,
                details=[
                    {
                        "Queries": "\n".join(
                            [widget_query.query for widget_query in widget_queries]
                        ),
                    }
                ],
            )
        )
        if extra_state and (
            failed_param_queries := self._count_failed_param_option_queries(extra_state)
        ):
            total_param_queries = len(extra_state["intermediate_tool_call_result"].data)
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="WARNING",
                    message=(
                        f"Some widget parameter lookups failed "
                        f"({failed_param_queries} of {total_param_queries}); "
                        "continuing with the rest."
                    ),
                )
            )
        try:
            query_widget_requests = self._prepare_query_widget_requests(
                widget_queries=widget_queries,
                extra_state=extra_state,
            )
            result = await self._copilot_data_service.query_widgets(
                query_widget_requests=query_widget_requests,
            )
            yield result
        except FunctionCallError as err:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Widget query failed",
                    details=[],
                )
            )
            raise err

    def _get_completed_query_widget_requests_from_extra_state_for_index(
        self,
        widget_queries: list[WidgetQueryRequest],
        widget_query_index: int,
        extra_state: dict[str, Any],
    ) -> QueryWidgetRequest | None:
        if completed_data_source_request_query_mapping := extra_state.get(
            "completed_data_source_request_query_mapping"
        ):
            for mapping in completed_data_source_request_query_mapping:
                if mapping["widget_query_index"] == widget_query_index:
                    return QueryWidgetRequest(
                        widget_uuid=widget_queries[
                            mapping["widget_query_index"]
                        ].widget_uuid,
                        widget_query=widget_queries[
                            mapping["widget_query_index"]
                        ].query,
                        use_current_inputs=mapping["data_source_request"].get(
                            "use_current_inputs", False
                        ),
                        partial_input_args=mapping["data_source_request"].get(
                            "input_args", {}
                        ),
                    )
        return None

    def _get_completed_query_extra_widget_requests_from_extra_state_for_index(
        self,
        search_queries: list[DataSourceSearchQuery],
        widget_query_index: int,
        extra_state: dict[str, Any],
    ) -> QueryExtraWidgetsRequest | None:
        if completed_data_source_request_query_mapping := extra_state.get(
            "completed_data_source_request_query_mapping"
        ):
            for mapping in completed_data_source_request_query_mapping:
                if mapping["widget_query_index"] == widget_query_index:
                    return QueryExtraWidgetsRequest(
                        data_source_description=search_queries[
                            mapping["widget_query_index"]
                        ].description,
                        widget_query=search_queries[
                            mapping["widget_query_index"]
                        ].query,
                        user_context=search_queries[
                            mapping["widget_query_index"]
                        ].user_context,
                        partial_input_args=mapping["data_source_request"]["input_args"],
                    )
        return None

    def _parse_param_options_results(self, data_items: list[Any]) -> list[Any]:
        """Parse param options results positionally, tolerating failures.

        Each element of the intermediate tool call's ``data`` corresponds to one
        param options sub-query. A sub-query that failed on the client comes back
        as a ``ClientFunctionCallError`` (no ``.items``); malformed successes are
        also possible. We keep the list index-aligned with the param query
        mapping by emitting ``None`` for anything we can't parse, so callers can
        best-effort skip the failed entries instead of crashing.
        """
        results: list[Any] = []
        for index, data_item in enumerate(data_items):
            if isinstance(data_item, ClientFunctionCallError):
                self._logging_service.warning(
                    "Param options sub-query %d failed on the client "
                    "(error_type=%s): %s",
                    index,
                    data_item.error_type,
                    data_item.content,
                )
                results.append(None)
                continue
            try:
                results.append(
                    json.loads(data_item.items[0].content)["param_options"][0]
                )
            except (AttributeError, KeyError, IndexError, TypeError, ValueError):
                self._logging_service.warning(
                    "Could not parse param options for sub-query %d (type=%s); "
                    "skipping it.",
                    index,
                    type(data_item).__name__,
                )
                results.append(None)
        return results

    def _count_failed_param_option_queries(
        self, extra_state: dict[str, Any] | None
    ) -> int:
        """Count how many param options sub-queries failed on the client.

        Used to surface a user-facing status message when we best-effort skip
        failed param lookups during a widget update.
        """
        if not extra_state:
            return 0
        intermediate_tool_call_result = extra_state.get("intermediate_tool_call_result")
        if (
            intermediate_tool_call_result is None
            or intermediate_tool_call_result.function != "get_params_options"
        ):
            return 0
        return sum(
            1
            for data_item in intermediate_tool_call_result.data
            if isinstance(data_item, ClientFunctionCallError)
        )

    def _get_partial_query_widget_requests_with_options_from_extra_state_for_index(
        self,
        widget_queries: list[WidgetQueryRequest],
        widget_query_index: int,
        extra_state: dict[str, Any],
    ) -> QueryWidgetRequest | None:
        if intermediate_tool_call_result := extra_state.get(
            "intermediate_tool_call_result"
        ):
            if intermediate_tool_call_result.function == "get_params_options":
                param_queries = intermediate_tool_call_result.input_arguments.get(
                    "param_options_queries"
                )
                # Build results positionally so indices stay aligned with the
                # param query mapping. A param options sub-query can fail on the
                # client and come back as a ClientFunctionCallError (which has no
                # `.items`), so we mark those (and any malformed entries) as None
                # and best-effort skip them below instead of crashing the stream.
                param_queries_results = self._parse_param_options_results(
                    intermediate_tool_call_result.data
                )

                # Get the relevant param queries and results for the
                # current widget query. We do this by looking at the
                # extra_state, which contains the mapping between the
                # widget query index and the param queries index.
                relevant_param_queries = []
                relevant_param_queries_results = []
                partial_input_args: dict[str, Any] = {}
                for param_query_index, mapping in enumerate(
                    extra_state.get("param_options_widget_query_mapping", [])
                ):
                    if mapping["widget_query_index"] == widget_query_index:
                        # Bounds check to prevent IndexError
                        if param_query_index >= len(
                            param_queries
                        ) or param_query_index >= len(param_queries_results):
                            self._logging_service.error(
                                (
                                    "Index mismatch in param queries: index=%d, "
                                    "queries_len=%d, results_len=%d"
                                ),
                                param_query_index,
                                len(param_queries),
                                len(param_queries_results),
                            )
                            continue

                        # This param options sub-query failed on the client;
                        # best-effort skip it and keep the successful ones.
                        if param_queries_results[param_query_index] is None:
                            continue

                        # Partial input args should be the same for all
                        # mappings with the same widget query index
                        partial_input_args = mapping.get("partial_input_args", {})
                        relevant_param_queries.append(param_queries[param_query_index])
                        relevant_param_queries_results.append(
                            param_queries_results[param_query_index]
                        )

                if relevant_param_queries and relevant_param_queries_results:
                    return QueryWidgetRequest(
                        widget_uuid=widget_queries[widget_query_index].widget_uuid,
                        widget_query=widget_queries[widget_query_index].query,
                        partial_input_args=partial_input_args,
                        extra_param_options=[
                            WidgetParamOptions(
                                widget_origin=param_query["origin"],
                                widget_id=param_query["id"],
                                param_name=param_query_result["param"],
                                options=param_query_result["options"],
                            )
                            for param_query, param_query_result in zip(
                                relevant_param_queries,
                                relevant_param_queries_results,
                                strict=True,
                            )
                        ],
                    )
        return None

    def _get_partial_query_extra_widget_requests_with_options_from_extra_state_for_index(  # noqa: E501
        self,
        search_queries: list[DataSourceSearchQuery],
        widget_query_index: int,
        extra_state: dict[str, Any],
    ) -> QueryExtraWidgetsRequest | None:
        if intermediate_tool_call_result := extra_state.get(
            "intermediate_tool_call_result"
        ):
            if intermediate_tool_call_result.function == "get_params_options":
                param_queries = intermediate_tool_call_result.input_arguments.get(
                    "param_options_queries"
                )
                # Build results positionally so indices stay aligned with the
                # param query mapping. A param options sub-query can fail on the
                # client and come back as a ClientFunctionCallError (which has no
                # `.items`), so we mark those (and any malformed entries) as None
                # and best-effort skip them below instead of crashing the stream.
                param_queries_results = self._parse_param_options_results(
                    intermediate_tool_call_result.data
                )

                # Get the relevant param queries and results for the
                # current widget query. We do this by looking at the
                # extra_state, which contains the mapping between the
                # widget query index and the param queries index.
                relevant_param_queries = []
                relevant_param_queries_results = []
                partial_input_args: dict[str, Any] = {}
                for param_query_index, mapping in enumerate(
                    extra_state.get("param_options_widget_query_mapping", [])
                ):
                    if mapping["widget_query_index"] == widget_query_index:
                        # Bounds check to prevent IndexError
                        if param_query_index >= len(
                            param_queries
                        ) or param_query_index >= len(param_queries_results):
                            self._logging_service.error(
                                (
                                    "Index mismatch in param queries: index=%d, "
                                    "queries_len=%d, results_len=%d"
                                ),
                                param_query_index,
                                len(param_queries),
                                len(param_queries_results),
                            )
                            continue

                        # This param options sub-query failed on the client;
                        # best-effort skip it and keep the successful ones.
                        if param_queries_results[param_query_index] is None:
                            continue

                        # Partial input args should be the same for all
                        # mappings with the same widget query index
                        partial_input_args = mapping.get("partial_input_args", {})
                        relevant_param_queries.append(param_queries[param_query_index])
                        relevant_param_queries_results.append(
                            param_queries_results[param_query_index]
                        )

                if relevant_param_queries and relevant_param_queries_results:
                    return QueryExtraWidgetsRequest(
                        data_source_description=search_queries[
                            widget_query_index
                        ].description,
                        widget_query=search_queries[widget_query_index].query,
                        user_context=search_queries[widget_query_index].user_context,
                        partial_input_args=partial_input_args,
                        extra_param_options=[
                            WidgetParamOptions(
                                widget_origin=param_query["origin"],
                                widget_id=param_query["id"],
                                param_name=param_query_result["param"],
                                options=param_query_result["options"],
                            )
                            for param_query, param_query_result in zip(
                                relevant_param_queries,
                                relevant_param_queries_results,
                                strict=True,
                            )
                        ],
                    )
        return None

    def _prepare_query_widget_requests(
        self,
        widget_queries: list[WidgetQueryRequest],
        extra_state: dict[str, Any] | None = None,
    ) -> list[QueryWidgetRequest]:
        self._logging_service.info(
            "Preparing query widget requests for widget_queries=%s, extra_state=%s",
            widget_queries,
            extra_state,
        )

        query_widget_requests: list[QueryWidgetRequest] = []

        # We loop through each widget query, since we need to preserve
        # the order of the widget queries.
        for widget_query_index, widget_query in enumerate(widget_queries):
            # If we have some extra state we need to handle...
            if extra_state:
                # First check if we have any already-completed requests in the
                # extra state for the current widget query
                if (
                    completed_widget_query_request
                    := self._get_completed_query_widget_requests_from_extra_state_for_index(  # noqa: E501
                        widget_queries=widget_queries,
                        widget_query_index=widget_query_index,
                        extra_state=extra_state,
                    )
                ):
                    query_widget_requests.append(completed_widget_query_request)
                    continue

                # Otherwise, we need to fetch the param options from the
                # intermediate tool call
                if (
                    partial_widget_query_request_with_options
                    := self._get_partial_query_widget_requests_with_options_from_extra_state_for_index(  # noqa: E501
                        widget_queries=widget_queries,
                        widget_query_index=widget_query_index,
                        extra_state=extra_state,
                    )
                ):
                    query_widget_requests.append(
                        partial_widget_query_request_with_options
                    )
                    continue

            # If there's no extra state, we just append a normal request.
            else:
                query_widget_requests.append(
                    QueryWidgetRequest(
                        widget_uuid=widget_query.widget_uuid,
                        widget_query=widget_query.query,
                        use_current_inputs=widget_query.use_current_inputs,
                    )
                )
        return query_widget_requests

    def _prepare_query_extra_widget_requests(
        self,
        search_queries: list[DataSourceSearchQuery],
        extra_state: dict[str, Any] | None = None,
    ) -> list[QueryExtraWidgetsRequest]:
        self._logging_service.info(
            "Preparing query extra widget requests for search_queries=%s, extra_state=%s",  # noqa: E501
            search_queries,
            extra_state,
        )
        query_extra_widget_requests: list[QueryExtraWidgetsRequest] = []
        for widget_query_index, search_query in enumerate(search_queries):
            if extra_state:
                if (
                    completed_extra_widget_query_request
                    := self._get_completed_query_extra_widget_requests_from_extra_state_for_index(  # noqa: E501
                        search_queries=search_queries,
                        widget_query_index=widget_query_index,
                        extra_state=extra_state,
                    )
                ):
                    query_extra_widget_requests.append(
                        completed_extra_widget_query_request
                    )
                    continue

                if (
                    partial_extra_widget_query_request_with_options
                    := self._get_partial_query_extra_widget_requests_with_options_from_extra_state_for_index(  # noqa: E501
                        search_queries=search_queries,
                        widget_query_index=widget_query_index,
                        extra_state=extra_state,
                    )
                ):
                    query_extra_widget_requests.append(
                        partial_extra_widget_query_request_with_options
                    )
                    continue

            else:
                query_extra_widget_requests.append(
                    QueryExtraWidgetsRequest(
                        data_source_description=search_query.description,
                        widget_query=search_query.query,
                        user_context=search_query.user_context,
                    )
                )
        return query_extra_widget_requests

    def is_client_function_call(self, response: FunctionCall) -> bool:
        fn_name = response.function.__name__
        if fn_name in [
            self.llm_query_widgets.__name__,
            self.llm_query_extra_widgets.__name__,
            self.llm_add_widget_to_dashboard.__name__,
            self.llm_update_widget_in_dashboard.__name__,
            self.llm_assign_tasks_to_agents.__name__,
            self.llm_execute_agent_tool.__name__,
            self.llm_manage_navigation_bar.__name__,
            self.llm_generate_widget_in_dashboard.__name__,
            self.llm_get_skill_content.__name__,
            self.llm_save_skill.__name__,
        ]:
            return True
        # Also match dynamically-generated flat MCP tool functions
        if fn_name in self._dynamic_mcp_tool_names:
            return True
        return False

    async def handle_function_calls(
        self, response: FunctionCall
    ) -> AsyncGenerator[list[Message[Any]] | StatusUpdateSSE | FunctionCallSSE, None]:
        function_name = response.function.__name__
        widget_queries = response.arguments.get("widget_queries", None)
        search_queries = response.arguments.get("search_queries", None)

        self._logging_service.info(
            "Function call to the client: %s with function=%s, widget_queries=%s, search_queries=%s",  # noqa: E501
            function_name,
            response.function.__qualname__,
            widget_queries,
            search_queries,
        )

        try:
            match function_name:
                case "llm_query_widgets" | "llm_update_widget_in_dashboard":
                    # We clear extra_state when fetching widgets primary /
                    # secondary widgets, except for the copilot function call arguments.
                    extra_state = response.arguments.get("extra_state", {}) or {}
                    response.arguments["extra_state"] = extra_state.get(
                        "copilot_function_call_arguments", {}
                    )
                    async for event in response():
                        if isinstance(event, StatusUpdateSSE):
                            yield event
                        elif isinstance(event, FunctionCallResponse):
                            # Then send the function call to the client
                            yield FunctionCallSSE(
                                data=FunctionCallSSEData(**event.model_dump())
                            )
                        else:
                            raise ValueError(f"Unexpected event: {type(event)}")
                case (
                    "llm_query_extra_widgets"
                    | "llm_add_widget_to_dashboard"
                    | "llm_assign_tasks_to_agents"
                    | "llm_execute_agent_tool"
                    | "llm_manage_navigation_bar"
                    | "llm_generate_widget_in_dashboard"
                    | "llm_get_skill_content"
                    | "llm_save_skill"
                ):  # noqa: E501
                    async for event in response():
                        if isinstance(event, StatusUpdateSSE):
                            yield event
                        elif isinstance(event, FunctionCallResponse):
                            yield FunctionCallSSE(
                                data=FunctionCallSSEData(**event.model_dump())
                            )
                case _ if function_name in self._dynamic_mcp_tool_names:
                    # Dynamically-generated flat MCP tool — same handling
                    # as llm_execute_agent_tool (the flat function delegates
                    # to it internally).
                    async for event in response():
                        if isinstance(event, StatusUpdateSSE):
                            yield event
                        elif isinstance(event, FunctionCallResponse):
                            yield FunctionCallSSE(
                                data=FunctionCallSSEData(**event.model_dump())
                            )
                case _:
                    raise ValueError(f"Unexpected function call: {function_name}")

        except FunctionCallError as err:
            self._logging_service.warning(
                "FunctionCallError caught in handle_function_calls "
                "for %s: %s — returning error to LLM",
                function_name,
                err,
            )
            llm_messages = [
                AssistantMessage(response),
                FunctionResultMessage(
                    content=str(err),
                    function_call=response,
                ),
            ]
            yield llm_messages

    def get_client_tool(self, function_name: str) -> Callable:
        if function_name == "get_widget_data":
            return self.llm_query_widgets
        elif function_name == "get_extra_widget_data":
            return self.llm_query_extra_widgets
        elif function_name == "add_widget_to_dashboard":
            return self.llm_add_widget_to_dashboard
        elif function_name == "update_widget_in_dashboard":
            return self.llm_update_widget_in_dashboard
        elif function_name == "assign_tasks_to_agents":
            return self.llm_assign_tasks_to_agents
        elif function_name == "execute_agent_tool":
            return self.llm_execute_agent_tool
        elif function_name == "manage_navigation_bar":
            return self.llm_manage_navigation_bar
        elif function_name == "add_generative_widget":
            return self.llm_generate_widget_in_dashboard
        elif function_name == "get_skill_content":
            return self.llm_get_skill_content
        elif function_name == "save_skill":
            return self.llm_save_skill
        raise HTTPException(
            status_code=500,
            detail="Attempted to map tool call message to an LLM function that doesn't exist.",  # noqa: E501
        )

    def get_function_call_spec(
        self, message: LlmClientFunctionCallResultMessage
    ) -> Tuple[Callable, dict[str, Any]]:
        arguments = (message.extra_state or {}).get(
            "copilot_function_call_arguments", {}
        )

        # Map execute_agent_tool results to the corresponding flat MCP
        # tool so the replayed history matches the current function list.
        if message.function == "execute_agent_tool":
            server_id = arguments.get("server_id", "")
            tool_name = arguments.get("tool_name", "")
            key = (server_id, tool_name)
            flat_fn = self._dynamic_mcp_tool_by_origin.get(key)

            # Fallback: try matching by tool_name alone (server_id may differ
            # between the historical call and the current MCP configuration).
            if flat_fn is None and tool_name:
                for (sid, tname), fn in self._dynamic_mcp_tool_by_origin.items():
                    if tname == tool_name:
                        flat_fn = fn
                        self._logging_service.debug(
                            "Flat MCP lookup: exact key %s missed, "
                            "matched by tool_name via server_id=%s",
                            key,
                            sid,
                        )
                        break

            if flat_fn is None:
                self._logging_service.warning(
                    "Flat MCP lookup failed for key=%s. Available keys: %s",
                    key,
                    list(self._dynamic_mcp_tool_by_origin.keys()),
                )

            if flat_fn:
                flat_args = {
                    "display_summary": arguments.get("summary", "Executing MCP tool"),
                    **arguments.get("tool_args", {}),
                }
                return flat_fn, flat_args

        tool = self.get_client_tool(message.function)
        # Here we create a pydantic model to parse the arguments into
        # the types expected by the function signature, including nested
        # types and pydantic models.
        # - Example:
        # Signature:
        # def get_data(widget_queries: list[WidgetQueryRequest], extra_state: dict[str, Any] | None = None):  #  noqa: E501
        # Arguments:
        # {"widget_queries": [{"origin": "my_origin", "id": "my_id"}]}
        # Pydantic model:
        # ArgsModel(widget_queries=[WidgetQueryRequest(origin="my_origin", id="my_id")], extra_state=None)  #  noqa: E501
        # Result:
        # {"widget_queries": [WidgetQueryRequest(origin="my_origin", id="my_id")]}  #  noqa: E501
        #
        # We do this because we want to pass the actual pydantic models to the
        # function, not the raw dicts.
        sig = signature(tool)
        fields = {
            name: (
                Any if param.annotation is Parameter.empty else param.annotation,
                ... if param.default is Parameter.empty else param.default,
            )
            for name, param in sig.parameters.items()
        }
        ArgsModel = create_model(tool.__name__, **fields)  # type: ignore[call-overload]
        try:
            model = ArgsModel(**arguments)
            model_dict: dict = model.__dict__
            validated_arguments = {
                k: v for k, v in model_dict.items() if k in arguments
            }
        except ValidationError as err:
            self._logging_service.warning(
                "Failed to validate function call arguments: %s", err
            )
            return tool, arguments
        return tool, validated_arguments

    def must_continue_with_function_call(
        self, messages: list[LlmClientMessage | LlmClientFunctionCallResultMessage]
    ) -> bool:
        if isinstance(messages[-1], LlmClientFunctionCallResultMessage):
            if messages[-1].extra_state and messages[-1].extra_state.get(
                "continue_from"
            ):
                return True
        return False

    def _prepare_continued_function_call(
        self, messages: list[LlmClientFunctionCallResultMessage | LlmClientMessage]
    ) -> FunctionCall:
        if not isinstance(messages[-1], LlmClientFunctionCallResultMessage):
            raise ValueError("Last message must be a function call result")
        if not messages[-1].extra_state or not messages[-1].extra_state.get(
            "continue_from"
        ):
            raise ValueError("Last message must contain a continue_from state")
        if not isinstance(messages[-2], LlmClientMessage):
            raise ValueError("Second last message must be a llm message")
        if not isinstance(messages[-2].content, LlmClientFunctionCall):
            raise ValueError("Second last message must contain a function call")

        function_to_call = self.get_client_tool(
            messages[-1].extra_state["continue_from"]
        )  # type: ignore
        # Extract the copilot function call arguments
        function_call_arguments = messages[-1].extra_state.pop(
            "copilot_function_call_arguments", {}
        )
        # Add the intermediate tool call result to the extra_state
        function_call_arguments["extra_state"] = {
            **messages[-1].extra_state,
            "intermediate_tool_call_result": messages[-1],
        }

        function_call = FunctionCall(
            function=function_to_call,
            **function_call_arguments,
        )
        return function_call

    async def continue_with_function_call(
        self,
        messages: list[LlmClientFunctionCallResultMessage | LlmClientMessage],
    ) -> AsyncGenerator[FunctionCallSSE | StatusUpdateSSE | Sequence[Message], None]:
        async for event in self.handle_function_calls(
            self._prepare_continued_function_call(messages)
        ):
            yield event
        return
