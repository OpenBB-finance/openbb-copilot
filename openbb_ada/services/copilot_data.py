import asyncio
import fnmatch
import hashlib
import io
import json
import random
from types import UnionType
from typing import (
    Any,
    Literal,
    Tuple,
    TypeVar,
    Union,
    get_args,
    get_origin,
)
from uuid import UUID

import httpx
import redis.asyncio as redis
from magentic import (
    prompt,
    prompt_chain,
)
from magentic.chat_model.base import ToolSchemaParseError
from magentic.prompt_chain import MaxFunctionCallsError
from openbb_ai.models import (
    Undefined,
    WidgetCollection,
    WidgetParam,
    WidgetParamOption,
)
from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.fields import FieldInfo

from .. import constants
from ..errors import (
    FunctionCallError,
)
from ..models import (
    DataSource,
    DataSourceInputField,
    DataSourceParamOptionsRequestPayload,
    DataSourceRequestPayload,
    DataSourceSearchFailure,
    DataSourceSearchResult,
    InputArgGenerationFailure,
    InputArgGenerationResult,
    QueryExtraWidgetsRequest,
    QueryExtraWidgetsResult,
    QueryWidgetRequest,
    QueryWidgetsResult,
    Widget,
    WidgetDataSourceParamOptionsRequest,
    WidgetDataSourceRequest,
    WidgetParamOptions,
)
from ..utils.ai import get_llm
from ..utils.utils import (
    masked_partial,
    retry_on_exception,
    sanitize_str,
)
from ..vector_db import VectorDb, VectorDbDocument
from ._logging import LoggingService, logfire
from .document import DocumentService
from .template import TemplateService


class CopilotDataService:
    """Provide functionality for retrieving data from external API endpoints."""

    def __init__(
        self,
        template_service: TemplateService,
        logging_service: LoggingService,
        openai_api_key: str | None,
        widget_collection: WidgetCollection | None = None,
    ):
        self._template_service = template_service
        self._logging_service = logging_service

        self._widget_collection = widget_collection
        self._data_sources_map = (
            self._convert_widgets_to_data_sources(
                widgets=widget_collection.primary + widget_collection.secondary
            )
            if widget_collection
            else {}
        )
        self._db_extra_widgets: VectorDb | None = None
        self._openai_api_key = openai_api_key

        # Increment this to invalidate the vector db cache (eg. when changing
        # the data source schema).
        self._db_extra_widgets_hash_seed = "hash_seed=3"

    async def _init(self) -> None:
        if self._data_sources_map is None:
            raise ValueError("Data sources not set.")
        self._logging_service.info(
            f"Loading {len(self._data_sources_map)} data sources into vector db"
        )
        self._db_extra_widgets = (
            await self._load_db_extra_widgets(
                self._widget_collection.extra,
                use_cache=constants.USE_EXTRA_WIDGETS_CACHE,
            )
            if self._widget_collection and self._widget_collection.extra
            else None
        )

    def get_widget_collection(self) -> WidgetCollection | None:
        return self._widget_collection

    def _get_model(self, **kwargs):
        return get_llm(
            model=constants.OPENBB_AGENT_MODEL_SMALL,
            temperature=0.1,
            api_key=self._openai_api_key,
            **kwargs,
        )

    # TODO: These type hints are a little confusing
    def _get_relevant_extra_param_options_for_data_source(
        self,
        data_source: DataSource,
        widgets_param_extra_options: list[WidgetParamOptions],
    ) -> dict[str, list[WidgetParamOption]]:
        return {
            extra_param_option.param_name: extra_param_option.options
            for extra_param_option in widgets_param_extra_options
            if (
                extra_param_option.widget_id == data_source.id
                and extra_param_option.widget_origin == data_source.origin
            )
        }

    def _data_sources_require_extra_param_options(
        self, data_sources: list[DataSource]
    ) -> bool:
        for data_source in data_sources:
            if data_source.input_fields:
                for param in data_source.input_fields.values():
                    if param.get_options:
                        return True
        return False

    async def _get_param_options_request_payloads(
        self, input_arg_generation_result: InputArgGenerationResult
    ) -> list[DataSourceParamOptionsRequestPayload]:
        data_source_params_options_requests: list[
            DataSourceParamOptionsRequestPayload
        ] = []
        for param_info in input_arg_generation_result.data_source.input_fields.values():
            if param_info.get_options:
                if not param_info.options_params:
                    self._logging_service.info(
                        "No options params for param: %s, for data source origin: %s, id: %s. Fetching options with no options endpoint args.",  # noqa: E501
                        param_info.title,
                        input_arg_generation_result.data_source.origin,
                        input_arg_generation_result.data_source.id,
                    )
                    data_source_params_options_requests.append(
                        DataSourceParamOptionsRequestPayload(
                            origin=input_arg_generation_result.data_source.origin,
                            id=input_arg_generation_result.data_source.id,
                            param=param_info.title,
                            options_endpoint_input_args={},
                        )
                    )
                else:
                    self._logging_service.info(
                        "Options params for param: %s, for data source origin: %s, id: %s. Fetching options with options endpoint args.",  # noqa: E501
                        param_info.title,
                        input_arg_generation_result.data_source.origin,
                        input_arg_generation_result.data_source.id,
                    )
                    options_endpoint_input_args = {}
                    # Process each parameter needed for the options endpoint
                    for option_param in param_info.options_params:
                        # If this parameter should inherit its value from another field
                        if option_param.inherit_value_from:
                            if (
                                option_param.inherit_value_from
                                not in input_arg_generation_result.input_args
                                and option_param.inherit_value_from
                                not in input_arg_generation_result.data_source.input_fields  # noqa: E501
                            ):
                                raise ValueError(
                                    f"Cannot inherit value for {option_param.name} from {option_param.inherit_value_from} as it doesn't exist in the generated input arguments or data source input fields."  # noqa: E501
                                )

                            # Check if the value exists in input_args first,
                            # otherwise get from data_source.input_fields
                            if (
                                option_param.inherit_value_from
                                in input_arg_generation_result.input_args
                            ):
                                value = input_arg_generation_result.input_args[
                                    option_param.inherit_value_from
                                ]
                                # When the inherited-from field has get_options
                                # (i.e. its value is also being resolved in this
                                # batch), the LLM may produce a label-style
                                # string (e.g. "north america") while the
                                # options endpoint expects the API-format value
                                # (e.g. "north_america"). Normalize to
                                # snake_case so dependent option fetches work.
                                inherited_field = input_arg_generation_result.data_source.input_fields.get(  # noqa: E501
                                    option_param.inherit_value_from
                                )
                                if (
                                    inherited_field
                                    and inherited_field.get_options
                                    and isinstance(value, str)
                                    and " " in value
                                ):
                                    value = value.strip().lower().replace(" ", "_")
                                options_endpoint_input_args[option_param.name] = value
                            else:
                                # Get the value from data_source.input_fields
                                field_info = input_arg_generation_result.data_source.input_fields[  # noqa: E501
                                    option_param.inherit_value_from
                                ]
                                value = field_info.current_value
                                if value is None:
                                    value = field_info.default
                                options_endpoint_input_args[option_param.name] = value

                    data_source_params_options_requests.append(
                        DataSourceParamOptionsRequestPayload(
                            origin=input_arg_generation_result.data_source.origin,
                            id=input_arg_generation_result.data_source.id,
                            param=param_info.title,
                            options_endpoint_input_args=options_endpoint_input_args,
                        )
                    )
        return data_source_params_options_requests

    async def _resolve_widget_request_types(
        self,
        input_arg_generation_results: list[InputArgGenerationResult],
    ) -> list[WidgetDataSourceRequest | WidgetDataSourceParamOptionsRequest]:
        """
        Algorithm for handling parameter options in widget data sources:

        1. First, we check if the data source requires any option parameters at all.
           If not, we can immediately create a regular data request.

        2. For data sources that do require options, we identify which parameters
           need dynamic option fetching (those with get_options=True).

        3. We then check if all required option parameters are present in the
           generated input args. If any are missing, we need to fetch options.

        4. Even if all option parameters are present, we check if their values
           have changed from the current values. If they have changed AND we
           haven't already fetched options for these new values, we need to
           fetch fresh options (as dependent options may have changed).

        5. The decision logic:
           - Request options if: parameters are missing OR values changed without
             having fetched options for those values
           - Otherwise: proceed with the regular data request

        6. When requesting options, we create a partial request containing only
           the non-option parameters, allowing the option fetching to proceed
           with the context it needs.

        Three main flows:
        - Flow 1: No options needed → Direct data request
        - Flow 2: Options needed but not ready → Param options request
        - Flow 3: Options ready and valid → Data request with complete args
        """
        result: list[WidgetDataSourceRequest | WidgetDataSourceParamOptionsRequest] = []
        for input_arg_generation_result in input_arg_generation_results:
            if not self._data_sources_require_extra_param_options(
                [input_arg_generation_result.data_source]
            ):
                result.append(
                    WidgetDataSourceRequest(
                        widget=input_arg_generation_result.data_source.widget,
                        payload=DataSourceRequestPayload(
                            widget_uuid=str(
                                input_arg_generation_result.data_source.widget.uuid
                            ),
                            origin=input_arg_generation_result.data_source.origin,
                            id=input_arg_generation_result.data_source.id,
                            input_args=input_arg_generation_result.input_args,
                        ),
                    )
                )
            else:
                input_fields_that_require_options = [
                    param_name
                    for param_name in input_arg_generation_result.data_source.input_fields.keys()  # noqa: E501
                    if input_arg_generation_result.data_source.input_fields[
                        param_name
                    ].get_options
                ]
                all_options_params_present = all(
                    param_name in input_arg_generation_result.input_args
                    for param_name in input_fields_that_require_options
                )

                option_values_changed = False
                if all_options_params_present:
                    for param_name in input_fields_that_require_options:
                        generated_value = input_arg_generation_result.input_args.get(
                            param_name
                        )
                        field = input_arg_generation_result.data_source.input_fields[
                            param_name
                        ]
                        current_value = field.current_value

                        if generated_value is None:
                            option_values_changed = True
                            break

                        generated_cmp = (
                            generated_value.strip()
                            if isinstance(generated_value, str)
                            else generated_value
                        )
                        current_cmp = (
                            current_value.strip()
                            if isinstance(current_value, str)
                            else current_value
                        )

                        if generated_cmp != current_cmp:
                            option_values_changed = True
                            break

                should_request_options = not all_options_params_present or (
                    option_values_changed
                    and not input_arg_generation_result.used_extra_param_options
                )

                if not should_request_options:
                    result.append(
                        WidgetDataSourceRequest(
                            widget=input_arg_generation_result.data_source.widget,
                            payload=DataSourceRequestPayload(
                                widget_uuid=str(
                                    input_arg_generation_result.data_source.widget.uuid
                                ),
                                origin=input_arg_generation_result.data_source.origin,
                                id=input_arg_generation_result.data_source.id,
                                input_args=input_arg_generation_result.input_args,
                            ),
                        )
                    )
                else:
                    # Create partial input args that exclude get_options parameters
                    non_options_input_args = {}
                    for (
                        param_name,
                        value,
                    ) in input_arg_generation_result.input_args.items():
                        input_field: DataSourceInputField | None = (
                            input_arg_generation_result.data_source.input_fields.get(
                                param_name
                            )
                        )
                        if input_field is None or not input_field.get_options:
                            non_options_input_args[param_name] = value

                    # Create a modified InputArgGenerationResult with only
                    # non-options params
                    partial_input_args_result = InputArgGenerationResult(
                        input_args=non_options_input_args,
                        data_source=input_arg_generation_result.data_source,
                        errors=input_arg_generation_result.errors,
                        used_extra_param_options=input_arg_generation_result.used_extra_param_options,
                    )

                    result.append(
                        WidgetDataSourceParamOptionsRequest(
                            payload=await self._get_param_options_request_payloads(
                                input_arg_generation_result=input_arg_generation_result,
                            ),
                            partial_input_args=partial_input_args_result,
                        )
                    )
        return result

    def _raise_function_call_error_on_input_arg_generation_failures(
        self,
        input_arg_generation_results: list[InputArgGenerationResult],
    ):
        failures = []
        for input_arg_generation_result in input_arg_generation_results:
            if input_arg_generation_result.errors:
                self._logging_service.error(
                    "Error generating input args for data source: %s. Error: %s",
                    input_arg_generation_result.data_source.id,
                    [error.reason for error in input_arg_generation_result.errors],
                )
                failures.extend(input_arg_generation_result.errors)
        if failures:
            function_call_error_message = "Failed to generate the following input arguments for data sources: \n\n"  # noqa: E501
            for failure in failures:
                function_call_error_message += (
                    f"Data Source: {failure.data_source.data_source_id}\n"
                    f"Parameter: {failure.param_name}\n"
                    f"Query: {failure.query}\n"
                    f"Reason: {failure.reason}\n"
                )
            function_call_error_message += (
                "NEXT ACTION: Ask the user for guidance.\n"
                "Suggest they select the appropriate input arguments on the widget itself"  # noqa: E501
                "for you, since this could help.\n"  # noqa: E501
            )  # noqa: E501
            if failure.example_values:
                function_call_error_message += f"Say, 'Some possible example inputs include: {', '.join(f'`{value}`' for value in failure.example_values)}, etc.'\n\n"  # noqa: E501

            raise FunctionCallError(function_call_error_message)

    @logfire.instrument("CopilotDataService.query_widgets")
    async def query_widgets(
        self,
        query_widget_requests: list[QueryWidgetRequest],
    ) -> QueryWidgetsResult:
        # Filter out widgets not in the data sources map (e.g. extra
        # widgets that the LLM referenced but aren't on the dashboard).
        valid_requests = []
        skipped_widgets = []
        for widget_query in query_widget_requests:
            if str(widget_query.widget_uuid) in self._data_sources_map:
                valid_requests.append(widget_query)
            else:
                skipped_widgets.append(str(widget_query.widget_uuid))

        if skipped_widgets:
            self._logging_service.warning(
                "Skipping widgets not found in dashboard: %s",
                skipped_widgets,
            )

        if not valid_requests:
            raise FunctionCallError(
                "None of the requested widgets were found on the dashboard."
            )

        query_widget_requests = valid_requests
        data_sources = [
            self._get_data_source_from_map(widget_query.widget_uuid)
            for widget_query in query_widget_requests
        ]

        # If we don't have option endpoints (or all queries require us to use
        # the current state), we can just generate input args and return the
        # data source requests.
        if all(
            request.use_current_inputs for request in query_widget_requests
        ) or not self._data_sources_require_extra_param_options(data_sources):
            tasks: list = []
            for widget_query in query_widget_requests:
                data_source = self._get_data_source_from_map(widget_query.widget_uuid)

                # Special handling for rich_note widgets: ensure name and
                # description are updated with content
                query_to_use = widget_query.widget_query
                # Check if this is a markdown/rich_note widget by checking params
                is_markdown_widget = (
                    data_source.widget.origin == "OpenBB Workspace"
                    and len(data_source.widget.params) == 1
                    and data_source.widget.params[0].name == "content"
                    and data_source.widget.params[0].type == "text"
                )
                if is_markdown_widget:
                    # Check if the query is about updating content
                    query_lower = widget_query.widget_query.lower()
                    content_keywords = [
                        "change",
                        "update",
                        "modify",
                        "write",
                        "rewrite",
                        "make",
                        "convert",
                        "transform",
                    ]
                    if any(keyword in query_lower for keyword in content_keywords):
                        # Enhance the query to explicitly request name and
                        # description updates
                        if "name" not in query_lower and "title" not in query_lower:
                            query_to_use = (
                                f"{widget_query.widget_query}. Also update the "
                                "widget name and description to match the new content."
                            )
                            self._logging_service.info(
                                "Enhanced rich_note update query to include "
                                "name/description: %s",
                                query_to_use,
                            )

                tasks.append(
                    self._generate_input_args(
                        data_source=data_source,
                        query=query_to_use,
                        use_current_inputs=widget_query.use_current_inputs,
                    )
                )
            # Execute concurrently
            input_arg_generation_results = await asyncio.gather(*tasks)
            self._raise_function_call_error_on_input_arg_generation_failures(
                input_arg_generation_results=input_arg_generation_results
            )
            # Then we convert the input arg generation results into data source requests
            return QueryWidgetsResult(
                content=await self._resolve_widget_request_types(
                    input_arg_generation_results
                )
            )

        # If we have option endpoints, but no the extra options yet, we generate
        # all the other input arguments first...
        if self._data_sources_require_extra_param_options(data_sources) and not any(
            widget_query.extra_param_options for widget_query in query_widget_requests
        ):
            tasks = []
            for widget_query in query_widget_requests:
                data_source = self._get_data_source_from_map(widget_query.widget_uuid)
                tasks.append(
                    self._generate_non_options_input_args(
                        data_source=data_source,
                        query=widget_query.widget_query,
                        use_current_inputs=widget_query.use_current_inputs,
                    )
                )
            input_arg_generation_results = await asyncio.gather(*tasks)
            self._raise_function_call_error_on_input_arg_generation_failures(
                input_arg_generation_results=input_arg_generation_results
            )
            return QueryWidgetsResult(
                content=await self._resolve_widget_request_types(
                    input_arg_generation_results
                )
            )
        # If we have option endpoints and the extra options, we can generate the
        # complete input arguments by re-using the already-generated partial
        # input arguments and only generating the input args for parameters with
        # option endpoints. We also re-use the already-generated data source
        # requests that didn't require fetching param options.
        if self._data_sources_require_extra_param_options(data_sources) and any(
            widget_query.extra_param_options for widget_query in query_widget_requests
        ):
            tasks = []
            for widget_query in query_widget_requests:
                data_source = self._get_data_source_from_map(widget_query.widget_uuid)
                tasks.append(
                    self._generate_input_args(
                        data_source=data_source,
                        query=widget_query.widget_query,
                        use_current_inputs=widget_query.use_current_inputs,
                        extra_param_options=widget_query.extra_param_options,
                        partial_input_args=widget_query.partial_input_args,
                    )
                )
            input_arg_generation_results = await asyncio.gather(*tasks)
            self._raise_function_call_error_on_input_arg_generation_failures(
                input_arg_generation_results=input_arg_generation_results
            )
            return QueryWidgetsResult(
                content=await self._resolve_widget_request_types(
                    input_arg_generation_results
                )
            )

        # If we reach this point without returning, something has gone wrong,
        # and we should bubble up the error.
        raise FunctionCallError("Unable to query widgets.")

    @logfire.instrument("CopilotDataService.query_extra_widgets")
    async def query_extra_widgets(
        self,
        query_extra_widget_requests: list[QueryExtraWidgetsRequest],
    ) -> QueryExtraWidgetsResult:
        # TODO: When continuing with the param options, we are still searching
        # through all the extra widgets again. This has two problems: 1. It's
        # slow and unnecessary.  2. If the LLM chooses different widget(s), we
        # may need to fetch options params *again* for the newly-selected
        # widget(s).
        #
        # The solution is to find a way to cache the data sources that we
        # originally retrieved, and then to re-use them. We can probably do this
        # using `extra_state`.
        tasks: list[Any] = []

        # First, we search for and select the right data sources.
        tasks = [
            self._get_extra_widgets(
                query_extra_widget_request=query_extra_widget_request,
            )
            for query_extra_widget_request in query_extra_widget_requests
        ]
        data_source_search_results: list[DataSourceSearchResult] = await asyncio.gather(
            *tasks
        )
        paired_search_results: list[
            tuple[QueryExtraWidgetsRequest, DataSourceSearchResult]
        ] = list(
            zip(query_extra_widget_requests, data_source_search_results, strict=True)
        )
        successful_search_pairs: list[tuple[QueryExtraWidgetsRequest, DataSource]] = [
            (search_request, search_result.data_source)
            for search_request, search_result in paired_search_results
            if search_result.data_source
        ]
        found_data_sources: list[DataSource] = [
            data_source for _, data_source in successful_search_pairs
        ]
        data_source_search_failures: list[DataSourceSearchFailure] = [
            result.errors[0]  # There should be only one error.
            for _, result in paired_search_results
            if result.errors
        ]
        # If we don't have option endpoints, we can just generate input args
        # and return the data source requests.
        if not self._data_sources_require_extra_param_options(found_data_sources):
            tasks = []
            for search_request, search_result in paired_search_results:
                if data_source := search_result.data_source:
                    # Use user_context if available, otherwise widget_query
                    base_query = (
                        search_request.user_context or search_request.widget_query
                    )

                    # Special handling for rich_note widgets: ensure name and
                    # description are generated
                    query_to_use = base_query
                    # Check if this is a markdown/rich_note widget
                    is_markdown_widget = (
                        data_source.widget.origin == "OpenBB Workspace"
                        and (
                            data_source.widget.widget_id == "rich_note"
                            or (
                                len(data_source.widget.params) > 0
                                and data_source.widget.params[0].name == "content"
                                and data_source.widget.params[0].type == "text"
                            )
                        )
                    )
                    if is_markdown_widget:
                        # For new widget creation, always request name and description
                        query_lower = base_query.lower()
                        if "name" not in query_lower and "title" not in query_lower:
                            query_to_use = (
                                f"{base_query}. Generate an appropriate widget name "
                                "and description based on the content."
                            )
                            self._logging_service.info(
                                (
                                    "Enhanced rich_note creation query to include "
                                    "name/description: %s"
                                ),
                                query_to_use,
                            )

                    tasks.append(
                        self._generate_input_args(
                            data_source=data_source,
                            query=query_to_use,
                        )
                    )
            input_arg_generation_results = await asyncio.gather(*tasks)
            self._raise_function_call_error_on_input_arg_generation_failures(
                input_arg_generation_results=input_arg_generation_results
            )
            return QueryExtraWidgetsResult(
                content=await self._resolve_widget_request_types(
                    input_arg_generation_results
                ),
                errors=data_source_search_failures,
            )
        # If we have option endpoints, but not extra options yet, we generate all
        # the non-options input args first...
        if self._data_sources_require_extra_param_options(
            found_data_sources
        ) and not any(
            request.extra_param_options for request in query_extra_widget_requests
        ):
            input_arg_generation_results = []
            tasks = []
            for request, data_source in successful_search_pairs:
                tasks.append(
                    self._generate_non_options_input_args(
                        data_source=data_source,
                        query=request.user_context or request.widget_query,
                    )
                )
            input_arg_generation_results = await asyncio.gather(*tasks)
            self._raise_function_call_error_on_input_arg_generation_failures(
                input_arg_generation_results=input_arg_generation_results
            )
            return QueryExtraWidgetsResult(
                content=await self._resolve_widget_request_types(
                    input_arg_generation_results
                ),
                errors=data_source_search_failures,
            )

        # If we have option endpoints and the extra options, we can generate the
        # complete input arguments by re-using the already-generated partial
        # input arguments and only generating the input args for parameters with
        # option endpoints. We also re-use the already-generated data source
        # requests that didn't require fetching param options.
        if self._data_sources_require_extra_param_options(found_data_sources) and any(
            request.extra_param_options for request in query_extra_widget_requests
        ):
            tasks = []
            for request, data_source in successful_search_pairs:
                tasks.append(
                    self._generate_input_args(
                        data_source=data_source,
                        query=request.user_context or request.widget_query,
                        extra_param_options=request.extra_param_options,
                        partial_input_args=request.partial_input_args,
                    )
                )
            input_arg_generation_results = await asyncio.gather(*tasks)
            self._raise_function_call_error_on_input_arg_generation_failures(
                input_arg_generation_results=input_arg_generation_results
            )
            return QueryExtraWidgetsResult(
                content=await self._resolve_widget_request_types(
                    input_arg_generation_results
                ),
                errors=data_source_search_failures,
            )

        raise FunctionCallError("Something went wrong querying extra widgets.")

    async def _get_extra_widgets(
        self,
        query_extra_widget_request: QueryExtraWidgetsRequest,
    ) -> DataSourceSearchResult:
        joint_query = f"{query_extra_widget_request.data_source_description}: {query_extra_widget_request.widget_query}"  # noqa: E501
        latest_candidate_ids: list[str] = []
        self._logging_service.info(
            "Searching relevant data source with guidance: %s", joint_query
        )

        async def _llm_search_data_sources(
            data_source_description: str, query: str
        ) -> str:
            """"""
            self._logging_service.info("Querying data sources: %s", query)
            candidate_data_sources: list[
                DataSource
            ] = await self._search_extra_data_sources(
                query=f"{data_source_description}: {query}"
            )
            data_sources_str = self._template_service.render_copilot_external_data_source_search_results(  # noqa: E501
                data_sources=candidate_data_sources
            )
            latest_candidate_ids[:] = [source.id for source in candidate_data_sources]
            self._logging_service.info(
                "Retrieved data sources: %s",
                latest_candidate_ids,
            )
            return data_sources_str

        ResponseModel = create_model(
            "ResponseModel",
            data_source_id=(
                Literal[*self._list_all_extra_data_sources()],  # type: ignore[valid-type]
                ...,
            ),
        )

        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @prompt_chain(
            self._template_service.render_copilot_external_data_source_search_prompt(
                user_query=query_extra_widget_request.widget_query,
                data_source_hint=query_extra_widget_request.data_source_description,
            ),
            max_calls=3,
            functions=[_llm_search_data_sources],
            model=self._get_model(),
        )
        async def _choose_data_source() -> ResponseModel | None: ...  # type: ignore[valid-type]

        try:
            response = await _choose_data_source()
        except MaxFunctionCallsError as err:
            self._logging_service.warning(
                "Max function calls reached: Unable to find data source."
            )
            self._logging_service.debug(
                "Extra widget chooser hit MaxFunctionCallsError "
                "for joint_query=%s with candidate_ids=%s; "
                "data_source_description=%s error=%s",
                joint_query,
                latest_candidate_ids,
                query_extra_widget_request.data_source_description,
                err,
            )
            return DataSourceSearchResult(
                errors=[
                    DataSourceSearchFailure(
                        data_source_description=query_extra_widget_request.data_source_description,
                        data_source_query=query_extra_widget_request.widget_query,
                        reason="Unable to find relevant widget",
                    )
                ],
            )
        except ToolSchemaParseError as err:
            self._logging_service.error("Error parsing tool schema: %s", err)
            self._logging_service.debug(
                "Extra widget chooser schema parse failed "
                "for joint_query=%s with candidate_ids=%s",
                joint_query,
                latest_candidate_ids,
            )
            response = None

        if response:
            self._logging_service.info(
                "Selected data source: %s for query: %s",
                response.data_source_id,
                joint_query,
            )
            if data_source := self.get_data_source_from_map_or_db(
                response.data_source_id
            ):
                return DataSourceSearchResult(
                    data_source=data_source,
                )
            else:
                return DataSourceSearchResult(
                    errors=[
                        DataSourceSearchFailure(
                            data_source_description=query_extra_widget_request.data_source_description,
                            data_source_query=query_extra_widget_request.widget_query,
                            reason="Unable to find data source.",
                        )
                    ],
                )

        if latest_candidate_ids:
            self._logging_service.warning(
                "Extra widget chooser did not select a data source "
                "for joint_query=%s despite candidate_ids=%s",
                joint_query,
                latest_candidate_ids,
            )
        return DataSourceSearchResult(
            errors=[
                DataSourceSearchFailure(
                    data_source_description=query_extra_widget_request.data_source_description,
                    data_source_query=query_extra_widget_request.widget_query,
                    reason="Unable to find relevant widget.",
                )
            ],
        )

    @staticmethod
    def _get_field_type_schema(param: WidgetParam) -> dict:
        schema: dict[str, Any] = {}
        schema["type"] = (
            CopilotDataService._convert_widget_param_type_into_openapi_type(  # noqa: E501
                param
            )
        )
        if param.options:
            schema["enum"] = param.options
        return schema

    def _convert_widgets_to_data_sources(
        self,
        widgets: list[Widget],
    ) -> dict[str, DataSource]:
        data_sources = {}
        for widget in widgets:
            try:
                input_fields = {}

                # Special handling for built-in widgets to ensure parameter
                # definitions. Check if this is a markdown/rich_note widget
                # (could be UUID for instances or "rich_note" for definitions)
                is_markdown_widget = widget.origin == "OpenBB Workspace" and (
                    widget.widget_id == "rich_note"
                    or (
                        len(widget.params) > 0
                        and widget.params[0].name == "content"
                        and widget.params[0].type == "text"
                        and len(widget.params) <= 3
                    )  # Allow up to 3 params (content, name, description)
                )
                if is_markdown_widget:
                    # Ensure rich_note has content, name, and description parameters
                    has_content_param = any(
                        param.name == "content" for param in (widget.params or [])
                    )
                    has_name_param = any(
                        param.name == "name" for param in (widget.params or [])
                    )
                    has_description_param = any(
                        param.name == "description" for param in (widget.params or [])
                    )

                    if (
                        not has_content_param
                        or not has_name_param
                        or not has_description_param
                    ):
                        # Create missing parameter definitions
                        from openbb_ai.models import WidgetParam

                        if not widget.params:
                            widget.params = []
                        else:
                            # Make mutable copy
                            widget.params = list(widget.params)

                        if not has_content_param:
                            content_param = WidgetParam(
                                name="content",
                                description=("The text content for the note widget"),
                                type="text",
                                default_value=(
                                    "# New Note\n\nClick to edit this note and "
                                    "add your content..."
                                ),
                            )
                            widget.params.append(content_param)

                        if not has_name_param:
                            name_param = WidgetParam(
                                name="name",
                                description=(
                                    "A short, descriptive title for the widget "
                                    "(max 50 characters). AI will generate this "
                                    "based on the content."
                                ),
                                type="text",
                                default_value="Note",
                            )
                            widget.params.append(name_param)

                        if not has_description_param:
                            description_param = WidgetParam(
                                name="description",
                                description=(
                                    "A brief description of what the content is "
                                    "about (max 100 characters). AI will generate "
                                    "this based on the content."
                                ),
                                type="text",
                                default_value="A note widget",
                            )
                            widget.params.append(description_param)

                if widget.params:
                    for param in widget.params:
                        param_dict = {}
                        param_dict.update(
                            {
                                "default": param.default_value,
                                "current_value": param.current_value,
                                "description": param.description,
                                "title": param.name,
                                "get_options": param.get_options,
                                "options_params": (
                                    param.options_params if param.options_params else []
                                ),
                                **self._get_field_type_schema(param),
                            }
                        )
                        input_fields[param.name] = DataSourceInputField(**param_dict)
                data_source = DataSource(
                    origin=widget.origin,
                    id=widget.widget_id,
                    name=widget.name,
                    description=widget.description,
                    input_fields=input_fields,
                    widget=widget,
                )
                data_sources[str(data_source.widget.uuid)] = data_source
            except ValidationError as err:  # TODO: Narrow down this exception
                self._logging_service.error(
                    "Failed to convert custom widget definition to data source: %s", err
                )
        return data_sources

    def _get_user_db(self) -> VectorDb | None:
        return self._db_extra_widgets

    def _get_data_source_from_map(self, data_source_id: str | UUID) -> DataSource:
        data_source_id_str = (
            str(data_source_id) if isinstance(data_source_id, UUID) else data_source_id
        )
        return self._data_sources_map[data_source_id_str]

    def get_data_source_from_map_or_db(self, data_source_id: str) -> DataSource | None:
        # First search data sources loaded from primary and secondary widgets
        if data_source := self._data_sources_map.get(data_source_id):
            return data_source
        if db := self._get_user_db():
            for doc in db.docs:
                if doc.metadata["data_source_id"] == data_source_id:
                    return DataSource.model_validate_json(doc.metadata["data_source"])
        return None

    def _list_all_extra_data_sources(self) -> list[str]:
        if db := self._get_user_db():
            return [doc.metadata["data_source_id"] for doc in db.docs]
        return []

    async def _search_extra_data_sources(self, query: str) -> list[DataSource]:
        if db := self._get_user_db():
            documents = await db.search(query, k=10)
            return [
                DataSource.model_validate_json(doc.metadata["data_source"])
                for doc in documents
            ]
        return []

    async def search_relevant_extra_widgets(
        self, queries: list[str], k: int = 10
    ) -> list[Widget]:
        """Search extra widgets VectorDB and return top-k relevant widgets.

        This is used for RAG-based widget context retrieval, where only the most
        relevant widget metadata is included in the system prompt instead of all
        available widgets.

        For each query (user message), retrieves k candidates with similarity scores.
        Results are aggregated across all queries, deduplicated by widget UUID
        (keeping the best score for each widget), and the top-k overall are returned.

        Args:
            queries: List of user queries to search against widget metadata.
            k: Maximum number of relevant widgets to return.

        Returns:
            List of Widget objects most relevant to the queries.
        """
        if not self._db_extra_widgets:
            self._logging_service.info(
                "No extra widgets VectorDB available for relevance search"
            )
            return []

        if not queries:
            return []

        # Aggregate results across all queries
        # Key: widget UUID (data_source_id), Value: (Widget, best_score)
        # Lower score = better (L2 distance)
        widget_scores: dict[str, tuple[Widget, float]] = {}

        # Run searches concurrently
        search_tasks = [
            self._db_extra_widgets.search_with_scores(query, k=k) for query in queries
        ]
        all_results = await asyncio.gather(*search_tasks)

        for query, results in zip(queries, all_results, strict=True):
            self._logging_service.info(
                "Retrieved %d candidates for query: %s",
                len(results),
                query[:100] if len(query) > 100 else query,
            )

            for doc, score in results:
                widget_id = doc.metadata["data_source_id"]
                widget = DataSource.model_validate_json(
                    doc.metadata["data_source"]
                ).widget

                # Keep the best (lowest) score for each widget
                if widget_id not in widget_scores:
                    widget_scores[widget_id] = (widget, score)  # First occurrence
                elif score < widget_scores[widget_id][1]:
                    widget_scores[widget_id] = (widget, score)  # Better (lower) score

        # Sort by score (ascending - lower is better) and take top k
        sorted_widgets = sorted(widget_scores.values(), key=lambda x: x[1])
        top_widgets = [widget for widget, _ in sorted_widgets[:k]]

        self._logging_service.info(
            "Final selection: %d relevant extra widgets from %d unique candidates",
            len(top_widgets),
            len(widget_scores),
        )
        return top_widgets

    def _get_field_definitions_from_data_source(
        self,
        data_source: DataSource,
        exclude_params_with_options_endpoints: bool = False,
        fixed_partial_input_args: dict[str, str] | None = None,
        use_current_inputs: bool = False,
    ) -> dict[str, Tuple[Any, FieldInfo]]:
        fixed_partial_input_args = fixed_partial_input_args or {}

        field_definitions: dict[str, Tuple[Any, FieldInfo]] = {}
        for (
            field_name,
            field_schema,
        ) in data_source.input_fields.items():
            if (
                exclude_params_with_options_endpoints
                and field_schema.get_options
                and not use_current_inputs
            ):
                has_fixed_value = field_name in fixed_partial_input_args
                current = field_schema.current_value
                has_current_value = (
                    current not in (None, "")
                    if isinstance(current, str)
                    else current is not None
                )
                if not has_fixed_value and not has_current_value:
                    continue

            field_default = field_schema.default
            field_description = field_schema.description
            # If we have a fixed value for this field, we can use that to
            # generate the field type, we make sure it is a Literal type
            # with the fixed value.
            if fixed_value := fixed_partial_input_args.get(field_name):
                field_info: FieldInfo = Field(
                    default=fixed_value, description=field_description
                )  # type: ignore[assignment]
                field_type = Literal[fixed_value]  # type: ignore
            # Otherwise, we use the spec.
            else:
                if field_default != Undefined.UNDEFINED:
                    field_info: FieldInfo = Field(  # type: ignore[no-redef, assignment]
                        default=field_default, description=field_description
                    )
                else:
                    field_info = Field(description=field_description)
                if field_schema:
                    field_type = self._convert_openapi_type_str_into_python_type(  # type: ignore
                        field_schema.type
                    )
                    if not field_type:
                        raise ValueError(
                            f"Failed to get 'field_type' from data source {data_source.name}"  # noqa: E501
                        )
                    if choices := field_schema.enum:
                        if get_origin(field_type) is list:
                            field_type = list[Literal[*choices]]  # type: ignore
                        else:
                            field_type = Literal[*choices]  # type: ignore
                    if field_default is None or field_schema.current_value is None:
                        field_type = field_type | None  # type: ignore
            field_definitions[field_name] = (field_type, field_info)
        return field_definitions

    async def _filter_input_arg_options(
        self,
        query: str,
        data_source: DataSource,
        extra_param_options: list[WidgetParamOptions],
    ) -> dict[str, list[WidgetParamOption]]:
        def _tool_filter_options(
            pattern: str, param_options: WidgetParamOptions
        ) -> str:
            pattern = pattern.lower()
            filtered_options: list[WidgetParamOption] = []
            seen = set()

            for option in param_options.options:
                if len(filtered_options) > 50:
                    break

                option_label_lower = option.label.lower()
                option_value_lower = option.value.lower()

                if fnmatch.fnmatch(option_label_lower, pattern) or fnmatch.fnmatch(
                    option_value_lower, pattern
                ):
                    if id(option) not in seen:
                        filtered_options.append(option)
                        seen.add(id(option))

            result_str = "Filtered options based on query pattern:\n"
            if len(filtered_options) >= 50:
                filtered_options = filtered_options[:50]
                result_str += "Warning: Only showing the first 50 options -- consider narrowing down your query.\n\n"  # noqa: E501

            for option in filtered_options:
                result_str += (
                    f"- value: {option.value}\n    (description: {option.label})\n"
                )

            self._logging_service.info(
                "Filtered options for %s, with pattern: %s, result: %s",
                param_options.param_name,
                pattern,
                result_str,
            )
            return result_str

        tasks = []
        for extra_param_option in extra_param_options:
            self._logging_service.info(
                "Filtering input argument options for %s: %s, with number of possible options: %s",  # noqa: E501
                data_source.name,
                extra_param_option.param_name,
                len(extra_param_option.options),
            )

            async def _filter_options_factory(
                user_query: str,
                data_source: DataSource,
                param_name: str,
                widget_param_options: list[WidgetParamOption],
                tool: Any,
            ):
                instructions = (
                    self._template_service.render_filter_input_arg_options_prompt(  # noqa: E501
                        user_query=user_query,
                        data_source=data_source,
                        param_name=param_name,
                        widget_param_options=widget_param_options,
                    )
                )

                @retry_on_exception(
                    max_retries=3, exceptions=(httpx.RemoteProtocolError,)
                )
                @prompt_chain(
                    "{instructions}",
                    model=self._get_model(),
                    functions=[tool],
                    max_calls=6,
                )
                async def _llm(instructions: str) -> list[str]: ...  # type: ignore

                # We skip if there are no options to filter.
                if len(widget_param_options) == 0:
                    return []

                pattern_results = []
                pattern_search_failed = False

                try:
                    # Try pattern-based LLM filtering first
                    pattern_results = await _llm(instructions=instructions)
                except MaxFunctionCallsError:
                    self._logging_service.warning(
                        "Max function calls reached for param: %s", param_name
                    )
                    pattern_search_failed = True

                # If pattern search succeeded and found results, return them
                if pattern_results:
                    return pattern_results

                # If pattern search failed or empty, try semantic search
                status = "failed" if pattern_search_failed else "returned empty results"
                self._logging_service.info(
                    "Pattern-based filtering %s for param: %s, trying semantic search",
                    status,
                    param_name,
                )

                try:
                    semantic_results = await self._semantic_search_options(
                        user_query=user_query,
                        param_options=widget_param_options,
                        top_k=min(20, len(widget_param_options)),
                    )

                    # Return semantic results (could be empty if no good matches found)
                    return semantic_results

                except Exception as e:
                    self._logging_service.error(
                        "Semantic search failed for param: %s with query '%s': %s",
                        param_name,
                        user_query,
                        str(e),
                    )
                    # Final fallback - return empty array to prevent token explosion
                    return []

            tasks.append(
                _filter_options_factory(
                    user_query=query,
                    data_source=data_source,
                    param_name=extra_param_option.param_name,
                    widget_param_options=extra_param_option.options,
                    # We have to do some tricks to make the tool work in
                    # `magentic` (due to function signatures) *and* where we
                    # pass in the param options to avoid falling victim to the
                    # late binding behaviour of Python.
                    # https://docs.python-guide.org/writing/gotchas/#late-binding-closures
                    tool=masked_partial(
                        _tool_filter_options, param_options=extra_param_option
                    ),
                )
            )

        filtered_option_values = await asyncio.gather(*tasks)
        filtered_extra_param_options = {}
        for widget_param_options, filtered_param_values in zip(
            extra_param_options, filtered_option_values, strict=True
        ):
            filtered_extra_param_options[widget_param_options.param_name] = [
                param_option
                for param_option in widget_param_options.options
                if param_option.value in filtered_param_values
            ]
        return filtered_extra_param_options

    async def _semantic_search_options(
        self,
        user_query: str,
        param_options: list[WidgetParamOption],
        top_k: int = 10,
    ) -> list[str]:
        """
        Use embedding-based semantic search to find relevant options.

        Args:
            user_query: The user's original query
            param_options: List of available parameter options
            top_k: Maximum number of options to return

        Returns:
            List of option values ranked by semantic similarity
        """
        if not param_options:
            return []

        self._logging_service.info(
            "Performing semantic search for query: %s with %d options",
            user_query,
            len(param_options),
        )

        try:
            # Create temporary VectorDb and add option documents
            temp_db = VectorDb()

            documents = []
            for option in param_options:
                doc = VectorDbDocument(
                    page_content=option.label.strip(), metadata={"value": option.value}
                )
                documents.append(doc)

            await temp_db.add(documents)

            # Search for similar options
            search_results = await temp_db.search(user_query, k=top_k)

            # Extract option values from search results
            result = [doc.metadata["value"] for doc in search_results]

            self._logging_service.info(
                "Semantic search found %d relevant options",
                len(result),
            )

            return result

        except Exception as e:
            self._logging_service.error("Error during semantic search: %s", str(e))
            return []

    async def _generate_non_options_input_args(
        self,
        data_source: DataSource,
        query: str,
        use_current_inputs: bool = False,
    ) -> InputArgGenerationResult:
        # Special handling for rich_note widgets: ensure name and
        # description are generated
        query_to_use = query
        # Check if this is a markdown/rich_note widget
        is_markdown_widget = data_source.widget.origin == "OpenBB Workspace" and (
            data_source.widget.widget_id == "rich_note"
            or (
                len(data_source.widget.params) > 0
                and data_source.widget.params[0].name == "content"
                and data_source.widget.params[0].type == "text"
            )
        )
        if is_markdown_widget:
            query_lower = query.lower()
            if "name" not in query_lower and "title" not in query_lower:
                query_to_use = (
                    f"{query}. Generate an appropriate widget name and "
                    "description based on the content."
                )
                self._logging_service.info(
                    "Enhanced rich_note query to include name/description: %s",
                    query_to_use,
                )

        input_arg_generation_results = await self._generate_input_args(
            data_source=data_source,
            query=query_to_use,
            exclude_params_with_options_endpoints=True,
            use_current_inputs=use_current_inputs,
        )
        return input_arg_generation_results

    @logfire.instrument("CopilotDataService._generate_input_args")
    async def _generate_input_args(
        self,
        query: str,
        data_source: DataSource,
        use_current_inputs: bool = False,
        extra_param_options: list[WidgetParamOptions] | None = None,
        partial_input_args: dict[str, str] | None = None,
        exclude_params_with_options_endpoints: bool = False,
    ) -> InputArgGenerationResult:
        self._logging_service.info(
            "Generating input arguments for %s for query: %s", data_source.name, query
        )
        if use_current_inputs:
            self._logging_service.info(
                "Using current state for %s for query: %s", data_source.name, query
            )
            if not any(
                input_field.current_value is None
                and input_field.default is Undefined.UNDEFINED
                for input_field in data_source.input_fields.values()
            ):
                return InputArgGenerationResult(
                    input_args={
                        widget_param.name: widget_param.current_value
                        for widget_param in data_source.widget.params
                        if widget_param.current_value is not None
                    },
                    data_source=data_source,
                    used_extra_param_options=bool(extra_param_options),
                )
            else:
                self._logging_service.warning(
                    "`use_current_inputs` was set to `True` for widget %s, but some input arguments have no current value and and are required. Forcing input arguments to be generated.",  # noqa: E501
                    data_source.widget.name,
                )

        if not data_source.input_fields:
            return InputArgGenerationResult(
                input_args={},
                data_source=data_source,
                used_extra_param_options=bool(extra_param_options),
            )

        field_definitions = self._get_field_definitions_from_data_source(
            data_source=data_source,
            exclude_params_with_options_endpoints=exclude_params_with_options_endpoints,
            fixed_partial_input_args=partial_input_args,
            use_current_inputs=use_current_inputs,
        )
        # TODO: Further optimization: if all of the fields are included in the
        # partial_input_args (i.e. fixed), then we don't need to generate
        # anything!
        if len(field_definitions) == 0:
            return InputArgGenerationResult(
                input_args={},
                data_source=data_source,
                used_extra_param_options=bool(extra_param_options),
            )
        filtered_extra_param_options = {}
        if extra_param_options:
            self._logging_service.info(
                "Filtering input argument options using LLM for %s: %s",
                data_source.name,
                [
                    extra_param_option.param_name
                    for extra_param_option in extra_param_options
                ],
            )
            filtered_extra_param_options = await self._filter_input_arg_options(
                query=query,
                data_source=data_source,
                extra_param_options=extra_param_options,
            )
            self._logging_service.info(
                "Filtered input argument options using LLM for %s: %s",
                data_source.name,
                filtered_extra_param_options,
            )

            optional_option_fields = {
                name
                for name, field in data_source.input_fields.items()
                if field.get_options
                and field.default is None
                and field.current_value is None
            }
            invalid_input_args = []
            for (
                param_name,
                filtered_param_options,
            ) in filtered_extra_param_options.items():
                if allowed_options := next(
                    (
                        option.options
                        for option in extra_param_options
                        if option.param_name == param_name
                    ),
                    None,
                ):
                    if not filtered_param_options:
                        if param_name in optional_option_fields:
                            continue
                        invalid_input_args.append(
                            InputArgGenerationFailure(
                                data_source=data_source,
                                query=query,
                                param_name=param_name,
                                reason="No valid options were found.",
                                example_values=random.sample(
                                    [option.label for option in allowed_options],
                                    min(4, len(allowed_options)),
                                ),
                            )
                        )
                        continue

                    if any(
                        filtered_option not in allowed_options
                        for filtered_option in filtered_param_options
                    ):
                        invalid_input_args.append(
                            InputArgGenerationFailure(
                                data_source=data_source,
                                query=query,
                                param_name=param_name,
                                reason="Did not match list of valid possible options",
                                example_values=random.sample(
                                    [option.label for option in allowed_options],
                                    min(4, len(allowed_options)),
                                ),
                            )
                        )
                        continue
                elif not filtered_param_options:
                    invalid_input_args.append(
                        InputArgGenerationFailure(
                            data_source=data_source,
                            query=query,
                            param_name=param_name,
                            reason="No valid options were found.",
                        )
                    )
                    continue

                # If we pass the check, we can update the field definition to
                # only allow the filtered options by using a `Literal` type.

                # We are making the assumption that the possible field types are:
                # - Primitives (int, float, str, bool)
                # - An optional primitive (int | None, float | None, str | None,
                # bool | None)
                # - A list of primitives (list[int], list[float], list[str], list[bool])
                # - An optional list of primitives (list[int] | None,
                # list[float] | None, list[str] | None, list[bool] | None)
                option_values = [option.value for option in filtered_param_options]
                if (
                    get_origin(field_definitions[param_name][0]) is Union
                    or get_origin(field_definitions[param_name][0]) is UnionType
                ):
                    make_type_optional = False
                    if type(None) in get_args(field_definitions[param_name][0]):
                        make_type_optional = True

                    field_type = field_definitions[param_name][0]
                    if get_origin(get_args(field_type)[0]) is list:
                        field_definitions[param_name] = (
                            list[Literal[*option_values]],  # type: ignore
                            field_definitions[param_name][1],
                        )
                    else:
                        field_definitions[param_name] = (
                            Literal[*option_values],  # type: ignore
                            field_definitions[param_name][1],
                        )

                    if make_type_optional:
                        field_definitions[param_name] = (
                            # This is the type of the field
                            field_definitions[param_name][0] | None,
                            # This is the FieldInfo object
                            field_definitions[param_name][1],
                        )
                elif get_origin(field_definitions[param_name][0]) is list:
                    field_definitions[param_name] = (
                        list[Literal[*option_values]],  # type: ignore
                        field_definitions[param_name][1],
                    )
                else:
                    field_definitions[param_name] = (
                        Literal[*option_values],  # type: ignore
                        field_definitions[param_name][1],
                    )

            if invalid_input_args:
                # If we weren't able to get any kind of list of suitable input
                # argument options after filtering, there's no point in
                # generating input arguments.
                return InputArgGenerationResult(
                    input_args={},
                    errors=invalid_input_args,
                    data_source=data_source,
                    used_extra_param_options=bool(extra_param_options),
                )

        input_model: BaseModel = create_model("InputModel", **field_definitions)  # type: ignore[call-overload]

        T = TypeVar("T", bound=BaseModel)

        async def _generate_input_args_from_input_model(
            query: str, data_source: DataSource, input_model: T
        ) -> InputArgGenerationResult:
            @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
            @prompt(
                sanitize_str(
                    self._template_service.render_generate_input_arguments_prompt(
                        query=query,
                        data_source=data_source,
                        extra_param_options=(
                            filtered_extra_param_options
                            if filtered_extra_param_options
                            else {}
                        ),
                    )
                ),
                model=self._get_model(),
                max_retries=3,
            )
            async def _llm() -> input_model: ...  # type: ignore

            input_args = await _llm()
            input_args_dict = input_args.model_dump()

            return InputArgGenerationResult(
                input_args=input_args_dict,
                data_source=data_source,
                used_extra_param_options=bool(extra_param_options),
            )

        input_arg_generation_result = await _generate_input_args_from_input_model(
            query, data_source, input_model
        )

        self._logging_service.info(
            "Input argument generation result: %s for %s (%s)",
            input_arg_generation_result.input_args,
            data_source.id,
            data_source.origin,
        )
        return input_arg_generation_result

    @staticmethod
    def _is_field_optional(annotation: Any) -> bool:
        return get_origin(annotation) is Union and type(None) in get_args(annotation)

    @staticmethod
    def _convert_widget_param_type_into_openapi_type(
        param: WidgetParam,
    ) -> str:
        param_type = "string"
        if param.type in ("text", "date", "ticker", "endpoint", "tabs"):
            param_type = "string"
        elif param.type in ("integer", "boolean", "number"):
            param_type = param.type

        if param.multi_select:
            param_type = f"array[{param_type}]"

        return param_type

    @staticmethod
    def _convert_openapi_type_str_into_python_type(
        type_str: str,
    ) -> type | None | str:
        python_type: type | None = None

        if "string" in type_str:
            python_type = str
        elif "boolean" in type_str:
            python_type = bool
        elif "integer" in type_str:
            python_type = int
        elif "number" in type_str:
            python_type = float
        elif "null" in type_str:
            python_type = None

        if "array" in type_str:
            if python_type:
                return list[python_type]  # type: ignore[valid-type]
            # Bare "array" without inner type — default to list[str]
            return list[str]
        return python_type

    @staticmethod
    def _get_aredis_client() -> redis.Redis:
        connection_pool = redis.ConnectionPool(
            host=constants.REDIS_HOST,
            port=constants.REDIS_PORT,
            db=0,
        )
        redis_client = redis.Redis(connection_pool=connection_pool)
        return redis_client

    def _get_widgets_hash(self, widgets: list[Widget]) -> str:
        data = [
            widget.model_dump_json(
                # Exclude current_value from params to avoid creating a new
                # vector db if only the param(s) current value(s) change
                exclude={
                    "uuid": True,
                    "params": {i: {"current_value"} for i in range(len(widget.params))},
                }
            )
            for widget in widgets
        ]
        data.append(self._db_extra_widgets_hash_seed)
        # Include embedding model in hash to invalidate cache when model changes
        embedding_model_key = f"embedding={constants.OPENBB_EMBEDDING_MODEL_PROVIDER}:{constants.OPENBB_EMBEDDING_MODEL}"  # noqa: E501
        data.append(embedding_model_key)
        json_data = json.dumps(data)
        sha256_hash = hashlib.sha256(json_data.encode("utf-8")).hexdigest()
        return sha256_hash

    async def _build_vector_db_from_widgets(self, widgets: list[Widget]) -> VectorDb:
        data_sources = list(self._convert_widgets_to_data_sources(widgets).values())
        db = VectorDb()
        documents = []

        for data_source in data_sources:
            widget = data_source.widget

            # Single format for all providers - most important info first
            # so truncation preserves key semantic content
            # Priority: name, description, table, category, columns
            content = f"{widget.name}: {widget.description}"

            # For SQL widgets, add table name early (important for matching)
            if widget.metadata and widget.metadata.get("schema"):
                try:
                    schema: dict = widget.metadata["schema"]
                    if tableName := schema.get("tableName"):
                        content += f" Table: {tableName}"
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

            # Add column names last (can be truncated if needed)
            if widget.metadata and widget.metadata.get("schema"):
                try:
                    schema = widget.metadata["schema"]
                    if columns := schema.get("columns"):
                        col_names = []
                        for col in columns:
                            if "name" in col:
                                # Strip any type info in parentheses
                                # e.g. VARCHART, FLOAT, etc
                                name = col["name"].split("(")[0].strip()
                                col_names.append(name)
                        if col_names:
                            content += f" Columns: {', '.join(col_names)}"
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    self._logging_service.warning(
                        "Failed to parse schema for widget %s: %s",
                        widget.widget_id,
                        e,
                    )

            # Add category info
            if category := getattr(widget, "category", None):
                content += f" [{category}]"
            if sub_category := getattr(widget, "sub_category", None):
                content += f" [{sub_category}]"

            doc = VectorDbDocument(
                page_content=content,
                metadata={
                    "data_source_id": data_source.data_source_id,
                    "data_source": data_source.model_dump_json(),
                },
            )
            documents.append(doc)

        # Use pooling=False for widget metadata (no chunking, truncate if needed)
        await db.add(documents, pooling=False)
        self._logging_service.info(
            "Built external data sources vector db from scratch."
        )
        return db

    async def _load_db_extra_widgets(
        self, widgets: list[Widget], use_cache: bool = True
    ) -> VectorDb:
        if use_cache:
            try:
                async with self._get_aredis_client() as aredis_client:
                    sha256_hash = self._get_widgets_hash(widgets)
                    key = f"extra_widgets_db:{sha256_hash}"
                    self._logging_service.info(
                        "Checking if cache exists for key: %s", key
                    )
                    if await aredis_client.exists(key):
                        self._logging_service.info("Loading vector db from cache.")
                        if vector_db_zip := await aredis_client.get(key):
                            db = await asyncio.to_thread(
                                DocumentService._extract_and_load_downloaded_vector_db,
                                vector_db_zip,
                            )
                            return db
                    self._logging_service.info(
                        "Building vector db from scratch and caching."
                    )
                    db = await self._build_vector_db_from_widgets(widgets)
                    buffer = io.BytesIO()
                    db.save_to_buffer(buffer)
                    KEY_EXPIRATION = 86400  # 1 day
                    await aredis_client.set(key, buffer.getvalue(), KEY_EXPIRATION)
                    return db
            except Exception as err:
                self._logging_service.error(
                    "Failed to build vector db with caching: %s", err
                )
        self._logging_service.info("Building vector db from scratch (cache disabled)")
        db = await self._build_vector_db_from_widgets(widgets)
        return db
