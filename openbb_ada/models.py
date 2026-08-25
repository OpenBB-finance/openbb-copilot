import json
import logging
import re
from enum import Enum
from typing import Any, Literal, cast
from uuid import UUID, uuid4

import xxhash
from fastapi import UploadFile
from openbb_ai.models import (
    BaseSSE,
    Citation,
    ClientArtifact,
    ClientCommandResult,
    ClientFunctionCallError,
    DataContent,
    DataFileReferences,
    DataSourceParamOptionsRequestPayload,
    DataSourceRequestPayload,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    OptionsEndpointParam,
    QueryRequest,
    RawObjectDataFormat,
    SourceInfo,
    Widget,
    WidgetParamOptions,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    computed_field,
    model_validator,
)

from . import constants
from .vector_db import VectorDbDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default input_args for SSRM (Server-Side Row Model) widget queries
DEFAULT_SSRM_INPUT_ARGS: dict[str, Any] = {
    "startRow": 0,
    "endRow": 10000,
    "rowGroupCols": [],
    "valueCols": [],
    "pivotCols": [],
    "pivotMode": False,
    "groupKeys": [],
    "filterModel": {},
    "sortModel": [],
}


class OpenBBValidationError(Exception):
    """Custom validation error to handle validation
    errors in a more user-friendly way."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"OpenBBValidationError: {self.message}"


_SEMANTIC_VIEW_FQN_RE = re.compile(r"^[A-Za-z0-9_$]+\.[A-Za-z0-9_$]+\.[A-Za-z0-9_$]+$")


class SkillCatalogEntry(BaseModel):
    """Lightweight skill entry sent with every query.

    Contains only the metadata needed for the model to decide
    if it needs the full content.
    """

    slug: str = Field(description="The unique slug identifier for the skill.")
    description: str = Field(description="Short description of the skill.")
    updated_at: str = Field(
        alias="updatedAt", description="ISO timestamp of last update."
    )


class SkillPayload(BaseModel):
    """Full skill payload for user-forced or model-requested skills."""

    slug: str = Field(description="The unique slug identifier for the skill.")
    description: str = Field(description="Short description of the skill.")
    content_markdown: str = Field(
        alias="contentMarkdown",
        description="Full markdown content with instructions for the AI.",
    )
    source: Literal["forced_slash", "model_selected"] = Field(
        description="How this skill was selected: forced by user or model."
    )


class DeferredFunctionCall(BaseModel):
    """Serializable tool-call metadata stored across client request boundaries."""

    model_config = ConfigDict(extra="forbid")

    function_name: str
    arguments: dict[str, Any]


class AvailableSemanticView(BaseModel):
    """Semantic view metadata discovered upstream and supplied to Ada."""

    model_config = ConfigDict(extra="ignore")

    fqn: str
    database: str | None = None
    schema_name: str | None = None
    view_name: str | None = None
    base_table: str | None = None
    comment: str | None = None

    @model_validator(mode="before")
    def normalize_field_names(cls, values: Any):
        if not isinstance(values, dict):
            return values

        normalized = dict(values)
        if "schema_name" not in normalized and "schema" in normalized:
            normalized["schema_name"] = normalized["schema"]
        if "view_name" not in normalized and "viewName" in normalized:
            normalized["view_name"] = normalized["viewName"]
        if "base_table" not in normalized and "baseTable" in normalized:
            normalized["base_table"] = normalized["baseTable"]
        return normalized

    @model_validator(mode="after")
    def derive_missing_coordinates_from_fqn(self) -> "AvailableSemanticView":
        parts = self.fqn.split(".")
        if len(parts) == 3:
            if not self.database:
                self.database = parts[0]
            if not self.schema_name:
                self.schema_name = parts[1]
            if not self.view_name:
                self.view_name = parts[2]
        return self


class AdaQueryRequest(QueryRequest):
    """Request for an Ada query."""

    workspace_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Workspace option values keyed by option id.",
    )

    semantic_views: list[str] | None = Field(
        default=None,
        description="List of Snowflake semantic view FQNs selected by the user.",
    )
    available_semantic_views: list[AvailableSemanticView] | None = Field(
        default=None,
        description=(
            "Semantic view candidates discovered upstream and supplied for "
            "Ada-side relevance selection."
        ),
    )

    # Skills support
    skills_catalog: list[SkillCatalogEntry] | None = Field(
        default=None,
        description="Lightweight catalog of available skills.",
    )
    selected_skills: list[SkillPayload] | None = Field(
        default=None,
        description="Full skill payloads for selected skills.",
    )

    @model_validator(mode="before")
    def filter_disabled_workspace_options(cls, values: Any):
        if not isinstance(values, dict):
            return values

        workspace_options = values.get("workspace_options")
        if workspace_options is None:
            values["workspace_options"] = {}
            return values

        if not isinstance(workspace_options, dict):
            return values

        filtered_workspace_options = dict(workspace_options)
        if constants.SNOWFLAKE_NATIVE_APP:
            filtered_workspace_options.pop("agent-orchestration", None)

        values["workspace_options"] = filtered_workspace_options
        return values

    @model_validator(mode="before")
    def filter_invalid_semantic_view_fqns(cls, values: Any):
        """Silently drop semantic view FQNs that don't match DB.SCHEMA.NAME format."""
        views = values.get("semantic_views")
        if views is not None:
            values["semantic_views"] = [
                v for v in views if _SEMANTIC_VIEW_FQN_RE.match(v)
            ] or None

        available_views = values.get("available_semantic_views")
        if available_views is not None:
            filtered_available_views = []
            for view in available_views:
                fqn = None
                if isinstance(view, dict):
                    fqn = view.get("fqn")
                elif hasattr(view, "fqn"):
                    fqn = view.fqn

                if isinstance(fqn, str) and _SEMANTIC_VIEW_FQN_RE.match(fqn):
                    filtered_available_views.append(view)

            values["available_semantic_views"] = filtered_available_views or None
        return values

    @model_validator(mode="before")
    def filter_incompatible_function_call_and_result_messages_before(cls, values: Any):
        # We need to check certain data before parsing, otherwise we raise a 422.
        # This is a workaround to avoid breaking changes.
        to_delete = []
        for index, message in enumerate(values["messages"]):
            if message["role"] == "tool":
                if "data" not in message:
                    to_delete.append(index)
                    continue
                for data_element in message["data"]:
                    try:
                        if "items" in data_element:
                            try:
                                DataContent.model_validate(data_element)
                            except ValidationError:
                                DataFileReferences.model_validate(data_element)
                        elif "error_type" in data_element:
                            ClientFunctionCallError.model_validate(data_element)
                        elif ClientCommandResult.model_validate(data_element):
                            continue
                        else:
                            raise ValueError(
                                f"Invalid data item or file reference: {data_element}"
                            )
                    except (ValueError, ValidationError):
                        # Remove the function call result message
                        to_delete.append(index)
                        # Remove the function call messages
                        to_delete.append(index - 1)

        values["messages"] = [
            msg for i, msg in enumerate(values["messages"]) if i not in to_delete
        ]
        return values

    @model_validator(mode="after")
    def filter_incompatible_function_call_and_result_messages_after(
        self,
    ) -> "AdaQueryRequest":
        """Filter out messages that are not compatible with our schema.

        Doing this, instead of throwing a 422, allows us to continue using old
        chats without breaking changes related to function call and function
        call result schema messages.
        """
        compatible_messages: list[
            LlmClientMessage | LlmClientFunctionCallResultMessage
        ] = []
        # Type assertion to help mypy understand the messages type
        messages: list[LlmClientMessage | LlmClientFunctionCallResultMessage] = (
            self.messages  # type: ignore[has-type]
        )
        for index, message in enumerate(messages):
            match message:
                case LlmClientMessage(
                    role="ai",
                    content=LlmClientFunctionCall(function="get_widget_data"),
                ):
                    next_message = (
                        messages[index + 1] if index + 1 < len(messages) else None
                    )
                    if not isinstance(next_message, LlmClientFunctionCallResultMessage):
                        raise OpenBBValidationError(
                            "Function call messages must always be followed by a function call result message."  # noqa: E501
                        )

                    message.content = cast(LlmClientFunctionCall, message.content)
                    try:
                        models = []
                        for data_source_request in message.content.input_arguments.get(
                            "data_sources", []
                        ):
                            model = DataSourceRequestPayload.model_validate(
                                data_source_request
                            )
                            models.append(model)
                        if not all(model.widget_uuid for model in models):
                            continue
                    except (ValidationError, OpenBBValidationError):
                        continue
                    compatible_messages.append(message)

                # Function call message (get_extra_widget_data)
                case LlmClientMessage(
                    role="ai",
                    content=LlmClientFunctionCall(function="get_extra_widget_data"),
                ):
                    message.content = cast(LlmClientFunctionCall, message.content)
                    for data_source_request in message.content.input_arguments.get(
                        "data_sources", []
                    ):
                        try:
                            DataSourceRequestPayload.model_validate(data_source_request)
                        except ValidationError:
                            continue
                    compatible_messages.append(message)

                # Function call result message
                #
                # We will also remove the previous function call message from
                # the list if it is present if the function call result message
                # is not compatible with the supported schemas.
                case LlmClientFunctionCallResultMessage(
                    function="get_widget_data" | "get_extra_widget_data"
                ):
                    message = cast(LlmClientFunctionCallResultMessage, message)
                    if not message.extra_state:
                        if isinstance(
                            compatible_messages[-1], LlmClientMessage
                        ) and isinstance(
                            compatible_messages[-1].content, LlmClientFunctionCall
                        ):
                            compatible_messages.pop()
                        continue

                    if widget_queries := message.extra_state.get(
                        "copilot_function_call_arguments", {}
                    ).get("widget_queries", []):
                        try:
                            [
                                WidgetQueryRequest.model_validate(widget_query)
                                for widget_query in widget_queries
                            ]
                        except ValidationError:
                            if isinstance(
                                compatible_messages[-1], LlmClientMessage
                            ) and isinstance(
                                compatible_messages[-1].content, LlmClientFunctionCall
                            ):
                                compatible_messages.pop()
                    elif search_queries := message.extra_state.get(
                        "copilot_function_call_arguments", {}
                    ).get("search_queries", []):
                        try:
                            [
                                DataSourceSearchQuery.model_validate(search_query)
                                for search_query in search_queries
                            ]
                        except ValidationError:
                            if isinstance(
                                compatible_messages[-1], LlmClientMessage
                            ) and isinstance(
                                compatible_messages[-1].content, LlmClientFunctionCall
                            ):
                                compatible_messages.pop()
                            continue

                    last_message = messages[index - 1] if index - 1 >= 0 else None
                    last_message_content = (
                        last_message.content
                        if last_message and isinstance(last_message, LlmClientMessage)
                        else None
                    )
                    if not isinstance(last_message_content, LlmClientFunctionCall):
                        raise OpenBBValidationError(
                            "Function call result messages must always be preceded by a function call message."  # noqa: E501
                        )

                    # If we made it this far, the function call result message
                    # is compatible with the supported schema.
                    compatible_messages.append(message)
                case _:
                    compatible_messages.append(message)
        self.messages = compatible_messages
        return self

    @model_validator(mode="after")
    def validate_sufficient_extra_state_is_present(self) -> "AdaQueryRequest":
        messages = cast(
            list[LlmClientMessage | LlmClientFunctionCallResultMessage], self.messages
        )
        last_message = messages[-1]
        if isinstance(last_message, LlmClientFunctionCallResultMessage):
            if last_message.extra_state:
                # First, we check if the function is a get_params_options and if
                # it is, we check if the param_options_widget_query_mapping is
                # present in the extra_state, since this is required to continue
                # executing the function call.
                if (
                    last_message.function == "get_params_options"
                    and "continue_from" in last_message.extra_state
                ):
                    if (
                        "param_options_widget_query_mapping"
                        not in last_message.extra_state
                    ):
                        raise ValueError(
                            "required param_options_widget_query_mapping is missing from extra_state. Was all extra state returned from the client?"  # noqa: E501
                        )

                # Next, we validate that all widget queries from the previous
                # function call are accounted for.  This ensures that the sum of
                # widgets referenced in param_options_widget_query_mapping and
                # completed_data_source_request_query_mapping equals the total
                # number of widget queries made in the original function call
                # (either get_widget_data or get_extra_widget_data).
                if continue_function := last_message.extra_state.get("continue_from"):
                    total_num_widget_queries = 0
                    match continue_function:
                        case "get_widget_data" | "update_widget_in_dashboard":
                            total_num_widget_queries = len(
                                last_message.extra_state.get(
                                    "copilot_function_call_arguments", {}
                                ).get("widget_queries", [])
                            )
                        case "get_extra_widget_data" | "add_widget_to_dashboard":  # noqa: E501
                            total_num_widget_queries = len(
                                last_message.extra_state.get(
                                    "copilot_function_call_arguments", {}
                                ).get("search_queries", [])
                            )
                        case "assign_tasks_to_agent":
                            pass
                        case _:
                            raise ValueError(
                                f"Unknown continue function: {continue_function}"
                            )

                    total_unique_widgets_in_params_options_query_mapping = len(
                        set(
                            el["widget_query_index"]
                            for el in last_message.extra_state.get(  # type: ignore
                                "param_options_widget_query_mapping"
                            )
                        )
                    )
                    total_complete_data_source_requests = len(
                        last_message.extra_state.get(
                            "completed_data_source_request_query_mapping", []
                        )
                    )
                    if (
                        total_unique_widgets_in_params_options_query_mapping
                        + total_complete_data_source_requests
                        != total_num_widget_queries
                    ):  # noqa: E501
                        raise ValueError(
                            "The number of widgets in the param_options_widget_query_mapping and completed_data_source_request_query_mapping in the extra_state does not match the number of widget queries."  # noqa: E501
                        )
        return self


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    DELETE = "DELETE"


class TaskRequest(BaseModel):
    """A task to be executed by an agent."""

    id: UUID = Field(
        default_factory=uuid4,
        description="A unique identifier for the task.",
    )
    description: str = Field(
        description="Start by tagging the agent with **@Agent Name** (not the id!) in markdown bold (**...**) and then a detailed description of the task to assign."  # noqa: E501
    )
    assigned_holder_url: str = Field(
        description="The URL of the agent holder to which the task is assigned. This is used to display the task in the workspace."  # noqa: E501
    )
    assigned_agent_id: str = Field(
        description="The ID of the agent to which the task is assigned. This is used to display the task in the workspace."  # noqa: E501
    )


class NavigationBarTab(BaseModel):
    """A tab in the navigation bar."""

    name: str = Field(description="The display name of the tab.")


class DataSourceSearchQuery(BaseModel):
    description: str = Field(
        description="A description hint of the data source you want to use."
    )
    query: str = Field(description="The data to retrieve from the data source.")
    user_context: str | None = Field(
        default=None,
        description=(
            "The original user query context for generating widget arguments. "
            "If provided, this will be used instead of 'query' for argument "
            "generation while 'query' is used only for widget searching."
        ),
    )
    inner_tab: str | None = Field(
        default=None,
        description=(
            "Optional: the tab name to place the widget in when a navigation "
            "bar exists on the dashboard. If not provided, the widget will be "
            "placed in the currently active tab."
        ),
    )


class WidgetQueryRequest(BaseModel):
    widget_uuid: UUID = Field(description="The widget UUID to request data from.")
    # The field below is purely to allow the LLM to provide some reasoning which
    # helps a lot with deciding whether to use the current input arguments for
    # the widget or not.
    does_query_match_input_args: str | None = Field(
        default=None,
        description="STEP-BY-STEP ANALYSIS: For each parameter, examine if the current value actually fulfills what the user is requesting. Compare the user's specific query against each parameter's current value and description. If any current values don't satisfy the user's request, explain specifically why not. Always populate this field, even though it is optional.",  # noqa: E501
    )
    use_current_inputs: bool = Field(
        default=False,
        description="Set to True ONLY after verifying that each parameter's current value actually satisfies the user's specific request. If any parameter needs a different value to fulfill the user's query, set to False.",  # noqa: E501
    )
    query: str = Field(
        description="The query of the data you want to retrieve from the data source."
    )


class SqlTableInfo(BaseModel):
    table_name: str
    sql_schema: str
    unique_column_values: dict[str, Any]
    description: str | None = None
    metadata: dict[str, Any] | None = None


class LlmFunctionResult(BaseModel):
    content: str


class ContextElement(BaseModel):
    content: Any = Field(description="Content to be specified by child classes.")
    source_info: SourceInfo


class ParsedContext(ContextElement):
    content: str = Field(description="The data content as a JSON string.")
    data_format: RawObjectDataFormat | None = None


class UnstructuredContext(ContextElement):
    content: str = Field(description="The data content as a string.")

    def to_client_artifact(self) -> ClientArtifact:
        content = self.content
        type_: Literal[
            "text", "table", "snowflake_query", "snowflake_python", "html"
        ] = "text"
        try:
            parse_as = self.source_info.metadata.get("parse_as")
            if parse_as == "table":
                content = json.loads(self.content)
                type_ = "table"
            elif parse_as in {"snowflake_query", "snowflake_python", "html"}:
                type_ = parse_as
        except Exception as _:
            logger.warning(
                "Failed to parse content as table. Content: %s", self.content
            )
            content = "Failed to parse the artifact inline, please check the artifact sent in the step-by-step reasoning."  # noqa: E501

        return ClientArtifact(
            uuid=self.source_info.uuid,
            name=self.source_info.name,
            description=self.source_info.description,
            type=type_,
            content=content,
            query_data_source=self.source_info.metadata.get("query_data_source"),
        )


class StructuredContext(ContextElement):
    sql_table_info: SqlTableInfo = Field(
        description="The data content as a structured SQL table format."
    )
    data_format: RawObjectDataFormat

    def to_client_artifact(self) -> ClientArtifact:
        return ClientArtifact(
            uuid=self.source_info.uuid,
            name=self.source_info.name,
            description=self.source_info.description,
            type=self.data_format.parse_as,
            content=json.loads(self.content),
            chart_params=self.data_format.chart_params,
            query_data_source=self.data_format.query_data_source,
        )


class CopilotArtifact(BaseModel):
    """A piece of data that is returned from a function call, and used internally."""

    content: str | list[dict]
    source_info: SourceInfo
    data_format: RawObjectDataFormat

    def to_parsed_context(self) -> ParsedContext:
        return ParsedContext(
            content=self.content
            if isinstance(self.content, str)
            else json.dumps(self.content),
            source_info=self.source_info,
            data_format=self.data_format,
        )

    def to_client_artifact(self) -> ClientArtifact:
        return ClientArtifact(
            uuid=self.source_info.uuid,
            name=self.source_info.name,
            description=self.source_info.description,
            type=self.data_format.parse_as,
            content=self.content,
            chart_params=self.data_format.chart_params,
            query_data_source=self.data_format.query_data_source,
        )


class AppArtifactLayoutInput(BaseModel):
    """Widget placement requested by the LLM for a dashboard app artifact."""

    origin: str
    widget_id: str
    x: int
    y: int
    w: int
    h: int
    state: dict[str, Any] | None = None


class AppArtifactTabInput(BaseModel):
    """Tab requested by the LLM for a dashboard app artifact."""

    id: str
    name: str
    layout: list[AppArtifactLayoutInput]


class AppArtifactLayoutItem(BaseModel):
    i: str
    x: int
    y: int
    w: int
    h: int
    state: dict[str, Any] | None = None


class AppArtifactTab(BaseModel):
    id: str
    name: str
    layout: list[AppArtifactLayoutItem]


class AppArtifactDef(BaseModel):
    name: str
    description: str
    allowCustomization: bool = True
    tabs: dict[str, AppArtifactTab]


class AppArtifactWidgetRef(BaseModel):
    i: str
    origin: str
    widget_id: str
    uuid: UUID | None = None
    name: str


class AppArtifact(BaseModel):
    type: Literal["app"] = "app"
    uuid: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    app: AppArtifactDef
    widget_refs: list[AppArtifactWidgetRef]


class AppArtifactSSE(BaseSSE):
    event: Literal["copilotMessageArtifact"] = "copilotMessageArtifact"
    data: AppArtifact


class DocumentQueryResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    artifacts: list[CopilotArtifact] | None = Field(default=None)


class DocumentAgentQueryResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    artifacts: list[CopilotArtifact] = Field(default_factory=list)


class SqlAgentQueryResult(BaseModel):
    answer: str
    queried_tables: list[str] = Field(default_factory=list)
    artifact: CopilotArtifact | None = None


class SqlWidgetContext(BaseModel):
    """SQL-enabled widget context passed to query generation.

    Note: Most fields are optional because the LLM may echo back partial data
    from the prompt. The actual widget lookup uses widget_uuid, so that's the
    only required field.
    """

    widget_uuid: str
    widget_id: str | None = None
    widget_name: str | None = None
    widget_origin: str | None = None
    widget_description: str | None = None
    sql_schema: dict[str, Any] | None = None  # Original JSON schema for SQL generation
    sql_schema_sanitized: str | None = None  # Sanitized for display in prompts
    current_sql: str | None = None


# Backwards-compatible alias
# (main used SqlWidgetDict, feature branch uses SqlWidgetContext)
SqlWidgetDict = SqlWidgetContext


class SqlQueryGenerationResult(BaseModel):
    """Result from SQL query generation for external databases."""

    sql_query: str = Field(description="The generated SQL query")
    widget_uuid: str = Field(description="UUID of the target widget")
    widget_id: str = Field(description="ID of the target widget")
    widget_origin: str = Field(description="Origin of the target widget")
    artifacts: list[CopilotArtifact] = Field(default_factory=list)


FuncCallKwargs = dict[Literal["input_args", "extra_state"], dict[str, Any]]


class SqlQueryFunctionCallResult(SqlQueryGenerationResult):
    """Result from SQL query generation for external databases via function call."""

    def get_function_call_kwargs(self) -> FuncCallKwargs:
        """Get the function call kwargs for executing the SQL query."""
        input_args = {**DEFAULT_SSRM_INPUT_ARGS, "query": self.sql_query}

        extra_state = {
            "copilot_function_call_arguments": {
                # Must match llm_query_widgets signature
                "widget_queries": [
                    {
                        "widget_uuid": (self.widget_uuid),
                        "query": (f"Execute SQL: {self.sql_query}"),
                        "use_current_inputs": False,
                    }
                ],
                "summary": "Executing SQL query",
            },
            # Store original SQL for reference
            "sql_query": self.sql_query,
            "sql_artifact_uuid": self.artifacts[0].source_info.uuid,
        }

        return {"input_args": input_args, "extra_state": extra_state}


class PythonWidgetContext(BaseModel):
    """Python widget context passed to code generation."""

    widget_uuid: str = Field(description="UUID of the widget")
    widget_id: str = Field(description="ID of the widget")
    widget_name: str = Field(description="Name of the widget")
    widget_origin: str = Field(description="Origin of the widget")
    widget_description: str = Field(description="Description of the widget")
    current_code: str | None = Field(default=None, description="Current widget code")


class PythonCodeGenerationResult(BaseModel):
    """Result from Python code generation for Snowflake Snowpark execution."""

    python_code: str = Field(description="The generated Python code")
    widget_uuid: str = Field(description="UUID of the target widget")
    widget_id: str = Field(description="ID of the target widget")
    widget_origin: str = Field(description="Origin of the target widget")
    artifacts: list[CopilotArtifact] = Field(default_factory=list)


class PythonCodeFunctionCallResult(PythonCodeGenerationResult):
    """Result from Python code generation for Snowflake Snowpark execution via function call."""  # noqa: E501

    def get_function_call_kwargs(self) -> FuncCallKwargs:
        """Get the function call kwargs for executing the Python code."""
        input_args = {"prompt": self.python_code}

        extra_state = {
            "copilot_function_call_arguments": {
                # Must match llm_query_widgets signature
                "widget_queries": [
                    {
                        "widget_uuid": (self.widget_uuid),
                        "query": (f"Execute Python code: {self.python_code}"),
                        "use_current_inputs": False,
                    }
                ],
                "summary": "Executing Python code",
            },
            # Store original Python code for reference
            "python_code": self.python_code,
            "python_artifact_uuid": self.artifacts[0].source_info.uuid,
        }

        return {"input_args": input_args, "extra_state": extra_state}


class ContextStructuredQueryResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    artifacts: list[CopilotArtifact] = Field(default_factory=list)


class ContextUnstructuredQueryResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class WebContext(BaseModel):
    url: str
    content: str
    citation: Citation | None = None


class RawContext(BaseModel):
    uuid: UUID
    name: str
    description: str | None = None
    data: DataContent | DataFileReferences
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional element metadata (eg. the selected ticker, endpoint used, etc.).",  # noqa: E501
    )


class SimpleDashboardWidget(BaseModel):
    name: str = Field(description="The name of the widget")
    description: str = Field(default="", description="The description of the widget")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Widget metadata"
    )


class LlmDashboardTitleGenerationRequest(BaseModel):
    widgets: list[SimpleDashboardWidget] | None = Field(
        default=None,
        description="A list of dashboard widgets used for dashboard title generation.",
    )
    openai_api_key: str | None = Field(
        default=None, description="Use a custom OpenAI API key for the request."
    )


class WidgetTitleDescriptionRequest(BaseModel):
    uuid: UUID | None = None
    name: str | None = None
    description: str | None = None
    widget_data: str = Field(description="The data contained in the new widget.")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional element metadata (eg. the selected ticker, endpoint used, etc.).",  # noqa: E501
    )
    openai_api_key: str | None = Field(
        default=None, description="Use a custom OpenAI API key for the request."
    )


class WidgetTitleDescriptionResponse(BaseModel):
    title: str
    description: str
    category: str = Field(default="", description="High-level category classification")
    subcategory: str = Field(
        default="", description="More specific subcategory classification"
    )


class SkillConversationMessage(BaseModel):
    """A single conversation message used for skill generation."""

    role: Literal["human", "ai"] = Field(description="The role of the message author.")
    content: str = Field(description="The text content of the message.")


class SkillGenerationRequest(BaseModel):
    conversation: list[SkillConversationMessage] = Field(
        min_length=1,
        description="The conversation messages to derive the skill from.",
    )
    name_hint: str | None = Field(
        default=None, description="Optional user-suggested name for the skill."
    )
    instructions: str | None = Field(
        default=None,
        description="Optional user guidance on what the skill should capture.",
    )
    existing_slugs: list[str] = Field(
        default_factory=list,
        description="Slugs already in use, to avoid collisions.",
    )
    openai_api_key: str | None = Field(
        default=None, description="Use a custom OpenAI API key for the request."
    )


class SkillGenerationResponse(BaseModel):
    slug: str = Field(
        description=(
            "Short kebab-case identifier for the skill "
            "(lowercase letters, numbers and hyphens, 2-50 characters)."
        )
    )
    description: str = Field(
        description="One or two sentences describing what the skill does."
    )
    content: str = Field(
        description="The full markdown instructions of the skill (the workflow)."
    )


class SemanticModelReference(BaseModel):
    """Reference to a semantic model source for Cortex Analyst."""

    semantic_model_file: str | None = None
    semantic_view: str | None = None

    @model_validator(mode="after")
    def validate_one_source(self) -> "SemanticModelReference":
        if not self.semantic_model_file and not self.semantic_view:
            raise ValueError(
                "Either semantic_model_file or semantic_view must be provided."
            )
        return self


class CodeGenerationRequest(BaseModel):
    """Request model for SQL/Python code generation."""

    widget_uuid: str
    user_prompt: str
    current_code: str | None = None
    language: Literal["sql", "python", "text"]
    sql_schema: dict[str, Any] | None = None
    data_sample: list[dict[str, Any]] | None = None
    semantic_model: str | None = None
    semantic_model_file: str | None = None
    semantic_view: str | None = None
    semantic_models: list[SemanticModelReference] | None = None

    @model_validator(mode="after")
    def validate_single_semantic_source(self) -> "CodeGenerationRequest":
        """Ensure at most one semantic configuration is set."""
        sources = sum(
            bool(v)
            for v in (
                self.semantic_model,
                self.semantic_model_file,
                self.semantic_view,
                self.semantic_models,
            )
        )
        if sources > 1:
            raise ValueError(
                "Only one of semantic_model, semantic_model_file, semantic_view, "
                "or semantic_models may be set."
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_semantic_config(self) -> bool:
        return bool(
            self.semantic_model
            or self.semantic_model_file
            or self.semantic_view
            or self.semantic_models
        )


class CodeGenerationResponse(BaseModel):
    """Response model for SQL/Python code generation."""

    generated_code: str | None = None
    generation_source: Literal["cortex_analyst", "llm"] | None = None


class DataSourceInputField(BaseModel):
    title: str
    description: str
    type: str
    enum: list[Any] | None = None
    default: Any | None = None
    current_value: Any
    get_options: bool
    options_params: list[OptionsEndpointParam] = Field(default_factory=list)


class DataSource(BaseModel):
    origin: str
    id: str
    name: str
    description: str
    # `input_fields`:
    #   - key is the field name
    #   - value is the openapi field schema {"type": ..., "description", ..., etc.}
    # We then convert this to a friendly format to be used with pydantic.create_model
    # {<key>: (<type>, FieldInfo(description="...", etc.))},
    # https://docs.pydantic.dev/latest/concepts/models/#dynamic-model-creation
    input_fields: dict[str, DataSourceInputField] = Field(
        default_factory=dict,  # type: ignore[arg-type]
    )
    widget: Widget

    @computed_field  # type: ignore[misc]
    @property
    def data_source_id(self) -> str:
        return str(self.widget.uuid)


class DataSourceSearchFailure(BaseModel):
    data_source_description: str
    data_source_query: str
    reason: str


class DataSourceSearchResult(BaseModel):
    data_source: DataSource | None = None
    errors: list[DataSourceSearchFailure] | None = None


class InputArgGenerationFailure(BaseModel):
    data_source: DataSource
    query: str
    param_name: str
    reason: str
    example_values: list[str] = Field(default_factory=list)


class InputArgGenerationResult(BaseModel):
    data_source: DataSource
    input_args: dict
    errors: list[InputArgGenerationFailure] | None = None
    used_extra_param_options: bool = False


class WidgetDataSourceRequest(BaseModel):
    payload: DataSourceRequestPayload = Field(
        description="The data source request to use."
    )
    widget: Widget = Field(description="The widget to request data from.")


class WidgetDataSourceParamOptionsRequest(BaseModel):
    payload: list[DataSourceParamOptionsRequestPayload] = Field(
        description="The data source param options request to use for a particular widget"  # noqa: E501
    )
    partial_input_args: InputArgGenerationResult = Field(
        description="The generated partial input arguments to re-use when we've fetched the options."  # noqa: E501
    )


class QueryWidgetRequest(BaseModel):
    widget_uuid: UUID
    widget_query: str
    use_current_inputs: bool = Field(default=False)
    partial_input_args: dict[str, Any] = Field(default_factory=dict)
    extra_param_options: list[WidgetParamOptions] = Field(default_factory=list)

    @model_validator(mode="after")
    def verify_mutually_exclusive_fields(self) -> "QueryWidgetRequest":
        if self.use_current_inputs and (
            self.partial_input_args or self.extra_param_options
        ):
            raise ValueError(
                "Cannot use both use_current_inputs and partial_input_args or extra_param_options"  # noqa: E501
            )
        return self


class QueryExtraWidgetsRequest(BaseModel):
    data_source_description: str
    widget_query: str
    partial_input_args: dict[str, Any] = Field(default_factory=dict)
    extra_param_options: list[WidgetParamOptions] = Field(default_factory=list)
    user_context: str | None = Field(
        default=None,
        description=(
            "The original user query context. If provided, this will be "
            "used for argument generation instead of widget_query."
        ),
    )


class QueryWidgetsResult(BaseModel):
    content: list[WidgetDataSourceRequest | WidgetDataSourceParamOptionsRequest]

    @computed_field
    def must_fetch_param_options(self) -> bool:
        """Returns True if we must fetch param options, False otherwise."""
        if any(
            isinstance(item, WidgetDataSourceParamOptionsRequest)
            for item in self.content
        ):
            return True
        return False


class QueryExtraWidgetsResult(BaseModel):
    content: list[WidgetDataSourceRequest | WidgetDataSourceParamOptionsRequest]
    errors: list[DataSourceSearchFailure] | None = None

    @computed_field
    def must_fetch_param_options(self) -> bool:
        """Returns True if we must fetch param options, False otherwise."""
        if any(
            isinstance(item, WidgetDataSourceParamOptionsRequest)
            for item in self.content
        ):
            return True
        return False


class WidgetFileDetails(BaseModel):
    widget_data: str
    filename: str | None = None
    columns: str | None = None
    index: str | None = None
    image: UploadFile | None = None


class UrlFileReference(BaseModel):
    url: HttpUrl
    filename: str
    extension: str
    source_info: SourceInfo


class UserFile(BaseModel):
    """Represents an UserFile uploaded to the Hub API."""

    file_uuid: UUID
    filename: str
    extension: str


class TargetUserFile(UserFile):
    source_info: SourceInfo


class Document(BaseModel):
    content: bytes
    filename: str
    extension: str
    file_uuid: UUID | None = None
    source_info: SourceInfo
    hash: str = Field(default="")

    @model_validator(mode="before")
    def compute_and_set_hash(cls, values: Any):
        values["hash"] = xxhash.xxh64_hexdigest(values["content"])
        return values

    # So we don't leak binary data
    def __repr__(self) -> str:
        return f"Document(hash='{self.hash}', filename='{self.filename}', extension='{self.extension}', content=Binary content of length {len(self.content)} bytes, source_info='{self.source_info}')"  # noqa: E501


class UnavailableDocument(BaseModel):
    error: str
    source_info: SourceInfo


class DocumentAgentArtifact(BaseModel):
    artifact_id: str
    document_hash: str | None = None
    content: str


class RetrievedDocumentChunk(VectorDbDocument):
    @computed_field
    def retrieved_chunk_id(self) -> str | None:
        document_hash = self.metadata.get("document_hash")
        page_number = self.metadata.get("page_number")
        if page_number:
            return f"{document_hash}-p{page_number}"
        return document_hash


class StructuredTableData(BaseModel):
    """Structured table data from MCP output."""

    rows: list[dict[str, Any]] = Field(
        description="List of row objects with consistent keys"
    )
    column_order: list[str] = Field(
        description="Preferred column order for display", default_factory=list
    )


class DocumentSourceInfo(BaseModel):
    content_id: str = Field(
        description="Retrieved chunk ID or artifact ID that was used to answer the query. Used when peeking / searching documents or returning artifacts."  # noqa: E501
    )
    relevant_direct_quotes: list[str] | None = Field(
        description="Direct quotes from the content_id that was used to answer the query. Never quote more than 2 sentences in a single quote.",  # noqa: E501
        default=None,
    )


class Word(BaseModel):
    top: float
    bottom: float
    x0: float
    x1: float
    text: str


class WordGroup(BaseModel):
    page: int
    words: list[Word]
