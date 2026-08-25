import base64
from typing import Annotated, Any, AsyncGenerator, Callable, cast
from uuid import UUID, uuid4

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openbb_ai.models import (
    ClientFunctionCallError,
    DataContent,
    DataFileFormat,
    DataFileReferences,
    DocxDataFormat,
    LlmClientFunctionCallResultMessage,
    PdfDataFormat,
    PlaintextDataFormat,
    SingleDataContent,
    SingleFileReference,
    SourceInfo,
    UserAPIKeys,
    Widget,
    WidgetCollection,
    WidgetParam,
    WorkspaceState,
)

from openbb_ada import constants

from .copilot import (
    CitationService,
    ClientFunctionCallService,
    ContextService,
    CopilotDataService,
    CopilotService,
    DocumentService,
    LoggingService,
    NativeFunctionCallService,
    TemplateService,
    UrlRetrievalService,
)
from .models import (
    AdaQueryRequest,
    AvailableSemanticView,
    Document,
    LlmDashboardTitleGenerationRequest,
    SkillCatalogEntry,
    SkillPayload,
    UnavailableDocument,
    UrlFileReference,
)
from .services import (
    ChatTitleGenerationService,
    DashboardTitleGenerationService,
    DocumentAgentService,
    EditorContentGenerationService,
    McpDataService,
    PromptEnhancementService,
    PythonCodeGenerationService,
    SqlAgentService,
    SqlQueryGenerationService,
    UserFileService,
    WebSearchLlmService,
)
from .snowflake_factory import build_snowflake_cortex_analyst_service
from .utils.utils import get_current_datetime


def get_trace_id(request: Request) -> str:
    return request.state.trace_id


def get_user_id(request: Request) -> str:
    return request.state.user_id


# We do this because we dynamically generate some values in this pydantic model
# (such as UUID values for widgets for internal use by Copilot), and we want to
# make sure they stay consistent across the various usages. By using dependency
# injection, we can ensure that the same instance of the AdaQueryRequest is used
# throughout every service that depends on this dependency.
def get_llm_query_request(
    llm_query_request: AdaQueryRequest,
) -> AdaQueryRequest:
    return llm_query_request


def get_llm_dashboard_title_generation_request(
    llm_dashboard_title_generation_request: LlmDashboardTitleGenerationRequest,
) -> LlmDashboardTitleGenerationRequest:
    return llm_dashboard_title_generation_request


def get_logging_service(
    name: str | None = None,
) -> Callable[[str | None], LoggingService]:
    def _get_logging_service(
        x_trace_id: Annotated[str | None, Depends(get_trace_id)],
    ) -> LoggingService:
        return LoggingService(
            trace_id=UUID(x_trace_id) if x_trace_id else None, name=name
        )

    return _get_logging_service


async def get_user_api_keys(request: Request) -> UserAPIKeys | None:
    payload = await request.json()
    if api_keys := payload.get("api_keys"):
        return UserAPIKeys(**api_keys)
    return None


async def get_user_file_service(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(auto_error=False)),
    ],
    user_id: str = Depends(get_user_id),
) -> AsyncGenerator[UserFileService, None]:
    if constants.OPENBB_PAYMENTS_BASE_URL is None:
        raise ValueError("OPENBB_PAYMENTS_BASE_URL is not set")
    service = UserFileService(
        base_url=constants.OPENBB_PAYMENTS_BASE_URL or "",
        access_token=credentials.credentials if credentials else "",
        user_id=user_id,
    )
    try:
        yield service
    finally:
        await service.close()


def get_current_datetime_with_timezone(
    query: Annotated[AdaQueryRequest, Depends(get_llm_query_request)],
) -> str:
    return get_current_datetime(query.timezone)


def get_workspace_state(
    query: Annotated[AdaQueryRequest, Depends(get_llm_query_request)],
) -> WorkspaceState | None:
    return query.workspace_state


def get_workspace_options(
    query: Annotated[AdaQueryRequest, Depends(get_llm_query_request)],
) -> dict[str, Any]:
    return query.workspace_options


def get_semantic_views(
    query: Annotated[AdaQueryRequest, Depends(get_llm_query_request)],
) -> list[str] | None:
    return query.semantic_views


def get_available_semantic_views(
    query: Annotated[AdaQueryRequest, Depends(get_llm_query_request)],
) -> list[AvailableSemanticView] | None:
    return query.available_semantic_views


def get_skills_catalog(
    query: Annotated[AdaQueryRequest, Depends(get_llm_query_request)],
) -> list[SkillCatalogEntry] | None:
    """Extract skills catalog from the request."""
    return query.skills_catalog


def get_selected_skills(
    query: Annotated[AdaQueryRequest, Depends(get_llm_query_request)],
) -> list[SkillPayload] | None:
    """Extract selected skills (from /slug commands) from the request."""
    return query.selected_skills


def get_template_service(
    current_datetime: Annotated[str, Depends(get_current_datetime_with_timezone)],
    workspace_state: Annotated[WorkspaceState | None, Depends(get_workspace_state)],
    workspace_options: Annotated[dict[str, Any], Depends(get_workspace_options)],
    semantic_views: Annotated[list[str] | None, Depends(get_semantic_views)],
    available_semantic_views: Annotated[
        list[AvailableSemanticView] | None, Depends(get_available_semantic_views)
    ],
) -> TemplateService:
    return TemplateService(
        current_datetime=current_datetime,
        workspace_state=workspace_state,
        workspace_options=workspace_options,
        semantic_views=semantic_views,
        available_semantic_views=available_semantic_views,
    )


def get_sql_agent_service(
    logging_service: Annotated[
        LoggingService, Depends(get_logging_service(name="SQL Agent"))
    ],
    template_service: Annotated[TemplateService, Depends(get_template_service)],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> SqlAgentService:
    return SqlAgentService(
        logging_service=logging_service,
        template_service=template_service,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )


def get_document_agent_service(
    logging_service: Annotated[
        LoggingService, Depends(get_logging_service(name="Document Agent"))
    ],
    template_service: Annotated[TemplateService, Depends(get_template_service)],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> DocumentAgentService:
    return DocumentAgentService(
        logging_service=logging_service,
        template_service=template_service,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )


async def get_widget_collection(
    llm_query_request: Annotated[AdaQueryRequest, Depends(get_llm_query_request)],
) -> WidgetCollection:
    if llm_query_request.widgets is None:
        return WidgetCollection()
    return llm_query_request.widgets


async def get_copilot_data_service(
    widget_collection: Annotated[WidgetCollection, Depends(get_widget_collection)],
    template_service: Annotated[TemplateService, Depends(get_template_service)],
    logging_service: Annotated[
        LoggingService,
        Depends(get_logging_service(name="Copilot External Data Service")),
    ],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> CopilotDataService:
    copilot_data_service = CopilotDataService(
        widget_collection=widget_collection,
        template_service=template_service,
        logging_service=logging_service,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )
    # Initializing this service requires some async code, so we handle it here
    # instead of in the __init__ (using the __init__ isn't possible without
    # hacks)
    await copilot_data_service._init()
    return copilot_data_service


def _get_widget_by_origin_and_id(
    widget_collection: WidgetCollection | None,
    origin: str,
    widget_id: str,
) -> Widget | None:
    all_widgets = (
        (
            widget_collection.primary
            + widget_collection.secondary
            + widget_collection.extra
        )
        if widget_collection
        else []
    )

    for widget in all_widgets:
        if widget.origin == origin and widget.widget_id == widget_id:
            return widget

    return None


def _is_valid_base64(data_content_item: SingleDataContent) -> bool:
    return (
        isinstance(data_content_item, SingleDataContent)
        and isinstance(
            data_content_item.data_format,
            (PdfDataFormat, DocxDataFormat, PlaintextDataFormat),
        )
        and data_content_item.data_format.data_type
        in ["pdf", "docx", "txt", "md", "html"]
        and bool(data_content_item.content)
    )


def _handle_split_param_widget(
    widget: Widget,
    data_source_request: dict,
    data_item: DataContent | DataFileReferences,
) -> list[Document | UrlFileReference]:
    results: list[Document | UrlFileReference] = []

    split_param = cast(WidgetParam, widget.split_param)
    base_input_args = data_source_request.get("input_args", {}).copy()
    base_metadata = {
        **widget.metadata,
        "widget_uuid": str(widget.uuid),
    }
    if split_param.name in base_input_args:
        split_input_arg = base_input_args.pop(split_param.name)
        # ... then we need to create a separate
        # SourceInfo for each split input arg.
        for (
            data_content_or_file_reference_item,
            split_input_arg_item,
        ) in zip(data_item.items, split_input_arg, strict=True):
            source_info = SourceInfo(
                type="widget",
                origin=widget.origin,
                widget_id=widget.widget_id,
                uuid=uuid4(),
                name=widget.name,
                description=widget.description,
                metadata={
                    **base_metadata,
                    "input_args": {
                        **base_input_args,
                        split_param.name: [
                            split_input_arg_item
                        ],  # Split params can only be set when multi select is True,
                        # which means param type is a list
                    },
                },
            )

            if isinstance(
                data_content_or_file_reference_item, SingleDataContent
            ) and _is_valid_base64(data_content_or_file_reference_item):
                data_format = cast(
                    DataFileFormat,
                    data_content_or_file_reference_item.data_format,
                )
                results.append(
                    Document(
                        content=base64.b64decode(
                            data_content_or_file_reference_item.content
                        ),
                        filename=data_format.filename,
                        extension=data_format.data_type,
                        source_info=source_info,
                    )
                )
            elif isinstance(data_content_or_file_reference_item, SingleFileReference):
                data_format = cast(
                    DataFileFormat,
                    data_content_or_file_reference_item.data_format,
                )
                results.append(
                    UrlFileReference(
                        url=data_content_or_file_reference_item.url,
                        filename=data_format.filename,
                        extension=data_format.data_type,
                        source_info=source_info,
                    )
                )

    return results


async def get_documents(
    llm_query_request: Annotated[AdaQueryRequest, Depends(get_llm_query_request)],
    logging_service: Annotated[LoggingService, Depends(get_logging_service())],
    user_file_service: Annotated[UserFileService, Depends(get_user_file_service)],
) -> list[Document | UnavailableDocument]:
    documents: list[Document | UnavailableDocument] = []
    url_file_references: list[UrlFileReference] = []
    for message in llm_query_request.messages:
        if isinstance(message, LlmClientFunctionCallResultMessage):
            if message.function in [
                "get_widget_data",
                "get_extra_widget_data",
            ]:
                if not message.input_arguments:
                    raise ValueError("input_arguments is required for get_widget_data")

                for data_source_request, data_item in zip(
                    message.input_arguments["data_sources"], message.data, strict=True
                ):
                    widget = _get_widget_by_origin_and_id(
                        widget_collection=llm_query_request.widgets,
                        origin=data_source_request["origin"],
                        widget_id=data_source_request["id"],
                    )
                    if isinstance(data_item, ClientFunctionCallError):
                        logging_service.warning(
                            "Function call '%s' returned an error of type %s with content %s",  # noqa: E501
                            message.function,
                            data_item.error_type,
                            data_item.content,
                        )
                        continue
                    if widget:
                        base_source_info = SourceInfo(
                            type="widget",
                            origin=widget.origin,
                            widget_id=widget.widget_id,
                            uuid=uuid4(),
                            name=widget.name,
                            description=widget.description,
                            metadata={
                                **widget.metadata,
                                "widget_uuid": str(widget.uuid),
                                "input_args": {
                                    **data_source_request.get("input_args", {}),
                                },
                            },
                        )

                        match data_item:
                            case DataContent():
                                # If we don't need to split the citations by
                                # a param...
                                if not widget.split_param:  # type: ignore[truth-function]
                                    for data_content_item in data_item.items:
                                        # ... then we can use the same source
                                        # info for every document.
                                        source_info = base_source_info
                                        if _is_valid_base64(data_content_item):
                                            data_format = cast(
                                                DataFileFormat,
                                                data_content_item.data_format,
                                            )
                                            documents.append(
                                                Document(
                                                    content=base64.b64decode(
                                                        data_content_item.content
                                                    ),
                                                    filename=data_format.filename,
                                                    extension=data_format.data_type,
                                                    source_info=source_info,
                                                )
                                            )
                                else:
                                    # If we need to split the citations by a
                                    # param...
                                    documents.extend(
                                        cast(
                                            list[Document],
                                            _handle_split_param_widget(
                                                widget=widget,
                                                data_source_request=data_source_request,
                                                data_item=data_item,
                                            ),
                                        )
                                    )
                            case DataFileReferences():
                                if not widget.split_param:  # type: ignore[truth-function]
                                    source_info = base_source_info
                                    for file_reference in data_item.items:
                                        url_file_references.append(
                                            UrlFileReference(
                                                url=file_reference.url,
                                                filename=file_reference.data_format.filename,  # type: ignore
                                                extension=file_reference.data_format.data_type,
                                                source_info=source_info,
                                            )
                                        )
                                else:
                                    url_file_references.extend(
                                        cast(
                                            list[UrlFileReference],
                                            _handle_split_param_widget(
                                                widget=widget,
                                                data_source_request=data_source_request,
                                                data_item=data_item,
                                            ),
                                        )
                                    )
                            case _:
                                logging_service.warning(
                                    "File widget referenced in function call result "
                                    "with origin %s and ID %s is not provided in request.",  # noqa: E501
                                    data_source_request["origin"],
                                    data_source_request["id"],
                                )

                if message.input_arguments and len(
                    message.input_arguments["data_sources"]
                ) != len(message.data):
                    logging_service.warning(
                        "Function call result with function %s does not have the same "
                        "number of data sources as data items.",  # noqa: E501
                        message.function,
                    )

    if url_file_references:
        documents.extend(
            await user_file_service.bulk_download_external_files(url_file_references)
        )
    return documents


async def get_document_service(
    user_file_service: Annotated[UserFileService, Depends(get_user_file_service)],
    template_service: Annotated[TemplateService, Depends(get_template_service)],
    document_agent_service: Annotated[
        DocumentAgentService, Depends(get_document_agent_service)
    ],
    sql_agent_service: Annotated[SqlAgentService, Depends(get_sql_agent_service)],
    logging_service: Annotated[
        LoggingService, Depends(get_logging_service(name="Document Service"))
    ],
    documents: Annotated[list[Document | UnavailableDocument], Depends(get_documents)],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> DocumentService:
    document_service = DocumentService(
        user_file_service=user_file_service,
        document_agent_service=document_agent_service,
        template_service=template_service,
        sql_agent_service=sql_agent_service,
        logging_service=logging_service,
        documents=documents,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )
    await document_service._init()
    return document_service


def get_context_service(
    template_service: Annotated[TemplateService, Depends(get_template_service)],
    sql_agent_service: Annotated[SqlAgentService, Depends(get_sql_agent_service)],
    logging_service: Annotated[
        LoggingService, Depends(get_logging_service(name="Context Service"))
    ],
) -> ContextService:
    context_service = ContextService(
        template_service=template_service,
        sql_agent_service=sql_agent_service,
        logging_service=logging_service,
    )
    return context_service


def get_web_search_llm_service(
    logging_service: Annotated[
        LoggingService,
        Depends(get_logging_service(name="Web Search LLM Service")),
    ],
    template_service: Annotated[TemplateService, Depends(get_template_service)],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> WebSearchLlmService:
    return WebSearchLlmService(
        logging_service=logging_service,
        template_service=template_service,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )


def get_dashboard_title_generation_service(
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> DashboardTitleGenerationService:
    return DashboardTitleGenerationService(
        template_service=TemplateService(),
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )


def get_chat_title_generation_service(
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> ChatTitleGenerationService:
    return ChatTitleGenerationService(
        template_service=TemplateService(),
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )


def get_prompt_enhancement_service(
    template_service: Annotated[TemplateService, Depends(get_template_service)],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> PromptEnhancementService:
    return PromptEnhancementService(
        template_service=template_service,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )


def get_sql_query_generation_service(
    template_service: Annotated[TemplateService, Depends(get_template_service)],
    logging_service: Annotated[
        LoggingService,
        Depends(get_logging_service(name="SQL Query Generation Service")),
    ],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> SqlQueryGenerationService | None:
    return SqlQueryGenerationService(
        template_service=template_service,
        logging_service=logging_service,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )


def get_python_code_generation_service(
    template_service: Annotated[TemplateService, Depends(get_template_service)],
    logging_service: Annotated[
        LoggingService,
        Depends(get_logging_service(name="Python Code Generation Service")),
    ],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> PythonCodeGenerationService | None:
    # Python widgets are only available in Snowflake mode.
    if not constants.SNOWFLAKE_NATIVE_APP:
        return None
    return PythonCodeGenerationService(
        template_service=template_service,
        logging_service=logging_service,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )


def get_sql_suggestion_service(
    logging_service: Annotated[
        LoggingService,
        Depends(get_logging_service(name="SQL Query Generation Service")),
    ],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> SqlQueryGenerationService:
    """Return SQL suggestion service for /v1/generate/code."""
    return SqlQueryGenerationService(
        template_service=TemplateService(
            current_datetime=get_current_datetime(),
            workspace_state=None,
            workspace_options=None,
        ),
        logging_service=logging_service,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )


def get_editor_content_generation_service(
    logging_service: Annotated[
        LoggingService,
        Depends(get_logging_service(name="Editor Content Generation Service")),
    ],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> EditorContentGenerationService:
    """Return editor code/content suggestion service for /v1/generate/code."""
    return EditorContentGenerationService(
        logging_service=logging_service,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )


def get_python_suggestion_service(
    logging_service: Annotated[
        LoggingService,
        Depends(get_logging_service(name="Python Code Generation Service")),
    ],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> PythonCodeGenerationService:
    """Return Python suggestion service for /v1/generate/code."""
    return PythonCodeGenerationService(
        template_service=TemplateService(
            current_datetime=get_current_datetime(),
            workspace_state=None,
            workspace_options=None,
        ),
        logging_service=logging_service,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )


def get_citation_service() -> CitationService:
    return CitationService()


def get_mcp_data_service(
    context_service: Annotated[ContextService, Depends(get_context_service)],
    logging_service: Annotated[
        LoggingService, Depends(get_logging_service(name="MCP Data Service"))
    ],
    template_service: Annotated[TemplateService, Depends(get_template_service)],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
) -> McpDataService:
    return McpDataService(
        context_service=context_service,
        logging_service=logging_service,
        template_service=template_service,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
    )


def get_copilot_service(
    request: Request,
    document_service: Annotated[DocumentService, Depends(get_document_service)],
    context_service: Annotated[ContextService, Depends(get_context_service)],
    template_service: Annotated[TemplateService, Depends(get_template_service)],
    url_retrieval_service: Annotated[UrlRetrievalService, Depends(UrlRetrievalService)],
    web_search_llm_service: Annotated[
        WebSearchLlmService, Depends(get_web_search_llm_service)
    ],
    copilot_data_service: Annotated[
        CopilotDataService, Depends(get_copilot_data_service)
    ],
    logging_service: Annotated[
        LoggingService, Depends(get_logging_service(name="Copilot Service"))
    ],
    citation_service: Annotated[CitationService, Depends(get_citation_service)],
    mcp_data_service: Annotated[McpDataService, Depends(get_mcp_data_service)],
    prompt_enhancement_service: Annotated[
        PromptEnhancementService, Depends(get_prompt_enhancement_service)
    ],
    sql_query_generation_service: Annotated[
        SqlQueryGenerationService | None, Depends(get_sql_query_generation_service)
    ],
    python_code_generation_service: Annotated[
        PythonCodeGenerationService | None, Depends(get_python_code_generation_service)
    ],
    semantic_views: Annotated[list[str] | None, Depends(get_semantic_views)],
    available_semantic_views: Annotated[
        list[AvailableSemanticView] | None, Depends(get_available_semantic_views)
    ],
    api_keys: Annotated[UserAPIKeys | None, Depends(get_user_api_keys)],
    user_id: Annotated[str, Depends(get_user_id)],
    workspace_state: Annotated[WorkspaceState | None, Depends(get_workspace_state)],
    workspace_options: Annotated[dict[str, Any], Depends(get_workspace_options)],
    skills_catalog: Annotated[
        list[SkillCatalogEntry] | None, Depends(get_skills_catalog)
    ],
    selected_skills: Annotated[list[SkillPayload] | None, Depends(get_selected_skills)],
) -> CopilotService:
    snowflake_cortex_analyst_service = None
    if constants.SNOWFLAKE_NATIVE_APP:
        snowflake_cortex_analyst_service = build_snowflake_cortex_analyst_service(
            request=request,
            logging_service=logging_service,
            sql_query_generation_service=sql_query_generation_service,
            python_code_generation_service=python_code_generation_service,
        )

    return CopilotService(
        user_id=user_id,
        document_service=document_service,
        context_service=context_service,
        template_service=template_service,
        url_retrieval_service=url_retrieval_service,
        copilot_data_service=copilot_data_service,
        logging_service=logging_service,
        client_function_call_service=ClientFunctionCallService(
            copilot_data_service=copilot_data_service,
            logging_service=logging_service,
        ),
        native_function_call_service=NativeFunctionCallService(
            document_service=document_service,
            logging_service=logging_service,
            web_search_llm_service=web_search_llm_service,
            context_service=context_service,
            template_service=template_service,
            citation_service=citation_service,
            prompt_enhancement_service=prompt_enhancement_service,
            sql_query_generation_service=sql_query_generation_service,
            python_code_generation_service=python_code_generation_service,
            snowflake_cortex_analyst_service=snowflake_cortex_analyst_service,
            semantic_views=semantic_views,
            available_semantic_views=available_semantic_views,
        ),
        citation_service=citation_service,
        mcp_data_service=mcp_data_service,
        openai_api_key=api_keys.openai_api_key if api_keys else None,
        workspace_options=workspace_options,
        workspace_state=workspace_state,
        prompt_enhancement_service=prompt_enhancement_service,
        sql_query_generation_service=sql_query_generation_service,
        python_code_generation_service=python_code_generation_service,
        skills_catalog=skills_catalog,
        selected_skills=selected_skills,
    )
