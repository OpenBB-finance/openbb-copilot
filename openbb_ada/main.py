import logging
import uuid
from typing import Annotated, AsyncGenerator, cast

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openbb_ai.models import (
    CitationCollectionSSE,
    FunctionCallSSE,
    LlmClientMessage,
    MessageChunkSSE,
    PromptSuggestionsSSE,
    StatusUpdateSSE,
)
from sse_starlette.sse import EventSourceResponse

from . import constants
from .constants import (
    AUTH_ENABLED,
    ENVIRONMENT,
    LLM_WEB_SEARCH_ENABLED,
    ORIGIN_REGEX,
    ORIGINS,
    SNOWFLAKE_NATIVE_APP,
    URL_RETRIEVAL_ENABLED,
)
from .copilot import (
    CopilotService,
    LoggingService,
)
from .dependencies import (
    get_chat_title_generation_service,
    get_copilot_service,
    get_dashboard_title_generation_service,
    get_editor_content_generation_service,
    get_llm_dashboard_title_generation_request,
    get_llm_query_request,
    get_logging_service,
    get_prompt_enhancement_service,
    get_sql_suggestion_service,
    get_web_search_llm_service,
)
from .errors import CodeGenerationError
from .logging import log_exception_with_trace_id, set_up_logging
from .models import (
    AdaQueryRequest,
    AppArtifactSSE,
    CodeGenerationRequest,
    CodeGenerationResponse,
    LlmDashboardTitleGenerationRequest,
    SkillGenerationRequest,
    SkillGenerationResponse,
    WidgetTitleDescriptionRequest,
    WidgetTitleDescriptionResponse,
)
from .services import (
    ChatTitleGenerationService,
    DashboardTitleGenerationService,
    EditorContentGenerationService,
    PromptEnhancementService,
    SkillGenerationService,
    SqlQueryGenerationService,
    WebSearchLlmService,
    WidgetMetadataGenerationService,
)
from .services.editor_content_generation import EditorContentLanguage
from .utils.utils import (
    get_file_details,
    get_human_readable_size,
    rate_limit,
    validate_and_sync,
)

set_up_logging()
logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


app = FastAPI(
    title="OpenBBAI",
    docs_url="/docs" if ENVIRONMENT == "TEST" else None,
    redoc_url="/redoc" if ENVIRONMENT == "TEST" else None,
    openapi_url="/openapi.json" if ENVIRONMENT == "TEST" else None,
)


app.add_exception_handler(Exception, log_exception_with_trace_id)


@app.exception_handler(CodeGenerationError)
async def code_generation_error_handler(
    request: Request, exc: CodeGenerationError
) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


if SNOWFLAKE_NATIVE_APP:
    # Mount Snowflake routes only in native mode.
    from .routers.snowflake import router as snowflake_router

    app.include_router(snowflake_router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_origin_regex=ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS origins: {ORIGINS}")
logger.info(f"CORS origins regex: {ORIGIN_REGEX}")


@app.middleware("http")
async def header_ids_middleware(request: Request, call_next):
    user_id = request.headers.get("X-User-Id")
    trace_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
    completion_id = str(uuid.uuid4())
    request.state.user_id = user_id
    request.state.trace_id = trace_id
    request.state.completion_id = completion_id
    response = await call_next(request)
    response.headers["Access-Control-Expose-Headers"] = "X-Trace-Id, X-Completion-Id"
    response.headers["X-Trace-Id"] = trace_id
    response.headers["X-Completion-Id"] = completion_id
    return response


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    response = await call_next(request)
    logger.info(
        f'{request.scope["client"][0]}:{request.scope["client"][1]} - "{request.method} {request.scope["path"]} {request.scope["scheme"].upper()}/{request.scope["http_version"]}" {response.status_code}',  # noqa: E501
        extra={"status_code": response.status_code, "method": request.method},
    )
    return response


@app.middleware("http")
async def openbb_auth_middleware(request: Request, call_next):
    """Middleware layer that authenticates the user's access token with the
    OpenBB backend, and retrieves additional user information.
    """
    if not AUTH_ENABLED:
        return await call_next(request)
    paths_to_exclude = [
        "/docs",
        "/openapi.json",
        "/status",
        "/health",
        "/healthz",
        "/copilots.json",
        "/agents.json",
    ]
    # Early exit for unprotected endpoints and OPTIONS requests
    if request.url.path in paths_to_exclude or request.method == "OPTIONS":
        response = await call_next(request)
        return response

    if not (
        access_token := request.headers.get("Authorization", "").replace("Bearer ", "")
    ):
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing access token."},
        )

    is_valid, error_detail = await validate_and_sync(access_token)

    if not is_valid:
        return JSONResponse(
            status_code=401,
            content={"detail": error_detail},
        )
    response = await call_next(request)
    return response


@app.get("/agents.json")
async def copilots_json():
    # Build features dict, conditionally including workspace-web-search
    features = {
        "streaming": True,
        "file-upload": True,
        "widget-dashboard-select": True,
        "widget-dashboard-search": True,
        "widget-global-search": True,
        "generative-ui": True,
        "mcp-tools": True,
    }

    if not SNOWFLAKE_NATIVE_APP:
        features["agent-orchestration"] = True

    # Only include workspace-web-search feature if LLM_WEB_SEARCH_ENABLED is True
    # (LLM_WEB_SEARCH_ENABLED is already False when SNOWFLAKE_NATIVE_APP is True)
    # This controls whether users can toggle web search functionality in the UI
    if LLM_WEB_SEARCH_ENABLED:
        features["workspace-web-search"] = {
            "label": "Web Search",
            "default": True,
            "description": "Allows the copilot to search the web.",
        }

    features["prompt-suggestions"] = {
        "label": "Follow-up Suggestions",
        "default": True,
        "description": "Show follow-up prompt suggestions after each response.",
    }

    data = {
        "openbb_ada": {
            "name": "OpenBB Copilot",
            "description": "OpenBB Copilot is the default agent for OpenBB Workspace.",
            "image": "https://openbb-assets.s3.us-east-1.amazonaws.com/docs/copilot/ada.png",
            "endpoints": {"query": "/v1/query"},
            "features": features,
        }
    }
    return JSONResponse(
        status_code=200,
        content=data,
    )


@app.post("/v1/query")
@rate_limit()
async def query_v1(
    request: Request,
    llm_query_request: Annotated[AdaQueryRequest, Depends(get_llm_query_request)],
    copilot_service: Annotated[CopilotService, Depends(get_copilot_service)],
    web_search_llm_service: Annotated[
        WebSearchLlmService, Depends(get_web_search_llm_service)
    ],
    logging_service: Annotated[LoggingService, Depends(get_logging_service())],
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> EventSourceResponse:
    # Check if this is a new conversation (only one message from user)
    is_new_conversation = len(llm_query_request.messages) == 1

    if is_new_conversation:
        logging_service.log_conversation_start()
    else:
        logging_service.log_new_message()

    extra_info = {
        "user_id": request.state.user_id,
        "messages": [
            str(msg.model_dump_json())[:512] for msg in llm_query_request.messages
        ],
        "context": (
            [
                f"{context.name}::{context.description}::{context.metadata}"
                for context in llm_query_request.context
            ]
            if llm_query_request.context
            else None
        ),
        "urls": llm_query_request.urls,
        # TODO:
        # "force_web_search" is deprecated on the client level and should be removed
        # both from Ada codebase and from the openbb-ai model.
        # This change is not breaking (the model is already implementing a factory that
        # defaults to None)
        "force_web_search": llm_query_request.force_web_search,
        "timezone": llm_query_request.timezone,
        "semantic_views": llm_query_request.semantic_views,
        "available_semantic_views_count": (
            len(llm_query_request.available_semantic_views)
            if llm_query_request.available_semantic_views
            else 0
        ),
    }
    extra_info["widgets"] = []
    if llm_query_request.widgets:
        extra_info["widgets"] += [
            {
                "priority": "primary",
                "name": widget.name,
                "widget_id": widget.widget_id,
                "description": widget.description,
            }
            for widget in llm_query_request.widgets.primary or []
        ]
        extra_info["widgets"] += [
            {
                "priority": "secondary",
                "name": widget.name,
                "widget_id": widget.widget_id,
                "description": widget.description,
            }
            for widget in llm_query_request.widgets.secondary or []
        ]
        extra_info["widgets"] += [
            {
                "priority": "extra",
                "message": f"Omitted: {len(llm_query_request.widgets.extra)} extra widgets.",  # noqa: E501
            }
        ]
    logging_service.info("Query request received.", extra=extra_info)

    events_stream: AsyncGenerator[
        CitationCollectionSSE
        | MessageChunkSSE
        | FunctionCallSSE
        | StatusUpdateSSE
        | AppArtifactSSE
        | PromptSuggestionsSSE,
        None,
    ]
    if (
        llm_query_request.force_web_search
        and "workspace-web-search" in llm_query_request.workspace_options
        and LLM_WEB_SEARCH_ENABLED
    ):
        # Only use dedicated web search service if both force_web_search is true
        # AND web search is enabled in the workspace options AND web search is enabled
        # in the backend.
        events_stream = web_search_llm_service.query(
            messages=cast(list[LlmClientMessage], llm_query_request.messages),
        )
    else:
        if llm_query_request.urls and not URL_RETRIEVAL_ENABLED:
            raise HTTPException(status_code=501, detail="URL retrieval is disabled")
        if llm_query_request.force_web_search and not LLM_WEB_SEARCH_ENABLED:
            raise HTTPException(status_code=501, detail="Web search is disabled")

        events_stream = copilot_service.query(
            messages=llm_query_request.messages,
            context=llm_query_request.context,
            urls=llm_query_request.urls,
            tools=llm_query_request.tools,
        )

    return EventSourceResponse(
        content=(event.model_dump(exclude_none=True) async for event in events_stream),
        media_type="text/event-stream",
    )


@app.post("/v1/generate/widget_info")
async def generate_widget_title_and_description_v1(
    widget_generation_request: WidgetTitleDescriptionRequest,
    widget_metadata_generation_service: Annotated[
        WidgetMetadataGenerationService, Depends(WidgetMetadataGenerationService)
    ],
    logging_service: Annotated[LoggingService, Depends(get_logging_service())],
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> WidgetTitleDescriptionResponse:
    extra_info = {
        "widget_data": widget_generation_request.widget_data[:512],
    }
    logging_service.info("Generate widget metadata request received.", extra=extra_info)

    return await widget_metadata_generation_service.generate_title_and_description(
        widget_generation_request=widget_generation_request
    )


@app.post("/v1/generate/skill_info")
async def generate_skill_info_v1(
    skill_generation_request: SkillGenerationRequest,
    skill_generation_service: Annotated[
        SkillGenerationService, Depends(SkillGenerationService)
    ],
    logging_service: Annotated[LoggingService, Depends(get_logging_service())],
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> SkillGenerationResponse:
    extra_info = {
        "conversation_messages": len(skill_generation_request.conversation),
        "name_hint": skill_generation_request.name_hint,
    }
    logging_service.info("Generate skill info request received.", extra=extra_info)

    return await skill_generation_service.generate_skill(
        skill_generation_request=skill_generation_request
    )


# TODO: Think about how to use custom OpenAI API key for this endpoint.
# (We cannot accept multi-form data *and* a JSON body.)
@app.post("/v1/generate/widget_info/file")
async def generate_widget_file_title_and_description_v1(
    widget_metadata_generation_service: Annotated[
        WidgetMetadataGenerationService, Depends(WidgetMetadataGenerationService)
    ],
    logging_service: Annotated[LoggingService, Depends(LoggingService)],
    file: UploadFile,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> WidgetTitleDescriptionResponse:
    if file.size and file.content_type and "image" in file.content_type:
        extra_info = {
            "widget_data": f"Image file {file.filename} of size {get_human_readable_size(file.size)}."  # noqa: E501
        }
        logging_service.info(
            "Generate widget metadata request received.", extra=extra_info
        )
        return await widget_metadata_generation_service.generate_title_and_description_for_image(  # noqa: E501
            file=file
        )
    else:
        widget_file_data = await get_file_details(file)

    extra_info = {
        "widget_data": widget_file_data.widget_data[:256],
    }
    logging_service.info("Generate widget metadata request received.", extra=extra_info)

    return await widget_metadata_generation_service.generate_title_and_description(
        widget_generation_request=WidgetTitleDescriptionRequest(
            widget_data=widget_file_data.widget_data,
            name=widget_file_data.filename,
            metadata={
                "columns": widget_file_data.columns,
                "index": widget_file_data.index,
                "filename": widget_file_data.filename,
            },
        )
    )


if not constants.SNOWFLAKE_NATIVE_APP:

    @app.post("/v1/generate/code", response_model=CodeGenerationResponse)
    async def generate_code(
        request: CodeGenerationRequest,
        sql_suggestion_service: Annotated[
            SqlQueryGenerationService,
            Depends(get_sql_suggestion_service),
        ],
        editor_content_generation_service: Annotated[
            EditorContentGenerationService,
            Depends(get_editor_content_generation_service),
        ],
        logging_service: Annotated[LoggingService, Depends(get_logging_service())],
        credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    ) -> CodeGenerationResponse:
        """Generate editor code/content for /v1/generate/code."""
        logging_service.info(
            "Code generation request received.",
            extra={
                "widget_uuid": request.widget_uuid,
                "language": request.language,
                "user_prompt": request.user_prompt[:256],
            },
        )

        if request.language == "sql":
            return await sql_suggestion_service.generate_sql_suggestion(
                user_request=request.user_prompt,
                sql_schema=request.sql_schema,
                current_sql=request.current_code,
                data_sample=request.data_sample,
            )

        return await editor_content_generation_service.generate_content_suggestion(
            user_request=request.user_prompt,
            language=cast(EditorContentLanguage, request.language),
            current_content=request.current_code,
        )


@app.post("/v1/generate/chat/title")
async def generate_chat_title(
    llm_query_request: AdaQueryRequest,
    chat_title_generation_service: Annotated[
        ChatTitleGenerationService, Depends(get_chat_title_generation_service)
    ],
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> str:
    """Generate a chat title based on the conversation."""

    return await chat_title_generation_service.generate_chat_title(
        messages=llm_query_request.messages,
    )


@app.post("/v1/enhance_prompt")
@rate_limit()
async def enhance_prompt(
    request: Request,
    llm_query_request: Annotated[AdaQueryRequest, Depends(get_llm_query_request)],
    prompt_enhancement_service: Annotated[
        PromptEnhancementService, Depends(get_prompt_enhancement_service)
    ],
    logging_service: Annotated[LoggingService, Depends(get_logging_service())],
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> str:
    """Enhance a user prompt to make it clearer and more specific."""
    extra_info = {
        "user_id": request.state.user_id,
        "messages": [
            str(msg.model_dump_json())[:512] for msg in llm_query_request.messages
        ],
    }
    logging_service.info("Prompt enhancement request received.", extra=extra_info)

    return await prompt_enhancement_service.enhance_prompt(
        messages=llm_query_request.messages,
        context=llm_query_request.context,
        widgets=llm_query_request.widgets,
        tools=llm_query_request.tools,
    )


@app.post("/v1/generate/dashboard/title")
async def generate_dashboard_title(
    llm_dashboard_title_request: Annotated[
        LlmDashboardTitleGenerationRequest,
        Depends(get_llm_dashboard_title_generation_request),
    ],
    dashboard_title_generation_service: Annotated[
        DashboardTitleGenerationService, Depends(get_dashboard_title_generation_service)
    ],
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> str:
    """Generate a dashboard title based on the present widgets."""

    return await dashboard_title_generation_service.generate_dashboard_title(
        widgets=llm_dashboard_title_request.widgets or [],
    )


@app.get("/status")
@app.get("/health")
@app.get("/healthz")
async def status():
    """Healthcheck endpoint."""
    return JSONResponse(status_code=200, content={"status": "OK"})
