import asyncio
import json
import re
import traceback
from contextlib import suppress
from typing import Any, AsyncGenerator, Callable, Literal, Sequence, cast
from uuid import UUID

import httpx
import openai
from fastapi import HTTPException
from magentic import (
    AssistantMessage,
    AsyncStreamedResponse,
    AsyncStreamedStr,
    FunctionCall,
    FunctionResultMessage,
    SystemMessage,
    UserMessage,
    chatprompt,
)
from magentic.chat_model.message import Message
from openbb_ai.models import (
    SSE,
    AgentTool,
    Citation,
    CitationCollection,
    CitationCollectionSSE,
    ClientArtifact,
    ClientCommandResult,
    ClientFunctionCallError,
    FunctionCallResponse,
    FunctionCallSSE,
    FunctionCallSSEData,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    MessageArtifactSSE,
    MessageChunkSSE,
    MessageChunkSSEData,
    PromptSuggestionsSSE,
    PromptSuggestionsSSEData,
    StatusUpdateSSE,
    StatusUpdateSSEData,
    WidgetCollection,
    WorkspaceState,
)
from pydantic import BaseModel, ValidationError

from . import constants
from .errors import ContextLimitExceededError, ToolLimitExceededError
from .models import (
    DEFAULT_SSRM_INPUT_ARGS,
    AppArtifactSSE,
    DataContent,
    DataFileReferences,
    DataSourceRequestPayload,
    DeferredFunctionCall,
    Document,
    ParsedContext,
    PythonCodeFunctionCallResult,
    PythonCodeGenerationResult,
    PythonWidgetContext,
    RawContext,
    SkillCatalogEntry,
    SkillPayload,
    SourceInfo,
    SqlQueryFunctionCallResult,
    SqlQueryGenerationResult,
    SqlWidgetContext,
    StructuredContext,
    UnstructuredContext,
    WebContext,
    WidgetQueryRequest,
)
from .services import (
    CitationService,
    ClientFunctionCallService,
    ContextService,
    CopilotDataService,
    DocumentService,
    LoggingService,
    McpDataService,
    NativeFunctionCallService,
    PromptEnhancementService,
    PythonCodeGenerationService,
    SqlQueryGenerationService,
    TemplateService,
    UrlRetrievalService,
)
from .services._logging import logfire, posthog_client
from .services.mcp_data import normalize_jsonish
from .utils.ai import (
    CONTEXT_LIMIT_BY_MODEL,
    CONTEXT_LIMIT_SAFETY_FACTOR,
    CONTEXT_LIMIT_UNDEFINED_MODEL,
    get_llm,
)
from .utils.mcp_diagnostics import (
    build_emitted_mcp_tool_call,
    restore_last_failed_mcp_call,
    summarize_mcp_tools,
)
from .utils.utils import (
    build_context_uuid,
    flatten_and_format_dict,
    handle_openai_error,
    instrument_async_generator,
    masked_partial,
    retry_on_exception,
    sanitize_str,
    wrapped_partial,
)

type NativeFunctionCallEvent = (
    list[Message[Any]]
    | StatusUpdateSSE
    | MessageChunkSSE
    | AppArtifactSSE
    | CitationCollectionSSE
    | SqlQueryGenerationResult
    | SqlQueryFunctionCallResult
    | PythonCodeGenerationResult
    | PythonCodeFunctionCallResult
    | FunctionCallResponse
)

type AdaSSE = SSE | AppArtifactSSE


class CopilotService:
    """Handle the orchestration of the Copilot."""

    MAX_CALLS = 30
    MAX_PARALLEL_NATIVE_CALLS = 4
    OPENAI_MAX_TOOLS_PER_REQUEST = 128
    _DEFERRED_FUNCTION_CALLS_KEY = "deferred_function_calls"
    # Keep this allowlist limited to native tools that cannot produce a client
    # boundary. Expanding it to tools that may yield FunctionCallResponse /
    # SqlQueryGenerationResult / PythonCodeGenerationResult requires a richer
    # carry-over protocol so unreplayed siblings in the same parallel batch are
    # preserved without duplicate execution.
    _PARALLEL_SAFE_NATIVE_FUNCTION_NAMES = {"llm_query_structured_data"}
    # Map mergeable client function names to the list argument that should be
    # concatenated when multiple same-type calls are folded into one boundary event.
    _MERGEABLE_CLIENT_FUNCTION_ARGUMENTS = {
        "llm_query_widgets": "widget_queries",
        "llm_update_widget_in_dashboard": "widget_queries",
        "llm_query_extra_widgets": "search_queries",
        "llm_add_widget_to_dashboard": "search_queries",
    }
    _MERGEABLE_CLIENT_FUNCTION_NAMES = set(_MERGEABLE_CLIENT_FUNCTION_ARGUMENTS)
    _SUGGESTIONS_OPEN_PATTERN = re.compile(r"<suggestions\b[^>]*>", re.IGNORECASE)
    _SUGGESTIONS_CLOSE_PATTERN = re.compile(r"</suggestions>", re.IGNORECASE)
    _SUGGESTION_ITEM_PATTERN = re.compile(
        r"<suggestion\b[^>]*>([\s\S]*?)</suggestion>",
        re.IGNORECASE,
    )

    def __init__(
        self,
        user_id: str,
        document_service: DocumentService,
        context_service: ContextService,
        template_service: TemplateService,
        url_retrieval_service: UrlRetrievalService,
        copilot_data_service: CopilotDataService,
        logging_service: LoggingService,
        client_function_call_service: ClientFunctionCallService,
        native_function_call_service: NativeFunctionCallService,
        citation_service: CitationService,
        mcp_data_service: McpDataService,
        openai_api_key: str | None,
        workspace_options: dict[str, Any] | None = None,
        workspace_state: WorkspaceState | None = None,
        prompt_enhancement_service: PromptEnhancementService | None = None,
        sql_query_generation_service: SqlQueryGenerationService | None = None,
        python_code_generation_service: PythonCodeGenerationService | None = None,
        skills_catalog: list[SkillCatalogEntry] | None = None,
        selected_skills: list[SkillPayload] | None = None,
    ):
        self._user_id = user_id
        self._document_service = document_service
        self._context_service = context_service
        self._template_service = template_service
        self._url_retrieval_service = url_retrieval_service
        self._copilot_data_service = copilot_data_service
        self._logging_service = logging_service
        self._client_function_call_service = client_function_call_service
        self._native_function_call_service = native_function_call_service
        self._citation_service = citation_service
        self._mcp_data_service = mcp_data_service
        self._prompt_enhancement_service = prompt_enhancement_service
        self._sql_query_generation_service = sql_query_generation_service
        self._python_code_generation_service = python_code_generation_service
        self._temperature = 0
        self._openai_api_key = openai_api_key
        self._workspace_options = workspace_options or {}
        self._workspace_state = workspace_state
        self._skills_catalog = skills_catalog or []
        self._selected_skills = selected_skills or []
        self._flat_mcp_functions: list[Callable] = []
        self._pending_prompt_suggestions: list[str] = []

    def _parse_prompt_suggestions(self, text: str) -> tuple[str, list[str]]:
        """Extract hidden follow-up suggestions from a final response."""
        open_match = self._SUGGESTIONS_OPEN_PATTERN.search(text)
        if not open_match:
            return text, []

        block_start = open_match.end()
        rest = text[block_start:]
        close_match = self._SUGGESTIONS_CLOSE_PATTERN.search(rest)
        block = rest[: close_match.start()] if close_match else rest
        after_block = rest[close_match.end() :] if close_match else ""
        clean_text = f"{text[: open_match.start()]}{after_block}".rstrip()

        if not close_match:
            return clean_text, []

        open_count = len(re.findall(r"<suggestion\b[^>]*>", block, re.IGNORECASE))
        close_count = len(re.findall(r"</suggestion>", block, re.IGNORECASE))
        if open_count != close_count:
            return clean_text, []

        suggestions: list[str] = []

        def collect_suggestion(match: re.Match[str]) -> str:
            suggestion = match.group(1).strip()
            if suggestion:
                suggestions.append(suggestion)
            return ""

        consumed = self._SUGGESTION_ITEM_PATTERN.sub(collect_suggestion, block)
        if consumed.strip():
            return clean_text, []
        return clean_text, suggestions

    def _log_available_mcp_tools_snapshot(
        self, tools: Sequence[AgentTool] | None
    ) -> None:
        if not tools:
            return

        tool_summaries, truncated = summarize_mcp_tools(tools)
        self._logging_service.debug(
            "available_mcp_tools_snapshot",
            extra={
                "event": "available_mcp_tools_snapshot",
                "tool_count": len(tools),
                "truncated": truncated,
                "tools": tool_summaries,
            },
        )

    def _raise_if_openai_tool_limit_exceeded(
        self,
        *,
        total_tool_count: int,
        mcp_tool_count: int,
    ) -> None:
        if total_tool_count <= self.OPENAI_MAX_TOOLS_PER_REQUEST:
            return

        raise ToolLimitExceededError(
            total_tool_count=total_tool_count,
            threshold=self.OPENAI_MAX_TOOLS_PER_REQUEST,
            mcp_tool_count=mcp_tool_count,
        )

    def _build_tool_limit_warning_sse(
        self,
        *,
        total_tool_count: int,
        mcp_tool_count: int,
        threshold: int,
    ) -> StatusUpdateSSE:
        non_mcp_tool_count = max(total_tool_count - mcp_tool_count, 0)
        return StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="WARNING",
                message=(
                    "Too many tools are enabled. "
                    f"Requests are capped at {threshold} tools."
                ),
                details=[
                    {
                        "Detail": (
                            f"This request would send {total_tool_count} tools "
                            f"total ({mcp_tool_count} MCP tools and "
                            f"{non_mcp_tool_count} built-in tools). "
                            "Reduce the selected MCP tools and try again."
                        )
                    }
                ],
            )
        )

    def _hydrate_loaded_skills_from_messages(
        self,
        messages: Sequence[LlmClientFunctionCallResultMessage | LlmClientMessage],
    ) -> None:
        """Promote previously loaded skills from message history into active state."""
        existing_slugs = {skill.slug for skill in self._selected_skills}
        restored_count = 0

        for message in messages:
            if not isinstance(message, LlmClientFunctionCallResultMessage):
                continue
            if message.function != "get_skill_content":
                continue

            for item in message.data:
                if not isinstance(item, ClientCommandResult):
                    continue
                if item.status != "success" or not item.data:
                    continue

                skill_data = item.data.get("skill")
                if not isinstance(skill_data, dict):
                    continue

                slug = skill_data.get("slug")
                description = skill_data.get("description", "")
                content_markdown = skill_data.get("contentMarkdown", "")
                if not isinstance(slug, str) or not slug:
                    continue

                self._remove_fetched_skill_from_catalog(slug)

                if slug in existing_slugs:
                    continue
                if (
                    not isinstance(content_markdown, str)
                    or not content_markdown.strip()
                ):
                    continue

                self._selected_skills.append(
                    SkillPayload(
                        slug=slug,
                        description=(
                            description if isinstance(description, str) else ""
                        ),
                        contentMarkdown=content_markdown,
                        source="model_selected",
                    )
                )
                existing_slugs.add(slug)
                restored_count += 1

        if restored_count:
            self._logging_service.info(
                "Restored %d loaded skill(s) from message history",
                restored_count,
            )

    @staticmethod
    def _parse_enhanced_query(text: str) -> str | None:
        """Extract the enhanced query text from an enhancement result."""
        prefix = "Enhanced query:"
        if not text.startswith(prefix):
            return None

        query = text[len(prefix) :].strip()
        if "\n\n" in query:
            query = query.split("\n\n", 1)[0].strip()
        return query or None

    def _extract_current_turn_enhanced_query(
        self, messages: Sequence[Message[Any] | AssistantMessage]
    ) -> str | None:
        """Return the latest enhanced query emitted in the active turn."""
        for message in reversed(messages):
            if isinstance(message, UserMessage):
                return None

            text: str | None = None
            if isinstance(message, FunctionResultMessage):
                text = message.content
            elif isinstance(message, AssistantMessage) and isinstance(
                message.content, str
            ):
                text = message.content

            if isinstance(text, str):
                if enhanced_query := self._parse_enhanced_query(text):
                    return enhanced_query

        return None

    def _current_turn_has_prompt_enhancement(
        self, messages: Sequence[Message[Any] | AssistantMessage]
    ) -> bool:
        """Check whether prompt enhancement already succeeded for the active turn."""
        return self._extract_current_turn_enhanced_query(messages) is not None

    def _current_turn_has_llm_think(
        self, messages: Sequence[Message[Any] | AssistantMessage]
    ) -> bool:
        """Check whether planning has already happened in the active turn."""
        think_function_name = self._native_function_call_service._llm_think.__name__
        for message in reversed(messages):
            if isinstance(message, UserMessage):
                return False

            function_call = None
            if isinstance(message, FunctionResultMessage):
                function_call = message.function_call
            elif isinstance(message, AssistantMessage) and isinstance(
                message.content, FunctionCall
            ):
                function_call = message.content

            if (
                isinstance(function_call, FunctionCall)
                and function_call.function.__name__ == think_function_name
            ):
                return True

        return False

    @instrument_async_generator("CopilotService.query")
    async def query(
        self,
        messages: list[LlmClientFunctionCallResultMessage | LlmClientMessage],
        context: list[RawContext] | None = None,
        urls: list[str] | None = None,
        tools: list[AgentTool] | None = None,
    ) -> AsyncGenerator[AdaSSE, None]:
        # Tools and their schemas are managed entirely by the frontend
        # Backend just passes them through to the system prompt
        self._pending_prompt_suggestions = []
        logfire.info(
            "openbb_trace_id: {trace_id}", trace_id=self._logging_service.trace_id
        )
        try:
            if context:
                self._context_service.load_explicit_context(context_elements=context)

                # Note: We intentionally do NOT register citations for
                # explicit context artifacts (charts, tables, HTML generated
                # by the AI in previous turns). These are still loaded as SQL
                # tables for querying, but should not produce citation tags
                # in the response. Citations are reserved for widget data
                # and uploaded documents/files.

            messages = messages or []
            self._hydrate_loaded_skills_from_messages(messages)

            if terminal_failure_message := self._get_terminal_data_load_failure_message(
                messages=messages
            ):
                yield MessageChunkSSE(
                    data=MessageChunkSSEData(delta=terminal_failure_message)
                )
                return

            # Pre-register flat MCP tools so that history replay in
            # _format_chat_messages can remap execute_agent_tool entries
            # to the correct flat function.  The built functions are
            # reused later by _compose_chain.
            if tools:
                self._flat_mcp_functions = (
                    self._client_function_call_service.build_flat_mcp_tools(tools)
                )
            else:
                self._flat_mcp_functions = []

            chat_messages: list[Message | AssistantMessage] = []
            async for event in self._format_chat_messages(messages):
                if isinstance(event, StatusUpdateSSE):
                    yield event
                else:
                    chat_messages.extend(event)

            if (
                messages
                and isinstance(messages[-1], LlmClientFunctionCallResultMessage)
                and messages[-1].extra_state
                and "intermediate_context" in messages[-1].extra_state
            ):
                intermediate_context = messages[-1].extra_state["intermediate_context"]
                if intermediate_context:
                    self._logging_service.info("Restoring intermediate context")
                    # Insert as AssistantMessage before the client function call
                    # so the LLM sees it already "said" this
                    insert_pos = len(chat_messages) - 2
                    if insert_pos < 0:
                        insert_pos = 0
                    chat_messages.insert(
                        insert_pos,
                        AssistantMessage(sanitize_str(intermediate_context)),
                    )

            # Restore intermediate citations from extra_state so the citation
            # service knows about citations registered in previous requests
            if (
                messages
                and isinstance(messages[-1], LlmClientFunctionCallResultMessage)
                and messages[-1].extra_state
                and "intermediate_citations" in messages[-1].extra_state
            ):
                intermediate_citations_data = messages[-1].extra_state[
                    "intermediate_citations"
                ]
                if intermediate_citations_data:
                    self._logging_service.info(
                        "Restoring %d intermediate citations",
                        len(intermediate_citations_data),
                    )
                    for c_data in intermediate_citations_data:
                        citation = Citation.model_validate(c_data)
                        self._citation_service.add_citation(citation)

            # Restore intermediate artifacts from extra_state so native
            # function call results (charts, tables) survive round-trips
            if (
                messages
                and isinstance(messages[-1], LlmClientFunctionCallResultMessage)
                and messages[-1].extra_state
                and "intermediate_artifacts" in messages[-1].extra_state
            ):
                artifacts_data = messages[-1].extra_state["intermediate_artifacts"]
                if artifacts_data:
                    self._logging_service.info(
                        "Restoring %d intermediate artifacts",
                        len(artifacts_data),
                    )
                    self._context_service.restore_roundtrip_artifacts(artifacts_data)

            (
                prompt_semantic_views,
                _prompt_semantic_reason,
            ) = self._native_function_call_service.get_prompt_semantic_context()

            # Track the original message count so we know which messages are new
            original_chat_messages_count = len(chat_messages)
            deferred_function_call_specs = self._get_deferred_function_call_specs(
                messages
            )

            web_pages = (
                await self._url_retrieval_service.retrieve_urls(urls) if urls else None
            )
            self._log_available_mcp_tools_snapshot(tools)
            last_failed_mcp_call, last_mcp_error_summary = restore_last_failed_mcp_call(
                messages,
                normalize_content=normalize_jsonish,
                is_mcp_error=self._mcp_data_service._is_mcp_error,
                get_mcp_error_content=self._mcp_data_service._get_mcp_error_content,
            )
            if last_failed_mcp_call:
                self._logging_service.debug(
                    "restored_mcp_retry_context",
                    extra={
                        "event": "restored_mcp_retry_context",
                        "tool_name": last_failed_mcp_call.get("tool_name"),
                        "server_id": last_failed_mcp_call.get("server_id"),
                        "missing_required_args": last_failed_mcp_call.get(
                            "missing_required_args", []
                        ),
                        "previous_mcp_error": last_mcp_error_summary,
                    },
                )
            if web_pages is not None:
                web_citations = [wp.citation for wp in web_pages if wp.citation]
                for citation in web_citations:
                    self._citation_service.add_citation(citation)

            # If we have to continue with a partially-completed function
            # call, we need to continue executing it before we kick-off the
            # execution loop and continue where we left off.
            if self._client_function_call_service.must_continue_with_function_call(
                messages
            ):
                self._logging_service.info(
                    "Continuing with partially-completed function call..."
                )
                async for (
                    fc_event
                ) in self._client_function_call_service.continue_with_function_call(
                    messages
                ):
                    if isinstance(fc_event, StatusUpdateSSE):
                        yield fc_event
                    elif isinstance(fc_event, FunctionCallSSE):
                        yield self._augment_client_function_call_event(
                            client_fc_event=fc_event,
                            chat_messages=chat_messages,
                            original_chat_messages_count=original_chat_messages_count,
                            deferred_function_call_specs=deferred_function_call_specs,
                        )
                        return  # We return here to the client.
                    # If we need to append messages (eg. when an error occurs)
                    elif isinstance(fc_event, list) and all(
                        isinstance(item, Message) for item in fc_event
                    ):
                        chat_messages.extend(fc_event)
                    else:
                        raise TypeError(f"Unexpected event type: {type(fc_event)}")

            widget_collection = self._copilot_data_service.get_widget_collection()
            app_widget_collection = (
                WidgetCollection(
                    primary=list(widget_collection.primary or []),
                    secondary=list(widget_collection.secondary or []),
                    extra=list(widget_collection.extra or []),
                )
                if widget_collection
                else None
            )

            # RAG-based filtering of extra widgets when global search is enabled
            if (
                "widget-global-search" in self._workspace_options
                and widget_collection
                and widget_collection.extra
            ):
                # Preserve Python-capable widgets BEFORE semantic search replaces extra
                # These should always be available when global data is enabled
                python_capable_widgets = [
                    w
                    for w in widget_collection.extra
                    if any(
                        getattr(p, "language", None) == "python"
                        for p in (w.params or [])
                    )
                ]
                # Extract all user queries from chat messages
                user_queries = [
                    msg.content
                    for msg in chat_messages
                    if isinstance(msg, UserMessage) and msg.content
                ]

                if user_queries:
                    relevant_widgets = (
                        await self._copilot_data_service.search_relevant_extra_widgets(
                            queries=user_queries, k=10
                        )
                    )

                    # Log what vector search found
                    self._logging_service.info(
                        "Vector search found %d relevant widgets: %s",
                        len(relevant_widgets),
                        [w.widget_id for w in relevant_widgets],
                    )

                    # Merge: semantic search results + preserved Python widgets
                    # Use UUIDs to avoid duplicates
                    seen_uuids = {str(w.uuid) for w in relevant_widgets}
                    for pw in python_capable_widgets:
                        if str(pw.uuid) not in seen_uuids:
                            relevant_widgets.append(pw)
                            seen_uuids.add(str(pw.uuid))

                    # Replace extra widgets with only the relevant ones for the prompt
                    widget_collection.extra = relevant_widgets

            # Extract SQL-enabled widgets early for system prompt
            # Include extra widgets only when global search is enabled
            include_extra_sql = "widget-global-search" in self._workspace_options

            sql_widgets = self._get_sql_enabled_widgets(
                widget_collection, include_extra=include_extra_sql
            )

            # Log which SQL widgets were found for debugging
            if sql_widgets:
                widget_ids = [w.widget_id for w in sql_widgets]
                self._logging_service.info(
                    "SQL-enabled widgets included in prompt: %s", widget_ids
                )

            # Extract Python code-enabled widgets early for system prompt
            # Include extra widgets only when global search is enabled (same as SQL)
            python_widgets = self._get_code_enabled_widgets(
                widget_collection, include_extra=include_extra_sql
            )

            # Remove SQL-enabled widgets from extra collection to avoid duplication
            # SQL widgets will be shown in the dedicated "SQL-Enabled Widgets" section
            # with full schemas, so we don't need them in the "Extra widgets" section
            if sql_widgets and widget_collection and widget_collection.extra:
                sql_widget_uuids = {w.widget_uuid for w in sql_widgets}
                widget_collection.extra = [
                    w
                    for w in widget_collection.extra
                    if str(w.uuid) not in sql_widget_uuids
                ]

            # Remove Python-enabled widgets from extra collection to avoid duplication
            if python_widgets and widget_collection and widget_collection.extra:
                python_widget_uuids = {w.widget_uuid for w in python_widgets}
                widget_collection.extra = [
                    w
                    for w in widget_collection.extra
                    if str(w.uuid) not in python_widget_uuids
                ]

            # If the previous request stopped at a client function call, the
            # remaining tool calls come back via extra_state. We restore them
            # before making a fresh LLM call so the logical tool queue continues
            # where it left off.
            function_registry = self._build_function_registry(
                messages=chat_messages,
                documents=self._document_service.documents,
                tools=tools,
                original_messages=messages,
                original_context=context,
                widget_collection=widget_collection,
                app_widget_collection=app_widget_collection,
                sql_widgets=sql_widgets,
                python_widgets=python_widgets,
            )
            self._raise_if_openai_tool_limit_exceeded(
                total_tool_count=len(function_registry),
                mcp_tool_count=len(self._flat_mcp_functions),
            )
            pending_function_calls = self._restore_deferred_function_calls(
                deferred_specs=deferred_function_call_specs,
                function_registry=function_registry,
            )
            queried_file_widget_uuids = self._get_previously_queried_file_widget_uuids(
                messages
            )

            call_count = 0
            response = None
            exit_loop = False
            step_id: str | None = None
            # Collect citations from native function calls to yield at end of stream
            # This ensures citations appear AFTER the chart/message content
            deferred_citations: list[Citation] = []
            pending_response_artifacts: list[ClientArtifact] = []
            visible_response_after_artifact = False

            while not exit_loop and call_count < self.MAX_CALLS:
                early_exit = False
                function_calls_to_process = pending_function_calls
                pending_function_calls = []

                if not function_calls_to_process:
                    # Compose the chain
                    step_id = (
                        f"{str(self._logging_service.trace_id)[:8]}-step-"
                        f"{call_count + 1}"
                    )
                    self._logging_service.debug(
                        "agent_step_start",
                        extra={
                            "event": "agent_step_start",
                            "step_id": step_id,
                            "call_index": call_count + 1,
                            "chat_message_count": len(chat_messages),
                            "available_mcp_tool_count": len(tools or []),
                            "previous_mcp_error": last_mcp_error_summary,
                        },
                    )
                    self._logging_service.debug("Preparing next call...")
                    run_agent_step = await self._compose_chain(
                        chat_messages=chat_messages,
                        documents=self._document_service.documents,
                        web_pages=web_pages,
                        tools=tools,
                        original_messages=messages,
                        original_context=context,
                        widget_collection=widget_collection,
                        app_widget_collection=app_widget_collection,
                        sql_widgets=sql_widgets,
                        python_widgets=python_widgets,
                        semantic_views_for_prompt=prompt_semantic_views,
                    )

                    # Run the agent (detailed request diagnostics logged in
                    # _compose_chain)
                    self._logging_service.info(
                        "Executing LLM call #%d (messages: %d)",
                        call_count + 1,
                        len(chat_messages),
                    )
                    with logfire.span(
                        "copilot.agent_step",
                        step_id=step_id,
                        call_index=call_count + 1,
                    ):
                        response = await run_agent_step()

                    call_count += 1

                    # magentic wraps mixed text + tool call responses in
                    # AsyncStreamedResponse. We stream any leading text
                    # immediately and collect every emitted tool call into the
                    # queue executor below.
                    if isinstance(response, AsyncStreamedResponse):
                        self._logging_service.debug("Handling AsyncStreamedResponse")
                        async for item in response:
                            if isinstance(item, AsyncStreamedStr):
                                if not function_calls_to_process:
                                    async for (
                                        stream_event
                                    ) in self._stream_copilot_events(item):
                                        if pending_response_artifacts and isinstance(
                                            stream_event,
                                            (MessageChunkSSE, MessageArtifactSSE),
                                        ):
                                            visible_response_after_artifact = True
                                        yield stream_event
                                else:
                                    # We now preserve every tool call in the
                                    # batch. The only unsupported shape left is
                                    # interleaved trailing text after the tool
                                    # queue has started.
                                    self._logging_service.warning(
                                        "Discarding streamed text emitted "
                                        "after function call %s",
                                        function_calls_to_process[-1].function.__name__,
                                    )
                                    await item.to_string()
                            else:
                                function_calls_to_process.append(item)

                        if not function_calls_to_process:
                            async for final_event in self._finalize_copilot_stream(
                                deferred_citations
                            ):
                                yield final_event
                            return
                    elif isinstance(response, FunctionCall):
                        function_calls_to_process = [response]

                    self._logging_service.debug(
                        "agent_step_response",
                        extra={
                            "event": "agent_step_response",
                            "step_id": step_id,
                            "call_index": call_count,
                            "response_type": type(response).__name__,
                            "function_name": (
                                function_calls_to_process[0].function.__name__
                                if function_calls_to_process
                                else None
                            ),
                            "function_count": len(function_calls_to_process),
                        },
                    )

                queue_index = 0
                while queue_index < len(function_calls_to_process):
                    current_call = function_calls_to_process[queue_index]
                    self._logging_service.debug(
                        "Function call: %s with args %s",
                        current_call.function.__name__,
                        current_call.arguments,
                    )
                    repeated_file_widget_queries = self._get_repeated_file_queries(
                        function_call=current_call,
                        previously_queried_file_widget_uuids=queried_file_widget_uuids,
                    )
                    raw_widget_queries = current_call.arguments.get("widget_queries")
                    if (
                        repeated_file_widget_queries
                        and isinstance(raw_widget_queries, list)
                        and len(repeated_file_widget_queries) == len(raw_widget_queries)
                    ):
                        self._logging_service.warning(
                            "Blocking repeated file widget request for widgets=%s",
                            [
                                str(widget_query.widget_uuid)
                                for widget_query in repeated_file_widget_queries
                            ],
                        )
                        chat_messages.extend(
                            self._build_repeated_file_widget_retry_messages(
                                function_call=current_call,
                                repeated_widget_queries=repeated_file_widget_queries,
                            )
                        )
                        queue_index += 1
                        continue
                    fn_name = current_call.function.__name__
                    # Check if this is an MCP tool call — either the legacy
                    # meta-tool or a dynamically-generated flat tool.
                    _cfs = self._client_function_call_service
                    is_execute_agent_tool_call = (
                        fn_name == _cfs.llm_execute_agent_tool.__name__
                        or fn_name in _cfs._dynamic_mcp_tool_names
                    )
                    if is_execute_agent_tool_call:
                        response_arguments = cast(
                            dict[str, Any], current_call.arguments
                        )
                        emitted_tool_call, emitted_tool_args_hash = (
                            build_emitted_mcp_tool_call(
                                function_name=fn_name,
                                response_arguments=response_arguments,
                                get_original_tool_info=_cfs.get_original_tool_info,
                                make_stable_hash=self._logging_service.make_stable_hash,
                            )
                        )
                        self._logging_service.debug(
                            "agent_tool_call_selected",
                            extra={
                                "event": "agent_tool_call_selected",
                                **({"step_id": step_id} if step_id else {}),
                                "previous_mcp_error": last_mcp_error_summary,
                                **emitted_tool_call,
                            },
                        )
                        if (
                            last_failed_mcp_call
                            and last_failed_mcp_call.get("tool_name")
                            == emitted_tool_call["tool_name"]
                            and last_failed_mcp_call.get("server_id")
                            == emitted_tool_call["server_id"]
                        ):
                            args_changed = (
                                last_failed_mcp_call.get("tool_args_hash")
                                != emitted_tool_args_hash
                            )
                            log_event = (
                                "mcp_retry_attempt_emitted"
                                if args_changed
                                else "repeated_invalid_tool_call"
                            )
                            log_extra = {
                                "event": log_event,
                                **({"step_id": step_id} if step_id else {}),
                                "previous_tool_attempt_id": last_failed_mcp_call.get(
                                    "tool_attempt_id"
                                ),
                                "previous_mcp_error": last_mcp_error_summary,
                                "args_changed": args_changed,
                                "missing_required_args": last_failed_mcp_call.get(
                                    "missing_required_args", []
                                ),
                                **emitted_tool_call,
                            }
                            if args_changed:
                                self._logging_service.debug(log_event, extra=log_extra)
                            else:
                                self._logging_service.warning(
                                    log_event, extra=log_extra
                                )

                            # Block repeated invalid tool calls with
                            # identical args to prevent infinite loops.
                            if not args_changed:
                                missing = last_failed_mcp_call.get(
                                    "missing_required_args", []
                                )
                                error_msg = (
                                    f"Blocked: You already called "
                                    f"`{emitted_tool_call['tool_name']}` "
                                    "with the same arguments and it failed. "
                                    "Previous error: "
                                    f"{last_mcp_error_summary}"
                                )
                                if missing:
                                    error_msg += (
                                        f"\nMissing required arguments: {missing}. "
                                        "You MUST include these in `tool_args`."
                                    )
                                error_msg += (
                                    "\nEither provide corrected `tool_args` with ALL "
                                    "required parameters, or explain to the user why "
                                    "you cannot proceed."
                                )
                                self._logging_service.warning(
                                    "Blocking repeated invalid MCP tool call "
                                    "for %s — injecting error",
                                    emitted_tool_call["tool_name"],
                                )
                                chat_messages.append(
                                    self._build_blocked_tool_feedback_message(
                                        current_call,
                                        error_msg,
                                    )
                                )
                                queue_index += 1
                                continue

                    if (
                        current_call.function.__name__
                        == self._native_function_call_service.llm_complete.__name__
                    ):
                        # Exhaust the generator for it to be considered "called"
                        async for _ in current_call():
                            pass
                        if (
                            pending_response_artifacts
                            and not visible_response_after_artifact
                        ):
                            yield MessageChunkSSE(
                                data=MessageChunkSSEData(
                                    delta="Created the requested artifact."
                                )
                            )
                            for artifact in pending_response_artifacts:
                                yield MessageArtifactSSE(data=artifact)
                        async for final_event in self._finalize_copilot_stream(
                            deferred_citations
                        ):
                            yield final_event
                        return

                    deferred_tail_specs = self._serialize_deferred_tail(
                        function_calls_to_process,
                        queue_index + 1,
                    )

                    # The following functions are executed by the client.
                    # The client acts as a data proxy, getting the data and
                    # returning it back to Copilot with a follow-up request.
                    if self._client_function_call_service.is_client_function_call(
                        current_call
                    ):
                        # Workspace frontend expects one client function call event at
                        # a time. We merge consecutive same-type widget calls
                        # into one boundary event and defer the rest of the
                        # queue through extra_state.
                        client_calls, next_index = self._get_client_call_batch(
                            function_calls_to_process,
                            queue_index,
                        )
                        if len(client_calls) > 1:
                            current_call = self._merge_client_function_calls(
                                client_calls
                            )
                            deferred_tail_specs = self._serialize_deferred_tail(
                                function_calls_to_process,
                                next_index,
                            )
                        (
                            client_events,
                            should_exit,
                        ) = await self._emit_client_function_call_events(
                            current_call,
                            chat_messages=chat_messages,
                            original_chat_messages_count=original_chat_messages_count,
                            deferred_tail_specs=deferred_tail_specs,
                        )
                        for client_event in client_events:
                            yield client_event
                        if should_exit:
                            early_exit = True
                            break
                        queue_index += len(client_calls)
                        continue

                    if self._is_parallel_safe_native_function_call(current_call):
                        parallel_group, next_index = self._get_parallel_native_group(
                            function_calls_to_process,
                            queue_index,
                        )

                        if len(parallel_group) > 1:
                            semaphore = asyncio.Semaphore(
                                self.MAX_PARALLEL_NATIVE_CALLS
                            )
                            status_queue: asyncio.Queue[StatusUpdateSSE] = (
                                asyncio.Queue()
                            )
                            native_event_groups: list[
                                list[NativeFunctionCallEvent] | None
                            ] = [None] * len(parallel_group)
                            parallel_failures: list[Exception | None] = [None] * len(
                                parallel_group
                            )
                            parallel_tasks: dict[
                                asyncio.Task[list[NativeFunctionCallEvent]],
                                int,
                            ] = {
                                asyncio.create_task(
                                    self._collect_native_function_call_events_with_limit(
                                        function_call,
                                        semaphore,
                                        status_queue=status_queue,
                                    )
                                ): group_index
                                for group_index, function_call in enumerate(
                                    parallel_group
                                )
                            }

                            # Keep the backend work parallel, but yield status
                            # updates as soon as each tool emits them instead
                            # of waiting for the whole batch to finish.
                            try:
                                while parallel_tasks:
                                    next_status_task: asyncio.Task[StatusUpdateSSE] = (
                                        asyncio.create_task(status_queue.get())
                                    )
                                    done, _ = await asyncio.wait(
                                        set(parallel_tasks) | {next_status_task},
                                        return_when=asyncio.FIRST_COMPLETED,
                                    )

                                    if next_status_task in done:
                                        status_event = next_status_task.result()
                                        self._logging_service.log_status_update_sse(
                                            status_event
                                        )
                                        if (
                                            status_event.data.message
                                            == "Artifact generated"
                                            and status_event.data.artifacts
                                        ):
                                            pending_response_artifacts = (
                                                status_event.data.artifacts
                                            )
                                            visible_response_after_artifact = False
                                        yield status_event
                                    else:
                                        next_status_task.cancel()
                                        with suppress(asyncio.CancelledError):
                                            await next_status_task

                                    for completed_task in done:
                                        if completed_task is next_status_task:
                                            continue
                                        native_task = cast(
                                            asyncio.Task[list[NativeFunctionCallEvent]],
                                            completed_task,
                                        )
                                        group_index = parallel_tasks.pop(native_task)
                                        try:
                                            native_event_groups[group_index] = (
                                                native_task.result()
                                            )
                                        except Exception as exc:
                                            parallel_failures[group_index] = exc
                                            function_name = parallel_group[
                                                group_index
                                            ].function.__name__
                                            self._logging_service.warning(
                                                "Parallel native function call failed "
                                                "for %s: %s",
                                                function_name,
                                                exc,
                                            )
                            finally:
                                if parallel_tasks:
                                    for pending_task in parallel_tasks:
                                        pending_task.cancel()
                                    await asyncio.gather(
                                        *parallel_tasks, return_exceptions=True
                                    )

                            while not status_queue.empty():
                                status_event = status_queue.get_nowait()
                                self._logging_service.log_status_update_sse(
                                    status_event
                                )
                                if (
                                    status_event.data.message == "Artifact generated"
                                    and status_event.data.artifacts
                                ):
                                    pending_response_artifacts = (
                                        status_event.data.artifacts
                                    )
                                    visible_response_after_artifact = False
                                yield status_event

                            # Replay each result group in the original emitted
                            # order even though collection happened in parallel.
                            for group_offset, grouped_native_events in enumerate(
                                native_event_groups
                            ):
                                failure = parallel_failures[group_offset]
                                if failure is not None:
                                    chat_messages.extend(
                                        self._build_function_call_error_messages(
                                            parallel_group[group_offset], failure
                                        )
                                    )
                                    continue
                                if grouped_native_events is None:
                                    continue
                                tail_specs = self._serialize_deferred_tail(
                                    function_calls_to_process,
                                    queue_index + group_offset + 1,
                                )
                                (
                                    native_events_to_yield,
                                    should_exit,
                                ) = await self._replay_native_function_call_events(
                                    grouped_native_events,
                                    chat_messages=chat_messages,
                                    original_chat_messages_count=original_chat_messages_count,
                                    deferred_citations=deferred_citations,
                                    deferred_tail_specs=tail_specs,
                                )
                                for native_event in native_events_to_yield:
                                    if pending_response_artifacts and isinstance(
                                        native_event,
                                        (MessageChunkSSE, MessageArtifactSSE),
                                    ):
                                        visible_response_after_artifact = True
                                    yield native_event
                                if should_exit:
                                    early_exit = True
                                    break
                            queue_index = next_index
                            if early_exit:
                                break
                            continue

                    # If instead we need to do a function call / reasoning step
                    # internally (eg. retrieve data from documents)...
                    native_events: list[NativeFunctionCallEvent] = []
                    async for (
                        native_fc_event
                    ) in self._native_function_call_service.handle_function_calls(
                        current_call
                    ):
                        if isinstance(native_fc_event, StatusUpdateSSE):
                            self._logging_service.log_status_update_sse(native_fc_event)
                            if (
                                native_fc_event.data.message == "Artifact generated"
                                and native_fc_event.data.artifacts
                            ):
                                pending_response_artifacts = (
                                    native_fc_event.data.artifacts
                                )
                                visible_response_after_artifact = False
                            yield native_fc_event
                        else:
                            native_events.append(native_fc_event)
                    (
                        native_events_to_yield,
                        should_exit,
                    ) = await self._replay_native_function_call_events(
                        native_events,
                        chat_messages=chat_messages,
                        original_chat_messages_count=original_chat_messages_count,
                        deferred_citations=deferred_citations,
                        deferred_tail_specs=deferred_tail_specs,
                    )
                    for native_event in native_events_to_yield:
                        if pending_response_artifacts and isinstance(
                            native_event,
                            (MessageChunkSSE, MessageArtifactSSE),
                        ):
                            visible_response_after_artifact = True
                        yield native_event
                    if should_exit:
                        early_exit = True
                        break
                    queue_index += 1

                if early_exit:
                    exit_loop = True
                    break

                # Handle case where max calls reached without completion
                if call_count >= self.MAX_CALLS:
                    yield StatusUpdateSSE(
                        data=StatusUpdateSSEData(
                            eventType="WARNING",
                            message=(
                                f"Maximum function calls ({self.MAX_CALLS}) reached"
                            ),
                            details=[
                                {
                                    "Detail": (
                                        "The task was complex and hit the "
                                        "iteration limit. Providing partial "
                                        "results and offering to continue."
                                    )
                                }
                            ],
                        )
                    )

                    # Force completion with instruction to summarize and offer
                    # continuation
                    chat_messages.append(
                        SystemMessage(
                            "SYSTEM: You have reached the maximum number of "
                            "function calls. Provide a comprehensive response "
                            "summarizing your work so far and naturally offer the "
                            "user the option to continue in a follow-up message."
                        )
                    )

                    try:
                        completion_chain = await self._compose_chain(
                            chat_messages=chat_messages,
                            documents=self._document_service.documents,
                            tools=None,  # No more tools allowed
                            original_messages=messages,
                            original_context=context,
                            widget_collection=widget_collection,
                            app_widget_collection=app_widget_collection,
                            sql_widgets=sql_widgets,
                            python_widgets=python_widgets,
                            semantic_views_for_prompt=prompt_semantic_views,
                        )
                        response = await completion_chain()
                    except Exception:
                        # If forced completion fails, just show a simple message
                        yield StatusUpdateSSE(
                            data=StatusUpdateSSEData(
                                eventType="INFO",
                                message="Task partially completed",
                                details=[
                                    {
                                        "Detail": (
                                            "The task reached the complexity limit but "
                                            "partial work was accomplished."
                                        )
                                    }
                                ],
                            )
                        )
                        return

                # If copilot gives a final answer, we stream it back
                if isinstance(response, AsyncStreamedStr):
                    self._logging_service.info(
                        "Accessed Citations: %s", self._citation_service.citations
                    )
                    self._logging_service.debug("Streaming tokens...")
                    async for stream_event in self._stream_copilot_events(response):
                        if pending_response_artifacts and isinstance(
                            stream_event,
                            (MessageChunkSSE, MessageArtifactSSE),
                        ):
                            visible_response_after_artifact = True
                        yield stream_event

                    async for final_event in self._finalize_copilot_stream(
                        deferred_citations
                    ):
                        yield final_event
                    return

        except ToolLimitExceededError as err:
            self._logging_service.warning(
                "OpenAI tool limit exceeded before request send "
                "(total=%d, mcp=%d, threshold=%d)",
                err.total_tool_count,
                err.mcp_tool_count,
                err.threshold,
            )
            yield self._build_tool_limit_warning_sse(
                total_tool_count=err.total_tool_count,
                mcp_tool_count=err.mcp_tool_count,
                threshold=err.threshold,
            )
            return
        except ContextLimitExceededError:
            logfire.error("Context limit exceeded")
            self._logging_service.warning("Context limit is too large.")
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Context limit exceeded",
                    details=[
                        {
                            "Detail": "This conversation has exceeded the context limit. Please try again, but consider using less data as input."  # noqa: E501
                        }
                    ],
                )
            )
        except openai.OpenAIError as err:
            logfire.error("LLM call failed: {err}", err=err)
            self._logging_service.error("LLM call failed: %s", err)
            # Log additional error details for debugging
            status_code = getattr(err, "status_code", "N/A")
            self._logging_service.error(
                "OpenAI Error Details - Type: %s, Status: %s",
                type(err).__name__,
                status_code,
            )
            # Extract request_id from error response if available
            request_id = None
            if hasattr(err, "response") and err.response is not None:
                headers = dict(err.response.headers)
                self._logging_service.error("Response headers: %s", headers)
                request_id = headers.get("x-request-id", headers.get("request-id"))
            if hasattr(err, "body") and err.body:
                self._logging_service.error("Response body: %s", err.body)
            # Log the last request context for debugging
            self._logging_service.error(
                "Failed after %d calls. Last chat_messages count: %d",
                call_count,
                len(chat_messages),
            )
            # Log full request diagnostics on failure to help debug 500 errors
            if status_code == 500 or status_code == "500":
                self._logging_service.error(
                    "=== FAILED REQUEST DIAGNOSTICS (HTTP %s) ===", status_code
                )
                self._log_request_diagnostics(
                    messages=chat_messages,
                    functions=None,  # Functions not available here
                    request_id=request_id,
                )
            yield handle_openai_error(err)
            return
        except Exception as err:
            logfire.error("Unhandled exception: {err}", err=err)
            self._logging_service.critical(
                "Unhandled exception: %s - %s\nStacktrace:\n%s",
                type(err).__name__,
                str(err),
                "".join(traceback.format_exception(type(err), err, err.__traceback__)),
            )
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR", message="An unexpected error has occurred."
                )
            )
            if constants.ENVIRONMENT == "TEST":
                raise err

    def _get_intermediate_results_text(
        self,
        messages: list[Message[Any] | AssistantMessage],
        original_count: int,
    ) -> str | None:
        """Extract text results from intermediate native function calls.

        Returns a simple text summary of what was done, or None if nothing new.
        """
        new_messages = messages[original_count:]
        results = []

        for msg in new_messages:
            if isinstance(msg, FunctionResultMessage):
                results.append(msg.content)

        if not results:
            return None

        return "\n\n".join(results)

    def _get_terminal_data_load_failure_message(
        self,
        messages: Sequence[LlmClientFunctionCallResultMessage | LlmClientMessage],
    ) -> str | None:
        if (
            not messages
            or not isinstance(messages[-1], LlmClientFunctionCallResultMessage)
            or messages[-1].function not in {"get_widget_data", "get_extra_widget_data"}
        ):
            return None

        raw_data_sources = (messages[-1].input_arguments or {}).get("data_sources", [])
        if not isinstance(raw_data_sources, list) or not raw_data_sources:
            return None

        for raw_data_source in raw_data_sources:
            try:
                data_source_request = DataSourceRequestPayload.model_validate(
                    raw_data_source
                )
            except ValidationError:
                return None

            data_source = self._copilot_data_service.get_data_source_from_map_or_db(
                data_source_request.widget_uuid
            )
            if data_source is None:
                return None

            if not self._document_service.get_unavailable_documents_by_data_source(
                data_source
            ):
                return None

        return (
            "I was unable to retrieve the requested document, so I cannot search "
            "its contents. Please refresh the widget, rerun the request, or "
            "provide a valid file or link."
        )

    def _resolve_artifact_reference(self, response: FunctionCall) -> None:
        """Resolve generic artifact_id to inline data for widget generation.

        When the LLM passes an artifact_id instead of inline data, this method
        looks up generic artifact data from the context service and injects it
        into the function call kwargs so the complete data reaches the frontend.
        SQL query artifacts are handled separately because the dashboard SQL
        widget flow needs widget identity plus raw SQL, not generic inline data.
        """
        if response.function.__name__ != "llm_generate_widget_in_dashboard":
            return

        artifact_id = cast(str | None, response._kwargs.get("artifact_id"))
        if not artifact_id:
            return

        self._logging_service.info(
            "Resolving artifact_id=%s for widget generation", artifact_id
        )

        full_data = self._context_service.get_full_table_data(artifact_id)
        if full_data:
            response._kwargs["data"] = full_data
            self._logging_service.info(
                "Resolved artifact_id=%s to %d rows",
                artifact_id,
                len(full_data),
            )
            return

        ctx = self._context_service.get_context_by_name(artifact_id)
        if ctx and isinstance(ctx, UnstructuredContext):
            if (ctx.source_info.metadata or {}).get("parse_as") == "snowflake_query":
                self._logging_service.info(
                    "Leaving snowflake_query artifact_id=%s "
                    "for Ada-side widget bridging",
                    artifact_id,
                )
                return
            response._kwargs["data"] = ctx.content
            self._logging_service.info(
                "Resolved artifact_id=%s from unstructured context (length=%d)",
                artifact_id,
                len(ctx.content),
            )
            return

        self._logging_service.warning(
            "Failed to resolve artifact_id=%s — not found in context",
            artifact_id,
        )

    @staticmethod
    def _extract_sql_from_query_artifact_content(content: str) -> str:
        """Normalize stored SQL artifact content back to raw SQL text."""
        normalized_content = content.strip()

        with suppress(json.JSONDecodeError):
            decoded_content = json.loads(normalized_content)
            if isinstance(decoded_content, str):
                normalized_content = decoded_content.strip()

        content_lines = normalized_content.splitlines()
        if (
            len(content_lines) >= 3
            and content_lines[0].startswith("```")
            and content_lines[-1].strip() == "```"
        ):
            return "\n".join(content_lines[1:-1]).strip()

        return normalized_content

    def _build_sql_artifact_dashboard_event(
        self,
        function_call: FunctionCall,
    ) -> FunctionCallSSE | None:
        """Convert a SQL query artifact into the concrete dashboard SQL widget request.

        `llm_generate_widget_in_dashboard` is a generic artifact-based tool, but
        SQL widgets are still created via `add_widget_to_dashboard` with the
        source widget identity plus the generated query. Keeping that translation
        in Ada lets the frontend stay generic.
        """
        if function_call.function.__name__ != "llm_generate_widget_in_dashboard":
            return None
        widget_type = function_call.arguments.get("widget_type")
        if widget_type not in {"table", "chart"}:
            return None

        artifact_id = function_call.arguments.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            return None

        artifact_context = self._context_service.get_context_by_name(artifact_id)
        if not isinstance(artifact_context, UnstructuredContext):
            return None

        metadata = artifact_context.source_info.metadata or {}
        if metadata.get("parse_as") != "snowflake_query":
            return None

        sql_query = self._extract_sql_from_query_artifact_content(
            artifact_context.content
        )
        if not sql_query:
            return None

        query_data_source = metadata.get("query_data_source")
        if not isinstance(query_data_source, dict):
            self._logging_service.info(
                "Unable to recover source widget for query artifact '%s'",
                artifact_id,
            )
            return None

        input_args = {**DEFAULT_SSRM_INPUT_ARGS, "query": sql_query}
        inner_tab = function_call.arguments.get("inner_tab")
        if isinstance(inner_tab, str) and inner_tab:
            input_args["inner_tab"] = inner_tab

        try:
            data_source_payload = DataSourceRequestPayload(
                widget_uuid=query_data_source["widget_uuid"],
                origin=query_data_source["origin"],
                id=query_data_source["id"],
                input_args=input_args,
            )
        except (KeyError, TypeError, ValidationError):
            self._logging_service.info(
                "Unable to validate source widget for query artifact '%s'",
                artifact_id,
            )
            return None

        search_query: dict[str, Any] = {
            "description": data_source_payload.id,
            "query": sql_query,
        }
        if isinstance(inner_tab, str) and inner_tab:
            search_query["inner_tab"] = inner_tab

        summary = function_call.arguments.get("summary")
        if not isinstance(summary, str) or not summary:
            summary = "Adding widget to dashboard"

        self._logging_service.info(
            "Bridging snowflake_query artifact '%s' to add_widget_to_dashboard "
            "for widget_type=%s",
            artifact_id,
            widget_type,
        )

        copilot_function_call_arguments: dict[str, Any] = {
            "summary": summary,
            "search_queries": [search_query],
        }
        if widget_type == "chart":
            copilot_function_call_arguments["widget_type"] = "chart"
            name = function_call.arguments.get("name")
            if isinstance(name, str) and name:
                copilot_function_call_arguments["name"] = name
            description = function_call.arguments.get("description")
            if isinstance(description, str) and description:
                copilot_function_call_arguments["description"] = description
            chart_params = function_call.arguments.get("chart_params")
            if isinstance(chart_params, dict):
                copilot_function_call_arguments["chart_params"] = chart_params

        extra_state: dict[str, Any] = {
            "copilot_function_call_arguments": copilot_function_call_arguments,
            "sql_query": sql_query,
        }
        if artifact_context.source_info.uuid is not None:
            extra_state["sql_artifact_uuid"] = str(artifact_context.source_info.uuid)

        return FunctionCallSSE(
            data=FunctionCallSSEData(
                function="add_widget_to_dashboard",
                input_arguments={"data_sources": [data_source_payload.model_dump()]},
                extra_state=extra_state,
            )
        )

    async def _format_chat_messages(
        self,
        messages: list[LlmClientMessage | LlmClientFunctionCallResultMessage],
    ) -> AsyncGenerator[list[Message[Any] | AssistantMessage] | StatusUpdateSSE, None]:
        chat_messages: list[Message[Any] | AssistantMessage] = []
        function_call = None

        for index, message in enumerate(messages):
            # Handle assistant messages
            if isinstance(message, LlmClientMessage):
                if message.role == "ai":
                    if isinstance(message.content, str):
                        # If an artifact tag is included in the message (meaning
                        # the LLM returned it inline), we insert a preview of
                        # the artifact to maintain consistency between the LLM's
                        # answer and native function calls, which do not persist
                        # across completions.
                        if "<|start_artifact_id|>" in message.content:
                            message_split = message.content.replace(
                                "<|end_artifact_id|>", "<|end_artifact_id|><<<SPLIT>>>"
                            ).split("<<<SPLIT>>>")
                            for index in range(0, len(message_split), 2):
                                message_part = message_split[index]
                                if (
                                    "<|start_artifact_id|>" in message_part
                                    and "<|end_artifact_id|>" in message_part
                                ):
                                    artifact_id = message_part.split(
                                        "<|start_artifact_id|>"
                                    )[1].split("<|end_artifact_id|>")[0]
                                    if (
                                        context_preview
                                        := self._context_service.preview_context(  # noqa: E501
                                            artifact_id
                                        )
                                    ):
                                        message_part += f"\n(SYSTEM: --- previewing artifact: {artifact_id} ---)\n{context_preview}\n(SYSTEM: --- end of preview ---)"  # noqa: E501
                                    message_split[index] = message_part
                            message.content = "".join(message_split)

                        assistant_message = ""
                        if message.agent_id:
                            assistant_message = sanitize_str(
                                f"(SYSTEM: The following message was sent by the agent with id '{message.agent_id}')\n\n"  # noqa: E501
                            )
                        assistant_message += message.content
                        chat_messages.append(
                            AssistantMessage(sanitize_str(assistant_message))
                        )
                    elif isinstance(message.content, LlmClientFunctionCall):
                        # We skip `get_params_options` function calls, because
                        # they are intermediary functions that we only used to
                        # fetch parameter options for generating input args for
                        # widgets / data sources
                        if message.content.function == "get_params_options":
                            continue

                        # We assume that after a LlmClientFunctionCall we have
                        # the corresponding LlmClientFunctionCallResultMessage
                        fc_result_message = cast(
                            LlmClientFunctionCallResultMessage, messages[index + 1]
                        )
                        (
                            tool,
                            arguments,
                        ) = self._client_function_call_service.get_function_call_spec(
                            fc_result_message
                        )
                        function_call = FunctionCall(tool, **arguments)
                        chat_messages.append(AssistantMessage(function_call))
                # Handle human messages
                elif message.role == "human" and isinstance(message.content, str):
                    chat_messages.append(UserMessage(sanitize_str(message.content)))

            elif isinstance(message, LlmClientFunctionCallResultMessage):
                # We skip `get_params_options` function call results, because
                # they are intermediary functions that we only used to fetch
                # parameter options for generating input args for widgets / data
                # sources
                if message.function == "get_params_options":
                    continue
                async for event in self._handle_function_call_result_message(
                    message=message,
                    function_call=function_call,
                    # We only yield warning SSEs to the client if it's the most
                    # recent function call result message (otherwise we'd raise
                    # warning SSEs for every function call result message that
                    # contains an error)
                    is_last_message=bool(index == len(messages) - 1),
                ):
                    if isinstance(event, StatusUpdateSSE):
                        yield event
                    else:
                        chat_messages.extend(event)
        yield chat_messages

    def _handle_function_call_result_error(
        self,
        client_function_call_error: ClientFunctionCallError,
        data_source_request: DataSourceRequestPayload,
    ) -> str:
        input_args = data_source_request.input_args or {}
        query_or_prompt = input_args.get("query") or input_args.get("prompt")
        latest_input_text = (
            f"\n  latest_input_text: {query_or_prompt}"
            if isinstance(query_or_prompt, str) and query_or_prompt.strip()
            else ""
        )
        latest_input_note = (
            "\n  note: The latest widget input text above is the current state "
            "sent by the UI. Data loading may fail even when the input text sync "
            "is correct (for example, invalid SQL syntax or unsupported characters)."
            if latest_input_text
            else ""
        )
        return (
            f"An error occurred:\n"
            f"  error_type: {client_function_call_error.error_type}\n"
            f"  error_content: {client_function_call_error.content}\n"
            f"  widget details:\n"
            f"    uuid: {data_source_request.widget_uuid}\n"
            f"    origin: {data_source_request.origin}\n"
            f"    id: {data_source_request.id}\n"
            f"    input_args={input_args}"
            f"{latest_input_text}"
            f"{latest_input_note}"
        )  # noqa: E501

    def _get_current_turn_messages(
        self,
        messages: list[LlmClientMessage | LlmClientFunctionCallResultMessage],
    ) -> list[LlmClientMessage | LlmClientFunctionCallResultMessage]:
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, LlmClientMessage) and message.role == "human":
                return messages[index:]
        return messages

    def _get_previously_queried_file_widget_uuids(
        self,
        messages: list[LlmClientMessage | LlmClientFunctionCallResultMessage],
    ) -> set[UUID]:
        queried_file_widget_uuids: set[UUID] = set()

        for message in self._get_current_turn_messages(messages):
            if not isinstance(message, LlmClientFunctionCallResultMessage):
                continue
            if message.function != "get_widget_data":
                continue

            raw_data_sources = (message.input_arguments or {}).get("data_sources", [])
            if not isinstance(raw_data_sources, list):
                continue

            for raw_data_source in raw_data_sources:
                try:
                    data_source_request = DataSourceRequestPayload.model_validate(
                        raw_data_source
                    )
                except ValidationError:
                    self._logging_service.warning(
                        "Skipping invalid data source request while checking for "
                        "repeated file widget queries: %s",
                        raw_data_source,
                    )
                    continue

                if data_source_request.id.startswith("file-"):
                    queried_file_widget_uuids.add(
                        UUID(str(data_source_request.widget_uuid))
                    )

        return queried_file_widget_uuids

    def _get_repeated_file_queries(
        self,
        function_call: FunctionCall,
        previously_queried_file_widget_uuids: set[UUID],
    ) -> list[WidgetQueryRequest]:
        if (
            function_call.function.__name__
            != self._client_function_call_service.llm_query_widgets.__name__
        ):
            return []

        raw_widget_queries = function_call.arguments.get("widget_queries")
        if not isinstance(raw_widget_queries, list):
            return []

        repeated_widget_queries: list[WidgetQueryRequest] = []

        for raw_widget_query in raw_widget_queries:
            try:
                widget_query = WidgetQueryRequest.model_validate(raw_widget_query)
            except ValidationError:
                self._logging_service.warning(
                    "Skipping invalid widget query while checking for repeated "
                    "file widget queries: %s",
                    raw_widget_query,
                )
                continue

            data_source = self._copilot_data_service.get_data_source_from_map_or_db(
                widget_query.widget_uuid
            )
            if not data_source or not data_source.id.startswith("file-"):
                continue

            if widget_query.widget_uuid in previously_queried_file_widget_uuids:
                repeated_widget_queries.append(widget_query)

        return repeated_widget_queries

    def _build_repeated_file_widget_retry_messages(
        self,
        function_call: FunctionCall,
        repeated_widget_queries: Sequence[WidgetQueryRequest],
    ) -> list[Message[Any]]:
        widget_list = ", ".join(
            str(widget_query.widget_uuid) for widget_query in repeated_widget_queries
        )
        content = (
            "Blocked repeated file widget request for widget_uuid(s): "
            f"{widget_list}. These file widgets were already queried earlier in "
            "this user turn. Do NOT call `llm_query_widgets` again for the same "
            "file widget, even if you rephrase the question. If the uploaded "
            "file references were already loaded, call "
            "`_llm_query_uploaded_files` next. Otherwise explain the failure to "
            "the user or choose a different strategy."
        )
        return [self._build_blocked_tool_feedback_message(function_call, content)]

    def _build_blocked_tool_feedback_message(
        self, function_call: FunctionCall, content: str
    ) -> AssistantMessage:
        """Represent an internal guardrail block as plain assistant text."""
        function_name = getattr(function_call.function, "__name__", "unknown_tool")
        message = (
            "(SYSTEM: Internal tool feedback. The following tool call was blocked "
            "and was not executed.)\n"
            f"Tool: `{function_name}`\n\n{content}"
        )
        return AssistantMessage(sanitize_str(message))

    def _extract_useful_widget_metadata(self, widget) -> dict[str, Any]:
        """Extract useful metadata from widget for SQL agent table selection."""
        extracted_metadata = {}

        # Get widget metadata if it exists
        if hasattr(widget, "metadata") and widget.metadata:
            # Extract ssmRequest information if present
            if "ssmRequest" in widget.metadata:
                ssm_request = widget.metadata["ssmRequest"]

                # Extract group keys (filtering criteria)
                if "groupKeys" in ssm_request and ssm_request["groupKeys"]:
                    extracted_metadata["filtered_by"] = ssm_request["groupKeys"]

                # Extract row group columns (what data is grouped by)
                if "rowGroupCols" in ssm_request and ssm_request["rowGroupCols"]:
                    group_info = []
                    for col in ssm_request["rowGroupCols"]:
                        if isinstance(col, dict) and "displayName" in col:
                            group_info.append(col["displayName"])
                    if group_info:
                        extracted_metadata["grouped_by"] = group_info

                # Extract pivot information
                if "pivotCols" in ssm_request and ssm_request["pivotCols"]:
                    extracted_metadata["pivot_columns"] = ssm_request["pivotCols"]

                # Add table name from ssmRequest if available
                if "name" in ssm_request:
                    extracted_metadata["table_display_name"] = ssm_request["name"]

                # Add row count information
                if "endRow" in ssm_request and "startRow" in ssm_request:
                    row_count = ssm_request["endRow"] - ssm_request["startRow"]
                    extracted_metadata["row_count_range"] = (
                        f"{ssm_request['startRow']}-{ssm_request['endRow']} "
                        f"(showing {row_count} rows)"
                    )

                self._logging_service.info(
                    "Extracted widget metadata for SQL agent: widget_id=%s, "
                    "extracted_metadata=%s",
                    getattr(widget, "widget_id", "unknown"),
                    extracted_metadata,
                )

        return extracted_metadata

    def _sanitize_sql_schema(self, schema: str | dict) -> str:
        """Sanitize SQL schema while preserving type information.

        Handles both JSON schema format and SQL DDL format.
        Removes verbose metadata but keeps column names and types.

        Examples:
            JSON Input: {"schema": {"tableName": "users", "columns": [...]}}
            JSON Output: users: id (INT), email (VARCHAR), name (VARCHAR)

            SQL Input:  CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(255))
            SQL Output: users: id (INT), name (VARCHAR)
        """

        # Try to parse as JSON first (most common format)
        try:
            schema_data: dict = schema  # type: ignore[assignment]
            if isinstance(schema, str):
                schema_data = json.loads(schema)

            # Check if this is already a proper schema dict with tableName and
            # columns. Don't navigate into nested "schema" key if it's just
            # the Snowflake schema name
            if isinstance(schema_data, dict):
                # If it has tableName/table_name and columns at top level,
                # use it directly
                has_table_name = (
                    "tableName" in schema_data or "table_name" in schema_data
                )
                has_columns = "columns" in schema_data

                # Only navigate into nested "schema" if it's a dict
                # (not a string schema name)
                if not (has_table_name or has_columns) and "schema" in schema_data:
                    if isinstance(schema_data["schema"], dict):
                        schema_data = schema_data["schema"]
                        if isinstance(schema_data, str):
                            schema_data = json.loads(schema_data)

            if not isinstance(schema_data, dict):
                return str(schema)

            # Extract full table name
            full_table_name = "UNKNOWN"

            database = schema_data.get("database", "")
            schema_name = schema_data.get("schema", "")
            if database:
                full_table_name = (
                    f"{database}.{schema_name}."
                    f"{schema_data.get('tableName', 'UNKNOWN')}"
                )

            # Extract columns
            columns = schema_data.get("columns", [])
            if not columns:
                keys = (
                    list(schema_data.keys()) if isinstance(schema_data, dict) else "N/A"
                )
                self._logging_service.warning(
                    "Schema sanitization skipped - no columns found. "
                    "Type: %s, Keys: %s, Structure: %s",
                    type(schema).__name__,
                    keys,
                    str(schema)[:200],
                )
                return str(schema)  # Can't sanitize, return original

            # Format as: column_name (TYPE)
            columns_formatted = [
                f"{col['name']} ({col['type']})"
                for col in columns
                if "name" in col and "type" in col
            ]

            if not columns_formatted:
                self._logging_service.warning(
                    "Schema sanitization skipped - columns missing name/type: %s",
                    str(schema)[:200],
                )
                return str(schema)  # Can't sanitize, return original

            # Create sanitized format with line breaks every 5 columns for readability
            sanitized = f"{full_table_name}:\n  "
            for i, col_def in enumerate(columns_formatted):
                sanitized += col_def
                if i < len(columns_formatted) - 1:
                    sanitized += ", "
                # Add line break every 5 columns for readability
                if (i + 1) % 5 == 0 and i < len(columns_formatted) - 1:
                    sanitized += "\n  "

            return sanitized

        except (json.JSONDecodeError, KeyError, TypeError) as sql_json_err:
            self._logging_service.warning(
                "Schema sanitization JSON error: %s", sql_json_err
            )

            # Fall back to SQL DDL parsing if JSON parsing fails
            import re

            schema_str = str(schema)

            # Extract table name from CREATE TABLE statement
            table_match = re.search(r"CREATE TABLE\s+(\w+)", schema_str, re.IGNORECASE)
            if not table_match:
                self._logging_service.warning(
                    "Schema sanitization failed - no CREATE TABLE found. "
                    "Type: %s, Structure: %s",
                    type(schema).__name__,
                    schema_str[:200],
                )
                return schema_str  # Return original if we can't parse it

            table_name = table_match.group(1)

            # Extract column definitions: column_name DATA_TYPE
            column_pattern = r"\n\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+([A-Z]+(?:\([^)]+\))?)"
            column_matches = re.findall(column_pattern, schema_str)

            if not column_matches:
                self._logging_service.warning(
                    "Schema sanitization failed - no columns found in DDL: %s",
                    schema_str[:200],
                )
                return schema_str  # Return original if we can't parse columns

            # Format as: column_name (TYPE)
            columns_formatted = [f"{col} ({typ})" for col, typ in column_matches]

            # Create sanitized format
            sanitized = f"{table_name}:\n  "
            for i, col_def in enumerate(columns_formatted):
                sanitized += col_def
                if i < len(columns_formatted) - 1:
                    sanitized += ", "
                if (i + 1) % 5 == 0 and i < len(columns_formatted) - 1:
                    sanitized += "\n  "

            return sanitized

    def _remove_fetched_skill_from_catalog(self, slug: str) -> None:
        """Remove a fetched skill from the catalog so the LLM won't re-fetch it.

        The skill content already lives in the conversation history as a
        ``FunctionResultMessage``.  Removing it from the catalog prevents
        the system prompt from listing it as "available to fetch".
        """
        before = len(self._skills_catalog)
        self._skills_catalog = [s for s in self._skills_catalog if s.slug != slug]
        if len(self._skills_catalog) < before:
            self._logging_service.info(
                "Removed fetched skill '%s' from catalog (remaining: %d)",
                slug,
                len(self._skills_catalog),
            )

    def _get_sql_enabled_widgets(
        self, widget_collection: WidgetCollection | None, include_extra: bool = False
    ) -> list[SqlWidgetContext]:
        """Extract SQL-enabled widgets (those with metadata.schema).

        Widgets are returned in priority order: primary, secondary, then extra.
        Extra widgets are only included when include_extra=True (global search).
        """
        if not widget_collection:
            return []

        sql_widgets: list[SqlWidgetContext] = []
        all_widgets = (widget_collection.primary or []) + (
            widget_collection.secondary or []
        )

        if include_extra:
            all_widgets += widget_collection.extra or []

        for widget in all_widgets:
            schema_raw = (widget.metadata or {}).get("schema")
            if not schema_raw:
                continue

            # Parse schema if it's a JSON string
            schema_dict = (
                json.loads(schema_raw) if isinstance(schema_raw, str) else schema_raw
            )

            # Sanitize the schema to reduce token usage while preserving types
            sanitized_schema = self._sanitize_sql_schema(schema_raw)

            # Get current SQL from params where name="query"
            query_param = next(
                (p for p in (widget.params or []) if p.name == "query"), None
            )
            current_sql = (
                (query_param.current_value or query_param.default_value)
                if query_param
                else None
            )

            sql_widgets.append(
                SqlWidgetContext(
                    widget_uuid=str(widget.uuid),
                    widget_id=widget.widget_id,
                    widget_name=widget.name,
                    widget_origin=widget.origin,
                    widget_description=widget.description,
                    sql_schema=schema_dict,  # Parsed dict for SQL generation
                    sql_schema_sanitized=sanitized_schema,  # Sanitized for prompts
                    current_sql=current_sql,
                )
            )

        return sql_widgets

    def _get_code_enabled_widgets(
        self, widget_collection: WidgetCollection | None, include_extra: bool = False
    ) -> list[PythonWidgetContext]:
        """Extract Python code-enabled widgets.

        Detection: Check if any param has language == "python".
        NOTE: Requires WidgetParam model in openbb-ai to have a `language` field.
        """
        if not widget_collection:
            return []

        python_widgets: list[PythonWidgetContext] = []
        all_widgets = (widget_collection.primary or []) + (
            widget_collection.secondary or []
        )
        if include_extra:
            all_widgets = all_widgets + (widget_collection.extra or [])

        for widget in all_widgets:
            # Detect Python widgets by checking if any param has language="python"
            has_python_param = any(
                getattr(p, "language", None) == "python" for p in (widget.params or [])
            )

            if not has_python_param:
                continue

            # Find the prompt param to get current code
            prompt_param = next(
                (
                    p
                    for p in (widget.params or [])
                    if getattr(p, "name", "") == "prompt"  # noqa: E501
                ),
                None,
            )

            # Get current code from the param
            current_code = (
                (prompt_param.current_value or prompt_param.default_value)
                if prompt_param
                else None
            )

            python_widgets.append(
                PythonWidgetContext(
                    widget_uuid=str(widget.uuid),
                    widget_id=widget.widget_id,
                    widget_name=widget.name,
                    widget_origin=widget.origin,
                    widget_description=widget.description,
                    current_code=current_code,
                )
            )

        # In Snowflake mode, only keep Snowflake-native Python widgets
        # so the LLM doesn't pick a generic widget that can't access
        # Snowflake data.
        if constants.SNOWFLAKE_NATIVE_APP and python_widgets:
            snow_widgets = [
                w for w in python_widgets if w.widget_origin == "SNOW backend"
            ]
            if snow_widgets:
                python_widgets = snow_widgets

        return python_widgets

    def _create_citable_python_source_info(
        self,
        source_info: SourceInfo,
        widget_name: str,
        widget_origin: str,
    ) -> SourceInfo:
        """Create a citable SourceInfo for Python code execution results.

        Python code results should always be citable so users can reference
        the widget that executed the code.

        Args:
            source_info: Original source info from the parsed context
            widget_name: Name of the widget
            widget_origin: Origin of the widget

        Returns:
            New SourceInfo with citable=True
        """
        return SourceInfo(
            uuid=source_info.uuid,
            type=source_info.type,
            origin=widget_origin,
            name=widget_name,
            widget_id=source_info.widget_id,
            description=source_info.description,
            metadata=source_info.metadata,
            citable=True,
        )

    def _process_snowflake_execution_result(
        self,
        parsed_context: ParsedContext,
        execution_type: Literal["sql", "python"],
        code_or_query: str,
        is_last_message: bool,
    ) -> tuple[str, list[StatusUpdateSSE]]:
        """Process Snowflake execution results and build content/artifacts.

        Args:
            parsed_context: The parsed context containing the result data
            execution_type: Either "sql" or "python"
            code_or_query: The SQL query or Python code that was executed
            is_last_message: Whether this is the last message in the execution cycle

        Returns:
            Tuple of (content string for LLM, list of StatusUpdateSSE to yield)
        """
        content = ""
        status_updates: list[StatusUpdateSSE] = []

        # Build header based on execution type
        code_preview = code_or_query[:100] + "..." if code_or_query else ""
        if execution_type == "sql":
            content += sanitize_str(
                f"Data from `{parsed_context.source_info.name}`"
                + (f" (query: `{code_preview}`)" if code_preview else "")
                + ":\n\n"
            )
        else:
            content += sanitize_str(
                f"Data from `{parsed_context.source_info.name}`"
                + (f" (code: `{code_preview}`)" if code_preview else "")
                + ":\n\n"
            )

        # Parse JSON content
        parsed_data = None
        if parsed_context.content:
            try:
                parsed_data = json.loads(parsed_context.content)
            except (json.JSONDecodeError, TypeError):
                preview = parsed_context.content[:500]
                content += f"Content: {preview}...\n\n"

        # Handle tabular results (list of dicts or list)
        if isinstance(parsed_data, list) and parsed_data:
            preview_rows = parsed_data[:15]
            total_rows = len(parsed_data)
            json_str = json.dumps(preview_rows, indent=2)
            content += f"```json\n{json_str}\n```\n\n"
            if total_rows > 15:
                content += f"(Showing 15/{total_rows} rows)\n\n"

            if is_last_message:
                preview_desc = (
                    f"Preview of {len(preview_rows)} rows"
                    + (f" (out of {total_rows} total)" if total_rows > 15 else "")
                    + f" from {parsed_context.source_info.name}"
                )
                src_info = parsed_context.source_info
                widget_name = src_info.name or "Unknown"
                widget_origin = src_info.origin or "Unknown"

                # Determine artifact type and content
                is_table_data = preview_rows and isinstance(preview_rows[0], dict)
                if is_table_data:
                    artifact_content: str | list[dict] = preview_rows
                    artifact_type: Literal["table", "text"] = "table"
                else:
                    artifact_content = json.dumps(preview_rows)
                    artifact_type = "text"

                # Build status message and artifact name based on type
                if execution_type == "sql":
                    msg = f"Query returned {total_rows} rows"
                    artifact_name = f"query_results_{widget_name}"
                else:
                    msg = f"Code executed - {total_rows} rows"
                    artifact_name = f"code_results_{widget_name}"

                status_updates.append(
                    StatusUpdateSSE(
                        data=StatusUpdateSSEData(
                            eventType="INFO",
                            message=msg,
                            artifacts=[
                                ClientArtifact(
                                    name=artifact_name,
                                    description=preview_desc,
                                    type=artifact_type,
                                    content=artifact_content,
                                )
                            ],
                        )
                    )
                )

                # Add citation
                if execution_type == "sql":
                    citation_source_info = parsed_context.source_info
                    citation_detail_key = "SQL"
                else:
                    citation_source_info = self._create_citable_python_source_info(
                        parsed_context.source_info, widget_name, widget_origin
                    )
                    citation_detail_key = "Code"

                citation = Citation(
                    source_info=citation_source_info,
                    details=[
                        {
                            "Widget": widget_name,
                            "Origin": widget_origin,
                            citation_detail_key: (
                                code_or_query[:200]
                                + ("..." if len(code_or_query) > 200 else "")
                            ),
                        }
                    ],
                )
                self._citation_service.add_citation(citation)

        elif is_last_message and execution_type == "python":
            # Python-only: handle non-tabular results
            src_info = parsed_context.source_info
            widget_name = src_info.name or "Unknown"
            widget_origin = src_info.origin or "Unknown"

            # Determine if there's output to show
            has_output = False
            result_desc = ""
            if parsed_data is not None:
                result_desc = str(parsed_data)[:200]
                has_output = True
            elif (
                parsed_context.content
                and parsed_context.content.strip()
                and parsed_context.content.strip().lower() != "none"
            ):
                result_desc = parsed_context.content[:200]
                has_output = True

            if has_output:
                status_updates.append(
                    StatusUpdateSSE(
                        data=StatusUpdateSSEData(
                            eventType="INFO",
                            message="Python code returned output",
                            artifacts=[
                                ClientArtifact(
                                    name=f"code_{widget_name}",
                                    description="Result",
                                    type="text",
                                    content=result_desc,
                                )
                            ],
                        )
                    )
                )
            else:
                status_updates.append(
                    StatusUpdateSSE(
                        data=StatusUpdateSSEData(
                            eventType="INFO",
                            message="Python code finished with no output",
                        )
                    )
                )

            # Add citation for Python non-tabular results
            citable_source_info = self._create_citable_python_source_info(
                parsed_context.source_info, widget_name, widget_origin
            )
            citation = Citation(
                source_info=citable_source_info,
                details=[
                    {
                        "Widget": widget_name,
                        "Origin": widget_origin,
                        "Code": code_or_query[:200],
                    }
                ],
            )
            self._citation_service.add_citation(citation)

        return content, status_updates

    async def _handle_function_call_result_message(
        self,
        message: LlmClientFunctionCallResultMessage,
        function_call: FunctionCall | None,
        is_last_message: bool = False,
    ) -> AsyncGenerator[list[FunctionResultMessage] | StatusUpdateSSE, None]:
        # TODO: This method should be split up into smaller helper methods
        self._logging_service.info(
            "[MCP Artifact] _handle_function_call_result_message called with "
            f"function: {message.function}"
        )
        if function_call:
            chat_messages: list[FunctionResultMessage] = []
            content = ""
            match message.function:
                case "get_widget_data" | "get_extra_widget_data":
                    successes: list[ParsedContext] = []
                    failures: list[str] = []
                    for data_source_request, data in zip(
                        (message.input_arguments or {}).get("data_sources", []),
                        message.data,
                        strict=True,
                    ):
                        data_source_request = DataSourceRequestPayload.model_validate(
                            data_source_request
                        )
                        origin = data_source_request.origin
                        id_ = data_source_request.id
                        data_source_id = data_source_request.widget_uuid
                        if (
                            data_source
                            := self._copilot_data_service.get_data_source_from_map_or_db(  # noqa: E501
                                data_source_id
                            )
                        ):
                            if (
                                unavailable_documents
                                := self._document_service.get_unavailable_documents_by_data_source(  # noqa: E501
                                    data_source
                                )
                            ):
                                for unavailable_document in unavailable_documents:
                                    failures.append(
                                        f"Could not retrieve document from {unavailable_document.source_info.origin} with id {unavailable_document.source_info.widget_id}: {unavailable_document.error}"  # noqa: E501
                                    )
                                continue

                            match data:
                                case DataContent():
                                    widget_uuid = UUID(data_source_request.widget_uuid)
                                    for item_index, data_content_item in enumerate(
                                        data.items
                                    ):
                                        # Extract useful metadata from widget for
                                        # better SQL agent table selection if SSRM
                                        # is enabled.
                                        widget_metadata = (
                                            self._extract_useful_widget_metadata(
                                                data_source.widget
                                            )
                                        )

                                        # Combine input_args with
                                        # extracted widget metadata
                                        combined_metadata = {
                                            "input_args": data_source_request.input_args,  # noqa: E501
                                            "widget_uuid": str(widget_uuid),
                                            **widget_metadata,
                                        }

                                        unique_context_uuid = build_context_uuid(
                                            widget_uuid,
                                            data_source_request.input_args,
                                            item_index,
                                        )

                                        parsed_context = ParsedContext(
                                            content=(
                                                ""
                                                if data_content_item.data_format.data_type  # noqa: E501
                                                == "pdf"
                                                else data_content_item.content
                                            ),
                                            source_info=SourceInfo(
                                                uuid=unique_context_uuid,
                                                type="widget",
                                                origin=data_source.origin,
                                                name=data_source.name,
                                                widget_id=data_source.id,
                                                description=data_source.description,
                                                metadata=combined_metadata,
                                                citable=data_content_item.citable,
                                            ),
                                        )
                                        successes.append(parsed_context)

                                case DataFileReferences():
                                    widget_uuid = UUID(data_source_request.widget_uuid)
                                    for item_index, file_reference in enumerate(
                                        data.items
                                    ):
                                        # Extract useful metadata from widget for
                                        # better SQL agent table selection
                                        widget_metadata = (
                                            self._extract_useful_widget_metadata(
                                                data_source.widget
                                            )
                                        )

                                        # Combine input_args with extracted widget
                                        #    metadata
                                        combined_metadata = {
                                            "input_args": data_source_request.input_args,  # noqa: E501
                                            "widget_uuid": str(widget_uuid),
                                            **widget_metadata,
                                        }

                                        unique_context_uuid = build_context_uuid(
                                            widget_uuid,
                                            data_source_request.input_args,
                                            item_index,
                                            extra_seed=getattr(
                                                file_reference, "url", None
                                            ),
                                        )

                                        parsed_context = ParsedContext(
                                            content="",
                                            source_info=SourceInfo(
                                                uuid=unique_context_uuid,
                                                type="widget",
                                                origin=data_source.origin,
                                                name=data_source.name,
                                                widget_id=data_source.id,
                                                description=data_source.description,
                                                metadata=combined_metadata,
                                                citable=file_reference.citable,
                                            ),
                                        )
                                        successes.append(parsed_context)
                                case ClientFunctionCallError():
                                    if is_last_message:
                                        yield StatusUpdateSSE(
                                            data=StatusUpdateSSEData(
                                                eventType="WARNING",
                                                message="An error occurred while fetching data from a widget",  # noqa: E501
                                                details=[
                                                    {
                                                        "Origin": data_source_request.origin,  # noqa: E501
                                                        "Widget Id": data_source_request.id,  # noqa: E501
                                                        **(
                                                            data_source_request.input_args
                                                            or {}
                                                        ),
                                                        "Error type": data.error_type,
                                                        "Error content": data.content,
                                                    }
                                                ],
                                            )
                                        )

                                    failures.append(
                                        self._handle_function_call_result_error(
                                            client_function_call_error=data,
                                            data_source_request=data_source_request,
                                        )
                                    )
                                    continue
                                case _:
                                    raise ValueError(
                                        f"Unexpected data type: {type(data)}"
                                    )

                            # Do not remove the is_last_message
                            # Only process citations for the last message in the current
                            # execution cycle (i.e. prevents accumulating citations
                            # across multiple messages)
                            if is_last_message and data.extra_citations:
                                for citation in data.extra_citations:
                                    if citation not in self._citation_service.citations:
                                        self._citation_service.add_citation(citation)

                        else:
                            failures.append(
                                f"Widget with origin '{origin}' and id '{id_}' not found in the current context. Has it been removed?"  # noqa: E501
                            )
                    # Check if this came from SQL/Python code generation
                    is_sql_query_result = (
                        hasattr(message, "extra_state")
                        and isinstance(message.extra_state, dict)
                        and "sql_query" in message.extra_state
                    )
                    is_python_code_result = (
                        hasattr(message, "extra_state")
                        and isinstance(message.extra_state, dict)
                        and "python_code" in message.extra_state
                    )

                    if successes:
                        # Load the context into the context service ONLY if it has
                        # content.  Note: We do this for DataFileReferences and base64
                        # PDFs because they have already been loaded into the
                        # DocumentService during dependency injection, so they don't
                        # have content here.
                        for parsed_context in successes:
                            if parsed_context.content:
                                self._context_service.load_context(
                                    elements=[parsed_context]
                                )
                                if is_last_message:
                                    source_info = parsed_context.source_info.model_copy(
                                        update={"citable": True}
                                    )
                                    citation_details = {
                                        "Origin": source_info.origin,
                                        "Data source": source_info.name,
                                    }
                                    input_args = (source_info.metadata or {}).get(
                                        "input_args"
                                    )
                                    if isinstance(input_args, dict):
                                        citation_details.update(
                                            flatten_and_format_dict(input_args)
                                        )
                                    self._citation_service.add_citation(
                                        Citation(
                                            source_info=source_info,
                                            details=[citation_details],
                                        )
                                    )

                        # Build header content based on execution type
                        if is_sql_query_result:
                            executed_code = message.extra_state.get("sql_query", "")
                            code_truncated = executed_code[:200]
                            code_suffix = "..." if len(executed_code) > 200 else ""
                            content = (
                                f"The SQL query has been executed. "
                                f"Query: `{code_truncated}{code_suffix}`\n\n"
                                "Output:\n\n"
                            )
                            execution_type: Literal["sql", "python"] | None = "sql"
                        elif is_python_code_result:
                            executed_code = message.extra_state.get("python_code", "")
                            code_truncated = executed_code[:200]
                            code_suffix = "..." if len(executed_code) > 200 else ""
                            content = (
                                f"Python code execution result. "
                                f"Code: `{code_truncated}{code_suffix}`\n\n"
                                "Output:\n\n"
                            )
                            execution_type = "python"
                        else:
                            content = "The data has been loaded:\n\n"
                            execution_type = None

                        for parsed_context in successes:
                            metadata = (
                                parsed_context.source_info.metadata
                                if parsed_context.source_info.metadata
                                else {}
                            )
                            input_args = metadata.get("input_args", None)

                            if execution_type:
                                # SQL or Python: use shared helper
                                code_or_query = (input_args or {}).get(
                                    "query" if execution_type == "sql" else "prompt",
                                    "",
                                )

                                ctx_content, status_updates = (
                                    self._process_snowflake_execution_result(
                                        parsed_context=parsed_context,
                                        execution_type=execution_type,
                                        code_or_query=code_or_query,
                                        is_last_message=is_last_message,
                                    )
                                )
                                content += ctx_content
                                for status_update in status_updates:
                                    yield status_update
                            else:
                                # Non-SQL/Python result: use original message format
                                content += sanitize_str(
                                    f"Data has been loaded after a function call for widget with origin `{parsed_context.source_info.origin}` and name `{parsed_context.source_info.name}`"  # noqa: E501
                                    + (
                                        f" with input arguments `{input_args}`.\n\n"
                                        if input_args
                                        else ".\n\n"
                                    )
                                )

                        _file_types = set(constants.FILE_EXTENSIONS_REQUIRING_QUERY)
                        has_file_like_tool_result = any(
                            (
                                isinstance(data_item, DataFileReferences)
                                and any(
                                    getattr(file_ref.data_format, "data_type", None)
                                    in _file_types
                                    for file_ref in data_item.items
                                )
                            )
                            or (
                                isinstance(data_item, DataContent)
                                and any(
                                    getattr(item.data_format, "data_type", None)
                                    in _file_types
                                    for item in data_item.items
                                )
                            )
                            for data_item in message.data
                        )
                        requires_uploaded_files_query = (
                            any(
                                # Widget IDs for user-uploaded files are prefixed
                                # with "file-" by convention in the workspace.
                                (parsed_ctx.source_info.widget_id or "").startswith(
                                    "file-"
                                )
                                for parsed_ctx in successes
                            )
                            or has_file_like_tool_result
                        )

                        # Add instruction for LLM
                        if execution_type == "sql":
                            content += (
                                "Review the output above. "
                                "If it succeeded, analyze the data and respond. "
                                "If it failed, fix the query and try again. "
                                "After using the data, you MUST include citation tags "
                                "in the final answer using "
                                "`<|start_citation_id|>...<|end_citation_id|>`. "
                                "Do NOT call llm_complete until you include at "
                                "least one citation when data is used."
                            )
                        elif execution_type == "python":
                            content += (
                                "Review the output above. "
                                "If it succeeded, analyze the data and respond. "
                                "If it failed, you MUST fix the code and call "
                                "llm_generate_python_code again. "
                                "Do NOT respond with an error — always retry."
                            )
                        else:
                            if requires_uploaded_files_query:
                                content += (
                                    "The loaded result contains uploaded file "
                                    "references, not the final file content. "
                                    "You MUST call `_llm_query_uploaded_files` next "
                                    "to read the file data and answer the user. "
                                    "Do NOT call `llm_query_widgets` again for this "
                                    "same file request. After using file content, "
                                    "you MUST include citation tags in the final "
                                    "answer using "
                                    "`<|start_citation_id|>...<|end_citation_id|>`. "
                                    "Do NOT call llm_complete until you include at "
                                    "least one citation when file content is used."
                                )
                            else:
                                content += (
                                    "You can now query or fetch this data to access "
                                    "it. If the user asked what data/results/rows "
                                    "were returned, call `llm_query_structured_data` "
                                    "now to inspect the loaded rows before answering. "
                                    "Do NOT call `llm_complete` until you have used "
                                    "the loaded data. This fetch did not persist "
                                    "widget or dashboard changes."
                                )
                        if (
                            hasattr(message, "extra_state")
                            and isinstance(message.extra_state, dict)
                            and "not_found_data_sources" in message.extra_state
                        ):
                            content += (
                                "\nExtra information received from the function: "
                            )
                            not_found_data_sources_messages = message.extra_state[
                                "not_found_data_sources"
                            ]
                            content += ", ".join(not_found_data_sources_messages)

                    if failures:
                        content += "\n\n----------\n\n"
                        content += "Failed to load some data:\n\n"
                        for failure in failures:
                            content += f"- {failure}\n"

                        if is_python_code_result:
                            content += (
                                "\nYou MUST fix the code and call "
                                "llm_generate_python_code again. "
                                "Do NOT respond with an error — always retry."
                            )
                        elif is_sql_query_result:
                            content += (
                                "\nYou MUST fix the query and try again. "
                                "Do NOT respond with an error — always retry."
                            )
                        elif not successes:
                            content += (
                                "\nNo data was loaded from this request. You are "
                                "unable to answer from this widget or file. Do NOT "
                                "call llm_query_widgets again for the same data "
                                "source. Tell the user the data could not be "
                                "retrieved and, when relevant, ask them to refresh "
                                "the widget, rerun the request, or provide a valid "
                                "file/link."
                            )

                case "add_widget_to_dashboard" | "update_widget_in_dashboard":
                    details: list[dict[str, Any] | str] = []
                    event_type: Literal["ERROR", "WARNING", "INFO"] = "INFO"
                    for _, data in zip(
                        (message.input_arguments or {}).get("data_sources", []),
                        message.data,
                        strict=True,
                    ):
                        result = cast(ClientCommandResult, data)
                        if is_last_message:
                            event_type = (
                                "ERROR"
                                if result.status == "error"
                                else "WARNING"
                                if result.status == "warning"
                                else "INFO"
                            )
                        details.append(
                            {
                                "Status": result.status,
                                "Message": result.message or "",
                            }
                        )
                        content += result.message or ""

                    # If this update came from SQL generation, inform
                    # the LLM so it knows the widget now has data and
                    # can query it via llm_query_widgets if needed.
                    is_sql_update = (
                        message.function
                        in (
                            "update_widget_in_dashboard",
                            "add_widget_to_dashboard",
                        )
                        and hasattr(message, "extra_state")
                        and isinstance(message.extra_state, dict)
                        and "sql_query" in message.extra_state
                    )
                    if is_sql_update:
                        sql_query = message.extra_state.get("sql_query", "")
                        content += (
                            f"\n\nThe SQL query has been applied to the widget. "
                            f"Query: `{sql_query[:200]}`. "
                            "The widget now displays the queried data. "
                            "Use `llm_query_widgets` to fetch this data "
                            "if you need it for further analysis."
                        )

                    if is_last_message and details:
                        yield StatusUpdateSSE(
                            data=StatusUpdateSSEData(
                                eventType=event_type,
                                message="Results from dashboard manipulation",
                                details=details,
                            )
                        )
                case "add_generative_widget":
                    gen_details: list[dict[str, Any] | str] = []
                    gen_event_type: Literal["ERROR", "WARNING", "INFO"] = "INFO"
                    for data in message.data:
                        result = cast(ClientCommandResult, data)
                        if is_last_message:
                            gen_event_type = (
                                "ERROR"
                                if result.status == "error"
                                else "WARNING"
                                if result.status == "warning"
                                else "INFO"
                            )
                        gen_details.append(
                            {
                                "Status": result.status,
                                "Message": result.message or "",
                            }
                        )
                        content += result.message or ""

                    if is_last_message and gen_details:
                        yield StatusUpdateSSE(
                            data=StatusUpdateSSEData(
                                eventType=gen_event_type,
                                message="Results from generative widget creation",
                                details=gen_details,
                            )
                        )
                case "assign_tasks_to_agents":
                    _details: list[dict[str, Any]] = []
                    _event_type: Literal["ERROR", "WARNING", "INFO"] = "INFO"
                    for _, data in zip(
                        (message.input_arguments or {}).get("task_requests", []),
                        message.data,
                        strict=False,
                    ):
                        result = cast(ClientCommandResult, data)
                        if is_last_message:
                            _event_type = (
                                "ERROR"
                                if result.status == "error"
                                else "WARNING"
                                if result.status == "warning"
                                else "INFO"
                            )
                        _details.append(
                            {
                                "Status": result.status,
                                "Message": result.message or "",
                            }
                        )
                        content += result.message or ""

                    if not message.data:
                        content = (
                            "The assigned agent has already "
                            "responded directly to the user. "
                            "Do NOT repeat, summarize, or "
                            "comment on the agent's response. "
                            "Call _llm_complete immediately."
                        )

                    if is_last_message and _details:
                        yield StatusUpdateSSE(
                            data=StatusUpdateSSEData(
                                eventType=_event_type,
                                message="Results from task execution",
                                details=_details,  # type: ignore[arg-type]
                            )
                        )

                case "execute_agent_tool":
                    tool_name = (message.input_arguments or {}).get(
                        "tool_name", "Unknown tool"
                    )
                    server_id = (message.input_arguments or {}).get(
                        "server_id", "Unknown server"
                    )
                    mcp_tool_diagnostics = {}
                    if isinstance(message.extra_state, dict):
                        raw_mcp_tool_diagnostics = message.extra_state.get(
                            "mcp_tool_diagnostics"
                        )
                        if isinstance(raw_mcp_tool_diagnostics, dict):
                            mcp_tool_diagnostics = raw_mcp_tool_diagnostics

                    # Collect all MCP response content for message
                    mcp_response_content = ""
                    all_raw_content = []

                    # Always populate content so the LLM can see historical tool
                    # results (prevents infinite loops with OpenAI models)
                    for data in message.data:
                        if not isinstance(data, DataContent):
                            continue
                        for item in data.items:
                            if item.content:
                                content += (
                                    f"\n\nResult from {tool_name}: {item.content}"
                                )
                                mcp_response_content += str(item.content)

                                # Normalize the MCP response: recursively
                                # unwrap single-element lists and
                                # double-encoded JSON strings.
                                normalized = normalize_jsonish(item.content)
                                all_raw_content.append(normalized)
                            else:
                                # Handle empty content consistently
                                content += item.content or ""
                                mcp_response_content += str(item.content or "")

                    # Only steer continuation for the latest MCP result in this cycle.
                    # Historical MCP results are replayed for context and should not
                    # repeatedly inject execution directives.
                    if is_last_message and mcp_response_content:
                        content += (
                            "\n\nThe MCP tool call has completed. Continue executing "
                            "your plan immediately. If the user's request is not "
                            "fully resolved, call the next relevant tool now "
                            "instead of narrating intent. Only call "
                            "`_llm_complete` when you are ready to provide the "
                            "final answer."
                            "\nReminder: when calling MCP tools, pass ALL "
                            "required parameters directly as function arguments. "
                            "The `summary` field is display-only."
                        )

                    # Determine the final raw content for artifact creation
                    raw_content = all_raw_content[-1] if all_raw_content else None
                    if is_last_message and raw_content is not None:
                        self._logging_service.debug(
                            "mcp_tool_result_received",
                            extra={
                                "event": "mcp_tool_result_received",
                                "tool_name": tool_name,
                                "server_id": server_id,
                                "tool_attempt_id": mcp_tool_diagnostics.get(
                                    "tool_attempt_id"
                                ),
                                "raw_content_type": type(raw_content).__name__,
                                "provided_arg_keys": mcp_tool_diagnostics.get(
                                    "provided_arg_keys", []
                                ),
                                "required_args": mcp_tool_diagnostics.get(
                                    "required_args", []
                                ),
                                "missing_required_args": mcp_tool_diagnostics.get(
                                    "missing_required_args", []
                                ),
                            },
                        )

                    if (
                        is_last_message
                        and raw_content is not None
                        and self._mcp_data_service._is_mcp_error(raw_content)
                    ):
                        extracted_error = self._mcp_data_service._get_mcp_error_content(
                            raw_content
                        )
                        error_details = extracted_error or "Unknown MCP error"
                        self._logging_service.warning(
                            "mcp_error_detected",
                            extra={
                                "event": "mcp_error_detected",
                                "tool_name": tool_name,
                                "server_id": server_id,
                                "tool_attempt_id": mcp_tool_diagnostics.get(
                                    "tool_attempt_id"
                                ),
                                "error_details": error_details,
                                "required_args": mcp_tool_diagnostics.get(
                                    "required_args", []
                                ),
                                "provided_arg_keys": mcp_tool_diagnostics.get(
                                    "provided_arg_keys", []
                                ),
                                "missing_required_args": mcp_tool_diagnostics.get(
                                    "missing_required_args", []
                                ),
                            },
                        )
                        content += (
                            f"\n\nMCP tool call failed.\nError: {error_details}\n"
                        )

                        if extracted_error:
                            self._logging_service.debug(
                                "mcp_retry_guidance_issued",
                                extra={
                                    "event": "mcp_retry_guidance_issued",
                                    "tool_name": tool_name,
                                    "server_id": server_id,
                                    "tool_attempt_id": mcp_tool_diagnostics.get(
                                        "tool_attempt_id"
                                    ),
                                    "retry_allowed": True,
                                    "missing_required_args": mcp_tool_diagnostics.get(
                                        "missing_required_args", []
                                    ),
                                },
                            )
                            content += (
                                "MCP failure handling overrides the previous "
                                "continuation instruction. If this is an "
                                "unexpected runtime/service error (for example "
                                "quota, billing, authentication, rate limit, "
                                "timeout, or server failure), do not retry "
                                "immediately and do not call fallback tools "
                                "such as `_llm_web_search`. In your next "
                                "user-facing reply, state which MCP tool "
                                "failed and why. If the error text includes "
                                "a suggested fix, next step, or URL, include "
                                "that solution explicitly. Then ask whether the user "
                                "wants you to retry, change parameters, or "
                                "use another source such as web search. After "
                                "asking, call `_llm_complete`. If this is only "
                                "a locally correctable argument-validation "
                                "error and the MCP schema provides enough "
                                "information, retry the tool exactly once with "
                                "corrected arguments. Do not reuse the same "
                                "arguments. Only skip `_llm_complete` when "
                                "you are making that corrected MCP retry."
                            )
                        else:
                            self._logging_service.debug(
                                "mcp_retry_guidance_issued",
                                extra={
                                    "event": "mcp_retry_guidance_issued",
                                    "tool_name": tool_name,
                                    "server_id": server_id,
                                    "tool_attempt_id": mcp_tool_diagnostics.get(
                                        "tool_attempt_id"
                                    ),
                                    "retry_allowed": False,
                                    "missing_required_args": mcp_tool_diagnostics.get(
                                        "missing_required_args", []
                                    ),
                                },
                            )
                            content += (
                                "No actionable error details were returned. "
                                "Do not blindly retry with guessed arguments. "
                                "Do not call fallback tools such as "
                                "`_llm_web_search` yet. Explain the failure "
                                "and ask the user whether they want you to "
                                "retry, provide missing inputs, or use another "
                                "source."
                            )

                    # Process MCP response and yield artifacts only for last message
                    if (
                        is_last_message
                        and mcp_response_content
                        and tool_name != "Unknown tool"
                        and raw_content is not None
                    ):
                        async for (
                            mcp_event
                        ) in self._mcp_data_service.process_mcp_tool_response(
                            tool_name=tool_name,
                            server_id=server_id,
                            raw_content=raw_content,
                            is_last_message=is_last_message,
                        ):
                            yield mcp_event

                case "get_skill_content":
                    # Handle skill content retrieved from the frontend
                    args = message.input_arguments or {}
                    slug = args.get("slug", "unknown")
                    skill_md_for_details = ""
                    for data in message.data:
                        result = cast(ClientCommandResult, data)
                        result_data = result.data
                        if (
                            result.status == "success"
                            and result_data is not None
                            and "skill" in result_data
                        ):
                            skill_data = result_data["skill"]
                            skill_slug = skill_data.get("slug", slug)
                            skill_desc = skill_data.get("description", "")
                            skill_md = skill_data.get("contentMarkdown", "")
                            skill_md_for_details = skill_md
                            content += f"\n\n## Skill Instructions: {skill_slug}\n"
                            content += f"**Description:** {skill_desc}\n\n"
                            content += skill_md

                            # Remove from catalog so the system prompt
                            # no longer lists it as "available to fetch".
                            # The content already lives in chat history.
                            self._remove_fetched_skill_from_catalog(skill_slug)
                        elif result.status == "error":
                            err = result.message or "Unknown error"
                            content += f"\n\nSkill '{slug}' not found: {err}"
                        else:
                            msg = result.message or "No content available"
                            content += f"\n\nSkill '{slug}': {msg}"

                    if is_last_message:
                        skill_details: list[dict[str, Any] | str] | None = (
                            [skill_md_for_details] if skill_md_for_details else None
                        )
                        yield StatusUpdateSSE(
                            data=StatusUpdateSSEData(
                                eventType="INFO",
                                message=f"Loaded skill: {slug}",
                                details=skill_details,
                            )
                        )

                case "save_skill":
                    # Handle the result of saving a skill on the frontend
                    saved_slug = ""
                    saved_description = ""
                    for data in message.data:
                        result = cast(ClientCommandResult, data)
                        result_data = result.data
                        if (
                            result.status == "success"
                            and result_data is not None
                            and isinstance(result_data.get("skill"), dict)
                        ):
                            skill_data = result_data["skill"]
                            saved_slug = skill_data.get("slug", "")
                            saved_description = skill_data.get("description", "")
                            content += (
                                f"\n\nThe conversation workflow was saved as the"
                                f" skill `/{saved_slug}`"
                            )
                            if saved_description:
                                content += f": {saved_description}"
                            content += (
                                "\n\nBriefly confirm to the user that the skill"
                                f" was saved and that they can reuse it by typing"
                                f" `/{saved_slug}` in the chat, or edit it in the"
                                " AI settings under Skills."
                            )
                        elif result.status == "error":
                            err = result.message or "Unknown error"
                            content += (
                                f"\n\nFailed to save the skill: {err}."
                                " Inform the user and suggest they try again or"
                                " create the skill manually in the AI settings"
                                " under Skills."
                            )
                        else:
                            msg = result.message or "The skill was not saved."
                            content += f"\n\n{msg}"

                    if is_last_message:
                        yield StatusUpdateSSE(
                            data=StatusUpdateSSEData(
                                eventType="INFO" if saved_slug else "WARNING",
                                message=(
                                    f"Skill saved: /{saved_slug}"
                                    if saved_slug
                                    else "Skill not saved"
                                ),
                                details=(
                                    [
                                        {
                                            "skill_slug": saved_slug,
                                            "skill_description": saved_description,
                                        }
                                    ]
                                    if saved_slug
                                    else None
                                ),
                            )
                        )

            chat_messages.append(
                FunctionResultMessage(
                    content=content,
                    # FunctionCall object must the same as the message
                    # preceding it (under-the-hood, we need to make sure
                    # their IDs match for OpenAI. Magentic does this
                    # using for us, but we have to make sure we use the
                    # same object!)
                    function_call=function_call,
                )
            )
            yield chat_messages
            return
        raise HTTPException(
            status_code=500,
            detail="Attempted to parse function call result message without a preceding function call message.",  # noqa: E501
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using character heuristic.

        Uses Anthropic's recommended heuristic of 1 token ≈ 3.5 characters.
        This is provider-agnostic and avoids dependency on specific tokenizers.
        """
        return int(len(text) / 3.5)

    def _count_tokens(self, messages: Sequence[Message]) -> int:
        """Estimate total tokens in messages using character heuristic."""
        message_totals = []
        for message in messages:
            if isinstance(message, Message):
                message_totals.append(self._estimate_tokens(str(message.content)))
        total = sum(message_totals)
        self._logging_service.debug("Token estimate: %s", total)
        return total

    def _log_token_breakdown(
        self,
        messages: Sequence[Message],
        widget_collection: WidgetCollection | None = None,
        documents: list[Document] | None = None,
        web_pages: list[WebContext] | None = None,
    ) -> None:
        """Log detailed token usage by component for debugging context limit issues."""
        # System prompt tokens (first message)
        system_tokens = 0
        if messages and isinstance(messages[0], SystemMessage):
            system_tokens = self._estimate_tokens(str(messages[0].content))

        # Conversation history tokens (all messages except system)
        history_tokens = 0
        history_message_count = 0
        for msg in messages[1:]:
            if isinstance(msg, Message):
                history_tokens += self._estimate_tokens(str(msg.content))
                history_message_count += 1

        # Widget tokens (broken down by type)
        widget_tokens = {"primary": 0, "secondary": 0, "extra": 0, "total": 0}
        widget_counts = {"primary": 0, "secondary": 0, "extra": 0}
        if widget_collection:
            if widget_collection.primary:
                widget_tokens["primary"] = self._estimate_tokens(
                    str(widget_collection.primary)
                )
                widget_counts["primary"] = len(widget_collection.primary)
            if widget_collection.secondary:
                widget_tokens["secondary"] = self._estimate_tokens(
                    str(widget_collection.secondary)
                )
                widget_counts["secondary"] = len(widget_collection.secondary)
            if widget_collection.extra:
                widget_tokens["extra"] = self._estimate_tokens(
                    str(widget_collection.extra)
                )
                widget_counts["extra"] = len(widget_collection.extra)
            widget_tokens["total"] = (
                widget_tokens["primary"]
                + widget_tokens["secondary"]
                + widget_tokens["extra"]
            )

        # RAG document tokens (estimate based on binary content size)
        doc_tokens = 0
        doc_count = 0
        if documents:
            for doc in documents:
                # Document.content is bytes; estimate ~4 bytes per token
                doc_tokens += len(doc.content) // 4
            doc_count = len(documents)

        # Web page tokens
        web_tokens = 0
        web_count = 0
        if web_pages:
            for page in web_pages:
                # WebContext has url and content attributes
                web_tokens += self._estimate_tokens(f"{page.url} {page.content}")
            web_count = len(web_pages)

        total_tokens = (
            system_tokens
            + history_tokens
            + widget_tokens["total"]
            + doc_tokens
            + web_tokens
        )

        self._logging_service.info(
            "Token breakdown (estimated): "
            f"Total={total_tokens:,} | "
            f"System={system_tokens:,} | "
            f"History={history_tokens:,} ({history_message_count} msgs) | "
            f"Widgets={widget_tokens['total']:,} "
            f"(primary={widget_tokens['primary']:,} [{widget_counts['primary']}], "
            f"secondary={widget_tokens['secondary']:,} [{widget_counts['secondary']}], "
            f"extra={widget_tokens['extra']:,} [{widget_counts['extra']}]) | "
            f"Docs={doc_tokens:,} ({doc_count}) | "
            f"Web={web_tokens:,} ({web_count})"
        )

    def _log_request_diagnostics(
        self,
        messages: Sequence[Message] | None,
        functions: list[Callable] | None = None,
        request_id: str | None = None,
    ) -> None:
        """Log detailed request diagnostics for debugging 500 errors.

        Logs request characteristics without the full message content to help
        identify patterns in failing requests (size, message types, data samples).

        This method is wrapped in try-except to ensure logging never crashes
        the main request flow.
        """
        try:
            # Generate request ID for correlation if not provided
            req_id = request_id or str(self._logging_service.trace_id)[:8]

            # Handle empty/None messages
            if not messages:
                self._logging_service.info(
                    "[REQ:%s] LLM Request Diagnostics - No messages to log",
                    req_id,
                )
                return

            # Count messages by type
            type_counts: dict[str, int] = {}
            type_sizes: dict[str, int] = {}
            message_details: list[tuple[str, int, str]] = []  # (type, size, preview)

            for _i, msg in enumerate(messages):
                if msg is None:
                    continue
                msg_type = type(msg).__name__
                try:
                    content = str(getattr(msg, "content", "") or "")
                except Exception:
                    content = "(error reading content)"
                content_len = len(content)

                type_counts[msg_type] = type_counts.get(msg_type, 0) + 1
                type_sizes[msg_type] = type_sizes.get(msg_type, 0) + content_len

                # Track individual message info for finding largest
                preview = content[:100].replace("\n", " ") if content else "(empty)"
                message_details.append((msg_type, content_len, preview))

            # Sort to find top 5 largest messages
            largest_messages = sorted(
                message_details, key=lambda x: x[1], reverse=True
            )[:5]

            # Estimate JSON payload size (rough approximation)
            try:
                # Build a simplified structure similar to OpenAI request
                payload_estimate = {
                    "messages": [
                        {
                            "role": type(m).__name__,
                            "content": str(getattr(m, "content", "") or ""),
                        }
                        for m in messages
                        if m is not None
                    ],
                }
                estimated_payload_bytes = len(
                    json.dumps(payload_estimate).encode("utf-8")
                )
            except Exception:
                # Fallback: sum up content lengths
                estimated_payload_bytes = sum(type_sizes.values())

            # Check for data samples in content (common pattern in prompts)
            data_sample_info = []
            for i, msg in enumerate(messages):
                if msg is None:
                    continue
                try:
                    content = str(getattr(msg, "content", "") or "")
                    if "DATA SAMPLE" in content or "data_sample" in content.lower():
                        data_sample_info.append(
                            f"msg[{i}]:{type(msg).__name__}:{len(content)} chars"
                        )
                except Exception:
                    # If we can't get the content, just skip it but log it
                    self._logging_service.warning(
                        f"[REQ:{req_id}] Could not get content for msg[{i}]:{type(msg).__name__}"  # noqa: E501
                    )
                    pass

            # Log the diagnostic summary
            self._logging_service.info(
                "[REQ:%s] LLM Request Diagnostics - "
                "Payload: ~%.1f KB | "
                "Messages: %d total | "
                "Functions: %d",
                req_id,
                estimated_payload_bytes / 1024,
                len(messages),
                len(functions) if functions else 0,
            )

            # Log message type breakdown
            type_summary = ", ".join(
                f"{t}={c} (~{type_sizes.get(t, 0) / 1024:.1f}KB)"
                for t, c in sorted(type_counts.items())
            )
            self._logging_service.info(
                "[REQ:%s] Message types: %s",
                req_id,
                type_summary,
            )

            # Log top 5 largest messages
            self._logging_service.info(
                "[REQ:%s] Top 5 largest messages:",
                req_id,
            )
            for idx, (msg_type, size, preview) in enumerate(largest_messages, 1):
                self._logging_service.info(
                    "[REQ:%s]   %d. %s: %d chars - %.50s...",
                    req_id,
                    idx,
                    msg_type,
                    size,
                    preview,
                )

            # Log data sample warnings if present
            if data_sample_info:
                self._logging_service.info(
                    "[REQ:%s] DATA SAMPLES found in: %s",
                    req_id,
                    ", ".join(data_sample_info),
                )

            # Log function names if present
            if functions:
                try:
                    func_names = [getattr(f, "__name__", str(f)) for f in functions]
                    self._logging_service.info(
                        "[REQ:%s] Functions: %s",
                        req_id,
                        ", ".join(func_names),
                    )
                except Exception:
                    self._logging_service.info(
                        "[REQ:%s] Functions: %d (names unavailable)",
                        req_id,
                        len(functions),
                    )

        except Exception as e:
            # Logging should never crash the main flow
            self._logging_service.warning(
                "Failed to log request diagnostics: %s", str(e)
            )

    def _is_within_context_limit(self, messages: Sequence[Message]) -> bool:
        model = self._get_model().model
        # Leave some wiggle room for inaccuracies, tool descriptions, etc.
        context_limit = (
            CONTEXT_LIMIT_BY_MODEL.get(model, CONTEXT_LIMIT_UNDEFINED_MODEL)
            * CONTEXT_LIMIT_SAFETY_FACTOR
        )
        return self._count_tokens(messages=messages) <= context_limit

    async def _compose_chain(
        self,
        chat_messages: list[Message[Any]],
        documents: list[Document] | None = None,
        web_pages: list[WebContext] | None = None,
        tools: list[AgentTool] | None = None,
        original_messages: (
            list[LlmClientFunctionCallResultMessage | LlmClientMessage] | None
        ) = None,
        original_context: list[RawContext] | None = None,
        widget_collection: WidgetCollection | None = None,
        app_widget_collection: WidgetCollection | None = None,
        sql_widgets: list[SqlWidgetContext] | None = None,
        python_widgets: list[PythonWidgetContext] | None = None,
        semantic_views_for_prompt: list[str] | None = None,
    ) -> Callable:
        """Compose the agent and message chain that will be completed by the LLM."""

        system_prompt = SystemMessage(
            sanitize_str(
                self._template_service.render_copilot_system_prompt(
                    widget_collection=widget_collection,
                    unstructured_context=self._context_service.unstructured_context,
                    structured_context=self._context_service.structured_context,
                    documents=documents,
                    web_pages=web_pages,
                    tools=tools,
                    sql_widgets=sql_widgets if sql_widgets else None,
                    python_widgets=python_widgets if python_widgets else None,
                    skills_catalog=self._skills_catalog or None,
                    selected_skills=self._selected_skills or None,
                    semantic_views=semantic_views_for_prompt,
                )
            )
        )
        # Normalize the most recent user message into the user-query template.
        # If this turn already produced an enhanced query, prefer that text so
        # later LLM calls continue from the clarified interpretation instead of
        # re-enhancing the original user message again.
        last_user_index = next(
            (
                index
                for index in range(len(chat_messages) - 1, -1, -1)
                if isinstance(chat_messages[index], UserMessage)
            ),
            None,
        )
        if last_user_index is not None:
            last_message = cast(UserMessage, chat_messages[last_user_index])
            raw_query_text = next(
                (
                    message.content
                    for message in reversed(original_messages or [])
                    if isinstance(message, LlmClientMessage)
                    and message.role == "human"
                    and isinstance(message.content, str)
                ),
                str(last_message.content),
            )
            query_text = raw_query_text
            if enhanced_query := self._extract_current_turn_enhanced_query(
                chat_messages
            ):
                query_text = enhanced_query
            new_message_content = self._template_service.render_copilot_user_prompt(
                query=query_text
            )
            chat_messages[last_user_index] = UserMessage(
                sanitize_str(new_message_content)
            )

        messages: list[Message[Any]] = []
        messages.append(system_prompt)
        messages += chat_messages

        # Log detailed token breakdown for debugging context limit issues
        self._log_token_breakdown(
            messages=messages,
            widget_collection=widget_collection,
            documents=documents,
            web_pages=web_pages,
        )

        self._set_prompt_enhancement_context(
            messages=original_messages,
            context=original_context,
            widget_collection=widget_collection,
            tools=tools,
        )
        functions = self._build_llm_functions(
            messages=messages,
            documents=documents,
            tools=tools,
            widget_collection=widget_collection,
            app_widget_collection=app_widget_collection,
            sql_widgets=sql_widgets,
            python_widgets=python_widgets,
        )
        self._raise_if_openai_tool_limit_exceeded(
            total_tool_count=len(functions),
            mcp_tool_count=len(self._flat_mcp_functions),
        )

        if not self._is_within_context_limit(messages=messages):
            raise ContextLimitExceededError("The context limit has been exceeded.")

        async def copilot() -> AsyncStreamedResponse: ...  # type: ignore

        # Log request diagnostics for debugging 500 errors
        # (replaces verbose message dump)
        self._log_request_diagnostics(messages=messages, functions=functions)

        chain = chatprompt(
            *messages,
            functions=functions if functions else None,
            model=self._get_model(),
            max_retries=3,
        )(copilot)
        chain_with_retry = retry_on_exception(
            max_retries=3,
            exceptions=(httpx.RemoteProtocolError,),
        )(chain)
        return chain_with_retry

    def _build_llm_functions(
        self,
        messages: Sequence[Message[Any]],
        documents: list[Document] | None,
        tools: list[AgentTool] | None,
        widget_collection: WidgetCollection | None,
        sql_widgets: list[SqlWidgetContext] | None,
        python_widgets: list[PythonWidgetContext] | None,
        app_widget_collection: WidgetCollection | None = None,
    ) -> list[Callable[..., AsyncGenerator]]:
        """Build the tool list for the current turn.

        This is shared by the normal LLM call path and deferred queue
        restoration so both flows resolve function names the same way.
        """
        functions: list[Callable[..., AsyncGenerator]] = []

        # Set up prompt enhancement if available
        if self._prompt_enhancement_service and not (
            self._current_turn_has_prompt_enhancement(messages)
        ):
            functions.append(self._native_function_call_service.llm_enhance_prompt)

        if not isinstance(messages[-1], UserMessage):
            functions.append(self._native_function_call_service.llm_complete)
        if not self._current_turn_has_llm_think(messages):
            functions.append(self._native_function_call_service._llm_think)
        if documents:
            functions.append(
                self._native_function_call_service.llm_query_uploaded_files
            )
        if self._context_service.structured_context:
            functions.append(
                self._native_function_call_service.llm_query_structured_data
            )
        if self._context_service.unstructured_context:
            functions.append(
                self._native_function_call_service.llm_query_unstructured_data
            )

        # Add web search tool if it's enabled via workspace option
        if "workspace-web-search" in self._workspace_options:
            functions.append(self._native_function_call_service.llm_web_search)

        # Add table creation from text function
        functions.append(self._native_function_call_service.llm_create_table_from_text)

        # Add chart creation from table function
        functions.append(self._native_function_call_service.llm_create_chart_from_table)

        # Add HTML artifact creation function
        functions.append(self._native_function_call_service.llm_create_html_artifact)

        if self._flat_mcp_functions:
            functions.extend(self._flat_mcp_functions)

        # Register skills content retrieval function if skills catalog is available
        if self._skills_catalog:
            functions.append(
                wrapped_partial(
                    self._client_function_call_service.llm_get_skill_content,
                    extra_state=None,
                )
            )

        # Always allow saving the current conversation's workflow as a skill
        functions.append(
            wrapped_partial(
                self._client_function_call_service.llm_save_skill,
                extra_state=None,
            )
        )

        app_catalog = app_widget_collection or widget_collection
        self._native_function_call_service.set_app_widget_collection(app_catalog)
        if app_catalog and (
            app_catalog.primary or app_catalog.secondary or app_catalog.extra
        ):
            functions.append(self._native_function_call_service.llm_search_widgets)
            functions.append(self._native_function_call_service.llm_create_app)

        if widget_collection:
            # to enabled/disable these functions from the FE
            if widget_collection.primary or widget_collection.secondary:
                functions.append(
                    masked_partial(
                        self._native_function_call_service.llm_get_widget_input_state,
                        widget_collection=widget_collection,
                    )
                )
                functions.append(
                    wrapped_partial(
                        self._client_function_call_service.llm_query_widgets,
                        extra_state=None,
                    )
                )

            if "widget-global-search" in self._workspace_options:
                # Global widget search enabled
                functions.append(
                    wrapped_partial(
                        self._client_function_call_service.llm_query_extra_widgets,
                        extra_state=None,
                    )
                )

            is_on_dashboard = (
                self._workspace_state is not None
                and getattr(self._workspace_state, "current_page_context", None)
                == "dashboard"
            )
            if "generative-ui" in self._workspace_options and is_on_dashboard:
                # Generative UI enabled (only on dashboard pages)
                functions.append(
                    wrapped_partial(
                        self._client_function_call_service.llm_update_widget_in_dashboard,
                        extra_state=None,
                    )
                )
                functions.append(
                    wrapped_partial(
                        self._client_function_call_service.llm_add_widget_to_dashboard,
                        extra_state=None,
                    )
                )
                functions.append(
                    self._client_function_call_service.llm_manage_navigation_bar
                )
                # Add generative widget creation for charts/tables directly on dashboard
                functions.append(
                    wrapped_partial(
                        self._client_function_call_service.llm_generate_widget_in_dashboard,
                        extra_state=None,
                    )
                )

            if "agent-orchestration" in self._workspace_options:
                functions.append(
                    self._client_function_call_service.llm_assign_tasks_to_agents
                )

        # Register SQL query generation function if SQL-enabled widgets were found
        if sql_widgets and self._sql_query_generation_service:
            if self._native_function_call_service.is_snowflake_mode:
                # Snowflake mode: register Cortex Analyst tool for writing
                # SQL queries, and the generic tool for reading existing
                # table data
                self._logging_service.info(
                    "Snowflake SQL query generation enabled for %d widgets",
                    len(sql_widgets),
                )
                functions.append(
                    masked_partial(
                        self._native_function_call_service.llm_generate_sql_query_snowflake,
                        sql_widgets=sql_widgets,
                    )
                )
            self._logging_service.info(
                "SQL query generation enabled for %d widgets", len(sql_widgets)
            )
            functions.append(
                masked_partial(
                    self._native_function_call_service.llm_generate_sql_query,
                    sql_widgets=sql_widgets,
                )
            )

        # Register Python code generation function if code-enabled widgets were found
        if python_widgets and self._python_code_generation_service:
            self._logging_service.info(
                "Python code generation enabled for %d widgets", len(python_widgets)
            )
            functions.append(
                masked_partial(
                    self._native_function_call_service.llm_generate_python_code,
                    python_widgets=python_widgets,
                    sql_widgets=sql_widgets,
                )
            )

        return functions

    def _build_function_registry(
        self,
        messages: Sequence[Message[Any]],
        documents: list[Document] | None,
        tools: list[AgentTool] | None,
        original_messages: list[LlmClientFunctionCallResultMessage | LlmClientMessage],
        original_context: list[RawContext] | None,
        widget_collection: WidgetCollection | None,
        sql_widgets: list[SqlWidgetContext] | None,
        python_widgets: list[PythonWidgetContext] | None,
        app_widget_collection: WidgetCollection | None = None,
    ) -> dict[str, Callable[..., AsyncGenerator]]:
        """Map function names to the exact callables available in this turn."""
        self._set_prompt_enhancement_context(
            messages=original_messages,
            context=original_context,
            widget_collection=widget_collection,
            tools=tools,
        )
        return {
            function.__name__: function
            for function in self._build_llm_functions(
                messages=messages,
                documents=documents,
                tools=tools,
                widget_collection=widget_collection,
                app_widget_collection=app_widget_collection,
                sql_widgets=sql_widgets,
                python_widgets=python_widgets,
            )
        }

    def _set_prompt_enhancement_context(
        self,
        messages: list[LlmClientFunctionCallResultMessage | LlmClientMessage] | None,
        context: list[RawContext] | None,
        widget_collection: WidgetCollection | None,
        tools: list[AgentTool] | None,
    ) -> None:
        """Prime prompt enhancement so deferred and live turns resolve identically."""
        if self._prompt_enhancement_service:
            self._prompt_enhancement_service.set_current_context(
                messages=messages,
                context=context,
                widgets=widget_collection,
                tools=tools,
            )

    def _serialize_function_call_value(self, value: Any) -> Any:
        """Convert deferred function-call payloads into JSON-safe values."""
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, dict):
            return {
                str(key): self._serialize_function_call_value(val)
                for key, val in value.items()
            }
        if isinstance(value, list | tuple):
            return [self._serialize_function_call_value(item) for item in value]
        if isinstance(value, BaseModel):
            return self._serialize_function_call_value(value.model_dump(mode="json"))
        return value

    def _serialize_function_call(
        self, function_call: FunctionCall
    ) -> DeferredFunctionCall:
        """Store a deferred call using only its public name and arguments."""
        return DeferredFunctionCall(
            function_name=function_call.function.__name__,
            arguments=self._serialize_function_call_value(function_call.arguments),
        )

    def _build_function_call_error_messages(
        self, function_call: FunctionCall, error: Exception
    ) -> list[Message[Any]]:
        """Represent a tool failure as assistant/tool messages for the next LLM turn."""
        error_type = type(error).__name__
        error_message = sanitize_str(str(error))
        content = (
            f"Tool call `{function_call.function.__name__}` failed with "
            f"{error_type}: {error_message}"
        )
        return [
            AssistantMessage(function_call),
            FunctionResultMessage(
                content=content,
                function_call=function_call,
            ),
        ]

    def _serialize_function_calls(
        self, function_calls: Sequence[FunctionCall]
    ) -> list[DeferredFunctionCall]:
        """Serialize a queue tail for storage in extra_state."""
        return [
            self._serialize_function_call(function_call)
            for function_call in function_calls
        ]

    def _get_deferred_function_call_specs(
        self,
        messages: list[LlmClientMessage | LlmClientFunctionCallResultMessage],
    ) -> list[DeferredFunctionCall]:
        if (
            messages
            and isinstance(messages[-1], LlmClientFunctionCallResultMessage)
            and messages[-1].extra_state
        ):
            raw_deferred_specs = messages[-1].extra_state.get(
                self._DEFERRED_FUNCTION_CALLS_KEY, []
            )
            if isinstance(raw_deferred_specs, list):
                deferred_specs: list[DeferredFunctionCall] = []
                for raw_spec in raw_deferred_specs:
                    try:
                        deferred_specs.append(
                            DeferredFunctionCall.model_validate(raw_spec)
                        )
                    except ValidationError:
                        self._logging_service.warning(
                            "Skipping invalid deferred function call spec: %s",
                            raw_spec,
                        )
                return deferred_specs
        return []

    def _restore_deferred_function_calls(
        self,
        deferred_specs: Sequence[DeferredFunctionCall | dict[str, Any]],
        function_registry: dict[str, Callable[..., AsyncGenerator]],
    ) -> list[FunctionCall]:
        """Rebuild deferred tool calls from extra_state for the next request."""
        restored_calls: list[FunctionCall] = []

        for raw_deferred_spec in deferred_specs:
            try:
                deferred_spec = (
                    raw_deferred_spec
                    if isinstance(raw_deferred_spec, DeferredFunctionCall)
                    else DeferredFunctionCall.model_validate(raw_deferred_spec)
                )
            except ValidationError:
                self._logging_service.warning(
                    "Skipping invalid deferred function call spec during restore: %s",
                    raw_deferred_spec,
                )
                continue
            function_name = deferred_spec.function_name
            arguments = deferred_spec.arguments
            function = function_registry.get(function_name)

            if function is None:
                self._logging_service.warning(
                    "Skipping deferred function call for unknown function %s",
                    function_name,
                )
                continue
            if not isinstance(arguments, dict):
                self._logging_service.warning(
                    "Skipping deferred function call %s with invalid arguments",
                    function_name,
                )
                continue

            try:
                restored_calls.append(FunctionCall(function, **arguments))
            except TypeError:
                self._logging_service.warning(
                    "Failed to restore deferred function call %s with arguments %s",
                    function_name,
                    arguments,
                )

        return restored_calls

    def _serialize_deferred_tail(
        self,
        function_calls: Sequence[FunctionCall],
        start_index: int,
    ) -> list[DeferredFunctionCall]:
        """Serialize the remaining queue tail starting at ``start_index``."""
        return self._serialize_function_calls(function_calls[start_index:])

    def _is_parallel_safe_native_function_call(self, response: FunctionCall) -> bool:
        return response.function.__name__ in self._PARALLEL_SAFE_NATIVE_FUNCTION_NAMES

    def _is_mergeable_client_function_call(self, response: FunctionCall) -> bool:
        return response.function.__name__ in self._MERGEABLE_CLIENT_FUNCTION_NAMES

    def _get_client_call_batch(
        self,
        function_calls: Sequence[FunctionCall],
        start_index: int,
    ) -> tuple[list[FunctionCall], int]:
        """Collect consecutive mergeable client calls of the same function."""
        current_call = function_calls[start_index]
        client_calls = [current_call]
        next_index = start_index + 1

        if not self._is_mergeable_client_function_call(current_call):
            return client_calls, next_index

        while (
            next_index < len(function_calls)
            and self._client_function_call_service.is_client_function_call(
                function_calls[next_index]
            )
            and function_calls[next_index].function.__name__
            == current_call.function.__name__
        ):
            client_calls.append(function_calls[next_index])
            next_index += 1

        return client_calls, next_index

    def _get_parallel_native_group(
        self,
        function_calls: Sequence[FunctionCall],
        start_index: int,
    ) -> tuple[list[FunctionCall], int]:
        """Collect consecutive same-name native calls that are safe to parallelize."""
        current_call = function_calls[start_index]
        parallel_group = [current_call]
        next_index = start_index + 1

        while (
            next_index < len(function_calls)
            and not self._client_function_call_service.is_client_function_call(
                function_calls[next_index]
            )
            and self._is_parallel_safe_native_function_call(function_calls[next_index])
            and function_calls[next_index].function.__name__
            == current_call.function.__name__
        ):
            parallel_group.append(function_calls[next_index])
            next_index += 1

        return parallel_group, next_index

    def _merge_client_function_calls(
        self, function_calls: Sequence[FunctionCall]
    ) -> FunctionCall:
        if len(function_calls) == 1:
            return function_calls[0]

        first_call = function_calls[0]
        merged_arguments = dict(first_call.arguments)
        function_name = first_call.function.__name__
        merge_argument_name = self._MERGEABLE_CLIENT_FUNCTION_ARGUMENTS.get(
            function_name
        )
        if merge_argument_name is None:
            return first_call

        merged_arguments[merge_argument_name] = [
            item
            for function_call in function_calls
            for item in function_call.arguments.get(merge_argument_name, [])
        ]

        return FunctionCall(first_call.function, **merged_arguments)

    async def _collect_native_function_call_events_with_limit(
        self,
        function_call: FunctionCall,
        semaphore: asyncio.Semaphore,
        status_queue: asyncio.Queue[StatusUpdateSSE] | None = None,
    ) -> list[NativeFunctionCallEvent]:
        """Collect native events while respecting bounded parallelism.

        Status updates can be forwarded to a shared queue immediately so the UI
        stays responsive while the rest of the native events are buffered for
        ordered replay.
        """
        async with semaphore:
            native_events: list[NativeFunctionCallEvent] = []
            async for (
                native_fc_event
            ) in self._native_function_call_service.handle_function_calls(
                function_call
            ):
                if status_queue is not None and isinstance(
                    native_fc_event, StatusUpdateSSE
                ):
                    await status_queue.put(native_fc_event)
                    continue
                native_events.append(native_fc_event)
            return native_events

    async def _replay_native_function_call_events(
        self,
        native_events: Sequence[NativeFunctionCallEvent],
        chat_messages: list[Message[Any] | AssistantMessage],
        original_chat_messages_count: int,
        deferred_citations: list[Citation],
        deferred_tail_specs: Sequence[DeferredFunctionCall] | None = None,
    ) -> tuple[list[AdaSSE], bool]:
        """Replay native events and stop only when control must return to the client."""
        outbound_events: list[AdaSSE] = []

        for native_fc_event in native_events:
            if isinstance(native_fc_event, StatusUpdateSSE):
                self._logging_service.log_status_update_sse(native_fc_event)
                if native_fc_event.data.artifacts:
                    self._logging_service.info(
                        "StatusUpdateSSE contains %d artifacts - yielding to frontend",
                        len(native_fc_event.data.artifacts),
                    )
                outbound_events.append(native_fc_event)
            elif isinstance(native_fc_event, list):
                # List implies LLM messages
                chat_messages.extend(native_fc_event)
                self._logging_service.info(
                    "Appended function call response. Continuing..."
                )
            elif isinstance(
                native_fc_event,
                (MessageChunkSSE, AppArtifactSSE, CitationCollectionSSE),
            ):
                if isinstance(native_fc_event, CitationCollectionSSE):
                    # Defer citations to end-of-stream so they appear after the
                    # message/chart content.
                    deferred_citations.extend(native_fc_event.data.citations)
                else:
                    outbound_events.append(native_fc_event)
                # Note: We no longer exit the loop after web search completes
                # to allow for follow-up actions with the retrieved information.
            elif isinstance(native_fc_event, SqlQueryFunctionCallResult):
                # SQL query generated for execution - send to frontend so the
                # widget data proxy runs the query and returns rows to Ada.
                func_call_kwargs = native_fc_event.get_function_call_kwargs()
                merged_extra_state = {**func_call_kwargs["extra_state"]}
                if deferred_tail_specs:
                    merged_extra_state[self._DEFERRED_FUNCTION_CALLS_KEY] = list(
                        spec.model_dump(mode="json") for spec in deferred_tail_specs
                    )

                for artifact in native_fc_event.artifacts:
                    self._context_service.load_context(
                        elements=[artifact.to_parsed_context()]
                    )

                outbound_events.append(
                    self._augment_client_function_call_event(
                        client_fc_event=FunctionCallSSE(
                            data=FunctionCallSSEData(
                                function="get_widget_data",
                                input_arguments={
                                    "data_sources": [
                                        DataSourceRequestPayload(
                                            widget_uuid=native_fc_event.widget_uuid,
                                            origin=native_fc_event.widget_origin,
                                            id=native_fc_event.widget_id,
                                            input_args=func_call_kwargs["input_args"],
                                        ).model_dump()
                                    ]
                                },
                                extra_state=merged_extra_state,
                            )
                        ),
                        chat_messages=chat_messages,
                        original_chat_messages_count=original_chat_messages_count,
                    )
                )
                return outbound_events, True
            elif isinstance(native_fc_event, PythonCodeFunctionCallResult):
                # Python code generated - send to frontend
                func_call_kwargs = native_fc_event.get_function_call_kwargs()
                merged_extra_state = {**func_call_kwargs["extra_state"]}
                if deferred_tail_specs:
                    merged_extra_state[self._DEFERRED_FUNCTION_CALLS_KEY] = list(
                        spec.model_dump(mode="json") for spec in deferred_tail_specs
                    )

                fc_function = "get_widget_data"

                # Build copilot_function_call_arguments so
                # get_function_call_spec can reconstruct valid args when
                # this comes back from the frontend.
                copilot_fc_args: dict[str, Any] = {
                    "summary": merged_extra_state.get(
                        "summary", "Executing Python code"
                    ),
                    "widget_queries": [
                        {
                            "widget_uuid": native_fc_event.widget_uuid,
                            "query": merged_extra_state.get("sql_query", ""),
                        }
                    ],
                }
                merged_extra_state["copilot_function_call_arguments"] = copilot_fc_args

                outbound_events.append(
                    self._augment_client_function_call_event(
                        client_fc_event=FunctionCallSSE(
                            data=FunctionCallSSEData(
                                function=fc_function,
                                input_arguments={
                                    "data_sources": [
                                        DataSourceRequestPayload(
                                            widget_uuid=native_fc_event.widget_uuid,
                                            origin=native_fc_event.widget_origin,
                                            id=native_fc_event.widget_id,
                                            input_args=func_call_kwargs["input_args"],
                                        ).model_dump()
                                    ]
                                },
                                extra_state=merged_extra_state,
                            )
                        ),
                        chat_messages=chat_messages,
                        original_chat_messages_count=original_chat_messages_count,
                    )
                )
                return outbound_events, True
            elif isinstance(native_fc_event, FunctionCallResponse):
                # Client function call (e.g., get_skill_content)
                outbound_events.append(
                    self._augment_client_function_call_event(
                        client_fc_event=FunctionCallSSE(
                            data=FunctionCallSSEData(**native_fc_event.model_dump())
                        ),
                        chat_messages=chat_messages,
                        original_chat_messages_count=original_chat_messages_count,
                        deferred_function_call_specs=deferred_tail_specs,
                    )
                )
                return outbound_events, True
            else:
                raise TypeError(f"Unexpected event type: {type(native_fc_event)}")

        return outbound_events, False

    async def _emit_client_function_call_events(
        self,
        function_call: FunctionCall,
        chat_messages: list[Message[Any] | AssistantMessage],
        original_chat_messages_count: int,
        deferred_tail_specs: Sequence[DeferredFunctionCall] | None = None,
    ) -> tuple[list[AdaSSE], bool]:
        """Emit the next client boundary event and preserve any deferred tail."""
        outbound_events: list[AdaSSE] = []

        if bridged_client_event := self._build_sql_artifact_dashboard_event(
            function_call
        ):
            summary = bridged_client_event.data.extra_state.get(
                "copilot_function_call_arguments", {}
            ).get("summary", "Adding widget to dashboard")
            status_event = StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message=str(summary),
                )
            )
            self._logging_service.log_status_update_sse(status_event)
            outbound_events.append(status_event)
            outbound_events.append(
                self._augment_client_function_call_event(
                    client_fc_event=bridged_client_event,
                    chat_messages=chat_messages,
                    original_chat_messages_count=original_chat_messages_count,
                    deferred_function_call_specs=deferred_tail_specs,
                )
            )
            return outbound_events, True

        # Resolve artifact references for widget generation
        self._resolve_artifact_reference(function_call)

        async for (
            client_fc_event
        ) in self._client_function_call_service.handle_function_calls(function_call):
            if isinstance(client_fc_event, StatusUpdateSSE):
                self._logging_service.log_status_update_sse(client_fc_event)
                outbound_events.append(client_fc_event)
            elif isinstance(client_fc_event, FunctionCallSSE):
                outbound_events.append(
                    self._augment_client_function_call_event(
                        client_fc_event=client_fc_event,
                        chat_messages=chat_messages,
                        original_chat_messages_count=original_chat_messages_count,
                        deferred_function_call_specs=deferred_tail_specs,
                    )
                )
                return outbound_events, True
            elif isinstance(client_fc_event, list):  # Error case
                chat_messages.extend(client_fc_event)

        return outbound_events, False

    def _augment_client_function_call_event(
        self,
        client_fc_event: FunctionCallSSE,
        chat_messages: list[Message[Any] | AssistantMessage],
        original_chat_messages_count: int,
        deferred_function_call_specs: Sequence[DeferredFunctionCall] | None = None,
    ) -> FunctionCallSSE:
        """Attach state that must survive the frontend request boundary."""
        intermediate_context = self._get_intermediate_results_text(
            chat_messages, original_chat_messages_count
        )
        if intermediate_context:
            # Store intermediate context (enhanced prompt, plan)
            self._logging_service.info("Storing intermediate context in extra_state")
            if client_fc_event.data.extra_state is None:
                client_fc_event.data.extra_state = {}
            client_fc_event.data.extra_state["intermediate_context"] = (
                intermediate_context
            )

        existing_citations = self._citation_service.citations
        is_add_generative_widget = (
            client_fc_event.data.function == "add_generative_widget"
            and isinstance(client_fc_event.data.input_arguments, dict)
        )
        input_arguments = (
            client_fc_event.data.input_arguments if is_add_generative_widget else None
        )
        citations_to_store = (
            self._get_generated_widget_citations(
                input_arguments=input_arguments,
                fallback_citations=existing_citations,
            )
            if input_arguments is not None
            else existing_citations
        )

        if citations_to_store:
            # Serialize citations so they survive across HTTP requests
            self._logging_service.info(
                "Storing %d intermediate citations in extra_state",
                len(citations_to_store),
            )
            if client_fc_event.data.extra_state is None:
                client_fc_event.data.extra_state = {}
            client_fc_event.data.extra_state["intermediate_citations"] = [
                citation.model_dump(mode="json") for citation in citations_to_store
            ]

        # Serialize artifact data so it survives across HTTP requests
        artifacts_to_store = self._context_service.dump_roundtrip_artifacts()
        if artifacts_to_store:
            self._logging_service.info(
                "Storing %d intermediate artifacts in extra_state",
                len(artifacts_to_store),
            )
            if client_fc_event.data.extra_state is None:
                client_fc_event.data.extra_state = {}
            client_fc_event.data.extra_state["intermediate_artifacts"] = (
                artifacts_to_store
            )

        if input_arguments is not None:
            if citations_to_store and not input_arguments.get("citations"):
                input_arguments["citations"] = [
                    citation.model_dump(mode="json") for citation in citations_to_store
                ]
            if artifacts_to_store and not input_arguments.get("artifacts"):
                input_arguments["artifacts"] = list(artifacts_to_store.values())

        if deferred_function_call_specs:
            if client_fc_event.data.extra_state is None:
                client_fc_event.data.extra_state = {}
            client_fc_event.data.extra_state[self._DEFERRED_FUNCTION_CALLS_KEY] = [
                spec.model_dump(mode="json") for spec in deferred_function_call_specs
            ]

        function_name = getattr(
            client_fc_event.data,
            "function",
            getattr(client_fc_event.data, "function_name", "unknown"),
        )
        input_arguments = client_fc_event.data.input_arguments
        data_sources = input_arguments.get("data_sources") if input_arguments else None
        row_count = len(data_sources) if isinstance(data_sources, list) else None
        # Log metadata only — avoid dumping full data payloads
        self._logging_service.info(
            "Yielding client function call event: function=%s, data_rows=%s",
            function_name,
            row_count,
        )
        return client_fc_event

    @staticmethod
    def _extract_inline_citation_ids(text: Any) -> list[str]:
        """Return citation IDs embedded in OpenAI inline citation markers."""
        if not isinstance(text, str):
            return []

        matches = re.findall(
            r"<\|start_citation_id\|>(.*?)<\|end_citation_id\|>",
            text,
            flags=re.DOTALL,
        )
        return [match.strip() for match in matches if match.strip()]

    def _get_generated_widget_citations(
        self,
        input_arguments: dict[str, Any],
        fallback_citations: Sequence[Citation],
    ) -> list[Citation]:
        """Prefer exact inline citation IDs over deduped same-source citations."""
        citations: list[Citation] = []
        seen_ids: set[str] = set()
        seen_signatures: set[str] = set()

        for citation_id in self._extract_inline_citation_ids(
            input_arguments.get("data")
        ):
            citation = self._citation_service.get_citation(citation_id)
            if not citation:
                continue

            serialized_id = str(citation.id)
            if serialized_id in seen_ids:
                continue

            citations.append(citation)
            seen_ids.add(serialized_id)
            seen_signatures.add(CitationService.signature(citation))

        for citation in fallback_citations:
            serialized_id = str(citation.id)
            signature = CitationService.signature(citation)
            if serialized_id in seen_ids or signature in seen_signatures:
                continue

            citations.append(citation)
            seen_ids.add(serialized_id)
            seen_signatures.add(signature)

        return citations

    def _prepare_function_response_message(
        self, response: FunctionCall
    ) -> AssistantMessage:
        function_call_text = f"Function call: {response.function.__name__}\n"
        function_call_text += f"Arguments: {response.arguments}"
        return AssistantMessage(content=sanitize_str(function_call_text))

    async def _stream_copilot_events(
        self, stream: AsyncStreamedStr
    ) -> AsyncGenerator[
        StatusUpdateSSE | MessageChunkSSE | MessageArtifactSSE | CitationCollectionSSE,
        None,
    ]:
        async for stream_event in self._handle_copilot_stream(stream):
            if isinstance(stream_event, tuple):
                yield stream_event[1]
            else:
                yield stream_event

    def _build_end_citations(
        self, deferred_citations: Sequence[Citation]
    ) -> list[Citation]:
        end_citations: list[Citation] = []
        seen_signatures: set[str] = set()

        for citation in deferred_citations:
            sig = CitationService.signature(citation)
            if sig in seen_signatures:
                continue
            end_citations.append(citation)
            seen_signatures.add(sig)

        for citation in self._citation_service.citations:
            sig = CitationService.signature(citation)
            if sig in seen_signatures:
                continue
            end_citations.append(citation)
            seen_signatures.add(sig)

        return end_citations

    async def _finalize_copilot_stream(
        self, deferred_citations: Sequence[Citation]
    ) -> AsyncGenerator[CitationCollectionSSE | PromptSuggestionsSSE, None]:
        end_citations = self._build_end_citations(deferred_citations)
        if end_citations:
            self._logging_service.info(
                "Streaming %d citations at end of stream",
                len(end_citations),
            )
            yield CitationCollectionSSE(
                data=CitationCollection(citations=end_citations)
            )
        self._citation_service.clear()
        if (
            "prompt-suggestions" in self._workspace_options
            and self._pending_prompt_suggestions
        ):
            yield PromptSuggestionsSSE(
                data=PromptSuggestionsSSEData(
                    suggestions=self._pending_prompt_suggestions[:3]
                )
            )
        self._pending_prompt_suggestions = []
        self._logging_service.info(
            "Stream completed successfully. Closing SSE connection."
        )

    async def _handle_copilot_stream(
        self, stream: AsyncStreamedStr
    ) -> AsyncGenerator[
        (
            MessageChunkSSE
            | MessageArtifactSSE
            | CitationCollectionSSE
            | tuple[str, MessageArtifactSSE]
            | tuple[str, CitationCollectionSSE]
        ),
        None,
    ]:
        MESSAGE_CHUNK_DELAY = 0.002
        START_ARTIFACT_TAG = "<|start_artifact_id|>"
        END_ARTIFACT_TAG = "<|end_artifact_id|>"
        START_CITATION_TAG = "<|start_citation_id|>"
        END_CITATION_TAG = "<|end_citation_id|>"
        START_SUGGESTIONS_TAG = "<suggestions"

        def extract_tag_value(buffer: str, start_tag: str, end_tag: str) -> str | None:
            _, start_separator, after_start = buffer.partition(start_tag)
            if not start_separator:
                return None
            tag_value, end_separator, _ = after_start.partition(end_tag)
            if not end_separator:
                return None
            return tag_value

        buffer = ""
        suggestion_buffer = ""
        in_suggestions_block = False
        async for chunk in stream:
            for letter in chunk:
                if in_suggestions_block:
                    suggestion_buffer += letter
                    if self._SUGGESTIONS_CLOSE_PATTERN.search(suggestion_buffer):
                        _, suggestions = self._parse_prompt_suggestions(
                            suggestion_buffer
                        )
                        self._pending_prompt_suggestions.extend(suggestions)
                        suggestion_buffer = ""
                        in_suggestions_block = False
                    continue

                # Possible start of an artifact tag
                if letter == "<" and not buffer:
                    buffer = letter
                    continue

                if buffer:  # Implies we're accumulating tokens
                    buffer += letter
                    buffer_lower = buffer.lower()
                    if buffer_lower.startswith(START_SUGGESTIONS_TAG):
                        if ">" in buffer:
                            suggestion_buffer = buffer
                            buffer = ""
                            in_suggestions_block = True
                            if self._SUGGESTIONS_CLOSE_PATTERN.search(
                                suggestion_buffer
                            ):
                                _, suggestions = self._parse_prompt_suggestions(
                                    suggestion_buffer
                                )
                                self._pending_prompt_suggestions.extend(suggestions)
                                suggestion_buffer = ""
                                in_suggestions_block = False
                        continue

                    if END_ARTIFACT_TAG in buffer:
                        self._logging_service.info(
                            "In-line artifact tag detected: %s", buffer
                        )
                        artifact_id = extract_tag_value(
                            buffer,
                            START_ARTIFACT_TAG,
                            END_ARTIFACT_TAG,
                        )
                        if artifact_id is None:
                            self._logging_service.warning(
                                "Incomplete artifact tag in buffer: %s", buffer
                            )
                            buffer = ""
                            continue
                        if (
                            artifact
                            := self._context_service.get_context_by_source_info_name(  # noqa: E501
                                artifact_id
                            )
                        ):
                            self._logging_service.info(
                                "Artifact found, yielding: %s", buffer
                            )
                            if self._user_id and posthog_client.feature_enabled(
                                "COPILOT-use-direct-inline-unstructured-artifacts",
                                self._user_id,
                            ):
                                self._logging_service.info(
                                    "Using direct inline unstructured artifacts"
                                )
                                client_artifact = artifact.to_client_artifact()
                                if isinstance(artifact, UnstructuredContext):
                                    if client_artifact.type == "text":
                                        for letter in artifact.content:
                                            await asyncio.sleep(MESSAGE_CHUNK_DELAY)
                                            yield MessageChunkSSE(
                                                data=MessageChunkSSEData(delta=letter)
                                            )
                                    elif client_artifact.type in {
                                        "table",
                                        "snowflake_query",
                                        "snowflake_python",
                                        "html",
                                    }:
                                        yield (
                                            buffer,
                                            MessageArtifactSSE(data=client_artifact),
                                        )
                                    else:
                                        self._logging_service.warning(
                                            "Artifact type not supported: %s",
                                            client_artifact.type,
                                        )
                                elif isinstance(artifact, StructuredContext):
                                    yield (
                                        buffer,
                                        MessageArtifactSSE(data=client_artifact),
                                    )
                            else:
                                self._logging_service.info(
                                    "Using unstructured artifact as separate object"
                                )
                                try:
                                    client_artifact = artifact.to_client_artifact()
                                    self._logging_service.info(
                                        "Client artifact created: type=%s, name=%s",
                                        client_artifact.type,
                                        client_artifact.name,
                                    )
                                    yield (
                                        buffer,
                                        MessageArtifactSSE(data=client_artifact),
                                    )
                                except Exception as e:
                                    self._logging_service.error(
                                        "Failed to create/yield MessageArtifactSSE: %s",
                                        e,
                                    )
                        else:
                            self._logging_service.warning(
                                "Artifact tag not found: %s", buffer
                            )
                        # Reset buffer
                        buffer = ""
                    elif END_CITATION_TAG in buffer:
                        self._logging_service.info(
                            "In-line citations tag detected: %s", buffer
                        )
                        citation_id = extract_tag_value(
                            buffer,
                            START_CITATION_TAG,
                            END_CITATION_TAG,
                        )
                        if citation_id is None:
                            self._logging_service.warning(
                                "Incomplete citation tag in buffer: %s",
                                buffer,
                            )
                            buffer = ""
                            continue

                        # Consume the tag silently — don't send raw text to frontend.
                        # Citations are yielded at end-of-stream via deferred_citations.
                        if self._citation_service.get_citation(citation_id):
                            self._citation_service.mark_as_cited(citation_id)
                        buffer = ""
                    elif (
                        len(buffer) > len(START_ARTIFACT_TAG)
                        and not buffer.startswith(START_ARTIFACT_TAG)
                        and not buffer.startswith(START_CITATION_TAG)
                    ):
                        # Invalid tag, yield the buffered content
                        for buffered_letter in buffer:
                            await asyncio.sleep(MESSAGE_CHUNK_DELAY)
                            yield MessageChunkSSE(
                                data=MessageChunkSSEData(delta=buffered_letter)
                            )
                        buffer = ""
                else:
                    # Regular message chunk
                    await asyncio.sleep(MESSAGE_CHUNK_DELAY)
                    yield MessageChunkSSE(data=MessageChunkSSEData(delta=letter))

    def _get_model(self):
        return get_llm(
            model=constants.OPENBB_AGENT_MODEL_MAIN,
            temperature=self._temperature,
            api_key=self._openai_api_key,
        )
