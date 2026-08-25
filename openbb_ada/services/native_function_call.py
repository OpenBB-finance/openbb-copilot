import json
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator

import pandas as pd
from magentic import (
    AssistantMessage,
    FunctionCall,
    FunctionResultMessage,
)
from magentic.chat_model.message import Message
from openbb_ai.models import (
    CitationCollection,
    CitationCollectionSSE,
    ClientArtifact,
    LlmClientMessage,
    MessageChunkSSE,
    RawObjectDataFormat,
    RoleEnum,
    SourceInfo,
    StatusUpdateSSE,
    StatusUpdateSSEData,
    Widget,
    WidgetCollection,
)

from ..errors import CodeGenerationError, FunctionCallError
from ..models import (
    AppArtifact,
    AppArtifactDef,
    AppArtifactLayoutItem,
    AppArtifactSSE,
    AppArtifactTab,
    AppArtifactTabInput,
    AppArtifactWidgetRef,
    AvailableSemanticView,
    CodeGenerationRequest,
    ContextStructuredQueryResult,
    ContextUnstructuredQueryResult,
    CopilotArtifact,
    DocumentQueryResult,
    PythonCodeFunctionCallResult,
    PythonCodeGenerationResult,
    PythonWidgetContext,
    SemanticModelReference,
    SqlQueryFunctionCallResult,
    SqlQueryGenerationResult,
    SqlWidgetContext,
)
from ..utils.chart_generation import generate_chart_parameters
from ..utils.data_extraction import extract_table_data
from ..utils.utils import sanitize_str
from ._logging import LoggingService
from .citation import CitationService
from .context import ContextService
from .document import DocumentService
from .prompt_enhancement import PromptEnhancementService
from .python_code_generation import PythonCodeGenerationService
from .sql_query_generation import SqlQueryGenerationService
from .template import TemplateService
from .web_search_llm import WebSearchLlmService

if TYPE_CHECKING:
    from .snowflake_cortex_analyst import SnowflakeCortexAnalystService


_GENERATED_ARTIFACT_WIDGET_IDS = {"rich_note", "copilot_table", "html"}
_GENERATED_ARTIFACT_WIDGET_PREFIXES = ("rich_note-", "copilot_table-", "html-")


class NativeFunctionCallService:
    def __init__(
        self,
        document_service: DocumentService,
        logging_service: LoggingService,
        web_search_llm_service: WebSearchLlmService,
        context_service: ContextService,
        template_service: TemplateService,
        citation_service: CitationService,
        prompt_enhancement_service: PromptEnhancementService | None = None,
        sql_query_generation_service: SqlQueryGenerationService | None = None,
        python_code_generation_service: PythonCodeGenerationService | None = None,
        snowflake_cortex_analyst_service: "SnowflakeCortexAnalystService | None" = None,
        semantic_views: list[str] | None = None,
        available_semantic_views: list[AvailableSemanticView] | None = None,
    ):
        self._document_service = document_service
        self._logging_service = logging_service
        self._web_search_llm_service = web_search_llm_service
        self._context_service = context_service
        self._template_service = template_service
        self._citation_service = citation_service
        self._prompt_enhancement_service = prompt_enhancement_service
        self._sql_query_generation_service = sql_query_generation_service
        self._python_code_generation_service = python_code_generation_service
        self._snowflake_cortex_analyst_service = snowflake_cortex_analyst_service
        self._semantic_views = semantic_views
        self._available_semantic_views = available_semantic_views
        self._reported_semantic_context_statuses: set[tuple[str, ...]] = set()
        self._app_widget_collection: WidgetCollection | None = None

    def set_app_widget_collection(
        self, widget_collection: WidgetCollection | None
    ) -> None:
        self._app_widget_collection = widget_collection

    @property
    def is_snowflake_mode(self) -> bool:
        return self._snowflake_cortex_analyst_service is not None

    @staticmethod
    def _find_widget(
        widget_uuid: str,
        widgets: list[SqlWidgetContext] | list[PythonWidgetContext] | None,
    ) -> SqlWidgetContext | PythonWidgetContext | None:
        return next(
            (w for w in (widgets or []) if w.widget_uuid == widget_uuid),
            None,
        )

    async def _llm_think(
        self, plan: str | list[str], summary: str = "Planning"
    ) -> AsyncGenerator[str | StatusUpdateSSE, None]:
        """Use this tool only when an explicit planning step is necessary.

        Do not call this tool when the user asks you to avoid planning, when the
        next useful action is obvious, or when a relevant widget/data tool can be
        queried directly. Do not call this tool more than once for the same user
        request; after planning, call a non-planning tool or answer directly.
        Never call this tool just to say that you are skipping planning.

        Call this tool when you need to:
        - Break down a complex query
        - Choose between multiple plausible data sources or tools
        - Organize a multi-step calculation before taking action
        - Organize your thoughts before taking action

        Parameters
        ----------
        plan : str | list[str]
            Your thoughts, reasoning, or plan for how to approach the user's query.
            Can be a single string or a list of strings for each step.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """
        full_plan = "\n".join(plan) if isinstance(plan, list) else plan
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
                details=[full_plan],
            )
        )
        yield full_plan

    async def llm_enhance_prompt(
        self,
        reasoning: str,
        summary: str = "Prompt enhancement",
    ) -> AsyncGenerator[str | StatusUpdateSSE, None]:
        """Enhance the user's prompt to make it clearer and more actionable.

        Use this tool BEFORE _llm_think when:
        - The user's query is vague or ambiguous
        - The request lacks specific details (e.g., "analyze the data"
          without specifying what analysis)
        - Multiple interpretations are possible
        - Financial terms need clarification (e.g., company names without tickers)
        - Time periods or dates are unclear
        - The user asks for "everything" or uses very broad language

        Do NOT use this tool when:
        - The query is already clear and specific
        - You understand exactly what the user wants
        - The request is simple and straightforward

        Parameters
        ----------
        reasoning : str
            Your analysis of why the query needs enhancement and what aspects
            need clarification.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """
        # Only proceed if prompt enhancement service is available
        if not self._prompt_enhancement_service:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message="Prompt enhancement unavailable",
                    details=["Proceeding with original query"],
                )
            )
            yield reasoning
            return

        try:
            (
                current_messages,
                current_context,
                current_widgets,
                current_tools,
            ) = self._prompt_enhancement_service.get_current_context()

            # Use the current context that was set
            enhanced_query = await self._prompt_enhancement_service.enhance_prompt(
                messages=current_messages or [],
                context=current_context,
                widgets=current_widgets,
                tools=current_tools,
            )

            if enhanced_query:
                yield StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType="INFO",
                        message=summary,
                        details=[enhanced_query],
                    )
                )
                result = (
                    f"Enhanced query: {enhanced_query}\n\n"
                    f"I will now proceed with this clarified interpretation."
                )
                yield result
            else:
                yield StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType="INFO",
                        message=summary,
                        details=["Proceeding with original query"],
                    )
                )
                yield f"Analysis: {reasoning}\n\nProceeding with the original query."

        except Exception as e:
            self._logging_service.warning(f"Prompt enhancement failed: {e}")
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="WARNING",
                    message="Enhancement failed",
                    details=[f"Error: {str(e)}"],
                )
            )
            yield reasoning

    async def llm_complete(
        self,
    ) -> AsyncGenerator[MessageChunkSSE, None]:
        """Signal that the conversation turn is finished.

        IMPORTANT RULES — violating any of these is a critical error:
        1. You MUST provide a text response to the user BEFORE calling this tool.
           That means your previous message in the conversation must be a user-facing
           text reply (not a tool call). If your last action was a tool call or a
           function result, you MUST first reply with a text message, then call this.
        2. NEVER call this tool as the immediate next step after any other tool call.
           Always produce a visible assistant message in between.
        """
        # This method signals the end of the conversation
        # It doesn't need to yield anything since it's just a signal to complete
        return
        yield  # This line is never reached but needed for the type checker

    async def llm_query_unstructured_data(
        self, unstructured_data_ids: list[str], summary: str = "Querying data"
    ) -> AsyncGenerator[
        ContextUnstructuredQueryResult | StatusUpdateSSE | CitationCollectionSSE, None
    ]:
        """Read loaded unstructured data by specifying the ids.

        Only for reading unstructured context.

        NB: This function can only be used to read unstructured data that has been
        loaded into context, not loaded files. To access original files, instead use
        the `_llm_query_uploaded_files` tool.

        Parameters
        ----------
        unstructured_data_ids : list[str]
            List of unstructured data ids to read from the context service.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """

        try:
            context_unstructured_query_results = (
                self._context_service.read_unstructured_context_by_ids(
                    unstructured_data_ids
                )
            )
            for context_unstructured_query_result in context_unstructured_query_results:
                # Add citations from context queries, but only if citable
                widget_citations = []
                for citation in context_unstructured_query_result.citations:
                    if citation.source_info.citable:  # Only add citable sources
                        self._citation_service.add_citation(citation)
                        # Auto-mark and collect widget citations when their data is used
                        if citation.source_info.type == "widget":
                            self._citation_service.mark_as_cited(str(citation.id))
                            widget_citations.append(citation)
                # Yield widget citations immediately so frontend receives them
                if widget_citations:
                    yield CitationCollectionSSE(
                        data=CitationCollection(citations=widget_citations)
                    )

                citations = context_unstructured_query_result.citations
                if citations:
                    source_type = citations[0].source_info.type
                    citation_details = citations[0].details
                    details: list[dict[str, Any] | str] = [
                        {
                            "Source type": source_type,
                            **(citation_details[0] if citation_details else {}),
                        }
                    ]
                else:
                    details = []

                yield StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType="INFO",
                        message=summary,
                        details=details,
                    )
                )
                yield context_unstructured_query_result

        except ValueError as err:
            self._logging_service.error(str(err))
            raise FunctionCallError(str(err)) from err

    async def llm_query_structured_data(
        self, query: str, summary: str = "Querying data"
    ) -> AsyncGenerator[
        ContextStructuredQueryResult | StatusUpdateSSE | CitationCollectionSSE, None
    ]:
        """Use natural language to fetch loaded structured data as a table or chart.

        You can combine queries to fetch multiple pieces of structured data.

        ALWAYS specify whether to return the result as a table or a kind of plot
        (eg line, bar, scatter) in your query, for example, "A table of ..." or
        "A line plot of ...", depending on what the user asks for.

        Each piece of structured data is already filtered using it's metadata.
        ONLY SPECIFY TICKER SYMBOLS OR SPECIFIC DATES IF THE USER QUERIES FOR IT EXPLICITLY.

        Parameters
        ----------
        query : str
            Your query.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with "Querying for" and then you complete based on the specific 'query' you are making (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """  # noqa: E501
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
                details=[
                    {
                        "Query": query,
                    }
                ],
            )
        )

        async for (
            event
        ) in self._context_service.query_structured_context_with_natural_language(
            query=query
        ):
            if isinstance(event, ContextStructuredQueryResult):
                widget_citations = []
                for citation in event.citations:
                    if citation.source_info.citable:  # Only add citable sources
                        self._citation_service.add_citation(citation)
                        # Auto-mark and collect widget citations when their data is used
                        if citation.source_info.type == "widget":
                            self._citation_service.mark_as_cited(str(citation.id))
                            widget_citations.append(citation)
                # Yield widget citations immediately so frontend receives them
                if widget_citations:
                    yield CitationCollectionSSE(
                        data=CitationCollection(citations=widget_citations)
                    )
            yield event

    async def llm_query_uploaded_files(
        self, query: str, summary: str = "Querying files"
    ) -> AsyncGenerator[
        list[DocumentQueryResult] | StatusUpdateSSE | CitationCollectionSSE, None
    ]:
        """Use a detailed NL query to access the original loaded user files.

        Your query can search for and summarize multiple files at once, just
        mention the files you want to use in your query.

        Specify precisely what the output must be.
        Capture the user's full intention is, in great detail.

        The query is usually best phrased as a question.

        Mention the name or names of the file(s) you want to query.

        You must specify additional information with your query
        if it has been provided by the user.

        Parameters
        ----------
        query : str
            Your query to search the user files with.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
                details=[
                    {
                        "Query": query,
                    }
                ],
            )
        )
        async for event in self._document_service.query_all_user_files(query):
            # Add citations from document queries (these are legitimate sources)
            if isinstance(event, list) and all(
                isinstance(r, DocumentQueryResult) for r in event
            ):
                all_citations = []
                for document_query_result in event:
                    for citation in document_query_result.citations:
                        if citation.source_info.citable:
                            self._citation_service.add_citation(citation)
                            # Auto-mark as cited when data is actually used
                            self._citation_service.mark_as_cited(str(citation.id))
                            all_citations.append(citation)
                # Yield citations immediately so frontend receives them
                if all_citations:
                    yield CitationCollectionSSE(
                        data=CitationCollection(citations=all_citations)
                    )
            yield event

    async def llm_get_widget_input_state(
        self,
        widget_uuid: str | None = None,
        summary: str = "Reading widget input state",
        widget_collection: WidgetCollection | None = None,
    ) -> AsyncGenerator[str | StatusUpdateSSE, None]:
        """Read the current widget input arguments already present in context.

        Use this tool when the user asks what prompt/query/input arguments are
        currently selected in a widget. Prefer this over `llm_query_widgets`
        when the user is asking about the input state itself (not asking to
        fetch data).

        Parameters
        ----------
        widget_uuid : str | None
            Optional widget UUID. Provide it when multiple widgets are present
            and the user refers to a specific widget.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
                details=[],
            )
        )

        if not widget_collection:
            yield json.dumps({"error": "no_widgets_in_context"})
            return

        widgets_by_priority: list[tuple[str, Widget]] = []
        widgets_by_priority.extend(
            [("primary", widget) for widget in (widget_collection.primary or [])]
        )
        widgets_by_priority.extend(
            [("secondary", widget) for widget in (widget_collection.secondary or [])]
        )

        if widget_uuid:
            widgets_by_priority = [
                (priority, widget)
                for priority, widget in widgets_by_priority
                if str(getattr(widget, "uuid", "") or "") == widget_uuid
            ]
            if not widgets_by_priority:
                yield json.dumps(
                    {
                        "error": "widget_not_found",
                        "widget_uuid": widget_uuid,
                    },
                )
                return

        if len(widgets_by_priority) != 1:
            yield json.dumps(
                {
                    "error": "ambiguous_widget",
                    "message": "Specify widget_uuid.",
                    "available_widgets": [
                        {
                            "widget_uuid": str(getattr(widget, "uuid", "") or ""),
                            "name": getattr(widget, "name", "widget"),
                            "priority": priority,
                        }
                        for priority, widget in widgets_by_priority
                    ],
                },
            )
            return

        priority, widget = widgets_by_priority[0]
        params = getattr(widget, "params", None) or []

        input_arguments = {
            str(getattr(param, "name", "unknown")): getattr(
                param, "current_value", None
            )
            for param in params
        }
        executed_params: dict[str, Any] = {}
        unapplied_param_names: list[str] = []
        for param in params:
            name = str(getattr(param, "name", "unknown"))
            executed_value = getattr(param, "executed_value", None)
            if executed_value is None:
                continue
            executed_params[name] = executed_value
            if executed_value != input_arguments.get(name):
                unapplied_param_names.append(name)

        payload: dict[str, Any] = {
            "widget": {
                "widget_uuid": str(getattr(widget, "uuid", "") or ""),
                "name": getattr(widget, "name", "widget"),
                "origin": getattr(widget, "origin", "unknown"),
                "priority": priority,
            },
            "input_arguments": input_arguments,
        }

        if executed_params:
            payload["param_sync"] = {
                "executed_params": executed_params,
                "unapplied_param_names": sorted(set(unapplied_param_names)),
                "has_unapplied_changes": bool(unapplied_param_names),
            }

        yield json.dumps(payload, default=str)

    async def llm_create_app(
        self,
        name: str,
        description: str,
        tabs: list[AppArtifactTabInput],
        summary: str = "Creating app",
    ) -> AsyncGenerator[str | StatusUpdateSSE | AppArtifactSSE, None]:
        """Assemble connected widgets into a multi-tab dashboard app artifact.

        Use this tool when the user asks you to create, build, or assemble a
        new dashboard/app made from existing connected widgets. The returned
        app artifact lets the user open it as a live dashboard.

        Call `llm_search_widgets` first with keywords from the user's request,
        then use only exact `(origin, widget_id)` pairs from those results.
        Do not use generated artifact widgets such as notes, tables, or HTML
        artifacts as app source widgets. Prefer data widgets. Use a 40-column
        grid and place only one widget per row by setting `x=0`, `w=40`, and
        increasing `y`. For widget inputs, put prompt tickers, sectors,
        countries, dates, and other filters under `state.params`, but only for
        parameter names that exist on that widget. Parameterized widgets must
        set explicit `state.params`; for example `{"params": {"symbol": "AAPL"}}`.
        Do not hand-write SQL in `state.params.query`; use an existing valid
        query from the widget/search result or an exact SQL query produced by
        the SQL generation tool.

        Parameters
        ----------
        name : str
            App display name. Use a descriptive dashboard/app name from the
            user's request, not a search term or individual widget name.
        description : str
            Short description of what the dashboard app shows.
        tabs : list[AppArtifactTabInput]
            Dashboard tabs. Tab ids must be lowercase kebab-case and unique.
            Each tab must contain at least one layout item.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period.
        """
        widget_collection = self._app_widget_collection
        catalog_widgets = (
            [
                *(widget_collection.primary or []),
                *(widget_collection.secondary or []),
                *(widget_collection.extra or []),
            ]
            if widget_collection
            else []
        )
        if not catalog_widgets:
            yield (
                "Error: cannot create app because no widgets are available in "
                "the current context."
            )
            return
        if not tabs or not any(tab.layout for tab in tabs):
            yield (
                "Error: cannot create app because no widgets were selected. "
                "Search for relevant widgets first, then retry with exact "
                "(origin, widget_id) pairs from the search results."
            )
            return

        errors: list[str] = []
        resolved_widgets: dict[tuple[str, str], Widget] = {}
        catalog_by_key = {
            (widget.origin, widget.widget_id): widget for widget in catalog_widgets
        }

        for tab in tabs:
            if not tab.layout:
                errors.append(f'tab "{tab.name}" has no widgets')
                continue
            for item in tab.layout:
                is_generated_artifact = (
                    item.widget_id in _GENERATED_ARTIFACT_WIDGET_IDS
                    or item.widget_id.startswith(_GENERATED_ARTIFACT_WIDGET_PREFIXES)
                )
                if is_generated_artifact:
                    errors.append(
                        f'generated artifact widget "{item.widget_id}" cannot be '
                        "used as an app source widget"
                    )
                    continue
                ref_key = (item.origin, item.widget_id)
                widget = catalog_by_key.get(ref_key)
                if not widget:
                    errors.append(
                        f'unknown widget (origin="{item.origin}", '
                        f'widget_id="{item.widget_id}")'
                    )
                    continue
                available_params = {
                    str(getattr(param, "name", "")): param
                    for param in (getattr(widget, "params", None) or [])
                    if getattr(param, "name", "")
                }
                state_params = (item.state or {}).get("params")
                if available_params and not state_params:
                    param_details = [
                        f"{name} ({getattr(param, 'type', '')}): "
                        f"{getattr(param, 'description', '')}"
                        for name, param in available_params.items()
                    ]
                    errors.append(
                        f'widget "{widget.name}" ({widget.widget_id}) has '
                        "configurable params and must set explicit state.params. "
                        f"Available params: {', '.join(param_details)}"
                    )
                    continue
                if state_params is not None:
                    if not isinstance(state_params, dict):
                        errors.append(
                            f'widget "{widget.name}" ({widget.widget_id}) has '
                            "`state.params`, but it must be an object"
                        )
                        continue
                    unknown_params = sorted(set(state_params) - set(available_params))
                    if unknown_params:
                        param_details = [
                            f"{name} ({getattr(param, 'type', '')}): "
                            f"{getattr(param, 'description', '')}"
                            for name, param in available_params.items()
                        ]
                        errors.append(
                            f'widget "{widget.name}" ({widget.widget_id}) does '
                            f"not support state.params {unknown_params}. "
                            f"Available params: {', '.join(param_details) or 'none'}"
                        )
                        continue
                    invalid_param_values: list[str] = []
                    for param_name, value in state_params.items():
                        param = available_params[param_name]
                        options = getattr(param, "options", None) or []
                        if not options:
                            continue
                        option_values = {
                            str(option.get("value", option.get("label", option)))
                            if isinstance(option, dict)
                            else str(getattr(option, "value", option))
                            for option in options
                        }
                        values = value if isinstance(value, list) else [value]
                        invalid_values = [
                            str(item)
                            for item in values
                            if str(item) not in option_values
                        ]
                        if invalid_values:
                            invalid_param_values.append(
                                f"{param_name}={invalid_values} "
                                f"(allowed: {', '.join(sorted(option_values))})"
                            )
                    if invalid_param_values:
                        errors.append(
                            f'widget "{widget.name}" ({widget.widget_id}) does '
                            "not support these state.params values: "
                            f"{'; '.join(invalid_param_values)}"
                        )
                        continue
                resolved_widgets[ref_key] = widget

        if errors:
            yield (
                "Error: cannot create app.\n- "
                + "\n- ".join(errors)
                + "\nUse the available widget context to select valid "
                "(origin, widget_id) pairs, then retry."
            )
            return

        selected_param_names = {
            str(getattr(param, "name", "")).casefold()
            for widget in resolved_widgets.values()
            for param in (getattr(widget, "params", None) or [])
            if getattr(param, "name", "")
        }
        app_name = " ".join(name.split()).strip()
        if not app_name or app_name.casefold() in selected_param_names:
            clean_description = " ".join(description.split())
            before_dashboard = clean_description.casefold().split(" dashboard", 1)[0]
            subject = clean_description[: len(before_dashboard)].strip(" ,.:;-")
            for article in ("a ", "an ", "the "):
                if subject.casefold().startswith(article):
                    subject = subject[len(article) :]
                    break
            app_name = (
                f"{subject.title()} Dashboard" if subject else "Generated Dashboard"
            )

        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
            )
        )

        app_tabs: dict[str, AppArtifactTab] = {}
        for tab in tabs:
            app_tabs[tab.id] = AppArtifactTab(
                id=tab.id,
                name=tab.name,
                layout=[
                    AppArtifactLayoutItem(
                        i=item.widget_id,
                        x=item.x,
                        y=item.y,
                        w=item.w,
                        h=item.h,
                        state=item.state,
                    )
                    for item in tab.layout
                ],
            )

        widget_refs: list[AppArtifactWidgetRef] = []
        seen_refs: set[tuple[str, str]] = set()
        for tab in tabs:
            for item in tab.layout:
                ref_key = (item.origin, item.widget_id)
                if ref_key in seen_refs:
                    continue
                seen_refs.add(ref_key)
                widget = resolved_widgets[ref_key]
                widget_refs.append(
                    AppArtifactWidgetRef(
                        i=widget.widget_id,
                        origin=widget.origin,
                        widget_id=widget.widget_id,
                        uuid=widget.uuid,
                        name=widget.name,
                    )
                )

        app = AppArtifactDef(
            name=app_name,
            description=description,
            tabs=app_tabs,
        )
        artifact = AppArtifact(
            name=app_name,
            description=description,
            app=app,
            widget_refs=widget_refs,
        )

        self._logging_service.info(
            "App artifact created: name=%s tabs=%d widgets=%d",
            app_name,
            len(tabs),
            len(widget_refs),
        )
        yield AppArtifactSSE(data=artifact)

        widget_count = sum(len(tab.layout) for tab in tabs)
        tab_count = len(tabs)
        yield (
            f'Created app "{app_name}" with {widget_count} '
            f"widget{'s' if widget_count != 1 else ''} across {tab_count} "
            f"tab{'s' if tab_count != 1 else ''}. The app is rendered in "
            "the workspace and can be opened as a dashboard."
        )

    async def llm_search_widgets(
        self,
        query: str,
        summary: str = "Searching widgets",
    ) -> AsyncGenerator[str | StatusUpdateSSE, None]:
        """Search connected Workspace widgets before choosing dashboard content.

        Use this before `llm_create_app` and whenever you need to inspect the
        user's connected widget catalog. Pass only search keywords, not the
        full user request. For broad topics, choose several adjacent keywords
        yourself. Pass an empty query to list available widgets. Results
        include exact `origin` and `widget_id` values for `llm_create_app`;
        the JSON result also includes each widget's available params with
        descriptions, current/default values, and static options when present.
        Generated note/table/html artifacts are omitted because they are not
        reusable app source widgets. If no widgets match, try another keyword
        query before concluding that relevant widgets are unavailable.
        """

        def compact(value: Any, max_length: int) -> str:
            text = " ".join(str(value or "").split())
            if len(text) <= max_length:
                return text
            return text[: max_length - 1].rstrip() + "…"

        widget_collection = self._app_widget_collection
        catalog_widgets = (
            [
                *[("primary", widget) for widget in (widget_collection.primary or [])],
                *[
                    ("secondary", widget)
                    for widget in (widget_collection.secondary or [])
                ],
                *[("extra", widget) for widget in (widget_collection.extra or [])],
            ]
            if widget_collection
            else []
        )
        terms = list(
            dict.fromkeys(
                query.casefold()
                .replace(",", " ")
                .replace("-", " ")
                .replace("_", " ")
                .split()
            )
        )
        matches: list[tuple[int, str, Widget]] = []
        seen_refs: set[tuple[str, str]] = set()
        for tier, widget in catalog_widgets:
            widget_id = str(getattr(widget, "widget_id", "") or "")
            is_generated_artifact = (
                widget_id in _GENERATED_ARTIFACT_WIDGET_IDS
                or widget_id.startswith(_GENERATED_ARTIFACT_WIDGET_PREFIXES)
            )
            if is_generated_artifact:
                continue
            ref_key = (widget.origin, widget.widget_id)
            if ref_key in seen_refs:
                continue

            params_text = " ".join(
                f"{getattr(param, 'name', '')} "
                f"{getattr(param, 'description', '')} "
                f"{getattr(param, 'current_value', '')} "
                f"{getattr(param, 'default_value', '')} "
                f"{getattr(param, 'options', '')}"
                for param in (getattr(widget, "params", None) or [])
            )
            metadata = getattr(widget, "metadata", None) or {}
            schema = metadata.get("schema") if isinstance(metadata, dict) else None
            schema_text = ""
            if isinstance(schema, dict):
                schema_text = " ".join(
                    str(schema.get(key, "") or "")
                    for key in ("tableName", "name", "description")
                )
            searchable = (
                " ".join(
                    str(value or "")
                    for value in (
                        widget.widget_id,
                        widget.name,
                        widget.description,
                        widget.origin,
                        widget.category,
                        widget.sub_category,
                        params_text,
                        schema_text,
                    )
                )
                .casefold()
                .replace(",", " ")
                .replace("-", " ")
                .replace("_", " ")
            )
            matched_term_count = sum(1 for term in terms if term in searchable)
            if terms and matched_term_count == 0:
                continue

            seen_refs.add(ref_key)
            matches.append((matched_term_count, tier, widget))

        matches.sort(key=lambda item: -item[0])
        returned_matches = matches[:12]
        results: list[dict[str, Any]] = []
        artifact_rows: list[dict[str, Any]] = []
        for _score, _tier, widget in returned_matches:
            results.append(
                {
                    "widget_id": compact(getattr(widget, "widget_id", ""), 120),
                    "name": compact(getattr(widget, "name", ""), 120),
                    "description": compact(
                        getattr(widget, "description", ""),
                        180,
                    ),
                    "category": compact(getattr(widget, "category", "") or "", 80),
                    "sub_category": compact(
                        getattr(widget, "sub_category", "") or "",
                        80,
                    ),
                    "origin": compact(getattr(widget, "origin", ""), 120),
                    "params": [
                        {
                            "name": compact(getattr(param, "name", ""), 80),
                            "type": compact(getattr(param, "type", ""), 40),
                            "description": compact(
                                getattr(param, "description", ""),
                                120,
                            ),
                            "current_value": getattr(param, "current_value", None),
                            "default_value": getattr(param, "default_value", None),
                            "options": [
                                option
                                for option in (getattr(param, "options", None) or [])[
                                    :12
                                ]
                            ],
                        }
                        for param in (getattr(widget, "params", None) or [])[:8]
                    ],
                }
            )
            artifact_rows.append(
                {
                    "Name": compact(getattr(widget, "name", ""), 120),
                    "Description": compact(getattr(widget, "description", ""), 220),
                    "Source": compact(getattr(widget, "origin", ""), 120),
                    "Category": compact(
                        getattr(widget, "category", "") or "",
                        120,
                    ),
                    "Subcategory": compact(
                        getattr(widget, "sub_category", "") or "",
                        120,
                    ),
                }
            )

        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
                details=[] if results else ["No matching widgets found."],
                artifacts=[
                    ClientArtifact(
                        uuid=uuid.uuid4(),
                        name=f"widget_search_results_{str(uuid.uuid4())[:5]}",
                        description=(
                            f"Widgets found for {query or 'available widgets'}"
                        ),
                        type="table",
                        content=artifact_rows,
                    )
                ]
                if artifact_rows
                else [],
            )
        )

        output_json = json.dumps(
            {
                "matches": results,
                "returned": len(results),
                "total": len(matches),
                "query_terms": terms,
                **(
                    {
                        "guidance": (
                            "No reusable widgets matched these terms. Retry once "
                            "with broader adjacent terms implied by the request. "
                            "If that still returns no relevant widgets, say the "
                            "dashboard cannot be created from the available "
                            "widgets."
                        )
                    }
                    if not results
                    else {}
                ),
            },
            default=str,
        )

        self._logging_service.info(
            "Widget search: query=%s returned=%d total=%d bytes=%d",
            query,
            len(results),
            len(matches),
            len(output_json),
        )
        yield output_json

    async def llm_create_table_from_text(
        self, text_data: str, summary: str = "Creating table"
    ) -> AsyncGenerator[ContextStructuredQueryResult | StatusUpdateSSE, None]:
        """Use this tool to convert unstructured text into a structured table format.

        Call this tool when you need to:
        - Transform text containing tabular data into a proper table
        - Convert lists, CSV-like text, or informal data into structured format
        - Create a table artifact that can be used for further analysis or visualization

        The resulting table will be stored as an artifact that you can reference
        later for creating charts or performing additional data operations.

        Parameters
        ----------
        text_data : str
            The unstructured text containing the data you want to convert to a table.
            This can be in any format: lists, CSV-like text, informal tables, etc.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """
        try:
            table_records = await extract_table_data(text_data, self._template_service)
            df = pd.DataFrame(table_records)
            table_json = df.to_json(orient="records", date_format="iso")
            table_artifact_id = str(uuid.uuid4())

            # Metadata - only include fields used by frontend
            metadata: dict[str, Any] = {}  # Removed unused metadata fields

            table_artifact = CopilotArtifact(
                content=table_json,
                source_info=SourceInfo(
                    type="artifact",
                    uuid=uuid.UUID(table_artifact_id),
                    name=f"table_{table_artifact_id[:8]}",
                    description=(
                        "Table created from text data using AI-driven extraction"
                    ),
                    metadata=metadata,
                    citable=False,  # Tables are tools, not sources
                ),
                data_format=RawObjectDataFormat(parse_as="table"),
            )

            self._logging_service.info("Successfully created table artifact from text.")

            table_client_artifact = table_artifact.to_client_artifact()
            if isinstance(table_client_artifact.content, str):
                try:
                    table_client_artifact.content = json.loads(
                        table_client_artifact.content
                    )
                except json.JSONDecodeError:
                    self._logging_service.warning(
                        "Unable to parse table_client_artifact as json"
                    )

            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message=summary,
                    artifacts=[table_client_artifact],
                )
            )

            answer = (
                f"Created structured table with {len(table_records)} data points "
                f"from text. You can now create charts from this data."
            )

            result = ContextStructuredQueryResult(
                answer=answer,
                citations=[],  # No citations - this is a tool, not a source
                artifacts=[table_artifact],
            )

            yield result

        except Exception as err:
            self._logging_service.error(f"Text extraction failed: {err}")
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Failed to extract data from text",
                    details=[{"Error": str(err)}],
                )
            )

    async def llm_create_chart_from_table(
        self,
        table_artifact_id: str,
        chart_type: str,
        chart_title: str = "",
        x_axis: str = "",
        y_axis: str = "",
        summary: str = "Creating chart",
    ) -> AsyncGenerator[ContextStructuredQueryResult | StatusUpdateSSE, None]:
        """Use this tool to create a chart visualization from an existing table
        artifact.

        Call this tool when you need to:
        - Visualize data from a table you previously created or loaded
        - Generate charts like line plots, bar charts, scatter plots, etc.
        - Create visual representations for better data understanding

        This tool requires a table artifact to already exist. First create or load
        a table, then use this tool with the table's artifact ID to visualize it.

        Parameters
        ----------
        table_artifact_id : str
            The ID of the table artifact to create a chart from.
            This should be the ID returned when you created or loaded the table.
        chart_type : str
            The type of chart to create (e.g., "line", "bar", "scatter", "pie", etc.).
        chart_title : str
            Optional title for the chart. If not provided, a default title will be used.
        x_axis : str
            Optional label for the x-axis.
        y_axis : str
            Optional label for the y-axis.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """
        try:
            # Get the table data from context (try multiple formats)
            table_context = self._context_service.get_context_by_name(table_artifact_id)
            if not table_context:
                # Try with table_ prefix if it's a UUID-like string
                table_name = (
                    f"table_{table_artifact_id[:8]}"
                    if "-" in table_artifact_id
                    else table_artifact_id
                )
                table_context = self._context_service.get_context_by_name(table_name)
            if not table_context:
                table_context = self._context_service.get_context_by_id(
                    table_artifact_id
                )
            if not table_context:
                raise ValueError(f"Table artifact {table_artifact_id} not found")

            # Parse table data
            table_data = json.loads(table_context.content)
            self._logging_service.info("Table data loaded")

            chart_params = await generate_chart_parameters(
                data=table_data,
                template_service=self._template_service,
                requested_chart_type=chart_type,
                context=f"Creating {chart_type} chart: {chart_title}",
            )

            chart_config = {
                "chart_type": chart_type,
                "title": chart_title or f"{chart_type.title()} Chart",
                "data": table_data,
            }
            self._logging_service.info(f"Chart being generated: {str(chart_title)}")

            # Add axis labels if provided
            if x_axis or y_axis:
                chart_config["axis_labels"] = {}
                if x_axis:
                    chart_config["axis_labels"]["x"] = x_axis
                if y_axis:
                    chart_config["axis_labels"]["y"] = y_axis

            # Frontend expects content to be the data array, not chart config
            chart_uuid = uuid.uuid4()
            chart_artifact_id = f"chart_artifact_{chart_uuid.hex[:8]}"

            chart_artifact = CopilotArtifact(
                content=json.dumps(
                    table_data
                ),  # Frontend expects content to be the data array
                source_info=SourceInfo(
                    type="artifact",
                    uuid=chart_uuid,
                    name=chart_artifact_id,
                    description=f"{chart_type.title()} chart visualization",
                    metadata={},  # Removed unused metadata fields
                    citable=False,  # Charts are visualizations, not sources
                ),
                data_format=RawObjectDataFormat(
                    parse_as="chart", chart_params=chart_params
                ),
            )

            chart_client_artifact = chart_artifact.to_client_artifact()
            if isinstance(chart_client_artifact.content, str):
                try:
                    chart_client_artifact.content = json.loads(
                        chart_client_artifact.content
                    )
                except json.JSONDecodeError:
                    self._logging_service.warning(
                        "Unable to parse chart_client_artifact as json"
                    )

            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message=summary,
                    artifacts=[chart_client_artifact],
                )
            )

            self._logging_service.info(
                f"Successfully created {chart_type} chart from table "
                f"{table_artifact_id}"
            )

            result = ContextStructuredQueryResult(
                answer=f"Created {chart_type} chart visualization from the data.",
                citations=[],  # No citations - visualization tool, not source
                artifacts=[chart_artifact],
            )

            yield result

        except Exception as err:
            self._logging_service.error(f"Chart creation failed: {err}")
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Failed to create chart",
                    details=[{"Error": str(err)}],
                )
            )
            raise FunctionCallError(f"Failed to create chart: {err}") from err

    async def llm_create_html_artifact(
        self,
        html_content: str,
        name: str,
        description: str,
        summary: str = "Creating HTML widget",
    ) -> AsyncGenerator[ContextStructuredQueryResult | StatusUpdateSSE, None]:
        """Use this tool to create and display HTML content.

        Use this tool when you need to create:
        - **HTML reports** (investment memos, research notes, analysis documents,
          skill-generated reports)
        - Metric cards or KPI displays (e.g., showing revenue, user counts, percentages)
        - Alert or notification boxes
        - Infographics or custom data summaries
        - Any styled HTML content including multi-page documents

        When generating HTML from skills or creating HTML reports, you MUST use
        this tool. Never stream raw HTML as text in your response - it will
        display as unrendered code.

        Do NOT use this tool for:
        - Plain text content (just respond with text)
        - Simple tables without custom styling (use table artifacts instead)

        IMPORTANT HTML REQUIREMENTS:
        - Use ONLY inline styles (style="...") - no <style> tags or external CSS
        - Do NOT include <script> tags - they will be stripped
        - Use semantic HTML (div, span, h1-h6, p, etc.)
        - Keep HTML concise and well-structured
        - Use modern CSS properties for styling (flexbox, grid, etc.)

        STYLING GUIDE (OpenBB Workspace):
        ALWAYS include a background color - never rely on page background.
        Use LIGHT cards with dark text for a clean, modern look.

        Colors:
        - Card background: #f1f5f9 (light gray)
        - Card border: #e2e8f0 (subtle border)
        - Text primary: #1e293b (dark)
        - Text secondary/labels: #64748b (muted)
        - Positive/growth: #16a34a (green)
        - Negative/decline: #dc2626 (red)
        - Warning: #d97706 (orange)

        Typography:
        - Font: system-ui, -apple-system, sans-serif
        - Large values: font-size: 1.75rem; font-weight: 700; color: #1e293b
        - Labels: font-size: 0.75rem; color: #64748b
        - Change indicators: font-size: 0.875rem

        Layout:
        - Border radius: 12px
        - Padding: 1rem
        - Border: 1px solid #e2e8f0
        - Use flexbox with gap for multiple cards

        IMPORTANT: Keep HTML compact. No unnecessary wrappers.

        Parameters
        ----------
        html_content : str
            The complete HTML content to display. Must use inline styles only.
            Follow the styling guide above for consistent appearance.
        name : str
            A short, descriptive name for this artifact (e.g., "revenue_metric_card").
            Use snake_case without spaces.
        description : str
            A brief description of what this visualization shows.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """
        # Validate HTML content - return error to LLM, not user-facing SSE
        if not html_content or not html_content.strip():
            self._logging_service.warning(
                "HTML artifact creation failed: empty content provided by LLM"
            )
            yield ContextStructuredQueryResult(
                answer=(
                    "Error: HTML content cannot be empty. "
                    "Please provide valid HTML content."
                ),
                citations=[],
                artifacts=[],
            )
            return

        # Size limit check (50KB) - return error to LLM, not user-facing SSE
        max_size = 50 * 1024
        if len(html_content) > max_size:
            self._logging_service.warning(
                "HTML artifact creation failed: content too large (%dKB > %dKB limit)",
                len(html_content) // 1024,
                max_size // 1024,
            )
            content_kb = len(html_content) // 1024
            limit_kb = max_size // 1024
            yield ContextStructuredQueryResult(
                answer=(
                    f"Error: HTML content exceeds {limit_kb}KB limit "
                    f"(provided: {content_kb}KB). Please reduce the content size."
                ),
                citations=[],
                artifacts=[],
            )
            return

        try:
            artifact_uuid = uuid.uuid4()
            html_artifact_name = f"html_{artifact_uuid.hex[:8]}"

            self._logging_service.info(
                "Creating HTML artifact: name=%s, content_length=%d",
                html_artifact_name,
                len(html_content),
            )

            html_artifact = CopilotArtifact(
                content=html_content,
                source_info=SourceInfo(
                    type="artifact",
                    uuid=artifact_uuid,
                    name=html_artifact_name,
                    description=description or "Custom HTML visualization",
                    metadata={"parse_as": "html"},  # For inline artifact rendering
                    citable=False,  # HTML widgets are visualizations, not sources
                ),
                data_format=RawObjectDataFormat(parse_as="html"),
            )

            # Load into context service for reference
            parsed_context = html_artifact.to_parsed_context()
            self._context_service.load_context(elements=[parsed_context])

            html_client_artifact = html_artifact.to_client_artifact()

            self._logging_service.info(
                "HTML artifact created successfully: uuid=%s", artifact_uuid
            )

            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message=summary,
                    artifacts=[html_client_artifact],
                )
            )

            result = ContextStructuredQueryResult(
                answer=f"Created HTML visualization '{html_artifact_name}'",
                citations=[],
                artifacts=[html_artifact],
            )

            yield result

        except Exception as err:
            self._logging_service.error("HTML artifact creation failed: %s", err)
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Failed to create HTML artifact",
                    details=[{"Error": str(err)}],
                )
            )

    async def llm_web_search(
        self, query: str, summary: str = "Searching the web"
    ) -> AsyncGenerator[
        StatusUpdateSSE | MessageChunkSSE | CitationCollectionSSE, None
    ]:
        """Use this tool to search the web for information.

        Use this tool when:
        - The user explicitly asks to search the web, OR
        - You cannot answer the user's question with the available tools and data, OR
        - A widget or tool call failed and you need an alternative data source.

        Parameters
        ----------
        query : str
            The query to search the web for.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with "Searching web for" and then you
            "complete the rest (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """
        async for event in self._web_search_llm_service.query(
            summary=summary,
            messages=[LlmClientMessage(role=RoleEnum.human, content=query)],
        ):
            yield event

    async def llm_generate_sql_query(
        self,
        user_request: str,
        widget_uuid: str,
        generate_query_only: bool,
        summary: str = "Generating SQL query",
        sql_widgets: list[SqlWidgetContext] | None = None,
    ) -> AsyncGenerator[SqlQueryGenerationResult | StatusUpdateSSE, None]:
        """Generate a SQL query from natural language for a SQL-enabled widget.

        Parameters
        ----------
        user_request : str
            Natural language description of the data request. When the user
            asks to fix, modify, or debug an existing query, ALWAYS include
            the current SQL in this field (e.g. "The current query is:
            SELECT ... — it returns no results. Fix it to ...").
        widget_uuid : str
            UUID of the SQL-enabled widget to query.
        generate_query_only : bool
            Whether to only generate the SQL query without additional context.
            If user only asked to create/write/generate SQL, set to True.
        summary : str
            Short action summary (3-5 words, starts with gerund, no period).
        sql_widgets : list[SqlWidgetContext] | None
            List of widgets that support SQL query generation.
        """
        widget_context = self._find_widget(widget_uuid, sql_widgets)
        if not widget_context:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Widget not found",
                    details=[f"Widget {widget_uuid} not available"],
                )
            )
            return

        if not self._sql_query_generation_service:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="SQL query generation not available",
                    details=["Service not configured"],
                )
            )
            return

        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
                details=[
                    {
                        "Origin": widget_context.widget_origin,
                        "Widget ID": widget_context.widget_id,
                        "Query": user_request,
                    }
                ],
            )
        )

        async for event in self._sql_query_generation_service.generate_query(
            user_request=user_request,
            widget_uuid=widget_uuid,
            generate_query_only=generate_query_only,
            sql_widget_dict=widget_context,
        ):
            yield event

    async def llm_generate_sql_query_snowflake(
        self,
        user_request: str,
        widget_uuid: str,
        generate_query_only: bool,
        semantic_view: str | None = None,
        summary: str = "Generating SQL query",
        sql_widgets: list[SqlWidgetContext] | None = None,
    ) -> AsyncGenerator[SqlQueryGenerationResult | StatusUpdateSSE, None]:
        """Generate a SQL query using Snowflake Cortex Analyst.

        This is the primary tool for generating SQL queries in Snowflake mode.
        Uses Cortex Analyst when a semantic view is available, with LLM
        fallback otherwise.

        Parameters
        ----------
        user_request : str
            Natural language description of the data request. When the user
            asks to fix, modify, or debug an existing query, ALWAYS include
            the current SQL in this field (e.g. "The current query is:
            SELECT ... — it returns no results. Fix it to ...").
        widget_uuid : str
            UUID of the SQL-enabled widget to query.
        generate_query_only : bool
            Whether to only generate the SQL query without additional context.
            If user only asked to create/write/generate SQL, set to True.
        semantic_view : str | None
            Optional semantic view FQN chosen from the available semantic views
            in the system prompt. Use only one of the provided FQNs. If a
            relevant semantic view exists, pass it here so Cortex Analyst can
            generate the query. Leave empty when none are relevant.
        summary : str
            Short action summary (3-5 words, starts with gerund, no period).
        sql_widgets : list[SqlWidgetContext] | None
            List of widgets that support SQL query generation.
        """
        widget_context = self._find_widget(widget_uuid, sql_widgets)
        if not widget_context:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Widget not found",
                    details=[f"Widget {widget_uuid} not available"],
                )
            )
            return

        if not self._snowflake_cortex_analyst_service:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Snowflake Cortex Analyst not available",
                    details=["Service not configured"],
                )
            )
            return

        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
                details=[
                    {
                        "Origin": widget_context.widget_origin,
                        "Widget ID": widget_context.widget_id,
                        "Query": user_request,
                    }
                ],
            )
        )

        async for event in self._generate_sql_via_cortex_analyst(
            user_request=user_request,
            widget_context=widget_context,
            generate_query_only=generate_query_only,
            requested_semantic_view=semantic_view,
        ):
            yield event

    async def _generate_sql_via_cortex_analyst(
        self,
        user_request: str,
        widget_context: SqlWidgetContext,
        generate_query_only: bool,
        requested_semantic_view: str | None = None,
    ) -> AsyncGenerator[SqlQueryGenerationResult | StatusUpdateSSE, None]:
        """Route SQL generation through SnowflakeCortexAnalystService.

        Builds a CodeGenerationRequest from the widget context and available
        semantic views (user-selected or chosen by the model from upstream
        candidates),
        delegates to
        SnowflakeCortexAnalystService.generate_code(),
        then maps the CodeGenerationResponse back to the streaming format expected
        by the copilot.
        """
        if not self._snowflake_cortex_analyst_service:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Snowflake Cortex Analyst not available",
                    details=["Service not configured"],
                )
            )
            return

        (
            semantic_view,
            semantic_models,
            selected_semantic_views,
            selection_reason,
        ) = self._build_semantic_request_config(
            requested_semantic_view=requested_semantic_view,
        )

        if semantic_status := self.build_semantic_view_status_update(
            selected_semantic_views, selection_reason
        ):
            yield semantic_status

        request = CodeGenerationRequest(
            widget_uuid=widget_context.widget_uuid,
            user_prompt=user_request,
            current_code=widget_context.current_sql,
            language="sql",
            sql_schema=widget_context.sql_schema,
            semantic_view=semantic_view,
            semantic_models=semantic_models,
        )

        try:
            response = await self._snowflake_cortex_analyst_service.generate_code(
                request
            )
        except CodeGenerationError as exc:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Could not generate SQL query",
                    details=[exc.message],
                )
            )
            return

        if not response.generated_code:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Could not generate SQL query",
                    details=["No SQL code was generated."],
                )
            )
            return

        generated_sql = response.generated_code.strip()
        is_cortex = response.generation_source == "cortex_analyst"
        self._logging_service.info("SQL code generated for widget editor flow")

        # Build the artifact in the same format as SqlQueryGenerationService
        query_uuid = uuid.uuid4()
        query_artifact_id = f"query_artifact_{str(query_uuid)[:8]}"
        query_data_source = {
            "origin": widget_context.widget_origin,
            "id": widget_context.widget_id,
            "widget_uuid": widget_context.widget_uuid,
        }

        query_artifact = CopilotArtifact(
            data_format=RawObjectDataFormat(
                parse_as="snowflake_query",
                query_data_source=query_data_source,
            ),
            content=SqlQueryGenerationService._format_sql_query(generated_sql),
            source_info=SourceInfo(
                type="artifact",
                uuid=query_uuid,
                name=query_artifact_id,
                description="Snowflake SQL query generated by AI copilot",
                metadata={
                    "parse_as": "snowflake_query",
                    "query_data_source": query_data_source,
                },
                citable=False,
            ),
        )

        query_client_artifact = query_artifact.to_client_artifact()
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message="Cortex Analyst generated SQL"
                if is_cortex
                else "SQL query generated",
                artifacts=[query_client_artifact] if not generate_query_only else [],
            )
        )

        result_model = (
            SqlQueryGenerationResult
            if generate_query_only
            else SqlQueryFunctionCallResult
        )
        yield result_model(
            sql_query=generated_sql,
            widget_uuid=widget_context.widget_uuid,
            widget_id=widget_context.widget_id,
            widget_origin=widget_context.widget_origin,
            artifacts=[query_artifact],
        )

    def _get_available_semantic_view_map(self) -> dict[str, AvailableSemanticView]:
        return {
            semantic_view.fqn: semantic_view
            for semantic_view in (self._available_semantic_views or [])
        }

    def _select_semantic_views_with_reason(
        self,
        requested_semantic_view: str | None = None,
    ) -> tuple[list[str] | None, str | None]:
        if self._semantic_views:
            self._logging_service.info(
                "Using explicitly selected semantic views: %s", self._semantic_views
            )
            return self._semantic_views, "Explicit selection"

        if not requested_semantic_view:
            return None, None

        available_semantic_view_map = self._get_available_semantic_view_map()
        if not available_semantic_view_map:
            self._logging_service.info(
                "Ignoring model-selected semantic view %s because no available "
                "semantic views were provided by the backend",
                requested_semantic_view,
            )
            return None, None

        if requested_semantic_view not in available_semantic_view_map:
            self._logging_service.info(
                "Ignoring model-selected semantic view %s because it is not in the "
                "available semantic view candidates",
                requested_semantic_view,
            )
            return None, None

        self._logging_service.info(
            "Using model-selected semantic view from available candidates: %s",
            requested_semantic_view,
        )
        return [requested_semantic_view], "Model selected available semantic view"

    def _build_semantic_model_refs(
        self, semantic_views: list[str] | None
    ) -> list[SemanticModelReference] | None:
        """Convert semantic views into SemanticModelReference list."""
        if not semantic_views or len(semantic_views) <= 1:
            return None

        return [SemanticModelReference(semantic_view=sv) for sv in semantic_views]

    def _build_semantic_request_config(
        self,
        requested_semantic_view: str | None = None,
    ) -> tuple[
        str | None,
        list[SemanticModelReference] | None,
        list[str] | None,
        str | None,
    ]:
        semantic_views, selection_reason = self._select_semantic_views_with_reason(
            requested_semantic_view=requested_semantic_view,
        )
        semantic_view = (
            semantic_views[0] if semantic_views and len(semantic_views) == 1 else None
        )
        semantic_models = (
            None if semantic_view else self._build_semantic_model_refs(semantic_views)
        )
        return semantic_view, semantic_models, semantic_views, selection_reason

    def build_semantic_view_status_update(
        self,
        semantic_views: list[str] | None,
        selection_reason: str | None,
    ) -> StatusUpdateSSE | None:
        if not semantic_views or not selection_reason:
            return None

        status_key = tuple(semantic_views)
        if status_key in self._reported_semantic_context_statuses:
            return None
        self._reported_semantic_context_statuses.add(status_key)

        return StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message="Using semantic view context",
                details=[
                    {
                        "Reason": selection_reason,
                        "Semantic Views": ", ".join(semantic_views),
                    }
                ],
            )
        )

    def get_prompt_semantic_context(
        self,
    ) -> tuple[list[str] | None, str | None]:
        if not self._semantic_views:
            return None, None
        return self._semantic_views, "Explicit selection"

    async def llm_generate_python_code(
        self,
        user_request: str,
        widget_uuid: str,
        generate_code_only: bool,
        semantic_view: str | None = None,
        summary: str = "Generating Python code",
        python_widgets: list[PythonWidgetContext] | None = None,
        sql_widgets: list[SqlWidgetContext] | None = None,
    ) -> AsyncGenerator[
        PythonCodeGenerationResult | PythonCodeFunctionCallResult | StatusUpdateSSE,
        None,
    ]:
        """Generate Python code from natural language for a code-enabled widget.

        The generated code has access to a `session` variable for data access.

        Parameters
        ----------
        user_request : str
            Natural language description of what the code should do.
        widget_uuid : str
            UUID of the Python-capable widget to generate code for.
        generate_code_only: bool
            Whether to only generate the Python code without additional context.
            If user only asked to create/write/generate code, set to True.
        semantic_view : str | None
            Optional semantic view FQN chosen from the available semantic views
            in the system prompt. Use only one of the provided FQNs. If a
            relevant semantic view exists, pass it here so Cortex Analyst can
            generate the SQL hint. Leave empty when none are relevant.
        summary : str
            Short action summary (3-5 words, starts with gerund, no period).
        python_widgets : list[PythonWidgetContext] | None
            List of widgets that support Python code generation.
        sql_widgets : list[SqlWidgetContext] | None
            List of widgets that support SQL query generation.
        """
        widget_context = self._find_widget(widget_uuid, python_widgets)
        if not widget_context:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Widget not found",
                    details=[f"Widget {widget_uuid} not available"],
                )
            )
            return

        if not self._python_code_generation_service:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Python code generation not available",
                    details=["Service not configured"],
                )
            )
            return

        # Yield initial status update showing which widget is being used
        widget_name = widget_context.widget_id or "Python widget"
        widget_origin = widget_context.widget_origin or ""
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
                details=[
                    {"Widget": widget_name, "Origin": widget_origin},
                ],
            )
        )

        (
            semantic_view,
            semantic_models,
            selected_semantic_views,
            selection_reason,
        ) = self._build_semantic_request_config(requested_semantic_view=semantic_view)

        # When semantic views are explicitly selected or chosen by the model
        # from upstream candidates, use Cortex Analyst to generate SQL first
        # and enrich the user prompt before the Python LLM call.
        if self._snowflake_cortex_analyst_service and (
            semantic_view or semantic_models
        ):
            if semantic_status := self.build_semantic_view_status_update(
                selected_semantic_views, selection_reason
            ):
                yield semantic_status
            try:
                sql_request = CodeGenerationRequest(
                    widget_uuid=widget_uuid,
                    user_prompt=user_request,
                    language="sql",
                    semantic_view=semantic_view,
                    semantic_models=semantic_models,
                )
                response = await self._snowflake_cortex_analyst_service.generate_code(
                    sql_request
                )
                if response.generated_code:
                    if response.generation_source == "cortex_analyst":
                        source_widget = (
                            sql_widgets[0] if sql_widgets else widget_context
                        )
                        query_data_source = {
                            "origin": source_widget.widget_origin,
                            "id": source_widget.widget_id,
                            "widget_uuid": source_widget.widget_uuid,
                        }
                        query_uuid = uuid.uuid4()
                        query_artifact = CopilotArtifact(
                            data_format=RawObjectDataFormat(
                                parse_as="snowflake_query",
                                query_data_source=query_data_source,
                            ),
                            content=SqlQueryGenerationService._format_sql_query(
                                response.generated_code
                            ),
                            source_info=SourceInfo(
                                type="artifact",
                                uuid=query_uuid,
                                name=f"query_artifact_{str(query_uuid)[:8]}",
                                description=(
                                    "Snowflake SQL context generated by Cortex "
                                    "Analyst for Python code generation"
                                ),
                                metadata={
                                    "parse_as": "snowflake_query",
                                    "query_data_source": query_data_source,
                                },
                                citable=False,
                            ),
                        )
                        yield StatusUpdateSSE(
                            data=StatusUpdateSSEData(
                                eventType="INFO",
                                message=(
                                    "Cortex Analyst generated SQL context for Python"
                                ),
                                artifacts=[query_artifact.to_client_artifact()],
                            )
                        )
                    user_request = (
                        f"{user_request}\n\n"
                        f"Use the following SQL query to retrieve the data:\n"
                        f"```sql\n{response.generated_code}\n```"
                    )
            except (CodeGenerationError, RuntimeError) as exc:
                self._logging_service.warning(
                    "Cortex Analyst SQL hint failed for Python "
                    "path, proceeding without: %s",
                    exc,
                )

        try:
            async for event in self._python_code_generation_service.generate_code(
                user_request=user_request,
                widget_uuid=widget_uuid,
                python_widget=widget_context,
                generate_code_only=generate_code_only,
                sql_widgets=sql_widgets,
            ):
                yield event
        except Exception as e:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Python code generation failed",
                    details=[str(e)],
                )
            )

    def _is_function_call_result(
        self,
        response: (
            ContextStructuredQueryResult
            | list[DocumentQueryResult]
            | ContextUnstructuredQueryResult
            | SqlQueryGenerationResult
            | PythonCodeGenerationResult
        ),
    ) -> bool:
        if isinstance(response, ContextStructuredQueryResult):
            return True
        if isinstance(response, list) and all(
            isinstance(item, DocumentQueryResult) for item in response
        ):
            return True
        if isinstance(response, ContextUnstructuredQueryResult):
            return True

        if isinstance(response, SqlQueryFunctionCallResult):
            return False

        if isinstance(response, SqlQueryGenerationResult):
            return True

        if isinstance(response, PythonCodeGenerationResult) and not isinstance(
            response, PythonCodeFunctionCallResult
        ):
            return True

        return False

    async def handle_function_calls(
        self, response: FunctionCall
    ) -> AsyncGenerator[
        list[Message[Any]]
        | StatusUpdateSSE
        | MessageChunkSSE
        | AppArtifactSSE
        | CitationCollectionSSE
        | SqlQueryGenerationResult
        | SqlQueryFunctionCallResult
        | PythonCodeGenerationResult
        | PythonCodeFunctionCallResult,
        None,
    ]:
        # Log function call with minimal schema info to avoid massive logs
        func_name = response.function.__name__
        args_summary = {
            k: v for k, v in response.arguments.items() if k != "sql_widgets"
        }
        if "sql_widgets" in response.arguments:
            num_widgets = len(response.arguments["sql_widgets"])
            args_summary["sql_widgets"] = f"[{num_widgets} widgets]"
        self._logging_service.info(
            "Native function call: %s with args: %s", func_name, args_summary
        )

        is_web_search = response.function.__name__ == self.llm_web_search.__name__

        try:
            if is_web_search:
                # Special handling for web search to accommodate streaming and final result  # noqa: E501
                streamed_content_buffer = ""
                collected_citations = []
                async for event in response():
                    if isinstance(event, MessageChunkSSE):
                        streamed_content_buffer += event.data.delta
                        continue
                    elif isinstance(event, CitationCollectionSSE):
                        # Add web search citations to the citation service
                        for citation in event.data.citations:
                            self._citation_service.add_citation(citation)
                            collected_citations.append(citation)
                    yield event

                # Enhance the function result with citation information for the AI
                enhanced_content = streamed_content_buffer
                if collected_citations:
                    citation_info = (
                        "\n\n=== CITATIONS FOR THIS CONTENT ===\n"
                        "You MUST cite these sources when using "
                        "information from them.\n\n"
                    )
                    for citation in collected_citations:
                        citation_info += (
                            f"Source: {citation.source_info.name}\n"
                            f"Citation format: "
                            f"<|start_citation_id|>{citation.id}<|end_citation_id|>\n\n"
                        )
                    citation_info += (
                        "IMPORTANT CITATION RULES:\n"
                        "1. Place the EXACT citation format shown above after "
                        "sentences using that source\n"
                        "2. Use the COMPLETE UUID inside the tags "
                        "(copy exactly as shown)\n"
                        "3. NEVER use just <|end_citation_id|> alone\n"
                        "4. Each citation must have both start AND end tags\n"
                        "Example: 'The tariff rate is 15%. "
                        "<|start_citation_id|>UUID-HERE<|end_citation_id|>'"
                    )
                    enhanced_content += citation_info

                llm_messages = [
                    AssistantMessage(response),
                    FunctionResultMessage(
                        content=sanitize_str(enhanced_content),
                        function_call=response,
                    ),
                ]
                yield llm_messages
                return

            # Default handling for other native function calls
            error_occurred = False
            error_message = ""
            async for event in response():
                if self._is_function_call_result(event):
                    # Create function call and result messages
                    llm_messages = [
                        AssistantMessage(response),
                        FunctionResultMessage(
                            content=self._handle_native_function_call_result(event),
                            function_call=response,
                        ),
                    ]
                    yield llm_messages
                elif isinstance(event, str):
                    # Handle simple string results (like from _llm_think)
                    llm_messages = [
                        AssistantMessage(response),
                        FunctionResultMessage(
                            content=event,
                            function_call=response,
                        ),
                    ]
                    yield llm_messages
                elif isinstance(event, StatusUpdateSSE):
                    # Check if this is an error event
                    if event.data.eventType == "ERROR":
                        error_occurred = True
                        details = [str(d) for d in (event.data.details or [])]
                        error_message = f"{event.data.message}: {', '.join(details)}"
                    yield event
                else:
                    yield event

            # If an error occurred but no result was yielded, add error to history
            if error_occurred:
                llm_messages = [
                    AssistantMessage(response),
                    FunctionResultMessage(
                        content=f"Error: {error_message}",
                        function_call=response,
                    ),
                ]
                yield llm_messages

        except FunctionCallError as err:
            self._logging_service.error("Function call failed: %s", err)
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message=str(err),
                    details=[response.arguments],
                )
            )
            llm_messages = [
                AssistantMessage(response),
                FunctionResultMessage(
                    content=str(err),
                    function_call=response,
                ),
            ]
            yield llm_messages

    def _handle_native_function_call_result(
        self,
        result: (
            list[DocumentQueryResult]
            | ContextStructuredQueryResult
            | ContextUnstructuredQueryResult
            | SqlQueryGenerationResult
            | PythonCodeGenerationResult
        ),
    ) -> str:
        output = ""
        results = result if isinstance(result, list) else [result]  # type: ignore[list-item]
        for res in results:
            # If we have artifacts, we need to add them to the context service.
            if (
                isinstance(res, DocumentQueryResult)
                or isinstance(res, ContextStructuredQueryResult)
                or isinstance(res, SqlQueryGenerationResult)
                or isinstance(res, PythonCodeGenerationResult)
            ) and res.artifacts:
                for artifact in res.artifacts:
                    self._logging_service.debug("Adding artifact to context service...")
                    # First add the artifact to the context service
                    parsed_context = artifact.to_parsed_context()
                    self._context_service.load_context(elements=[parsed_context])[0]
                    # Then produce the result message we'll show to Copilot
                    if artifact.data_format.parse_as == "table":
                        MAX_TABLE_ROWS = 50

                        table_data = (
                            json.loads(artifact.content)
                            if isinstance(artifact.content, str)
                            else artifact.content
                        )
                        table = pd.DataFrame(table_data)
                        table_preview = table.head(MAX_TABLE_ROWS).to_json(
                            orient="records", date_format="iso", lines=True
                        )
                        output += self._template_service.render_copilot_native_function_call_result(  # noqa: E501
                            answer=res.answer,
                            artifact=artifact,
                            table_preview=table_preview,
                            has_more_rows=len(table) > MAX_TABLE_ROWS,
                            remaining_rows=len(table) - MAX_TABLE_ROWS,
                            citations=res.citations or None,
                        )
                    elif artifact.data_format.parse_as == "chart":
                        output += self._template_service.render_copilot_native_function_call_result(  # noqa: E501
                            answer=res.answer,
                            artifact=artifact,
                            citations=res.citations or None,
                        )
                    elif artifact.data_format.parse_as == "text":
                        output += self._template_service.render_copilot_native_function_call_result(  # noqa: E501
                            answer=res.answer,
                            artifact=artifact,
                            citations=res.citations or None,
                        )
                    elif artifact.data_format.parse_as == "html":
                        # HTML artifacts - provide a brief summary for the LLM
                        output += self._template_service.render_copilot_native_function_call_result(  # noqa: E501
                            answer=res.answer,
                            artifact=artifact,
                            citations=res.citations or None,
                        )
                    elif artifact.data_format.parse_as == "snowflake_query":
                        if isinstance(res, SqlQueryGenerationResult):
                            output += self._template_service.render_copilot_snowflake_query_result(  # noqa: E501
                                sql_query=res.sql_query,
                                artifact=artifact,
                                generate_query_only=not isinstance(
                                    res, SqlQueryFunctionCallResult
                                ),
                            )
                    elif artifact.data_format.parse_as == "snowflake_python":
                        if isinstance(res, PythonCodeGenerationResult):
                            output += self._template_service.render_copilot_python_code_result(  # noqa: E501
                                python_code=res.python_code,
                                artifact=artifact,
                            )
            # If we don't have artifacts, we can just return the answer.
            else:
                output += (
                    self._template_service.render_copilot_native_function_call_result(  # noqa: E501
                        answer=res.answer,
                        citations=res.citations or None,
                    )
                )
        return sanitize_str(output)
