import json
import re
from pathlib import Path
from typing import (
    Any,
)

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from openbb_ai.models import (
    AgentTool,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    RoleEnum,
    WidgetCollection,
    WidgetParamOption,
    WorkspaceState,
)

from .. import constants
from ..models import (
    AvailableSemanticView,
    Citation,
    CopilotArtifact,
    DataSource,
    Document,
    DocumentAgentArtifact,
    PythonWidgetContext,
    RetrievedDocumentChunk,
    SkillCatalogEntry,
    SkillGenerationRequest,
    SkillPayload,
    SqlTableInfo,
    SqlWidgetContext,
    StructuredContext,
    UnstructuredContext,
    WebContext,
    Widget,
    WidgetTitleDescriptionRequest,
)


class TemplateService:
    """Provide templating functionality for prompts and other text formatting."""

    def __init__(
        self,
        base_path: Path | None = None,
        current_datetime: str | None = None,
        workspace_state: WorkspaceState | None = None,
        workspace_options: dict[str, Any] | None = None,
        semantic_views: list[str] | None = None,
        available_semantic_views: list[AvailableSemanticView] | None = None,
    ):
        base_path = base_path if base_path else Path(__file__).parent.parent
        self._current_datetime = current_datetime
        self._workspace_state = workspace_state
        self._workspace_options = workspace_options or {}
        self._semantic_views = semantic_views
        self._available_semantic_views = available_semantic_views
        self._env = Environment(
            loader=FileSystemLoader(base_path / "templates"),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._env.filters["llm_safe_string"] = self._filter_escape_string

    @staticmethod
    def _filter_escape_string(value: str) -> str:
        value = str(value)
        cleaned_message = re.sub(r"(?<!\{)\{(?!{)", "{{", value)
        cleaned_message = re.sub(r"(?<!\})\}(?!})", "}}", cleaned_message)
        return cleaned_message

    def _get_template(self, template_name: str) -> Template:
        return self._env.get_template(template_name)

    def render_template(
        self, template_name: str, template_context: dict[str, Any] | None = None
    ) -> str:
        template_context = template_context or {}

        template = self._get_template(template_name)
        return template.render(**template_context)

    def render_copilot_system_prompt(
        self,
        widget_collection: WidgetCollection | None = None,
        unstructured_context: list[UnstructuredContext] | None = None,
        structured_context: list[StructuredContext] | None = None,
        documents: list[Document] | None = None,
        web_pages: list[WebContext] | None = None,
        tools: list[AgentTool] | None = None,
        sql_widgets: list[SqlWidgetContext] | None = None,
        python_widgets: list[PythonWidgetContext] | None = None,
        skills_catalog: list[SkillCatalogEntry] | None = None,
        selected_skills: list[SkillPayload] | None = None,
        semantic_views: list[str] | None = None,
        available_semantic_views: list[AvailableSemanticView] | None = None,
    ) -> str:
        return self.render_template(
            template_name="copilot_system_prompt_template.jinja",
            template_context={
                "unstructured_context": unstructured_context,
                "structured_context": structured_context,
                "documents": documents,
                "widget_collection": widget_collection,
                "web_pages": web_pages,
                "tools": tools,
                "sql_widgets": sql_widgets,
                "python_widgets": python_widgets,
                "skills_catalog": skills_catalog,
                "selected_skills": selected_skills,
                "current_datetime": self._current_datetime,
                "workspace_state": self._workspace_state,
                "workspace_options": self._workspace_options,
                "llm_web_search_enabled": constants.LLM_WEB_SEARCH_ENABLED,
                "snowflake_native_app": constants.SNOWFLAKE_NATIVE_APP,
                "semantic_views": (
                    semantic_views
                    if semantic_views is not None
                    else self._semantic_views
                ),
                "available_semantic_views": (
                    available_semantic_views
                    if available_semantic_views is not None
                    else self._available_semantic_views
                ),
            },
        )

    def render_copilot_user_prompt(self, query: str) -> str:
        return self.render_template(
            template_name="copilot_user_query_prompt_template.jinja",
            template_context={"query": query, "workspace_state": self._workspace_state},
        )

    def render_copilot_document_agent_retrieved_document_chunks(
        self, retrieved_document_chunks: list[RetrievedDocumentChunk]
    ) -> str:
        return self.render_template(
            template_name="copilot_document_agent_retrieved_document_chunks_template.jinja",
            template_context={"retrieved_document_chunks": retrieved_document_chunks},
        )

    def render_copilot_docs_create_sql_table_name_prompt(
        self, filename_or_sheet_name: str
    ) -> str:
        return self.render_template(
            template_name="copilot_docs_create_sql_table_name_prompt.jinja",
            template_context={"filename_or_sheet_name": filename_or_sheet_name},
        )

    def render_copilot_docs_query_image_prompt(self) -> str:
        return self.render_template(
            template_name="copilot_docs_query_image_prompt_template.jinja",
            template_context={
                "current_datetime": self._current_datetime,
            },
        )

    def render_copilot_sql_agent_system_prompt(
        self,
        sql_tables_info: list[SqlTableInfo],
    ) -> str:
        # Get table relationships for normalized data
        table_relationships: dict[str, dict[str, Any]] = {}
        query_examples: list[dict[str, str]] = []

        # Check if we have any normalized tables (by looking for tables with
        # relationship metadata)
        has_normalized_tables = any(
            table.metadata and table.metadata.get("table_type") in ["parent", "child"]
            for table in sql_tables_info
        )

        if has_normalized_tables:
            # Create a temporary SqlAgentService to get relationships
            # We need to recreate the relationships from the table info
            for table in sql_tables_info:
                if table.metadata and table.metadata.get("table_type") == "parent":
                    parent_name = table.table_name

                    if parent_name not in table_relationships:
                        table_relationships[parent_name] = {"child_tables": []}

                    # Find child tables for this parent
                    for child_table in sql_tables_info:
                        if (
                            child_table.metadata
                            and child_table.metadata.get("table_type") == "child"
                            and child_table.metadata.get("parent_table") == parent_name
                        ):
                            original_column = child_table.metadata["parent_column"]
                            data_type = child_table.metadata["nested_data_type"]

                            child_info = {
                                "table_name": child_table.table_name,
                                "original_column": original_column,
                                "join_condition": (
                                    f"{child_table.table_name}.group_id = "
                                    f"{parent_name}.{original_column}"
                                ),
                                "data_type": data_type,
                                "description": (
                                    f"Child table containing {data_type} data "
                                    f"from {original_column} column"
                                ),
                            }

                            table_relationships[parent_name]["child_tables"].append(
                                child_info
                            )

                            # Generate simplified query examples (one per data type)
                            if data_type == "array" and not any(
                                ex.get("data_type") == "array" for ex in query_examples
                            ):
                                query_examples.append(
                                    {
                                        "data_type": "array",
                                        "description": (
                                            "Join array data with parent table"
                                        ),
                                        "query": (
                                            f"SELECT parent.*, child.sequence, "
                                            f"child.value\n"
                                            f"FROM {parent_name} parent\n"
                                            f"JOIN {child_table.table_name} child ON "
                                            f"child.group_id = "
                                            f"parent.{original_column}\n"
                                            f"ORDER BY parent.id, child.sequence;"
                                        ),
                                    }
                                )
                            elif data_type == "object" and not any(
                                ex.get("data_type") == "object" for ex in query_examples
                            ):
                                query_examples.append(
                                    {
                                        "data_type": "object",
                                        "description": (
                                            "Join object data with parent table"
                                        ),
                                        "query": (
                                            f"SELECT parent.*, child.key, child.value\n"
                                            f"FROM {parent_name} parent\n"
                                            f"JOIN {child_table.table_name} child ON "
                                            f"child.group_id = "
                                            f"parent.{original_column};"
                                        ),
                                    }
                                )

        # Use self._current_datetime to get the user's datetime, not the server's
        return self.render_template(
            "copilot_sql_agent_system_prompt_template.jinja",
            {
                "sql_tables_info": sql_tables_info,
                "current_datetime": self._current_datetime,
                "table_relationships": table_relationships,
                "query_examples": query_examples,
            },
        )

    def render_copilot_document_agent_system_prompt(
        self,
        documents: list[Document],
        artifacts: dict[str, DocumentAgentArtifact],
    ) -> str:
        return self.render_template(
            template_name="copilot_document_agent_system_prompt_template.jinja",
            template_context={
                "documents": documents,
                "artifacts": artifacts,
                "current_datetime": self._current_datetime,
            },
        )

    def render_copilot_external_data_source_search_prompt(
        self, user_query: str, data_source_hint: str
    ) -> str:
        template_name = "copilot_external_data_source_search_prompt_template.jinja"
        return self.render_template(
            template_name=template_name,
            template_context={
                "user_query": user_query,
                "data_source_hint": data_source_hint,
            },
        )

    def render_copilot_external_data_source_search_results(
        self, data_sources: list[DataSource]
    ) -> str:
        template_name = "copilot_external_data_source_search_results_template.jinja"
        return self.render_template(
            template_name=template_name,
            template_context={"data_sources": data_sources},
        )

    def render_copilot_native_function_call_result(
        self,
        answer: str,
        artifact: CopilotArtifact | None = None,
        table_preview: str | None = None,
        has_more_rows: bool | None = False,
        remaining_rows: int | None = None,
        citations: list[Citation] | None = None,
    ) -> str:
        return self.render_template(
            template_name="copilot_native_function_call_result_template.jinja",
            template_context={
                "answer": answer,
                "artifact": artifact,
                "table_preview": table_preview,
                "has_more_rows": has_more_rows,
                "remaining_rows": remaining_rows,
                "citations": citations,
            },
        )

    def render_copilot_snowflake_query_result(
        self,
        sql_query: str,
        artifact: CopilotArtifact | None = None,
        generate_query_only: bool = False,
    ) -> str:
        return self.render_template(
            template_name="copilot_snowflake_query_result_template.jinja",
            template_context={
                "sql_query": sql_query,
                "artifact": artifact,
                "generate_query_only": generate_query_only,
            },
        )

    def render_copilot_python_code_result(
        self, python_code: str, artifact: CopilotArtifact | None = None
    ) -> str:
        return self.render_template(
            template_name="copilot_python_code_result_template.jinja",
            template_context={"python_code": python_code, "artifact": artifact},
        )

    def render_copilot_web_search_system_prompt(
        self, messages: list[LlmClientMessage]
    ) -> str:
        return self.render_template(
            template_name="copilot_web_search_system_prompt_template.jinja",
            template_context={"messages": messages},
        )

    def render_generate_widget_title_and_description_prompt(
        self,
        widget_generation_request: WidgetTitleDescriptionRequest,
    ) -> str:
        return self.render_template(
            "generate_widget_title_and_description_prompt_template.jinja",
            widget_generation_request.model_dump(),
        )

    def render_generate_skill_prompt(
        self,
        skill_generation_request: SkillGenerationRequest,
    ) -> str:
        return self.render_template(
            "generate_skill_prompt_template.jinja",
            skill_generation_request.model_dump(),
        )

    def render_generate_input_arguments_prompt(
        self,
        query: str,
        data_source: DataSource,
        extra_param_options: dict[str, list[WidgetParamOption]] | None = None,
    ) -> str:
        return self.render_template(
            template_name="copilot_generate_input_arguments_prompt_template.jinja",
            template_context={
                "current_datetime": self._current_datetime,
                "query": query,
                "data_source": data_source,
                "extra_param_options": extra_param_options,
            },
        )

    def render_filter_input_arg_options_prompt(
        self,
        user_query: str,
        param_name: str,
        data_source: DataSource,
        widget_param_options: list[WidgetParamOption],
    ):
        return self.render_template(
            template_name="copilot_filter_input_arg_options_prompt_template.jinja",
            template_context={
                "user_query": user_query,
                "data_source": data_source,
                "param_name": param_name,
                "param_description": getattr(
                    data_source.input_fields.get(param_name, {}), "description", None
                ),
                "widget_param_options": widget_param_options,
            },
        )

    def render_generate_input_arg_options_query_prompt(
        self,
        user_query: str,
        data_source: DataSource,
        param_name: str,
    ) -> str:
        return self.render_template(
            template_name="copilot_generate_input_arg_options_query_template.jinja",
            template_context={
                "user_query": user_query,
                "data_source": data_source,
                "param_name": param_name,
            },
        )

    def render_generate_chat_title_prompt(
        self,
        messages: (
            list[LlmClientMessage | LlmClientFunctionCallResultMessage] | None
        ) = None,
    ) -> str:
        result = self.render_template(
            template_name="generate_chat_title_prompt_template.jinja",
            template_context={
                "messages": messages,
            },
        )
        return result

    def render_generate_dashboard_title_prompt(
        self,
        widgets: list[Widget],
    ) -> str:
        result = self.render_template(
            template_name="generate_dashboard_title_prompt_template.jinja",
            template_context={
                "widgets": widgets,
            },
        )
        return result

    def render_prompt_enhancement_prompt(
        self,
        messages: (
            list[LlmClientMessage | LlmClientFunctionCallResultMessage] | None
        ) = None,
        context: list[Any] | None = None,
        widgets: Any | None = None,
        tools: list[AgentTool] | None = None,
    ) -> str:
        # Extract user query from the last human message
        user_query = ""
        if messages:
            for message in reversed(messages):
                if (
                    isinstance(message, LlmClientMessage)
                    and message.role == RoleEnum.human
                    and isinstance(message.content, str)
                ):
                    user_query = message.content
                    break

        return self.render_template(
            template_name="prompt_enhancement_template.jinja",
            template_context={
                "messages": messages,
                "context": context,
                "widgets": widgets,
                "user_query": user_query,
                "tools": tools,
                "workspace_options": self._workspace_options,
                "workspace_state": self._workspace_state,
                "llm_web_search_enabled": constants.LLM_WEB_SEARCH_ENABLED,
            },
        )

    def render_copilot_context(self) -> str:
        """Render the copilot context template."""
        return self.render_template(
            template_name="copilot_context_template.jinja",
            template_context={
                "current_datetime": self._current_datetime,
                "workspace_state": self._workspace_state,
                "workspace_options": self._workspace_options,
                "llm_web_search_enabled": constants.LLM_WEB_SEARCH_ENABLED,
                "snowflake_native_app": constants.SNOWFLAKE_NATIVE_APP,
            },
        )

    def render_mcp_to_table(self, tool_name: str, raw_output: str) -> str:
        """Render the MCP to table conversion template."""
        return self.render_template(
            template_name="mcp_to_table.jinja",
            template_context={
                "tool_name": tool_name,
                "raw_output": raw_output,
            },
        )

    def _truncate_schema_for_llm(
        self,
        sql_schema: dict,
        current_sql: str | None = None,
        max_chars: int = 100_000,
    ) -> dict:
        """Truncate SQL schema to fit within token limits.

        Prioritizes keeping tables that are referenced in current_sql.
        Limits to approximately max_chars of JSON output.

        Parameters
        ----------
        sql_schema : dict
            The full SQL schema dictionary.
        current_sql : str | None
            Current SQL query - tables referenced here are prioritized.
        max_chars : int
            Maximum characters for the JSON output (default ~100K chars ≈ 25K tokens).

        Returns
        -------
        dict
            Truncated schema that fits within limits.
        """
        if not sql_schema:
            return sql_schema

        # Quick check - if already small enough, return as-is
        schema_json = json.dumps(sql_schema, indent=2)
        if len(schema_json) <= max_chars:
            return sql_schema

        # Schema is too large - we need to truncate
        # Strategy: Prioritize tables referenced in current_sql, then by size
        current_sql_upper = (current_sql or "").upper()

        # Separate into referenced tables (priority) and others
        referenced_tables = []
        other_tables = []

        for key, value in sql_schema.items():
            table_json = json.dumps({key: value}, indent=2)
            table_size = len(table_json)

            # Check if table name appears in current SQL
            # key format is typically "DATABASE.SCHEMA.TABLE" or just table name
            table_parts = key.upper().split(".")
            is_referenced = any(part in current_sql_upper for part in table_parts)

            if is_referenced:
                referenced_tables.append((key, value, table_size))
            else:
                other_tables.append((key, value, table_size))

        # Sort other tables by size (smaller first)
        other_tables.sort(key=lambda x: x[2])

        # Build truncated schema: referenced tables first, then fill with others
        truncated = {}
        current_size = 2  # for "{}"

        # Add all referenced tables first (they're needed for the query)
        for key, value, size in referenced_tables:
            truncated[key] = value
            current_size += size

        # Fill remaining space with other tables
        for key, value, size in other_tables:
            if current_size + size > max_chars:
                break
            truncated[key] = value
            current_size += size

        # Add truncation notice if we truncated
        omitted_count = len(sql_schema) - len(truncated)
        if omitted_count > 0:
            truncated["__truncated__"] = (
                f"Schema truncated. {omitted_count} tables omitted due to size limits. "
                f"Only tables relevant to the current query are included."
            )

        return truncated

    def render_sql_query_generation_prompt(
        self,
        sql_schema: dict,
        current_sql: str | None = None,
        data_sample: list[dict] | None = None,
    ) -> str:
        """Render the SQL query generation prompt template.

        Parameters
        ----------
        sql_schema : dict
            The SQL schema as a dictionary with database, table, and column info.
        current_sql : str | None
            The current SQL query being viewed (optional).
        data_sample : list[dict] | None
            Sample rows from previous widget execution (optional).

        Returns
        -------
        str
            The rendered prompt template.
        """
        # Truncate schema to prevent exceeding model context limits
        truncated_schema = self._truncate_schema_for_llm(sql_schema, current_sql)

        return self.render_template(
            template_name="copilot_sql_query_generation_prompt_template.jinja",
            template_context={
                "sql_schema": json.dumps(truncated_schema, indent=2),
                "current_sql": current_sql,
                "data_sample": json.dumps(data_sample, indent=2, default=str)
                if data_sample
                else None,
            },
        )

    def render_python_code_generation_prompt(
        self,
        current_code: str | None = None,
        sql_widgets: list[SqlWidgetContext] | None = None,
    ) -> str:
        """Render the Python code generation prompt template.

        Parameters
        ----------
        current_code : str | None
            The current Python code in the widget (optional).
        sql_widgets : list[SqlWidgetContext] | None
            List of SQL widget contexts available (optional).

        Returns
        -------
        str
            The rendered prompt template.
        """
        return self.render_template(
            template_name="copilot_python_code_generation_prompt_template.jinja",
            template_context={
                "current_code": current_code,
                "sql_widgets": sql_widgets,
            },
        )
