"""SQL Query Generation Service for external SQL databases."""

import uuid
from typing import AsyncGenerator

import httpx
import sqlparse
from magentic import prompt
from openbb_ai.models import StatusUpdateSSE, StatusUpdateSSEData
from pydantic import BaseModel, Field

from openbb_ada.errors import CodeGenerationError
from openbb_ada.models import (
    CodeGenerationResponse,
    CopilotArtifact,
    RawObjectDataFormat,
    SourceInfo,
    SqlQueryFunctionCallResult,
    SqlQueryGenerationResult,
    SqlWidgetContext,
)

from ..constants import OPENBB_AGENT_MODEL_MAIN
from ..utils.ai import get_llm
from ..utils.utils import retry_on_exception, sanitize_str
from ._logging import LoggingService
from .template import TemplateService


class SqlGenerationResponse(BaseModel):
    """Structured response from SQL generation LLM."""

    success: bool = Field(description="Whether SQL query was successfully generated")
    sql_query: str | None = Field(
        default=None,
        description="The generated SQL query if successful, None otherwise",
    )
    error_message: str | None = Field(
        default=None,
        description="Error explanation if query could not be generated",
    )


class SqlQueryGenerationService:
    """Generate SQL queries from natural language for external SQL databases."""

    def __init__(
        self,
        template_service: TemplateService,
        logging_service: LoggingService,
        openai_api_key: str | None,
    ):
        self._template_service = template_service
        self._logging_service = logging_service
        self._openai_api_key = openai_api_key

    async def generate_query(
        self,
        user_request: str,
        widget_uuid: str,
        sql_widget_dict: SqlWidgetContext,
        generate_query_only: bool = True,
    ) -> AsyncGenerator[str | StatusUpdateSSE, None]:
        """Generate SQL query from natural language request.

        Parameters
        ----------
        user_request: str
            Natural language request from user
        widget_uuid: str
            UUID of the SQL widget requesting the query generation
        sql_widget_dict: SqlWidgetContext
            Context of the SQL widget including schema and current SQL
        generate_query_only: bool
            If True, only generate SQL query without executing it
        """
        widget_id = sql_widget_dict.widget_id
        widget_origin = sql_widget_dict.widget_origin
        sql_schema = sql_widget_dict.sql_schema
        current_sql = sql_widget_dict.current_sql

        system_prompt = self._template_service.render_sql_query_generation_prompt(
            sql_schema=sql_schema or {},
            current_sql=current_sql,
        )

        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @prompt(
            sanitize_str(system_prompt) + "\n\nUser request: {user_request}",
            model=get_llm(
                model=OPENBB_AGENT_MODEL_MAIN,
                temperature=0.1,
                api_key=self._openai_api_key,
                max_completion_tokens=2048,
            ),
        )
        async def _generate_sql(user_request: str) -> SqlGenerationResponse: ...  # type: ignore[empty-body]

        try:
            response = await _generate_sql(user_request=user_request)
        except Exception as e:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Failed to generate SQL query",
                    details=[str(e)],
                )
            )
            return

        # Check if LLM successfully generated SQL
        if not response.success or not response.sql_query:
            error_detail = (
                response.error_message
                or "The requested data may not be available in the schema."
            )
            self._logging_service.warning("SQL generation failed: %s", error_detail)
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Could not generate SQL query",
                    details=[error_detail],
                )
            )
            return

        generated_sql = response.sql_query.strip()

        query_uuid = uuid.uuid4()
        query_artifact_id = f"query_artifact_{str(query_uuid)[:8]}"
        query_data_source = {
            "origin": widget_origin,
            "id": widget_id,
            "widget_uuid": widget_uuid,
        }

        query_artifact = CopilotArtifact(
            data_format=RawObjectDataFormat(
                parse_as="snowflake_query",
                query_data_source=query_data_source,
            ),
            content=self._format_sql_query(generated_sql),
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
                message="SQL query generated",
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
            widget_uuid=widget_uuid,
            widget_id=widget_id,
            widget_origin=widget_origin,
            artifacts=[query_artifact],
        )

    async def generate_sql_suggestion(
        self,
        user_request: str,
        sql_schema: dict | None = None,
        current_sql: str | None = None,
        data_sample: list[dict] | None = None,
    ) -> CodeGenerationResponse:
        """Generate SQL suggestion from natural language for widget code generation.

        Parameters
        ----------
        user_request: str
            Natural language request from user
        sql_schema: dict | None
            SQL schema context (tables, columns, etc.)
        current_sql: str | None
            Current SQL in the editor for modification context
        data_sample: list[dict] | None
            Sample rows from previous widget execution results
        """
        system_prompt = self._template_service.render_sql_query_generation_prompt(
            sql_schema=sql_schema or {},
            current_sql=current_sql,
            data_sample=data_sample,
        )

        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @prompt(
            sanitize_str(system_prompt) + "\n\nUser request: {user_request}",
            model=get_llm(
                model=OPENBB_AGENT_MODEL_MAIN,
                temperature=0.1,
                api_key=self._openai_api_key,
                max_completion_tokens=2048,
            ),
        )
        async def _generate_sql(user_request: str) -> SqlGenerationResponse: ...  # type: ignore[empty-body]

        try:
            response = await _generate_sql(user_request=user_request)
            if response.success and response.sql_query:
                formatted_sql = self._format_sql_query(
                    response.sql_query.strip(), wrap_markdown=False
                )
                return CodeGenerationResponse(generated_code=formatted_sql)
            raise CodeGenerationError(
                status_code=422,
                message=response.error_message or "Failed to generate SQL.",
            )
        except CodeGenerationError:
            raise
        except Exception as e:
            self._logging_service.error("SQL suggestion generation failed: %s", str(e))
            raise CodeGenerationError(status_code=422, message=str(e)) from e

    @staticmethod
    def _format_sql_query(sql_query: str, wrap_markdown: bool = True) -> str:
        """Format SQL query with proper indentation and keyword casing.

        Args:
            sql_query: Raw SQL query string
            wrap_markdown: Whether to wrap in markdown code block

        Returns:
            Formatted SQL query, optionally wrapped in markdown code block
        """
        try:
            formatted_query = sqlparse.format(
                sql_query,
                reindent=True,
                keyword_case="upper",
                identifier_case="preserve",
                strip_comments=False,
                use_space_around_operators=True,
                indent_width=2,
            )
        except Exception:
            formatted_query = sql_query

        if wrap_markdown:
            return f"```sql\n{formatted_query}\n```"
        return formatted_query
