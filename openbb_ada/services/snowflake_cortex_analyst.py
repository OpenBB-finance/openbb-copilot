"""Snowflake Cortex Analyst service for SQL and Python widget helper endpoint."""

import json
from typing import Any

import httpx
from openbb_ai.models import StatusUpdateSSE

from .. import constants
from ..errors import CodeGenerationError
from ..models import (
    CodeGenerationRequest,
    CodeGenerationResponse,
    PythonCodeGenerationResult,
    PythonWidgetContext,
    SqlQueryGenerationResult,
    SqlWidgetContext,
)
from ..utils.auth import (
    get_login_token,
    get_snowflake_host,
    is_running_in_spcs,
)
from ._logging import LoggingService
from .python_code_generation import PythonCodeGenerationService
from .sql_query_generation import SqlQueryGenerationService


class CortexAnalystClient:
    """Call Snowflake Cortex Analyst REST API and extract generated SQL."""

    def __init__(
        self, logging_service: LoggingService, ingress_user_token: str | None = None
    ):
        self._logging_service = logging_service
        # The ingress user token is threaded in from the FastAPI request so that
        # we can flip Cortex Analyst back to caller's rights with a one-line
        # change in `_get_auth_headers` (replace `None` with
        # `self._ingress_user_token`). It is intentionally unused today —
        # see the docstring of `_get_auth_headers` for why.
        self._ingress_user_token = ingress_user_token

    def _get_auth_headers(self) -> dict[str, str]:
        """Owner's-rights auth headers for the Cortex Analyst REST API.

        We deliberately pass `None` to `get_login_token` instead of
        `self._ingress_user_token` so the call uses **only** the SPCS service
        token — i.e. owner's rights as the application identity.

        Concatenating the ingress user token (`service.user`) would put the
        session in Snowflake's "restricted caller's rights context", where:
          * only `CALLER` / `INHERITED CALLER` grants are visible (plain
            `GRANT … TO APPLICATION` is invisible), and
          * `CALLER` grants cannot legally be applied to objects in IMPORTED
            DATABASEs (Marketplace shares).
        Together that makes share-backed semantic views unreachable through
        Cortex Analyst.

        The plumbing for `ingress_user_token` is kept on the constructor and
        the factory so we can flip back to caller's rights here without
        touching the call sites — useful if Snowflake later relaxes the
        restricted-mode constraints on shares, or if we add a per-user
        authorization check in front of the Cortex Analyst call.
        """
        token = get_login_token(None)
        token_type = "OAUTH" if is_running_in_spcs() else "PROGRAMMATIC_ACCESS_TOKEN"
        return {
            "Authorization": f"Bearer {token}",
            "X-Snowflake-Authorization-Token-Type": token_type,
        }

    # ------------------------------------------------------------------
    # SQL generation via Cortex Analyst REST API
    # ------------------------------------------------------------------

    async def generate_sql(
        self,
        request: CodeGenerationRequest,
    ) -> tuple[str, list[str]]:
        current_prompt = request.user_prompt
        last_error: str | None = None

        for attempt in range(1 + constants.MAX_CORTEX_ANALYST_RETRIES):
            sql, warnings, error, suggestions = await self._call_cortex_analyst(
                request, current_prompt
            )

            if sql:
                if attempt > 0:
                    self._logging_service.info(
                        "Cortex Analyst succeeded after reformulation (attempt %d)",
                        attempt + 1,
                    )
                return sql, warnings

            # Cortex returned feedback instead of SQL — build a reformulated prompt
            feedback = error or "Cortex Analyst could not generate SQL."
            last_error = feedback

            if attempt < constants.MAX_CORTEX_ANALYST_RETRIES:
                current_prompt = self._reformulate_prompt(
                    original_prompt=request.user_prompt,
                    cortex_feedback=feedback,
                    suggestions=suggestions,
                )
                self._logging_service.info(
                    "Cortex Analyst returned feedback, reformulating prompt "
                    "(attempt %d/%d): %s",
                    attempt + 1,
                    constants.MAX_CORTEX_ANALYST_RETRIES,
                    feedback[:200],
                )

        raise CodeGenerationError(
            status_code=502,
            message=last_error or "Cortex Analyst returned no SQL statement.",
        )

    async def _call_cortex_analyst(
        self,
        request: CodeGenerationRequest,
        user_prompt: str,
    ) -> tuple[str | None, list[str], str | None, list[str]]:
        """Call Cortex Analyst and return (sql, warnings, error, suggestions)."""
        modified_request = request.model_copy(update={"user_prompt": user_prompt})
        payload = self._build_payload(modified_request)
        host = get_snowflake_host()
        url = f"https://{host}/{constants.CORTEX_ANALYST_API_ENDPOINT}"
        headers = {
            "Content-Type": "application/json",
            **self._get_auth_headers(),
        }

        async with httpx.AsyncClient(
            timeout=constants.CORTEX_ANALYST_REQUEST_TIMEOUT
        ) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            detail = response.text[:500]
            raise CodeGenerationError(
                status_code=response.status_code,
                message=f"Cortex Analyst returned {response.status_code}: {detail}",
            )

        try:
            response_data = response.json()
        except json.JSONDecodeError as exc:
            raise CodeGenerationError(
                status_code=502,
                message=f"Cortex Analyst returned malformed JSON: {exc}",
            ) from exc

        self._logging_service.debug("Cortex Analyst raw response: %s", response_data)
        message = response_data.get("message", {})
        if isinstance(message, str):
            return message, [], None, []

        content = message.get("content", [])
        warnings = [
            warning.get("message", "")
            for warning in response_data.get("warnings", [])
            if isinstance(warning, dict) and warning.get("message")
        ]

        for block in content:
            if not isinstance(block, dict) or block.get("type") != "sql":
                continue
            statement = block.get("statement")
            if statement is None:
                continue
            sql_statement = str(statement).strip()
            if sql_statement:
                return sql_statement, warnings, None, []

        # No SQL — extract text explanation and suggestions as feedback
        text_parts: list[str] = []
        suggestions: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    text_parts.append(block["text"])
                elif block.get("type") == "suggestions":
                    suggestions = block.get("suggestions", []) or []

        error_msg = text_parts[0] if text_parts else None
        return None, [], error_msg, suggestions

    @staticmethod
    def _reformulate_prompt(
        original_prompt: str,
        cortex_feedback: str,
        suggestions: list[str],
    ) -> str:
        parts = [
            f"Original request: {original_prompt}",
            f"\nThe semantic model returned this feedback: {cortex_feedback}",
        ]
        if suggestions:
            parts.append(
                "\nSuggested reformulations:\n"
                + "\n".join(f"- {s}" for s in suggestions[:3])
            )
        parts.append(
            "\nPlease use the column names and dimensions referenced in the "
            "feedback above to answer the original request."
        )
        return "\n".join(parts)

    @staticmethod
    def _build_payload(request: CodeGenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": request.user_prompt}],
                }
            ],
            "stream": False,
        }

        # Semantic configuration must already be attached to the request.
        if request.semantic_models:
            payload["semantic_models"] = [
                ref.model_dump(exclude_none=True) for ref in request.semantic_models
            ]
        elif request.semantic_model:
            payload["semantic_model"] = request.semantic_model
        elif request.semantic_model_file:
            payload["semantic_model_file"] = request.semantic_model_file
        elif request.semantic_view:
            payload["semantic_view"] = request.semantic_view

        return payload


class SnowflakeCortexAnalystService:
    """Generate SQL/Python code with Cortex Analyst + LLM fallback for SQL."""

    def __init__(
        self,
        logging_service: LoggingService,
        sql_query_generation_service: SqlQueryGenerationService | None,
        python_code_generation_service: PythonCodeGenerationService | None,
        cortex_analyst_client: CortexAnalystClient | None,
    ):
        self._logging_service = logging_service
        self._sql_query_generation_service = sql_query_generation_service
        self._python_code_generation_service = python_code_generation_service
        self._cortex_analyst_client = cortex_analyst_client

    async def generate_code(
        self, request: CodeGenerationRequest
    ) -> CodeGenerationResponse:
        if request.language == "python":
            return await self._generate_python_code(request)
        if request.language != "sql":
            raise CodeGenerationError(
                status_code=503,
                message="Only SQL and Python code generation are available in Snowflake mode.",  # noqa: E501
            )
        return await self._generate_sql_code(request)

    async def _generate_sql_code(
        self, request: CodeGenerationRequest
    ) -> CodeGenerationResponse:
        # Try Cortex Analyst only when semantic configuration is already attached
        # to the request by upstream selection or Ada-side candidate matching.
        should_try_cortex = (
            self._cortex_analyst_client is not None and request.has_semantic_config
        )

        # Try Cortex Analyst, fall back to LLM on failure.
        if should_try_cortex:
            assert self._cortex_analyst_client is not None
            try:
                (
                    generated_sql,
                    cortex_warnings,
                ) = await self._cortex_analyst_client.generate_sql(request)
                for w in cortex_warnings:
                    self._logging_service.warning("Cortex Analyst warning: %s", w)
                return CodeGenerationResponse(
                    generated_code=generated_sql,
                    generation_source="cortex_analyst",
                )
            except CodeGenerationError as exc:
                self._logging_service.warning(
                    "Cortex Analyst failed, falling back to LLM: %s", exc
                )

        # 4. LLM fallback
        if self._sql_query_generation_service is None:
            raise CodeGenerationError(
                status_code=503,
                message="SQL generation service is not available.",
            )

        widget_context = SqlWidgetContext(
            widget_uuid=request.widget_uuid,
            widget_id=request.widget_uuid,
            widget_name="SQL Widget",
            widget_origin="OpenBB Workspace",
            widget_description="SQL widget context for direct code generation endpoint",
            sql_schema=request.sql_schema,
            current_sql=request.current_code,
        )
        llm_sql, error_message = await self._generate_sql_with_llm(
            request=request,
            widget_context=widget_context,
        )
        if error_message:
            raise CodeGenerationError(status_code=422, message=error_message)

        return CodeGenerationResponse(
            generated_code=llm_sql,
            generation_source="llm",
        )

    async def _generate_python_code(
        self, request: CodeGenerationRequest
    ) -> CodeGenerationResponse:
        if self._python_code_generation_service is None:
            raise CodeGenerationError(
                status_code=503,
                message="Python code generation service is not available.",
            )

        python_context = PythonWidgetContext(
            widget_uuid=request.widget_uuid,
            widget_id=request.widget_uuid,
            widget_name="Python Widget",
            widget_origin="OpenBB Workspace",
            widget_description=(
                "Python widget context for direct code generation endpoint"
            ),
            current_code=request.current_code,
        )

        sql_widgets = (
            [
                SqlWidgetContext(
                    widget_uuid=request.widget_uuid,
                    widget_id=request.widget_uuid,
                    widget_name="SQL Context Widget",
                    widget_origin="OpenBB Workspace",
                    widget_description=(
                        "SQL context attached to python code generation endpoint"
                    ),
                    sql_schema=request.sql_schema,
                    current_sql=None,
                )
            ]
            if request.sql_schema
            else None
        )
        generated_python, error_message = await self._generate_python_with_llm(
            request=request,
            python_context=python_context,
            sql_widgets=sql_widgets,
        )
        if error_message:
            raise CodeGenerationError(status_code=422, message=error_message)

        return CodeGenerationResponse(
            generated_code=generated_python,
            generation_source="llm",
        )

    async def _generate_sql_with_llm(
        self,
        request: CodeGenerationRequest,
        widget_context: SqlWidgetContext,
    ) -> tuple[str | None, str | None]:
        if self._sql_query_generation_service is None:
            return None, "SQL generation service is not available."

        events = self._sql_query_generation_service.generate_query(
            user_request=request.user_prompt,
            widget_uuid=request.widget_uuid,
            sql_widget_dict=widget_context,
            generate_query_only=True,
        )
        try:
            async for event in events:
                if isinstance(event, SqlQueryGenerationResult):
                    return event.sql_query, None
                if (
                    isinstance(event, StatusUpdateSSE)
                    and event.data.eventType == "ERROR"
                ):
                    return None, self._extract_status_error(event)
        finally:
            await events.aclose()

        return None, "SQL generation completed without a SQL result."

    async def _generate_python_with_llm(
        self,
        request: CodeGenerationRequest,
        python_context: PythonWidgetContext,
        sql_widgets: list[SqlWidgetContext] | None = None,
    ) -> tuple[str | None, str | None]:
        if self._python_code_generation_service is None:
            return None, "Python code generation service is not available."

        events = self._python_code_generation_service.generate_code(
            user_request=request.user_prompt,
            widget_uuid=request.widget_uuid,
            python_widget=python_context,
            generate_code_only=True,
            sql_widgets=sql_widgets,
        )
        try:
            async for event in events:
                if isinstance(event, PythonCodeGenerationResult):
                    return event.python_code, None
                if (
                    isinstance(event, StatusUpdateSSE)
                    and event.data.eventType == "ERROR"
                ):
                    return None, self._extract_status_error(event)
        finally:
            await events.aclose()

        return None, "Python code generation completed without a code result."

    @staticmethod
    def _extract_status_error(event: StatusUpdateSSE) -> str:
        details = event.data.details or []
        if details:
            if isinstance(details[0], str):
                return details[0]
            return str(details[0])
        return event.data.message or "Code generation failed."
