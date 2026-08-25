"""Python Code Generation Service for Snowflake Snowpark execution."""

import ast
import re
import uuid
from typing import AsyncGenerator

import httpx
from magentic import prompt
from openbb_ai.models import (
    RawObjectDataFormat,
    SourceInfo,
    StatusUpdateSSE,
    StatusUpdateSSEData,
)

from ..constants import OPENBB_AGENT_MODEL_MAIN
from ..errors import CodeGenerationError
from ..models import (
    CodeGenerationResponse,
    CopilotArtifact,
    PythonCodeFunctionCallResult,
    PythonCodeGenerationResult,
    PythonWidgetContext,
    SqlWidgetContext,
)
from ..utils.ai import get_llm
from ..utils.utils import retry_on_exception, sanitize_str
from ._logging import LoggingService
from .template import TemplateService


class PythonCodeGenerationService:
    """Generate Python code from natural language for Snowflake Snowpark execution."""

    def __init__(
        self,
        template_service: TemplateService,
        logging_service: LoggingService,
        openai_api_key: str | None,
    ):
        self._template_service = template_service
        self._logging_service = logging_service
        self._openai_api_key = openai_api_key

    async def generate_code(
        self,
        user_request: str,
        widget_uuid: str,
        python_widget: PythonWidgetContext,
        generate_code_only: bool = True,
        sql_widgets: list[SqlWidgetContext] | None = None,
    ) -> AsyncGenerator[
        StatusUpdateSSE | PythonCodeGenerationResult | PythonCodeFunctionCallResult,
        None,
    ]:
        """Generate Python code from natural language request.


        Parameters
        ----------
        user_request: str
            Natural language request from user
        widget_uuid: str
            UUID of the target widget
        python_widget: PythonWidgetContext
            Context of the target Python widget
        generate_code_only: bool
            If True, only generate code without executing it
        sql_widgets: list[SqlWidgetContext] | None
            List of SQL widget contexts available (optional)
        """
        system_prompt = self._template_service.render_python_code_generation_prompt(
            current_code=python_widget.current_code,
            sql_widgets=sql_widgets,
        )

        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @prompt(
            sanitize_str(system_prompt) + "\n\nUser request: {user_request}",
            model=get_llm(
                model=OPENBB_AGENT_MODEL_MAIN,
                temperature=0.1,
                api_key=self._openai_api_key,
                max_completion_tokens=4096,
            ),
        )
        async def _generate_python(user_request: str) -> str:  # type: ignore[empty-body]
            ...

        try:
            generated_code = await _generate_python(user_request=user_request)
            generated_code = self._sanitize_code(generated_code)
        except Exception as e:
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="ERROR",
                    message="Failed to generate Python code",
                    details=[str(e)],
                )
            )
            return

        # Validate syntax (warn but proceed - Snowpark handles runtime errors)
        syntax_valid = self._validate_syntax(generated_code)
        if not syntax_valid:
            self._logging_service.warning(
                "Generated Python code has syntax issues, proceeding anyway"
            )

        self._logging_service.info(
            "Python code generated (%d chars)", len(generated_code)
        )

        # Create artifact for frontend display
        code_uuid = uuid.uuid4()
        code_artifact_id = f"python_artifact_{str(code_uuid)[:8]}"
        query_data_source = {
            "origin": python_widget.widget_origin,
            "id": python_widget.widget_id,
            "widget_uuid": widget_uuid,
        }

        code_artifact = CopilotArtifact(
            data_format=RawObjectDataFormat(
                parse_as="snowflake_python",
                query_data_source=query_data_source,
            ),
            content=self._format_python_code(generated_code),
            source_info=SourceInfo(
                type="artifact",
                uuid=code_uuid,
                name=code_artifact_id,
                description="Python code generated for Snowflake Snowpark",
                metadata={
                    "parse_as": "snowflake_python",
                    "query_data_source": query_data_source,
                },
                citable=False,
            ),
        )

        code_client_artifact = code_artifact.to_client_artifact()
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message="Python code generated",
                artifacts=[code_client_artifact] if not generate_code_only else [],
            )
        )

        result_model = (
            PythonCodeGenerationResult
            if generate_code_only
            else PythonCodeFunctionCallResult
        )

        yield result_model(
            python_code=generated_code,
            widget_uuid=widget_uuid,
            widget_id=python_widget.widget_id,
            widget_origin=python_widget.widget_origin,
            artifacts=[code_artifact],
        )

    async def generate_python_suggestion(
        self,
        user_request: str,
        current_code: str | None = None,
    ) -> CodeGenerationResponse:
        """Generate Python suggestion from natural language for widget code generation.

        Parameters
        ----------
        user_request: str
            Natural language request from user
        current_code: str | None
            Current Python code in the editor for modification context
        """
        system_prompt = self._template_service.render_python_code_generation_prompt(
            current_code=current_code,
            sql_widgets=None,
        )

        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @prompt(
            sanitize_str(system_prompt) + "\n\nUser request: {user_request}",
            model=get_llm(
                model=OPENBB_AGENT_MODEL_MAIN,
                temperature=0.1,
                api_key=self._openai_api_key,
                max_completion_tokens=4096,
            ),
        )
        async def _generate_python(user_request: str) -> CodeGenerationResponse: ...  # type: ignore[empty-body]

        try:
            response = await _generate_python(user_request=user_request)
            if response.generated_code:
                generated_code = self._sanitize_code(response.generated_code)
                return CodeGenerationResponse(generated_code=generated_code)
            raise CodeGenerationError(
                status_code=422,
                message="Failed to generate Python code.",
            )
        except CodeGenerationError:
            raise
        except Exception as e:
            self._logging_service.error(
                "Python suggestion generation failed: %s", str(e)
            )
            raise CodeGenerationError(status_code=422, message=str(e)) from e

    @staticmethod
    def _format_python_code(code: str) -> str:
        """Format Python code for display in markdown.

        Args:
            code: Raw Python code string

        Returns:
            Python code wrapped in markdown code block
        """
        return f"```python\n{code}\n```"

    @staticmethod
    def _sanitize_code(code: str) -> str:
        """Sanitize generated Python code by removing markdown artifacts.

        The LLM sometimes returns code wrapped in markdown code blocks
        despite being instructed not to. This method strips those artifacts.

        Args:
            code: Raw code string from LLM

        Returns:
            Cleaned Python code
        """
        code = code.strip()

        # Remove markdown code fences (```python or ```)
        # Pattern handles: ```python\n...```, ```\n...```, etc.
        fence_pattern = r"^```(?:python)?\s*\n?(.*?)\n?```$"
        match = re.match(fence_pattern, code, re.DOTALL)
        if match:
            code = match.group(1).strip()

        return code

    def _validate_syntax(self, code: str) -> bool:
        """Validate that the code is syntactically correct Python.

        Args:
            code: Python code to validate

        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            self._logging_service.warning(
                "Syntax error in generated code at line %s: %s",
                e.lineno,
                e.msg,
            )
            return False
