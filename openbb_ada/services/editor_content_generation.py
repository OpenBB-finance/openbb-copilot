"""Editor content generation service."""

import re
from typing import Literal

import httpx
from magentic import prompt
from pydantic import BaseModel, Field

from ..constants import OPENBB_AGENT_MODEL_MAIN
from ..errors import CodeGenerationError
from ..models import CodeGenerationResponse
from ..utils.ai import get_llm
from ..utils.utils import retry_on_exception, sanitize_str
from ._logging import LoggingService

EditorContentLanguage = Literal["python", "text"]


class EditorContentGenerationResponse(BaseModel):
    """Structured response from editor-content generation."""

    generated_content: str | None = Field(default=None)
    error_message: str | None = Field(default=None)


class EditorContentGenerationService:
    """Generate content for widget editors."""

    def __init__(
        self,
        logging_service: LoggingService,
        openai_api_key: str | None,
    ):
        self._logging_service = logging_service
        self._openai_api_key = openai_api_key

    async def generate_content_suggestion(
        self,
        user_request: str,
        language: EditorContentLanguage,
        current_content: str | None,
    ) -> CodeGenerationResponse:
        system_prompt = self._build_generation_prompt(
            language=language,
            current_content=current_content,
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
            max_retries=3,
        )
        async def _generate_editor_content(
            user_request: str,
        ) -> EditorContentGenerationResponse:
            raise NotImplementedError

        try:
            response = await _generate_editor_content(user_request=user_request)
            if response.generated_content is not None:
                return CodeGenerationResponse(
                    generated_code=self._strip_wrapping_code_fence(
                        response.generated_content
                    ),
                    generation_source="llm",
                )
            raise CodeGenerationError(
                status_code=422,
                message=response.error_message or "Failed to generate content.",
            )
        except CodeGenerationError:
            raise
        except Exception as e:
            self._logging_service.error("Editor content generation failed: %s", str(e))
            raise CodeGenerationError(status_code=422, message=str(e)) from e

    @staticmethod
    def _build_generation_prompt(
        language: EditorContentLanguage,
        current_content: str | None = None,
    ) -> str:
        content_label = "Python code" if language == "python" else "plain text content"
        result_label = "code" if language == "python" else "content"
        prompt_parts = [
            f"You generate and edit {content_label} for an OpenBB Workspace omni widget editor.",  # noqa: E501
            "Follow the user's request directly.",
            (
                "If the request modifies existing content, return the full updated "
                f"{result_label}, not a diff, summary, or explanation."
            ),
            "Do not wrap the result in markdown code fences.",
            "Respond with a structured object containing generated_content.",
        ]

        if current_content:
            prompt_parts.extend(["", f"Current {content_label}:", current_content])

        return "\n".join(prompt_parts)

    @staticmethod
    def _strip_wrapping_code_fence(content: str) -> str:
        content = content.strip()
        match = re.match(r"^```[\w+-]*\s*\n?(.*?)\n?```$", content, re.DOTALL)
        if match:
            return match.group(1).strip()
        return content
