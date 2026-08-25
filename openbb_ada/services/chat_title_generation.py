from typing import Annotated

import httpx
from fastapi import Depends
from magentic import (
    prompt,
)
from openbb_ai.models import (
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
)

from ..constants import OPENBB_AGENT_MODEL_SMALL
from ..utils.ai import get_llm
from ..utils.utils import (
    retry_on_exception,
    sanitize_str,
)
from .template import TemplateService


class ChatTitleGenerationService:
    """Provide functionality to generate chat title based on messages."""

    def __init__(
        self,
        template_service: Annotated[TemplateService, Depends(TemplateService)],
        openai_api_key: str | None,
    ):
        self._template_service = template_service
        self._openai_api_key = openai_api_key

    def _get_model(self, **kwargs):
        return get_llm(
            model=OPENBB_AGENT_MODEL_SMALL,
            temperature=0.1,
            api_key=self._openai_api_key,
            **kwargs,
        )

    async def generate_chat_title(
        self,
        messages: list[LlmClientMessage | LlmClientFunctionCallResultMessage],
    ) -> str:
        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @prompt(
            sanitize_str(
                self._template_service.render_generate_chat_title_prompt(
                    messages=messages,
                )
            ),
            model=self._get_model(max_completion_tokens=256),
        )
        async def _generate_chat_title() -> str: ...  # type: ignore

        result = await _generate_chat_title()
        return result
