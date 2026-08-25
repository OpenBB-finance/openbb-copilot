from typing import Annotated, Any

import httpx
from fastapi import Depends
from magentic import prompt
from openbb_ai.models import (
    AgentTool,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    RawContext,
    WidgetCollection,
)

from ..constants import OPENBB_AGENT_MODEL_SMALL
from ..utils.ai import get_llm
from ..utils.utils import retry_on_exception, sanitize_str
from .template import TemplateService


class PromptEnhancementService:
    """Provide functionality to enhance user prompts for better AI comprehension."""

    def __init__(
        self,
        template_service: Annotated[TemplateService, Depends(TemplateService)],
        openai_api_key: str | None,
    ):
        self._template_service = template_service
        self._openai_api_key = openai_api_key

        # Store context for prompt enhancement
        self._current_messages: (
            list[LlmClientMessage | LlmClientFunctionCallResultMessage] | None
        ) = None
        self._current_context: list[Any] | None = None
        self._current_widgets: Any | None = None
        self._current_tools: list[Any] | None = None

    def set_current_context(
        self,
        messages: (
            list[LlmClientMessage | LlmClientFunctionCallResultMessage] | None
        ) = None,
        context: list | None = None,
        widgets: Any | None = None,
        tools: list | None = None,
    ):
        """Set the current context for prompt enhancement."""
        self._current_messages = messages
        self._current_context = context
        self._current_widgets = widgets
        self._current_tools = tools

    def get_current_context(
        self,
    ) -> tuple[
        list[LlmClientMessage | LlmClientFunctionCallResultMessage] | None,
        list[RawContext] | None,
        WidgetCollection | None,
        list[AgentTool] | None,
    ]:
        """Return the cached context for prompt enhancement."""
        return (
            self._current_messages,
            self._current_context,
            self._current_widgets,
            self._current_tools,
        )

    def _get_model(self, **kwargs):
        return get_llm(
            model=OPENBB_AGENT_MODEL_SMALL,
            temperature=0.3,
            api_key=self._openai_api_key,
            **kwargs,
        )

    async def enhance_prompt(
        self,
        messages: list[LlmClientMessage | LlmClientFunctionCallResultMessage],
        context: list[RawContext] | None = None,
        widgets: WidgetCollection | None = None,
        tools: list[AgentTool] | None = None,
    ) -> str:
        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @prompt(
            sanitize_str(
                self._template_service.render_prompt_enhancement_prompt(
                    messages=messages,
                    context=context,
                    widgets=widgets,
                    tools=tools,
                )
            ),
            model=self._get_model(max_completion_tokens=2048),
        )
        async def _enhance_prompt() -> str: ...  # type: ignore

        result = await _enhance_prompt()
        return result
