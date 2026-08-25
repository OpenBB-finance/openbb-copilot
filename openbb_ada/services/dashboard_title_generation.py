from typing import Annotated, Any

import httpx
from fastapi import Depends
from magentic import prompt

from ..constants import OPENBB_AGENT_MODEL_SMALL
from ..utils.ai import get_llm
from ..utils.utils import retry_on_exception, sanitize_str
from ._logging import logfire
from .template import TemplateService


class DashboardTitleGenerationService:
    """Generate dashboard titles based on the widgets present on the dashboard."""

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
            temperature=0.7,
            api_key=self._openai_api_key,
            **kwargs,
        )

    @logfire.instrument("DashboardTitleGenerationService.generate_dashboard_title")
    async def generate_dashboard_title(
        self,
        widgets: list[Any],
    ) -> str:
        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @prompt(
            sanitize_str(
                self._template_service.render_generate_dashboard_title_prompt(
                    widgets=widgets,
                )
            ),
            model=self._get_model(max_completion_tokens=256),
        )
        async def _generate_dashboard_title() -> str: ...  # type: ignore

        result = await _generate_dashboard_title()
        return result
