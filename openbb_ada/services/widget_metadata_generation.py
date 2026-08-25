import json
from typing import Annotated

import httpx
import pandas as pd
from fastapi import Depends, UploadFile
from magentic import (
    SystemMessage,
    chatprompt,
    prompt,
)
from magentic.vision import UserImageMessage

from ..constants import OPENBB_AGENT_MODEL_SMALL
from ..models import (
    WidgetTitleDescriptionRequest,
    WidgetTitleDescriptionResponse,
)
from ..utils.ai import get_llm
from ..utils.utils import (
    retry_on_exception,
)
from . import TemplateService


class WidgetMetadataGenerationService:
    """Provide functionality for generating widget metadata."""

    def __init__(
        self,
        template_service: Annotated[TemplateService, Depends(TemplateService)],
        openai_api_key: str | None = None,
    ):
        self._template_service = template_service
        self._openai_api_key = openai_api_key

    def _get_model(self, api_key: str | None = None, **kwargs):
        return get_llm(
            model=OPENBB_AGENT_MODEL_SMALL,
            temperature=0.1,
            api_key=api_key or self._openai_api_key,
            **kwargs,
        )

    async def generate_title_and_description(
        self, widget_generation_request: WidgetTitleDescriptionRequest
    ) -> WidgetTitleDescriptionResponse:
        # Strip metadata fields that bias the LLM (pre-existing answers)
        # or are irrelevant noise (infrastructure details, redundant params).
        # Keep only fields useful for generation (e.g. query, type).
        if widget_generation_request.metadata:
            keys_to_strip = {
                "name",
                "description",
                "category",
                "subCategory",
                "endpoint",
                "params",
            }
            widget_generation_request.metadata = {
                k: v
                for k, v in widget_generation_request.metadata.items()
                if k not in keys_to_strip
            }

        try:
            widget_data = pd.DataFrame(
                json.loads(widget_generation_request.widget_data)
            )
            # TODO: It's probably a little ugly to update the
            # widget_generation_request object in-place.
            widget_generation_request.widget_data = widget_data.head(20).to_json(
                orient="records", date_format="iso", lines=True
            )
        # In case we can't parse as structured data, we treat it as a string.
        except (ValueError, TypeError, json.JSONDecodeError):
            widget_generation_request.widget_data = (
                widget_generation_request.widget_data[:1024]
            )

        rendered_prompt = (
            self._template_service.render_generate_widget_title_and_description_prompt(
                widget_generation_request=widget_generation_request,
            )
        )

        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @prompt(
            rendered_prompt,
            model=self._get_model(
                api_key=widget_generation_request.openai_api_key,
            ),
        )
        async def _generate_title_and_description() -> (  # type: ignore[empty-body]
            WidgetTitleDescriptionResponse
        ): ...

        response = await _generate_title_and_description()
        return response

    async def generate_title_and_description_for_image(
        self, file: UploadFile
    ) -> WidgetTitleDescriptionResponse:
        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @chatprompt(
            SystemMessage(
                self._template_service.render_generate_widget_title_and_description_prompt(
                    widget_generation_request=WidgetTitleDescriptionRequest(
                        widget_data="The widget data is in the image.",
                        name=file.filename,
                    )
                )
            ),
            UserImageMessage(await file.read()),
            model=self._get_model(max_completion_tokens=2048),
        )
        async def _query_image() -> WidgetTitleDescriptionResponse: ...  # type: ignore

        response = await _query_image()
        return response
