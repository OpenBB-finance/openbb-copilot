"""Utility for converting text to structured data."""

import json
from typing import Any

import httpx
from magentic import SystemMessage, UserMessage, chatprompt

from ..services.template import TemplateService
from .ai import get_llm
from .utils import retry_on_exception


async def extract_table_data(
    text_data: str,
    template_service: TemplateService,
) -> list[dict[str, Any]]:
    """Extract structured data from text using LLM analysis.

    Args:
        text_data: The unstructured text containing data
        template_service: Service for rendering templates

    Returns:
        List of dictionaries containing extracted data
    """
    # Render the extraction prompt
    prompt = template_service.render_template(
        "data_extraction_template.jinja",
        {"text_data": text_data},
    )

    @retry_on_exception(max_retries=2, exceptions=(httpx.RemoteProtocolError,))
    @chatprompt(
        SystemMessage("Extract data and return only valid JSON array."),
        UserMessage(prompt),
        model=get_llm(max_completion_tokens=2048, temperature=0),
    )
    async def call_llm() -> str: ...  # type: ignore

    response = await call_llm()

    try:
        table_records = json.loads(response)
        if not isinstance(table_records, list):
            table_records = [table_records]  # Wrap single object in array
        return table_records
    except json.JSONDecodeError as e:
        # Raise a clear error
        raise ValueError(
            "Failed to extract table data. The AI could not parse the text "
            "into structured format. Please try rephrasing or formatting your "
            "data more clearly."
        ) from e
