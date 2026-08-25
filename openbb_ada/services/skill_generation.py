import logging
import re
from typing import Annotated

import httpx
from fastapi import Depends
from magentic import OpenaiChatModel, prompt

from ..constants import OPENBB_AGENT_MODEL_SMALL
from ..models import (
    SkillConversationMessage,
    SkillGenerationRequest,
    SkillGenerationResponse,
)
from ..utils.ai import context_char_budget, get_llm
from ..utils.utils import (
    retry_on_exception,
)
from .template import TemplateService

logger = logging.getLogger(__name__)

# Tokens reserved out of the model context for the prompt template, user
# hints, existing slugs, and the generated skill output.
PROMPT_RESERVED_TOKENS = 12_000
# A single message cannot consume the whole budget; generous ceiling for
# data-heavy messages (tables, reports pasted into the chat).
MAX_MESSAGE_CONTENT_LENGTH = 50_000

SLUG_REGEX = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def slugify(value: str) -> str:
    """Normalize a string into a valid skill slug (mirrors the frontend rules)."""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s-]", "", value)
    value = re.sub(r"[\s_]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value[:50].strip("-")


def fit_conversation_to_budget(
    conversation: list[SkillConversationMessage],
    budget_chars: int,
) -> list[SkillConversationMessage]:
    """Keep the newest whole messages that fit within the character budget.

    Each message is first capped at MAX_MESSAGE_CONTENT_LENGTH. Once a message
    no longer fits, it and everything older are dropped so the kept history
    stays contiguous. The newest message is always kept, truncated to the
    budget if necessary.
    """
    kept: list[SkillConversationMessage] = []
    used = 0
    for message in reversed(conversation):
        content = message.content[:MAX_MESSAGE_CONTENT_LENGTH]
        remaining = budget_chars - used
        if len(content) > remaining:
            if kept:
                break
            content = content[:remaining]
        message.content = content
        kept.append(message)
        used += len(content)
    kept.reverse()
    if len(kept) < len(conversation):
        logger.warning(
            "Skill generation conversation truncated to fit the model context:"
            " kept %d of %d messages (%d chars, budget %d)",
            len(kept),
            len(conversation),
            used,
            budget_chars,
        )
    return kept


class SkillGenerationService:
    """Generate skill metadata and content from a conversation."""

    def __init__(
        self,
        template_service: Annotated[TemplateService, Depends(TemplateService)],
    ):
        self._template_service = template_service

    def _get_model(self, api_key: str | None = None, **kwargs) -> OpenaiChatModel:
        return get_llm(
            model=OPENBB_AGENT_MODEL_SMALL,
            temperature=0.2,
            api_key=api_key,
            **kwargs,
        )

    async def generate_skill(
        self, skill_generation_request: SkillGenerationRequest
    ) -> SkillGenerationResponse:
        """Generate a skill (slug, description, markdown content) from a
        conversation using the small model and the skill prompt template."""
        skill_generation_request.conversation = fit_conversation_to_budget(
            skill_generation_request.conversation,
            context_char_budget(
                OPENBB_AGENT_MODEL_SMALL, reserved_tokens=PROMPT_RESERVED_TOKENS
            ),
        )

        rendered_prompt = self._template_service.render_generate_skill_prompt(
            skill_generation_request=skill_generation_request,
        )

        @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
        @prompt(
            rendered_prompt,
            model=self._get_model(
                api_key=skill_generation_request.openai_api_key,
            ),
        )
        async def _generate_skill() -> (  # type: ignore[empty-body]
            SkillGenerationResponse
        ): ...

        response = await _generate_skill()

        # The frontend enforces slug uniqueness; here we only guarantee that
        # the slug is well-formed so the save cannot fail on validation.
        if not SLUG_REGEX.fullmatch(response.slug) or not 2 <= len(response.slug) <= 50:
            fallback = slugify(response.slug) or slugify(
                skill_generation_request.name_hint or ""
            )
            logger.warning(
                "Generated skill slug %r is invalid; falling back to %r",
                response.slug,
                fallback or "saved-skill",
            )
            response.slug = fallback if len(fallback) >= 2 else "saved-skill"
        return response
