"""
Tests for SkillGenerationService.

Covers slug normalization (slugify / SLUG_REGEX), SkillGenerationRequest
validation, and generate_skill conversation truncation and slug-fallback
logic with the LLM call mocked.
"""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from openbb_ada.models import (
    SkillConversationMessage,
    SkillGenerationRequest,
    SkillGenerationResponse,
)
from openbb_ada.services import TemplateService
from openbb_ada.services.skill_generation import (
    MAX_MESSAGE_CONTENT_LENGTH,
    SLUG_REGEX,
    SkillGenerationService,
    fit_conversation_to_budget,
    slugify,
)


@pytest.fixture
def skill_generation_service() -> SkillGenerationService:
    """Create a SkillGenerationService instance for testing."""
    return SkillGenerationService(template_service=TemplateService())


def _request(**overrides) -> SkillGenerationRequest:
    defaults: dict = {
        "conversation": [
            SkillConversationMessage(role="human", content="Show me AAPL earnings"),
            SkillConversationMessage(role="ai", content="Here is the analysis."),
        ]
    }
    defaults.update(overrides)
    return SkillGenerationRequest(**defaults)


def _mock_llm(mock_prompt, response: SkillGenerationResponse) -> None:
    mock_prompt.return_value = lambda func: AsyncMock(return_value=response)


class TestSlugify:
    """Tests for the slugify helper."""

    def test_normalizes_basic_title(self):
        assert slugify("My Cool Skill") == "my-cool-skill"

    def test_removes_special_characters(self):
        assert slugify("Hello, World! @2026") == "hello-world-2026"

    def test_collapses_whitespace_and_hyphens(self):
        assert slugify("a  b--c") == "a-b-c"

    def test_strips_leading_and_trailing_hyphens(self):
        assert slugify("--abc--") == "abc"

    def test_truncates_then_strips_no_trailing_hyphen(self):
        """Truncation to 50 chars must not leave a trailing hyphen."""
        value = "x" * 49 + "-abc"
        result = slugify(value)
        assert result == "x" * 49
        assert not result.endswith("-")

    def test_empty_string_returns_empty(self):
        assert slugify("") == ""

    def test_symbols_only_returns_empty(self):
        assert slugify("!!!") == ""

    def test_non_empty_results_match_slug_regex(self):
        inputs = [
            "My Cool Skill",
            "Hello, World! @2026",
            "--abc--",
            "x" * 49 + "-abc",
            "snowflake_html report",
        ]
        for value in inputs:
            result = slugify(value)
            assert SLUG_REGEX.fullmatch(result), f"{value!r} -> {result!r}"


class TestSlugRegex:
    """Tests for SLUG_REGEX."""

    @pytest.mark.parametrize("slug", ["abc", "abc-123", "a", "1-2-3"])
    def test_valid_slugs(self, slug: str):
        assert SLUG_REGEX.fullmatch(slug)

    @pytest.mark.parametrize(
        "slug", ["-abc", "abc-", "ab--c", "Abc", "abc_def", "a b", ""]
    )
    def test_invalid_slugs(self, slug: str):
        assert not SLUG_REGEX.fullmatch(slug)


class TestSkillGenerationRequestValidation:
    """Tests for SkillGenerationRequest model validation."""

    def test_empty_conversation_rejected(self):
        with pytest.raises(ValidationError):
            SkillGenerationRequest(conversation=[])

    def test_bad_role_rejected(self):
        with pytest.raises(ValidationError):
            SkillGenerationRequest(
                conversation=[{"role": "system", "content": "not allowed"}]
            )


class TestGenerateSkill:
    """Tests for generate_skill with the LLM prompt call mocked."""

    @pytest.mark.asyncio
    async def test_valid_slug_is_preserved(
        self, skill_generation_service: SkillGenerationService
    ):
        with patch("openbb_ada.services.skill_generation.prompt") as mock_prompt:
            _mock_llm(
                mock_prompt,
                SkillGenerationResponse(
                    slug="earnings-analysis",
                    description="Analyze earnings.",
                    content="# Earnings Analysis",
                ),
            )

            response = await skill_generation_service.generate_skill(_request())

        assert response.slug == "earnings-analysis"
        assert response.description == "Analyze earnings."
        assert response.content == "# Earnings Analysis"

    @pytest.mark.asyncio
    async def test_invalid_slug_falls_back_to_slugified_slug(
        self, skill_generation_service: SkillGenerationService
    ):
        with patch("openbb_ada.services.skill_generation.prompt") as mock_prompt:
            _mock_llm(
                mock_prompt,
                SkillGenerationResponse(
                    slug="Earnings Analysis!",
                    description="Analyze earnings.",
                    content="# Earnings Analysis",
                ),
            )

            response = await skill_generation_service.generate_skill(_request())

        assert response.slug == "earnings-analysis"

    @pytest.mark.asyncio
    async def test_unusable_slug_falls_back_to_name_hint(
        self, skill_generation_service: SkillGenerationService
    ):
        with patch("openbb_ada.services.skill_generation.prompt") as mock_prompt:
            _mock_llm(
                mock_prompt,
                SkillGenerationResponse(
                    slug="!!!",
                    description="A skill.",
                    content="Instructions",
                ),
            )

            response = await skill_generation_service.generate_skill(
                _request(name_hint="My Workflow")
            )

        assert response.slug == "my-workflow"

    @pytest.mark.asyncio
    async def test_unusable_slug_without_name_hint_falls_back_to_saved_skill(
        self, skill_generation_service: SkillGenerationService
    ):
        with patch("openbb_ada.services.skill_generation.prompt") as mock_prompt:
            _mock_llm(
                mock_prompt,
                SkillGenerationResponse(
                    slug="!!!",
                    description="A skill.",
                    content="Instructions",
                ),
            )

            response = await skill_generation_service.generate_skill(_request())

        assert response.slug == "saved-skill"

    @pytest.mark.asyncio
    async def test_too_short_slug_falls_back_to_saved_skill(
        self, skill_generation_service: SkillGenerationService
    ):
        """A single-char slug matches SLUG_REGEX but fails the length check,
        and its fallback is also too short."""
        with patch("openbb_ada.services.skill_generation.prompt") as mock_prompt:
            _mock_llm(
                mock_prompt,
                SkillGenerationResponse(
                    slug="a",
                    description="A skill.",
                    content="Instructions",
                ),
            )

            response = await skill_generation_service.generate_skill(_request())

        assert response.slug == "saved-skill"

    @pytest.mark.asyncio
    async def test_generate_skill_fits_conversation_to_budget(
        self, skill_generation_service: SkillGenerationService
    ):
        """generate_skill drops the oldest messages when the derived context
        budget is exceeded."""
        conversation = [
            SkillConversationMessage(
                role="human" if i % 2 == 0 else "ai",
                content=f"message-{i:02d} " + "x" * 90,
            )
            for i in range(10)
        ]
        request = _request(conversation=conversation)

        with (
            patch(
                "openbb_ada.services.skill_generation.context_char_budget",
                return_value=250,
            ),
            patch("openbb_ada.services.skill_generation.prompt") as mock_prompt,
        ):
            _mock_llm(
                mock_prompt,
                SkillGenerationResponse(
                    slug="my-skill",
                    description="A skill.",
                    content="Instructions",
                ),
            )

            await skill_generation_service.generate_skill(request)

        # Each message is 101 chars, so only the newest two fit in 250.
        assert len(request.conversation) == 2
        assert request.conversation[0].content.startswith("message-08")
        assert request.conversation[1].content.startswith("message-09")


class TestFitConversationToBudget:
    """Tests for the fit_conversation_to_budget helper."""

    def _conversation(self, count: int, size: int) -> list[SkillConversationMessage]:
        return [
            SkillConversationMessage(
                role="human" if i % 2 == 0 else "ai",
                content=f"message-{i:02d} " + "x" * (size - 11),
            )
            for i in range(count)
        ]

    def test_keeps_everything_when_within_budget(self):
        conversation = self._conversation(5, 100)
        result = fit_conversation_to_budget(conversation, budget_chars=1_000)
        assert len(result) == 5
        assert all(len(m.content) == 100 for m in result)

    def test_keeps_newest_whole_messages_in_order(self):
        conversation = self._conversation(5, 100)
        result = fit_conversation_to_budget(conversation, budget_chars=250)
        assert len(result) == 2
        assert result[0].content.startswith("message-03")
        assert result[1].content.startswith("message-04")

    def test_caps_individual_messages(self):
        conversation = self._conversation(2, MAX_MESSAGE_CONTENT_LENGTH + 1_000)
        result = fit_conversation_to_budget(
            conversation, budget_chars=MAX_MESSAGE_CONTENT_LENGTH * 3
        )
        assert len(result) == 2
        assert all(len(m.content) == MAX_MESSAGE_CONTENT_LENGTH for m in result)

    def test_newest_message_survives_a_tiny_budget_truncated(self):
        conversation = self._conversation(3, 100)
        result = fit_conversation_to_budget(conversation, budget_chars=40)
        assert len(result) == 1
        assert len(result[0].content) == 40
        assert result[0].content.startswith("message-02")
