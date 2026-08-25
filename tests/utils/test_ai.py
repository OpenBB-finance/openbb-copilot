from unittest.mock import Mock

import pytest

from openbb_ada.utils.ai import _map_temperature_to_verbosity, get_llm


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch OpenaiChatModel and required module-level constants for get_llm tests."""
    mock_model = Mock()
    monkeypatch.setattr("openbb_ada.utils.ai.OpenaiChatModel", mock_model)
    monkeypatch.setattr(
        "openbb_ada.utils.ai.OPENBB_AGENT_MODEL_PROVIDER", "openai", raising=False
    )
    monkeypatch.setattr(
        "openbb_ada.utils.ai.OPENBB_AGENT_OPENAI_BASE_URL",
        "https://api.openai.com",
        raising=False,
    )
    monkeypatch.setattr(
        "openbb_ada.utils.ai.OPENBB_AGENT_OPENAI_API_KEY", "test-key", raising=False
    )
    return mock_model


def test_map_temperature_to_verbosity():
    """Map temperature values to low/medium/high verbosity based on thresholds."""
    assert _map_temperature_to_verbosity(0.0) == "low"
    assert _map_temperature_to_verbosity(0.29) == "low"

    assert _map_temperature_to_verbosity(0.3) == "medium"
    assert _map_temperature_to_verbosity(0.5) == "medium"
    assert _map_temperature_to_verbosity(0.79) == "medium"

    assert _map_temperature_to_verbosity(0.8) == "high"
    assert _map_temperature_to_verbosity(1.0) == "high"
    assert _map_temperature_to_verbosity(1.5) == "high"


@pytest.mark.parametrize(
    ("model", "temperature", "expected_verbosity"),
    [
        ("gpt-5.5", 0.9, "high"),
        ("gpt-5.4-mini", 0.1, "low"),
    ],
)
def test_gpt5_parameter_mapping(mock_llm, model, temperature, expected_verbosity):
    """Test full parameter mapping for supported GPT-5 models."""
    get_llm(model=model, temperature=temperature, max_tokens=1000)

    call_kwargs = mock_llm.call_args.kwargs

    assert call_kwargs["model"] == model
    assert call_kwargs["verbosity"] == expected_verbosity
    assert "reasoning_effort" not in call_kwargs
    assert "temperature" not in call_kwargs
    assert "max_tokens" not in call_kwargs
    assert call_kwargs["max_completion_tokens"] == 1000


def test_gpt4_compatibility(mock_llm):
    """Test that GPT-4 models strip GPT-5-specific parameters."""
    get_llm(
        model="gpt-4.1",
        temperature=0.7,
        max_tokens=500,
        reasoning_effort="medium",
        verbosity="high",
    )

    call_kwargs = mock_llm.call_args.kwargs

    assert call_kwargs["model"] == "gpt-4.1"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["max_tokens"] == 500
    assert "reasoning_effort" not in call_kwargs
    assert "verbosity" not in call_kwargs
