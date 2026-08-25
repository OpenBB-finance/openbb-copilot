import logging

from magentic import OpenaiChatModel

from ..constants import (
    OPENBB_AGENT_MODEL_MAIN,
    OPENBB_AGENT_MODEL_PROVIDER,
    OPENBB_AGENT_OPENAI_API_KEY,
    OPENBB_AGENT_OPENAI_BASE_URL,
)
from .auth import get_login_token, get_snowflake_host

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model context registry shared by the copilot loop and generation services.
CONTEXT_LIMIT_UNDEFINED_MODEL = 200_000
CONTEXT_LIMIT_SAFETY_FACTOR = 0.8
CONTEXT_LIMIT_BY_MODEL = {
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-5.5": 1_050_000,
    "gpt-5.4-mini": 400_000,
}
# Provider-agnostic heuristic: 1 token ~= 3.5 characters.
TOKEN_CHARS_RATIO = 3.5


def context_char_budget(model: str, reserved_tokens: int = 0) -> int:
    """Usable prompt budget in characters for a model, after the safety
    factor and a reservation for fixed prompt content and output."""
    limit = CONTEXT_LIMIT_BY_MODEL.get(model, CONTEXT_LIMIT_UNDEFINED_MODEL)
    usable_tokens = int(limit * CONTEXT_LIMIT_SAFETY_FACTOR) - reserved_tokens
    return max(0, int(usable_tokens * TOKEN_CHARS_RATIO))


def _is_gpt5_model(model_name: str) -> bool:
    """Check if the model is GPT-5.x which has different parameter requirements."""
    return "gpt-5" in model_name.lower()


def _map_temperature_to_verbosity(temperature: float) -> str:
    """Map temperature to GPT-5's verbosity parameter when reasoning_effort is none."""
    if temperature < 0.3:
        return "low"
    elif temperature < 0.8:
        return "medium"
    return "high"


def get_llm(**kwargs) -> OpenaiChatModel:
    if "model" not in kwargs:
        kwargs["model"] = OPENBB_AGENT_MODEL_MAIN

    model_name = kwargs["model"]
    is_gpt5 = _is_gpt5_model(model_name)

    # Handle backward compatibility for GPT-4.x vs GPT-5.x parameter differences
    if is_gpt5:
        # GPT-5.x: Convert max_tokens to max_completion_tokens
        if "max_tokens" in kwargs:
            logger.debug(
                "Converting max_tokens to max_completion_tokens for GPT-5 model: %s",
                model_name,
            )
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

        if "temperature" in kwargs:
            temp_value = kwargs.pop("temperature")
            verbosity = _map_temperature_to_verbosity(temp_value)
            kwargs["verbosity"] = verbosity
            logger.debug(
                "Mapped temperature=%s to verbosity=%s for GPT-5 model: %s",
                temp_value,
                verbosity,
                model_name,
            )

        # Do not set a default reasoning_effort here. This wrapper only constructs
        # the model; magentic adds function tools later, and GPT-5 Chat Completions
        # rejects function tools when reasoning_effort is explicitly present.
    else:
        # GPT-4.x: Remove GPT-5 specific parameters
        if "reasoning_effort" in kwargs:
            removed_value = kwargs.pop("reasoning_effort")
            logger.debug(
                "Removed reasoning_effort=%s (not supported in GPT-4 model: %s)",
                removed_value,
                model_name,
            )

        if "verbosity" in kwargs:
            removed_value = kwargs.pop("verbosity")
            logger.debug(
                "Removed verbosity=%s (not supported in GPT-4 model: %s)",
                removed_value,
                model_name,
            )

    # We also use the OpenAI API if the user is providing us with an API key.
    if OPENBB_AGENT_MODEL_PROVIDER == "openai" or kwargs.get("api_key"):
        kwargs["api_type"] = "openai"
    elif OPENBB_AGENT_MODEL_PROVIDER == "azure":
        kwargs["api_type"] = "azure"
    else:
        raise ValueError("OPENBB_AGENT_MODEL_PROVIDER is not set")

    # Explicitly set the base URL to our proxy if using OpenAI API
    if kwargs["api_type"] == "openai":
        kwargs["base_url"] = OPENBB_AGENT_OPENAI_BASE_URL

    # Explicitly set the API key if the user API key is not provided
    if not kwargs.get("api_key") and kwargs["api_type"] == "openai":
        kwargs["api_key"] = OPENBB_AGENT_OPENAI_API_KEY

    # The "snowflake" API key value is explicitly used in snowpark container environment
    # If the API key is "snowflake", replace it with the Snowflake OAuth token
    # and construct the correct base URL dynamically
    if kwargs.get("api_key") == "snowflake":
        kwargs["api_key"] = get_login_token()

        # Dynamically construct base URL with correct Snowflake host
        snowflake_host = get_snowflake_host()
        kwargs["base_url"] = f"https://{snowflake_host}/api/v2/cortex/openai"

        logger.info(
            f"[SNOWFLAKE MODE] Using Snowflake OAuth token with host: {snowflake_host}"
        )

    return OpenaiChatModel(**kwargs)
