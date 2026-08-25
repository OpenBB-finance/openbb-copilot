import logging
import os
from pathlib import Path

from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SPCS_TOKEN_FILE = Path("/snowflake/session/token")


def is_running_in_spcs() -> bool:
    """Check if running inside a Snowpark Container Services environment."""
    return SPCS_TOKEN_FILE.exists()


def get_login_token(ingress_user_token: str | None = None) -> str:
    """Get the OAuth token provided by Snowflake in the container"""
    if SPCS_TOKEN_FILE.exists():
        token = SPCS_TOKEN_FILE.read_text().strip()
        logger.info("[SNOWFLAKE MODE] Snowflake OAuth token found")
        if ingress_user_token:
            # Concatenate service token with user token for caller's rights
            token += "." + ingress_user_token
        return token

    if os.environ.get("SNOWFLAKE_NATIVE_APP", "false").lower() == "true":
        token = os.getenv("SNOWFLAKE_TOKEN", "")
        if token:
            logger.info("[LOCAL MODE] Snowflake token from SNOWFLAKE_TOKEN env var")
            return token

    _error_message = (
        "[SNOWFLAKE MODE] Snowflake OAuth token not found. "
        "Are you running inside a Snowpark container?"
    )
    logger.error(_error_message)
    raise HTTPException(
        status_code=500,
        detail=_error_message,
    ) from None


def get_snowflake_host() -> str:
    """Get Snowflake host from environment or connection info"""
    # Try environment variable first
    host = os.getenv("SNOWFLAKE_HOST")
    if not host:
        _error_message = "[SNOWFLAKE MODE] SNOWFLAKE_HOST not found in environment"
        logger.error(_error_message)
        raise HTTPException(
            status_code=500,
            detail=_error_message,
        )
    return host
