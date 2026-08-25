import json
import logging
import os
from typing import Literal, cast

import dotenv

logger = logging.getLogger("uvicorn.error")

dotenv.load_dotenv()

ENVIRONMENT = os.environ.get("ENVIRONMENT")
if ENVIRONMENT not in ["PROD", "DEV", "TEST"]:
    raise EnvironmentError(f"Invalid environment: {ENVIRONMENT}")
ENVIRONMENT = cast(Literal["PROD", "DEV", "TEST"], ENVIRONMENT)

CORS_ORIGINS_LIST = os.environ.get("CORS_ORIGINS_LIST") or "[]"
try:
    ORIGINS: list[str] = json.loads(CORS_ORIGINS_LIST)
    if not ORIGINS:
        logger.warning("CORS_ORIGINS_LIST is empty")
except json.JSONDecodeError as err:
    raise ValueError("Invalid CORS_ORIGINS_LIST") from err

ORIGIN_REGEX = os.environ.get("CORS_ORIGIN_REGEX")

# Application Configuration
RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED") == "true"
AUTH_ENABLED = os.environ.get("AUTH_ENABLED") == "true"

# Base URLs for external services
OPENBB_PAYMENTS_BASE_URL = os.environ.get("OPENBB_PAYMENTS_BASE_URL")
JINA_AI_BASE_URL = os.environ.get("JINA_AI_BASE_URL")

# API keys and secrets
OPENBB_PAYMENTS_API_SECRET_KEY = os.getenv("OPENBB_PAYMENTS_API_SECRET_KEY")
JINA_AI_API_KEY = os.environ.get("JINA_AI_API_KEY")
POSTHOG_PROJECT_KEY = os.environ.get("POSTHOG_PROJECT_KEY")
POSTHOG_HOST_URL = os.environ.get("POSTHOG_HOST_URL")

# LLM service providers configuration
OPENBB_AGENT_MODEL_PROVIDER = os.environ.get("OPENBB_AGENT_MODEL_PROVIDER", "openai")
OPENBB_AGENT_MODEL_MAIN = os.environ.get("OPENBB_AGENT_MODEL_MAIN", "gpt-4.1")
OPENBB_AGENT_MODEL_SMALL = os.environ.get("OPENBB_AGENT_MODEL_SMALL", "gpt-4.1-mini")
OPENBB_AGENT_MODEL_VISION = os.environ.get("OPENBB_AGENT_MODEL_VISION", "gpt-4.1")
OPENBB_AGENT_OPENAI_BASE_URL = os.environ.get(
    "OPENBB_AGENT_OPENAI_BASE_URL", "https://api.openai.com/v1"
)
OPENBB_AGENT_OPENAI_API_KEY = os.environ.get("OPENBB_AGENT_OPENAI_API_KEY")

# Embedding model configuration
OPENBB_EMBEDDING_MODEL_PROVIDER = os.environ.get(
    "OPENBB_EMBEDDING_MODEL_PROVIDER", "openai"
)
OPENBB_EMBEDDING_MODEL = os.environ.get(
    "OPENBB_EMBEDDING_MODEL", "text-embedding-3-small"
)
OPENBB_EMBEDDING_BASE_URL = (
    os.environ.get("OPENBB_EMBEDDING_BASE_URL") or "https://api.openai.com/v1"
)
OPENBB_EMBEDDING_API_KEY = os.environ.get("OPENBB_EMBEDDING_API_KEY")

# Search model configuration
OPENBB_SEARCH_PROVIDER = os.environ.get("OPENBB_SEARCH_PROVIDER", "openai")
OPENBB_SEARCH_BASE_URL = os.environ.get(
    "OPENBB_SEARCH_BASE_URL", "https://api.openai.com/v1"
)
OPENBB_SEARCH_MODEL = os.environ.get("OPENBB_SEARCH_MODEL", "gpt-4.1")
OPENBB_SEARCH_API_KEY = os.environ.get("OPENBB_SEARCH_API_KEY")

# Feature toggles
URL_RETRIEVAL_ENABLED = bool(JINA_AI_BASE_URL and JINA_AI_API_KEY)
SNOWFLAKE_NATIVE_APP = os.environ.get("SNOWFLAKE_NATIVE_APP", "false").lower() == "true"
MAX_CORTEX_ANALYST_RETRIES = 2
CORTEX_ANALYST_REQUEST_TIMEOUT = 45.0
CORTEX_ANALYST_API_ENDPOINT = "api/v2/cortex/analyst/message"

# Web search feature
LLM_WEB_SEARCH_ENABLED = (
    os.environ.get("LLM_WEB_SEARCH_ENABLED", "true").lower() == "true"
)

# Logging configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Telemetry and Analytics
POSTHOG_ENABLED = bool(
    POSTHOG_PROJECT_KEY and POSTHOG_HOST_URL and ENVIRONMENT != "TEST"
)
LOGFIRE_TOKEN = os.environ.get("LOGFIRE_TOKEN")
LLM_TRACING_ENABLED = bool(LOGFIRE_TOKEN) and ENVIRONMENT != "TEST"

# File extension configurations
STRUCTURED_FILE_EXTENSIONS = ["csv", "xlsx"]
IMAGE_FILE_EXTENSIONS = ["png", "jpg", "jpeg"]
UNSTRUCTURED_FILE_EXTENSIONS = ["txt", "md", "pdf", "docx", "html"]
SUPPORTED_FILE_EXTENSIONS = (
    UNSTRUCTURED_FILE_EXTENSIONS + STRUCTURED_FILE_EXTENSIONS + IMAGE_FILE_EXTENSIONS
)
# File types that cannot be inlined as structured data and must be fetched via
# _llm_query_uploaded_files before the LLM can reason about their content.
FILE_EXTENSIONS_REQUIRING_QUERY = UNSTRUCTURED_FILE_EXTENSIONS + IMAGE_FILE_EXTENSIONS

# Field exclusion lists (keep lowercase)
EXCLUDE_CITATION_DETAILS_FIELDS = [
    "lastupdated",
    "source",
    "id",
    "uuid",
    "storedfileuuid",
    "url",
    "datakey",
    "originalfilename",
    "extension",
    "category",
    "subcategory",
    "transcript_url",
]
EXCLUDE_STATUS_UPDATE_DETAILS_FIELDS = [
    "lastupdated",
    "source",
    "id",
    "uuid",
    "storedfileuuid",
    "url",
    "datakey",
    "originalfilename",
    "extension",
    "category",
    "subcategory",
    "transcript_url",
]

# Redis configurations
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT") or 6379)
USE_EXTRA_WIDGETS_CACHE = os.getenv("USE_EXTRA_WIDGETS_CACHE") == "true"
USE_DOCUMENT_CACHE = os.getenv("USE_DOCUMENT_CACHE") == "true"
