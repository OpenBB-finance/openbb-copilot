import hashlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any
from uuid import UUID

import logfire
import posthog

from .. import constants

if TYPE_CHECKING:
    from openbb_ai.models import StatusUpdateSSE

if constants.POSTHOG_ENABLED:
    posthog_client = posthog.Posthog(
        constants.POSTHOG_PROJECT_KEY,
        constants.POSTHOG_HOST_URL,
    )
else:
    posthog_client = posthog.Posthog("", "")
    posthog_client.disabled = True


logging.basicConfig(level=getattr(logging, constants.LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# Only configure logfire if tracing is enabled to avoid OpenTelemetry
# network connections in restricted environments (like Snowflake)
if constants.LLM_TRACING_ENABLED:
    logfire.configure(
        service_name="openbb_ai",
        send_to_logfire=True,
        environment=constants.ENVIRONMENT,
    )
    logfire.instrument_openai()
else:
    # Minimal configuration without any exporters
    # Note:
    # OTEL_SDK_DISABLED environment variable must me set to "true"
    # to fully disable attempts to do external network calls
    logfire.configure(
        send_to_logfire=False,
        console=False,  # Disable console logging from logfire
    )


class LoggingService:
    """Provide logging and tracing functionality."""

    def __init__(
        self,
        trace_id: UUID | None = None,
        name: str | None = None,
    ):
        self._trace_id = trace_id or uuid.uuid4()
        self._name = name
        self._prefix = "" if self._name is None else self._name + ": "

    @property
    def trace_id(self) -> UUID:
        return self._trace_id

    @staticmethod
    def make_stable_hash(value: Any) -> str:
        """Generate a short stable hash for structured log correlation."""
        try:
            serialized = json.dumps(
                value,
                sort_keys=True,
                default=str,
                ensure_ascii=True,
            )
        except TypeError:
            serialized = repr(value)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    def info(self, message: str, *args, extra: dict[str, Any] | None = None):
        extra = dict(extra or {})
        extra["trace_id"] = self.trace_id
        logger.info(self._prefix + message, *args, extra=extra)

    def warning(self, message: str, *args, extra: dict[str, Any] | None = None):
        extra = dict(extra or {})
        extra["trace_id"] = self.trace_id
        logger.warning(self._prefix + message, *args, extra=extra)

    def error(self, message: str, *args, extra: dict[str, Any] | None = None):
        extra = dict(extra or {})
        extra["trace_id"] = self.trace_id
        logger.error(self._prefix + message, *args, extra=extra)

    def critical(self, message: str, *args, extra: dict[str, Any] | None = None):
        extra = dict(extra or {})
        extra["trace_id"] = self.trace_id
        logger.critical(self._prefix + message, *args, extra=extra)

    def debug(self, message: str, *args, extra: dict[str, Any] | None = None):
        extra = dict(extra or {})
        extra["trace_id"] = self.trace_id
        logger.debug(self._prefix + message, *args, extra=extra)

    def log_conversation_start(self):
        """Log ASCII art for a new conversation start."""
        logger.debug("")
        logger.debug("=" * 80)
        logger.debug("NEW AI COPILOT CONVERSATION".center(80))
        logger.debug("=" * 80)
        logger.debug("")

    def log_new_message(self):
        """Log ASCII art for a new user message in ongoing conversation."""
        logger.debug("")
        logger.debug("-" * 80)
        logger.debug("NEW USER MESSAGE".center(80))
        logger.debug("-" * 80)
        logger.debug("")

    def log_status_update_sse(self, event: "StatusUpdateSSE"):
        """Log a StatusUpdateSSE event with all relevant details."""
        artifact_length = (
            len(event.data.artifacts[0].content) if event.data.artifacts else 0
        )
        self.info(
            "Yielding event: eventType=%s, message=%s, details=%s, artifact_length=%s",
            event.data.eventType,
            event.data.message,
            event.data.details,
            artifact_length,
        )
