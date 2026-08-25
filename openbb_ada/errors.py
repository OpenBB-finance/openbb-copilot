class StructuredContextError(Exception):
    """Class for structured context errors."""


class ContextLimitExceededError(Exception):
    """Class for context limit exceeded errors."""


class ToolLimitExceededError(Exception):
    """Raised when the outbound OpenAI tool count exceeds the allowed limit."""

    def __init__(
        self,
        total_tool_count: int,
        threshold: int,
        mcp_tool_count: int = 0,
    ) -> None:
        self.total_tool_count = total_tool_count
        self.threshold = threshold
        self.mcp_tool_count = mcp_tool_count
        super().__init__(
            f"Tool count {total_tool_count} exceeds the limit of {threshold}."
        )


class FunctionCallError(Exception):
    """Class for function call errors."""


class SqlAgentError(Exception):
    """Class for SQL agent errors."""


class IterationLimitExceededError(Exception):
    """Class for iteration limit exceeded errors."""


class HTTPError(Exception):
    """Class for HTTP errors."""


class RetryExceededError(Exception):
    """Class for retry exceeded errors."""


class CodeGenerationError(Exception):
    """Raised when code generation fails."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(message)
