from typing import TYPE_CHECKING

from fastapi import Request

from .services import (
    LoggingService,
    PythonCodeGenerationService,
    SqlQueryGenerationService,
)

if TYPE_CHECKING:
    from .services.snowflake_cortex_analyst import SnowflakeCortexAnalystService


def build_snowflake_cortex_analyst_service(
    request: Request,
    logging_service: LoggingService,
    sql_query_generation_service: SqlQueryGenerationService | None,
    python_code_generation_service: PythonCodeGenerationService | None,
) -> "SnowflakeCortexAnalystService":
    """Create SnowflakeCortexAnalystService.

    The ingress user token is read from the request and threaded into the
    Cortex Analyst client. The client currently ignores it (Cortex Analyst
    runs as the application owner — see
    `CortexAnalystClient._get_auth_headers`), but the plumbing is preserved
    so we can re-enable caller's rights without touching the route layer.
    """
    from .services.snowflake_cortex_analyst import (
        CortexAnalystClient,
        SnowflakeCortexAnalystService,
    )

    headers = request.headers if request else {}
    ingress_user_token = headers.get("Sf-Context-Current-User-Token")

    return SnowflakeCortexAnalystService(
        logging_service=logging_service,
        sql_query_generation_service=sql_query_generation_service,
        python_code_generation_service=python_code_generation_service,
        cortex_analyst_client=CortexAnalystClient(
            logging_service=logging_service, ingress_user_token=ingress_user_token
        ),
    )
