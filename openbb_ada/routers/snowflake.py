from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..dependencies import (
    get_logging_service,
    get_python_suggestion_service,
    get_sql_suggestion_service,
)
from ..models import CodeGenerationRequest, CodeGenerationResponse
from ..services import (
    LoggingService,
    PythonCodeGenerationService,
    SqlQueryGenerationService,
)
from ..snowflake_factory import build_snowflake_cortex_analyst_service

router = APIRouter()
security = HTTPBearer(auto_error=False)


def get_snowflake_cortex_analyst_service(
    request: Request,
    logging_service: Annotated[
        LoggingService,
        Depends(get_logging_service(name="Snowflake Cortex Analyst Service")),
    ],
    sql_query_generation_service: Annotated[
        SqlQueryGenerationService, Depends(get_sql_suggestion_service)
    ],
    python_code_generation_service: Annotated[
        PythonCodeGenerationService, Depends(get_python_suggestion_service)
    ],
):
    """Return Snowflake code generation service."""
    return build_snowflake_cortex_analyst_service(
        request=request,
        logging_service=logging_service,
        sql_query_generation_service=sql_query_generation_service,
        python_code_generation_service=python_code_generation_service,
    )


@router.post("/v1/generate/code", response_model=CodeGenerationResponse)
async def generate_code(
    request: CodeGenerationRequest,
    snowflake_cortex_analyst_service: Annotated[
        Any, Depends(get_snowflake_cortex_analyst_service)
    ],
    logging_service: Annotated[LoggingService, Depends(get_logging_service())],
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> CodeGenerationResponse:
    """Generate code for Snowflake /v1/generate/code."""
    logging_service.info(
        "Generate code request received.",
        extra={
            "widget_uuid": request.widget_uuid,
            "language": request.language,
            "has_semantic_config": request.has_semantic_config,
        },
    )
    return await snowflake_cortex_analyst_service.generate_code(request)
