import logging.config
import traceback

from fastapi import Request
from fastapi.responses import JSONResponse

from . import constants
from .services import LoggingService

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        }
    },
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "json"}},
    "loggers": {
        "uvicorn": {
            "handlers": ["console"],
            "level": constants.LOG_LEVEL,
        },
        # disable uvicorn.access, since we use custom access logs in our middleware.
        "uvicorn.access": {
            "handlers": [],
            "level": "INFO",
            "propagate": False,
        },
        "openbb_ada.main": {
            "handlers": ["console"],
            "level": constants.LOG_LEVEL,
            "propagate": False,
        },
        "openbb_ada.services": {
            "handlers": ["console"],
            "level": constants.LOG_LEVEL,
            "propagate": False,
        },
        "openbb_ada.utils": {
            "handlers": ["console"],
            "level": constants.LOG_LEVEL,
            "propagate": False,
        },
    },
}


def set_up_logging():
    logging.config.dictConfig(LOGGING_CONFIG)


async def log_exception_with_trace_id(request: Request, exc: Exception):
    logging_service = LoggingService(trace_id=request.state.trace_id)
    logging_service.critical(
        "Unhandled exception: %s - %s\nStacktrace:\n%s",
        type(exc).__name__,
        str(exc),
        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Something unexpected happened."},
    )
