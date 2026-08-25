import logging
import uuid
from contextlib import contextmanager

from openbb_ada.services import LoggingService

LOGGER_NAME = "openbb_ada.services._logging"


@contextmanager
def capture_logger(caplog, level: int):
    logger = logging.getLogger(LOGGER_NAME)
    previous_level = logger.level
    logger.addHandler(caplog.handler)
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous_level)


def test_logging_service_info(caplog):
    trace_id = uuid.uuid4()
    logging_service = LoggingService(trace_id=trace_id)

    with capture_logger(caplog, logging.INFO):
        logging_service.info("test", extra={"some_key": "some_value"})

    record = caplog.records[-1]
    assert record.name == LOGGER_NAME
    assert record.levelname == "INFO"
    assert record.msg == "test"
    assert record.args == ()
    assert record.trace_id == trace_id
    assert record.some_key == "some_value"


def test_logging_service_info_preserves_percent_style_formatting(caplog):
    trace_id = uuid.uuid4()
    logging_service = LoggingService(trace_id=trace_id)

    with capture_logger(caplog, logging.INFO):
        logging_service.info("value=%s count=%d", "abc", 3)

    record = caplog.records[-1]
    assert record.msg == "value=%s count=%d"
    assert record.args == ("abc", 3)
    assert record.getMessage() == "value=abc count=3"
    assert record.trace_id == trace_id


def test_logging_service_info_propagates_event_fields_via_extra(caplog):
    trace_id = uuid.uuid4()
    logging_service = LoggingService(trace_id=trace_id)

    with capture_logger(caplog, logging.INFO):
        logging_service.info(
            "agent_step_start",
            extra={
                "event": "agent_step_start",
                "step_id": "abc-step-1",
                "call_index": 1,
            },
        )

    record = caplog.records[-1]
    assert record.msg == "agent_step_start"
    assert record.trace_id == trace_id
    assert record.event == "agent_step_start"
    assert record.step_id == "abc-step-1"
    assert record.call_index == 1


def test_logging_service_info_does_not_mutate_input_extra(caplog):
    trace_id = uuid.uuid4()
    logging_service = LoggingService(trace_id=trace_id)
    original_extra = {"event": "tool_call_emitted"}

    with capture_logger(caplog, logging.INFO):
        logging_service.info("tool_call_emitted", extra=original_extra)

    assert original_extra == {"event": "tool_call_emitted"}


def test_logging_service_warning(caplog):
    trace_id = uuid.uuid4()
    logging_service = LoggingService(trace_id=trace_id)

    with capture_logger(caplog, logging.WARNING):
        logging_service.warning("test", extra={"some_key": "some_value"})

    record = caplog.records[-1]
    assert record.name == LOGGER_NAME
    assert record.levelname == "WARNING"
    assert record.msg == "test"
    assert record.trace_id == trace_id
    assert record.some_key == "some_value"


def test_logging_service_error(caplog):
    trace_id = uuid.uuid4()
    logging_service = LoggingService(trace_id=trace_id)

    with capture_logger(caplog, logging.ERROR):
        logging_service.error("test", extra={"some_key": "some_value"})

    record = caplog.records[-1]
    assert record.name == LOGGER_NAME
    assert record.levelname == "ERROR"
    assert record.msg == "test"
    assert record.trace_id == trace_id
    assert record.some_key == "some_value"
