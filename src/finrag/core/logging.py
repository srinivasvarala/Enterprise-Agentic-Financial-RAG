"""Structured JSON logging. Ingestion logs carry tenant_id, document_id and job_id."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

_RESERVED = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}
_HANDLER_NAME = "finrag-json"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    """Idempotently attach a JSON handler to the `finrag` logger."""
    logger = logging.getLogger("finrag")
    logger.setLevel(level.upper())
    if not any(h.get_name() == _HANDLER_NAME for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
