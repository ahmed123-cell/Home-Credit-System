"""Shared logging configuration and reusable logging helpers across training modules."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, event, <fields>, exception."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-friendly variant: `time LEVEL logger event key=value ...`."""

    def format(self, record: logging.LogRecord) -> str:
        fields = " ".join(f"{k}={v}" for k, v in getattr(record, "fields", {}).items())
        line = (
            f"{self.formatTime(record, '%H:%M:%S')} "
            f"{record.levelname:<7} {record.name} {record.getMessage()} {fields}"
        ).rstrip()
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging(
    log_format: str | None = None, log_level: str | None = None
) -> None:
    """Configure root logging. Safe to call once from entry points.

    Args:
        log_format: 'json' or 'console'. If None, reads LOG_FORMAT env var (defaults to 'console').
        log_level: 'DEBUG', 'INFO', 'WARNING', etc. If None, reads LOG_LEVEL env var (defaults to 'INFO').
    """
    if log_format is None:
        log_format = os.getenv("LOG_FORMAT", "console")
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")

    handler = logging.StreamHandler(sys.stdout)
    use_console = log_format.lower() == "console"
    handler.setFormatter(ConsoleFormatter() if use_console else JsonFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(log_level.upper())

    # Suppress verbose third-party loggers unless explicitly in DEBUG
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("mlflow").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger instance with standard naming."""
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any
) -> None:
    """Emit a structured log event with arbitrary extra keyword arguments."""
    logger.log(level, event, extra={"fields": fields})


@contextmanager
def timed(
    logger: logging.Logger, event: str, **fields: Any
) -> Iterator[dict[str, Any]]:
    """Log `<event>.started`, then `<event>.completed` (or `.failed`) with duration in seconds.

    Yields a dict that the caller can populate with result metrics:
        with timed(logger, "step") as ctx:
            ctx["processed_items"] = 100
    """
    result: dict[str, Any] = {}
    start = time.perf_counter()
    log_event(logger, f"{event}.started", **fields)
    try:
        yield result
    except Exception:
        duration = round(time.perf_counter() - start, 2)
        logger.exception(
            f"{event}.failed",
            extra={"fields": {**fields, "duration_s": duration}},
        )
        raise
    duration = round(time.perf_counter() - start, 2)
    log_event(logger, f"{event}.completed", duration_s=duration, **fields, **result)
