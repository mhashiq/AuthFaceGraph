"""
AuthBrain AI Face Analysis Engine
Structured Logging Configuration

Uses structlog for JSON-formatted, contextual log output.
Logs include timestamp, level, logger name, and all context variables.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from app.core.config import get_settings

settings = get_settings()


def add_app_context(logger: Any, method: str, event_dict: EventDict) -> EventDict:
    """Add application-level context to every log entry."""
    event_dict["app"] = settings.APP_NAME
    event_dict["version"] = settings.APP_VERSION
    event_dict["env"] = settings.ENVIRONMENT
    return event_dict


def configure_logging() -> None:
    """
    Configure structlog with JSON rendering for production
    or pretty console rendering for development.
    """
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        add_app_context,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.ENVIRONMENT == "production":
        # JSON output for log aggregation (Datadog, Loki, CloudWatch)
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        # Pretty colored output for development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    # Suppress noisy third-party loggers
    for noisy_logger in ["uvicorn.access", "multipart"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Returns a named structlog logger."""
    return structlog.get_logger(name)
