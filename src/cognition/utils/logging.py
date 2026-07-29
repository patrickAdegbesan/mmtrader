"""Structured (JSON) logging setup, shared across every module.

Every decision the system makes later (agent votes, risk-engine
rejections, order placement) should go through this logger so the whole
pipeline is auditable from log files alone.

Every logger created here is a child of a single "cognition" parent and
propagates to it. get_logger() alone only attaches a console handler (so
importing a module never touches the filesystem); a CLI entrypoint calls
configure_file_logging() exactly once to attach a rotating file handler
to the parent, and from then on EVERY cognition.* logger's records reach
that file too — including loggers created at import time before the log
directory was known (downloader, bybit_client, paper_trader, risk_engine,
...), which previously never wrote to a file at all.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path

_PARENT_NAME = "cognition"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        name = record.name
        if name.startswith(_PARENT_NAME + "."):
            name = name[len(_PARENT_NAME) + 1:]
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """A console-only JSON logger, namespaced under "cognition" so it
    picks up a file handler later if/when configure_file_logging() runs.
    Safe to call at import time.
    """
    parent = logging.getLogger(_PARENT_NAME)
    parent.propagate = False  # never bubble into the root logger

    logger = logging.getLogger(f"{_PARENT_NAME}.{name}")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    logger.propagate = True

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(JsonFormatter())
    logger.addHandler(console)

    return logger


def configure_file_logging(
    log_dir: Path,
    name: str,
    level: int = logging.INFO,
    max_bytes: int = 20_000_000,
    backup_count: int = 5,
) -> Path:
    """Attach a rotating file handler to the "cognition" parent logger, so
    every module's log records for this process — not just the calling
    CLI's own logger — land in log_dir/{name}.log. Call once, near the
    start of main(). Safe to call again (e.g. across tests in the same
    process): replaces rather than accumulates the parent's file handler.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}.log"

    parent = logging.getLogger(_PARENT_NAME)
    parent.propagate = False
    for handler in [h for h in parent.handlers if isinstance(h, logging.handlers.RotatingFileHandler)]:
        parent.removeHandler(handler)
        handler.close()

    file_handler = logging.handlers.RotatingFileHandler(
        path, maxBytes=max_bytes, backupCount=backup_count,
    )
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(level)
    parent.addHandler(file_handler)
    return path


def log_with_fields(logger: logging.Logger, level: int, message: str, **fields) -> None:
    logger.log(level, message, extra={"extra_fields": fields})
