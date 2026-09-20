"""Logging that never touches stdout.

The MCP stdio transport owns stdout. A single stray diagnostic line there
corrupts the protocol stream, so every component logs to a rotating file under
the state directory and, optionally, to stderr.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

_CONFIGURED: set[str] = set()


def setup_logging(cfg: "Config", component: str, *, stderr: bool | None = None) -> logging.Logger:
    if component in _CONFIGURED:
        return logging.getLogger(component)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(cfg.log_level).upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(name)s pid=%(process)d] %(message)s"
    )
    fh = logging.handlers.RotatingFileHandler(
        cfg.log_dir / f"{component}.log", maxBytes=8 * 1024 * 1024,
        backupCount=3, encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    if stderr is None:
        stderr = os.environ.get("AMOEBA_STDERR_LOG", "1") != "0"
    if stderr:
        sh = logging.StreamHandler(stream=sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    _CONFIGURED.add(component)
    return logging.getLogger(component)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
