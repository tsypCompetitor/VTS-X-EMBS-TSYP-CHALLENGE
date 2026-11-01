"""Structured logging setup for CABINGUARD-ADI."""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(level: str = "INFO", log_dir: str | None = None, run_id: str | None = None) -> logging.Logger:
    """Configure root logger with console and optional file handler."""
    level_int = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level_int)
    root.handlers.clear()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level_int)
    ch.setFormatter(formatter)
    root.addHandler(ch)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        tag = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(Path(log_dir) / f"run_{tag}.log", encoding="utf-8")
        fh.setLevel(level_int)
        fh.setFormatter(formatter)
        root.addHandler(fh)
    return root

