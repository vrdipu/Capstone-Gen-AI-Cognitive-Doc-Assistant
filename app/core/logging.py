from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from app.core.config import get_settings


def setup_logging():
    settings = get_settings()
    logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    logger.add(sys.stdout, format=log_format, level="DEBUG" if settings.debug else "INFO", colorize=True)
    if settings.app_env == "production":
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logger.add(log_dir / "app.log", format=log_format, level="INFO", rotation="10 MB", retention="7 days")
    return logger


app_logger = setup_logging()
