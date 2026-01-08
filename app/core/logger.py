from loguru import logger
import sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger.remove()

logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>.<cyan>{function}</cyan>:"
        "<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    level="DEBUG",
    colorize=True,
    backtrace=True,
    diagnose=True,
)

logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    enqueue=True,
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    level="DEBUG",
    encoding="utf-8",
)

logger.add(
    LOG_DIR / "errors.log",
    enqueue=True,
    level="ERROR",
    rotation="10 MB",
    retention="30 days",
)

__all__ = ["logger"]
