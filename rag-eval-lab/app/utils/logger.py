"""
app/utils/logger.py
====================
Centralized logging setup using Loguru.

📚 LESSON — Why Loguru over Python's built-in logging?
---------------------------------------------------------
Python's standard `logging` module requires ~20 lines of boilerplate to get a
useful logger. Loguru gives you a beautiful, structured logger in 2 lines:

    from app.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Loaded {count} documents", count=42)

Output (with colors in terminal):
    2024-01-15 10:23:01 | INFO | app.ingestion.loader:load:45 - Loaded 42 documents

Key Loguru features we use:
  - Automatic source location (file, function, line number)
  - Structured key=value logging (great for tracing)
  - Single global sink — configure once, works everywhere
  - Context binding: logger.bind(experiment="v1") propagates to all log calls
"""

import sys

from loguru import logger as _loguru_logger


# We only configure the logger once — this flag prevents double-configuration
# if this module is imported multiple times.
_configured = False


def _configure_logger() -> None:
    """Configure the global Loguru logger with our preferred format."""
    global _configured
    if _configured:
        return

    # Gracefully fall back to INFO if settings can't be loaded yet
    # (e.g., .env file missing during smoke tests or initial setup).
    try:
        from app.config.settings import get_settings
        log_level = get_settings().log_level
    except Exception:
        log_level = "INFO"

    # Remove Loguru's default handler (it has a different format)
    _loguru_logger.remove()

    # Add our custom handler to stderr
    _loguru_logger.add(
        sys.stderr,
        level=log_level,
        # Format breakdown:
        #   {time:HH:mm:ss}   — short timestamp
        #   {level: <8}       — level name, left-aligned, 8 chars wide
        #   {name}            — module name (e.g., app.ingestion.chunking.fixed)
        #   {function}:{line} — where the log was called
        #   {message}         — the actual log message
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,   # show full traceback on exceptions
        diagnose=True,    # show variable values in tracebacks (dev only)
    )

    _configured = True


def get_logger(name: str):
    """
    Return a Loguru logger bound to the given module name.

    Usage:
        logger = get_logger(__name__)
        logger.info("Starting chunker", strategy="fixed", chunk_size=512)

    Args:
        name: Typically pass __name__ so logs show the module path.

    Returns:
        A Loguru logger instance bound with the module name.
    """
    _configure_logger()
    # .bind() creates a child logger with extra context fields.
    # Every log message from this logger will include the module name.
    return _loguru_logger.bind(module=name)
