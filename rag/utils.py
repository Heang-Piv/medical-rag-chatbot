"""
Shared utilities — currently just lightweight logging setup.

Call get_logger(__name__) once per module. All loggers share a single
root handler (level from config.log_level, configured on first use) so
nothing needs to configure logging more than once.
"""

import logging

from config import config

_configured = False


def _configure_root_logger() -> None:
    """Set up the root handler once, at WARNING, so third-party libraries
    (httpx, sentence_transformers, chromadb, ...) stay quiet by default.
    Our own modules opt into config.log_level individually in get_logger()."""
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger for `name`, set to config.log_level regardless of the
    (quieter) root default — so our own pipeline events are always visible
    without turning on every third-party library's logging too."""
    _configure_root_logger()
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    return logger
