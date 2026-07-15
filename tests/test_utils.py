"""Tests for the lightweight logging setup in rag/utils.py (M10)."""

import logging

from rag.utils import get_logger


def test_get_logger_returns_a_logger_with_the_given_name() -> None:
    logger = get_logger("some.module.name")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "some.module.name"


def test_get_logger_is_idempotent_across_calls() -> None:
    # Calling repeatedly (as every module does at import time) must not
    # raise or duplicate handlers.
    get_logger("a")
    get_logger("b")
    root_handlers_after_first = len(logging.getLogger().handlers)
    get_logger("c")
    assert len(logging.getLogger().handlers) == root_handlers_after_first
