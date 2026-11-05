"""Scoped console progress and spawn-safe forwarding using standard logging."""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from logging.handlers import QueueHandler, QueueListener
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue
from time import perf_counter

LOGGER = logging.getLogger("cosmosalpha")
LOGGER.addHandler(logging.NullHandler())
_CONTEXT: ContextVar[dict[str, str] | None] = ContextVar(
    "cosmosalpha_log_context", default=None
)
_HANDLER_NAME = "cosmosalpha.console"


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "progress_context"):
            record.progress_context = "".join(
                f" {name}={value}"
                for name, value in (_CONTEXT.get() or {}).items()
                if name != "step" or record.levelno != logging.INFO
            )
        return True


class _ConsoleFormatter(logging.Formatter):
    def __init__(self, encoding: str | None) -> None:
        super().__init__(
            "%(asctime)s %(levelname)s%(progress_context)s %(message)s",
            datefmt="%H:%M:%S",
        )
        self._encoding = encoding or "utf-8"

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        text = "".join(
            (
                (f"\\x{ord(char):02x}" if ord(char) < 256 else f"\\u{ord(char):04x}")
                if not char.isprintable()
                else char
            )
            for char in text
        )
        return text.encode(self._encoding, errors="backslashreplace").decode(
            self._encoding
        )


@contextmanager
def log_context(**values: str) -> Iterator[None]:
    """Bind progress identity in this thread and restore it on every exit."""
    token = _CONTEXT.set({**(_CONTEXT.get() or {}), **values})
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def contextual[**P, T](function: Callable[P, T]) -> Callable[P, T]:
    """Bind existing keyword run/cycle identifiers without changing call inputs."""

    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        values = {
            name.removesuffix("_id"): value
            for name in ("run_id", "cycle_id")
            if isinstance(value := kwargs.get(name), str)
        }
        with log_context(**values):
            return function(*args, **kwargs)

    return wrapped


def error_code(error: BaseException) -> str:
    """Expose a stable error code without printing provider bodies or secrets."""
    prefix = str(error).partition(":")[0]
    return (
        prefix
        if re.fullmatch(
            r"(?:LLM|GRAPH|MCTS|WORKFLOW|CONFIG|DATA)_[A-Z0-9_]{1,70}", prefix
        )
        else type(error).__name__
    )


@contextmanager
def progress_stage(
    label: str,
    *,
    level: int = logging.INFO,
    report_completion: bool = True,
) -> Iterator[None]:
    """Report an existing operation without changing its result or exception."""
    started = perf_counter()
    LOGGER.log(level, "%s started", label)
    try:
        yield
    except Exception as error:
        LOGGER.warning(
            "%s failed error=%s elapsed=%.2fs",
            label,
            error_code(error),
            perf_counter() - started,
        )
        raise
    else:
        if report_completion:
            LOGGER.log(
                level, "%s completed elapsed=%.2fs", label, perf_counter() - started
            )


@contextmanager
def console_logging(level: str) -> Iterator[None]:
    """Enable CLI-owned stderr logging without changing application handlers."""
    previous_level, previous_propagate = LOGGER.level, LOGGER.propagate
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setLevel(level)
    handler.addFilter(_ContextFilter())
    handler.setFormatter(_ConsoleFormatter(getattr(sys.stderr, "encoding", None)))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False
    try:
        yield
    finally:
        LOGGER.removeHandler(handler)
        handler.close()
        LOGGER.setLevel(previous_level)
        LOGGER.propagate = previous_propagate


def configure_worker_logging(queue: Queue | None, level: int) -> None:
    """Forward process-local records, including context, to the parent console."""
    if queue is None:
        return
    handler = QueueHandler(queue)
    handler.addFilter(_ContextFilter())
    LOGGER.addHandler(handler)
    LOGGER.setLevel(level)
    LOGGER.propagate = False


@contextmanager
def forward_worker_logs(context: BaseContext) -> Iterator[tuple[Queue | None, int]]:
    """Drain child logs until workers exit, then release queue resources."""
    handlers = tuple(h for h in LOGGER.handlers if h.name == _HANDLER_NAME)
    level = LOGGER.getEffectiveLevel()
    if not handlers:
        yield None, level
        return
    queue = context.Queue()
    listener = QueueListener(queue, *handlers, respect_handler_level=True)
    listener.start()
    try:
        yield queue, level
    finally:
        listener.stop()
        queue.close()
        queue.join_thread()
