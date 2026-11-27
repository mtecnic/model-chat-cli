"""Ctrl+C handling that works with asyncio.

Since Python 3.11, asyncio.run() turns SIGINT into a cancellation of the
*main* task instead of raising KeyboardInterrupt, so `except KeyboardInterrupt`
around awaits never fires and blocking prompts ignore Ctrl+C until Enter.

The app restores Python's default SIGINT handler (so synchronous prompts
raise KeyboardInterrupt as usual) and wraps long-running async operations in
run_interruptible(), which routes Ctrl+C to cancelling just that operation.
"""
import asyncio
import signal
from typing import Any, Awaitable, Tuple


def install_default_sigint() -> None:
    """Make Ctrl+C raise KeyboardInterrupt (undoes asyncio.run's handler)."""
    signal.signal(signal.SIGINT, signal.default_int_handler)


async def run_interruptible(awaitable: Awaitable) -> Tuple[Any, bool]:
    """Await `awaitable`, letting Ctrl+C cancel it instead of the whole app.

    Returns (result, interrupted). If interrupted and the operation did not
    swallow the cancellation, result is None. Do not nest these calls.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(awaitable)
    interrupted = False

    def on_sigint():
        nonlocal interrupted
        interrupted = True
        task.cancel()

    try:
        loop.add_signal_handler(signal.SIGINT, on_sigint)
        installed = True
    except (NotImplementedError, RuntimeError, ValueError):
        installed = False  # e.g. Windows / non-main thread: KeyboardInterrupt still works

    try:
        try:
            result = await task
        except asyncio.CancelledError:
            if interrupted:
                return None, True
            raise
        return result, interrupted
    finally:
        if installed:
            # Restores signal.default_int_handler for SIGINT
            loop.remove_signal_handler(signal.SIGINT)
