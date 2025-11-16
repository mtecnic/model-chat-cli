"""Retry logic with exponential backoff for network operations."""
import asyncio
import logging
from typing import TypeVar, Callable, Optional
from functools import wraps

T = TypeVar('T')

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 16.0,
    exceptions: tuple = (Exception,)
) -> T:
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts (default: 4)
        base_delay: Initial delay in seconds (default: 2.0)
        max_delay: Maximum delay between retries (default: 16.0)
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e

            if attempt == max_retries:
                # Last attempt failed
                logger.error(f"All {max_retries} retries exhausted for {func.__name__}")
                raise

            # Calculate backoff delay: 2^attempt * base_delay, capped at max_delay
            delay = min(base_delay * (2 ** attempt), max_delay)

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {str(e)}. "
                f"Retrying in {delay:.1f}s..."
            )

            await asyncio.sleep(delay)

    # Should never reach here, but just in case
    raise last_exception


def with_retry(
    max_retries: int = 4,
    base_delay: float = 2.0,
    max_delay: float = 16.0,
    exceptions: tuple = (Exception,)
):
    """Decorator to add retry logic with exponential backoff to async functions.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to catch and retry

    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        async def fetch_data():
            async with httpx.AsyncClient() as client:
                response = await client.get("https://api.example.com/data")
                return response.json()
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async def call_func():
                return await func(*args, **kwargs)

            return await retry_with_backoff(
                call_func,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                exceptions=exceptions
            )
        return wrapper
    return decorator
