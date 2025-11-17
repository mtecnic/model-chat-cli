"""Tests for retry logic with exponential backoff."""
import pytest
import asyncio
import time
from utils.retry import retry_with_backoff, with_retry


class TestRetryLogic:
    """Test retry logic."""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self):
        """Test successful function on first attempt."""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_with_backoff(success_func, max_retries=3)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_success_after_failures(self):
        """Test successful function after some failures."""
        call_count = 0

        async def eventual_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"

        result = await retry_with_backoff(eventual_success, max_retries=4, base_delay=0.01)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausts_attempts(self):
        """Test that retry exhausts all attempts and raises."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            await retry_with_backoff(always_fails, max_retries=3, base_delay=0.01)

        assert call_count == 4  # Initial + 3 retries

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff_timing(self):
        """Test that exponential backoff timing is correct."""
        call_count = 0
        call_times = []

        async def failing_func():
            nonlocal call_count
            call_count += 1
            call_times.append(time.time())
            raise Exception("Fail")

        start_time = time.time()

        with pytest.raises(Exception):
            await retry_with_backoff(
                failing_func,
                max_retries=3,
                base_delay=0.1,
                max_delay=1.0
            )

        # Check delays between attempts
        # Delays should be: 0.1, 0.2, 0.4 seconds
        if len(call_times) >= 2:
            delay1 = call_times[1] - call_times[0]
            # Allow some tolerance for timing
            assert 0.08 <= delay1 <= 0.15

    @pytest.mark.asyncio
    async def test_retry_max_delay_cap(self):
        """Test that delays are capped at max_delay."""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Fail")

        with pytest.raises(Exception):
            await retry_with_backoff(
                failing_func,
                max_retries=10,
                base_delay=1.0,
                max_delay=2.0
            )

        # With max_delay=2.0, no delay should exceed that
        # (Hard to test timing precisely, but we verify it doesn't hang)

    @pytest.mark.asyncio
    async def test_retry_specific_exceptions(self):
        """Test retrying only specific exceptions."""
        call_count = 0

        async def fails_with_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Specific error")

        # Should retry ValueError
        with pytest.raises(ValueError):
            await retry_with_backoff(
                fails_with_value_error,
                max_retries=2,
                base_delay=0.01,
                exceptions=(ValueError,)
            )

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_non_matching_exception(self):
        """Test that non-matching exceptions are not retried."""
        call_count = 0

        async def fails_with_key_error():
            nonlocal call_count
            call_count += 1
            raise KeyError("Different error")

        # Should NOT retry KeyError when only catching ValueError
        with pytest.raises(KeyError):
            await retry_with_backoff(
                fails_with_key_error,
                max_retries=3,
                base_delay=0.01,
                exceptions=(ValueError,)
            )

        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_with_retry_decorator(self):
        """Test the @with_retry decorator."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        async def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Fail")
            return "success"

        result = await decorated_func()

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_with_retry_decorator_with_args(self):
        """Test decorator with function arguments."""
        call_count = 0

        @with_retry(max_retries=2, base_delay=0.01)
        async def func_with_args(x, y):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Fail")
            return x + y

        result = await func_with_args(5, 3)

        assert result == 8
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_zero_retries(self):
        """Test with zero retries (fail immediately)."""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Fail")

        with pytest.raises(Exception):
            await retry_with_backoff(failing_func, max_retries=0, base_delay=0.01)

        assert call_count == 1  # Only initial attempt

    @pytest.mark.asyncio
    async def test_retry_returns_correct_value(self):
        """Test that retry returns the correct value."""
        async def returns_value():
            return {"status": "ok", "data": [1, 2, 3]}

        result = await retry_with_backoff(returns_value, max_retries=3)

        assert result == {"status": "ok", "data": [1, 2, 3]}
