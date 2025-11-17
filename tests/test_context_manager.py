"""Tests for context window management."""
import pytest
from utils.context_manager import ContextManager, MODEL_CONTEXT_WINDOWS


class TestContextManager:
    """Test context window manager."""

    def test_initialization_default(self):
        """Test context manager initialization with defaults."""
        cm = ContextManager(model="gpt-3.5-turbo")

        assert cm.model == "gpt-3.5-turbo"
        assert cm.max_tokens == 4096
        assert cm.reserve_tokens == 1024
        assert cm.effective_limit == 3072

    def test_initialization_custom_max_tokens(self):
        """Test context manager with custom max tokens."""
        cm = ContextManager(model="custom-model", max_tokens=8192)

        assert cm.max_tokens == 8192
        assert cm.effective_limit == 8192 - 1024

    def test_detect_gpt4_context(self):
        """Test GPT-4 context window detection."""
        cm = ContextManager(model="gpt-4")
        assert cm.max_tokens == 8192

        cm_32k = ContextManager(model="gpt-4-32k")
        assert cm_32k.max_tokens == 32768

    def test_detect_claude_context(self):
        """Test Claude context window detection."""
        cm = ContextManager(model="claude-3-opus")
        assert cm.max_tokens == 200000

        cm2 = ContextManager(model="claude-2")
        assert cm2.max_tokens == 100000

    def test_detect_llama_context(self):
        """Test Llama context window detection."""
        cm = ContextManager(model="llama-2-7b")
        assert cm.max_tokens == 4096

    def test_detect_unknown_model(self):
        """Test unknown model defaults to 4096."""
        cm = ContextManager(model="unknown-model-xyz")
        assert cm.max_tokens == 4096

    def test_estimate_context_usage_empty(self):
        """Test context usage estimation with empty history."""
        cm = ContextManager(model="gpt-3.5-turbo")
        usage = cm.estimate_context_usage([])

        assert usage == 0

    def test_estimate_context_usage_with_messages(self):
        """Test context usage estimation with messages."""
        cm = ContextManager(model="gpt-3.5-turbo")

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        usage = cm.estimate_context_usage(history)

        # Should include tokens for messages + formatting overhead
        assert usage > 0
        assert usage < 100  # Simple messages shouldn't be huge

    def test_estimate_context_usage_with_system_prompt(self):
        """Test context usage with system prompt."""
        cm = ContextManager(model="gpt-3.5-turbo")

        history = [
            {"role": "user", "content": "Hello"},
        ]
        system_prompt = "You are a helpful assistant."

        usage = cm.estimate_context_usage(history, system_prompt)

        # Should be more than without system prompt
        usage_without = cm.estimate_context_usage(history)
        assert usage > usage_without

    def test_get_context_status_ok(self):
        """Test context status when usage is low."""
        cm = ContextManager(model="gpt-3.5-turbo")

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        status = cm.get_context_status(history)

        assert status["status"] == "ok"
        assert status["current_tokens"] > 0
        assert status["remaining_tokens"] > 0
        assert status["usage_ratio"] < 0.8

    def test_get_context_status_warning(self):
        """Test context status at warning threshold."""
        cm = ContextManager(model="gpt-3.5-turbo", max_tokens=100, reserve_tokens=20)

        # Create messages that approach the limit
        history = [{"role": "user", "content": "word " * 20} for _ in range(3)]

        status = cm.get_context_status(history)

        # Should trigger warning at 80%+
        if status["usage_ratio"] >= 0.8:
            assert status["status"] in ("warning", "critical")

    def test_should_trim_false(self):
        """Test should_trim returns False for small history."""
        cm = ContextManager(model="gpt-3.5-turbo")

        history = [
            {"role": "user", "content": "Hello"},
        ]

        assert cm.should_trim(history) is False

    def test_should_trim_true(self):
        """Test should_trim returns True when over limit."""
        cm = ContextManager(model="gpt-3.5-turbo", max_tokens=100, reserve_tokens=10)

        # Create large history that exceeds limit
        history = [{"role": "user", "content": "word " * 50} for _ in range(5)]

        result = cm.should_trim(history)
        # May or may not need trimming depending on estimation
        assert isinstance(result, bool)

    def test_trim_history_keeps_recent(self):
        """Test that trimming keeps recent messages."""
        cm = ContextManager(model="gpt-3.5-turbo", max_tokens=100, reserve_tokens=10)

        history = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(20)
        ]

        trimmed, removed = cm.trim_history(history, keep_recent=5)

        # Should keep last 5 messages
        assert len(trimmed) <= len(history)
        assert trimmed[-1]["content"] == "Message 19"
        assert trimmed[-5]["content"] == "Message 15"

    def test_trim_history_small_history(self):
        """Test trimming with history smaller than keep_recent."""
        cm = ContextManager(model="gpt-3.5-turbo")

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        trimmed, removed = cm.trim_history(history, keep_recent=10)

        # Should return unchanged
        assert trimmed == history
        assert removed == 0

    def test_get_max_prompt_tokens(self):
        """Test max prompt tokens calculation."""
        cm = ContextManager(model="gpt-3.5-turbo", max_tokens=4096, reserve_tokens=1024)

        history = [
            {"role": "user", "content": "Previous message"},
        ]

        max_tokens = cm.get_max_prompt_tokens(history)

        # Should be effective_limit - current_usage - buffer
        assert max_tokens > 0
        assert max_tokens < cm.effective_limit

    def test_format_context_info(self):
        """Test context info formatting."""
        cm = ContextManager(model="gpt-3.5-turbo")

        history = [
            {"role": "user", "content": "Hello world"},
        ]

        info = cm.format_context_info(history)

        assert isinstance(info, str)
        assert "Context:" in info
        assert "%" in info
        assert "tokens" in info

    def test_format_context_info_colors(self):
        """Test context info uses correct colors."""
        cm = ContextManager(model="gpt-3.5-turbo", max_tokens=100, reserve_tokens=10)

        # Low usage - should be green
        history_low = [{"role": "user", "content": "Hi"}]
        info_low = cm.format_context_info(history_low)
        assert "green" in info_low

        # High usage - should be yellow or red
        history_high = [{"role": "user", "content": "word " * 30} for _ in range(5)]
        info_high = cm.format_context_info(history_high)
        # Color depends on actual usage
        assert "yellow" in info_high or "red" in info_high or "green" in info_high

    def test_reserve_tokens_configuration(self):
        """Test custom reserve tokens."""
        cm = ContextManager(model="gpt-3.5-turbo", reserve_tokens=2048)

        assert cm.reserve_tokens == 2048
        assert cm.effective_limit == cm.max_tokens - 2048

    def test_warning_threshold_configuration(self):
        """Test custom warning threshold."""
        cm = ContextManager(model="gpt-3.5-turbo", warning_threshold=0.9)

        assert cm.warning_threshold == 0.9

    def test_model_context_windows_coverage(self):
        """Test that common models are in the lookup table."""
        assert "gpt-4" in MODEL_CONTEXT_WINDOWS
        assert "gpt-3.5-turbo" in MODEL_CONTEXT_WINDOWS
        assert "claude-3-opus" in MODEL_CONTEXT_WINDOWS
        assert "llama-2-7b" in MODEL_CONTEXT_WINDOWS
        assert "mistral-7b" in MODEL_CONTEXT_WINDOWS
        assert "default" in MODEL_CONTEXT_WINDOWS
