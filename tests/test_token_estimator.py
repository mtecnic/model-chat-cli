"""Tests for token estimation utilities."""
import pytest
from utils.token_estimator import estimate_tokens, estimate_response_tokens


class TestTokenEstimator:
    """Test token estimation functions."""

    def test_estimate_tokens_empty_string(self):
        """Test token estimation with empty string."""
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_estimate_tokens_simple_text(self):
        """Test token estimation with simple text."""
        text = "Hello world"
        tokens = estimate_tokens(text)

        # Should estimate ~2 words * 1.3 = ~2-3 tokens
        assert tokens >= 2
        assert tokens <= 5

    def test_estimate_tokens_with_punctuation(self):
        """Test token estimation with punctuation."""
        text = "Hello, world! How are you?"
        tokens = estimate_tokens(text)

        # 5 words * 1.3 + 3 punctuation * 0.5 = ~8 tokens
        assert tokens >= 6
        assert tokens <= 12

    def test_estimate_tokens_with_newlines(self):
        """Test token estimation with newlines."""
        text = "Line 1\nLine 2\nLine 3"
        tokens = estimate_tokens(text)

        # 6 words * 1.3 + 2 newlines * 1.0 = ~9-10 tokens
        assert tokens >= 8
        assert tokens <= 12

    def test_estimate_tokens_code(self):
        """Test token estimation with code."""
        text = """def hello():
    print("Hello, world!")
    return True"""

        tokens = estimate_tokens(text)

        # Code has more special chars, should be higher
        assert tokens >= 10
        assert tokens <= 30

    def test_estimate_tokens_consistency(self):
        """Test that estimation is consistent."""
        text = "This is a test message with some content."

        tokens1 = estimate_tokens(text)
        tokens2 = estimate_tokens(text)

        assert tokens1 == tokens2

    def test_estimate_tokens_length_correlation(self):
        """Test that longer text has more tokens."""
        short = "Hello"
        medium = "Hello world, how are you today?"
        long = "Hello world, how are you today? I hope everything is going well for you."

        tokens_short = estimate_tokens(short)
        tokens_medium = estimate_tokens(medium)
        tokens_long = estimate_tokens(long)

        assert tokens_short < tokens_medium < tokens_long

    def test_estimate_response_tokens_default(self):
        """Test response token estimation with default multiplier."""
        prompt = "What is the meaning of life?"
        prompt_tokens = estimate_tokens(prompt)
        response_tokens = estimate_response_tokens(prompt)

        # Default multiplier is 1.5
        assert response_tokens == int(prompt_tokens * 1.5)

    def test_estimate_response_tokens_custom_multiplier(self):
        """Test response token estimation with custom multiplier."""
        prompt = "Tell me a story."
        prompt_tokens = estimate_tokens(prompt)
        response_tokens = estimate_response_tokens(prompt, multiplier=2.0)

        assert response_tokens == int(prompt_tokens * 2.0)

    def test_estimate_tokens_minimum(self):
        """Test that token estimation has minimum of 1."""
        # Even for very short text, should return at least 1
        text = "a"
        tokens = estimate_tokens(text)

        assert tokens >= 1

    def test_estimate_tokens_accuracy_heuristic(self):
        """Test that estimation is within reasonable bounds.

        Target: Within 10-15% of actual tokens (hard to test without real tokenizer)
        For this test, we verify the formula makes sense.
        """
        # Known example: "The quick brown fox" should be ~4-5 tokens
        text = "The quick brown fox"
        tokens = estimate_tokens(text)

        # 4 words * 1.3 = ~5.2 tokens
        assert tokens >= 4
        assert tokens <= 7

    def test_estimate_tokens_unicode(self):
        """Test token estimation with unicode characters."""
        text = "Hello 世界 🌍"
        tokens = estimate_tokens(text)

        # Should handle unicode gracefully
        assert tokens >= 2
        assert tokens <= 10
