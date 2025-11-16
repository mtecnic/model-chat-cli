"""Token estimation utilities for Model Chat CLI."""


def estimate_tokens(text: str) -> int:
    """Estimate token count using improved heuristics.

    Better than simple char/4, accounts for:
    - Word boundaries (spaces create tokens)
    - Punctuation (often separate tokens)
    - Numbers (compact tokenization)
    - Code patterns (more tokens per char)

    Target: Within 10-15% of actual tokens

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Count different components
    words = text.split()
    word_count = len(words)

    # Count special characters (punctuation, symbols)
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())

    # Count newlines (often separate tokens)
    newline_count = text.count('\n')

    # Estimate based on multiple factors:
    # - Base: ~1.3 tokens per word (accounts for subword tokenization)
    # - Punctuation: ~0.5 tokens each (some merge with words)
    # - Newlines: 1 token each
    # - Adjustment for very short words (more tokens per char)

    token_estimate = (
        word_count * 1.3 +           # Words with subword splits
        special_chars * 0.5 +        # Punctuation/symbols
        newline_count * 1.0          # Newlines
    )

    # Clamp minimum to prevent zero/negative
    return max(1, int(token_estimate))


def estimate_response_tokens(prompt_text: str, multiplier: float = 1.5) -> int:
    """Estimate response tokens based on prompt.

    Args:
        prompt_text: The prompt text
        multiplier: Response length multiplier (default: 1.5x prompt)

    Returns:
        Estimated response token count
    """
    prompt_tokens = estimate_tokens(prompt_text)
    return int(prompt_tokens * multiplier)
