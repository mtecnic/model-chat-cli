"""Context window management for Model Chat CLI."""
from typing import List, Dict, Optional, Tuple
from utils.token_estimator import estimate_tokens


# Common model context window sizes (in tokens)
MODEL_CONTEXT_WINDOWS = {
    # OpenAI models
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16384,

    # Anthropic Claude models
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-2": 100000,
    "claude-instant": 100000,

    # Open source models (typical defaults)
    "llama-2-7b": 4096,
    "llama-2-13b": 4096,
    "llama-2-70b": 4096,
    "mistral-7b": 8192,
    "mixtral-8x7b": 32768,
    "codellama": 16384,
    "phi-2": 2048,
    "gemma-7b": 8192,

    # Default for unknown models
    "default": 4096,
}


class ContextManager:
    """Manages context window limits and automatic history trimming."""

    def __init__(
        self,
        model: str,
        max_tokens: Optional[int] = None,
        reserve_tokens: int = 1024,
        warning_threshold: float = 0.8
    ):
        """Initialize context manager.

        Args:
            model: Model name
            max_tokens: Override context window size
            reserve_tokens: Tokens to reserve for response (default: 1024)
            warning_threshold: Warn when context usage exceeds this ratio (default: 0.8)
        """
        self.model = model
        self.reserve_tokens = reserve_tokens
        self.warning_threshold = warning_threshold

        # Determine context window size
        if max_tokens:
            self.max_tokens = max_tokens
        else:
            # Try to match model name to known context windows
            self.max_tokens = self._detect_context_window(model)

        # Calculate effective limit (minus reserve for response)
        self.effective_limit = self.max_tokens - self.reserve_tokens

    def _detect_context_window(self, model: str) -> int:
        """Detect context window size from model name.

        Args:
            model: Model name

        Returns:
            Context window size in tokens
        """
        model_lower = model.lower()

        # Check for exact matches first
        if model_lower in MODEL_CONTEXT_WINDOWS:
            return MODEL_CONTEXT_WINDOWS[model_lower]

        # Check for partial matches
        for known_model, context_size in MODEL_CONTEXT_WINDOWS.items():
            if known_model in model_lower:
                return context_size

        # Default fallback
        return MODEL_CONTEXT_WINDOWS["default"]

    def estimate_context_usage(self, history: List[Dict], system_prompt: str = "") -> int:
        """Estimate total tokens in current context.

        Args:
            history: Conversation history
            system_prompt: System prompt text

        Returns:
            Estimated token count
        """
        total_tokens = 0

        # Add system prompt tokens
        if system_prompt:
            total_tokens += estimate_tokens(system_prompt)

        # Add message tokens
        for msg in history:
            # Add role tokens (approximately 4 tokens per message for formatting)
            total_tokens += 4
            # Add content tokens
            total_tokens += estimate_tokens(msg.get("content", ""))

        return total_tokens

    def get_context_status(
        self,
        history: List[Dict],
        system_prompt: str = ""
    ) -> Dict:
        """Get current context window status.

        Args:
            history: Conversation history
            system_prompt: System prompt text

        Returns:
            Status dict with:
                - current_tokens: Current token count
                - max_tokens: Maximum allowed tokens
                - effective_limit: Limit minus reserve
                - usage_ratio: Current / effective_limit
                - remaining_tokens: Tokens remaining
                - status: "ok", "warning", or "critical"
        """
        current_tokens = self.estimate_context_usage(history, system_prompt)
        usage_ratio = current_tokens / self.effective_limit

        # Determine status
        if usage_ratio >= 1.0:
            status = "critical"
        elif usage_ratio >= self.warning_threshold:
            status = "warning"
        else:
            status = "ok"

        return {
            "current_tokens": current_tokens,
            "max_tokens": self.max_tokens,
            "effective_limit": self.effective_limit,
            "usage_ratio": usage_ratio,
            "remaining_tokens": max(0, self.effective_limit - current_tokens),
            "status": status,
        }

    def should_trim(self, history: List[Dict], system_prompt: str = "") -> bool:
        """Check if history should be trimmed.

        Args:
            history: Conversation history
            system_prompt: System prompt text

        Returns:
            True if trimming is needed
        """
        current_tokens = self.estimate_context_usage(history, system_prompt)
        return current_tokens >= self.effective_limit

    def trim_history(
        self,
        history: List[Dict],
        system_prompt: str = "",
        keep_recent: int = 10
    ) -> Tuple[List[Dict], int]:
        """Trim history to fit within context window.

        Keeps the most recent messages and removes older ones from the middle.

        Args:
            history: Conversation history
            system_prompt: System prompt text
            keep_recent: Number of recent messages to always keep

        Returns:
            Tuple of (trimmed_history, removed_count)
        """
        if len(history) <= keep_recent:
            return history, 0

        current_tokens = self.estimate_context_usage(history, system_prompt)

        if current_tokens < self.effective_limit:
            return history, 0

        # Keep most recent messages
        recent_messages = history[-keep_recent:]
        removed_count = 0

        # Calculate tokens for recent messages
        recent_tokens = self.estimate_context_usage(recent_messages, system_prompt)

        # Check if even recent messages fit
        if recent_tokens >= self.effective_limit:
            # Need more aggressive trimming
            keep_recent = max(2, keep_recent // 2)
            recent_messages = history[-keep_recent:]
            removed_count = len(history) - keep_recent
            return recent_messages, removed_count

        # Add older messages one by one until we approach limit
        remaining_budget = self.effective_limit - recent_tokens
        older_messages = []

        for i in range(len(history) - keep_recent - 1, -1, -1):
            msg = history[i]
            msg_tokens = estimate_tokens(msg.get("content", "")) + 4

            if msg_tokens < remaining_budget:
                older_messages.insert(0, msg)
                remaining_budget -= msg_tokens
            else:
                break

        removed_count = len(history) - len(older_messages) - keep_recent
        trimmed_history = older_messages + recent_messages

        return trimmed_history, removed_count

    def get_max_prompt_tokens(self, history: List[Dict], system_prompt: str = "") -> int:
        """Calculate maximum tokens available for next prompt.

        Args:
            history: Conversation history
            system_prompt: System prompt text

        Returns:
            Maximum tokens available for next user prompt
        """
        current_tokens = self.estimate_context_usage(history, system_prompt)
        remaining = self.effective_limit - current_tokens

        # Reserve some space for message formatting
        return max(0, remaining - 100)

    def format_context_info(self, history: List[Dict], system_prompt: str = "") -> str:
        """Format context status as a human-readable string.

        Args:
            history: Conversation history
            system_prompt: System prompt text

        Returns:
            Formatted status string
        """
        status = self.get_context_status(history, system_prompt)

        percentage = status["usage_ratio"] * 100
        bar_width = 20
        filled = int(bar_width * status["usage_ratio"])
        bar = "█" * filled + "░" * (bar_width - filled)

        # Color based on status
        if status["status"] == "critical":
            color = "red"
        elif status["status"] == "warning":
            color = "yellow"
        else:
            color = "green"

        return (
            f"[{color}]Context: {bar} {percentage:.1f}%[/{color}] "
            f"({status['current_tokens']:,}/{status['effective_limit']:,} tokens)"
        )
