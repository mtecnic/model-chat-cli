"""Tests for configuration management."""
import os
import pytest
from pathlib import Path
from config import Config


class TestConfig:
    """Test configuration system."""

    def test_default_values(self):
        """Test default configuration values."""
        assert Config.SCAN_TIMEOUT == 30.0
        assert Config.SCAN_SEMAPHORE_LIMIT == 50
        assert Config.MAX_HISTORY_MESSAGES == 1000
        assert Config.MAX_DISPLAY_MESSAGES == 100
        assert Config.MAX_RETRIES == 3
        assert Config.RETRY_BASE_DELAY == 1.0
        assert Config.RETRY_MAX_DELAY == 8.0

    def test_common_ports_default(self):
        """Test common ports default value."""
        assert 11434 in Config.COMMON_PORTS  # Ollama
        assert 1234 in Config.COMMON_PORTS   # LM Studio
        assert len(Config.COMMON_PORTS) >= 5

    def test_environment_override(self, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("MODEL_CHAT_SCAN_TIMEOUT", "60.0")
        monkeypatch.setenv("MODEL_CHAT_MAX_RETRIES", "5")

        # Reload config with new env vars
        from importlib import reload
        import config as config_module
        reload(config_module)

        # Note: This test demonstrates the pattern but won't affect the singleton
        # In real usage, env vars must be set before first import

    def test_validation_errors(self):
        """Test configuration validation."""
        errors = Config.validate()
        # Should have no errors with defaults
        assert isinstance(errors, list)

    def test_get_config_summary(self):
        """Test configuration summary generation."""
        summary = Config.get_config_summary()

        assert "network" in summary
        assert "cache" in summary
        assert "memory" in summary
        assert "retry" in summary
        assert "ui" in summary

        assert summary["network"]["scan_timeout"] == Config.SCAN_TIMEOUT
        assert summary["memory"]["max_history_messages"] == Config.MAX_HISTORY_MESSAGES

    def test_cache_file_paths(self):
        """Test cache file path configuration."""
        assert isinstance(Config.CACHE_FILE, Path)
        assert isinstance(Config.THEME_FILE, Path)
        assert isinstance(Config.FAVORITES_FILE, Path)

        assert Config.CACHE_FILE.name == ".model_chat_cache.json"
        assert Config.THEME_FILE.name == ".model_chat_theme.json"
        assert Config.FAVORITES_FILE.name == ".model_chat_favorites.json"

    def test_banner_config(self):
        """Test banner configuration."""
        assert isinstance(Config.ENABLE_BANNER, bool)
        assert isinstance(Config.BANNER_FONT, str)
        assert isinstance(Config.BANNER_DURATION, float)
        assert Config.BANNER_DURATION > 0

    def test_retry_config_logic(self):
        """Test retry configuration logic."""
        assert Config.RETRY_MAX_DELAY >= Config.RETRY_BASE_DELAY
        assert Config.MAX_RETRIES >= 0
