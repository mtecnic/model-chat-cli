"""Pytest configuration and fixtures."""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Get project root path."""
    return project_root


@pytest.fixture
def mock_server():
    """Mock server configuration."""
    return {
        "url": "http://localhost:11434",
        "type": "ollama",
        "ip": "127.0.0.1",
        "port": 11434,
        "models": ["llama2", "codellama"],
        "status": "healthy",
        "response_time": 0.05,
    }


@pytest.fixture
def sample_chat_history():
    """Sample chat history for testing."""
    return [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language."},
        {"role": "user", "content": "Tell me more about it."},
        {"role": "assistant", "content": "Python is known for its simplicity and readability."},
    ]
