"""Tests for conversation management."""
import pytest
import json
import tempfile
from pathlib import Path
from utils.conversation_manager import ConversationManager


@pytest.fixture
def temp_conv_dir():
    """Create temporary conversation directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def conv_manager(temp_conv_dir):
    """Create conversation manager with temp directory."""
    return ConversationManager(conversations_dir=temp_conv_dir)


@pytest.fixture
def sample_history():
    """Sample conversation history."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]


@pytest.fixture
def sample_server():
    """Sample server configuration."""
    return {
        "url": "http://localhost:11434",
        "type": "ollama",
        "ip": "127.0.0.1",
        "port": 11434,
    }


class TestConversationManager:
    """Test conversation manager."""

    def test_initialization(self, temp_conv_dir):
        """Test conversation manager initialization."""
        cm = ConversationManager(conversations_dir=temp_conv_dir)

        assert cm.conversations_dir == temp_conv_dir
        assert temp_conv_dir.exists()

    def test_save_conversation(self, conv_manager, sample_history, sample_server):
        """Test saving a conversation."""
        conv_id = conv_manager.save_conversation(
            history=sample_history,
            server=sample_server,
            model="llama2",
        )

        assert isinstance(conv_id, str)
        assert len(conv_id) > 0

        # Check file was created
        conv_file = conv_manager.conversations_dir / f"{conv_id}.json"
        assert conv_file.exists()

    def test_save_conversation_with_name(self, conv_manager, sample_history, sample_server):
        """Test saving conversation with custom name."""
        conv_id = conv_manager.save_conversation(
            history=sample_history,
            server=sample_server,
            model="llama2",
            name="my_test_conversation",
        )

        assert "my_test_conversation" in conv_id

    def test_save_conversation_with_metadata(self, conv_manager, sample_history, sample_server):
        """Test saving conversation with metadata."""
        metadata = {"tps_samples": [50, 60, 55], "avg_tps": 55.0}

        conv_id = conv_manager.save_conversation(
            history=sample_history,
            server=sample_server,
            model="llama2",
            metadata=metadata,
        )

        # Load and verify metadata
        conv = conv_manager.load_conversation(conv_id)
        assert conv["metadata"] == metadata

    def test_load_conversation(self, conv_manager, sample_history, sample_server):
        """Test loading a conversation."""
        # Save first
        conv_id = conv_manager.save_conversation(
            history=sample_history,
            server=sample_server,
            model="llama2",
        )

        # Load
        loaded = conv_manager.load_conversation(conv_id)

        assert loaded is not None
        assert loaded["history"] == sample_history
        assert loaded["model"] == "llama2"
        assert loaded["server"]["url"] == sample_server["url"]

    def test_load_nonexistent_conversation(self, conv_manager):
        """Test loading a conversation that doesn't exist."""
        loaded = conv_manager.load_conversation("nonexistent_id")
        assert loaded is None

    def test_list_conversations_empty(self, conv_manager):
        """Test listing conversations when none exist."""
        conversations = conv_manager.list_conversations()
        assert conversations == []

    def test_list_conversations(self, conv_manager, sample_history, sample_server):
        """Test listing conversations."""
        # Save multiple conversations
        conv_manager.save_conversation(sample_history, sample_server, "model1")
        conv_manager.save_conversation(sample_history, sample_server, "model2")
        conv_manager.save_conversation(sample_history, sample_server, "model3")

        conversations = conv_manager.list_conversations()

        assert len(conversations) == 3
        assert all("id" in c for c in conversations)
        assert all("model" in c for c in conversations)
        assert all("message_count" in c for c in conversations)

    def test_list_conversations_limit(self, conv_manager, sample_history, sample_server):
        """Test listing conversations with limit."""
        # Save multiple conversations
        for i in range(5):
            conv_manager.save_conversation(sample_history, sample_server, f"model{i}")

        conversations = conv_manager.list_conversations(limit=3)

        assert len(conversations) == 3

    def test_list_conversations_filter_by_model(self, conv_manager, sample_history, sample_server):
        """Test filtering conversations by model."""
        conv_manager.save_conversation(sample_history, sample_server, "llama2")
        conv_manager.save_conversation(sample_history, sample_server, "llama2")
        conv_manager.save_conversation(sample_history, sample_server, "gpt-4")

        conversations = conv_manager.list_conversations(model="llama2")

        assert len(conversations) == 2
        assert all(c["model"] == "llama2" for c in conversations)

    def test_delete_conversation(self, conv_manager, sample_history, sample_server):
        """Test deleting a conversation."""
        # Save
        conv_id = conv_manager.save_conversation(sample_history, sample_server, "llama2")

        # Verify exists
        assert conv_manager.load_conversation(conv_id) is not None

        # Delete
        result = conv_manager.delete_conversation(conv_id)
        assert result is True

        # Verify deleted
        assert conv_manager.load_conversation(conv_id) is None

    def test_delete_nonexistent_conversation(self, conv_manager):
        """Test deleting a conversation that doesn't exist."""
        result = conv_manager.delete_conversation("nonexistent_id")
        assert result is False

    def test_export_conversation_json(self, conv_manager, sample_history, sample_server, temp_conv_dir):
        """Test exporting conversation to JSON."""
        # Save
        conv_id = conv_manager.save_conversation(sample_history, sample_server, "llama2")

        # Export
        output_path = temp_conv_dir / "export.json"
        result = conv_manager.export_conversation(conv_id, output_path, format="json")

        assert result is True
        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            data = json.load(f)
            assert data["history"] == sample_history

    def test_export_conversation_txt(self, conv_manager, sample_history, sample_server, temp_conv_dir):
        """Test exporting conversation to text."""
        conv_id = conv_manager.save_conversation(sample_history, sample_server, "llama2")

        output_path = temp_conv_dir / "export.txt"
        result = conv_manager.export_conversation(conv_id, output_path, format="txt")

        assert result is True
        assert output_path.exists()

        # Verify content
        content = output_path.read_text()
        assert "USER:" in content
        assert "ASSISTANT:" in content
        assert "Hello" in content

    def test_export_conversation_md(self, conv_manager, sample_history, sample_server, temp_conv_dir):
        """Test exporting conversation to markdown."""
        conv_id = conv_manager.save_conversation(sample_history, sample_server, "llama2")

        output_path = temp_conv_dir / "export.md"
        result = conv_manager.export_conversation(conv_id, output_path, format="md")

        assert result is True
        assert output_path.exists()

        # Verify content
        content = output_path.read_text()
        assert "###" in content  # Markdown headers
        assert "Hello" in content

    def test_export_nonexistent_conversation(self, conv_manager, temp_conv_dir):
        """Test exporting a conversation that doesn't exist."""
        output_path = temp_conv_dir / "export.json"
        result = conv_manager.export_conversation("nonexistent", output_path, format="json")

        assert result is False
        assert not output_path.exists()

    def test_sanitize_filename(self, conv_manager):
        """Test filename sanitization."""
        unsafe_name = 'my/conversation\\with:special*chars?'
        safe_name = conv_manager._sanitize_filename(unsafe_name)

        assert "/" not in safe_name
        assert "\\" not in safe_name
        assert ":" not in safe_name
        assert "*" not in safe_name
        assert "?" not in safe_name

    def test_sanitize_filename_length(self, conv_manager):
        """Test filename length limit."""
        long_name = "a" * 200
        safe_name = conv_manager._sanitize_filename(long_name)

        assert len(safe_name) <= 100

    def test_duplicate_name_handling(self, conv_manager, sample_history, sample_server):
        """Test handling of duplicate conversation names."""
        # Save with same name twice
        conv_id1 = conv_manager.save_conversation(
            sample_history, sample_server, "llama2", name="duplicate"
        )
        conv_id2 = conv_manager.save_conversation(
            sample_history, sample_server, "llama2", name="duplicate"
        )

        # Should create different IDs
        assert conv_id1 != conv_id2

        # Both should exist
        assert conv_manager.load_conversation(conv_id1) is not None
        assert conv_manager.load_conversation(conv_id2) is not None

    def test_conversation_roundtrip(self, conv_manager, sample_history, sample_server):
        """Test save and load roundtrip preserves data."""
        metadata = {"test_key": "test_value"}

        # Save
        conv_id = conv_manager.save_conversation(
            history=sample_history,
            server=sample_server,
            model="llama2",
            metadata=metadata,
        )

        # Load
        loaded = conv_manager.load_conversation(conv_id)

        # Verify all data preserved
        assert loaded["history"] == sample_history
        assert loaded["model"] == "llama2"
        assert loaded["server"]["url"] == sample_server["url"]
        assert loaded["metadata"] == metadata
        assert "created_at" in loaded
        assert "message_count" in loaded
        assert loaded["message_count"] == len(sample_history)
