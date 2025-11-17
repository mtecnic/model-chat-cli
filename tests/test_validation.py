"""Tests for input validation and security."""
import pytest
import tempfile
from pathlib import Path
from utils.validation import InputValidator, ValidationError


class TestInputValidator:
    """Test input validation."""

    def test_validate_message_valid(self):
        """Test validating a valid message."""
        message = "Hello, this is a test message."
        result = InputValidator.validate_message(message)
        assert result == message

    def test_validate_message_empty(self):
        """Test validating empty message."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            InputValidator.validate_message("")

        with pytest.raises(ValidationError, match="cannot be empty"):
            InputValidator.validate_message("   ")

    def test_validate_message_too_long(self):
        """Test validating overly long message."""
        long_message = "a" * (InputValidator.MAX_MESSAGE_LENGTH + 1)

        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_message(long_message)

    def test_validate_message_not_string(self):
        """Test validating non-string message."""
        with pytest.raises(ValidationError, match="must be a string"):
            InputValidator.validate_message(123)

        with pytest.raises(ValidationError, match="must be a string"):
            InputValidator.validate_message(None)

    def test_validate_system_prompt_valid(self):
        """Test validating valid system prompt."""
        prompt = "You are a helpful assistant."
        result = InputValidator.validate_system_prompt(prompt)
        assert result == prompt

    def test_validate_system_prompt_too_long(self):
        """Test validating overly long system prompt."""
        long_prompt = "a" * (InputValidator.MAX_SYSTEM_PROMPT_LENGTH + 1)

        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_system_prompt(long_prompt)

    def test_validate_filename_valid(self):
        """Test validating valid filename."""
        filename = "conversation_20240115.json"
        result = InputValidator.validate_filename(filename)
        assert result == filename

    def test_validate_filename_path_traversal(self):
        """Test detecting path traversal."""
        with pytest.raises(ValidationError, match="Path traversal"):
            InputValidator.validate_filename("../etc/passwd")

        with pytest.raises(ValidationError, match="Path traversal"):
            InputValidator.validate_filename("..\\windows\\system32")

    def test_validate_filename_invalid_chars(self):
        """Test detecting invalid characters."""
        invalid_names = [
            "file<test>.txt",
            "file|test.txt",
            'file"test.txt',
            "file?test.txt",
            "file*test.txt",
        ]

        for name in invalid_names:
            with pytest.raises(ValidationError, match="invalid characters"):
                InputValidator.validate_filename(name)

    def test_validate_filename_dangerous_extension(self):
        """Test detecting dangerous file extensions."""
        dangerous = [
            "malware.exe",
            "script.bat",
            "program.sh",
            "virus.dll",
        ]

        for filename in dangerous:
            with pytest.raises(ValidationError, match="Dangerous file extension"):
                InputValidator.validate_filename(filename)

    def test_validate_conversation_name_valid(self):
        """Test validating valid conversation name."""
        name = "My Important Conversation"
        result = InputValidator.validate_conversation_name(name)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_validate_conversation_name_sanitizes(self):
        """Test that conversation name is sanitized."""
        name = 'Conversation<with>invalid:chars'
        result = InputValidator.validate_conversation_name(name)

        # Should replace invalid chars with underscore
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result

    def test_validate_conversation_name_too_long(self):
        """Test conversation name length limit."""
        long_name = "a" * (InputValidator.MAX_CONV_NAME_LENGTH + 1)

        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_conversation_name(long_name)

    def test_validate_port_valid(self):
        """Test validating valid port numbers."""
        assert InputValidator.validate_port(80) == 80
        assert InputValidator.validate_port(443) == 443
        assert InputValidator.validate_port(8080) == 8080
        assert InputValidator.validate_port(65535) == 65535

    def test_validate_port_invalid(self):
        """Test validating invalid port numbers."""
        with pytest.raises(ValidationError, match="between 1 and 65535"):
            InputValidator.validate_port(0)

        with pytest.raises(ValidationError, match="between 1 and 65535"):
            InputValidator.validate_port(65536)

        with pytest.raises(ValidationError, match="between 1 and 65535"):
            InputValidator.validate_port(-1)

        with pytest.raises(ValidationError, match="must be an integer"):
            InputValidator.validate_port("8080")

    def test_validate_url_valid(self):
        """Test validating valid URLs."""
        urls = [
            "http://localhost:8080",
            "https://example.com",
            "http://192.168.1.1:11434",
        ]

        for url in urls:
            result = InputValidator.validate_url(url)
            assert result == url

    def test_validate_url_invalid_protocol(self):
        """Test detecting invalid URL protocols."""
        with pytest.raises(ValidationError, match="must start with http"):
            InputValidator.validate_url("ftp://example.com")

        with pytest.raises(ValidationError, match="must start with http"):
            InputValidator.validate_url("javascript:alert(1)")

    def test_validate_url_invalid_chars(self):
        """Test detecting invalid characters in URL."""
        with pytest.raises(ValidationError, match="invalid characters"):
            InputValidator.validate_url("http://example.com/<script>")

    def test_check_disk_space(self, tmp_path):
        """Test disk space checking."""
        # Should pass for small requirement
        result = InputValidator.check_disk_space(tmp_path, required_bytes=1024)
        assert result is True

    def test_validate_file_size(self, tmp_path):
        """Test file size validation."""
        # Create a small file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, world!")

        size = InputValidator.validate_file_size(test_file, max_size=1024)
        assert size > 0
        assert size < 1024

    def test_validate_file_size_too_large(self, tmp_path):
        """Test file size validation with large file."""
        # Create a file
        test_file = tmp_path / "large.txt"
        test_file.write_text("a" * 1000)

        # Validate with small limit
        with pytest.raises(ValidationError, match="too large"):
            InputValidator.validate_file_size(test_file, max_size=500)

    def test_validate_file_size_nonexistent(self, tmp_path):
        """Test file size validation with nonexistent file."""
        test_file = tmp_path / "nonexistent.txt"

        with pytest.raises(ValidationError, match="does not exist"):
            InputValidator.validate_file_size(test_file)

    def test_validate_path_absolute(self, tmp_path):
        """Test path validation returns absolute path."""
        relative = Path("relative/path")
        resolved = InputValidator.validate_path(relative)

        assert resolved.is_absolute()

    def test_validate_path_traversal(self):
        """Test path validation detects traversal."""
        with pytest.raises(ValidationError, match="traversal not allowed"):
            InputValidator.validate_path(Path("../../../etc/passwd"))

    def test_dangerous_extensions_coverage(self):
        """Test that common dangerous extensions are blocked."""
        assert '.exe' in InputValidator.DANGEROUS_EXTENSIONS
        assert '.dll' in InputValidator.DANGEROUS_EXTENSIONS
        assert '.sh' in InputValidator.DANGEROUS_EXTENSIONS
        assert '.bat' in InputValidator.DANGEROUS_EXTENSIONS
