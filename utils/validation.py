"""Input validation and security utilities."""
import re
from pathlib import Path
from typing import Optional


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class InputValidator:
    """Validates user inputs for security and correctness."""

    # Maximum lengths to prevent memory exhaustion
    MAX_MESSAGE_LENGTH = 100000  # 100k characters
    MAX_SYSTEM_PROMPT_LENGTH = 10000  # 10k characters
    MAX_FILENAME_LENGTH = 255
    MAX_CONV_NAME_LENGTH = 100

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERN = re.compile(r'\.\.[/\\]')

    # Dangerous file extensions
    DANGEROUS_EXTENSIONS = {
        '.exe', '.dll', '.so', '.dylib', '.sh', '.bat', '.cmd',
        '.ps1', '.vbs', '.js', '.jar', '.app'
    }

    @classmethod
    def validate_message(cls, message: str) -> str:
        """Validate user message input.

        Args:
            message: User message

        Returns:
            Validated message

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(message, str):
            raise ValidationError("Message must be a string")

        if not message.strip():
            raise ValidationError("Message cannot be empty")

        if len(message) > cls.MAX_MESSAGE_LENGTH:
            raise ValidationError(
                f"Message too long (max {cls.MAX_MESSAGE_LENGTH} characters)"
            )

        return message

    @classmethod
    def validate_system_prompt(cls, prompt: str) -> str:
        """Validate system prompt.

        Args:
            prompt: System prompt

        Returns:
            Validated prompt

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(prompt, str):
            raise ValidationError("System prompt must be a string")

        if len(prompt) > cls.MAX_SYSTEM_PROMPT_LENGTH:
            raise ValidationError(
                f"System prompt too long (max {cls.MAX_SYSTEM_PROMPT_LENGTH} characters)"
            )

        return prompt

    @classmethod
    def validate_filename(cls, filename: str) -> str:
        """Validate filename for security.

        Args:
            filename: Filename to validate

        Returns:
            Validated filename

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(filename, str):
            raise ValidationError("Filename must be a string")

        if not filename.strip():
            raise ValidationError("Filename cannot be empty")

        if len(filename) > cls.MAX_FILENAME_LENGTH:
            raise ValidationError(
                f"Filename too long (max {cls.MAX_FILENAME_LENGTH} characters)"
            )

        # Check for path traversal attempts
        if cls.PATH_TRAVERSAL_PATTERN.search(filename):
            raise ValidationError("Path traversal not allowed in filename")

        # Check for dangerous characters
        if any(c in filename for c in '<>:"|?*\x00'):
            raise ValidationError("Filename contains invalid characters")

        # Check for dangerous extensions
        path = Path(filename)
        if path.suffix.lower() in cls.DANGEROUS_EXTENSIONS:
            raise ValidationError(f"Dangerous file extension: {path.suffix}")

        return filename

    @classmethod
    def validate_conversation_name(cls, name: str) -> str:
        """Validate conversation name.

        Args:
            name: Conversation name

        Returns:
            Validated name

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(name, str):
            raise ValidationError("Conversation name must be a string")

        if not name.strip():
            raise ValidationError("Conversation name cannot be empty")

        if len(name) > cls.MAX_CONV_NAME_LENGTH:
            raise ValidationError(
                f"Conversation name too long (max {cls.MAX_CONV_NAME_LENGTH} characters)"
            )

        # Basic sanitization
        sanitized = re.sub(r'[<>:"/\\|?*\x00]', '_', name)

        return sanitized

    @classmethod
    def validate_path(cls, path: Path, must_exist: bool = False) -> Path:
        """Validate file path for security.

        Args:
            path: Path to validate
            must_exist: Whether path must exist

        Returns:
            Validated path

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(path, (str, Path)):
            raise ValidationError("Path must be a string or Path object")

        path = Path(path)

        # Resolve to absolute path to detect traversal
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValidationError(f"Invalid path: {e}")

        # Check if path escapes expected directory
        # (This is a basic check, more sophisticated checks may be needed)
        if ".." in path.parts:
            raise ValidationError("Path traversal not allowed")

        # Check if path exists (if required)
        if must_exist and not resolved.exists():
            raise ValidationError(f"Path does not exist: {path}")

        # Check for symlink attacks
        if resolved.is_symlink():
            raise ValidationError("Symlinks not allowed for security")

        return resolved

    @classmethod
    def validate_port(cls, port: int) -> int:
        """Validate network port number.

        Args:
            port: Port number

        Returns:
            Validated port

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(port, int):
            raise ValidationError("Port must be an integer")

        if not (1 <= port <= 65535):
            raise ValidationError("Port must be between 1 and 65535")

        return port

    @classmethod
    def validate_url(cls, url: str) -> str:
        """Validate URL format.

        Args:
            url: URL to validate

        Returns:
            Validated URL

        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(url, str):
            raise ValidationError("URL must be a string")

        if not url.strip():
            raise ValidationError("URL cannot be empty")

        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            raise ValidationError("URL must start with http:// or https://")

        # Check for suspicious patterns
        if any(char in url for char in ['<', '>', '"', '\x00']):
            raise ValidationError("URL contains invalid characters")

        return url

    @classmethod
    def check_disk_space(cls, path: Path, required_bytes: int = 10 * 1024 * 1024) -> bool:
        """Check if sufficient disk space is available.

        Args:
            path: Path to check
            required_bytes: Required space in bytes (default: 10MB)

        Returns:
            True if sufficient space available

        Raises:
            ValidationError: If insufficient space
        """
        import shutil

        try:
            stat = shutil.disk_usage(path.parent if path.is_file() else path)
            if stat.free < required_bytes:
                raise ValidationError(
                    f"Insufficient disk space: {stat.free / (1024*1024):.1f}MB available, "
                    f"{required_bytes / (1024*1024):.1f}MB required"
                )
            return True
        except OSError as e:
            raise ValidationError(f"Cannot check disk space: {e}")

    @classmethod
    def validate_file_size(cls, path: Path, max_size: int = 100 * 1024 * 1024) -> int:
        """Validate file size is within limits.

        Args:
            path: File path
            max_size: Maximum size in bytes (default: 100MB)

        Returns:
            File size in bytes

        Raises:
            ValidationError: If file too large
        """
        if not path.exists():
            raise ValidationError(f"File does not exist: {path}")

        size = path.stat().st_size

        if size > max_size:
            raise ValidationError(
                f"File too large: {size / (1024*1024):.1f}MB "
                f"(max {max_size / (1024*1024):.1f}MB)"
            )

        return size
