"""Conversation save/load management for Model Chat CLI."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class ConversationManager:
    """Manages saving and loading conversations with metadata."""

    def __init__(self, conversations_dir: Optional[Path] = None):
        """Initialize conversation manager.

        Args:
            conversations_dir: Directory to store conversations
                              (default: ~/.model_chat_conversations/)
        """
        if conversations_dir is None:
            conversations_dir = Path.home() / ".model_chat_conversations"

        self.conversations_dir = Path(conversations_dir)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

    def save_conversation(
        self,
        history: List[Dict],
        server: Dict,
        model: str,
        metadata: Optional[Dict] = None,
        name: Optional[str] = None
    ) -> str:
        """Save a conversation to disk.

        Args:
            history: Conversation history (list of message dicts)
            server: Server configuration dict
            model: Model name
            metadata: Optional additional metadata
            name: Optional custom conversation name

        Returns:
            Conversation ID (filename without extension)
        """
        timestamp = datetime.now()

        # Generate conversation ID
        if name:
            # Use custom name, sanitize it
            conv_id = self._sanitize_filename(name)
        else:
            # Auto-generate from timestamp
            conv_id = timestamp.strftime("%Y%m%d_%H%M%S")

        # Build conversation data
        conversation = {
            "id": conv_id,
            "created_at": timestamp.isoformat(),
            "server": {
                "url": server.get("url"),
                "type": server.get("type"),
                "ip": server.get("ip"),
                "port": server.get("port"),
            },
            "model": model,
            "message_count": len(history),
            "metadata": metadata or {},
            "history": history,
        }

        # Save to file
        filepath = self.conversations_dir / f"{conv_id}.json"

        # Handle duplicates by appending counter
        counter = 1
        while filepath.exists():
            filepath = self.conversations_dir / f"{conv_id}_{counter}.json"
            counter += 1

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)

        return filepath.stem

    def load_conversation(self, conv_id: str) -> Optional[Dict]:
        """Load a conversation from disk.

        Args:
            conv_id: Conversation ID (filename without extension)

        Returns:
            Conversation dict or None if not found
        """
        filepath = self.conversations_dir / f"{conv_id}.json"

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def list_conversations(
        self,
        limit: Optional[int] = None,
        model: Optional[str] = None
    ) -> List[Dict]:
        """List all saved conversations.

        Args:
            limit: Maximum number of conversations to return
            model: Filter by model name

        Returns:
            List of conversation metadata dicts (sorted by created_at descending)
        """
        conversations = []

        for filepath in self.conversations_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    conv = json.load(f)

                    # Filter by model if specified
                    if model and conv.get("model") != model:
                        continue

                    # Extract metadata only (not full history)
                    conversations.append({
                        "id": conv["id"],
                        "created_at": conv["created_at"],
                        "server": conv["server"],
                        "model": conv["model"],
                        "message_count": conv["message_count"],
                        "metadata": conv.get("metadata", {}),
                    })
            except (json.JSONDecodeError, IOError, KeyError):
                continue

        # Sort by created_at (most recent first)
        conversations.sort(key=lambda c: c["created_at"], reverse=True)

        # Apply limit
        if limit:
            conversations = conversations[:limit]

        return conversations

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation.

        Args:
            conv_id: Conversation ID

        Returns:
            True if deleted, False if not found
        """
        filepath = self.conversations_dir / f"{conv_id}.json"

        if not filepath.exists():
            return False

        try:
            filepath.unlink()
            return True
        except IOError:
            return False

    def export_conversation(
        self,
        conv_id: str,
        output_path: Path,
        format: str = "json"
    ) -> bool:
        """Export a conversation to a different format.

        Args:
            conv_id: Conversation ID
            output_path: Output file path
            format: Export format ("json", "txt", "md")

        Returns:
            True if exported successfully
        """
        conversation = self.load_conversation(conv_id)

        if not conversation:
            return False

        try:
            if format == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(conversation, f, indent=2, ensure_ascii=False)

            elif format == "txt":
                with open(output_path, 'w', encoding='utf-8') as f:
                    self._export_to_text(conversation, f)

            elif format == "md":
                with open(output_path, 'w', encoding='utf-8') as f:
                    self._export_to_markdown(conversation, f)

            else:
                return False

            return True
        except IOError:
            return False

    def _export_to_text(self, conversation: Dict, file):
        """Export conversation to plain text."""
        file.write(f"Conversation: {conversation['id']}\n")
        file.write(f"Created: {conversation['created_at']}\n")
        file.write(f"Model: {conversation['model']}\n")
        file.write(f"Server: {conversation['server']['url']}\n")
        file.write("=" * 60 + "\n\n")

        for msg in conversation['history']:
            role = msg['role'].upper()
            content = msg['content']
            file.write(f"{role}:\n{content}\n\n")

    def _export_to_markdown(self, conversation: Dict, file):
        """Export conversation to markdown."""
        file.write(f"# Conversation: {conversation['id']}\n\n")
        file.write(f"**Created:** {conversation['created_at']}  \n")
        file.write(f"**Model:** {conversation['model']}  \n")
        file.write(f"**Server:** {conversation['server']['url']}  \n\n")
        file.write("---\n\n")

        for msg in conversation['history']:
            role = msg['role']
            content = msg['content']

            if role == "user":
                file.write(f"### 👤 User\n\n{content}\n\n")
            else:
                file.write(f"### 🤖 Assistant\n\n{content}\n\n")

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a filename to be filesystem-safe.

        Args:
            name: Original filename

        Returns:
            Sanitized filename
        """
        # Replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            name = name.replace(char, '_')

        # Limit length
        name = name[:100]

        # Remove leading/trailing spaces and dots
        name = name.strip('. ')

        return name or "conversation"


# Singleton instance
conversation_manager = ConversationManager()
