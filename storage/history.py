"""Chat history persistence for the Rich-based Model Chat CLI.

Conversations are saved as JSON files in ~/.model_chat_history/.
"""
import json
import os
from datetime import datetime
from typing import List, Optional


class ChatHistoryManager:
    """Save/load/search/delete conversations as JSON files."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = root or os.path.join(
            os.path.expanduser("~"), ".model_chat_history")
        os.makedirs(self.root, exist_ok=True)

    def save_conversation(self, model: str, server: str,
                          messages: List[dict], **kwargs) -> str:
        """Persist a conversation and return the saved file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = model.replace("/", "_").replace(":", "_")
        safe_server = server.replace(":", "_").replace("/", "_")
        filename = f"chat_{safe_model}_{safe_server}_{timestamp}.json"
        path = os.path.join(self.root, filename)
        with open(path, "w") as f:
            json.dump({"model": model, "server": server,
                       "messages": messages}, f, indent=2)
        return path

    def load_conversation(self, path: str) -> dict:
        """Load a conversation from a JSON file."""
        with open(path) as f:
            return json.load(f)

    def list_conversations(self) -> List[str]:
        """Return all saved conversation file paths, newest first."""
        files = [os.path.join(self.root, f) for f in os.listdir(self.root)
                 if f.startswith("chat_") and f.endswith(".json")]
        return sorted(files, reverse=True)
