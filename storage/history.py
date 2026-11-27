"""Chat history persistence for the Rich-based Model Chat CLI.

Conversations are saved as JSON files in ~/.model_chat_history/.
Written by /export in the chat view, browsed and loaded with /history.
"""
import json
import os
from datetime import datetime
from typing import List, Optional


class ChatHistoryManager:
    """Save/load/list conversations as JSON files."""

    def __init__(self, root: Optional[str] = None) -> None:
        self.root = root or os.path.join(
            os.path.expanduser("~"), ".model_chat_history")
        os.makedirs(self.root, exist_ok=True)

    def save_conversation(self, model: str, server: str,
                          messages: List[dict],
                          system_prompt: str = "", **kwargs) -> str:
        """Persist a conversation and return the saved file path."""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        safe_model = model.replace("/", "_").replace(":", "_")
        safe_server = server.replace(":", "_").replace("/", "_")
        filename = f"chat_{safe_model}_{safe_server}_{timestamp}.json"
        path = os.path.join(self.root, filename)
        with open(path, "w") as f:
            json.dump({
                "model": model,
                "server": server,
                "saved_at": now.isoformat(timespec="seconds"),
                "system_prompt": system_prompt or "",
                "messages": messages,
                **kwargs,
            }, f, indent=2)
        return path

    def load_conversation(self, path: str) -> dict:
        """Load a conversation from a JSON file."""
        with open(path) as f:
            return json.load(f)

    def list_conversations(self) -> List[str]:
        """Return all saved conversation file paths, newest first."""
        files = [os.path.join(self.root, f) for f in os.listdir(self.root)
                 if f.startswith("chat_") and f.endswith(".json")]
        # Filenames start with the model name, so sort by mtime, not name
        return sorted(files, key=os.path.getmtime, reverse=True)

    def list_summaries(self, limit: int = 15) -> List[dict]:
        """Return [{path, date, model, preview}] for the newest conversations.

        Unreadable files are skipped.
        """
        summaries = []
        for path in self.list_conversations():
            if len(summaries) >= limit:
                break
            try:
                data = self.load_conversation(path)
            except (OSError, json.JSONDecodeError):
                continue
            saved_at = data.get("saved_at")
            if saved_at:
                date = saved_at.replace("T", " ")[:16]
            else:
                date = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
            first_user = next((m.get("content", "") for m in data.get("messages", [])
                               if m.get("role") == "user"), "")
            summaries.append({
                "path": path,
                "date": date,
                "model": data.get("model", "?"),
                "preview": " ".join(first_user.split()),
            })
        return summaries
