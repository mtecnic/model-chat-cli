"""Application configuration management.

Stores user settings in ~/.model_chat/config.json. Access via dotted paths,
e.g. config.get("github.repo_path").
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


CONFIG_DIR = Path.home() / ".model_chat"
CONFIG_FILE = CONFIG_DIR / "config.json"


DEFAULTS: dict = {
    "version": 2,
    "ui": {
        "theme": "modelchat",
        "compact": False,
    },
    "scan": {
        "extra_ips": "",
        "concurrency": 500,
        "tcp_timeout": 0.5,
        "auto_scan": True,
    },
    "chat": {
        "thinking_default": False,
        "system_prompt": "",
        "auto_export": True,
    },
    "stress": {
        "max_tokens": 256,
        "system_prompt": "",
    },
    "output": {
        "root": str(CONFIG_DIR / "exports"),
    },
    "github": {
        "repo_path": "",
        "branch": "main",
        "prefix": "results",
        "push": True,
        "dry_run": False,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into a copy of base (dicts merged, other values replaced)."""
    merged = {}
    for key, value in base.items():
        if key in override:
            if isinstance(value, dict) and isinstance(override[key], dict):
                merged[key] = _deep_merge(value, override[key])
            else:
                merged[key] = override[key]
        else:
            merged[key] = value
    for key, value in override.items():
        if key not in base:
            merged[key] = value
    return merged


class ConfigManager:
    """Load, access and persist application settings."""

    def __init__(self, path: Path = CONFIG_FILE):
        self.path = Path(path)
        self.data = dict(DEFAULTS)
        self.load()

    def load(self) -> None:
        """Load config from disk, merging over defaults."""
        try:
            if self.path.exists():
                with open(self.path, "r") as f:
                    stored = json.load(f)
                if isinstance(stored, dict):
                    self.data = _deep_merge(DEFAULTS, stored)
        except Exception:
            self.data = dict(DEFAULTS)

    def save(self) -> None:
        """Atomically write config to disk."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self.path.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self.data, f, indent=2)
                os.replace(tmp_name, self.path)
            except Exception:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
                raise
        except Exception:
            pass  # config is best-effort; the app still runs

    def get(self, dotted: str, default: Any = None) -> Any:
        """Get a value by dotted path, e.g. get("github.repo_path")."""
        node: Any = self.data
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, dotted: str, value: Any) -> None:
        """Set a value by dotted path, creating intermediate dicts."""
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def reset(self) -> None:
        """Restore defaults (in memory; call save() to persist)."""
        self.data = dict(DEFAULTS)

    @property
    def exports_root(self) -> Path:
        root = Path(self.get("output.root", str(CONFIG_DIR / "exports"))).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return root
