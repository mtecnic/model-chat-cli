"""Model Chat CLI - Textual application shell."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Static

from config import ConfigManager
from storage.exports import ResultStore
from ui.base import MUTED
from ui.theme import modelchat_theme

from ui.home import HomeScreen
from ui.discovery import DiscoveryScreen
from ui.chat import ChatScreen
from ui.stress import StressScreen
from ui.prompt_arena import PromptArenaScreen
from ui.model_arena import ModelArenaScreen
from ui.history import HistoryScreen
from ui.settings import SettingsScreen
from ui.publish_screen import PublishScreen

SCREENS: Dict[str, type[Screen]] = {
    "home": HomeScreen,
    "discovery": DiscoveryScreen,
    "chat": ChatScreen,
    "stress": StressScreen,
    "prompt_arena": PromptArenaScreen,
    "model_arena": ModelArenaScreen,
    "history": HistoryScreen,
    "settings": SettingsScreen,
    "publish": PublishScreen,
}


class ModelChatApp(App):
    """Top-level app: shared state, global keys, status bar."""

    SCREENS = SCREENS
    CSS = """
    Screen {
        background: $background;
    }
    """
    BINDINGS = [
        Binding("m", "home", "Menu", show=True),
        Binding("s", "settings", "Settings", show=True),
        Binding("h", "history", "History", show=True),
        Binding("g", "publish", "Publish", show=True),
        Binding("f1", "help", "Keys", show=True),
    ]

    model: Optional[str] = reactive(None)
    thinking: bool = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self.config = ConfigManager()
        self.store = ResultStore(self.config.exports_root)
        self.servers: List[Dict[str, Any]] = []
        self.server: Optional[Dict[str, Any]] = None
        self._results_count = 0

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def on_mount(self) -> ComposeResult:
        if "modelchat" not in self.available_themes:
            self.register_theme(modelchat_theme())
        stored = self.config.get("ui.theme", "modelchat")
        if stored in self.available_themes:
            self.theme = stored
        self.thinking = self.config.get("chat.thinking_default", False)
        self._results_count = len(self.store.list_results())
        self.push_screen("home")
        self.run_worker(self._migrate_legacy, name="migrate", thread=True)

    def _migrate_legacy(self) -> None:
        """One-time: pull legacy outputs (cwd chat_*.md / arena_*.md, logs/)
        into the unified exports directory."""
        cwd = Path.cwd()
        moved = self.store.migrate_legacy(cwd)
        if moved:
            self._results_count = len(self.store.list_results())
            self.notify(f"Migrated {len(moved)} old result file(s) to "
                        f"{self.config.exports_root}", title="Migrated", severity="info")

    # ------------------------------------------------------------------ #
    # shared state helpers
    # ------------------------------------------------------------------ #

    @property
    def results_count(self) -> int:
        return self._results_count

    def note_saved(self) -> None:
        self._results_count += 1

    def git_state(self) -> str:
        repo = self.config.get("github.repo_path", "")
        if not repo:
            return "git: no repo"
        prefix = self.config.get("github.prefix", "results").strip("/")
        mode = "dry-run" if self.config.get("github.dry_run", False) else "push"
        return f"git: {prefix}/ → {repo} ({mode})"

    def save_chat(self, messages, system_prompt: str) -> Optional[Path]:
        if not self.server or not self.model:
            return None
        path = self.store.save_chat(
            model=self.model,
            server=f"{self.server['ip']}:{self.server['port']}",
            system_prompt=system_prompt or None,
            messages=messages,
        )
        self.note_saved()
        return path

    # ------------------------------------------------------------------ #
    # global actions
    # ------------------------------------------------------------------ #

    def action_home(self) -> None:
        self.switch_screen("home")

    def action_settings(self) -> None:
        self.push_screen("settings")

    def action_history(self) -> None:
        self.push_screen("history")

    def action_publish(self) -> None:
        self.push_screen("publish")

    def action_help(self) -> None:
        t = Text(justify="left")
        t.append("Global keys\n\n", style="bold")
        t.append("  m", style="bold")
        t.append("   menu (home)\n")
        t.append("  s", style="bold")
        t.append("   settings\n")
        t.append("  h", style="bold")
        t.append("   history / results\n")
        t.append("  g", style="bold")
        t.append("   publish to repo\n")
        t.append("  f1", style="bold")
        t.append("   this help\n")
        t.append("  ctrl+q", style="bold")
        t.append("  quit\n\n")
        t.append("Each screen has its own keys, shown in its hints line.", style=MUTED)
        self.notify(t, title="Keys", severity="information",
                    timeout=15, markup=False)
