"""Settings screen - theme, scan, chat, output and repo config."""
from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (Button, Checkbox, Input, Select, Static,
                             TextArea)

from config import ConfigManager
from storage.exports import ResultStore
from ui.base import BaseScreen, MUTED, make_header
from ui.theme import modelchat_theme


class SettingsScreen(BaseScreen):
    DEFAULT_CSS = """
    SettingsScreen #set-header {
        height: 2;
        content-align: center middle;
    }
    SettingsScreen VerticalScroll { height: 1fr; padding: 0 2; }
    SettingsScreen .set-section {
        margin: 1 0;
    }
    SettingsScreen .set-title {
        text-style: bold;
        color: $accent;
        height: 1;
    }
    SettingsScreen .set-row {
        height: 2;
        content-align: left middle;
    }
    SettingsScreen .set-label {
        width: 18;
        color: $text-muted;
    }
    SettingsScreen .set-input {
        width: 40;
    }
    SettingsScreen #set-system {
        height: 5;
        margin: 0 0 0 18;
        width: 60;
        display: none;
    }
    SettingsScreen .set-hint {
        height: 1;
        color: $text-muted;
    }
    SettingsScreen #set-actions {
        height: 2;
        align: center middle;
    }
    SettingsScreen #set-actions Button { width: 20; }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+s", "save", "Save"),
    ]

    def build_content(self) -> ComposeResult:
        yield Static("", id="set-header")
        with VerticalScroll():
            yield Static("Interface", classes="set-title set-section")
            with Horizontal(classes="set-row"):
                yield Static("theme", classes="set-label")
                yield Select([], prompt="Theme",
                             id="set-theme", classes="set-input")
            yield Static("", classes="set-hint")

            yield Static("Scan", classes="set-title set-section")
            with Horizontal(classes="set-row"):
                yield Static("extra ips", classes="set-label")
                yield Input(id="set-ips", classes="set-input")
            with Horizontal(classes="set-row"):
                yield Static("concurrency", classes="set-label")
                yield Input(id="set-conc", classes="set-input")
            with Horizontal(classes="set-row"):
                yield Static("tcp timeout", classes="set-label")
                yield Input(id="set-timeout", classes="set-input")
            yield Static("extra ips: comma-separated IPs to always include", classes="set-hint")

            yield Static("Chat", classes="set-title set-section")
            with Horizontal(classes="set-row"):
                yield Static("", classes="set-label")
                yield Checkbox("thinking on by default", id="set-think")
            with Horizontal(classes="set-row"):
                yield Static("", classes="set-label")
                yield Checkbox("auto-export chats on close", id="set-autoexp")
            with Horizontal(classes="set-row"):
                yield Static("system prompt", classes="set-label")
                yield Button("edit…", id="set-sysbtn")
            yield TextArea("", id="set-system")

            yield Static("Output", classes="set-title set-section")
            with Horizontal(classes="set-row"):
                yield Static("exports root", classes="set-label")
                yield Input(id="set-out", classes="set-input")

            yield Static("GitHub", classes="set-title set-section")
            with Horizontal(classes="set-row"):
                yield Static("repo path", classes="set-label")
                yield Input(id="set-repo", classes="set-input")
            with Horizontal(classes="set-row"):
                yield Static("branch", classes="set-label")
                yield Input(id="set-branch", classes="set-input")
            with Horizontal(classes="set-row"):
                yield Static("prefix", classes="set-label")
                yield Input(id="set-prefix", classes="set-input")
            with Horizontal(classes="set-row"):
                yield Static("", classes="set-label")
                yield Checkbox("push after commit", id="set-push")
            with Horizontal(classes="set-row"):
                yield Static("", classes="set-label")
                yield Checkbox("dry-run (no writes)", id="set-dry")
            yield Static("results are committed to <repo>/<prefix>/ via your local git", classes="set-hint")

        with Horizontal(id="set-actions"):
            yield Button("Save", id="set-save", variant="primary")
            yield Button("Defaults", id="set-reset")
            yield Button("Back", id="set-back")

    def on_mount(self) -> None:
        app = self.app
        self.query_one("#set-header", Static).update(
            make_header("Settings", str(app.config.path)))
        themes = ["modelchat"] + [t for t in self.app.available_themes if t != "modelchat"]
        sel = self.query_one("#set-theme")
        sel.set_options([(t, t) for t in themes])
        sel.value = app.config.get("ui.theme", "modelchat")
        extra = app.config.get("scan.extra_ips", "")
        if isinstance(extra, list):
            extra = ", ".join(str(x) for x in extra)
        self.query_one("#set-ips", Input).value = str(extra)
        self.query_one("#set-conc", Input).value = str(app.config.get("scan.concurrency", 500))
        self.query_one("#set-timeout", Input).value = str(app.config.get("scan.tcp_timeout", 0.5))
        self.query_one("#set-think", Checkbox).value = app.config.get("chat.thinking_default", False)
        self.query_one("#set-autoexp", Checkbox).value = app.config.get("chat.auto_export", True)
        self.query_one("#set-system", TextArea).text = app.config.get("chat.system_prompt", "")
        self.query_one("#set-out", Input).value = str(app.config.exports_root)
        self.query_one("#set-repo", Input).value = app.config.get("github.repo_path", "")
        self.query_one("#set-branch", Input).value = app.config.get("github.branch", "main")
        self.query_one("#set-prefix", Input).value = app.config.get("github.prefix", "results")
        self.query_one("#set-push", Checkbox).value = app.config.get("github.push", True)
        self.query_one("#set-dry", Checkbox).value = app.config.get("github.dry_run", False)

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_save(self) -> None:
        self._save()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "set-back":
            self.app.pop_screen()
        elif event.button.id == "set-save":
            self._save()
        elif event.button.id == "set-reset":
            self.app.config.reset()
            self.app.config.save()
            self.notify("Defaults restored - check the form and save",
                        title="Reset", severity="info", timeout=4)
            self.app.pop_screen()
        elif event.button.id == "set-sysbtn":
            area = self.query_one("#set-system", TextArea)
            area.styles.display = "block" if area.styles.display == "none" else "none"

    def _save(self) -> None:
        app = self.app
        cfg = app.config
        theme = self.query_one("#set-theme").value or "modelchat"
        cfg.set("ui.theme", theme)
        cfg.set("scan.extra_ips", self.query_one("#set-ips", Input).value)
        try:
            cfg.set("scan.concurrency", int(float(self.query_one("#set-conc", Input).value or 500)))
        except ValueError:
            self.notify("concurrency must be a number", severity="error", timeout=3)
            return
        try:
            cfg.set("scan.tcp_timeout", float(self.query_one("#set-timeout", Input).value or 0.5))
        except ValueError:
            self.notify("tcp timeout must be a number", severity="error", timeout=3)
            return
        cfg.set("chat.thinking_default", self.query_one("#set-think", Checkbox).value)
        cfg.set("chat.auto_export", self.query_one("#set-autoexp", Checkbox).value)
        cfg.set("chat.system_prompt", self.query_one("#set-system", TextArea).text)
        cfg.set("output.root", self.query_one("#set-out", Input).value or str(app.config.exports_root))
        cfg.set("github.repo_path", self.query_one("#set-repo", Input).value)
        cfg.set("github.branch", self.query_one("#set-branch", Input).value or "main")
        cfg.set("github.prefix", self.query_one("#set-prefix", Input).value or "results")
        cfg.set("github.push", self.query_one("#set-push", Checkbox).value)
        cfg.set("github.dry_run", self.query_one("#set-dry", Checkbox).value)
        cfg.save()
        # apply theme + re-point store if the exports root changed
        if theme not in app.available_themes:
            app.register_theme(modelchat_theme())
        if theme in app.available_themes:
            app.theme = theme
        app.config = ConfigManager(cfg.path)
        app.store = ResultStore(app.config.exports_root)
        self.notify("Saved", title="Settings", severity="success", timeout=3)
        app.pop_screen()
