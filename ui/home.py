"""Home screen - the main menu."""
from rich.text import Text
from textual.widgets.option_list import Option
from textual.app import ComposeResult
from textual.widgets import OptionList, Static

from ui.base import BaseScreen, MUTED

MENU = [
    ("chat", "Chat", "talk with the selected model"),
    ("discovery", "Models & Servers", "scan the network, pick a model"),
    ("stress", "Stress Lab", "throughput / token / sustained / tool-bench"),
    ("prompt_arena", "Prompt Arena", "system prompts battle on the same question"),
    ("model_arena", "Model Arena", "models vs models, blind judged"),
    ("history", "History", "all saved results - search, open, delete"),
    ("publish", "Publish", "push results to your git repo"),
    ("settings", "Settings", "theme, scan, chat, output, repo"),
    ("quit", "Quit", "exit the app"),
]


class HomeScreen(BaseScreen):
    DEFAULT_CSS = """
    HomeScreen #home-header {
        height: 4;
        content-align: center middle;
        text-style: bold;
    }
    HomeScreen #home-current {
        height: 1;
        content-align: center middle;
        width: 100%;
    }
    HomeScreen OptionList {
        height: 1fr;
        width: 70%;
        align: center middle;
        padding: 0 1;
    }
    HomeScreen #home-hints {
        height: 3;
        content-align: center middle;
        color: $text-muted;
    }
    """

    def build_content(self) -> ComposeResult:
        yield Static("", id="home-header")
        yield Static("", id="home-current")
        yield OptionList(id="home-menu")
        yield Static("", id="home-hints")

    def on_mount(self) -> None:
        self._header()
        for key, title, desc in MENU:
            text = Text.assemble(
                (f"  {title}", "bold bright_cyan"),
                (f"   -   {desc}", MUTED),
            )
            self.query_one("#home-menu", OptionList).add_option(Option(text, id=key))
        self.query_one("#home-menu", OptionList).highlighted = 0
        self.query_one("#home-hints", Static).update(
            "enter: run action    m: home    s: settings    h: history    f1: keys"
        )

    def _header(self) -> None:
        t = Text(justify="center")
        t.append("MODEL CHAT", style="bold bright_cyan")
        t.append("  —  local LLM utility\n")
        self.query_one("#home-header", Static).update(t)

    def _current(self) -> None:
        app = self.app
        t = Text(justify="center")
        if app.model:
            t.append("current: ", style=MUTED)
            t.append(f"{app.model}", style="bright_cyan bold")
            if app.server:
                t.append(f" @ {app.server['ip']}:{app.server['port']}", style=MUTED)
        else:
            t.append("no model selected yet - open Models & Servers", style=MUTED)
        self.query_one("#home-current", Static).update(t)

    def action_home(self) -> None:
        self._current()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        app = self.app
        choice = event.option_id
        if choice == "quit":
            app.exit()
        elif choice == "chat":
            if app.model:
                app.push_screen("chat")
            else:
                app.push_screen("discovery")
        elif choice == "discovery":
            app.push_screen(choice)
        elif choice in ("stress", "prompt_arena", "model_arena"):
            if not app.model:
                app.notify("Pick a model first (Models & Servers)",
                           title="No model", severity="warning", timeout=4)
                return
            app.push_screen(choice)
        else:  # history, publish, settings
            app.push_screen(choice)
