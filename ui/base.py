"""Shared screen base with the global status bar."""
from rich.text import Text
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static

MUTED = "#8b9ab0"
"""Solid muted-slate tone: replaces the 50%-alpha 'dim' Rich style for
informational text (hints, descriptions, metadata) so it stays readable."""


def make_header(title: str, subtitle: str = "") -> Text:
    """Standard screen header: bold cyan title + muted context subtitle."""
    t = Text.assemble((title, "bold bright_cyan"))
    if subtitle:
        t.append(f"   ·   {subtitle}", style=MUTED)
    return t


class StatusBar(Static):
    """Bottom bar: current model, thinking state, result count, repo state."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        border-top: solid $panel;
        text-style: bold;
    }
    """

    def on_mount(self) -> None:
        self.set_interval(0.5, self._refresh)

    def _refresh(self) -> None:
        app = self.app
        t = Text()
        if app.model:
            t.append("model ", style="bold")
            t.append(f"{app.model}", style="bright_cyan")
            if app.server:
                t.append(f" @ {app.server['ip']}:{app.server['port']}", style=MUTED)
        else:
            t.append("no model selected - press [m]", style=MUTED)
        t.append("  |  think ", style="bold")
        t.append("on" if app.thinking else "off",
                 style="green" if app.thinking else "dim")
        t.append(f"  |  results {app.results_count}", style=MUTED)
        t.append(f"  |  {app.git_state()}", style=MUTED)
        self.update(t)


class BaseScreen(Screen):
    """All screens share the status bar and global layout."""

    DEFAULT_CSS = """
    BaseScreen {
        layout: vertical;
    }
    """

    def compose(self) -> ComposeResult:
        yield from self.build_content()
        yield StatusBar()

    def build_content(self) -> ComposeResult:
        """Override in each screen."""
        return ()