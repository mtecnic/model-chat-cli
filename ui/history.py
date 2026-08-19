"""History screen - browse, open, publish, delete saved results."""
from __future__ import annotations

from pathlib import Path
from rich.text import Text
from textual import work
from textual.widgets.option_list import Option
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (Button, Input, Markdown, OptionList,
                              Static)

from storage.exports import StoredResult
from ui.base import BaseScreen, MUTED, make_header


class ResultModal(ModalScreen):
    """Open a result's markdown in a modal."""

    DEFAULT_CSS = """
    ResultModal { align: center middle; }
    ResultModal > Vertical {
        width: 80%;
        height: 80%;
        border: round $primary;
        padding: 1 2;
    }
    ResultModal VerticalScroll { height: 1fr; }
    ResultModal #rm-actions { height: 2; align: center right; }
    ResultModal Button { width: 12; }
    """

    BINDINGS = [Binding("escape", "close", "Close")]

    def __init__(self, markdown: str) -> None:
        super().__init__()
        self.markdown = markdown

    def compose(self) -> ComposeResult:
        with Vertical():
            with VerticalScroll():
                yield Markdown(self.markdown)
            with Horizontal(id="rm-actions"):
                yield Button("Close", id="rm-close")

    def on_mount(self) -> None:
        self.query_one("#rm-close", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rm-close":
            self.dismiss()

    def action_close(self) -> None:
        self.dismiss()


class HistoryScreen(BaseScreen):
    DEFAULT_CSS = """
    HistoryScreen #hist-header {
        height: 2;
        content-align: center middle;
        border-bottom: solid $panel;
    }
    HistoryScreen OptionList { height: 1fr; }
    HistoryScreen #hist-filter {
        height: 3;
        width: 40%;
        margin: 0 0 0 4;
    }
    HistoryScreen #hist-actions {
        height: 2;
        width: 30%;
        align: right middle;
    }
    HistoryScreen #hist-actions Button { width: 14; }
    HistoryScreen #hist-hints {
        height: 2;
        content-align: center middle;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("o", "open", "Open"),
        Binding("p", "publish", "Publish"),
        Binding("d", "delete", "Delete"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._results: list[StoredResult] = []
        self._published: set = set()

    def build_content(self) -> ComposeResult:
        yield Static("", id="hist-header")
        yield OptionList(id="hist-list")
        yield Input(placeholder="filter (model, kind)…", id="hist-filter")
        with Horizontal(id="hist-actions"):
            yield Button("Open", id="hist-open")
            yield Button("Publish", id="hist-pub", variant="primary")
            yield Button("Delete", id="hist-del")
        yield Static("", id="hist-hints")

    def on_mount(self) -> None:
        self.query_one("#hist-header", Static).update(
            make_header("History", self.app.config.exports_root))
        self.query_one("#hist-hints", Static).update(
            "o: open   p: publish   d: delete   /: filter   esc: back")
        self._worker = self.run_worker(self._load, name="hist-load", exclusive=True)

    async def _load(self) -> None:
        self._results = self.app.store.list_results()
        self._published = await self._published_set()
        if not self._results:
            lst = self.query_one("#hist-list", OptionList)
            lst.clear_options()
            lst.add_option(
                Option(Text("  (no saved results yet)", style=MUTED), id="__empty"))
            return
        self._rebuild("")

    async def _published_set(self) -> set:
        from publish import Publisher
        cfg = self.app.config
        if not cfg.get("github.repo_path", ""):
            return set()
        pub = Publisher(cfg.get("github.repo_path", ""),
                        branch=cfg.get("github.branch", "main"),
                        prefix=cfg.get("github.prefix", "results"))
        try:
            return set(await pub.list_published())
        except Exception:
            return set()

    def _rebuild(self, q: str) -> None:
        q = q.strip().lower()
        lst = self.query_one("#hist-list", OptionList)
        lst.clear_options()
        count = 0
        for r in self._results:
            hay = f"{r.title} {r.kind} {r.path.name}".lower()
            if q and q not in hay:
                continue
            pub = "  •published" if f"{self.app.config.get('github.prefix', 'results')}/{r.path.name}" in self._published else ""
            text = Text.assemble(
                (f"  {r.title}", "bright_cyan"),
                (f"\n  {r.kind} · {r.created[:10]}{pub}", MUTED),
            )
            lst.add_option(Option(text, id=str(r.path)))
            count += 1
        if count:
            lst.highlighted = 0
        else:
            lst.add_option(Option(Text("  (no matches)", style=MUTED), id="__empty"))

    def _selected(self) -> StoredResult | None:
        lst = self.query_one("#hist-list", OptionList)
        idx = lst.highlighted
        if idx is None:
            return None
        opt = lst.get_option_at_index(idx)
        if opt is None:
            return None
        path = Path(str(opt.id))
        return next((r for r in self._results if r.path == path), None)

    def on_input_changed(self, event: Input.Changed) -> None:
        self._rebuild(event.value)

    def _action(self, handler) -> None:
        r = self._selected()
        if r is None:
            self.app.notify("Select a result first", severity="info", timeout=2)
            return
        handler(r)

    def action_open(self) -> None:
        self._action(lambda r: self.app.push_screen(
            ResultModal(r.path.read_text())))

    def action_publish(self) -> None:
        self._action(lambda r: self.run_worker(
            self._publish_one(r), name="hist-pub"))

    async def _publish_one(self, r: StoredResult) -> None:
        from publish import Publisher
        cfg = self.app.config
        pub = Publisher(cfg.get("github.repo_path", ""),
                        branch=cfg.get("github.branch", "main"),
                        prefix=cfg.get("github.prefix", "results"),
                        push=cfg.get("github.push", True),
                        dry_run=cfg.get("github.dry_run", False))
        ok, desc = pub.validate()
        if not ok:
            self.app.notify(f"Publish failed: {desc}", severity="error", timeout=5)
            return
        prefix = str(cfg.get("github.prefix", "results")).strip("/")
        rel = f"{prefix}/{r.path.name}"
        result = await pub.publish([r.path], rel_paths=[rel],
                                   commit_message=f"results: {r.kind}")
        self.app.notify(result.summary(), title="Publish",
                        severity="success" if result.ok else "error", timeout=6)
        self._published = await self._published_set()
        self._rebuild(self.query_one("#hist-filter", Input).value)

    def action_delete(self) -> None:
        self._action(lambda r: self._delete(r))

    def _delete(self, r: StoredResult) -> None:
        try:
            r.path.unlink()
            r.path.with_suffix(".json").unlink(missing_ok=True)
        except Exception as e:
            self.app.notify(f"delete failed: {e}", severity="error", timeout=3)
            return
        self.app._results_count = max(0, self.app._results_count - 1)
        self._results = [x for x in self._results if x.path != r.path]
        self._rebuild(self.query_one("#hist-filter", Input).value)
        self.app.notify("Deleted", severity="info", timeout=2)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "hist-open":
            self.action_open()
        elif event.button.id == "hist-pub":
            self.action_publish()
        elif event.button.id == "hist-del":
            self.action_delete()

    def action_back(self) -> None:
        self.app.pop_screen()
