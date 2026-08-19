"""Publish screen - push saved results to the configured git repo."""
from __future__ import annotations

from pathlib import Path
from rich.text import Text
from textual import work
from textual.widgets.option_list import Option
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, OptionList, RichLog, Static

from ui.base import BaseScreen, MUTED, make_header


class PublishScreen(BaseScreen):
    DEFAULT_CSS = """
    PublishScreen #pub-header {
        height: 2;
        content-align: center middle;
        border-bottom: solid $panel;
    }
    PublishScreen #pub-status {
        height: 2;
        content-align: center middle;
        border: round $primary;
        width: 80%;
        align: center middle;
    }
    PublishScreen OptionList { height: 7fr; }
    PublishScreen #pub-log-wrap { height: 3fr; }
    PublishScreen #pub-log {
        height: 100%;
        border: round $panel;
        border-title-color: $primary;
    }
    PublishScreen #pub-actions {
        height: 2;
        align: center middle;
    }
    PublishScreen #pub-actions Button { width: 20; }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("p", "publish_sel", "Publish"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._all: list = []
        self._published: set = set()
        self._worker = None

    def build_content(self) -> ComposeResult:
        yield Static("", id="pub-header")
        yield Static("", id="pub-status")
        yield OptionList(id="pub-list")
        with Vertical(id="pub-log-wrap"):
            yield RichLog(id="pub-log")
        with Horizontal(id="pub-actions"):
            yield Button("Publish selected", id="pub-one", variant="primary")
            yield Button("Publish all new", id="pub-all")
            yield Button("Back", id="pub-back")
        yield Static("", id="pub-hints")

    def on_mount(self) -> None:
        self.query_one("#pub-log", RichLog).border_title = "log"
        self.query_one("#pub-header", Static).update(
            make_header("Publish", self.app.config.exports_root))
        self.query_one("#pub-hints", Static).update(
            "p: publish selected   (unmarked results are new)")
        self._worker = self.run_worker(self._load, name="pub-load", exclusive=True)

    def _publisher(self, dry: bool = False):
        from publish import Publisher
        cfg = self.app.config
        return Publisher(
            cfg.get("github.repo_path", ""),
            branch=cfg.get("github.branch", "main"),
            prefix=cfg.get("github.prefix", "results"),
            push=cfg.get("github.push", True),
            dry_run=dry or cfg.get("github.dry_run", False),
        )

    async def _load(self) -> None:
        cfg = self.app.config
        pub = self._publisher()
        ok, desc = pub.validate()
        status = self.query_one("#pub-status", Static)
        if ok:
            status.update(Text(desc, style="green"))
        else:
            status.update(Text(desc, style="bright_red"))
            self.query_one("#pub-list", OptionList).add_option(
                Option(Text("  (no valid repo - set one in Settings [g/s])", style="yellow"),
                      id="__empty"))
            return
        self._published = set(await pub.list_published())
        self._all = self.app.store.list_results()
        self._rebuild()
        if self._all:
            self.query_one("#pub-list", OptionList).highlighted = 0
        else:
            self.query_one("#pub-list", OptionList).add_option(
                Option(Text("  (no results yet)", style=MUTED), id="__empty"))

    def _rebuild(self) -> None:
        prefix = str(self.app.config.get("github.prefix", "results")).strip("/")
        lst = self.query_one("#pub-list", OptionList)
        lst.clear_options()
        for r in self._all:
            is_pub = f"{prefix}/{r.path.name}" in self._published
            mark = "  ✓ in repo" if is_pub else "  ✗ new"
            text = Text.assemble(
                (f"  {r.title}", "bright_cyan"),
                (f"\n  {r.kind} · {r.created[:10]}", MUTED),
                (mark, "green" if is_pub else "yellow"),
            )
            lst.add_option(Option(text, id=str(r.path)))

    def _selected(self):
        lst = self.query_one("#pub-list", OptionList)
        idx = lst.highlighted
        if idx is None:
            return None
        opt = lst.get_option_at_index(idx)
        if opt is None:
            return None
        path = Path(str(opt.id))
        return next((r for r in self._all if r.path == path), None)

    async def _publish_paths(self, paths: list, commit: str) -> None:
        cfg = self.app.config
        pub = self._publisher()
        log = self.query_one("#pub-log", RichLog)
        ok, desc = pub.validate()
        if not ok:
            self.app.notify(f"Publish failed: {desc}", severity="error", timeout=5)
            return
        prefix = str(cfg.get("github.prefix", "results")).strip("/")
        rels = [f"{prefix}/{p.name}" for p in paths]
        result = await pub.publish(list(paths), rel_paths=rels,
                                   commit_message=commit)
        for line in result.output:
            log.write(Text(line[:140], style=MUTED))
        log.write(Text(f"→ {result.summary()}",
                       style="green" if result.ok else "bright_red"))
        self.app.notify(result.summary(),
                        title="Publish" + (" (dry-run)" if result.dry_run else ""),
                        severity="success" if result.ok else "error", timeout=8)
        self._published = set(await pub.list_published())
        self._rebuild()

    def action_publish_sel(self) -> None:
        r = self._selected()
        if r is None:
            return
        self._worker = self.run_worker(
            self._publish_paths([r.path], f"results: {r.kind} {r.title}".strip()),
            name="pub", group="pub", exclusive=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pub-back":
            self.app.pop_screen()
        elif event.button.id == "pub-one":
            self.action_publish_sel()
        elif event.button.id == "pub-all":
            prefix = str(self.app.config.get("github.prefix", "results")).strip("/")
            new = [r.path for r in self._all
                   if f"{prefix}/{r.path.name}" not in self._published]
            if not new:
                self.app.notify("Everything is already in the repo",
                                severity="info", timeout=3)
                return
            self._worker = self.run_worker(
                self._publish_paths(new,
                                    f"results: {len(new)} new file(s)"),
                name="pub", group="pub", exclusive=True)

    def action_back(self) -> None:
        self.app.pop_screen()
