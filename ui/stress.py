"""Stress Lab screen - 6 load-testing modes with live progress."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.text import Text
from textual import work
from textual.widgets.option_list import Option
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (Button, DataTable, OptionList, RichLog,
                             Select, Static)

from stress_tester import StressTester
from storage.reports import stress_markdown
from ui.base import BaseScreen, MUTED, make_header

MODES = [
    ("throughput", "Throughput", "concurrent requests at max load"),
    ("token_stress", "Token Stress", "varied prompt lengths (500→5000)"),
    ("sustained", "Sustained Load", "steady-rate over time, thermal test"),
    ("consistency", "Consistency", "same prompt N× — hardware variance"),
    ("realistic", "Realistic User", "human-like sessions over time"),
    ("tool_bench", "Tool Bench", "agentic tool-calling accuracy"),
]

# mode -> list of (label, id, choices)
MODE_CONFIG: Dict[str, list] = {
    "throughput": [
        ("Requests", "reqs", ["10", "25", "50", "100"]),
    ],
    "token_stress": [
        ("Token sizes", "sizes", ["500 1000 2000 5000", "256 512 1024", "1000 4096"]),
    ],
    "sustained": [
        ("Minutes", "minutes", ["1", "3", "5", "10"]),
        ("Requests / min", "rpm", ["5", "10", "20", "30"]),
    ],
    "consistency": [
        ("Iterations", "iters", ["10", "25", "50"]),
    ],
    "realistic": [
        ("Minutes", "minutes", ["2", "5", "10"]),
        ("Mean requests / min", "rpm", ["10", "30", "60"]),
    ],
    "tool_bench": [
        ("Subset", "subset", ["quick", "full"]),
        ("Concurrency", "conc", ["1", "2", "4"]),
    ],
}


class StressScreen(BaseScreen):
    DEFAULT_CSS = """
    StressScreen #stress-header {
        height: 2;
        content-align: center middle;
        border-bottom: solid $panel;
    }
    StressScreen OptionList {
        height: auto;
        max-height: 10;
        border: round $panel;
        border-title-color: $primary;
    }
    StressScreen #stress-config {
        height: auto;
        min-height: 4;
        width: 100%;
        display: none;
    }
    StressScreen #stress-config Select {
        width: 1fr;
        margin: 0 1;
    }
    StressScreen #stress-run-btn {
        height: 2;
        width: 20;
        align: center middle;
    }
    StressScreen #stress-live {
        height: 1fr;
        width: 100%;
    }
    StressScreen #stress-table {
        width: 60%;
        border: round $panel;
        border-title-color: $primary;
    }
    StressScreen #stress-side {
        width: 40%;
        height: 1fr;
        padding: 0 1;
    }
    StressScreen #stress-stats {
        height: auto;
        border: round $panel;
        border-title-color: $primary;
        padding: 0 1;
    }
    StressScreen #stress-errors {
        height: 1fr;
        border: round $panel;
        border-title-color: $primary;
    }
    StressScreen #stress-summary {
        display: none;
        height: 1fr;
    }
    StressScreen #summary-scroll {
        height: 1fr;
        padding: 0 1;
    }
    StressScreen #summary-actions {
        height: 2;
        align: center middle;
    }
    StressScreen #summary-actions Button {
        width: 14;
    }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "run", "Run"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._mode: Optional[str] = None
        self._tester: Optional[StressTester] = None
        self._worker: Optional[work.Worker] = None
        self._summary_md: Optional[str] = None
        self._summary_extra: Optional[Dict[str, str]] = None
        self._stats = None
        self._results: List[Any] = []

    def build_content(self) -> ComposeResult:
        yield Static("", id="stress-header")
        yield OptionList(id="stress-modes")
        yield Horizontal(id="stress-config")
        yield Button("▶ Run", id="stress-run-btn", variant="primary")
        with Horizontal(id="stress-live"):
            yield DataTable(id="stress-table")
            with Vertical(id="stress-side"):
                yield Static("", id="stress-stats")
                yield RichLog(id="stress-errors")
        with Vertical(id="stress-summary"):
            with Vertical(id="summary-scroll"):
                pass
            with Horizontal(id="summary-actions"):
                yield Button("Save", id="stress-save", variant="primary")
                yield Button("Publish", id="stress-pub")
                yield Button("New Test", id="stress-again")
                yield Button("Back", id="stress-back")

    def on_mount(self) -> None:
        t = make_header("Stress Lab", self.app.model)
        self.query_one("#stress-header", Static).update(t)
        for key, title, desc in MODES:
            self.query_one("#stress-modes", OptionList).add_option(
                Option(Text.assemble((f"  {title}", "bold"), (f"  -  {desc}", MUTED)), id=key))
        self._table_setup()
        self.query_one("#stress-modes", OptionList).border_title = "mode"
        self.query_one("#stress-table", DataTable).border_title = "requests"
        self.query_one("#stress-stats", Static).border_title = "stats"
        self.query_one("#stress-errors", RichLog).border_title = "errors"
        self.query_one("#stress-modes", OptionList).highlighted = 0
        self._mode = "throughput"
        self._rebuild_config()

    def _table_setup(self) -> None:
        table = self.query_one("#stress-table", DataTable)
        table.cursor_type = None
        table.row_key_column = 0
        for col in ("#", "status", "time(s)", "tps", "ttft(s)", "tok"):
            table.add_column(col)

    # ------------------------------------------------------------------ #

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._mode = event.option_id
        self._rebuild_config()

    def _rebuild_config(self) -> None:
        cfg = self.query_one("#stress-config", Horizontal)
        cfg.remove_children()
        items = MODE_CONFIG.get(self._mode or "", [])
        for label, cid, choices in items:
            sel = Select([(c, c) for c in choices], value=choices[0],
                         prompt=label, id=f"cfg-{cid}")
            cfg.mount(sel)
        cfg.styles.display = "block" if items else "none"
        self._render_stats()
        if self._summary_md is not None:
            self._show_summary()

    def _render_stats(self) -> None:
        s = self._tester.stats if self._tester else None
        if s:
            bits = [
                ("done", f"{s.completed}/{s.total}"),
                ("ok / fail", f"{s.success} / {s.failed}"),
                ("decode", f"{s.avg_decode_tps or s.avg_tps:.1f} t/s"),
                ("ttft", f"{s.avg_ttft:.2f}s"),
                ("wall", f"{s.wall_clock_time:.0f}s"),
            ]
        else:
            bits = [("mode", self._mode or "—"),
                    ("status", "select a mode and hit Run")]
        t = Text()
        for label, val in bits:
            t.append(f"  {label:<9}", style=MUTED)
            t.append(f"{val}\n", style="bold bright_cyan")
        self.query_one("#stress-stats", Static).update(t)

    def _config_value(self, cid: str) -> str:
        try:
            return self.query_one(f"#cfg-{cid}", Select).value
        except Exception:
            return ""

    # ------------------------------------------------------------------ #

    def action_run(self) -> None:
        if not self.app.model or not self.app.server:
            self.app.notify("Pick a model first", title="No model", severity="warning")
            return
        if self._mode is None or self._worker and self._worker.is_running:
            return
        self._worker = self.run_worker(self._run(), name="stress", group="stress",
                                       exclusive=True)

    async def _run(self) -> None:
        app = self.app
        extra: Dict[str, str] = {}
        stats = None
        mode = self._mode
        max_tokens = int(app.config.get("stress.max_tokens", 256))
        tester = StressTester(app.server, app.model,
                              max_tokens=max_tokens,
                              system_prompt=app.config.get("stress.system_prompt", ""))
        self._tester = tester

        async def cb(result):
            self._update_row(result)

        try:
            if mode == "throughput":
                stats = await tester.run_throughput_test(
                    int(self._config_value("reqs") or 25), cb)
                extra["Requests"] = self._config_value("reqs")
            elif mode == "token_stress":
                sizes = [int(x) for x in (self._config_value("sizes") or "500 1000 2000 5000").split()]
                stats = await tester.run_token_stress_test(sizes, cb)
                extra["Token sizes"] = " ".join(map(str, sizes))
            elif mode == "sustained":
                stats = await tester.run_sustained_load_test(
                    int(self._config_value("minutes") or 3),
                    int(self._config_value("rpm") or 10), cb)
                extra["Duration"] = f"{self._config_value('minutes')} min @ {self._config_value('rpm')} rpm"
            elif mode == "consistency":
                stats = await tester.run_consistency_test(
                    "count from 1 to 10",
                    int(self._config_value("iters") or 25),
                    update_callback=cb)
                extra["Iterations"] = self._config_value("iters")
            elif mode == "realistic":
                stats = await tester.run_realistic_user_test(
                    int(self._config_value("minutes") or 5),
                    float(self._config_value("rpm") or 30))
                extra["Duration"] = f"{self._config_value('minutes')} min @ {self._config_value('rpm')} rpm"
            elif mode == "tool_bench":
                stats = await tester.run_tool_bench_test(
                    self._config_value("subset") or "full",
                    int(self._config_value("conc") or 1), cb)
                extra["Subset"] = self._config_value("subset")
                extra["Concurrency"] = self._config_value("conc")
        except Exception as e:
            self.query_one("#stress-errors", RichLog).write(Text(f"run failed: {e}", style="bright_red"))
            return
        self._summary_md = stress_markdown(mode, app.model,
                                            f"{app.server['ip']}:{app.server['port']}",
                                            stats, tester.results, extra)
        self._summary_extra = dict(extra, **{"Max Tokens": str(max_tokens)})
        self._stats = stats
        self._results = list(tester.results)
        self._show_summary()

    def _update_row(self, result) -> None:
        table = self.query_one("#stress-table", DataTable)
        row = (
            str(result.request_id),
            result.status,
            f"{result.duration:.2f}",
            f"{result.decode_tps:.0f}" if result.decode_tps else "-",
            f"{result.ttft:.3f}" if result.ttft else "-",
            str(result.completion_tokens or result.token_count),
        )
        try:
            table.add_row(row, key=result.request_id)
        except Exception:
            pass
        self._render_stats()

    def _show_summary(self) -> None:
        from textual.widgets import Markdown
        scroller = self.query_one("#summary-scroll")
        scroller.remove_children()
        scroller.mount(Markdown(self._summary_md))
        self.query_one("#stress-live").styles.display = "none"
        self.query_one("#stress-config").styles.display = "none"
        self.query_one("#stress-run-btn").styles.display = "none"
        self.query_one("#stress-summary").styles.display = "block"

    def _hide_summary(self) -> None:
        self.query_one("#stress-summary").styles.display = "none"
        self.query_one("#stress-live").styles.display = "block"
        self.query_one("#stress-run-btn").styles.display = "block"
        if self._mode:
            self.query_one("#stress-config").styles.display = "block"
        table = self.query_one("#stress-table", DataTable)
        table.clear()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "stress-back":
            self.app.pop_screen()
        elif event.button.id == "stress-again":
            self._hide_summary()
        elif event.button.id == "stress-save":
            self._do_save()
        elif event.button.id == "stress-pub":
            self._do_save(publish=True)

    def _do_save(self, publish: bool = False) -> None:
        app = self.app
        if self._stats is None:
            return
        path = app.store.save_stress(
            self._mode or "unknown", app.model,
            f"{app.server['ip']}:{app.server['port']}",
            self._stats, self._results,
            markdown=self._summary_md or "",
            extra=self._summary_extra,
        )
        app.note_saved()
        self.query_one("#stress-save", Button).label = "Saved ✓"
        self.app.notify(f"Saved: {path}", title="Saved", severity="success", timeout=4)
        if publish:
            self.run_worker(self._publish_file(path), name="publish-stress")

    async def _publish_file(self, path) -> None:
        from pathlib import Path
        from publish import Publisher
        cfg = self.app.config
        pub = Publisher(
            cfg.get("github.repo_path", ""),
            branch=cfg.get("github.branch", "main"),
            prefix=cfg.get("github.prefix", "results"),
            push=cfg.get("github.push", True),
            dry_run=cfg.get("github.dry_run", False),
        )
        ok, desc = pub.validate()
        if not ok:
            self.app.notify(f"Publish failed: {desc}", severity="error", timeout=5)
            return
        rel = f"{str(cfg.get('github.prefix', 'results')).strip('/')}/{Path(path).name}"
        msg = f"results: stress {self.app.model}"
        result = await pub.publish([Path(path)], rel_paths=[rel], commit_message=msg)
        self.app.notify(result.summary(), title="Publish",
                        severity="success" if result.ok else "error", timeout=6)

    def action_back(self) -> None:
        self.app.pop_screen()
