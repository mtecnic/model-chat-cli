"""Accuracy benchmark view (/bench): GSM8K, MMLU-Pro, ARC-Challenge, HumanEval."""
import time
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from benchmarks import ALL_BENCHMARKS
from benchmarks.base import Benchmark, BenchSummary, ItemResult
from benchmarks.datasets import DatasetError, load_cached
from benchmarks.runner import (THINKING_TOKEN_FACTOR, load_items, previous_runs,
                               run_bench, save_run)
from client import ModelClient
from ui.components import CellGrid, format_menu_item
from ui.interrupt import run_interruptible

DEFAULT_SEED = 1234

_ICONS = {"pending": ("·", "chrome"), "running": ("●", "status.running"),
          "correct": ("✓", "status.ok"), "wrong": ("✗", "status.warn"),
          "error": ("!", "status.error")}


def _item_cell(r: ItemResult) -> Text:
    icon, style = _ICONS.get(r.status, ("?", "chrome"))
    text = Text()
    text.append(icon + " ", style=style)
    text.append(r.item_id.split("/")[-1], style="accent")
    text.append("\n")
    if r.status in ("correct", "wrong"):
        text.append(f"{r.extracted or '-'}", style=style)
        text.append(f" / {r.gold}", style="chrome")
        text.append(f"  {r.latency:.0f}s", style="metric.label")
    elif r.status == "error":
        text.append(r.error or "error", style="status.error")
    elif r.status == "running":
        text.append("thinking...", style="status.running")
    else:
        text.append("queued", style="chrome")
    return text


class BenchView:
    """Pick benchmarks, run them with a live dashboard, show scores."""

    def __init__(self, console: Console, server: dict, model: str):
        self.console = console
        self.server = server
        self.model = model
        self.client = ModelClient(server, model)
        self.code_exec_ok = False

    # ------------------------------------------------------------------ #
    # Menu
    # ------------------------------------------------------------------ #
    async def run(self) -> str:
        self.console.clear()
        self.console.print()
        self.console.print(Panel(
            f"[accent.bold]Accuracy Benchmarks[/accent.bold]  [chrome]{self.model}[/chrome]",
            border_style="chrome.border", padding=(0, 1)))
        self.console.print()

        benches = self._select_benchmarks()
        if not benches:
            return "back"
        n = self._select_sample_size()
        concurrency = int(Prompt.ask("[prompt]Concurrent requests[/prompt]",
                                     choices=["1", "2", "4", "8", "16"], default="4"))
        thinking = Confirm.ask("[prompt]Enable thinking/reasoning[/prompt]", default=False)
        seed = DEFAULT_SEED
        self.console.print()

        summaries = []
        for bench in benches:
            if bench.executes_code and not self._confirm_code_exec():
                self.console.print(f"  [chrome]Skipping {bench.name}.[/chrome]\n")
                continue
            summary, stopped = await self._run_one(bench, n, seed, concurrency, thinking)
            if summary is not None:
                summaries.append(summary)
            if stopped:
                break

        if len(summaries) > 1:
            self._print_overview(summaries)

        try:
            Prompt.ask("\n  [chrome]Press Enter to return to chat[/chrome]",
                       default="", show_default=False)
        except (KeyboardInterrupt, EOFError):
            pass
        return "back"

    def _select_benchmarks(self) -> List[Benchmark]:
        self.console.print("  [chrome.header]Benchmarks[/chrome.header]\n")
        for i, cls in enumerate(ALL_BENCHMARKS, 1):
            cached = " [status.ok](cached)[/status.ok]" if load_cached(cls.key) is not None else ""
            self.console.print(format_menu_item(str(i), f"{cls.name}{cached}", cls.description))
        self.console.print(format_menu_item("A", "All of the above"))
        self.console.print("  [chrome]Q.[/chrome] Cancel\n")

        while True:
            raw = Prompt.ask("[prompt]Select (e.g. 1,3)[/prompt]", default="1").strip().lower()
            if raw == "q":
                return []
            if raw == "a":
                return [cls() for cls in ALL_BENCHMARKS]
            try:
                picks = [int(p) for p in raw.replace(" ", ",").split(",") if p]
            except ValueError:
                picks = []
            if picks and all(1 <= p <= len(ALL_BENCHMARKS) for p in picks):
                seen = []
                for p in picks:
                    if p not in seen:
                        seen.append(p)
                return [ALL_BENCHMARKS[p - 1]() for p in seen]
            self.console.print("  [status.error]Enter numbers from the list, A or Q.[/status.error]")

    def _select_sample_size(self) -> Optional[int]:
        self.console.print()
        self.console.print("  [chrome]Smaller samples are faster but noisier "
                           "(200 items gives roughly ±6 points).[/chrome]")
        raw = Prompt.ask("[prompt]Questions per benchmark[/prompt]",
                         choices=["25", "50", "100", "200", "500", "all"], default="200")
        return None if raw == "all" else int(raw)

    def _confirm_code_exec(self) -> bool:
        if self.code_exec_ok:
            return True
        self.console.print(Panel(
            "HumanEval grades answers by [bold]running the model's Python code on this "
            "machine[/bold].\nIt runs in a separate process with a CPU/memory/time limit and an "
            "empty temp directory, but it is not a real sandbox: the code can still reach "
            "the network and your files.",
            title="[status.warn]Runs generated code[/status.warn]", border_style="status.warn"))
        self.code_exec_ok = Confirm.ask("[prompt]Run HumanEval anyway?[/prompt]", default=False)
        return self.code_exec_ok

    # ------------------------------------------------------------------ #
    # Running
    # ------------------------------------------------------------------ #
    async def _load(self, bench: Benchmark, n, seed):
        """Load (download on first use) and sample the dataset."""
        if load_cached(bench.key) is not None:
            return await load_items(bench, n, seed)

        progress = Progress(TextColumn("  [accent]Downloading {task.description}[/accent]"),
                            BarColumn(), TextColumn("{task.completed}/{task.total} pages"),
                            console=self.console, transient=True)
        with progress:
            task = progress.add_task(bench.name, total=None)

            async def on_progress(done, total):
                progress.update(task, completed=done, total=total)

            items, interrupted = await run_interruptible(load_items(bench, n, seed, on_progress))
        if interrupted:
            return None
        return items

    async def _run_one(self, bench: Benchmark, n, seed, concurrency, thinking):
        """Run one benchmark. Returns (summary or None, stopped_by_user)."""
        try:
            items = await self._load(bench, n, seed)
        except DatasetError as e:
            self.console.print(Text(f"  Could not download {bench.name}: {e}", style="status.error"))
            return None, False
        if items is None:
            self.console.print("  [status.warn]Download cancelled.[/status.warn]")
            return None, True

        results: List[ItemResult] = []
        started: List[ItemResult] = []  # grid order: when each question started
        started_ids = set()
        layout = self._create_layout(bench, len(items), concurrency, started, results)
        wall_start = time.monotonic()

        async def on_result(r: ItemResult):
            if r.status == "running" and r.index not in started_ids:
                started_ids.add(r.index)
                started.append(r)
            self._update_layout(layout, started, results, len(items), wall_start)

        max_tokens = bench.max_tokens * (THINKING_TOKEN_FACTOR if thinking else 1)
        error = None
        try:
            with Live(layout, console=self.console, screen=True, refresh_per_second=4,
                      redirect_stdout=True, redirect_stderr=True):
                _, interrupted = await run_interruptible(run_bench(
                    bench, self.client, items, results, concurrency=concurrency,
                    enable_thinking=thinking, callback=on_result, max_tokens=max_tokens))
        except Exception as e:
            interrupted, error = False, e

        wall = time.monotonic() - wall_start
        if error is not None:
            self.console.print(Text(f"  {bench.name} failed: {type(error).__name__}: {error}",
                                    style="status.error"))
        summary = BenchSummary.from_results(bench.key, self.model, len(items), results, wall,
                                            stopped_early=interrupted or error is not None)
        if summary.finished == 0:
            self.console.print(f"  [status.warn]{bench.name}: no questions finished.[/status.warn]")
            return None, interrupted

        config = {"server": self.server.get("url"), "server_type": self.server.get("type"),
                  "n": n, "seed": seed, "concurrency": concurrency, "thinking": thinking,
                  "max_tokens": max_tokens}
        path = save_run(summary, results, config)
        self._print_summary(bench, summary, results, path, max_tokens)
        return summary, interrupted

    def _create_layout(self, bench, total, concurrency, started, results) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=6),
        )
        layout["header"].update(Panel(
            f"[accent.bold]{bench.name}[/accent.bold]  [chrome]{self.model} · {total} questions "
            f"· {concurrency} at a time   (Ctrl+C to stop)[/chrome]", border_style="chrome.border"))
        layout["main"].update(self._grid(started, total))
        layout["footer"].update(self._stats_panel(results, total, 0.0))
        return layout

    @staticmethod
    def _grid(started, total) -> CellGrid:
        return CellGrid(started, _item_cell, title="Questions started",
                        cell_min_w=22, empty_text="Sending first questions...")

    def _update_layout(self, layout, started, results, total, wall_start):
        layout["main"].update(self._grid(started, total))
        layout["footer"].update(self._stats_panel(results, total, time.monotonic() - wall_start))

    @staticmethod
    def _stats_panel(results: List[ItemResult], total: int, elapsed: float) -> Panel:
        done = [r for r in results if r.finished]
        correct = sum(r.status == "correct" for r in done)
        wrong = sum(r.status == "wrong" for r in done)
        errors = sum(r.status == "error" for r in done)
        toks = sum(r.completion_tokens for r in done)
        table = Table.grid(padding=(0, 2))
        for _ in range(4):
            table.add_column()
        acc = f"{100 * correct / len(done):.1f}%" if done else "-"
        eta = "-"
        if done and elapsed > 0 and len(done) < total:
            eta = f"{(total - len(done)) * elapsed / len(done):.0f}s"
        table.add_row("[chrome.header]Progress[/chrome.header]", f"{len(done)}/{total}",
                      "[chrome.header]Accuracy[/chrome.header]", f"[metric]{acc}[/metric]")
        table.add_row("[chrome.header]Correct[/chrome.header]", f"[status.ok]{correct}[/status.ok]",
                      "[chrome.header]Elapsed[/chrome.header]", f"{elapsed:.0f}s")
        table.add_row("[chrome.header]Wrong[/chrome.header]", f"[status.warn]{wrong}[/status.warn]",
                      "[chrome.header]ETA[/chrome.header]", eta)
        table.add_row("[chrome.header]Errors[/chrome.header]", f"[status.error]{errors}[/status.error]",
                      "[chrome.header]Output tok/s[/chrome.header]",
                      f"{toks / elapsed:.0f}" if elapsed > 0 and toks else "-")
        return Panel(table, title="Score", border_style="chrome.border")

    # ------------------------------------------------------------------ #
    # Results
    # ------------------------------------------------------------------ #
    def _print_summary(self, bench, s: BenchSummary, results: List[ItemResult], path,
                       max_tokens: int):
        lo, hi = s.ci
        title = f"{bench.name} — " + ("Partial Results" if s.stopped_early else "Results")
        table = Table(title=f"[accent.bold]{title}[/accent.bold]", show_header=False,
                      border_style="chrome.border")
        table.add_column(style="chrome.header")
        table.add_column()
        table.add_row("Accuracy", f"[metric]{100 * s.accuracy:.1f}%[/metric]  "
                                  f"[chrome](95% CI {100 * lo:.1f}–{100 * hi:.1f})[/chrome]")
        table.add_row("Scored", f"{s.correct} correct / {s.finished} answered"
                                + (f" of {s.n_requested}" if s.finished < s.n_requested else ""))
        table.add_row("Wrong", f"[status.warn]{s.wrong}[/status.warn]")
        table.add_row("Errors", (f"[status.error]{s.errors}[/status.error] (counted as wrong)"
                                 if s.errors else "0"))
        if s.avg_completion_tokens:
            table.add_row("Avg output", f"{s.avg_completion_tokens:.0f} tokens")
        capped = sum(1 for r in results if r.finished and r.completion_tokens >= max_tokens)
        if capped:
            table.add_row("Hit token limit", f"[status.warn]{capped}[/status.warn] "
                          f"[chrome](cut off at {max_tokens} tokens, usually scored wrong)[/chrome]")
        table.add_row("Avg latency", f"{s.avg_latency:.1f}s")
        table.add_row("Wall time", f"{s.wall_time:.0f}s")
        self.console.print()
        self.console.print(table)
        shown = str(path)
        home = str(Path.home())
        if shown.startswith(home):
            shown = "~" + shown[len(home):]
        self.console.print(Text(f"  Saved to {shown}", style="chrome", overflow="fold"))

        prev = previous_runs(bench.key, self.model, limit=5, exclude=path)
        if prev:
            pt = Table(title="Previous runs (this model)", border_style="chrome.border",
                       title_style="chrome.header")
            pt.add_column("When", style="chrome")
            pt.add_column("n", justify="right")
            pt.add_column("Think", justify="center")
            pt.add_column("Accuracy", justify="right")
            pt.add_column("95% CI", justify="right", style="chrome")
            for run in prev:
                ps, pc = run.get("summary") or {}, run.get("config") or {}
                pt.add_row((run.get("saved_at") or "")[:16].replace("T", " "),
                           str(ps.get("finished", "")),
                           "on" if pc.get("thinking") else "off",
                           f"{100 * ps.get('accuracy', 0):.1f}%",
                           f"{100 * ps.get('ci_low', 0):.0f}–{100 * ps.get('ci_high', 0):.0f}")
            self.console.print()
            self.console.print(pt)

        failures = [r for r in results if r.status in ("wrong", "error")]
        if failures:
            try:
                show = Confirm.ask(f"\n  [prompt]Show some of the {len(failures)} misses?[/prompt]",
                                   default=False)
            except (KeyboardInterrupt, EOFError):
                show = False
            if show:
                self._print_failures(failures[:5], max_tokens)

    def _print_failures(self, failures: List[ItemResult], max_tokens: int):
        for r in failures:
            head = Text()
            head.append(f"\n  {r.item_id}  ", style="accent.bold")
            if r.status == "error":
                head.append(f"error: {r.error}", style="status.error")
            else:
                head.append(f"got {r.extracted!r}, expected {r.gold!r}", style="status.warn")
                if r.completion_tokens >= max_tokens:
                    head.append("  (hit token limit)", style="chrome")
            self.console.print(head)
            tail = r.reply.strip()[-600:]
            if tail:
                self.console.print(Panel(Text(tail), border_style="chrome.border",
                                         title="end of reply", title_align="left"))

    def _print_overview(self, summaries: List[BenchSummary]):
        names = {cls.key: cls.name for cls in ALL_BENCHMARKS}
        table = Table(title="[accent.bold]Benchmark Overview[/accent.bold]",
                      border_style="chrome.border")
        table.add_column("Benchmark")
        table.add_column("Accuracy", justify="right")
        table.add_column("95% CI", justify="right", style="chrome")
        table.add_column("n", justify="right")
        table.add_column("Errors", justify="right")
        for s in summaries:
            lo, hi = s.ci
            table.add_row(names.get(s.bench, s.bench), f"[metric]{100 * s.accuracy:.1f}%[/metric]",
                          f"{100 * lo:.0f}–{100 * hi:.0f}", str(s.finished), str(s.errors))
        self.console.print()
        self.console.print(table)
