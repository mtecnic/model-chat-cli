"""Stress test view for load testing AI models."""
import asyncio
from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.prompt import IntPrompt, Prompt

from stress_tester import StressTester, TestResult, TestStats
from ui.components import format_menu_item


class StressTestView:
    """Handle stress test interface and visualization."""

    def __init__(self, console: Console, server: dict, model: str):
        self.console = console
        self.server = server
        self.model = model
        self.max_tokens = 256
        self.system_prompt = ""
        self.tester = StressTester(server, model)
        self.current_results: List[TestResult] = []
        self.current_stats = TestStats()

    def _configure_test(self):
        """Configure max_tokens and system prompt before running a test."""
        self.console.print("  [chrome.header]Test Configuration[/chrome.header]\n")

        # Max tokens
        max_tok = Prompt.ask(
            "[prompt]Max output tokens[/prompt]",
            choices=["128", "256", "512", "1024"],
            default="256"
        )
        self.max_tokens = int(max_tok)

        # System prompt
        self.console.print()
        self.console.print(format_menu_item("1", "None", "No system prompt"))
        self.console.print(format_menu_item("2", "Helpful Assistant", "You are a helpful assistant."))
        self.console.print(format_menu_item("3", "Technical Expert", "You are a precise technical expert. Be thorough and accurate."))
        self.console.print(format_menu_item("4", "Custom", "Enter your own"))
        self.console.print()

        sp_choice = Prompt.ask(
            "[prompt]System prompt[/prompt]",
            choices=["1", "2", "3", "4"],
            default="1"
        )

        if sp_choice == "2":
            self.system_prompt = "You are a helpful assistant."
        elif sp_choice == "3":
            self.system_prompt = "You are a precise technical expert. Be thorough and accurate."
        elif sp_choice == "4":
            self.system_prompt = Prompt.ask("[prompt]Enter system prompt[/prompt]")
        else:
            self.system_prompt = ""

        # Apply to tester
        self.tester = StressTester(
            self.server, self.model,
            max_tokens=self.max_tokens, system_prompt=self.system_prompt
        )
        self.console.print()

    async def run(self) -> str:
        """Run the stress test interface."""
        self.console.clear()
        self.console.print()
        self.console.rule(
            f"[accent.bold]Stress Test[/accent.bold]  [chrome]{self.model}[/chrome]",
            style="chrome.border"
        )
        self.console.print()

        mode = self._select_mode()
        if mode is None:
            return "back"

        self._configure_test()

        if mode == 1:
            await self._run_throughput_mode()
        elif mode == 2:
            await self._run_token_stress_mode()
        elif mode == 3:
            await self._run_sustained_load_mode()
        elif mode == 4:
            await self._run_consistency_mode()

        self._show_summary()

        self.console.print("\n  [chrome]Press Enter to return to chat...[/chrome]")
        input()

        return "back"

    def _select_mode(self) -> int:
        """Show mode selection menu."""
        self.console.print("  [chrome.header]Select Mode[/chrome.header]\n")
        self.console.print(format_menu_item("1", "Throughput Test", "Concurrent requests to test parallelization"))
        self.console.print()
        self.console.print(format_menu_item("2", "Token Stress Test", "Performance with increasing prompt lengths"))
        self.console.print()
        self.console.print(format_menu_item("3", "Sustained Load", "Stability over extended period"))
        self.console.print()
        self.console.print(format_menu_item("4", "Consistency Test", "Same prompt N times — isolate hardware-level noise (thermals, DVFS, drivers)"))
        self.console.print()
        self.console.print("  [chrome]Q.[/chrome] Cancel\n")

        try:
            choice = Prompt.ask(
                "[prompt]Select mode[/prompt]",
                choices=["1", "2", "3", "4", "q", "Q"]
            )
            if choice.lower() == 'q':
                return None
            return int(choice)
        except (ValueError, KeyboardInterrupt):
            return None

    async def _run_throughput_mode(self):
        """Run throughput test mode."""
        self.console.print("\n  [accent.bold]Throughput Test[/accent.bold]\n")
        self.console.print("  [chrome]Spawns multiple concurrent requests to test parallelization[/chrome]\n")

        num_requests = int(IntPrompt.ask(
            "[prompt]Number of concurrent requests[/prompt]",
            choices=["5", "10", "20", "50"],
            default=10
        ))

        self.console.print(f"\n  [status.info]Starting test with {num_requests} concurrent requests...[/status.info]")
        self.console.print(f"  [chrome]{self.server['url']} | {self.model}[/chrome]\n")

        try:
            layout = self._create_layout(f"Throughput Test | {num_requests} concurrent")

            with Live(layout, console=self.console, refresh_per_second=4) as live:
                async def update_callback(result: TestResult):
                    existing = next((r for r in self.current_results if r.request_id == result.request_id), None)
                    if existing:
                        idx = self.current_results.index(existing)
                        self.current_results[idx] = result
                    else:
                        self.current_results.append(result)
                    self._update_layout(layout, self.tester.stats)

                self.current_stats = await self.tester.run_throughput_test(
                    num_requests, update_callback=update_callback
                )

            self.current_results = self.tester.results

        except Exception as e:
            self.console.print(f"\n  [status.error]Test failed: {type(e).__name__} - {str(e)}[/status.error]")
            import traceback
            self.console.print(f"  [chrome]{traceback.format_exc()}[/chrome]")
            raise

    async def _run_token_stress_mode(self):
        """Run token stress test mode."""
        self.console.print("\n  [accent.bold]Token Stress Test[/accent.bold]\n")
        self.console.print("  [chrome]Tests performance with progressively longer prompts[/chrome]\n")

        self.console.print(format_menu_item("1", "Quick", "500, 1000, 2000 tokens"))
        self.console.print(format_menu_item("2", "Standard", "500, 1000, 2000, 5000 tokens"))
        self.console.print(format_menu_item("3", "Extended", "500, 1000, 2000, 5000, 10000 tokens"))
        self.console.print()

        choice = int(IntPrompt.ask(
            "[prompt]Select test size[/prompt]",
            choices=["1", "2", "3"],
            default=2
        ))

        token_sizes = {
            1: [500, 1000, 2000],
            2: [500, 1000, 2000, 5000],
            3: [500, 1000, 2000, 5000, 10000]
        }[choice]

        self.console.print(f"\n  [status.info]Starting test with token sizes: {token_sizes}...[/status.info]")
        self.console.print(f"  [chrome]{self.server['url']} | {self.model}[/chrome]\n")

        try:
            layout = self._create_layout(f"Token Stress Test | {len(token_sizes)} sizes")

            with Live(layout, console=self.console, refresh_per_second=4) as live:
                async def update_callback(result: TestResult):
                    existing = next((r for r in self.current_results if r.request_id == result.request_id), None)
                    if existing:
                        idx = self.current_results.index(existing)
                        self.current_results[idx] = result
                    else:
                        self.current_results.append(result)
                    self._update_layout(layout, self.tester.stats)

                self.current_stats = await self.tester.run_token_stress_test(
                    token_sizes, update_callback=update_callback
                )

            self.current_results = self.tester.results

        except Exception as e:
            self.console.print(f"\n  [status.error]Test failed: {type(e).__name__} - {str(e)}[/status.error]")
            import traceback
            self.console.print(f"  [chrome]{traceback.format_exc()}[/chrome]")
            raise

    async def _run_sustained_load_mode(self):
        """Run sustained load test mode."""
        self.console.print("\n  [accent.bold]Sustained Load Test[/accent.bold]\n")
        self.console.print("  [chrome]Tests stability with continuous requests over time[/chrome]\n")

        duration = int(IntPrompt.ask(
            "[prompt]Duration in minutes[/prompt]",
            choices=["1", "5", "10", "60", "240", "720", "1440"],
            default=1
        ))

        rate = int(IntPrompt.ask(
            "[prompt]Requests per minute[/prompt]",
            choices=["5", "10", "20"],
            default=5
        ))

        total_requests = duration * rate

        self.console.print(f"\n  [status.info]Starting: {duration}min @ {rate} req/min ({total_requests} total)...[/status.info]")
        self.console.print(f"  [chrome]{self.server['url']} | {self.model}[/chrome]\n")

        try:
            layout = self._create_layout(f"Sustained Load | {duration}min @ {rate}/min")

            with Live(layout, console=self.console, refresh_per_second=4) as live:
                async def update_callback(result: TestResult):
                    existing = next((r for r in self.current_results if r.request_id == result.request_id), None)
                    if existing:
                        idx = self.current_results.index(existing)
                        self.current_results[idx] = result
                    else:
                        self.current_results.append(result)
                    self._update_layout(layout, self.tester.stats)

                self.current_stats = await self.tester.run_sustained_load_test(
                    duration, rate, update_callback=update_callback
                )

            self.current_results = self.tester.results

        except Exception as e:
            self.console.print(f"\n  [status.error]Test failed: {type(e).__name__} - {str(e)}[/status.error]")
            import traceback
            self.console.print(f"  [chrome]{traceback.format_exc()}[/chrome]")
            raise

    async def _run_consistency_mode(self):
        """Run consistency test mode — same prompt N times, serial."""
        self.console.print("\n  [accent.bold]Consistency Test[/accent.bold]\n")
        self.console.print("  [chrome]Runs one fixed prompt N times in series. Variance in the[/chrome]")
        self.console.print("  [chrome]results reflects hardware-level noise (thermals, DVFS, drivers,[/chrome]")
        self.console.print("  [chrome]kernel scheduling) rather than model or request differences.[/chrome]\n")

        prompt = self._pick_prompt()
        if prompt is None:
            return

        iterations = int(IntPrompt.ask(
            "[prompt]Iterations[/prompt]",
            choices=["5", "10", "25", "50", "100"],
            default=10
        ))

        self.console.print()
        self.console.print("  [chrome.header]Run notes[/chrome.header]")
        self.console.print("  [chrome]Free-text header for the report — describe the hardware state[/chrome]")
        self.console.print("  [chrome]under test (e.g. 'fan 100%, CPU boost off, freshly booted'). Blank OK.[/chrome]")
        notes = Prompt.ask("[prompt]Notes[/prompt]", default="")

        self.console.print(f"\n  [status.info]Starting: {iterations} runs of the same prompt...[/status.info]")
        self.console.print(f"  [chrome]{self.server['url']} | {self.model}[/chrome]")
        if notes:
            self.console.print(f"  [chrome]Notes: {notes}[/chrome]")
        self.console.print()

        try:
            layout = self._create_layout(f"Consistency | {iterations} runs")

            with Live(layout, console=self.console, refresh_per_second=4) as live:
                async def update_callback(result: TestResult):
                    existing = next((r for r in self.current_results if r.request_id == result.request_id), None)
                    if existing:
                        idx = self.current_results.index(existing)
                        self.current_results[idx] = result
                    else:
                        self.current_results.append(result)
                    self._update_layout(layout, self.tester.stats)

                self.current_stats = await self.tester.run_consistency_test(
                    prompt, iterations, notes, update_callback=update_callback
                )

            self.current_results = self.tester.results

        except Exception as e:
            self.console.print(f"\n  [status.error]Test failed: {type(e).__name__} - {str(e)}[/status.error]")
            import traceback
            self.console.print(f"  [chrome]{traceback.format_exc()}[/chrome]")
            raise

    def _pick_prompt(self) -> str:
        """Pick one fixed prompt from the bank. Returns None on cancel."""
        from rich.markup import escape

        self.console.print("  [chrome.header]Prompt bank[/chrome.header]\n")
        self.console.print(format_menu_item("1", "Short", f"{len(StressTester.PROMPTS_SHORT)} prompts, ~5-20 tokens each"))
        self.console.print(format_menu_item("2", "Medium", f"{len(StressTester.PROMPTS_MEDIUM)} prompts, ~50-150 tokens each"))
        self.console.print(format_menu_item("3", "Long", f"{len(StressTester.PROMPTS_LONG)} prompts, ~500-2000 tokens each"))
        self.console.print()

        bank_choice = Prompt.ask(
            "[prompt]Bank[/prompt]",
            choices=["1", "2", "3"],
            default="2"
        )
        bank = {
            "1": StressTester.PROMPTS_SHORT,
            "2": StressTester.PROMPTS_MEDIUM,
            "3": StressTester.PROMPTS_LONG,
        }[bank_choice]

        self.console.print()
        for i, p in enumerate(bank):
            first_line = p.split("\n", 1)[0]
            preview = first_line[:72] + ("..." if len(first_line) > 72 else "")
            self.console.print(f"  [accent]{i + 1}.[/accent] [chrome]{escape(preview)}[/chrome]")
        self.console.print()

        idx = int(IntPrompt.ask(
            "[prompt]Pick prompt #[/prompt]",
            default=1
        )) - 1

        if idx < 0 or idx >= len(bank):
            self.console.print("  [status.error]Invalid selection.[/status.error]")
            return None

        return bank[idx]

    def _create_layout(self, title: str) -> Layout:
        """Create layout for stress test display."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=3),
            Layout(name="footer", ratio=1)
        )
        layout["main"].split_row(
            Layout(name="conversations", ratio=2),
            Layout(name="errors", ratio=1)
        )

        layout["header"].update(Panel(title, style="accent.bold"))
        layout["conversations"].update(Panel("[chrome]Waiting for results...[/chrome]", title="Conversations"))
        layout["errors"].update(Panel("[chrome]No errors yet[/chrome]", title="Error Log"))
        layout["footer"].update(self._create_stats_panel(TestStats()))

        return layout

    def _update_layout(self, layout: Layout, stats: TestStats):
        """Update layout with current results."""
        layout["conversations"].update(self._create_conversations_panel())
        layout["footer"].update(self._create_stats_panel(stats))
        layout["errors"].update(self._create_errors_panel(stats))

    def _create_conversations_panel(self) -> Panel:
        """Create panel showing conversation status."""
        if not self.current_results:
            return Panel("[chrome]No conversations yet...[/chrome]", title="Conversations")

        max_display = 24
        display_results = self.current_results[-max_display:]

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_column(justify="left")

        for i in range(0, len(display_results), 3):
            row_results = display_results[i:i+3]
            cells = []

            for result in row_results:
                status_icon = {
                    "pending": "\u23f8",
                    "running": "\u23f3",
                    "success": "\u2713",
                    "error": "\u2717"
                }.get(result.status, "?")

                status_style = {
                    "pending": "chrome",
                    "running": "status.running",
                    "success": "status.ok",
                    "error": "status.error"
                }.get(result.status, "chrome")

                cell = Text()
                cell.append(f"#{result.request_id} ", style="accent.bold")
                cell.append(status_icon, style=status_style)
                cell.append("\n", style="chrome")

                if result.status == "success":
                    cell.append(f"{result.duration:.1f}s  {result.decode_tps:.0f} t/s", style="metric.label")
                    if result.ttft > 0:
                        cell.append(f"  TTFT {result.ttft:.2f}s", style="chrome")
                elif result.status == "error":
                    error_short = result.error_msg[:22] + "..." if len(result.error_msg) > 22 else result.error_msg
                    cell.append(error_short, style="status.error")
                elif result.status == "running":
                    cell.append("streaming...", style="status.running")
                else:
                    cell.append("waiting...", style="chrome")

                cells.append(Panel(cell, width=28, height=5))

            while len(cells) < 3:
                cells.append(Panel("", width=28, height=5, border_style="chrome.border"))

            table.add_row(*cells)

        return Panel(table, title=f"Conversations ({len(display_results)} of {len(self.current_results)})")

    def _create_stats_panel(self, stats: TestStats) -> Panel:
        """Create statistics panel."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="chrome.header")
        table.add_column()

        table.add_row("Total:", str(stats.total))
        table.add_row("Completed:", f"{stats.completed}/{stats.total}")
        table.add_row("Success:", f"[status.ok]{stats.success}[/status.ok]")
        table.add_row("Failed:", f"[status.error]{stats.failed}[/status.error]")

        if stats.avg_response_time > 0:
            table.add_row("Avg Response:", f"{stats.avg_response_time:.2f}s")
        if stats.avg_decode_tps > 0:
            table.add_row("Decode TPS:", f"[metric]{stats.avg_decode_tps:.1f}[/metric]")
        if stats.avg_ttft > 0:
            table.add_row("Avg TTFT:", f"{stats.avg_ttft:.2f}s")
        if stats.total_throughput_tps > 0:
            suffix = " [chrome]~est[/chrome]" if stats.throughput_estimated else ""
            table.add_row("Throughput:", f"[metric]{stats.total_throughput_tps:.1f} t/s[/metric]{suffix}")

        return Panel(table, title="Statistics", border_style="chrome.border")

    def _create_errors_panel(self, stats: TestStats) -> Panel:
        """Create errors panel."""
        if not stats.errors:
            return Panel("[chrome]No errors[/chrome]", title="Error Log", border_style="status.ok")

        recent_errors = stats.errors[-10:]
        error_text = "\n".join(recent_errors)

        return Panel(error_text, title=f"Error Log ({len(stats.errors)} total)", border_style="status.error")

    def _show_summary(self):
        """Show test summary."""
        s = self.current_stats
        self.console.print("\n  [status.ok]Test Complete[/status.ok]\n")

        # Run-notes banner (consistency test or any run with notes).
        if s.notes:
            from rich.markup import escape
            self.console.print(Panel(
                f"[chrome]{escape(s.notes)}[/chrome]",
                title="Run notes",
                border_style="accent",
            ))
            self.console.print()

        # Main results
        summary = Table(title="Test Summary", show_header=True, header_style="accent.bold")
        summary.add_column("Metric", style="chrome.header")
        summary.add_column("Value", justify="right")

        summary.add_row("Total Requests", str(s.total))
        summary.add_row("Successful", f"[status.ok]{s.success}[/status.ok]")
        summary.add_row("Failed", f"[status.error]{s.failed}[/status.error]")
        summary.add_row("Success Rate", f"{(s.success / s.total * 100):.1f}%" if s.total > 0 else "N/A")
        summary.add_row("Max Tokens", str(self.max_tokens))

        self.console.print(summary)

        # Performance metrics
        perf = Table(title="Performance", show_header=True, header_style="accent.bold")
        perf.add_column("Metric", style="chrome.header")
        perf.add_column("Value", justify="right")

        if s.avg_decode_tps > 0:
            perf.add_row("Avg Decode TPS", f"[metric]{s.avg_decode_tps:.1f}[/metric]")
        if s.total_throughput_tps > 0:
            label = "System Throughput"
            if s.throughput_estimated:
                label += " (est.)"
            perf.add_row(label, f"[metric]{s.total_throughput_tps:.1f} t/s[/metric]")
        if s.avg_ttft > 0:
            perf.add_row("Avg TTFT", f"{s.avg_ttft:.2f}s")
        perf.add_row("Avg Response Time", f"{s.avg_response_time:.2f}s")
        if s.total_output_tokens > 0:
            perf.add_row("Total Output Tokens", str(s.total_output_tokens))
        if s.wall_clock_time > 0:
            perf.add_row("Wall Clock", f"{s.wall_clock_time:.1f}s")
        if s.max_concurrent > 0:
            perf.add_row("Peak Concurrency", str(s.max_concurrent))

        self.console.print(perf)

        # Per-size breakdown for token stress mode: render when we can see
        # that prompt sizes varied across successful requests.
        successful = [r for r in self.current_results if r.status == "success"]
        prompt_tokens_seen = {r.prompt_tokens for r in successful if r.prompt_tokens > 0}
        if len(prompt_tokens_seen) >= 2:
            breakdown = Table(title="Per-Request Breakdown", show_header=True, header_style="accent.bold")
            breakdown.add_column("#", style="chrome", justify="right")
            breakdown.add_column("Prompt tok", justify="right")
            breakdown.add_column("Output tok", justify="right")
            breakdown.add_column("TTFT", justify="right")
            breakdown.add_column("Decode TPS", justify="right", style="metric")
            breakdown.add_column("Duration", justify="right")

            for r in successful:
                breakdown.add_row(
                    str(r.request_id),
                    str(r.prompt_tokens) if r.prompt_tokens else "—",
                    str(r.completion_tokens or r.token_count),
                    f"{r.ttft:.2f}s" if r.ttft > 0 else "—",
                    f"{r.decode_tps:.1f}" if r.decode_tps > 0 else "—",
                    f"{r.duration:.2f}s",
                )

            self.console.print(breakdown)

        # Percentiles (only if we have enough data)
        if s.p50_response_time > 0:
            pct = Table(title="Latency Distribution", show_header=True, header_style="accent.bold")
            pct.add_column("Metric", style="chrome.header")
            pct.add_column("p50", justify="right")
            pct.add_column("p95", justify="right")
            pct.add_column("p99", justify="right")

            pct.add_row(
                "Response Time",
                f"{s.p50_response_time:.2f}s",
                f"{s.p95_response_time:.2f}s" if s.p95_response_time > 0 else "—",
                f"{s.p99_response_time:.2f}s" if s.p99_response_time > 0 else "—",
            )
            if s.p50_ttft > 0:
                pct.add_row(
                    "TTFT",
                    f"{s.p50_ttft:.3f}s",
                    f"{s.p95_ttft:.3f}s" if s.p95_ttft > 0 else "—",
                    "—",
                )
            if s.p50_decode_tps > 0:
                pct.add_row(
                    "Decode TPS",
                    f"{s.p50_decode_tps:.1f}",
                    "—",
                    "—",
                )

            self.console.print(pct)

        # Variance / spread — emitted whenever we have enough samples. Most
        # useful for the consistency test (same prompt, serial), where the
        # remaining variance is attributable to environmental noise.
        if s.stddev_decode_tps > 0 or s.stddev_ttft > 0:
            var = Table(title="Variance (hardware noise)", show_header=True, header_style="accent.bold")
            var.add_column("Metric", style="chrome.header")
            var.add_column("Mean", justify="right")
            var.add_column("Stddev", justify="right")
            var.add_column("Min", justify="right")
            var.add_column("Max", justify="right")
            var.add_column("CV%", justify="right", style="metric")

            def _cv(mean: float, sd: float) -> str:
                return f"{(sd / mean * 100):.2f}%" if mean > 0 else "—"

            if s.avg_decode_tps > 0:
                var.add_row(
                    "Decode TPS",
                    f"{s.avg_decode_tps:.1f}",
                    f"{s.stddev_decode_tps:.2f}",
                    f"{s.min_decode_tps:.1f}",
                    f"{s.max_decode_tps:.1f}",
                    _cv(s.avg_decode_tps, s.stddev_decode_tps),
                )
            if s.avg_ttft > 0:
                var.add_row(
                    "TTFT",
                    f"{s.avg_ttft:.3f}s",
                    f"{s.stddev_ttft:.3f}s",
                    f"{s.min_ttft:.3f}s",
                    f"{s.max_ttft:.3f}s",
                    _cv(s.avg_ttft, s.stddev_ttft),
                )
            if s.avg_response_time > 0 and s.stddev_response_time > 0:
                var.add_row(
                    "Response",
                    f"{s.avg_response_time:.2f}s",
                    f"{s.stddev_response_time:.2f}s",
                    "—", "—",
                    _cv(s.avg_response_time, s.stddev_response_time),
                )

            self.console.print(var)

        # Drift — first-half vs second-half means. Thermal throttling or
        # driver warmup typically show up as a negative delta in TPS / positive
        # delta in TTFT across the run.
        if s.first_half_decode_tps > 0 and s.second_half_decode_tps > 0:
            drift = Table(title="Drift (first half vs second half)", show_header=True, header_style="accent.bold")
            drift.add_column("Metric", style="chrome.header")
            drift.add_column("First half", justify="right")
            drift.add_column("Second half", justify="right")
            drift.add_column("Delta", justify="right", style="metric")

            def _delta(first: float, second: float) -> str:
                if first <= 0:
                    return "—"
                pct = (second - first) / first * 100
                sign = "+" if pct >= 0 else ""
                return f"{sign}{pct:.2f}%"

            drift.add_row(
                "Decode TPS",
                f"{s.first_half_decode_tps:.1f}",
                f"{s.second_half_decode_tps:.1f}",
                _delta(s.first_half_decode_tps, s.second_half_decode_tps),
            )
            if s.first_half_ttft > 0 and s.second_half_ttft > 0:
                drift.add_row(
                    "TTFT",
                    f"{s.first_half_ttft:.3f}s",
                    f"{s.second_half_ttft:.3f}s",
                    _delta(s.first_half_ttft, s.second_half_ttft),
                )

            self.console.print(drift)

        if s.errors:
            self.console.print(f"\n  [status.error]Encountered {len(s.errors)} errors[/status.error]")
            self.console.print("  [chrome]Check logs/stress_test_*.log for details[/chrome]")
