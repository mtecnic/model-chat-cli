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
        self.tester = StressTester(server, model)
        self.current_results: List[TestResult] = []
        self.current_stats = TestStats()

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

        if mode == 1:
            await self._run_throughput_mode()
        elif mode == 2:
            await self._run_token_stress_mode()
        elif mode == 3:
            await self._run_sustained_load_mode()

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
        self.console.print("  [chrome]Q.[/chrome] Cancel\n")

        try:
            choice = Prompt.ask(
                "[prompt]Select mode[/prompt]",
                choices=["1", "2", "3", "q", "Q"]
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
                    cell.append(f"{result.duration:.1f}s  {result.tokens_per_sec:.0f} t/s", style="metric.label")
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
        if stats.avg_tps > 0:
            table.add_row("Avg TPS:", f"[metric]{stats.avg_tps:.1f}[/metric]")

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
        self.console.print("\n  [status.ok]Test Complete[/status.ok]\n")

        summary = Table(title="Test Summary", show_header=True, header_style="accent.bold")
        summary.add_column("Metric", style="chrome.header")
        summary.add_column("Value", justify="right")

        summary.add_row("Total Requests", str(self.current_stats.total))
        summary.add_row("Successful", f"[status.ok]{self.current_stats.success}[/status.ok]")
        summary.add_row("Failed", f"[status.error]{self.current_stats.failed}[/status.error]")
        summary.add_row("Success Rate", f"{(self.current_stats.success / self.current_stats.total * 100):.1f}%" if self.current_stats.total > 0 else "N/A")
        summary.add_row("Avg Response Time", f"{self.current_stats.avg_response_time:.2f}s")
        summary.add_row("Avg Tokens/sec", f"[metric]{self.current_stats.avg_tps:.1f}[/metric]")

        self.console.print(summary)

        if self.current_stats.errors:
            self.console.print(f"\n  [status.error]Encountered {len(self.current_stats.errors)} errors[/status.error]")
            self.console.print("  [chrome]Check logs/stress_test_*.log for details[/chrome]")
