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


class StressTestView:
    """Handle stress test interface and visualization."""

    def __init__(self, console: Console, server: dict, model: str):
        """Initialize stress test view.

        Args:
            console: Rich Console instance
            server: Server dictionary
            model: Model name
        """
        self.console = console
        self.server = server
        self.model = model
        self.tester = StressTester(server, model)
        self.current_results: List[TestResult] = []
        self.current_stats = TestStats()

    async def run(self) -> str:
        """Run the stress test interface.

        Returns:
            Control signal ("back" or "quit")
        """
        # Show title
        self.console.clear()
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]Stress Test[/bold cyan] - {self.model}",
                style="title"
            ),
            justify="center"
        )
        self.console.print()

        # Mode selection
        mode = self._select_mode()
        if mode is None:
            return "back"

        # Get parameters and run test
        if mode == 1:
            await self._run_throughput_mode()
        elif mode == 2:
            await self._run_token_stress_mode()
        elif mode == 3:
            await self._run_sustained_load_mode()

        # Show summary
        self._show_summary()

        # Wait for user to return
        self.console.print("\n[dim]Press Enter to return to chat...[/dim]")
        input()

        return "back"

    def _select_mode(self) -> int:
        """Show mode selection menu.

        Returns:
            Selected mode number (1-3) or None to cancel
        """
        self.console.print("[bold]Select Stress Test Mode:[/bold]\n")
        self.console.print("  [cyan]1.[/cyan] Throughput Test - Concurrent requests")
        self.console.print("     [dim]Test server's ability to handle multiple simultaneous users[/dim]\n")

        self.console.print("  [cyan]2.[/cyan] Token Stress Test - Long context handling")
        self.console.print("     [dim]Test performance with increasing prompt lengths[/dim]\n")

        self.console.print("  [cyan]3.[/cyan] Sustained Load Test - Endurance testing")
        self.console.print("     [dim]Test stability over extended period[/dim]\n")

        self.console.print("  [dim]Q.[/dim] Cancel and return\n")

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
        self.console.print("\n[bold cyan]Throughput Test Mode[/bold cyan]\n")
        self.console.print("[dim]Spawns multiple concurrent requests to test parallelization[/dim]\n")

        # Get parameters
        num_requests = int(IntPrompt.ask(
            "[prompt]Number of concurrent requests[/prompt]",
            choices=["5", "10", "20", "50"],
            default=10
        ))

        self.console.print(f"\n[info]Starting test with {num_requests} concurrent requests...[/info]")
        self.console.print(f"[dim]Server: {self.server['url']} | Model: {self.model}[/dim]\n")

        try:
            # Create layout for live display
            layout = self._create_layout(f"Throughput Test | {num_requests} concurrent")

            # Run test with live updates
            with Live(layout, console=self.console, refresh_per_second=4) as live:
                async def update_callback(result: TestResult):
                    # Update current results
                    existing = next((r for r in self.current_results if r.request_id == result.request_id), None)
                    if existing:
                        idx = self.current_results.index(existing)
                        self.current_results[idx] = result
                    else:
                        self.current_results.append(result)

                    # Update layout
                    self._update_layout(layout, self.tester.stats)

                self.current_stats = await self.tester.run_throughput_test(
                    num_requests,
                    update_callback=update_callback
                )

            self.current_results = self.tester.results

        except Exception as e:
            self.console.print(f"\n[error]Test failed: {type(e).__name__} - {str(e)}[/error]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise

    async def _run_token_stress_mode(self):
        """Run token stress test mode."""
        self.console.print("\n[bold cyan]Token Stress Test Mode[/bold cyan]\n")
        self.console.print("[dim]Tests performance with progressively longer prompts[/dim]\n")

        # Get parameters
        self.console.print("Available token sizes:")
        self.console.print("  [cyan]1.[/cyan] Quick test: 500, 1000, 2000")
        self.console.print("  [cyan]2.[/cyan] Standard test: 500, 1000, 2000, 5000")
        self.console.print("  [cyan]3.[/cyan] Extended test: 500, 1000, 2000, 5000, 10000\n")

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

        self.console.print(f"\n[info]Starting test with token sizes: {token_sizes}...[/info]")
        self.console.print(f"[dim]Server: {self.server['url']} | Model: {self.model}[/dim]\n")

        try:
            # Create layout
            layout = self._create_layout(f"Token Stress Test | {len(token_sizes)} sizes")

            # Run test with live updates
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
                    token_sizes,
                    update_callback=update_callback
                )

            self.current_results = self.tester.results

        except Exception as e:
            self.console.print(f"\n[error]Test failed: {type(e).__name__} - {str(e)}[/error]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise

    async def _run_sustained_load_mode(self):
        """Run sustained load test mode."""
        self.console.print("\n[bold cyan]Sustained Load Test Mode[/bold cyan]\n")
        self.console.print("[dim]Tests stability with continuous requests over time[/dim]\n")

        # Get parameters
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

        self.console.print(f"\n[info]Starting test: {duration}min @ {rate} req/min ({total_requests} total)...[/info]")
        self.console.print(f"[dim]Server: {self.server['url']} | Model: {self.model}[/dim]\n")

        try:
            # Create layout
            layout = self._create_layout(f"Sustained Load | {duration}min @ {rate}/min")

            # Run test with live updates
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
                    duration,
                    rate,
                    update_callback=update_callback
                )

            self.current_results = self.tester.results

        except Exception as e:
            self.console.print(f"\n[error]Test failed: {type(e).__name__} - {str(e)}[/error]")
            import traceback
            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")
            raise

    def _create_layout(self, title: str) -> Layout:
        """Create layout for stress test display.

        Args:
            title: Test title

        Returns:
            Layout object
        """
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

        # Initialize with placeholders
        layout["header"].update(Panel(title, style="bold cyan"))
        layout["conversations"].update(Panel("[dim]Waiting for results...[/dim]", title="Conversations"))
        layout["errors"].update(Panel("[dim]No errors yet[/dim]", title="Error Log"))
        layout["footer"].update(self._create_stats_panel(TestStats()))

        return layout

    def _update_layout(self, layout: Layout, stats: TestStats):
        """Update layout with current results.

        Args:
            layout: Layout to update
            stats: Current test statistics
        """
        # Update conversations panel
        layout["conversations"].update(self._create_conversations_panel())

        # Update stats panel
        layout["footer"].update(self._create_stats_panel(stats))

        # Update errors panel
        layout["errors"].update(self._create_errors_panel(stats))

    def _create_conversations_panel(self) -> Panel:
        """Create panel showing conversation status.

        Returns:
            Panel with conversation grid
        """
        if not self.current_results:
            return Panel("[dim]No conversations yet...[/dim]", title="Conversations")

        # Show up to 24 most recent conversations (8 rows x 3 columns)
        # This allows seeing up to 24 concurrent requests at once
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
                    "pending": "⏸",
                    "running": "⏳",
                    "success": "✓",
                    "error": "✗"
                }.get(result.status, "?")

                status_style = {
                    "pending": "dim",
                    "running": "cyan",
                    "success": "green",
                    "error": "red"
                }.get(result.status, "white")

                # Create cell content
                cell = Text()
                cell.append(f"#{result.request_id} ", style="bold")
                cell.append(status_icon, style=status_style)
                cell.append("\n", style="dim")

                if result.status == "success":
                    cell.append(f"{result.duration:.1f}s, {result.tokens_per_sec:.0f}t/s", style="dim")
                elif result.status == "error":
                    error_short = result.error_msg[:20] + "..." if len(result.error_msg) > 20 else result.error_msg
                    cell.append(error_short, style="red dim")
                elif result.status == "running":
                    cell.append("Streaming...", style="cyan dim")
                else:
                    cell.append("Waiting...", style="dim")

                cells.append(Panel(cell, width=20, height=4))

            # Pad row if needed
            while len(cells) < 3:
                cells.append(Panel("", width=20, height=4, border_style="dim"))

            table.add_row(*cells)

        return Panel(table, title=f"Conversations (showing {len(display_results)} of {len(self.current_results)})")

    def _create_stats_panel(self, stats: TestStats) -> Panel:
        """Create statistics panel.

        Args:
            stats: Test statistics

        Returns:
            Panel with stats table
        """
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()

        table.add_row("Total:", str(stats.total))
        table.add_row("Completed:", f"{stats.completed}/{stats.total}")
        table.add_row("Success:", f"[green]{stats.success}[/green]")
        table.add_row("Failed:", f"[red]{stats.failed}[/red]")

        if stats.avg_response_time > 0:
            table.add_row("Avg Response:", f"{stats.avg_response_time:.2f}s")
        if stats.avg_tps > 0:
            table.add_row("Avg TPS:", f"[yellow]{stats.avg_tps:.1f}[/yellow]")

        return Panel(table, title="Statistics", border_style="blue")

    def _create_errors_panel(self, stats: TestStats) -> Panel:
        """Create errors panel.

        Args:
            stats: Test statistics

        Returns:
            Panel with error log
        """
        if not stats.errors:
            return Panel("[dim]No errors[/dim]", title="Error Log", border_style="green")

        # Show last 10 errors
        recent_errors = stats.errors[-10:]
        error_text = "\n".join(recent_errors)

        return Panel(error_text, title=f"Error Log ({len(stats.errors)} total)", border_style="red")

    def _show_summary(self):
        """Show test summary."""
        self.console.print("\n[bold green]Test Complete![/bold green]\n")

        # Summary table
        summary = Table(title="Test Summary", show_header=True, header_style="bold cyan")
        summary.add_column("Metric", style="bold")
        summary.add_column("Value", justify="right")

        summary.add_row("Total Requests", str(self.current_stats.total))
        summary.add_row("Successful", f"[green]{self.current_stats.success}[/green]")
        summary.add_row("Failed", f"[red]{self.current_stats.failed}[/red]")
        summary.add_row("Success Rate", f"{(self.current_stats.success / self.current_stats.total * 100):.1f}%" if self.current_stats.total > 0 else "N/A")
        summary.add_row("Avg Response Time", f"{self.current_stats.avg_response_time:.2f}s")
        summary.add_row("Avg Tokens/sec", f"[yellow]{self.current_stats.avg_tps:.1f}[/yellow]")

        self.console.print(summary)

        # Show errors if any
        if self.current_stats.errors:
            self.console.print(f"\n[red]Encountered {len(self.current_stats.errors)} errors[/red]")
            self.console.print("[dim]Check logs/stress_test_*.log for details[/dim]")
