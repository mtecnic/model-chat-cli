"""Discovery view for scanning and selecting models."""
import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt

from scanner import scan_network, check_server_health, load_cache, save_cache, quick_validate_cache
from ui.components import create_model_table, _model_label


class DiscoveryView:
    """Handle model discovery and selection."""

    def __init__(self, console: Console):
        self.console = console
        self.servers = []
        self.models = []  # List of (server, model_name) tuples

    async def run(self) -> tuple:
        """Run the discovery process and return selected model."""
        self.console.print()
        self.console.rule("[accent.bold]Model Discovery[/accent.bold]", style="chrome.border")
        self.console.print()

        # Try cache first
        cached_servers = load_cache()

        if cached_servers:
            self.console.print("  [status.info]Validating cached servers...[/status.info]")
            validated_servers = await self._validate_cache(cached_servers)

            if validated_servers:
                self.servers = validated_servers
                save_cache(validated_servers)
                self._display_servers(validated_servers)
            else:
                self.servers = await self._scan_network()
                if self.servers:
                    save_cache(self.servers)
                    self._display_servers(self.servers)
        else:
            self.servers = await self._scan_network()
            if self.servers:
                save_cache(self.servers)
                self._display_servers(self.servers)

        if not self.servers:
            self.console.print("  [status.warn]No models found on local network.[/status.warn]")
            return None, None

        # Model selection loop
        while True:
            self.models = []
            for server in self.servers:
                for model in server.get("models", []):
                    # models are dicts ({name, max_context}); selection uses the name
                    self.models.append((server, _model_label(model)))

            self.console.print()
            valid_choices = [str(i) for i in range(1, len(self.models) + 1)] + ['r', 'R']
            choice = Prompt.ask(
                "[prompt]Select model (number) or [accent]R[/accent] to rescan[/prompt]",
                choices=valid_choices
            )

            if choice.upper() == 'R':
                self.console.print()
                self.servers = await self._scan_network()
                if self.servers:
                    save_cache(self.servers)
                    self._display_servers(self.servers)
                else:
                    self.console.print("  [status.warn]No models found on local network.[/status.warn]")
                    return None, None
                continue

            server, model = self.models[int(choice) - 1]
            return server, model

    async def _scan_network(self) -> list:
        """Perform full network scan with progress display."""
        self.console.print("  [status.info]Scanning local network...[/status.info]\n")

        servers = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Scanning network...", total=100)

            async def update_progress(current, total):
                progress.update(task, completed=current, total=total)

            servers = await scan_network(progress_callback=update_progress)

        self.console.print()
        return servers

    async def _validate_cache(self, cached_servers: list) -> list:
        """Validate cached servers with progress."""
        validated_servers = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console
        ) as progress:
            task = progress.add_task("Validating cache...", total=len(cached_servers))

            async def update_progress(current, total):
                progress.update(task, completed=current, total=total)

            validated_servers = await quick_validate_cache(
                cached_servers,
                progress_callback=update_progress
            )

        return validated_servers

    def _display_servers(self, servers: list):
        """Display discovered servers as a table."""
        if not servers:
            return

        table = create_model_table(servers)
        self.console.print(table)
        self.console.print()
        self.console.print("[chrome]Ctrl+C to quit[/chrome]", justify="center")
