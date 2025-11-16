"""Discovery view for scanning and selecting models."""
import asyncio
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

from scanner import scan_network, check_server_health, load_cache, save_cache, quick_validate_cache, load_favorites, save_favorite, is_favorite
from ui.components import create_model_table
from ui.effects import create_gradient_text, create_glass_panel


class DiscoveryView:
    """Handle model discovery and selection."""

    def __init__(self, console: Console):
        """Initialize discovery view.

        Args:
            console: Rich Console instance
        """
        self.console = console
        self.servers = []
        self.models = []  # List of (server, model_name) tuples

    async def run(self) -> tuple:
        """Run the discovery process and return selected model.

        Returns:
            Tuple of (server_dict, model_name)
        """
        # Show title with gradient
        self.console.print()
        title = create_gradient_text("✨ MODEL DISCOVERY ✨", ['cyan', 'bright_cyan', 'blue', 'bright_blue'])
        self.console.print(
            create_glass_panel(title, accent_color="cyan"),
            justify="center"
        )
        self.console.print()

        # Check for favorites first
        favorites = load_favorites()
        if favorites:
            self.console.print(f"[success]Found {len(favorites)} favorite(s)[/success]")
            self.console.print()
            use_favorites = Prompt.ask(
                "[prompt]Use favorites? (Y/n)[/prompt]",
                choices=["y", "Y", "n", "N", ""],
                default="y"
            )

            if use_favorites.lower() in ["y", ""]:
                # Use favorites
                self.servers = [fav["server"] for fav in favorites]
                self._display_servers(self.servers)
            else:
                # Proceed with normal discovery
                self.console.print()
                cached_servers = load_cache()
                if cached_servers:
                    self.console.print("[info]Validating cached servers...[/info]")
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
        else:
            # No favorites, try cache first
            cached_servers = load_cache()

            if cached_servers:
                # Automatically validate and use cached servers
                self.console.print("[info]Validating cached servers...[/info]")
                validated_servers = await self._validate_cache(cached_servers)

                if validated_servers:
                    self.servers = validated_servers
                    save_cache(validated_servers)
                    self._display_servers(validated_servers)
                else:
                    # Cache validation failed, do full scan
                    self.servers = await self._scan_network()
                    if self.servers:
                        save_cache(self.servers)
                        self._display_servers(self.servers)
            else:
                # No cache, do full network scan
                self.servers = await self._scan_network()
                if self.servers:
                    save_cache(self.servers)
                    self._display_servers(self.servers)

        # No models found
        if not self.servers:
            self.console.print("[warning]No models found on local network[/warning]")
            return None, None

        # Model selection loop (allows rescanning with 'R' and filtering with 'F')
        filtered_servers = self.servers
        filter_text = ""

        while True:
            # Build model list from filtered servers
            self.models = []
            for server in filtered_servers:
                for model in server.get("models", []):
                    self.models.append((server, model))

            # Prompt for selection
            self.console.print()
            if filter_text:
                self.console.print(f"[dim]Filter active: '{filter_text}' ({len(self.models)} models)[/dim]")

            valid_choices = [str(i) for i in range(1, len(self.models) + 1)] + ['r', 'R', 'f', 'F', 'c', 'C']
            choice = Prompt.ask(
                "[prompt]Select model or [cyan]R[/cyan]:rescan [cyan]F[/cyan]:filter [cyan]C[/cyan]:clear filter[/prompt]",
                choices=valid_choices + ['']  # Allow empty for just showing menu
            )

            # Handle rescan
            if choice.upper() == 'R':
                self.console.print()
                self.servers = await self._scan_network()
                if self.servers:
                    save_cache(self.servers)
                    filtered_servers = self.servers
                    filter_text = ""
                    self._display_servers(self.servers)
                else:
                    self.console.print("[warning]No models found on local network[/warning]")
                    return None, None
                continue

            # Handle filter
            if choice.upper() == 'F':
                filter_text = Prompt.ask("[prompt]Filter by (model name/IP/type)[/prompt]")
                filtered_servers = self._filter_servers(self.servers, filter_text)
                self._display_servers(filtered_servers)
                continue

            # Handle clear filter
            if choice.upper() == 'C':
                filter_text = ""
                filtered_servers = self.servers
                self._display_servers(self.servers)
                continue

            # Return selected model
            if choice and choice.isdigit():
                server, model = self.models[int(choice) - 1]

                # Ask if they want to favorite this model
                if not is_favorite(server["url"], model):
                    self.console.print()
                    add_fav = Prompt.ask(
                        "[prompt]Add this model to favorites? (y/N)[/prompt]",
                        choices=["y", "Y", "n", "N", ""],
                        default="n"
                    )
                    if add_fav.lower() == 'y':
                        save_favorite(server, model)
                        self.console.print("[success]Added to favorites![/success]")

                return server, model

    async def _scan_network(self) -> list:
        """Perform full network scan with progress display.

        Returns:
            List of discovered servers
        """
        self.console.print("[info]Scanning local network...[/info]\n")

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
        """Validate cached servers with progress.

        Args:
            cached_servers: Servers from cache

        Returns:
            Validated servers or empty list if validation fails
        """
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
        """Display discovered servers as a table.

        Args:
            servers: List of server dictionaries
        """
        if not servers:
            return

        table = create_model_table(servers)
        self.console.print(table)
        self.console.print()
        self.console.print("[dim]Press Ctrl+C to quit[/dim]", justify="center")

    def _filter_servers(self, servers: list, filter_text: str) -> list:
        """Filter servers by model name, IP, or type.

        Args:
            servers: List of server dictionaries
            filter_text: Filter string

        Returns:
            Filtered list of servers
        """
        if not filter_text:
            return servers

        filter_lower = filter_text.lower()
        filtered = []

        for server in servers:
            # Check if filter matches IP
            if filter_lower in server['ip'].lower():
                filtered.append(server)
                continue

            # Check if filter matches type
            if filter_lower in server['type'].lower():
                filtered.append(server)
                continue

            # Check if filter matches status
            if filter_lower in server.get('status', '').lower():
                filtered.append(server)
                continue

            # Check if any model name matches
            for model in server.get('models', []):
                if filter_lower in model.lower():
                    filtered.append(server)
                    break

        return filtered
