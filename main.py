#!/usr/bin/env python3
"""
Model Chat CLI - A modern terminal interface for discovering and chatting with local AI models.
"""
import asyncio
from enum import Enum
from rich.console import Console

from ui.theme import get_theme, load_theme
from ui.discovery import DiscoveryView
from ui.chat import ChatView
from ui.stress_test import StressTestView
from ui.effects import create_banner, animate_banner_fade


class AppState(Enum):
    """Application states."""
    DISCOVERY = "discovery"
    CHAT = "chat"
    STRESS_TEST = "stress_test"
    QUIT = "quit"


class ModelChatCLI:
    """Main application controller."""

    def __init__(self):
        """Initialize the CLI application."""
        # Load theme preference
        theme_name = load_theme()
        current_theme = get_theme(theme_name)
        self.console = Console(theme=current_theme)
        self.state = AppState.DISCOVERY
        self.selected_server = None
        self.selected_model = None
        self.show_banner = True  # Show banner on first run

    async def run(self):
        """Run the main application loop."""
        try:
            # Show animated banner on first run
            if self.show_banner:
                banner = create_banner("MODEL CHAT CLI", font="slant")
                await animate_banner_fade(self.console, banner, duration=0.8)
                self.show_banner = False
                await asyncio.sleep(0.5)

            while self.state != AppState.QUIT:
                if self.state == AppState.DISCOVERY:
                    await self._discovery_mode()
                elif self.state == AppState.CHAT:
                    await self._chat_mode()
                elif self.state == AppState.STRESS_TEST:
                    await self._stress_test_mode()

        except KeyboardInterrupt:
            self.console.print("\n[info]Goodbye![/info]")
        except Exception as e:
            self.console.print(f"\n[error]An error occurred: {e}[/error]")
            raise

    async def _discovery_mode(self):
        """Handle discovery mode - scan network and select model."""
        discovery = DiscoveryView(self.console)

        try:
            server, model = await discovery.run()

            if server and model:
                self.selected_server = server
                self.selected_model = model
                self.state = AppState.CHAT
            else:
                # No models found or user cancelled
                self.state = AppState.QUIT

        except KeyboardInterrupt:
            self.state = AppState.QUIT

    async def _chat_mode(self):
        """Handle chat mode - interact with selected model."""
        chat = ChatView(self.console, self.selected_server, self.selected_model)

        try:
            result = await chat.run()

            if result == "switch":
                # User wants to switch models
                self.state = AppState.DISCOVERY
            elif result == "stress_test":
                # User wants to run stress tests
                self.state = AppState.STRESS_TEST
            else:
                # User quit
                self.state = AppState.QUIT

        except KeyboardInterrupt:
            # Return to discovery on Ctrl+C
            self.state = AppState.DISCOVERY

    async def _stress_test_mode(self):
        """Handle stress test mode - load test the selected model."""
        stress_test = StressTestView(self.console, self.selected_server, self.selected_model)

        try:
            result = await stress_test.run()

            if result == "back":
                # Return to chat
                self.state = AppState.CHAT
            else:
                # User quit
                self.state = AppState.QUIT

        except KeyboardInterrupt:
            # Return to chat on Ctrl+C
            self.state = AppState.CHAT


def main():
    """Entry point for the application."""
    app = ModelChatCLI()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
