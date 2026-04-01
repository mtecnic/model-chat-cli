#!/usr/bin/env python3
"""
Model Chat CLI - A modern terminal interface for discovering and chatting with local AI models.
"""
import asyncio
from enum import Enum
from rich.console import Console

from ui.theme import APP_THEME
from ui.discovery import DiscoveryView
from ui.chat import ChatView
from ui.stress_test import StressTestView
from ui.arena import ArenaView
from ui.multi_arena import MultiArenaView


class AppState(Enum):
    """Application states."""
    DISCOVERY = "discovery"
    CHAT = "chat"
    STRESS_TEST = "stress_test"
    PROMPT_ARENA = "prompt_arena"
    ARENA = "arena"
    QUIT = "quit"


class ModelChatCLI:
    """Main application controller."""

    def __init__(self):
        """Initialize the CLI application."""
        self.console = Console(theme=APP_THEME)
        self.state = AppState.DISCOVERY
        self.selected_server = None
        self.selected_model = None

    async def run(self):
        """Run the main application loop."""
        try:
            while self.state != AppState.QUIT:
                if self.state == AppState.DISCOVERY:
                    await self._discovery_mode()
                elif self.state == AppState.CHAT:
                    await self._chat_mode()
                elif self.state == AppState.STRESS_TEST:
                    await self._stress_test_mode()
                elif self.state == AppState.PROMPT_ARENA:
                    await self._prompt_arena_mode()
                elif self.state == AppState.ARENA:
                    await self._arena_mode()

        except KeyboardInterrupt:
            self.console.print("\n[chrome]Goodbye![/chrome]")
        except Exception as e:
            self.console.print(f"\n[status.error]An error occurred: {e}[/status.error]")
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
                self.state = AppState.QUIT

        except KeyboardInterrupt:
            self.state = AppState.QUIT

    async def _chat_mode(self):
        """Handle chat mode - interact with selected model."""
        chat = ChatView(self.console, self.selected_server, self.selected_model)

        try:
            result = await chat.run()

            if result == "switch":
                self.state = AppState.DISCOVERY
            elif result == "stress_test":
                self.state = AppState.STRESS_TEST
            elif result == "prompt_arena":
                self.state = AppState.PROMPT_ARENA
            elif result == "arena":
                self.state = AppState.ARENA
            else:
                self.state = AppState.QUIT

        except KeyboardInterrupt:
            self.state = AppState.DISCOVERY

    async def _stress_test_mode(self):
        """Handle stress test mode - load test the selected model."""
        stress_test = StressTestView(self.console, self.selected_server, self.selected_model)

        try:
            result = await stress_test.run()

            if result == "back":
                self.state = AppState.CHAT
            else:
                self.state = AppState.QUIT

        except KeyboardInterrupt:
            self.state = AppState.CHAT

    async def _prompt_arena_mode(self):
        """Handle prompt arena mode - compare system prompts."""
        arena = ArenaView(self.console, self.selected_server, self.selected_model)

        try:
            result = await arena.run()

            if result == "back":
                self.state = AppState.CHAT
            else:
                self.state = AppState.QUIT

        except KeyboardInterrupt:
            self.state = AppState.CHAT

    async def _arena_mode(self):
        """Handle multi-server arena mode - compare models side by side."""
        arena = MultiArenaView(self.console, self.selected_server, self.selected_model)

        try:
            result = await arena.run()

            if result == "back":
                self.state = AppState.CHAT
            else:
                self.state = AppState.QUIT

        except KeyboardInterrupt:
            self.state = AppState.CHAT


def main():
    """Entry point for the application."""
    app = ModelChatCLI()
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
