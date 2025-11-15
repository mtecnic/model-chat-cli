"""Chat view for conversing with selected model."""
import asyncio
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.layout import Layout
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import HTML

from client import ModelClient
from ui.components import create_chat_message, create_header, create_footer, create_typing_indicator


class ChatView:
    """Handle chat interface with streaming responses."""

    def __init__(self, console: Console, server: dict, model: str):
        """Initialize chat view.

        Args:
            console: Rich Console instance
            server: Server dictionary
            model: Model name
        """
        self.console = console
        self.server = server
        self.model = model
        self.client = ModelClient(server, model)
        self.history = []
        self.session = PromptSession()

        # Token per second tracking
        self.tps_samples = []
        self.avg_tps = 0.0

        # Message display tracking
        self.all_messages = []  # List of rendered message panels

        # System prompt
        self.system_prompt = ""  # Optional system prompt

    async def run(self):
        """Run the chat interface."""
        # Initial display with header and welcome message
        self._refresh_display()

        # Chat loop
        while True:
            try:
                # Get user input
                with patch_stdout():
                    user_input = await self.session.prompt_async(
                        HTML('<ansiblue><b>┃ You ▶</b></ansiblue> '),
                        multiline=False
                    )

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    command_result = await self._handle_command(user_input)
                    if command_result == "quit":
                        break
                    elif command_result == "switch":
                        return "switch"  # Signal to main to switch models
                    elif command_result == "stress_test":
                        return "stress_test"  # Signal to main to start stress test
                    continue

                # Add user message to history
                self.history.append({"role": "user", "content": user_input})

                # Add user message to display and refresh
                user_msg = create_chat_message("user", user_input)
                self.all_messages.append(user_msg)
                self._refresh_display()

                # Stream assistant response
                await self._stream_response(user_input)

            except KeyboardInterrupt:
                # Ctrl+C - ask if they want to quit or go back
                self.console.print("\n[dim]Ctrl+C: Quit | Ctrl+D: Back to menu[/dim]")
                try:
                    confirm = await self.session.prompt_async(HTML('<ansigray>Q to quit, M for menu, or Enter to continue: </ansigray>'))
                    confirm = confirm.strip().lower()
                    if confirm == 'q':
                        return "quit"
                    elif confirm == 'm':
                        return "switch"
                except (KeyboardInterrupt, EOFError):
                    return "quit"
                continue
            except EOFError:
                # Ctrl+D - back to menu
                self.console.print("\n[info]Returning to menu...[/info]")
                return "switch"

    async def _stream_response(self, message: str):
        """Stream the assistant's response with live updates.

        Args:
            message: User's message
        """
        import time
        import asyncio

        full_response = ""
        start_time = time.time()
        token_count = 0
        interrupted = False

        # Show typing indicator first
        typing_panel = create_typing_indicator()

        # Build history with system prompt if set
        messages_with_system = self._build_message_history()

        try:
            # Create live display for streaming
            with Live(
                typing_panel,
                console=self.console,
                refresh_per_second=10,
                transient=True  # Remove after streaming done
            ) as live:
                # Stream response chunks
                try:
                    async for chunk in self.client.chat_stream(message, messages_with_system):
                        full_response += chunk
                        # Improved token estimation
                        token_count = self._estimate_tokens(full_response)
                        # Update display with current response
                        live.update(create_chat_message("assistant", full_response))
                except (asyncio.CancelledError, KeyboardInterrupt):
                    # Stream was interrupted
                    interrupted = True
                    full_response += "\n\n[dim italic](interrupted)[/dim italic]"

            # Calculate tokens per second for this response
            elapsed_time = time.time() - start_time
            current_tps = 0
            if elapsed_time > 0 and token_count > 0:
                current_tps = token_count / elapsed_time
                self.tps_samples.append(current_tps)

                # Keep only last 10 samples for rolling average
                if len(self.tps_samples) > 10:
                    self.tps_samples.pop(0)

                # Calculate average
                self.avg_tps = sum(self.tps_samples) / len(self.tps_samples)

            # After Live context exits, add the final message to display
            if full_response:
                final_msg = create_chat_message("assistant", full_response)
                self.all_messages.append(final_msg)

                # Show TPS stats for this response (includes avg in header format)
                if current_tps > 0 and not interrupted:
                    tps_info = Text(f"{token_count} tokens in {elapsed_time:.1f}s | ", style="dim")
                    tps_info.append(f"{current_tps:.1f}", style="bold yellow")
                    tps_info.append(" tok/s | avg: ", style="dim")
                    tps_info.append(f"{self.avg_tps:.1f}", style="bold yellow")
                    tps_info.append(" tok/s", style="dim")
                    self.all_messages.append(tps_info)

                # Add to history
                self.history.append({"role": "assistant", "content": full_response})

                # Refresh display to show header at top with all messages
                self._refresh_display()

            # Re-raise if interrupted so outer handler can deal with it
            if interrupted:
                raise KeyboardInterrupt()

        except KeyboardInterrupt:
            # Let this propagate to the outer handler
            raise
        except Exception as e:
            error_msg = f"[error]Error: {str(e)}[/error]"
            self.console.print(Panel(error_msg, border_style="red"))
            self.console.print()

    async def _handle_command(self, command: str) -> str:
        """Handle special commands.

        Args:
            command: Command string starting with /

        Returns:
            Command result ("quit", "switch", or "continue")
        """
        cmd = command.lower().split()[0]

        if cmd == "/quit" or cmd == "/q":
            self.console.print("[info]Goodbye![/info]")
            return "quit"

        elif cmd == "/switch":
            self.console.print("[info]Switching models...[/info]")
            return "switch"

        elif cmd == "/clear":
            self.history = []
            self.tps_samples = []
            self.avg_tps = 0.0
            self.all_messages = []
            self._refresh_display()
            self.console.print("[success]Conversation cleared![/success]")
            self.console.print()

        elif cmd == "/export":
            filename = await self._export_conversation()
            if filename:
                self.console.print(f"[success]Conversation exported to {filename}[/success]")
            self.console.print()

        elif cmd == "/system":
            # Handle system prompt
            await self._handle_system_prompt()

        elif cmd == "/stress":
            return "stress_test"

        elif cmd == "/help":
            self.console.print(Panel(
                """[bold]Available Commands:[/bold]

/quit, /q     - Exit the chat
/switch       - Switch to a different model
/clear        - Clear conversation history
/export       - Export conversation to markdown
/system       - View/edit system prompt
/stress       - Run stress tests on the model server
/help         - Show this help message

[bold]Keyboard Shortcuts:[/bold]

Ctrl+D        - Back to main menu
Ctrl+C        - Quit application (with confirmation)
                """,
                title="Help",
                border_style="blue"
            ))
            self.console.print()

        else:
            self.console.print(f"[warning]Unknown command: {cmd}[/warning]")
            self.console.print("[dim]Type /help for available commands[/dim]")
            self.console.print()

        return "continue"

    def _create_header_panel(self) -> Panel:
        """Create the header panel with current stats."""
        server_addr = f"{self.server['ip']}:{self.server['port']}"

        # Create header text
        header_text = Text()
        header_text.append("┃ ", style="bold blue")
        header_text.append(self.model, style="bold cyan")
        header_text.append(" @ ", style="dim")
        header_text.append(server_addr, style="dim magenta")

        if self.avg_tps > 0:
            header_text.append(" │ ", style="dim")
            header_text.append(f"{self.avg_tps:.1f}", style="bold yellow")
            header_text.append(" tok/s", style="dim")

        header_text.append(" ┃", style="bold blue")

        # Return panel
        return Panel(header_text, style="on #1e293b", border_style="blue", padding=(0, 1))

    def _refresh_display(self):
        """Refresh the display with header at top and all messages below."""
        # Clear screen
        self.console.clear()

        # Print header at top (always visible)
        self.console.print(self._create_header_panel())
        self.console.print()

        # Print all accumulated messages
        if self.all_messages:
            for msg in self.all_messages:
                self.console.print(msg)
                self.console.print()
        else:
            # Show welcome message if no messages yet
            self.console.print(
                Panel(
                    "[dim italic]Start typing to chat. Commands: /quit, /switch, /export, /clear, /system, /stress[/dim italic]\n"
                    "[dim]Ctrl+D: Back to menu | Ctrl+C: Quit[/dim]",
                    border_style="dim"
                )
            )
            self.console.print()

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using improved heuristics.

        Better than simple char/4, accounts for:
        - Word boundaries (spaces create tokens)
        - Punctuation (often separate tokens)
        - Numbers (compact tokenization)
        - Code patterns (more tokens per char)

        Target: Within 10-15% of actual tokens

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Count different components
        words = text.split()
        word_count = len(words)

        # Count special characters (punctuation, symbols)
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())

        # Count newlines (often separate tokens)
        newline_count = text.count('\n')

        # Estimate based on multiple factors:
        # - Base: ~1.3 tokens per word (accounts for subword tokenization)
        # - Punctuation: ~0.5 tokens each (some merge with words)
        # - Newlines: 1 token each
        # - Adjustment for very short words (more tokens per char)

        token_estimate = (
            word_count * 1.3 +           # Words with subword splits
            special_chars * 0.5 +        # Punctuation/symbols
            newline_count * 1.0          # Newlines
        )

        # Clamp minimum to prevent zero/negative
        return max(1, int(token_estimate))

    def _build_message_history(self) -> list:
        """Build message history with system prompt if set.

        Returns:
            List of messages including system prompt
        """
        messages = []

        # Add system prompt if set
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add conversation history
        messages.extend(self.history)

        return messages

    async def _handle_system_prompt(self):
        """Handle system prompt view/edit."""
        if self.system_prompt:
            # Show current system prompt
            self.console.print(Panel(
                self.system_prompt,
                title="[bold cyan]Current System Prompt[/bold cyan]",
                border_style="cyan"
            ))
            self.console.print()

            # Ask if they want to edit or clear
            with patch_stdout():
                action = await self.session.prompt_async(
                    HTML('<ansigray>E to edit, C to clear, or Enter to keep: </ansigray>')
                )

            action = action.strip().lower()

            if action == 'e':
                # Edit prompt
                self.console.print("[dim]Enter new system prompt (Ctrl+D when done):[/dim]")
                lines = []
                try:
                    with patch_stdout():
                        while True:
                            line = await self.session.prompt_async("", multiline=False)
                            if not line:  # Empty line signals done
                                break
                            lines.append(line)
                except EOFError:
                    pass

                new_prompt = "\n".join(lines).strip()
                if new_prompt:
                    self.system_prompt = new_prompt
                    self.console.print("[success]System prompt updated![/success]")
                else:
                    self.console.print("[warning]System prompt unchanged[/warning]")

            elif action == 'c':
                # Clear prompt
                self.system_prompt = ""
                self.console.print("[success]System prompt cleared![/success]")

            self.console.print()

        else:
            # No system prompt set, ask if they want to create one
            self.console.print("[dim]No system prompt set. Create one?[/dim]")
            self.console.print("[dim]Enter system prompt (Ctrl+D or empty line when done):[/dim]")
            lines = []
            try:
                with patch_stdout():
                    while True:
                        line = await self.session.prompt_async("", multiline=False)
                        if not line:  # Empty line signals done
                            break
                        lines.append(line)
            except EOFError:
                pass

            new_prompt = "\n".join(lines).strip()
            if new_prompt:
                self.system_prompt = new_prompt
                self.console.print("[success]System prompt set![/success]")
                self.console.print(Panel(
                    self.system_prompt,
                    title="[bold cyan]System Prompt[/bold cyan]",
                    border_style="cyan"
                ))
            else:
                self.console.print("[dim]No system prompt set[/dim]")

            self.console.print()

    async def _export_conversation(self) -> str:
        """Export conversation to markdown file.

        Returns:
            Filename if successful, None otherwise
        """
        if not self.history:
            self.console.print("[warning]No conversation to export[/warning]")
            return None

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{self.model.replace('/', '_')}_{timestamp}.md"

        try:
            with open(filename, "w") as f:
                f.write(f"# Chat with {self.model}\n\n")
                f.write(f"Server: {self.server['ip']}:{self.server['port']}\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Include system prompt if set
                if self.system_prompt:
                    f.write(f"**System Prompt:** {self.system_prompt}\n\n")

                f.write("---\n\n")

                for msg in self.history:
                    role = msg["role"].upper()
                    content = msg["content"]
                    f.write(f"## {role}\n\n{content}\n\n")

            return filename
        except Exception as e:
            self.console.print(f"[error]Failed to export: {e}[/error]")
            return None
