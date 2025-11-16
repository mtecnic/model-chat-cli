"""Chat view for conversing with selected model."""
import asyncio
import os
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
from ui.effects import (
    create_sparkline,
    create_glass_panel,
    create_status_bar,
    create_confetti,
    create_particle_burst
)
from ui.theme import list_themes, save_theme, get_theme


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

        # Multiline mode
        self.multiline_mode = False

        # Message metadata for timestamps
        self.message_metadata = []  # List of dicts with metadata
        self.show_timestamps = False  # Toggle for showing timestamps/metadata

    async def run(self):
        """Run the chat interface."""
        # Initial display with header and welcome message
        self._refresh_display()

        # Chat loop
        while True:
            try:
                # Get user input
                prompt_text = '<ansiblue><b>┃ You ▶</b></ansiblue> '
                if self.multiline_mode:
                    prompt_text += '<ansigray>[multiline: Alt+Enter to send]</ansigray> '

                with patch_stdout():
                    user_input = await self.session.prompt_async(
                        HTML(prompt_text),
                        multiline=self.multiline_mode
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

                # Add user message to display with metadata
                import time
                user_metadata = {"timestamp": time.time(), "tokens": 0, "tps": 0, "duration": 0}
                self.message_metadata.append(user_metadata)
                user_msg = create_chat_message("user", user_input, metadata=user_metadata, show_metadata=self.show_timestamps)
                self.all_messages.append(user_msg)

                # Show particle burst effect when sending message
                particles = Text(create_particle_burst("✨", count=15), style="cyan dim")
                self.all_messages.append(particles)

                self._refresh_display()

                # Stream assistant response
                await self._stream_response(user_input)

                # Remove particle effect after response
                if particles in self.all_messages:
                    self.all_messages.remove(particles)

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

        # Estimate response tokens (assume similar to prompt length, with multiplier)
        prompt_tokens = self._estimate_tokens(message)
        estimated_response_tokens = int(prompt_tokens * 1.5)  # Responses typically 1.5x prompt length

        # Show typing indicator with estimate
        typing_panel = create_typing_indicator(estimated_response_tokens, self.avg_tps)

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
                # Create metadata for assistant message
                assistant_metadata = {
                    "timestamp": start_time,
                    "tokens": token_count,
                    "tps": current_tps,
                    "duration": elapsed_time
                }
                self.message_metadata.append(assistant_metadata)

                final_msg = create_chat_message("assistant", full_response, metadata=assistant_metadata, show_metadata=self.show_timestamps)
                self.all_messages.append(final_msg)

                # Show TPS stats for this response (includes avg in header format) - only if timestamps not shown
                if current_tps > 0 and not interrupted and not self.show_timestamps:
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
            self.message_metadata = []
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

        elif cmd == "/multiline":
            self.multiline_mode = not self.multiline_mode
            status = "enabled" if self.multiline_mode else "disabled"
            self.console.print(f"[success]Multiline mode {status}[/success]")
            if self.multiline_mode:
                self.console.print("[dim]Press Alt+Enter to send message, Enter for new line[/dim]")
            self.console.print()

        elif cmd == "/timestamps":
            self.show_timestamps = not self.show_timestamps
            status = "enabled" if self.show_timestamps else "disabled"
            self.console.print(f"[success]Timestamps {status}[/success]")
            # Rebuild all messages with new timestamp setting
            self._rebuild_messages()
            self._refresh_display()

        elif cmd.startswith("/copy"):
            # Copy a message to clipboard (just print it for now)
            parts = command.split()
            if len(parts) > 1 and parts[1].isdigit():
                msg_num = int(parts[1]) - 1
                if 0 <= msg_num < len(self.history):
                    content = self.history[msg_num]["content"]
                    # Try to copy to clipboard using pbcopy/xclip/clip
                    import subprocess
                    try:
                        # Try different clipboard commands
                        if os.system("which pbcopy > /dev/null 2>&1") == 0:
                            subprocess.run(["pbcopy"], input=content.encode(), check=True)
                        elif os.system("which xclip > /dev/null 2>&1") == 0:
                            subprocess.run(["xclip", "-selection", "clipboard"], input=content.encode(), check=True)
                        elif os.system("which clip > /dev/null 2>&1") == 0:
                            subprocess.run(["clip"], input=content.encode(), check=True)
                        else:
                            raise Exception("No clipboard command found")
                        self.console.print(f"[success]Message #{msg_num + 1} copied to clipboard![/success]")
                    except:
                        self.console.print(f"[warning]Could not copy to clipboard. Here's the content:[/warning]")
                        self.console.print(Panel(content, border_style="cyan"))
                else:
                    self.console.print(f"[error]Invalid message number. Valid range: 1-{len(self.history)}[/error]")
            else:
                self.console.print("[error]Usage: /copy <message_number>[/error]")
            self.console.print()

        elif cmd.startswith("/regenerate") or cmd.startswith("/regen"):
            # Regenerate the last assistant response
            if len(self.history) >= 2 and self.history[-1]["role"] == "assistant":
                # Remove last assistant message
                self.history.pop()
                self.all_messages.pop()  # Remove the message panel
                if self.all_messages and not isinstance(self.all_messages[-1], Panel):
                    self.all_messages.pop()  # Remove TPS stats if present
                if self.message_metadata:
                    self.message_metadata.pop()

                # Get the last user message
                last_user_msg = self.history[-1]["content"]
                self._refresh_display()

                # Regenerate response
                self.console.print("[info]Regenerating response...[/info]")
                await self._stream_response(last_user_msg)
            else:
                self.console.print("[warning]No assistant message to regenerate[/warning]")
                self.console.print()

        elif cmd =="/theme":
            # Theme selection
            themes = list_themes()
            self.console.print("\n[bold cyan]Available Themes:[/bold cyan]\n")

            for i, theme_name in enumerate(themes, 1):
                self.console.print(f"  [{i}] {theme_name}")

            self.console.print()
            choice = await self.session.prompt_async(HTML('<ansiblue>Select theme (number or name): </ansiblue>'))
            choice = choice.strip()

            if choice.isdigit() and 1 <= int(choice) <= len(themes):
                theme_name = themes[int(choice) - 1]
            elif choice in themes:
                theme_name = choice
            else:
                self.console.print("[error]Invalid theme selection[/error]")
                self.console.print()
                return "continue"

            # Save and apply theme
            save_theme(theme_name)
            new_theme = get_theme(theme_name)
            self.console = Console(theme=new_theme)
            self.console.print(f"[success]Theme changed to '{theme_name}'! Restart for full effect.[/success]")

            # Show celebration
            confetti = create_confetti()
            self.console.print(confetti)
            self.console.print()

        elif cmd == "/compare":
            # Compare responses from multiple models
            self.console.print("[bold cyan]Model Comparison Mode[/bold cyan]")
            self.console.print("[dim]Send the same prompt to multiple models and compare responses[/dim]\n")

            # Get comparison prompt
            with patch_stdout():
                prompt_text = await self.session.prompt_async(
                    HTML('<ansiblue><b>Enter prompt to compare:</b></ansiblue> '),
                    multiline=False
                )

            if not prompt_text.strip():
                self.console.print("[warning]Comparison cancelled[/warning]")
                self.console.print()
                return "continue"

            # For now, just compare with current model (can be extended to multi-model)
            self.console.print(f"\n[info]Comparing: {self.model}[/info]\n")

            # Get response from current model
            import time
            start = time.time()
            response = ""

            typing_panel = create_typing_indicator(0, 0)
            with Live(typing_panel, console=self.console, refresh_per_second=10, transient=True) as live:
                async for chunk in self.client.chat_stream(prompt_text, []):
                    response += chunk
                    live.update(create_chat_message("assistant", response))

            duration = time.time() - start
            tokens = self._estimate_tokens(response)
            tps = tokens / duration if duration > 0 else 0

            # Show comparison result
            from rich.columns import Columns

            col1 = Panel(
                f"[bold cyan]{self.model}[/bold cyan]\n\n{response}\n\n"
                f"[dim]⏱ {duration:.1f}s | 📊 {tokens} tokens | ⚡ {tps:.0f} tok/s[/dim]",
                title="Model Response",
                border_style="cyan",
                padding=(1, 2)
            )

            self.console.print(col1)
            self.console.print()
            self.console.print("[dim]Note: Multi-model comparison requires selecting multiple models at startup[/dim]")
            self.console.print("[dim]For now, showing single model response with detailed metrics[/dim]")
            self.console.print()

        elif cmd == "/history" or cmd.startswith("/hist"):
            # Show conversation history with message numbers
            if not self.history:
                self.console.print("[warning]No conversation history[/warning]")
                self.console.print()
            else:
                from rich.table import Table

                table = Table(title="Conversation History", show_header=True, header_style="bold cyan")
                table.add_column("#", style="dim", width=4, justify="right")
                table.add_column("Role", style="bold", width=12)
                table.add_column("Preview", style="dim")
                table.add_column("Tokens", justify="right", width=8)

                for i, msg in enumerate(self.history):
                    role = msg["role"].upper()
                    content_preview = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
                    # Get token count from metadata if available
                    tokens = ""
                    if i < len(self.message_metadata) and self.message_metadata[i].get("tokens", 0) > 0:
                        tokens = str(self.message_metadata[i]["tokens"])

                    role_style = "blue" if msg["role"] == "user" else "green"
                    table.add_row(
                        str(i + 1),
                        f"[{role_style}]{role}[/{role_style}]",
                        content_preview,
                        tokens
                    )

                self.console.print(table)
                self.console.print()
                self.console.print("[dim]Use /copy <num>, /edit <num> to interact with messages[/dim]")
                self.console.print()

        elif cmd.startswith("/edit"):
            # Edit and resend a message
            parts = command.split()
            if len(parts) > 1 and parts[1].isdigit():
                msg_num = int(parts[1]) - 1
                if 0 <= msg_num < len(self.history) and self.history[msg_num]["role"] == "user":
                    original = self.history[msg_num]["content"]
                    self.console.print(f"[dim]Original message:[/dim]\n{original}\n")
                    self.console.print("[dim]Enter edited message (Ctrl+D when done):[/dim]")

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

                    edited_msg = "\n".join(lines).strip()
                    if edited_msg:
                        # Truncate history at this point and resend
                        self.history = self.history[:msg_num]
                        self.message_metadata = self.message_metadata[:msg_num]
                        self._rebuild_messages()

                        # Add edited message
                        self.history.append({"role": "user", "content": edited_msg})
                        import time
                        user_metadata = {"timestamp": time.time(), "tokens": 0, "tps": 0, "duration": 0}
                        self.message_metadata.append(user_metadata)
                        user_msg = create_chat_message("user", edited_msg, metadata=user_metadata, show_metadata=self.show_timestamps)
                        self.all_messages.append(user_msg)
                        self._refresh_display()

                        # Get new response
                        await self._stream_response(edited_msg)
                    else:
                        self.console.print("[warning]Edit cancelled[/warning]")
                else:
                    self.console.print(f"[error]Invalid message number or not a user message[/error]")
            else:
                self.console.print("[error]Usage: /edit <message_number>[/error]")
            self.console.print()

        elif cmd == "/help":
            self.console.print(Panel(
                """[bold]Available Commands:[/bold]

/quit, /q       - Exit the chat
/switch         - Switch to a different model
/clear          - Clear conversation history
/export         - Export conversation to markdown
/system         - View/edit system prompt
/stress         - Run stress tests on the model server
/multiline      - Toggle multiline input mode
/timestamps     - Toggle timestamp/metadata display
/copy <num>     - Copy message to clipboard
/regenerate     - Regenerate last assistant response
/edit <num>     - Edit and resend a user message
/history        - View conversation history
/compare        - Compare model responses (experimental)
/theme          - Change color theme
/help           - Show this help message

[bold]Keyboard Shortcuts:[/bold]

Ctrl+D          - Back to main menu
Ctrl+C          - Quit application (with confirmation)
Alt+Enter       - Send message (in multiline mode)
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
        """Create the header panel with current stats and sparklines."""
        server_addr = f"{self.server['ip']}:{self.server['port']}"

        # Create status bar with icons
        health = "good" if self.avg_tps > 50 else "medium" if self.avg_tps > 20 else "poor"
        status_bar = create_status_bar(
            server=server_addr,
            tps=self.avg_tps,
            messages=len(self.history),
            health=health if self.avg_tps > 0 else "good"
        )

        # Create sparkline for TPS history
        sparkline = ""
        if len(self.tps_samples) > 1:
            sparkline = create_sparkline(self.tps_samples, width=20)
            sparkline_text = Text()
            sparkline_text.append("\n⚡ TPS: ", style="dim")
            sparkline_text.append(sparkline, style="yellow")
        else:
            sparkline_text = Text()

        # Combine model name and status
        header_content = Group(
            Text(f"✨ {self.model}", style="bold cyan"),
            status_bar,
            sparkline_text if sparkline else Text("")
        )

        # Return glassmorphism-style panel
        return create_glass_panel(header_content, accent_color="cyan")

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
                    "[dim italic]Start typing to chat. Type /help for all commands.[/dim italic]\n"
                    "[dim]Ctrl+D: Back to menu | Ctrl+C: Quit[/dim]",
                    border_style="dim"
                )
            )
            self.console.print()

        # Show command palette footer
        self._show_command_palette()

    def _show_command_palette(self):
        """Show command palette footer with available commands."""
        from rich.text import Text

        footer = Text()
        footer.append("[", style="dim")
        footer.append("/switch", style="cyan")
        footer.append("] ", style="dim")
        footer.append("Switch Model  ", style="dim")

        footer.append("[", style="dim")
        footer.append("/export", style="cyan")
        footer.append("] ", style="dim")
        footer.append("Export  ", style="dim")

        footer.append("[", style="dim")
        footer.append("/clear", style="cyan")
        footer.append("] ", style="dim")
        footer.append("Clear  ", style="dim")

        footer.append("[", style="dim")
        footer.append("/timestamps", style="cyan")
        footer.append("] ", style="dim")
        ts_status = "ON" if self.show_timestamps else "OFF"
        footer.append(f"Timestamps:{ts_status}  ", style="dim")

        footer.append("[", style="dim")
        footer.append("/multiline", style="cyan")
        footer.append("] ", style="dim")
        ml_status = "ON" if self.multiline_mode else "OFF"
        footer.append(f"Multiline:{ml_status}  ", style="dim")

        footer.append("[", style="dim")
        footer.append("/help", style="cyan")
        footer.append("] ", style="dim")
        footer.append("Help", style="dim")

        self.console.print(Panel(footer, style="on #0f172a", border_style="dim blue", padding=(0, 1)))

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

    def _rebuild_messages(self):
        """Rebuild all message displays with current timestamp setting."""
        new_messages = []
        metadata_idx = 0

        for i, msg_dict in enumerate(self.history):
            role = msg_dict["role"]
            content = msg_dict["content"]

            # Get corresponding metadata if available
            metadata = self.message_metadata[metadata_idx] if metadata_idx < len(self.message_metadata) else None

            # Create message with current timestamp setting
            msg_panel = create_chat_message(role, content, metadata=metadata, show_metadata=self.show_timestamps)
            new_messages.append(msg_panel)

            # Add TPS stats line for assistant messages if timestamps are off
            if role == "assistant" and not self.show_timestamps and metadata:
                if metadata.get("tps", 0) > 0:
                    from rich.text import Text
                    tps_info = Text(f"{metadata['tokens']} tokens in {metadata['duration']:.1f}s | ", style="dim")
                    tps_info.append(f"{metadata['tps']:.1f}", style="bold yellow")
                    tps_info.append(" tok/s | avg: ", style="dim")
                    tps_info.append(f"{self.avg_tps:.1f}", style="bold yellow")
                    tps_info.append(" tok/s", style="dim")
                    new_messages.append(tps_info)

            metadata_idx += 1

        self.all_messages = new_messages

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
