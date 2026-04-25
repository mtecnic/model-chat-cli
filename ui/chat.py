"""Chat view for conversing with selected model."""
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import HTML

from client import ModelClient
from storage.history import ChatHistoryManager
from ui.components import format_stats_line, estimate_tokens
from think_parser import parsed_chat_stream, ChunkType


class ChatView:
    """Handle chat interface with streaming responses."""

    def __init__(self, console: Console, server: dict, model: str):
        self.console = console
        self.server = server
        self.model = model
        self.client = ModelClient(server, model)
        self.history = []
        self.session = PromptSession()

        # Token per second tracking
        self.tps_samples = []
        self.avg_tps = 0.0

        # System prompt
        self.system_prompt = ""

        # Thinking mode toggle
        self.thinking_enabled = False

    async def run(self):
        """Run the chat interface."""
        self._print_header()
        self._print_welcome()

        while True:
            try:
                with patch_stdout():
                    user_input = await self.session.prompt_async(
                        HTML('<ansicyan><b>&gt; </b></ansicyan>'),
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
                        return "switch"
                    elif command_result == "stress_test":
                        return "stress_test"
                    elif command_result == "arena":
                        return "arena"
                    elif command_result == "prompt_arena":
                        return "prompt_arena"
                    continue

                # Add user message to history
                self.history.append({"role": "user", "content": user_input})

                # Print user message inline
                self._print_user_message(user_input)

                # Stream assistant response
                await self._stream_response(user_input)

            except KeyboardInterrupt:
                self.console.print("\n  [chrome]Q to quit, M for menu, Enter to continue[/chrome]")
                try:
                    confirm = await self.session.prompt_async(
                        HTML('<ansicyan><b>&gt; </b></ansicyan>')
                    )
                    confirm = confirm.strip().lower()
                    if confirm == 'q':
                        return "quit"
                    elif confirm == 'm':
                        return "switch"
                except (KeyboardInterrupt, EOFError):
                    return "quit"
                continue
            except EOFError:
                self.console.print("\n  [chrome]Returning to menu...[/chrome]")
                return "switch"

    async def _stream_response(self, message: str):
        """Stream the assistant's response with direct stdout writes."""
        import time
        import sys

        thinking_text = ""
        content_text = ""
        request_start = time.time()
        first_token_time = None
        end_time = None
        token_count = 0
        interrupted = False
        was_thinking = False

        messages_with_system = self._build_message_history()

        try:
            # Print assistant label
            label = Text("  Assistant ", style="role.assistant")
            label.append("\u25b8", style="chrome")
            self.console.print(label)

            # Stream chunks directly to stdout — Rich.Live freezes on long output
            # due to lock contention between its refresh thread and the async loop
            try:
                stream = parsed_chat_stream(
                    self.client.chat_stream(
                        None, messages_with_system,
                        enable_thinking=self.thinking_enabled,
                    ),
                    start_thinking=self.thinking_enabled,
                )
                async for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.time()
                    if chunk.chunk_type == ChunkType.THINKING:
                        if not was_thinking:
                            # Entering thinking — italic only (content stays normal)
                            sys.stdout.write("\033[3m")
                            was_thinking = True
                        thinking_text += chunk.text
                        sys.stdout.write(chunk.text)
                        sys.stdout.flush()
                    else:
                        if was_thinking:
                            # Leaving thinking — reset style, add separator
                            sys.stdout.write("\033[0m\n\n")
                            was_thinking = False
                        content_text += chunk.text
                        sys.stdout.write(chunk.text)
                        sys.stdout.flush()
                end_time = time.time()
            except (asyncio.CancelledError, KeyboardInterrupt):
                end_time = time.time()
                interrupted = True
                content_text += "\n\n(interrupted)"

            # Reset any lingering ANSI style
            if was_thinking:
                sys.stdout.write("\033[0m")
            sys.stdout.write('\n')
            sys.stdout.flush()

            # If thinking was enabled but model didn't actually think
            # (no </think> seen, all text classified as thinking), treat as content
            if thinking_text and not content_text:
                content_text = thinking_text
                thinking_text = ""

            # Estimate tokens once after streaming (content only for stats)
            token_count = estimate_tokens(content_text)
            think_token_count = estimate_tokens(thinking_text) if thinking_text else 0

            # Decode tok/s: generated tokens / time from first token → last token.
            # Excludes TTFT (queue + prompt eval + network RTT) so the number
            # reflects actual model decode speed, matching stress_tester's decode_tps.
            if end_time is None:
                end_time = time.time()
            ttft = first_token_time - request_start if first_token_time else 0.0
            decode_elapsed = (end_time - first_token_time) if first_token_time else 0.0
            total_generated = token_count + think_token_count
            current_tps = 0
            if decode_elapsed > 0 and total_generated > 0:
                current_tps = total_generated / decode_elapsed
                self.tps_samples.append(current_tps)
                if len(self.tps_samples) > 10:
                    self.tps_samples.pop(0)
                self.avg_tps = sum(self.tps_samples) / len(self.tps_samples)

            # Print stats line
            if content_text or thinking_text:
                if current_tps > 0 and not interrupted:
                    stats = format_stats_line(
                        token_count,
                        decode_elapsed,
                        current_tps,
                        ttft=ttft,
                        think_tokens=think_token_count if self.thinking_enabled else None,
                    )
                    self.console.print(stats)
                self.console.print()
                entry = {"role": "assistant", "content": content_text}
                if thinking_text:
                    entry["thinking"] = thinking_text
                self.history.append(entry)

            if interrupted:
                raise KeyboardInterrupt()

        except KeyboardInterrupt:
            raise
        except Exception as e:
            self.console.print(f"  [status.error]Error: {str(e)}[/status.error]")
            self.console.print()

    async def _handle_command(self, command: str) -> str:
        """Handle special commands."""
        cmd = command.lower().split()[0]

        if cmd in ("/quit", "/q"):
            self.console.print("  [chrome]Goodbye![/chrome]")
            return "quit"

        elif cmd == "/switch":
            self.console.print("  [chrome]Switching models...[/chrome]")
            return "switch"

        elif cmd == "/clear":
            self.history = []
            self.tps_samples = []
            self.avg_tps = 0.0
            self._refresh_display()
            self.console.print("  [status.ok]Conversation cleared.[/status.ok]")
            self.console.print()

        elif cmd == "/export":
            filename = await self._export_conversation()
            if filename:
                self.console.print(f"  [status.ok]Exported to {filename}[/status.ok]")
            self.console.print()

        elif cmd == "/system":
            await self._handle_system_prompt()

        elif cmd == "/stress":
            return "stress_test"

        elif cmd == "/arena":
            return "arena"

        elif cmd == "/promptarena":
            return "prompt_arena"

        elif cmd == "/think":
            self.thinking_enabled = not self.thinking_enabled
            state = "ON" if self.thinking_enabled else "OFF"
            style = "status.ok" if self.thinking_enabled else "chrome"
            self.console.print(f"  [{style}]Thinking: {state}[/{style}]")
            self.console.print()

        elif cmd == "/help":
            self.console.print()
            self.console.print("  [accent.bold]Commands[/accent.bold]")
            self.console.print("  [accent]/quit[/accent], [accent]/q[/accent]     Exit")
            self.console.print("  [accent]/switch[/accent]       Switch model")
            self.console.print("  [accent]/clear[/accent]        Clear history")
            self.console.print("  [accent]/export[/accent]       Export to markdown")
            self.console.print("  [accent]/system[/accent]       System prompt")
            self.console.print("  [accent]/think[/accent]        Toggle reasoning mode")
            self.console.print("  [accent]/stress[/accent]       Stress test")
            self.console.print("  [accent]/arena[/accent]        Multi-model arena (side by side)")
            self.console.print("  [accent]/promptarena[/accent]  Prompt comparison tournament")
            self.console.print()
            self.console.print("  [chrome]Ctrl+D[/chrome] back  [chrome]Ctrl+C[/chrome] quit")
            self.console.print()

        else:
            self.console.print(f"  [status.warn]Unknown command: {cmd}[/status.warn]")
            self.console.print("  [chrome]Type /help for commands[/chrome]")
            self.console.print()

        return "continue"

    def _print_header(self):
        """Print the chat header."""
        server_addr = f"{self.server['ip']}:{self.server['port']}"
        header = Text()
        header.append(self.model, style="model.name")
        header.append("  ", style="chrome")
        header.append(server_addr, style="model.server")
        if self.thinking_enabled:
            header.append("  think", style="status.ok")
        self.console.print()
        self.console.rule(header, style="chrome.border")
        self.console.print()

    def _print_welcome(self):
        """Print the welcome hint."""
        self.console.print("  [chrome.muted]Type a message to begin. /help for commands.[/chrome.muted]")
        self.console.print()

    def _print_user_message(self, text: str):
        """Print a user message inline."""
        self.console.print()
        label = Text("  You ", style="role.user")
        label.append("\u25b8 ", style="chrome")
        self.console.print(label)
        for line in text.split('\n'):
            self.console.print(f"  {line}")
        self.console.print()

    def _refresh_display(self):
        """Clear screen and reprint header. Used only for /clear."""
        self.console.clear()
        self._print_header()
        self._print_welcome()

    def _build_message_history(self) -> list:
        """Build message history with system prompt if set."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.history)
        return messages

    async def _handle_system_prompt(self):
        """Handle system prompt view/edit."""
        if self.system_prompt:
            self.console.print()
            self.console.print("  [accent.bold]System Prompt[/accent.bold]")
            self.console.print(f"  {self.system_prompt}")
            self.console.print()

            with patch_stdout():
                action = await self.session.prompt_async(
                    HTML('<ansicyan>  E to edit, C to clear, Enter to keep: </ansicyan>')
                )
            action = action.strip().lower()

            if action == 'e':
                self.console.print("  [chrome]Enter new system prompt (empty line when done):[/chrome]")
                lines = []
                try:
                    with patch_stdout():
                        while True:
                            line = await self.session.prompt_async("  ")
                            if not line:
                                break
                            lines.append(line)
                except EOFError:
                    pass
                new_prompt = "\n".join(lines).strip()
                if new_prompt:
                    self.system_prompt = new_prompt
                    self.console.print("  [status.ok]System prompt updated.[/status.ok]")
                else:
                    self.console.print("  [chrome]System prompt unchanged.[/chrome]")

            elif action == 'c':
                self.system_prompt = ""
                self.console.print("  [status.ok]System prompt cleared.[/status.ok]")

            self.console.print()

        else:
            self.console.print("  [chrome]No system prompt set. Enter one (empty line when done):[/chrome]")
            lines = []
            try:
                with patch_stdout():
                    while True:
                        line = await self.session.prompt_async("  ")
                        if not line:
                            break
                        lines.append(line)
            except EOFError:
                pass
            new_prompt = "\n".join(lines).strip()
            if new_prompt:
                self.system_prompt = new_prompt
                self.console.print("  [status.ok]System prompt set.[/status.ok]")
                self.console.print(f"  [chrome]{self.system_prompt}[/chrome]")
            else:
                self.console.print("  [chrome]No system prompt set.[/chrome]")
            self.console.print()

    async def _export_conversation(self) -> str:
        """Export conversation to markdown file."""
        if not self.history:
            self.console.print("  [status.warn]No conversation to export.[/status.warn]")
            return None

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{self.model.replace('/', '_')}_{timestamp}.md"

        try:
            with open(filename, "w") as f:
                f.write(f"# Chat with {self.model}\n\n")
                f.write(f"Server: {self.server['ip']}:{self.server['port']}\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                if self.system_prompt:
                    f.write(f"**System Prompt:** {self.system_prompt}\n\n")
                f.write("---\n\n")
                for msg in self.history:
                    role = msg["role"].upper()
                    content = msg["content"]
                    if role == "ASSISTANT" and msg.get("thinking"):
                        f.write(f"## {role}\n\n")
                        f.write(f"<details>\n<summary>Thinking</summary>\n\n{msg['thinking']}\n\n</details>\n\n")
                        f.write(f"{content}\n\n")
                    else:
                        f.write(f"## {role}\n\n{content}\n\n")

            try:
                ChatHistoryManager().save_conversation(
                    model=self.model,
                    server=f"{self.server['ip']}:{self.server['port']}",
                    messages=self.history,
                )
            except Exception as e:
                self.console.print(f"  [status.warn]History persist failed: {e}[/status.warn]")

            return filename
        except Exception as e:
            self.console.print(f"  [status.error]Failed to export: {e}[/status.error]")
            return None
