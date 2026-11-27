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
from ui.interrupt import run_interruptible


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
                    elif command_result == "bench":
                        return "bench"
                    continue

                # Add user message to history
                self.history.append({"role": "user", "content": user_input})

                # Print user message inline
                self._print_user_message(user_input)

                # Stream assistant response
                await self._stream_response(user_input)

            except KeyboardInterrupt:
                # Ctrl+C at the input prompt (generation handles its own Ctrl+C)
                self.console.print("\n  [chrome]Q to quit, M for model picker, Enter to continue[/chrome]")
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

    # With /think on we assume the model starts inside a thinking block (the
    # template puts <think> in the prompt). Until a </think> or explicit
    # reasoning confirms that, hold back up to this many chars instead of
    # printing them in italics, so a model that doesn't think isn't shown
    # as "thinking" for its whole answer.
    THINK_HOLDBACK_CHARS = 400

    async def _stream_response(self, message: str):
        """Stream the assistant's response with direct stdout writes.

        The user message is already the last entry in self.history. On error
        or an empty reply it is removed again, so history never contains two
        user turns in a row. Ctrl+C cancels generation and keeps the partial
        reply.
        """
        import time
        import sys

        state = {
            "thinking": "", "content": "",
            "first_token": None, "end": None,
            "italic": False,       # currently writing in italics
            "decided": not self.thinking_enabled,  # thinking-vs-content resolved
            "pending": "",         # held-back text while undecided
        }
        request_start = time.time()
        messages_with_system = self._build_message_history()

        def write(text: str):
            sys.stdout.write(text)
            sys.stdout.flush()

        def set_italic(on: bool):
            if on and not state["italic"]:
                write("\033[3m")
            elif not on and state["italic"]:
                write("\033[0m")
            state["italic"] = on

        def flush_pending(as_thinking: bool):
            if state["pending"]:
                set_italic(as_thinking)
                write(state["pending"])
                state["pending"] = ""
            state["decided"] = True

        async def consume():
            stream = parsed_chat_stream(
                self.client.chat_stream(
                    None, messages_with_system,
                    enable_thinking=self.thinking_enabled,
                ),
                start_thinking=self.thinking_enabled,
            )
            async for chunk in stream:
                if state["first_token"] is None:
                    state["first_token"] = time.time()
                if chunk.chunk_type == ChunkType.THINKING:
                    state["thinking"] += chunk.text
                    if not state["decided"]:
                        state["pending"] += chunk.text
                        if len(state["pending"]) >= self.THINK_HOLDBACK_CHARS:
                            flush_pending(as_thinking=True)
                        continue
                    set_italic(True)
                    write(chunk.text)
                else:
                    if not state["decided"]:
                        # Content after thinking means </think> was seen:
                        # the held-back text really was reasoning
                        flush_pending(as_thinking=True)
                    if state["italic"]:
                        set_italic(False)
                        write("\n\n")
                    state["content"] += chunk.text
                    write(chunk.text)
            state["end"] = time.time()

        # Print assistant label
        label = Text("  Assistant ", style="role.assistant")
        label.append("▸", style="chrome")
        self.console.print(label)

        error = None
        interrupted = False
        try:
            # Stream chunks directly to stdout — Rich.Live freezes on long output
            # due to lock contention between its refresh thread and the async loop
            _, interrupted = await run_interruptible(consume())
        except Exception as e:
            error = e

        if state["end"] is None:
            state["end"] = time.time()

        thinking_text, content_text = state["thinking"], state["content"]
        # /think on but the model never closed a thinking block: it wasn't
        # thinking, it was answering. Show held-back text as normal content.
        if thinking_text and not content_text:
            if not state["decided"]:
                flush_pending(as_thinking=False)
            content_text, thinking_text = thinking_text, ""

        set_italic(False)
        write("\n")

        if interrupted:
            self.console.print("  [chrome](interrupted)[/chrome]")

        if error is not None:
            # Text, not markup: server error bodies often contain [brackets]
            self.console.print(Text(f"  Error: {error}", style="status.error"))
            self.console.print()

        if not content_text.strip():
            # Nothing usable came back: drop the pending user turn
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            if error is None and not interrupted:
                self.console.print("  [status.warn]Empty response from model.[/status.warn]")
                self.console.print()
            return

        # Estimate tokens once after streaming (content only for stats)
        token_count = estimate_tokens(content_text)
        think_token_count = estimate_tokens(thinking_text) if thinking_text else 0

        # Decode tok/s: generated tokens / time from first token → last token.
        # Excludes TTFT (queue + prompt eval + network RTT) so the number
        # reflects actual model decode speed, matching stress_tester's decode_tps.
        first_token_time = state["first_token"]
        ttft = first_token_time - request_start if first_token_time else 0.0
        decode_elapsed = (state["end"] - first_token_time) if first_token_time else 0.0
        total_generated = token_count + think_token_count
        current_tps = 0
        if decode_elapsed > 0 and total_generated > 0:
            current_tps = total_generated / decode_elapsed
            self.tps_samples.append(current_tps)
            if len(self.tps_samples) > 10:
                self.tps_samples.pop(0)
            self.avg_tps = sum(self.tps_samples) / len(self.tps_samples)

        if current_tps > 0 and not interrupted and error is None:
            stats = format_stats_line(
                token_count,
                decode_elapsed,
                current_tps,
                ttft=ttft,
                think_tokens=think_token_count if self.thinking_enabled else None,
            )
            self.console.print(stats)
        self.console.print()

        # Keep partial replies (interrupt / mid-stream error) so the
        # conversation stays coherent; the "(interrupted)" marker is display-only.
        entry = {"role": "assistant", "content": content_text}
        if thinking_text:
            entry["thinking"] = thinking_text
        self.history.append(entry)

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

        elif cmd == "/bench":
            return "bench"

        elif cmd == "/history":
            await self._handle_history()

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
            self.console.print("  [accent]/export[/accent]       Export to markdown (+ save to history)")
            self.console.print("  [accent]/history[/accent]      Load a saved conversation")
            self.console.print("  [accent]/system[/accent]       System prompt")
            self.console.print("  [accent]/think[/accent]        Toggle reasoning mode")
            self.console.print("  [accent]/stress[/accent]       Stress test")
            self.console.print("  [accent]/bench[/accent]        Accuracy benchmarks (GSM8K, MMLU-Pro, ARC, HumanEval)")
            self.console.print("  [accent]/arena[/accent]        Multi-model arena (side by side)")
            self.console.print("  [accent]/promptarena[/accent]  Prompt comparison tournament")
            self.console.print()
            self.console.print("  [chrome]Ctrl+C[/chrome] stop generation / menu  [chrome]Ctrl+D[/chrome] model picker")
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
        # Only role/content go to the server; extra keys (e.g. "thinking",
        # kept for export) can be rejected by strict OpenAI-compatible servers
        messages.extend({"role": m["role"], "content": m["content"]} for m in self.history)
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
                    system_prompt=self.system_prompt,
                )
            except Exception as e:
                self.console.print(f"  [status.warn]History persist failed: {e}[/status.warn]")

            return filename
        except Exception as e:
            self.console.print(f"  [status.error]Failed to export: {e}[/status.error]")
            return None

    async def _handle_history(self):
        """List saved conversations and load one into this chat."""
        manager = ChatHistoryManager()
        entries = manager.list_summaries(limit=15)
        if not entries:
            self.console.print("  [status.warn]No saved conversations yet.[/status.warn]")
            self.console.print("  [chrome]/export saves the current chat to history.[/chrome]")
            self.console.print()
            return

        self.console.print()
        self.console.print("  [accent.bold]Saved Conversations[/accent.bold]")
        for i, e in enumerate(entries, 1):
            preview = e["preview"][:50] + ("..." if len(e["preview"]) > 50 else "")
            line = Text(f"  {i:>2}. ", style="accent")
            line.append(e["date"], style="chrome")
            line.append("  ")
            line.append(e["model"], style="model.name")
            line.append(f"  {preview}")
            self.console.print(line)
        self.console.print()

        with patch_stdout():
            choice = await self.session.prompt_async(
                HTML('<ansicyan>  Number to load, Enter to cancel: </ansicyan>')
            )
        choice = choice.strip()
        if not choice:
            self.console.print()
            return
        if not choice.isdigit() or not 1 <= int(choice) <= len(entries):
            self.console.print("  [status.warn]Invalid choice.[/status.warn]")
            self.console.print()
            return

        entry = entries[int(choice) - 1]
        try:
            data = manager.load_conversation(entry["path"])
        except Exception as e:
            self.console.print(f"  [status.error]Failed to load: {e}[/status.error]")
            self.console.print()
            return

        self.history = [m for m in data.get("messages", [])
                        if m.get("role") in ("user", "assistant") and "content" in m]
        self.system_prompt = data.get("system_prompt") or ""
        self.tps_samples = []
        self.avg_tps = 0.0

        self._refresh_display()
        if data.get("model") and data["model"] != self.model:
            self.console.print(
                f"  [chrome]Saved with {data['model']}; continuing with {self.model}.[/chrome]")
        for m in self.history:
            if m["role"] == "user":
                self._print_user_message(m["content"])
            else:
                label = Text("  Assistant ", style="role.assistant")
                label.append("\u25b8", style="chrome")
                self.console.print(label)
                self.console.print(m["content"], markup=False, highlight=False)
                self.console.print()
        self.console.print(f"  [status.ok]Loaded {len(self.history)} messages.[/status.ok]")
        self.console.print()
