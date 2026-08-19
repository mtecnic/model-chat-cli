"""Chat screen - streaming conversation with slash commands."""
from __future__ import annotations

import time
from typing import List, Optional

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Input, Markdown, Static, TextArea

from client import ModelClient
from think_parser import ChunkType, parsed_chat_stream
from ui.base import BaseScreen, MUTED

USER_LABEL = "#4cc9f0"

COMMANDS = {
    "/help": "show chat commands",
    "/quit": "back to menu",
    "/q": "back to menu",
    "/switch": "pick another model",
    "/clear": "clear this conversation",
    "/system": "set the system prompt",
    "/think": "toggle thinking mode",
    "/export": "save this conversation",
    "/publish": "save + publish this conversation",
    "/stress": "open the stress lab",
    "/arena": "open the model arena",
    "/promptarena": "open the prompt arena",
}


class AssistantMessage(Widget):
    """One assistant reply: dim thinking block + markdown body + stats line.

    While streaming, the body is a plain Static (fast, no markdown re-parse).
    On completion it is swapped for a Markdown widget so code blocks get
    syntax highlighting (the original Rich behavior, without the lock issue).
    """

    DEFAULT_CSS = """
    AssistantMessage {
        width: 100%;
        height: auto;
        padding: 0 1;
    }
    AssistantMessage #am-think {
        width: 100%;
        height: auto;
        display: none;
    }
    AssistantMessage #am-body {
        width: 100%;
        height: auto;
    }
    AssistantMessage #am-stats {
        width: 100%;
        height: auto;
        display: none;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.thinking = ""
        self.body = ""
        self.error: Optional[str] = None
        self._t0: Optional[float] = None
        self._ttft: Optional[float] = None
        self._first_seen = False
        self._flush_at = 0.0
        self._content_tokens = 0
        self._body_widget: Optional[Widget] = None
        self._finished = False

    def compose(self) -> ComposeResult:
        yield Static("", id="am-think")
        yield Static("…", id="am-body", classes="streaming")
        yield Static("", id="am-stats")

    async def on_mount(self) -> None:
        self._body_widget = self.query_one("#am-body")
        if self.thinking:
            t = self.query_one("#am-think")
            t.update(Text(self.thinking, style="italic dim"))
            t.styles.display = "block"
        if self._finished:
            await self._finalize_body()

    # -- streaming ------------------------------------------------------- #
    def add_chunk(self, text: str, is_thinking: bool) -> None:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        if not is_thinking and not self._first_seen and text:
            self._first_seen = True
            self._ttft = now - (self._t0 if self._t0 else now)
        if is_thinking:
            self.thinking += text
            t = self.query_one("#am-think")
            t.update(Text(self.thinking, style="italic dim"))
            t.styles.display = "block"
        else:
            self.body += text
            self._content_tokens += 1
        if now - self._flush_at >= 0.1:
            self._flush_at = now
            self._flush()

    def _flush(self) -> None:
        if self._body_widget is not None and not self._finished:
            self._body_widget.update(Text(self.body or "…",
                                          style="dim" if not self.body else "default"))

    async def _finalize_body(self) -> None:
        """Swap the streaming Static for a Markdown widget."""
        if self._finished or self._body_widget is None:
            return
        self._finished = True
        content = (self.body or "").strip()
        md = Markdown(content if content else "*(empty)*")
        stats = self.query_one("#am-stats")
        await self._body_widget.remove()
        await self.mount(md, before=stats)
        self._body_widget = md

    async def set_metrics(self, tokens: int, tps: float, ttft: float, total: float) -> None:
        if tokens:
            self._content_tokens = tokens
        self._finalize_body()
        bits = []
        if tokens:
            bits.append(f"{tokens} tok")
        if total:
            bits.append(f"{total:.1f}s")
        if tps:
            bits.append(f"{tps:.0f} t/s")
        if ttft:
            bits.append(f"TTFT {ttft:.2f}s")
        if self.error:
            bits.append(f"[{self.error}]")
        stats = self.query_one("#am-stats")
        stats.update(Text("  ".join(bits), style=MUTED))
        stats.styles.display = "block"

    async def fail(self, msg: str) -> None:
        self.error = msg
        await self._finalize_body()
        stats = self.query_one("#am-stats")
        stats.update(Text(msg, style="bright_red"))
        stats.styles.display = "block"

    async def mark_stopped(self) -> None:
        await self._finalize_body()
        elapsed = (time.monotonic() - self._t0) if self._t0 else 0.0
        stats = self.query_one("#am-stats")
        stats.update(Text(f"[stopped]   {elapsed:.1f}s", style=MUTED))
        stats.styles.display = "block"

    @classmethod
    def from_record(cls, content: str, thinking: str = "") -> "AssistantMessage":
        """Build a static message from a stored conversation record."""
        am = cls()
        am.body = content
        am.thinking = thinking
        am._finished = True
        return am


class SystemPromptModal(ModalScreen):
    DEFAULT_CSS = """
    SystemPromptModal {
        align: center middle;
    }
    SystemPromptModal > Vertical {
        width: 70;
        height: 24;
        border: round $primary;
        padding: 1 2;
    }
    SystemPromptModal TextArea {
        height: 1fr;
    }
    SystemPromptModal > Vertical > Horizontal {
        height: 3;
        content-align: right middle;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, current: str) -> None:
        super().__init__()
        self.current = current
        self.result: Optional[str] = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("System prompt (applies to new turns)", style="bold")
            yield TextArea(self.current or "", id="sys-area")
            with Horizontal():
                yield Button("Save", id="sys-save", variant="primary")
                yield Button("Cancel", id="sys-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "sys-save":
            self.result = self.query_one("#sys-area", TextArea).text
            self.dismiss()
        else:
            self.dismiss()

    def action_cancel(self) -> None:
        self.dismiss()


class ChatScreen(BaseScreen):
    DEFAULT_CSS = """
    ChatScreen #chat-messages {
        height: 1fr;
        padding: 1 1;
    }
    ChatScreen #chat-input {
        height: 3;
        width: 100%;
        margin: 0;
    }
    ChatScreen #chat-cmds {
        height: 1;
        width: 60%;
        margin: 0;
        color: $text-muted;
    }
    ChatScreen #chat-hints {
        height: 2;
        content-align: center middle;
        width: 100%;
        color: $text-muted;
    }
    .msg-user {
        width: 100%;
        padding: 0 1;
        height: auto;
    }
    """

    BINDINGS = [
        Binding("escape", "interrupt", "Stop", show=False),
        Binding("ctrl+t", "toggle_think", "Think", show=False),
        Binding("ctrl+l", "back", "Menu", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.messages: List[dict] = []
        self.system_prompt = ""
        self._worker: Optional[work.Worker] = None

    def build_content(self) -> ComposeResult:
        yield VerticalScroll(id="chat-messages")
        yield Static("", id="chat-hints")
        yield Static("", id="chat-cmds")
        yield Input(placeholder="Ask the model…  (/help for commands)", id="chat-input")

    def on_mount(self) -> None:
        app = self.app
        box = self.query_one("#chat-messages", VerticalScroll)
        if self.messages:
            for m in self.messages:
                self._add_user_widget(m["role"], m.get("content", ""))
        else:
            app = self.app
            t = Text(justify="center")
            t.append("Chatting with ", style=MUTED)
            t.append(f"{app.model}", style="bright_cyan bold")
            if app.server:
                t.append(f" @ {app.server['ip']}:{app.server['port']}", style=MUTED)
            t.append("\n\n", )
            t.append("/help for commands    ·    esc while streaming = stop", style=MUTED)
            box.mount(Static(t))
        self.system_prompt = self._load_system_prompt()
        self.query_one("#chat-hints", Static).update(
            f"think: {'on' if app.thinking else 'off'} (ctrl+t)   ·   ctrl+l: menu   ·   esc: stop stream"
        )
        self.query_one("#chat-input", Input).focused = True

    def _load_system_prompt(self) -> str:
        return self.app.config.get("chat.system_prompt", "")

    # ------------------------------------------------------------------ #
    # input
    # ------------------------------------------------------------------ #

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if text.startswith("/"):
            self._handle_command(text)
        else:
            self.query_one("#chat-input", Input).value = ""
            self._send(text)
        if self._worker and self._worker.is_running:
            return
        self.query_one("#chat-input", Input).focused = True

    def on_input_changed(self, event: Input.Changed) -> None:
        v = event.value
        if v.startswith("/") and len(v) >= 1:
            matches = [c for c in sorted(COMMANDS) if c.startswith(v)][:6]
            self.query_one("#chat-cmds", Static).update(
                "  ".join(f"{c} {COMMANDS[c]}" for c in matches) if matches else ""
            )
        else:
            self.query_one("#chat-cmds", Static).update("")

    def _handle_command(self, text: str) -> None:
        cmd = text.split()[0].lower()
        app = self.app
        if cmd in ("/quit", "/q"):
            app.pop_screen()
        elif cmd == "/switch":
            app.pop_screen()
            app.push_screen("discovery")
        elif cmd == "/clear":
            self.messages = []
            box = self.query_one("#chat-messages", VerticalScroll)
            box.remove_children()
            self.notify("Conversation cleared", severity="info", timeout=2)
        elif cmd == "/system":
            app.push_screen(
                SystemPromptModal(self.system_prompt),
                callback=lambda result: self._system_saved(result),
            )
        elif cmd == "/think":
            app.thinking = not app.thinking
            self.query_one("#chat-hints", Static).update(
                f"think: {'on' if app.thinking else 'off'} (ctrl+t)   ·    ctrl+l: menu   ·   esc: stop stream"
            )
            self.notify(f"Thinking mode: {'on' if app.thinking else 'off'}",
                        severity="info", timeout=2)
        elif cmd == "/export":
            self._do_export()
        elif cmd == "/publish":
            self._do_export(publish=True)
        elif cmd in ("/stress",):
            app.pop_screen()
            app.push_screen("stress")
        elif cmd in ("/arena",):
            app.pop_screen()
            app.push_screen("model_arena")
        elif cmd in ("/promptarena",):
            app.pop_screen()
            app.push_screen("prompt_arena")
        elif cmd == "/help":
            t = Text()
            for c in sorted(COMMANDS):
                t.append(f"  {c:<14}", style="bold")
                t.append(COMMANDS[c] + "\n", style=MUTED)
            t.append("\n  ctrl+t toggle thinking · esc stop stream · ctrl+l menu", style=MUTED)
            self.app.notify(t, title="Chat commands", severity="information", timeout=15, markup=False)
        else:
            self.notify(f"unknown command {cmd} — try /help", severity="warning", timeout=3)

    def _system_saved(self, result) -> None:
        if result is None:
            return
        self.system_prompt = result
        self.app.config.set("chat.system_prompt", result)
        self.app.config.save()
        self.notify("System prompt saved", title="System", severity="success", timeout=3)

    # ------------------------------------------------------------------ #
    # streaming
    # ------------------------------------------------------------------ #

    def _add_user_widget(self, role: str, content: str, thinking: str = "") -> None:
        box = self.query_one("#chat-messages", VerticalScroll)
        if role == "user":
            t = Text()
            t.append("you →  ", style=f"bold {USER_LABEL}")
            t.append(content)
            box.mount(Static(t, classes="msg-user"))
        elif role == "assistant":
            box.mount(AssistantMessage.from_record(content, thinking))
        box.scroll_end(animate=False, force=True)

    def _send(self, text: str) -> None:
        app = self.app
        if not app.model or not app.server:
            self.notify("No model selected", severity="warning", timeout=3)
            return
        self.messages.append({"role": "user", "content": text})
        box = self.query_one("#chat-messages", VerticalScroll)
        t = Text()
        t.append("you →  ", style=f"bold {USER_LABEL}")
        t.append(text)
        box.mount(Static(t, classes="msg-user"))
        am = AssistantMessage()
        box.mount(am)
        box.scroll_end(animate=False, force=True)

        if self.system_prompt:
            self.messages.insert(0, {"role": "system", "content": self.system_prompt})

        self._current_am = am
        self._worker = self.run_worker(
            self._stream_worker(text, am),
            name="chat",
            group="chat",
            exclusive=True,
        )

    async def _stream_worker(self, text: str, am: AssistantMessage) -> None:
        app = self.app
        # history = everything before the current user turn (system prompt + prior turns)
        history = self.messages[:-1]
        client = ModelClient(app.server, app.model)
        full = ""
        t_start = time.monotonic()
        try:
            stream = client.chat_stream(text, history, enable_thinking=app.thinking)
            async for chunk in parsed_chat_stream(stream, start_thinking=app.thinking):
                is_think = chunk.chunk_type == ChunkType.THINKING
                am.add_chunk(chunk.text, is_think)
                full += chunk.text if not is_think else ""
        except Exception as e:
            await am.fail(str(e))
            return
        t_end = time.monotonic()
        total = t_end - t_start
        m = client.last_metrics
        tokens = m.completion_tokens or am._content_tokens
        tps = tokens / total if total > 0 and tokens else 0.0
        ttft = m.ttft or am._ttft or 0.0
        await am.set_metrics(tokens, tps, ttft, total)

        content = am.body.strip()
        thinking = am.thinking.strip()
        rec = {"role": "assistant", "content": content or full.strip()}
        if thinking:
            rec["thinking"] = thinking
        self.messages.append(rec)
        box = self.query_one("#chat-messages", VerticalScroll)
        box.scroll_end(animate=False, force=True)

    def _do_export(self, publish: bool = False) -> None:
        if len(self.messages) < 1:
            self.notify("Nothing to export yet", severity="info", timeout=2)
            return
        msgs = [m for m in self.messages if m["role"] != "system"]
        path = self.app.save_chat(msgs, self.system_prompt)
        if path:
            self.notify(f"Saved: {path}", title="Exported", severity="success", timeout=5)
            if publish:
                self.run_worker(self._publish_file(path), name="publish-chat")

    async def _publish_file(self, path) -> None:
        from publish import Publisher
        cfg = self.app.config
        pub = Publisher(
            cfg.get("github.repo_path", ""),
            branch=cfg.get("github.branch", "main"),
            prefix=cfg.get("github.prefix", "results"),
            push=cfg.get("github.push", True),
            dry_run=cfg.get("github.dry_run", False),
        )
        ok, desc = pub.validate()
        if not ok:
            self.app.notify(f"Publish failed: {desc}", severity="error", timeout=5)
            return
        rel = f"{str(cfg.get('github.prefix', 'results')).strip('/')}/{path.name}"
        result = await pub.publish([path], rel_paths=[rel],
                                   commit_message=f"results: chat {self.app.model}")
        self.app.notify(result.summary(),
                        title="Publish" + (" (dry-run)" if result.dry_run else ""),
                        severity="success" if result.ok else "error", timeout=6)

    # ------------------------------------------------------------------ #
    # keys
    # ------------------------------------------------------------------ #

    async def action_interrupt(self) -> None:
        if self._worker and self._worker.is_running:
            self._worker.cancel()
            self.notify("Stopped", severity="info", timeout=2)
            am = getattr(self, "_current_am", None)
            if am is not None and not am.error:
                await am.mark_stopped()

    def action_toggle_think(self) -> None:
        self.app.thinking = not self.app.thinking
        self._handle_command("/think")

    def action_back(self) -> None:
        self.app.pop_screen()
