"""Prompt Arena screen - system-prompt tournament."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.text import Text
from textual import work
from textual.widgets.option_list import Option
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (Button, DataTable, Input, Markdown,
                             OptionList, RichLog, Select, Static, TextArea)

from prompt_arena import (PromptArena, TEST_QUESTIONS,
                          PromptResponse, ArenaMatchup, ArenaStats)
from storage.reports import prompt_arena_markdown
from ui.base import BaseScreen, MUTED, make_header

MODES = [
    ("single", "Single Question", "all prompts face one question, round-robin judging"),
    ("multi", "Multi-Round Battle", "several questions, aggregated standings"),
]


class PromptModal(ModalScreen):
    """Add a custom system prompt."""

    DEFAULT_CSS = """
    PromptModal { align: center middle; }
    PromptModal > Vertical {
        width: 80;
        height: 22;
        border: round $primary;
        padding: 1 2;
    }
    PromptModal TextArea { height: 1fr; }
    PromptModal #pm-row {
        height: 3;
        content-align: left middle;
    }
    PromptModal Input { width: 25; }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    result: Optional[Dict[str, str]] = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("New custom system prompt\n", style="bold")
            with Horizontal(id="pm-row"):
                yield Input(placeholder="key (e.g. mystyle)", id="pm-key")
                yield Input(placeholder="display name", id="pm-name")
            yield TextArea("", id="pm-body")
            with Horizontal():
                yield Button("Add", id="pm-add", variant="primary")
                yield Button("Cancel", id="pm-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pm-add":
            key = self.query_one("#pm-key", Input).value.strip().lower().replace(" ", "_")
            name = self.query_one("#pm-name", Input).value.strip() or key
            body = self.query_one("#pm-body", TextArea).text
            if key and body.strip():
                self.result = {"key": key, "name": name, "prompt": body}
                self.dismiss(self.result)
        else:
            self.dismiss()

    def action_cancel(self) -> None:
        self.dismiss()


class PromptArenaScreen(BaseScreen):
    DEFAULT_CSS = """
    PromptArenaScreen #pa-header {
        height: 2;
        content-align: center middle;
        border-bottom: solid $panel;
    }
    PromptArenaScreen #pa-modes {
        height: auto;
        max-height: 8;
        border: round $panel;
        border-title-color: $primary;
    }
    PromptArenaScreen #pa-category {
        width: 26;
        margin: 0 1;
    }
    PromptArenaScreen #pa-question { height: 3; width: 80%; align: center middle; }
    PromptArenaScreen #pa-prompts {
        height: auto;
        max-height: 10;
        border: round $panel;
        border-title-color: $primary;
    }
    PromptArenaScreen #pa-prompt-actions { height: 2; }
    PromptArenaScreen #pa-prompt-actions Button { width: 16; margin: 0 1; }
    PromptArenaScreen #pa-run-btn { height: 2; content-align: center middle; width: 12; align: center middle; }
    PromptArenaScreen #pa-live { height: 1fr; width: 100%; }
    PromptArenaScreen #pa-table {
        width: 55%;
        border: round $panel;
        border-title-color: $primary;
    }
    PromptArenaScreen #pa-log {
        width: 45%;
        height: 1fr;
        border: round $panel;
        border-title-color: $primary;
    }
    PromptArenaScreen #pa-summary { display: none; height: 1fr; }
    PromptArenaScreen #pa-scroll { height: 1fr; padding: 0 1; }
    PromptArenaScreen #pa-actions { height: 2; align: center middle; }
    PromptArenaScreen #pa-actions Button { width: 14; }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "run", "Run"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._mode = "single"
        self._category = "reasoning"
        self._arena: PromptArena | None = None
        self._worker: Optional[work.Worker] = None
        self._data: Optional[dict] = None
        self._summary_md: Optional[str] = None

    def build_content(self) -> ComposeResult:
        yield Static("", id="pa-header")
        yield OptionList(id="pa-modes")
        yield Select([("reasoning", "reasoning"), ("creative", "creative"),
                     ("technical", "technical"), ("analysis", "analysis"),
                     ("practical", "practical")],
                     prompt="Question bank", value="reasoning", id="pa-category",
                     classes="pa-cat")
        yield Input(placeholder="enter a question… (or leave for bank)", id="pa-question")
        yield OptionList(id="pa-prompts")
        with Horizontal(id="pa-prompt-actions"):
            yield Button("➕ Add prompt", id="pa-add")
            yield Button("↺ Reset", id="pa-reset")
        yield Button("▶ Run", id="pa-run-btn", variant="primary")
        with Horizontal(id="pa-live"):
            yield DataTable(id="pa-table")
            yield RichLog(id="pa-log")
        with Vertical(id="pa-summary"):
            with Vertical(id="pa-scroll"):
                pass
            with Horizontal(id="pa-actions"):
                yield Button("Save", id="pa-save", variant="primary")
                yield Button("Publish", id="pa-pub")
                yield Button("Back", id="pa-back")

    def on_mount(self) -> None:
        t = make_header("Prompt Arena", self.app.model)
        self.query_one("#pa-header", Static).update(t)
        for key, title, desc in MODES:
            self.query_one("#pa-modes", OptionList).add_option(
                Option(Text.assemble((f"  {title}", "bold"), (f"  -  {desc}", MUTED)), id=key))
        self.query_one("#pa-modes", OptionList).highlighted = 0
        self._apply_mode()
        table = self.query_one("#pa-table", DataTable)
        table.cursor_type = None
        for c in ("prompt", "status", "time(s)", "tok"):
            table.add_column(c)
        self.query_one("#pa-modes", OptionList).border_title = "mode"
        self.query_one("#pa-prompts", OptionList).border_title = "prompts"
        self.query_one("#pa-table", DataTable).border_title = "leaderboard"
        self.query_one("#pa-log", RichLog).border_title = "matchups"
        self._arena = PromptArena(self.app.server, self.app.model)
        self._populate_prompts()

    def _populate_prompts(self) -> None:
        lst = self.query_one("#pa-prompts", OptionList)
        lst.clear_options()
        for key, p in self._arena.get_active_prompts().items():
            custom = " ·custom" if key not in ("basic", "cot", "aot", "deep_cot",
                                               "failure_first", "methodical", "concise") else ""
            lst.add_option(Option(Text.assemble(
                (f"  {p['name']}", "bright_cyan"), (f"  ({key}{custom})", MUTED)), id=key))
        if lst.option_count:
            lst.highlighted = 0

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id in ("single", "multi"):
            self._mode = event.option_id
            self._apply_mode()
        elif event.option_id in self._arena.get_active_prompts():
            p = self._arena.get_active_prompts()[event.option_id]
            self.query_one("#pa-log", RichLog).write(
                Text(f"── {p['name']} ──\n{p['prompt'][:400]}", style=MUTED))

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "pa-category":
            self._category = str(event.value)

    def _apply_mode(self) -> None:
        cat = self.query_one("#pa-category")
        q = self.query_one("#pa-question")
        if self._mode == "multi":
            cat.styles.display = "block"
            q.styles.display = "none"
            q.value = ""
        else:
            cat.styles.display = "none"
            q.styles.display = "block"

    # ------------------------------------------------------------------ #

    def action_run(self) -> None:
        app = self.app
        if not app.model or not app.server:
            return
        if self._worker and self._worker.is_running:
            return
        question = self.query_one("#pa-question", Input).value.strip()
        if self._mode == "single":
            if not question:
                self.app.notify("Type a question first", severity="warning", timeout=3)
                return
            questions = [question]
        else:
            questions = TEST_QUESTIONS.get(self._category, [])[:2]
        self._worker = self.run_worker(self._run(questions), name="arena", group="arena",
                                       exclusive=True)

    async def _run(self, questions: List[str]) -> None:
        app = self.app
        self.query_one("#pa-live").styles.display = "block"
        self.query_one("#pa-summary").styles.display = "none"

        table = self.query_one("#pa-table", DataTable)
        log = self.query_one("#pa-log", RichLog)
        table.clear()

        async def cb(item: Any) -> None:
            if isinstance(item, PromptResponse):
                row = (item.prompt_name, item.status,
                       f"{item.duration:.1f}" if item.end_time else "-",
                       str(item.token_count or "-"))
                try:
                    table.add_row(row, key=item.prompt_key)
                except Exception:
                    pass
            elif isinstance(item, ArenaMatchup):
                j = item.judge_result
                verdict = {
                    "A": f"{item.response_a.prompt_name} beats",
                    "B": f"{item.response_b.prompt_name} beats",
                    "TIE": "tie between",
                }.get(j.winner, "?")
                log.write(Text(f"  {verdict} "
                               f"{item.response_a.prompt_name} vs {item.response_b.prompt_name} "
                               f"({j.score_a}–{j.score_b})", style="bright_cyan"))
            elif isinstance(item, ArenaStats):
                log.write(Text(f"  round done: {item.completed_rounds}/{item.total_rounds}",
                               style=MUTED))

        if len(questions) == 1:
            result = await self._arena.run_tournament(questions[0], cb)
            self._data = self._tournament_data(result)
        else:
            stats = await self._arena.run_multi_round(questions, cb)
            self._data = self._multi_round_data(stats)
        self._summary_md = prompt_arena_markdown(
            app.model, f"{app.server['ip']}:{app.server['port']}", self._data)
        self._show_summary()

    # ------------------------------------------------------------------ #

    def _tournament_data(self, r) -> dict:
        prompts = {k: v["name"] for k, v in self._arena.get_active_prompts().items()}
        responses = [{
            "prompt_key": x.prompt_key,
            "response": x.response,
            "status": x.status,
            "duration": round(x.duration, 1),
            "error_msg": x.error_msg,
        } for x in r.responses]
        matchups = [{
            "a": m.response_a.prompt_name,
            "b": m.response_b.prompt_name,
            "winner": m.judge_result.winner,
            "score_a": m.judge_result.score_a,
            "score_b": m.judge_result.score_b,
            "explanation": m.judge_result.explanation,
        } for m in r.matchups]
        return {
            "mode": "tournament",
            "prompts": prompts,
            "question": r.question,
            "responses": responses,
            "matchups": matchups,
            "rankings": r.rankings,
            "avg_scores": {k: r.get_avg_score(k) for k in r.rankings},
            "winner": r.winner,
        }

    def _multi_round_data(self, s: ArenaStats) -> dict:
        prompts = {k: v["name"] for k, v in self._arena.get_active_prompts().items()}
        rounds = [{
            "question": r.question,
            "rankings": r.rankings,
            "winner": r.winner,
            "avg_scores": {k: r.get_avg_score(k) for k in r.rankings},
        } for r in s.results]
        return {
            "mode": "multi_round",
            "prompts": prompts,
            "questions": [r.question for r in s.results],
            "rounds": rounds,
            "totals": s.prompt_wins,
            "win_rates": {k: s.get_win_rate(k) for k in s.prompt_wins},
            "avg_scores": {k: s.get_avg_score(k) for k in s.prompt_wins},
        }

    def _show_summary(self) -> None:
        from textual.widgets import Markdown
        scroller = self.query_one("#pa-scroll")
        scroller.remove_children()
        scroller.mount(Markdown(self._summary_md))
        self.query_one("#pa-live").styles.display = "none"
        self.query_one("#pa-summary").styles.display = "block"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "pa-back":
            self.app.pop_screen()
        elif event.button.id == "pa-save":
            self._do_save()
        elif event.button.id == "pa-pub":
            self._do_save(publish=True)
        elif event.button.id == "pa-add":
            self.app.push_screen(PromptModal(), callback=self._prompt_added)
        elif event.button.id == "pa-reset":
            self._arena.reset_to_defaults()
            self._populate_prompts()
            self.notify("Prompts reset to the 7 built-ins", severity="info", timeout=3)

    def _prompt_added(self, result: Optional[Dict[str, str]]) -> None:
        if not result:
            return
        ok = self._arena.add_custom_prompt(
            result["key"], result["name"], result["prompt"])
        if ok:
            self._populate_prompts()
            self.notify(f"Added custom prompt: {result['name']}",
                        title="Custom prompt", severity="success", timeout=3)
        else:
            self.notify(f"Could not add '{result['key']}' (invalid or duplicate key)",
                        severity="error", timeout=4)

    def _do_save(self, publish: bool = False) -> None:
        if self._data is None:
            return
        app = self.app
        path = app.store.save_prompt_arena(
            app.model, f"{app.server['ip']}:{app.server['port']}",
            self._data, markdown=self._summary_md or "")
        app.note_saved()
        self.query_one("#pa-save", Button).label = "Saved ✓"
        app.notify(f"Saved: {path}", title="Saved", severity="success", timeout=4)
        if publish:
            self._publish(path)

    async def _publish(self, path) -> None:
        from pathlib import Path
        from publish import Publisher
        cfg = self.app.config
        pub = Publisher(cfg.get("github.repo_path", ""),
                        branch=cfg.get("github.branch", "main"),
                        prefix=cfg.get("github.prefix", "results"),
                        push=cfg.get("github.push", True),
                        dry_run=cfg.get("github.dry_run", False))
        ok, desc = pub.validate()
        if not ok:
            self.app.notify(f"Publish failed: {desc}", severity="error", timeout=5)
            return
        rel = f"{str(cfg.get('github.prefix', 'results')).strip('/')}/{Path(path).name}"
        result = await pub.publish([Path(path)], rel_paths=[rel],
                                   commit_message=f"results: prompt_arena {self.app.model}")
        self.app.notify(result.summary(), title="Publish",
                        severity="success" if result.ok else "error", timeout=6)

    def action_back(self) -> None:
        self.app.pop_screen()
