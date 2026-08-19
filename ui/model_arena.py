"""Model Arena screen - models vs models with blind judging."""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from rich.text import Text
from textual import work
from textual.widgets.option_list import Option
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (Button, Checkbox, Input, Markdown,
                             OptionList, RichLog, Select, Static)

from client import ModelClient
from think_parser import split_thinking
from ui.base import BaseScreen, MUTED, make_header


@dataclass
class ArenaResponse:
    model_key: str
    server: dict
    model: str
    text: str = ""
    ttft: float = 0.0
    total_time: float = 0.0
    token_count: int = 0
    tps: float = 0.0
    error: str = ""


@dataclass
class RoundResult:
    round_num: int
    prompt: str
    responses: List[ArenaResponse] = field(default_factory=list)
    winner_key: str = ""
    judge_scores: Dict[str, int] = field(default_factory=dict)
    judge_explanation: str = ""


TEST_SUITES: Dict[str, dict] = {
    "reasoning": {
        "name": "Reasoning",
        "criteria": [
            "LOGICAL VALIDITY - Is the reasoning sound with no logical fallacies?",
            "CORRECTNESS - Is the final answer right?",
            "EXPLANATION - Is the chain of reasoning clear and followable?",
        ],
        "prompts": [
            "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
            "A bat and ball cost $1.10 together. The bat costs $1 more than the ball. How much does the ball cost?",
            "There are 3 switches outside a room with 1 light inside. You can only enter once. Which switch controls it?",
        ],
    },
    "coding": {
        "name": "Coding",
        "criteria": [
            "CORRECTNESS - Does the code work? Any bugs?",
            "CODE QUALITY - Clean, idiomatic, well-structured?",
            "EDGE CASES - Are boundary conditions handled?",
        ],
        "prompts": [
            "Write a Python function that finds the longest common subsequence of two strings.",
            "Explain the difference between a mutex and a semaphore with code examples.",
            "Debug this code: `def fib(n): return fib(n-1) + fib(n-2)`",
        ],
    },
    "creative": {
        "name": "Creative",
        "criteria": [
            "ORIGINALITY - Is it fresh and surprising?",
            "CRAFT - Quality of writing, word choice, structure?",
            "CONSTRAINT ADHERENCE - Does it follow the prompt's rules?",
        ],
        "prompts": [
            "Write a haiku about a programming bug that turned out to be a feature.",
            "Describe the color blue to someone who has never seen it.",
            "Write a 3-sentence story with a twist ending.",
        ],
    },
    "instruction": {
        "name": "Instruction Following",
        "criteria": [
            "COMPLIANCE - Does it follow every instruction exactly?",
            "PRECISION - No extra content beyond what was asked?",
            "FORMAT - Correct structure/format as requested?",
        ],
        "prompts": [
            "List exactly 5 African countries that start with the letter 'M'. No more, no less.",
            "Translate 'The quick brown fox jumps over the lazy dog' into French, then back to English.",
            "Write a recipe for scrambled eggs using only imperative sentences.",
        ],
    },
    "analysis": {
        "name": "Analysis",
        "criteria": [
            "DEPTH - Goes beyond surface-level observations?",
            "BALANCE - Considers multiple perspectives fairly?",
            "EVIDENCE - Claims are supported, not hand-waved?",
        ],
        "prompts": [
            "What are the main arguments for and against remote work?",
            "Compare the environmental impacts of electric vs gasoline vehicles.",
            "What would happen to the global economy if all fossil fuels ran out tomorrow?",
        ],
    },
}

DEFAULT_CRITERIA = [
    "ACCURACY - Is the information correct?",
    "COMPLETENESS - Does it fully address the question?",
    "CLARITY - Easy to follow and well-organized?",
    "USEFULNESS - Would this actually help someone?",
]


def build_judge_system(criteria: List[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(criteria))
    return f"""You are an impartial judge evaluating AI responses to the same prompt.

Score each response 1-10 on:
{numbered}

Be critical. 7 is "good." Reserve 9-10 for exceptional.
Pick a clear winner. Ties are cop-outs.
Reply with ONLY valid JSON, no other text."""


class VoteModal(ModalScreen):
    """User vote: which response won this round?"""

    DEFAULT_CSS = """
    VoteModal { align: center middle; }
    VoteModal > Vertical {
        width: 60;
        height: 16;
        border: round $primary;
        padding: 1 2;
    }
    VoteModal OptionList { height: 1fr; }
    """

    BINDINGS = [Binding("escape", "skip", "Skip")]

    def __init__(self, prompt: str, labels: List[str]) -> None:
        super().__init__()
        self.labels = labels
        self.result: Optional[str] = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Round winner?", style="bold")
            yield Input(value=self.prompt[:80], disabled=True, id="vm-prompt")
            yield OptionList(id="vm-choices")

    def on_mount(self) -> None:
        lst = self.query_one("#vm-choices", OptionList)
        for i, lab in enumerate(self.labels, 1):
            lst.add_option(Option(Text(f"  {lab}", style="bright_cyan"), id=lab))
        lst.add_option(Option(Text("  skip (no vote)", style=MUTED), id="__skip"))
        lst.highlighted = 0
        lst.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        label = event.option_id
        self.result = None if label == "__skip" else label
        self.dismiss()

    def action_skip(self) -> None:
        self.dismiss()


MODES = [
    ("quick", "Quick Compare", "one custom prompt, you pick the winner"),
    ("battle", "Battle", "suite of prompts, you pick each winner"),
    ("tournament", "Tournament", "suite + LLM judge scores every round"),
]


class ModelArenaScreen(BaseScreen):
    DEFAULT_CSS = """
    ModelArenaScreen #ma-header {
        height: 2;
        content-align: center middle;
    }
    ModelArenaScreen #ma-models {
        height: auto;
        max-height: 10;
    }
    ModelArenaScreen #ma-models-note { height: 1; color: $text-muted; }
    ModelArenaScreen #ma-setup { height: 3; }
    ModelArenaScreen #ma-setup OptionList { height: 100%; width: 4fr; margin: 0 1; }
    ModelArenaScreen #ma-setup Select { height: 100%; width: 3fr; margin: 0 1; }
    ModelArenaScreen #ma-setup Checkbox { height: 100%; margin: 0 1; }
    ModelArenaScreen #ma-prompt { height: 3; width: 80%; align: center middle; }
    ModelArenaScreen #ma-run-btn { height: 2; width: 20; align: center middle; }
    ModelArenaScreen #ma-live { height: 1fr; width: 100%; }
    ModelArenaScreen #ma-panes { height: 1fr; }
    ModelArenaScreen .ma-pane {
        height: 1fr;
        width: 1fr;
        border: round $primary;
        padding: 0 1;
    }
    ModelArenaScreen #ma-log {
        width: 30%;
        height: 1fr;
    }
    ModelArenaScreen #ma-summary { display: none; height: 1fr; }
    ModelArenaScreen #ma-scroll { height: 1fr; padding: 0 1; }
    ModelArenaScreen #ma-actions { height: 2; align: center middle; }
    ModelArenaScreen #ma-actions Button { width: 14; }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "run", "Run"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._all_models: List[Tuple[str, str, str, dict]] = []  # (key, server_ip_port, model, server)
        self._selected: set = set()
        self._mode = "quick"
        self._worker: Optional[work.Worker] = None
        self._rounds: List[RoundResult] = []
        self._labels: List[str] = []
        self._blind = True
        self._summary_md: Optional[str] = None
        self._summary_data: Optional[dict] = None
        self._final_scores: Dict[str, int] = {}
        self._models: List[Tuple[str, str, str, dict]] = []
        self._key_by_label: Dict[str, str] = {}
        self._label_by_key: Dict[str, str] = {}
        self._pane_bodies: List[Static] = []

    # ------------------------------------------------------------------ #

    def build_content(self) -> ComposeResult:
        yield Static("", id="ma-header")
        yield OptionList(id="ma-models")
        yield Static("", id="ma-models-note")
        with Horizontal(id="ma-setup"):
            yield OptionList(id="ma-modes")
            yield Select([("suite: reasoning", "reasoning"),
                         ("suite: coding", "coding"),
                         ("suite: creative", "creative"),
                         ("suite: instruction", "instruction"),
                         ("suite: analysis", "analysis")],
                         prompt="Suite", value="reasoning", id="ma-suite")
            yield Select([("auto", "auto"),
                         ("same as first model", "same as first model")],
                         prompt="Judge", value="auto", id="ma-judge")
            yield Checkbox("blind", value=True, id="ma-blind")
        yield Input(placeholder="custom prompt (quick mode)…", id="ma-prompt")
        yield Button("▶ Run", id="ma-run-btn", variant="primary")
        with Horizontal(id="ma-live"):
            with Horizontal(id="ma-panes"):
                pass
            yield RichLog(id="ma-log")
        with Vertical(id="ma-summary"):
            with Vertical(id="ma-scroll"):
                pass
            with Horizontal(id="ma-actions"):
                yield Button("Save", id="ma-save", variant="primary")
                yield Button("Publish", id="ma-pub")
                yield Button("Again", id="ma-again")
                yield Button("Back", id="ma-back")

    def on_mount(self) -> None:
        t = make_header("Model Arena", f"{len(self.app.servers)} server(s) discovered")
        self.query_one("#ma-header", Static).update(t)
        self._fill_models()
        for key, title, desc in MODES:
            self.query_one("#ma-modes", OptionList).add_option(
                Option(Text.assemble((f"  {title}", "bold"), (f"  -  {desc}", MUTED)), id=key))
        self.query_one("#ma-modes", OptionList).highlighted = 0
        self.query_one("#ma-models-note", Static).update(
            "space: toggle model (min 2)   enter: also toggles   r: run when ready")

    def _fill_models(self) -> None:
        lst = self.query_one("#ma-models", OptionList)
        lst.clear_options()
        self._all_models = []
        for s in self.app.servers:
            for m in s.get("models", []):
                name = m if isinstance(m, str) else (m.get("name") or "?")
                key = f"{s['ip']}:{s['port']}/{name}"
                self._all_models.append((key, f"{s['ip']}:{s['port']}", name, s))
                lst.add_option(Option(Text.assemble(
                    (name, "bright_cyan"), (f"  {s['ip']}:{s['port']}", MUTED)), id=key))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        oid = event.option_id
        if oid in ("quick", "battle", "tournament"):
            self._mode = oid
            self._prompt_visible()
        elif oid in {k for k, *_ in self._all_models}:
            if oid in self._selected:
                self._selected.discard(oid)
            else:
                self._selected.add(oid)
            self._update_selection_note()
            self._prompt_visible()

    def _update_selection_note(self) -> None:
        n = len(self._selected)
        note = f"{n} model(s) selected" + ("   (need at least 2)" if n < 2 else "")
        self.query_one("#ma-models-note", Static).update(
            Text(note, style="yellow" if n < 2 else "green"))

    def _prompt_visible(self) -> None:
        q = self.query_one("#ma-prompt")
        q.styles.display = "block" if self._mode == "quick" else "none"

    # ------------------------------------------------------------------ #

    def action_run(self) -> None:
        app = self.app
        if not app.model:
            return
        if len(self._selected) < 2:
            app.notify("Select at least 2 models", title="Arena", severity="warning", timeout=3)
            return
        if self._mode == "quick" and not self.query_one("#ma-prompt", Input).value.strip():
            app.notify("Enter a prompt for quick compare", severity="warning", timeout=3)
            return
        if self._worker and self._worker.is_running:
            return
        self._worker = self.run_worker(self._run(), name="model-arena", group="ma",
                                       exclusive=True)

    async def _run(self) -> None:
        app = self.app
        log = self.query_one("#ma-log", RichLog)
        panes = self.query_one("#ma-panes")
        panes.remove_children()

        models = [(k, ip_port, name, s) for k, ip_port, name, s in self._all_models
                  if k in self._selected]
        self._rounds = []
        letters = [chr(65 + i) for i in range(len(models))]
        if self.query_one("#ma-blind", Checkbox).value:
            self._blind = True
            order = list(range(len(models)))
            random.shuffle(order)
            self._labels = [letters[i] for i in order]
            self._key_by_label = {self._labels[i]: models[i][0] for i in range(len(models))}
            self._label_by_key = {v: k for k, v in self._key_by_label.items()}
        else:
            self._blind = False
            self._labels = [m[2] for m in models]
            self._key_by_label = {m[2]: m[0] for m in models}
            self._label_by_key = {v: k for k, v in self._key_by_label.items()}

        # one pane per model
        self._pane_bodies: List[Static] = []
        for i, (key, ip_port, name, s) in enumerate(models):
            from textual.containers import Vertical as V
            body = Static("…", id=f"pane-body-{i}")
            pane = V(
                body,
                classes="ma-pane", id=f"pane-{i}",
            )
            panes.mount(pane)
            self._pane_bodies.append(body)
        self._models = models

        # build rounds
        if self._mode == "quick":
            prompt = self.query_one("#ma-prompt", Input).value.strip()
            rounds = [(prompt, DEFAULT_CRITERIA)]
        else:
            suite_key = self.query_one("#ma-suite", Select).value
            suite = TEST_SUITES[suite_key]
            rounds = [(p, suite["criteria"]) for p in suite["prompts"]]

        # judge
        judge_server, judge_model = self._pick_judge(models)

        for i, (prompt, criteria) in enumerate(rounds, 1):
            log.write(Text(f"── round {i}/{len(rounds)} ──", style="bold"))
            log.write(Text(f"  {prompt[:90]}", style=MUTED))
            responses = await self._stream_all(prompt)
            winner_key, scores, expl = "", {}, ""
            if self._mode == "tournament":
                winner_key, scores, expl = await self._auto_judge(
                    judge_server, judge_model, prompt, responses, criteria)
                wlab = self._label_by_key.get(winner_key, "?")
                log.write(Text(f"  judged: {wlab} wins — {expl[:80]}", style="bright_cyan"))
            else:
                modal = VoteModal(prompt, self._labels)
                result = await app.push_screen(modal, wait_for_dismiss=True)
                if result:
                    winner_key = self._key_by_label[result]
                    log.write(Text(f"  voted: {result} wins", style="bright_cyan"))
                else:
                    log.write(Text("  (skipped)", style=MUTED))
            rr = RoundResult(round_num=i, prompt=prompt, responses=responses,
                             winner_key=winner_key, judge_scores=scores,
                             judge_explanation=expl)
            self._rounds.append(rr)
            if winner_key:
                self._final_scores[winner_key] = self._final_scores.get(winner_key, 0) + 1

        self._show_final()

    def _pick_judge(self, models) -> Tuple[dict, str]:
        sel = self.query_one("#ma-judge", Select).value
        if sel == "same as first model":
            return models[0][3], models[0][2]
        # auto: first discovered model that is not competing
        for key, ip_port, name, s in self._all_models:
            if key not in self._selected:
                return s, name
        return models[0][3], models[0][2]

    async def _stream_all(self, prompt: str) -> List[ArenaResponse]:
        import asyncio
        n = len(self._models)
        buffers: List[List[str]] = [[] for _ in range(n)]
        done = [asyncio.Event() for _ in range(n)]
        starts = [0.0] * n
        ends = [0.0] * n
        ttfts = [0.0] * n
        errors = [""] * n
        clients: List[Optional[ModelClient]] = [None] * n
        flush_at = 0.0

        async def stream_one(idx: int):
            _, ip_port, name, s = self._models[idx]
            client = ModelClient(s, name)
            clients[idx] = client
            starts[idx] = time.monotonic()
            first = True
            try:
                async for chunk in client.chat_stream(prompt, []):
                    if first:
                        ttfts[idx] = time.monotonic() - starts[idx]
                        first = False
                    buffers[idx].append(chunk)
            except Exception as e:
                errors[idx] = str(e)
            finally:
                ends[idx] = time.monotonic()
                done[idx].set()

        tasks = [asyncio.create_task(stream_one(i)) for i in range(n)]

        while not all(d.is_set() for d in done):
            await asyncio.sleep(0.2)
            now = time.monotonic()
            if now - flush_at >= 0.1:
                flush_at = now
                for i in range(n):
                    self._pane_bodies[i].update(
                        Text("".join(buffers[i]) or "…",
                             style="dim" if not buffers[i] else "default"))
        for i in range(n):
            self._pane_bodies[i].update(
                Text("".join(buffers[i]) or "(empty)", style="default"))

        await asyncio.gather(*tasks, return_exceptions=True)

        responses: List[ArenaResponse] = []
        for i in range(n):
            key, ip_port, name, s = self._models[i]
            raw = "".join(buffers[i])
            parsed = split_thinking(raw)
            elapsed = ends[i] - starts[i] if ends[i] > 0 else 0.0
            m = clients[i].last_metrics if clients[i] else None
            tokens = m.completion_tokens if m and m.completion_tokens > 0 else 0
            if m and m.eval_duration_ns > 0 and tokens > 0:
                tps = tokens / (m.eval_duration_ns / 1e9)
            elif ttfts[i] > 0 and elapsed > ttfts[i] and tokens > 0:
                tps = tokens / (elapsed - ttfts[i])
            elif elapsed > 0 and tokens > 0:
                tps = tokens / elapsed
            else:
                tps = 0.0
            responses.append(ArenaResponse(
                model_key=key, server=s, model=name,
                text=parsed.content, ttft=ttfts[i], total_time=elapsed,
                token_count=tokens, tps=tps, error=errors[i]))
        return responses

    async def _auto_judge(self, judge_server: dict, judge_model: str,
                          prompt: str, responses: List[ArenaResponse],
                          criteria: List[str]) -> Tuple[str, Dict[str, int], str]:
        n = len(responses)
        letters = [chr(65 + i) for i in range(n)]
        response_text = ""
        for i, r in enumerate(responses):
            txt = r.text[:2000] if r.text else "(error or empty)"
            response_text += f"\nResponse {letters[i]}:\n{txt}\n"
        score_keys = ", ".join(f'"{l}": <1-10>' for l in letters)
        letters_pick = '" or "'.join(letters)
        judge_prompt = (
            f"Question: {prompt}\n{response_text}\n"
            f'Score each response 1-10 and pick a winner. Reply with ONLY this JSON: '
            f'{{"winner": "{letters_pick}", "scores": {{{score_keys}}}, "explanation": "<one sentence>"}}'
        )
        client = ModelClient(judge_server, judge_model)
        try:
            result = await client.chat(judge_prompt,
                                       [{"role": "system", "content": build_judge_system(criteria)}])
            result = split_thinking(result).content
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(result[start:end])
                winner_letter = data.get("winner", "")
                winner_key = self._key_by_label.get(winner_letter, "")
                scores = {self._key_by_label.get(l, l): int(v)
                          for l, v in data.get("scores", {}).items()}
                return winner_key, scores, data.get("explanation", "")
        except Exception as e:
            self.query_one("#ma-log", RichLog).write(
                Text(f"  judge error: {e}", style="bright_red"))
        return "", {}, ""

    # ------------------------------------------------------------------ #

    def _show_final(self) -> None:
        from storage.reports import model_arena_markdown
        models = [k for k, *_ in self._models]
        display_models = [f"{k}" for k in models]
        data = {
            "mode": self._mode,
            "judge": "auto" if self._mode == "tournament" else "user",
            "blind": self._blind,
            "criteria": TEST_SUITES.get(
                self.query_one("#ma-suite", Select).value,
                {}).get("criteria", DEFAULT_CRITERIA) if self._mode != "quick"
                else DEFAULT_CRITERIA,
            "rounds": [
                {
                    "round_num": r.round_num,
                    "prompt": r.prompt,
                    "responses": [
                        {
                            "label": self._label_by_key.get(x.model_key, x.model),
                            "text": x.text,
                            "token_count": x.token_count,
                            "total_time": round(x.total_time, 2),
                            "tps": round(x.tps, 1),
                            "ttft": round(x.ttft, 3),
                            "error": x.error,
                        } for x in r.responses
                    ],
                    "winner_label": self._label_by_key.get(r.winner_key, "") if r.winner_key else "",
                    "judge_explanation": r.judge_explanation,
                } for r in self._rounds
            ],
            "final_scores": {self._label_by_key.get(k, k): v
                             for k, v in self._final_scores.items()},
        }
        self._summary_md = model_arena_markdown(display_models, data)
        self._summary_data = data
        scroller = self.query_one("#ma-scroll")
        scroller.remove_children()
        scroller.mount(Markdown(self._summary_md))
        self.query_one("#ma-live").styles.display = "none"
        self.query_one("#ma-summary").styles.display = "block"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "ma-back":
            self.app.pop_screen()
        elif event.button.id == "ma-again":
            self.query_one("#ma-summary").styles.display = "none"
            self.query_one("#ma-live").styles.display = "block"
            panes = self.query_one("#ma-panes")
            panes.remove_children()
            self.query_one("#ma-models").focused = True
        elif event.button.id == "ma-save":
            self._do_save()
        elif event.button.id == "ma-pub":
            self._do_save(publish=True)

    def _do_save(self, publish: bool = False) -> None:
        if getattr(self, "_summary_md", None) is None:
            return
        app = self.app
        labels = [self._label_by_key.get(k, k) for k, *_ in self._models]
        path = app.store.save_model_arena(labels, self._summary_data,
                                          markdown=self._summary_md)
        app.note_saved()
        self.query_one("#ma-save", Button).label = "Saved ✓"
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
                                   commit_message="results: model_arena")
        self.app.notify(result.summary(), title="Publish",
                        severity="success" if result.ok else "error", timeout=6)

    def action_back(self) -> None:
        self.app.pop_screen()
