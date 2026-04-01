"""Multi-server arena view for comparing models side by side."""
import asyncio
import json
import random
import time
import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.prompt import Prompt, IntPrompt

from client import ModelClient
from scanner import load_cache
from ui.components import format_stats_line, format_menu_item, estimate_tokens


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ArenaResponse:
    """Single model's response to a prompt."""
    model_key: str
    server: dict
    model: str
    text: str = ""
    ttft: float = 0.0      # time to first token
    total_time: float = 0.0
    token_count: int = 0
    tps: float = 0.0
    error: str = ""


@dataclass
class RoundResult:
    """Result of one arena round."""
    round_num: int
    prompt: str
    responses: List[ArenaResponse] = field(default_factory=list)
    winner: str = ""               # model_key voted as winner
    judge_scores: Dict[str, int] = field(default_factory=dict)
    judge_winner: str = ""
    judge_explanation: str = ""


# ── Test suites ──────────────────────────────────────────────────────────────

TEST_SUITES = {
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
            "There are 3 switches outside a room with 1 light inside. You can only enter the room once. How do you figure out which switch controls the light?",
        ],
    },
    "coding": {
        "name": "Coding",
        "criteria": [
            "CORRECTNESS - Does the code work? Any bugs?",
            "CODE QUALITY - Clean, idiomatic, well-structured?",
            "EXPLANATION - Are the concepts explained clearly?",
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
            "CONSTRAINT ADHERENCE - Does it follow the prompt's rules (e.g. '3 sentences')?",
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
            "List exactly 5 countries in Africa that start with the letter 'M'. No more, no less.",
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
            "CALIBRATION - Acknowledges uncertainty where appropriate?",
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
    """Build judge system prompt from suite-specific criteria."""
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))
    return f"""You are an impartial judge evaluating AI responses to the same prompt.

Score each response 1-10 on:
{numbered}

Be critical. 7 is "good." Reserve 9-10 for exceptional.
Pick a clear winner. Ties are cop-outs.
Reply with ONLY valid JSON, no other text."""


# ── Main view ────────────────────────────────────────────────────────────────

class MultiArenaView:
    """Compare multiple models by streaming the same prompt side by side."""

    def __init__(self, console: Console, selected_server: dict, selected_model: str):
        self.console = console
        self.selected_server = selected_server
        self.selected_model = selected_model

        # Arena state
        self.models: List[Tuple[dict, str]] = []
        self.labels: List[str] = []
        self.model_keys: List[str] = []
        self.system_prompt: str = ""
        self.blind_mode: bool = False
        self.blind_labels: List[str] = []   # shuffled display labels
        self.blind_order: List[int] = []    # shuffled index order

        # Scoring
        self.scores: Dict[str, float] = {}
        self.rounds: List[RoundResult] = []

    async def run(self) -> str:
        """Run the multi-server arena."""
        self.console.clear()
        self.console.print()
        self.console.rule("[accent.bold]Arena[/accent.bold]", style="chrome.border")
        self.console.print()

        # 1. Build model list
        choices = self._build_model_list()
        if len(choices) < 2:
            self.console.print("  [status.warn]Need at least 2 models for the arena.[/status.warn]")
            self.console.print("  [chrome]Found {} model(s). Run discovery first.[/chrome]".format(len(choices)))
            self.console.print("\n  [chrome]Press Enter to return...[/chrome]")
            input()
            return "back"

        # 2. Model selection
        self._display_model_list(choices)
        selected = self._get_selections(choices)
        if not selected:
            return "back"

        self.models = selected
        self.labels = [self._make_label(s, m) for s, m in selected]
        self.model_keys = [f"{s['url']}:{m}" for s, m in selected]
        self.scores = {k: 0.0 for k in self.model_keys}

        # 3. Settings
        await self._configure_settings()

        # 4. Mode selection
        mode = self._select_mode()
        if mode is None:
            return "back"

        # 5. Run
        if mode == 1:
            await self._run_quick_compare()
        elif mode == 2:
            await self._run_battle()
        elif mode == 3:
            await self._run_tournament(choices)

        # 6. Offer export
        self._offer_export()

        self.console.print("\n  [chrome]Press Enter to return to chat...[/chrome]")
        input()
        return "back"

    # ── Model selection ──────────────────────────────────────────────────────

    def _build_model_list(self) -> List[Tuple[dict, str]]:
        """Build flat list of (server, model) from cache + current."""
        choices = []
        seen = set()

        key = f"{self.selected_server['url']}:{self.selected_model}"
        choices.append((self.selected_server, self.selected_model))
        seen.add(key)

        cached = load_cache()
        if cached:
            for server in cached:
                for model in server.get("models", []):
                    key = f"{server['url']}:{model}"
                    if key not in seen:
                        choices.append((server, model))
                        seen.add(key)

        return choices

    def _display_model_list(self, choices: List[Tuple[dict, str]]):
        """Display numbered list of available models."""
        table = Table(show_header=True, header_style="accent.bold")
        table.add_column("#", style="chrome", width=4, justify="right")
        table.add_column("Model", style="model.name")
        table.add_column("Server", style="model.server")
        table.add_column("Type", style="chrome", justify="center")

        for i, (server, model) in enumerate(choices, 1):
            marker = " [accent]*[/accent]" if (
                server["url"] == self.selected_server["url"]
                and model == self.selected_model
            ) else ""
            table.add_row(str(i), f"{model}{marker}",
                          f"{server['ip']}:{server['port']}", server["type"].upper())

        self.console.print(table)

    def _get_selections(self, choices: List[Tuple[dict, str]]) -> List[Tuple[dict, str]]:
        """Let user pick up to 6 models (comma-separated)."""
        self.console.print()
        self.console.print("  [chrome]Enter model numbers separated by commas (2-6)[/chrome]")
        self.console.print("  [chrome.muted]Example: 1,3,5[/chrome.muted]")
        self.console.print()

        try:
            raw = Prompt.ask("[prompt]Models[/prompt]")
        except (KeyboardInterrupt, EOFError):
            return []

        selected = []
        seen = set()
        for part in raw.split(","):
            part = part.strip()
            if not part.isdigit():
                continue
            idx = int(part) - 1
            if 0 <= idx < len(choices) and idx not in seen:
                selected.append(choices[idx])
                seen.add(idx)
            if len(selected) >= 6:
                break

        if len(selected) < 2:
            self.console.print("  [status.warn]Need at least 2 models.[/status.warn]")
            return []

        return selected

    def _make_label(self, server: dict, model: str) -> str:
        """Short label for a model."""
        name = model if len(model) <= 25 else model[:22] + "..."
        return f"{name} @ {server['ip']}:{server['port']}"

    # ── Settings ─────────────────────────────────────────────────────────────

    async def _configure_settings(self):
        """Configure arena settings before running."""
        self.console.print()
        self.console.print("  [chrome.header]Settings[/chrome.header]\n")

        # System prompt
        sp = Prompt.ask("  [prompt]System prompt[/prompt]", default="none")
        if sp.lower() != "none" and sp.strip():
            self.system_prompt = sp.strip()
            self.console.print(f"  [status.ok]System prompt set.[/status.ok]")
        self.console.print()

        # Blind mode
        blind = Prompt.ask("  [prompt]Blind mode? (hide model names)[/prompt]",
                           choices=["y", "n"], default="n")
        self.blind_mode = blind.lower() == "y"
        if self.blind_mode:
            self.console.print("  [status.ok]Blind mode ON — identities hidden until reveal.[/status.ok]")
            self._shuffle_blind()
        self.console.print()

    def _shuffle_blind(self):
        """Create shuffled blind labels."""
        n = len(self.models)
        letters = [chr(65 + i) for i in range(n)]  # A, B, C, ...
        self.blind_order = list(range(n))
        random.shuffle(self.blind_order)
        self.blind_labels = [""] * n
        for display_pos, real_idx in enumerate(self.blind_order):
            self.blind_labels[real_idx] = f"Model {letters[display_pos]}"

    def _get_display_label(self, idx: int) -> str:
        """Get display label (blind or real)."""
        if self.blind_mode:
            return self.blind_labels[idx]
        return self.labels[idx]

    def _get_display_key(self, idx: int) -> str:
        """Get display key for scoring (blind or real)."""
        if self.blind_mode:
            return self.blind_labels[idx]
        return self.model_keys[idx]

    # ── Mode selection ───────────────────────────────────────────────────────

    def _select_mode(self) -> Optional[int]:
        """Show mode selection menu."""
        self.console.print("  [chrome.header]Select Mode[/chrome.header]\n")
        self.console.print(format_menu_item("1", "Quick Compare",
                                            "Single prompt, see all responses"))
        self.console.print()
        self.console.print(format_menu_item("2", "Battle",
                                            "Multi-round with manual voting, scoreboard"))
        self.console.print()
        self.console.print(format_menu_item("3", "Tournament",
                                            "Test suite with auto-judge model"))
        self.console.print()
        self.console.print("  [chrome]Q.[/chrome] Cancel\n")

        try:
            choice = Prompt.ask("[prompt]Mode[/prompt]", choices=["1", "2", "3", "q", "Q"])
            if choice.lower() == "q":
                return None
            return int(choice)
        except (ValueError, KeyboardInterrupt):
            return None

    # ── Quick Compare ────────────────────────────────────────────────────────

    async def _run_quick_compare(self):
        """Single prompt, all models respond side by side."""
        self.console.print()
        prompt_text = Prompt.ask("[prompt]Enter prompt[/prompt]")
        if not prompt_text.strip():
            return
        prompt_text = prompt_text.strip()

        responses = await self._stream_all(prompt_text)
        self.rounds.append(RoundResult(round_num=1, prompt=prompt_text, responses=responses))

        self._print_full_responses(responses)

    # ── Battle mode ──────────────────────────────────────────────────────────

    async def _run_battle(self):
        """Multi-round with manual voting."""
        self.console.print()
        self.console.print("  [chrome]Enter prompts each round. /scores to see leaderboard, /quit to end.[/chrome]")
        self.console.print()

        round_num = 0
        while True:
            try:
                raw = Prompt.ask(f"[prompt]Round {round_num + 1}[/prompt]")
            except (KeyboardInterrupt, EOFError):
                break

            raw = raw.strip()
            if not raw:
                continue
            if raw.lower() in ("/quit", "/q"):
                break
            if raw.lower() == "/scores":
                self._show_scoreboard()
                continue

            round_num += 1
            responses = await self._stream_all(raw)

            rr = RoundResult(round_num=round_num, prompt=raw, responses=responses)

            # Print responses
            self._print_full_responses(responses)

            # Vote
            winner_idx = self._vote(responses)
            if winner_idx is not None:
                key = self.model_keys[winner_idx]
                display = self._get_display_label(winner_idx)
                rr.winner = key
                self.scores[key] = self.scores.get(key, 0) + 1
                self.console.print(f"  [status.ok]Vote: {display}[/status.ok]")
            else:
                self.console.print("  [chrome]No vote recorded.[/chrome]")

            self.rounds.append(rr)
            self.console.print()
            self._show_scoreboard()
            self.console.print()

        # Final results
        if self.blind_mode and self.rounds:
            self._reveal_identities()
        if self.rounds:
            self._show_final_results()

    def _vote(self, responses: List[ArenaResponse]) -> Optional[int]:
        """Ask user to vote for the best response."""
        n = len(responses)
        self.console.print()
        self.console.print("  [chrome.header]Vote for best response[/chrome.header]")
        for i in range(n):
            display = self._get_display_label(i)
            self.console.print(f"  [accent]{i + 1}.[/accent] {display}")
        self.console.print("  [chrome]S.[/chrome] Skip\n")

        try:
            choice = Prompt.ask("[prompt]Winner[/prompt]")
            if choice.lower() == "s":
                return None
            idx = int(choice) - 1
            if 0 <= idx < n:
                return idx
        except (ValueError, KeyboardInterrupt):
            pass
        return None

    # ── Tournament mode ──────────────────────────────────────────────────────

    async def _run_tournament(self, all_choices: List[Tuple[dict, str]]):
        """Run test suite with auto-judge."""
        # Select judge model
        judge_server, judge_model = self._select_judge(all_choices)
        if judge_server is None:
            return

        judge_client = ModelClient(judge_server, judge_model)
        judge_label = self._make_label(judge_server, judge_model)
        self.console.print(f"  [status.ok]Judge: {judge_label}[/status.ok]\n")

        # Select test suite — returns (prompts, prompt_to_criteria mapping)
        prompts, prompt_criteria_map = self._select_test_suite()
        if not prompts:
            return

        # Optional: repeat count
        repeats = 1
        try:
            r = Prompt.ask("  [prompt]Repeats per prompt[/prompt]", default="1")
            repeats = max(1, min(5, int(r)))
        except (ValueError, KeyboardInterrupt):
            pass

        total_rounds = len(prompts) * repeats
        self.console.print(f"\n  [status.info]Running {total_rounds} rounds ({len(prompts)} prompts x {repeats} repeats)...[/status.info]\n")

        round_num = 0
        for prompt in prompts:
            for rep in range(repeats):
                round_num += 1
                self.console.print(f"  [chrome.header]Round {round_num}/{total_rounds}[/chrome.header]")
                preview = prompt[:60] + "..." if len(prompt) > 60 else prompt
                self.console.print(f"  [chrome]{preview}[/chrome]\n")

                responses = await self._stream_all(prompt)
                rr = RoundResult(round_num=round_num, prompt=prompt, responses=responses)

                # Auto-judge with suite-specific criteria
                criteria = prompt_criteria_map.get(prompt, DEFAULT_CRITERIA)
                judge_result = await self._auto_judge(judge_client, prompt, responses, criteria)
                if judge_result:
                    rr.judge_scores = judge_result.get("scores", {})
                    rr.judge_winner = judge_result.get("winner", "")
                    rr.judge_explanation = judge_result.get("explanation", "")

                    # Map letter winner back to model key
                    winner_letter = rr.judge_winner
                    letters = [chr(65 + i) for i in range(len(responses))]
                    if winner_letter in letters:
                        winner_idx = letters.index(winner_letter)
                        key = self.model_keys[winner_idx]
                        self.scores[key] = self.scores.get(key, 0) + 1

                    # Apply score points
                    for letter, score in rr.judge_scores.items():
                        if letter in letters:
                            idx = letters.index(letter)
                            # Track as fractional score points too
                            pass

                    self.console.print(f"  [metric]Winner: {rr.judge_winner}[/metric] — {rr.judge_explanation}")
                else:
                    self.console.print("  [status.warn]Judge failed to return valid scores.[/status.warn]")

                self.rounds.append(rr)
                self.console.print()

        if self.blind_mode:
            self._reveal_identities()
        self._show_final_results()

    def _select_judge(self, all_choices: List[Tuple[dict, str]]) -> Tuple[Optional[dict], Optional[str]]:
        """Let user pick a judge model."""
        self.console.print("  [chrome.header]Select Judge Model[/chrome.header]\n")
        self.console.print("  [chrome]The judge evaluates and scores all responses.[/chrome]\n")

        table = Table(show_header=True, header_style="accent.bold")
        table.add_column("#", style="chrome", width=4, justify="right")
        table.add_column("Model", style="model.name")
        table.add_column("Server", style="model.server")

        for i, (server, model) in enumerate(all_choices, 1):
            table.add_row(str(i), model, f"{server['ip']}:{server['port']}")

        self.console.print(table)
        self.console.print()

        try:
            choice = Prompt.ask("[prompt]Judge model number[/prompt]")
            idx = int(choice) - 1
            if 0 <= idx < len(all_choices):
                return all_choices[idx]
        except (ValueError, KeyboardInterrupt):
            pass

        self.console.print("  [status.warn]Invalid selection.[/status.warn]")
        return None, None

    def _select_test_suite(self) -> Tuple[List[str], Dict[str, List[str]]]:
        """Select a test suite. Returns (prompts, {prompt: criteria})."""
        self.console.print("  [chrome.header]Select Test Suite[/chrome.header]\n")

        suites = list(TEST_SUITES.items())
        for i, (key, suite) in enumerate(suites, 1):
            criteria_preview = ", ".join(c.split(" - ")[0] for c in suite["criteria"])
            self.console.print(format_menu_item(
                str(i), suite["name"],
                f"{len(suite['prompts'])} prompts — judged on: {criteria_preview}"
            ))
            self.console.print()

        self.console.print(format_menu_item(str(len(suites) + 1), "All suites",
                                            f"{sum(len(s['prompts']) for s in TEST_SUITES.values())} prompts, per-suite criteria"))
        self.console.print()
        self.console.print(format_menu_item(str(len(suites) + 2), "Custom",
                                            "Enter your own prompts (generic criteria)"))
        self.console.print()

        try:
            choice = Prompt.ask("[prompt]Suite[/prompt]")
            idx = int(choice)
            if 1 <= idx <= len(suites):
                suite = suites[idx - 1][1]
                criteria_map = {p: suite["criteria"] for p in suite["prompts"]}
                return suite["prompts"], criteria_map
            elif idx == len(suites) + 1:
                # All suites — each prompt maps to its suite's criteria
                all_prompts = []
                criteria_map = {}
                for s in TEST_SUITES.values():
                    for p in s["prompts"]:
                        all_prompts.append(p)
                        criteria_map[p] = s["criteria"]
                return all_prompts, criteria_map
            elif idx == len(suites) + 2:
                custom = self._get_custom_prompts()
                return custom, {}  # empty map → falls back to DEFAULT_CRITERIA
        except (ValueError, KeyboardInterrupt):
            pass
        return [], {}

    def _get_custom_prompts(self) -> List[str]:
        """Get custom prompts from user."""
        self.console.print("\n  [chrome]Enter prompts (one per line, empty to finish):[/chrome]\n")
        prompts = []
        while True:
            q = input(f"  P{len(prompts) + 1}: ").strip()
            if not q:
                break
            prompts.append(q)
        return prompts

    async def _auto_judge(self, judge_client: ModelClient, prompt: str,
                          responses: List[ArenaResponse],
                          criteria: Optional[List[str]] = None) -> Optional[Dict]:
        """Have the judge model score all responses using suite-specific criteria."""
        n = len(responses)
        letters = [chr(65 + i) for i in range(n)]

        response_text = ""
        for i, resp in enumerate(responses):
            text = resp.text[:2000] if resp.text else "(error or empty)"
            response_text += f"\nResponse {letters[i]}:\n{text}\n"

        score_keys = ", ".join(f'"{l}": <1-10>' for l in letters)
        judge_prompt = f"""Question: {prompt}
{response_text}
Score each response 1-10 and pick a winner. Reply with ONLY this JSON:
{{"winner": "{'" or "'.join(letters)}", "scores": {{{score_keys}}}, "explanation": "<one sentence>"}}"""

        system = build_judge_system(criteria or DEFAULT_CRITERIA)
        messages = [{"role": "system", "content": system}]

        try:
            result = await judge_client.chat(judge_prompt, messages)
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(result[start:end])
        except Exception as e:
            self.console.print(f"  [status.error]Judge error: {e}[/status.error]")

        return None

    # ── Streaming core ───────────────────────────────────────────────────────

    async def _stream_all(self, prompt: str) -> List[ArenaResponse]:
        """Stream all models in parallel and display side by side."""
        n = len(self.models)
        buffers: List[List[str]] = [[] for _ in range(n)]
        done_events = [asyncio.Event() for _ in range(n)]
        start_times = [0.0] * n
        end_times = [0.0] * n
        ttfts = [0.0] * n
        errors: List[str] = [""] * n

        # Grid layout
        cols = min(n, 3) if n > 2 else n
        rows_count = (n + cols - 1) // cols

        term_height = self.console.size.height
        cell_height = max(6, (term_height - 4) // rows_count - 3)

        layout = self._build_grid_layout(n, cols, rows_count, cell_height)

        # Display labels (blind or real)
        display_labels = [self._get_display_label(i) for i in range(n)]

        # Stream producers
        async def stream_one(idx: int):
            server, model = self.models[idx]
            client = ModelClient(server, model)
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})

            start_times[idx] = time.time()
            try:
                first_chunk = True
                async for chunk in client.chat_stream(prompt, messages):
                    if first_chunk:
                        ttfts[idx] = time.time() - start_times[idx]
                        first_chunk = False
                    buffers[idx].append(chunk)
            except Exception as e:
                errors[idx] = str(e)
            finally:
                end_times[idx] = time.time()
                done_events[idx].set()

        tasks = [asyncio.create_task(stream_one(i)) for i in range(n)]

        # Display loop
        try:
            with Live(layout, console=self.console, refresh_per_second=4) as live:
                while not all(e.is_set() for e in done_events):
                    await asyncio.sleep(0.25)
                    self._update_grid(layout, n, buffers, done_events, errors,
                                      display_labels, cell_height)
                # Final
                self._update_grid(layout, n, buffers, done_events, errors,
                                  display_labels, cell_height)

        except KeyboardInterrupt:
            for t in tasks:
                t.cancel()
            self.console.print("\n  [chrome](cancelled)[/chrome]")

        await asyncio.gather(*tasks, return_exceptions=True)

        # Build response objects
        responses = []
        for i in range(n):
            text = "".join(buffers[i])
            elapsed = end_times[i] - start_times[i] if end_times[i] > 0 else 0
            tokens = estimate_tokens(text)
            tps = tokens / elapsed if elapsed > 0 and tokens > 0 else 0

            responses.append(ArenaResponse(
                model_key=self.model_keys[i],
                server=self.models[i][0],
                model=self.models[i][1],
                text=text,
                ttft=ttfts[i],
                total_time=elapsed,
                token_count=tokens,
                tps=tps,
                error=errors[i],
            ))

        return responses

    def _build_grid_layout(self, n: int, cols: int, rows_count: int,
                           cell_height: int) -> Layout:
        """Build the Rich grid layout for streaming display."""
        layout = Layout()
        row_layouts = []
        cell_idx = 0
        display_labels = [self._get_display_label(i) for i in range(n)]

        for r in range(rows_count):
            row = Layout(name=f"row{r}")
            col_layouts = []
            for c in range(cols):
                if cell_idx < n:
                    cell = Layout(name=f"cell{cell_idx}")
                    cell.update(Panel(
                        "[chrome]waiting...[/chrome]",
                        title=f"[accent]{display_labels[cell_idx]}[/accent]",
                        border_style="chrome.border",
                        height=cell_height,
                    ))
                    col_layouts.append(cell)
                else:
                    filler = Layout(name=f"filler{r}_{c}")
                    filler.update(Panel("", border_style="chrome.border", height=cell_height))
                    col_layouts.append(filler)
                cell_idx += 1
            row.split_row(*col_layouts)
            row_layouts.append(row)

        layout.split_column(*row_layouts)
        return layout

    def _update_grid(self, layout: Layout, n: int,
                     buffers: List[List[str]], done_events: List[asyncio.Event],
                     errors: List[str], display_labels: List[str],
                     cell_height: int):
        """Update all grid cells with current buffer tails."""
        max_lines = cell_height - 3

        for i in range(n):
            text = "".join(buffers[i])
            lines = text.split('\n')

            if errors[i]:
                visible = f"[status.error]{errors[i][:100]}[/status.error]"
                border = "status.error"
            elif done_events[i].is_set():
                visible = '\n'.join(lines[-max_lines:]) if len(lines) > max_lines else text
                border = "status.ok"
            elif text:
                visible = '\n'.join(lines[-max_lines:]) if len(lines) > max_lines else text
                border = "status.running"
            else:
                visible = "[chrome]waiting...[/chrome]"
                border = "chrome.border"

            status_icon = "\u2713" if done_events[i].is_set() else "\u23f3"
            title = f"[accent]{display_labels[i]}[/accent] {status_icon}"

            layout[f"cell{i}"].update(Panel(
                visible, title=title, border_style=border, height=cell_height,
            ))

    # ── Display helpers ──────────────────────────────────────────────────────

    def _print_full_responses(self, responses: List[ArenaResponse]):
        """Print all responses sequentially with stats."""
        self.console.print()
        self.console.rule("[accent.bold]Responses[/accent.bold]", style="chrome.border")

        for i, resp in enumerate(responses):
            display = self._get_display_label(i)
            self.console.print()
            label = Text(f"  {display} ", style="accent.bold")
            label.append("\u25b8", style="chrome")
            self.console.print(label)

            if resp.error:
                self.console.print(f"  [status.error]Error: {resp.error}[/status.error]")
            elif resp.text:
                for line in resp.text.split('\n'):
                    self.console.print(f"  {line}")
                # Stats line with TTFT
                stats = Text()
                stats.append("  \u21b3 ", style="chrome")
                stats.append(str(resp.token_count), style="metric")
                stats.append(" tok ", style="metric.label")
                stats.append("\u00b7 ", style="chrome")
                stats.append(f"{resp.total_time:.1f}s", style="metric")
                stats.append(" \u00b7 ", style="chrome")
                stats.append(f"{resp.tps:.1f}", style="metric")
                stats.append(" t/s", style="metric.label")
                if resp.ttft > 0:
                    stats.append(" \u00b7 ", style="chrome")
                    stats.append(f"TTFT {resp.ttft:.2f}s", style="metric.label")
                self.console.print(stats)
            else:
                self.console.print("  [chrome](empty response)[/chrome]")

        self.console.print()

    def _show_scoreboard(self):
        """Show current scores."""
        self.console.print()
        table = Table(title="Scoreboard", show_header=True, header_style="accent.bold")
        table.add_column("Rank", width=6)
        table.add_column("Model", width=30)
        table.add_column("Wins", justify="right", width=8)

        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (key, score) in enumerate(sorted_scores, 1):
            idx = self.model_keys.index(key)
            display = self._get_display_label(idx)
            style = "status.ok" if rank == 1 else None
            table.add_row(f"#{rank}", display,
                          f"{score:.0f}" if score == int(score) else f"{score:.1f}",
                          style=style)

        self.console.print(table)
        self.console.print()

    def _reveal_identities(self):
        """Reveal blind mode identities."""
        self.console.print()
        self.console.rule("[accent.bold]Identity Reveal[/accent.bold]", style="chrome.border")
        self.console.print()

        for i in range(len(self.models)):
            blind_label = self.blind_labels[i]
            real_label = self.labels[i]
            self.console.print(f"  [metric]{blind_label}[/metric] \u2192 [accent]{real_label}[/accent]")

        self.console.print()

    def _show_final_results(self):
        """Show final results summary."""
        self.console.print()
        self.console.rule("[accent.bold]Final Results[/accent.bold]", style="chrome.border")
        self.console.print()

        # Leaderboard (using real labels now)
        table = Table(title=f"Leaderboard ({len(self.rounds)} rounds)",
                      show_header=True, header_style="accent.bold")
        table.add_column("Rank", width=6)
        table.add_column("Model", width=35)
        table.add_column("Wins", justify="right", width=8)
        table.add_column("Avg TPS", justify="right", width=10)
        table.add_column("Avg TTFT", justify="right", width=10)

        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)

        for rank, (key, score) in enumerate(sorted_scores, 1):
            idx = self.model_keys.index(key)
            real_label = self.labels[idx]

            # Compute average metrics
            tps_vals = [r.responses[idx].tps for r in self.rounds
                        if idx < len(r.responses) and r.responses[idx].tps > 0]
            ttft_vals = [r.responses[idx].ttft for r in self.rounds
                         if idx < len(r.responses) and r.responses[idx].ttft > 0]

            avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0
            avg_ttft = sum(ttft_vals) / len(ttft_vals) if ttft_vals else 0

            style = "status.ok" if rank == 1 else None
            table.add_row(
                f"#{rank}", real_label,
                f"{score:.0f}" if score == int(score) else f"{score:.1f}",
                f"[metric]{avg_tps:.1f}[/metric]" if avg_tps > 0 else "-",
                f"{avg_ttft:.2f}s" if avg_ttft > 0 else "-",
                style=style,
            )

        self.console.print(table)

        # Winner
        if sorted_scores and sorted_scores[0][1] > 0:
            winner_idx = self.model_keys.index(sorted_scores[0][0])
            winner_label = self.labels[winner_idx]
            self.console.print()
            self.console.print(Panel(
                f"[metric]Champion: {winner_label}[/metric]",
                border_style="metric"
            ))

        self.console.print()

    # ── Export ───────────────────────────────────────────────────────────────

    def _offer_export(self):
        """Offer to export results to markdown."""
        if not self.rounds:
            return

        try:
            choice = Prompt.ask("  [prompt]Export results to markdown?[/prompt]",
                                choices=["y", "n"], default="n")
            if choice.lower() == "y":
                filename = self._export_markdown()
                if filename:
                    self.console.print(f"  [status.ok]Exported to {filename}[/status.ok]")
        except (KeyboardInterrupt, EOFError):
            pass

    def _export_markdown(self) -> Optional[str]:
        """Export arena results to markdown."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arena_{ts}.md"

        try:
            with open(filename, "w") as f:
                f.write("# Arena Results\n\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Models
                f.write("## Models\n\n")
                for i, (server, model) in enumerate(self.models):
                    f.write(f"- **{self.labels[i]}** ({server['type']})\n")
                f.write("\n")

                if self.system_prompt:
                    f.write(f"**System Prompt:** {self.system_prompt}\n\n")

                # Scores
                f.write("## Scores\n\n")
                f.write("| Model | Wins |\n|---|---|\n")
                sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
                for key, score in sorted_scores:
                    idx = self.model_keys.index(key)
                    label = self.labels[idx]
                    f.write(f"| {label} | {score:.0f} |\n")
                f.write("\n")

                # Rounds
                f.write("## Rounds\n\n")
                for rr in self.rounds:
                    f.write(f"### Round {rr.round_num}\n\n")
                    f.write(f"**Prompt:** {rr.prompt}\n\n")
                    if rr.winner:
                        idx = self.model_keys.index(rr.winner)
                        f.write(f"**Winner:** {self.labels[idx]}\n\n")
                    if rr.judge_explanation:
                        f.write(f"**Judge:** {rr.judge_explanation}\n\n")
                    for resp in rr.responses:
                        idx = self.model_keys.index(resp.model_key)
                        f.write(f"#### {self.labels[idx]}\n\n")
                        f.write(f"{resp.text}\n\n")
                        f.write(f"*{resp.token_count} tok, {resp.total_time:.1f}s, "
                                f"{resp.tps:.1f} t/s, TTFT {resp.ttft:.2f}s*\n\n")

            return filename
        except Exception as e:
            self.console.print(f"  [status.error]Export failed: {e}[/status.error]")
            return None

    # ── Utilities ────────────────────────────────────────────────────────────

