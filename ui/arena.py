"""Arena view for comparing system prompts."""
import asyncio
import random
import traceback
from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.prompt import Prompt

from prompt_arena import (
    PromptArena,
    PromptResponse,
    JudgeResult,
    ArenaMatchup,
    ArenaResult,
    ArenaStats,
    TEST_QUESTIONS,
)
from ui.components import format_menu_item


class ArenaView:
    """Handle arena interface and visualization."""

    def __init__(self, console: Console, server: dict, model: str):
        self.console = console
        self.server = server
        self.model = model
        self.arena = PromptArena(server, model)
        self.current_responses: List[PromptResponse] = []
        self.current_matchups: List[ArenaMatchup] = []
        self.phase = "idle"  # idle, generating, judging

    async def run(self) -> str:
        """Run the arena interface."""
        self.console.clear()
        self.console.print()
        self.console.rule(
            f"[accent.bold]Prompt Arena[/accent.bold]  [chrome]{self.model}[/chrome]",
            style="chrome.border"
        )
        self.console.print()

        mode = self._select_mode()
        if mode is None:
            return "back"

        if mode == 1:
            await self._run_single_question_mode()
        elif mode == 2:
            await self._run_multi_round_mode()
        elif mode == 3:
            await self._run_custom_prompt_mode()

        self.console.print("\n  [chrome]Press Enter to return to chat...[/chrome]")
        input()

        return "back"

    def _select_mode(self) -> Optional[int]:
        """Show mode selection menu."""
        prompts = self.arena.get_active_prompts()
        self.console.print(f"  [chrome]Active prompts: {', '.join(p['name'] for p in prompts.values())}[/chrome]\n")

        self.console.print("  [chrome.header]Select Mode[/chrome.header]\n")
        self.console.print(format_menu_item("1", "Single Question Tournament", "All prompts compete head-to-head on one question"))
        self.console.print()
        self.console.print(format_menu_item("2", "Multi-Round Battle", "Multiple questions, aggregate statistics"))
        self.console.print()
        self.console.print(format_menu_item("3", "Custom Prompt Setup", "Add or remove system prompts"))
        self.console.print()
        self.console.print("  [chrome]Q.[/chrome] Return to chat\n")

        try:
            choice = Prompt.ask(
                "[prompt]Select mode[/prompt]",
                choices=["1", "2", "3", "q", "Q"]
            )
            if choice.lower() == 'q':
                return None
            return int(choice)
        except (ValueError, KeyboardInterrupt):
            return None

    async def _run_single_question_mode(self):
        """Run single question tournament."""
        self.console.print("\n  [accent.bold]Single Question Tournament[/accent.bold]\n")
        self.console.print("  [chrome]All prompts compete on one question. Best prompt wins![/chrome]\n")

        question = self._get_user_question()
        if not question:
            return

        self.current_responses = []
        self.current_matchups = []

        prompts = self.arena.get_active_prompts()
        layout = self._create_layout(f"Tournament | {len(prompts)} prompts")

        try:
            with Live(layout, console=self.console, refresh_per_second=4) as live:
                async def update_callback(item):
                    if isinstance(item, PromptResponse):
                        existing = next(
                            (r for r in self.current_responses if r.prompt_key == item.prompt_key),
                            None
                        )
                        if existing:
                            idx = self.current_responses.index(existing)
                            self.current_responses[idx] = item
                        else:
                            self.current_responses.append(item)
                        self.phase = "generating"
                    elif isinstance(item, ArenaMatchup):
                        self.current_matchups.append(item)
                        self.phase = "judging"

                    self._update_layout(layout, question)

                result = await self.arena.run_tournament(question, update_callback)

            self._show_tournament_results(result)

        except Exception as e:
            self.console.print(f"\n  [status.error]Tournament failed: {type(e).__name__} - {str(e)}[/status.error]")
            self.console.print(f"  [chrome]{traceback.format_exc()}[/chrome]")

    async def _run_multi_round_mode(self):
        """Run multi-round battle mode."""
        self.console.print("\n  [accent.bold]Multi-Round Battle[/accent.bold]\n")
        self.console.print("  [chrome]Run multiple tournaments, track aggregate performance[/chrome]\n")

        questions = self._select_questions()
        if not questions:
            return

        self.console.print(f"\n  [status.info]Running {len(questions)} rounds...[/status.info]\n")

        try:
            layout = self._create_multi_round_layout(f"Battle | {len(questions)} rounds")

            with Live(layout, console=self.console, refresh_per_second=2) as live:
                async def update_callback(item):
                    if isinstance(item, ArenaStats):
                        self._update_multi_round_layout(layout, item)

                stats = await self.arena.run_multi_round(questions, update_callback)

            self._show_multi_round_summary(stats)

        except Exception as e:
            self.console.print(f"\n  [status.error]Battle failed: {type(e).__name__} - {str(e)}[/status.error]")
            self.console.print(f"  [chrome]{traceback.format_exc()}[/chrome]")

    async def _run_custom_prompt_mode(self):
        """Run custom prompt management."""
        self.console.print("\n  [accent.bold]Custom Prompt Setup[/accent.bold]\n")

        while True:
            prompts = self.arena.get_active_prompts()

            self.console.print("  [chrome.header]Current Prompts[/chrome.header]\n")
            for i, (key, info) in enumerate(prompts.items(), 1):
                preview = info["prompt"][:60] + "..." if len(info["prompt"]) > 60 else info["prompt"]
                self.console.print(f"  [accent]{i}.[/accent] {info['name']} ({key})")
                self.console.print(f"     [chrome.muted]{preview}[/chrome.muted]\n")

            self.console.print("  [chrome.header]Options[/chrome.header]")
            self.console.print("  [accent]A[/accent] Add custom prompt")
            self.console.print("  [accent]R[/accent] Remove prompt")
            self.console.print("  [accent]D[/accent] Reset to defaults")
            self.console.print("  [accent]Q[/accent] Done\n")

            choice = Prompt.ask(
                "[prompt]Select option[/prompt]",
                choices=["a", "A", "r", "R", "d", "D", "q", "Q"]
            ).lower()

            if choice == 'q':
                break
            elif choice == 'a':
                self._add_custom_prompt()
            elif choice == 'r':
                self._remove_prompt()
            elif choice == 'd':
                self.arena.reset_to_defaults()
                self.console.print("  [status.ok]Reset to default prompts.[/status.ok]\n")

    def _add_custom_prompt(self):
        """Add a custom system prompt."""
        self.console.print("\n  [accent.bold]Add Custom Prompt[/accent.bold]\n")

        key = Prompt.ask("[prompt]Prompt key (short identifier)[/prompt]")
        if not key:
            return

        name = Prompt.ask("[prompt]Display name[/prompt]")
        if not name:
            return

        self.console.print("  [chrome]Enter the system prompt (press Enter twice to finish):[/chrome]")

        lines = []
        while True:
            line = input()
            if line == "":
                if lines and lines[-1] == "":
                    break
            lines.append(line)

        prompt = "\n".join(lines).strip()
        if not prompt:
            self.console.print("  [status.warn]Empty prompt, cancelled.[/status.warn]")
            return

        if self.arena.add_custom_prompt(key, name, prompt):
            self.console.print(f"  [status.ok]Added prompt '{name}'.[/status.ok]\n")
        else:
            self.console.print(f"  [status.warn]Key '{key}' already exists.[/status.warn]\n")

    def _remove_prompt(self):
        """Remove a prompt."""
        prompts = self.arena.get_active_prompts()
        keys = list(prompts.keys())

        self.console.print("\n  [chrome.header]Remove Prompt[/chrome.header]")
        self.console.print("  [chrome]Note: 'basic' cannot be removed[/chrome]\n")

        for i, key in enumerate(keys, 1):
            self.console.print(f"  {i}. {prompts[key]['name']} ({key})")

        try:
            choice = Prompt.ask("\n[prompt]Enter number to remove (or Q to cancel)[/prompt]")
            if choice.lower() == 'q':
                return

            idx = int(choice) - 1
            if 0 <= idx < len(keys):
                key = keys[idx]
                if self.arena.remove_prompt(key):
                    self.console.print(f"  [status.ok]Removed '{key}'.[/status.ok]\n")
                else:
                    self.console.print(f"  [status.warn]Cannot remove '{key}'.[/status.warn]\n")
        except (ValueError, IndexError):
            self.console.print("  [status.warn]Invalid selection.[/status.warn]\n")

    def _get_user_question(self) -> Optional[str]:
        """Get question from user."""
        self.console.print("  [chrome.header]Enter your question[/chrome.header]")
        self.console.print("  [chrome]Or press Enter for a sample question[/chrome]\n")

        question = Prompt.ask("[prompt]Question[/prompt]", default="")

        if not question:
            all_questions = []
            for qs in TEST_QUESTIONS.values():
                all_questions.extend(qs)
            question = random.choice(all_questions)
            self.console.print(f"  [chrome]Using sample: {question}[/chrome]\n")

        return question

    def _select_questions(self) -> List[str]:
        """Select questions for multi-round mode."""
        self.console.print("  [chrome.header]Select Questions[/chrome.header]\n")
        self.console.print(format_menu_item("1", "Quick battle", "3 questions"))
        self.console.print(format_menu_item("2", "Standard battle", "5 questions"))
        self.console.print(format_menu_item("3", "Extended battle", "10 questions"))
        self.console.print(format_menu_item("4", "Custom questions"))
        self.console.print()

        choice = Prompt.ask(
            "[prompt]Select option[/prompt]",
            choices=["1", "2", "3", "4"],
            default="2"
        )

        if choice == "4":
            return self._get_custom_questions()

        all_questions = self.arena.get_test_questions()
        counts = {"1": 3, "2": 5, "3": 10}
        count = min(counts[choice], len(all_questions))

        return random.sample(all_questions, count)

    def _get_custom_questions(self) -> List[str]:
        """Get custom questions from user."""
        self.console.print("\n  [chrome]Enter questions (one per line, empty line to finish):[/chrome]\n")

        questions = []
        while True:
            q = input(f"  Q{len(questions)+1}: ").strip()
            if not q:
                break
            questions.append(q)

        return questions

    def _create_layout(self, title: str) -> Layout:
        """Create layout for tournament display."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="question", size=5),
            Layout(name="main", ratio=3),
            Layout(name="footer", ratio=1)
        )
        layout["main"].split_row(
            Layout(name="responses", ratio=2),
            Layout(name="matchups", ratio=1)
        )

        layout["header"].update(Panel(title, style="accent.bold"))
        layout["question"].update(Panel("[chrome]Waiting...[/chrome]", title="Question"))
        layout["responses"].update(Panel("[chrome]Generating responses...[/chrome]", title="Responses"))
        layout["matchups"].update(Panel("[chrome]Waiting for judging...[/chrome]", title="Matchups"))
        layout["footer"].update(Panel("[chrome]Starting...[/chrome]", title="Progress"))

        return layout

    def _update_layout(self, layout: Layout, question: str):
        """Update layout with current state."""
        q_text = question[:100] + "..." if len(question) > 100 else question
        layout["question"].update(Panel(q_text, title="Question", border_style="chrome.border"))
        layout["responses"].update(self._create_responses_panel())
        layout["matchups"].update(self._create_matchups_panel())
        layout["footer"].update(self._create_progress_panel())

    def _create_responses_panel(self) -> Panel:
        """Create panel showing response status."""
        if not self.current_responses:
            return Panel("[chrome]Generating responses...[/chrome]", title="Responses")

        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_column(justify="left")

        for i in range(0, len(self.current_responses), 3):
            row_responses = self.current_responses[i:i+3]
            cells = []

            for resp in row_responses:
                status_icon = {
                    "pending": "...",
                    "running": "...",
                    "success": "\u2713",
                    "error": "\u2717"
                }.get(resp.status, "?")

                status_style = {
                    "pending": "chrome",
                    "running": "status.running",
                    "success": "status.ok",
                    "error": "status.error"
                }.get(resp.status, "chrome")

                cell = Text()
                cell.append(f"{resp.prompt_name}\n", style="accent.bold")
                cell.append(status_icon, style=status_style)

                if resp.status == "success":
                    cell.append(f"\n{resp.duration:.1f}s", style="metric.label")

                cells.append(Panel(cell, width=24, height=5))

            while len(cells) < 3:
                cells.append(Panel("", width=24, height=5, border_style="chrome.border"))

            table.add_row(*cells)

        completed = sum(1 for r in self.current_responses if r.status == "success")
        total = len(self.current_responses)

        return Panel(table, title=f"Responses ({completed}/{total})")

    def _create_matchups_panel(self) -> Panel:
        """Create panel showing matchup results."""
        if not self.current_matchups:
            if self.phase == "generating":
                return Panel("[chrome]Waiting for responses...[/chrome]", title="Matchups")
            return Panel("[chrome]Starting judging...[/chrome]", title="Matchups")

        text = Text()
        for matchup in self.current_matchups[-8:]:
            if matchup.judge_result:
                jr = matchup.judge_result
                a_name = matchup.response_a.prompt_name
                b_name = matchup.response_b.prompt_name

                if jr.winner == "A":
                    text.append(f"{a_name}", style="status.ok")
                    text.append(f" > {b_name}\n", style="chrome")
                elif jr.winner == "B":
                    text.append(f"{a_name}", style="chrome")
                    text.append(" < ", style="chrome")
                    text.append(f"{b_name}\n", style="status.ok")
                else:
                    text.append(f"{a_name} = {b_name}\n", style="status.warn")

        total_matchups = len(self.arena.prompts) * (len(self.arena.prompts) - 1) // 2
        completed = len(self.current_matchups)

        return Panel(text, title=f"Matchups ({completed}/{total_matchups})")

    def _create_progress_panel(self) -> Panel:
        """Create progress panel."""
        prompts = self.arena.get_active_prompts()
        total_prompts = len(prompts)
        completed_responses = sum(1 for r in self.current_responses if r.status == "success")
        total_matchups = total_prompts * (total_prompts - 1) // 2
        completed_matchups = len(self.current_matchups)

        if self.phase == "generating":
            status = f"Generating responses: {completed_responses}/{total_prompts}"
        elif self.phase == "judging":
            status = f"Judging matchups: {completed_matchups}/{total_matchups}"
        else:
            status = "Starting..."

        return Panel(status, title="Progress", border_style="chrome.border")

    def _create_multi_round_layout(self, title: str) -> Layout:
        """Create layout for multi-round display."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=4),
            Layout(name="footer", size=5)
        )

        layout["header"].update(Panel(title, style="accent.bold"))
        layout["main"].update(Panel("[chrome]Starting battle...[/chrome]", title="Leaderboard"))
        layout["footer"].update(Panel("[chrome]Round 0/0[/chrome]", title="Progress"))

        return layout

    def _update_multi_round_layout(self, layout: Layout, stats: ArenaStats):
        """Update multi-round layout."""
        table = Table(show_header=True, header_style="accent.bold")
        table.add_column("Rank", width=6)
        table.add_column("Prompt", width=20)
        table.add_column("Wins", justify="right", width=8)
        table.add_column("Avg Score", justify="right", width=10)

        sorted_prompts = sorted(
            stats.prompt_wins.items(),
            key=lambda x: (x[1], stats.get_avg_score(x[0])),
            reverse=True
        )

        for rank, (key, wins) in enumerate(sorted_prompts, 1):
            avg_score = stats.get_avg_score(key)
            prompt_info = self.arena.prompts.get(key, {"name": key})
            table.add_row(
                f"#{rank}",
                prompt_info["name"],
                f"{wins:.1f}" if wins % 1 else str(int(wins)),
                f"{avg_score:.1f}" if avg_score > 0 else "-"
            )

        layout["main"].update(Panel(table, title="Leaderboard"))
        layout["footer"].update(Panel(
            f"Round {stats.completed_rounds}/{stats.total_rounds}",
            title="Progress",
            border_style="chrome.border"
        ))

    def _show_tournament_results(self, result: ArenaResult):
        """Display final tournament results."""
        self.console.print("\n  [status.ok]Tournament Complete[/status.ok]\n")

        if result.winner:
            winner_info = self.arena.prompts.get(result.winner, {"name": result.winner})
            self.console.print(Panel(
                f"[metric]Winner: {winner_info['name']}[/metric]",
                border_style="metric"
            ))
            self.console.print()

        table = Table(title="Final Standings", show_header=True, header_style="accent.bold")
        table.add_column("Rank", width=6)
        table.add_column("Prompt", width=20)
        table.add_column("Wins", justify="right", width=8)
        table.add_column("Avg Score", justify="right", width=10)
        table.add_column("Time", justify="right", width=10)

        sorted_prompts = sorted(
            result.rankings.items(),
            key=lambda x: (x[1], result.get_avg_score(x[0])),
            reverse=True
        )

        for rank, (key, wins) in enumerate(sorted_prompts, 1):
            avg_score = result.get_avg_score(key)
            prompt_info = self.arena.prompts.get(key, {"name": key})
            resp = next((r for r in result.responses if r.prompt_key == key), None)
            time_str = f"{resp.duration:.1f}s" if resp else "-"

            style = "status.ok" if rank == 1 else None
            table.add_row(
                f"#{rank}",
                prompt_info["name"],
                f"{wins:.1f}" if wins % 1 else str(int(wins)),
                f"{avg_score:.1f}" if avg_score > 0 else "-",
                time_str,
                style=style
            )

        self.console.print(table)

        if result.matchups:
            self.console.print("\n  [chrome.header]Sample Matchups[/chrome.header]\n")
            for matchup in result.matchups[:3]:
                if matchup.judge_result:
                    jr = matchup.judge_result
                    a = matchup.response_a.prompt_name
                    b = matchup.response_b.prompt_name

                    if jr.winner == "A":
                        self.console.print(f"  [status.ok]{a}[/status.ok] > {b}: {jr.explanation}")
                    elif jr.winner == "B":
                        self.console.print(f"  {a} < [status.ok]{b}[/status.ok]: {jr.explanation}")
                    else:
                        self.console.print(f"  [status.warn]{a} = {b}[/status.warn]: {jr.explanation}")

    def _show_multi_round_summary(self, stats: ArenaStats):
        """Display multi-round battle summary."""
        self.console.print("\n  [status.ok]Battle Complete[/status.ok]\n")

        if stats.prompt_wins:
            winner_key = max(
                stats.prompt_wins.keys(),
                key=lambda k: (stats.prompt_wins[k], stats.get_avg_score(k))
            )
            winner_info = self.arena.prompts.get(winner_key, {"name": winner_key})

            self.console.print(Panel(
                f"[metric]Champion: {winner_info['name']}[/metric]\n"
                f"[chrome]{stats.prompt_wins[winner_key]} total wins | "
                f"Avg score: {stats.get_avg_score(winner_key):.1f}[/chrome]",
                border_style="metric"
            ))
            self.console.print()

        table = Table(
            title=f"Final Standings ({stats.completed_rounds} rounds)",
            show_header=True, header_style="accent.bold"
        )
        table.add_column("Rank", width=6)
        table.add_column("Prompt", width=20)
        table.add_column("Total Wins", justify="right", width=12)
        table.add_column("Win Rate", justify="right", width=10)
        table.add_column("Avg Score", justify="right", width=10)

        sorted_prompts = sorted(
            stats.prompt_wins.items(),
            key=lambda x: (x[1], stats.get_avg_score(x[0])),
            reverse=True
        )

        total_matchups = sum(stats.prompt_wins.values())

        for rank, (key, wins) in enumerate(sorted_prompts, 1):
            prompt_info = self.arena.prompts.get(key, {"name": key})
            win_rate = (wins / total_matchups * 100) if total_matchups > 0 else 0
            avg_score = stats.get_avg_score(key)

            style = "status.ok" if rank == 1 else None
            table.add_row(
                f"#{rank}",
                prompt_info["name"],
                f"{wins:.1f}" if wins % 1 else str(int(wins)),
                f"{win_rate:.1f}%",
                f"{avg_score:.1f}" if avg_score > 0 else "-",
                style=style
            )

        self.console.print(table)

        self.console.print("\n  [chrome.header]Round Winners[/chrome.header]\n")
        for i, result in enumerate(stats.results, 1):
            winner_info = self.arena.prompts.get(result.winner, {"name": result.winner})
            q_preview = result.question[:40] + "..." if len(result.question) > 40 else result.question
            self.console.print(f"  R{i}: [status.ok]{winner_info['name']}[/status.ok] - [chrome]{q_preview}[/chrome]")
