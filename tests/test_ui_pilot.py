"""Headless Textual pilot tests for ModelChatApp.

Runs from an empty CWD (legacy migration is a no-op) with HOME pointed at a
temp dir, and network scanning monkeypatched to a fake server. Each test
boots a fresh app via run_test(). Also runnable standalone:

    venv/bin/python tests/test_ui_pilot.py
"""
import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace

from contextlib import asynccontextmanager

TESTHOME = tempfile.mkdtemp(prefix="mc_pilot_home_")
TESTCWD = tempfile.mkdtemp(prefix="mc_pilot_cwd_")
os.environ["HOME"] = TESTHOME
os.chdir(TESTCWD)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scanner  # noqa: E402
import stress_tester  # noqa: E402
import prompt_arena as prompt_arena_mod  # noqa: E402
import ui.stress as stress_mod  # noqa: E402
import ui.model_arena as model_arena_mod  # noqa: E402

FAKE_SERVERS = [
    {
        "ip": "127.0.0.1", "port": 9999, "url": "http://127.0.0.1:9999",
        "type": "openai", "models": ["fake-model-a", "fake-model-b"],
        "status": "healthy", "response_time": 3,
    },
]


async def fake_validate(servers, progress_callback=None):
    return list(servers)


async def fake_scan(progress_callback=None, **kwargs):
    return [dict(s) for s in FAKE_SERVERS]


scanner.quick_validate_cache = fake_validate
scanner.scan_network = fake_scan

from ui.app import ModelChatApp  # noqa: E402
from ui.home import HomeScreen  # noqa: E402
from ui.discovery import DiscoveryScreen  # noqa: E402
from ui.chat import ChatScreen  # noqa: E402
from ui.stress import StressScreen  # noqa: E402
from ui.prompt_arena import PromptArenaScreen  # noqa: E402
from ui.model_arena import ModelArenaScreen  # noqa: E402
from ui.history import HistoryScreen  # noqa: E402
from ui.settings import SettingsScreen  # noqa: E402
from ui.publish_screen import PublishScreen  # noqa: E402

# --------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------- #

_ORIGINAL_STRESS_TESTER = stress_mod.StressTester
_ORIGINAL_PA_CLIENT = prompt_arena_mod.ModelClient
_ORIGINAL_MA_CLIENT = model_arena_mod.ModelClient


class FakeStressTester(stress_tester.StressTester):
    """Real StressTester with instant run methods (no network)."""

    async def run_throughput_test(self, n, update_callback=None):
        return self._done(n)

    async def run_token_stress_test(self, sizes, update_callback=None):
        return self._done(len(sizes))

    async def run_sustained_load_test(self, minutes, rpm, update_callback=None):
        return self._done(minutes)

    async def run_consistency_test(self, prompt, iterations, update_callback=None):
        return self._done(iterations)

    async def run_realistic_user_test(self, minutes, rpm):
        return self._done(minutes)

    async def run_tool_bench_test(self, subset, concurrency, update_callback=None):
        return self._done(concurrency)

    def _done(self, n):
        self.stats = stress_tester.TestStats(
            total=n, completed=n, success=n, failed=0,
            avg_decode_tps=25.0, avg_ttft=0.2,
            wall_clock_time=1.5, total_output_tokens=n * 64)
        self.results = [
            stress_tester.TestResult(request_id=1, status="success",
                                     prompt="p", response="ok",
                                     completion_tokens=64, decode_tps=25.0),
        ]
        return self.stats


class FakeChatClient:
    """Canned ModelClient for the prompt arena: real tournament logic runs."""

    def __init__(self, server, model):
        self.last_metrics = SimpleNamespace(completion_tokens=10, ttft=0.1,
                                            eval_duration_ns=0)

    async def chat(self, question, messages):
        sys_content = messages[0]["content"] if messages else ""
        if "judge" in sys_content.lower():
            return ('{"winner": "TIE", "score_a": 5, "score_b": 5, '
                    '"confidence": "medium", "explanation": "both acceptable"}')
        return f"A clear answer to: {question[:60]}"


class FakeArenaClient:
    """Canned ModelClient for the model arena."""

    def __init__(self, server, model):
        self.last_metrics = SimpleNamespace(completion_tokens=42, ttft=0.2,
                                            eval_duration_ns=2_000_000_000)

    async def chat_stream(self, prompt, history, enable_thinking=False):
        yield "fake response text"
        yield " chunk two"

    async def chat(self, prompt, messages):
        return ('{"winner": "A", "scores": {"A": 9, "B": 8}, '
                '"explanation": "clearer and more complete"}')


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #


@asynccontextmanager
async def boot():
    app = ModelChatApp()
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause(0.6)
        yield app, pilot


async def goto_menu(pilot, index: int):
    await pilot.click("#home-menu")
    await pilot.press("home")
    for _ in range(index):
        await pilot.press("down")
    await pilot.press("enter")
    await pilot.pause(0.4)


async def pick_model(pilot):
    models = pilot.app.screen.query_one("#models")
    models.focus()
    models.highlighted = 0
    await pilot.pause(0.1)
    await pilot.press("enter")
    await pilot.pause(0.4)


def hex_of(color):
    return getattr(color, "hex", None) or str(color)


# --------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------- #


def test_theme_palette():
    asyncio.run(_test_theme_palette())


async def _test_theme_palette():
    async with boot() as (app, pilot):
        assert app.theme == "modelchat"
        theme = app.get_theme(app.theme)
        assert theme is not None
        gen = theme.to_color_system().generate()
        assert hex_of(gen["background"]).lower() == "#0b1220", hex_of(gen["background"])
        assert hex_of(gen["surface"]).lower() == "#121b2a", hex_of(gen["surface"])
        assert hex_of(gen["text-muted"]).lower() == "#8b9ab0", hex_of(gen["text-muted"])


def test_settings():
    asyncio.run(_test_settings())


async def _test_settings():
    async with boot() as (app, pilot):
        assert isinstance(app.screen, HomeScreen), type(app.screen)
        menu = app.screen.query_one("#home-menu")
        assert menu.option_count == 9, f"home menu {menu.option_count} != 9"
        await pilot.press("s")
        await pilot.pause(0.4)
        assert isinstance(app.screen, SettingsScreen), type(app.screen)
        theme_sel = app.screen.query_one("#set-theme")
        assert len(theme_sel._options) >= 3, theme_sel._options
        assert theme_sel.value == "modelchat"
        await pilot.press("escape")
        await pilot.pause(0.3)
        assert isinstance(app.screen, HomeScreen)


def test_discovery_and_chat():
    asyncio.run(_test_discovery_and_chat())


async def _test_discovery_and_chat():
    async with boot() as (app, pilot):
        await goto_menu(pilot, 1)
        assert isinstance(app.screen, DiscoveryScreen), type(app.screen)
        models = app.screen.query_one("#models")
        assert models.option_count == 2, f"discovery {models.option_count} != 2"
        await pick_model(pilot)
        assert isinstance(app.screen, ChatScreen), type(app.screen)
        assert app.model == "fake-model-a", app.model
        # send a message -> worker hits dead fake server -> error path
        await pilot.click("#chat-input")
        for ch in "hello":
            await pilot.press(ch)
        await pilot.press("enter")
        await pilot.pause(2.5)
        am = app.screen.query_one("#chat-messages").children[-1]
        stats = am.query_one("#am-stats")
        assert stats.styles.display == "block", "no stats line after stream error"
        await pilot.press("ctrl+l")
        await pilot.pause(0.3)
        assert isinstance(app.screen, HomeScreen)


def test_stress_run_summary():
    asyncio.run(_test_stress_run_summary())


async def _test_stress_run_summary():
    async with boot() as (app, pilot):
        app.model = "fake-model-a"
        app.server = dict(FAKE_SERVERS[0])
        stress_mod.StressTester = FakeStressTester
        try:
            await goto_menu(pilot, 2)
            assert isinstance(app.screen, StressScreen), type(app.screen)
            modes = app.screen.query_one("#stress-modes")
            assert modes.option_count == 6, f"stress modes {modes.option_count} != 6"
            cfg = app.screen.query_one("#stress-config")
            assert cfg.styles.display == "block", "config hidden without Enter"
            assert len(cfg.children) == 1, f"preselect config: {len(cfg.children)}"
            await pilot.press("r")
            await pilot.pause(1.5)
            summary = app.screen.query_one("#stress-summary")
            assert summary.styles.display == "block", "stress summary not shown"
        finally:
            stress_mod.StressTester = _ORIGINAL_STRESS_TESTER


def test_prompt_arena_run_summary():
    asyncio.run(_test_prompt_arena_run_summary())


async def _test_prompt_arena_run_summary():
    async with boot() as (app, pilot):
        app.model = "fake-model-a"
        app.server = dict(FAKE_SERVERS[0])
        prompt_arena_mod.ModelClient = FakeChatClient
        try:
            await goto_menu(pilot, 3)
            assert isinstance(app.screen, PromptArenaScreen), type(app.screen)
            prompts = app.screen.query_one("#pa-prompts")
            assert prompts.option_count == 7, f"prompts {prompts.option_count} != 7"
            app.screen.query_one("#pa-question").value = "What is the capital of France?"
            await pilot.press("r")
            await pilot.pause(2.5)
            summary = app.screen.query_one("#pa-summary")
            assert summary.styles.display == "block", "pa summary not shown"
        finally:
            prompt_arena_mod.ModelClient = _ORIGINAL_PA_CLIENT


def test_model_arena_run_summary():
    asyncio.run(_test_model_arena_run_summary())


async def _test_model_arena_run_summary():
    async with boot() as (app, pilot):
        app.model = "fake-model-a"
        app.server = dict(FAKE_SERVERS[0])
        app.servers = [dict(FAKE_SERVERS[0])]
        model_arena_mod.ModelClient = FakeArenaClient
        try:
            await goto_menu(pilot, 4)
            assert isinstance(app.screen, ModelArenaScreen), type(app.screen)
            mlist = app.screen.query_one("#ma-models")
            assert mlist.option_count == 2, f"model arena {mlist.option_count} != 2"
            mlist.focus()
            for i in (0, 1):
                mlist.highlighted = i
                await pilot.press("enter")
                await pilot.pause(0.05)
            assert len(app.screen._selected) == 2, app.screen._selected
            # switch to tournament mode (auto-judge, no vote modal)
            modes = app.screen.query_one("#ma-modes")
            modes.focus()
            modes.highlighted = 2
            await pilot.press("enter")
            await pilot.pause(0.1)
            await pilot.press("r")
            await pilot.pause(0.4)  # panes mounted, still streaming
            panes = app.screen.query_one("#ma-panes")
            assert len(panes.children) == 2, len(panes.children)
            for child in panes.children:
                assert child.size.width > 10, f"pane too narrow: {child.size.width}"
            await pilot.pause(3.0)
            summary = app.screen.query_one("#ma-summary")
            assert summary.styles.display == "block", "ma summary not shown"
        finally:
            model_arena_mod.ModelClient = _ORIGINAL_MA_CLIENT


def test_history_delete_count():
    asyncio.run(_test_history_delete_count())


async def _test_history_delete_count():
    async with boot() as (app, pilot):
        app.store.save_chat(
            model="fake-model-a", server="127.0.0.1:9999", system_prompt=None,
            messages=[{"role": "user", "content": "hello"}])
        app._results_count = 1
        await goto_menu(pilot, 5)
        assert isinstance(app.screen, HistoryScreen), type(app.screen)
        await pilot.pause(0.5)
        lst = app.screen.query_one("#hist-list")
        assert lst.option_count == 1, f"history options: {lst.option_count}"
        lst.focus()
        lst.highlighted = 0
        await pilot.press("d")
        await pilot.pause(0.4)
        assert app.results_count == 0, f"count after delete: {app.results_count}"
        opt = lst.get_option_at_index(0)
        assert opt.id == "__empty", opt.id


def test_publish_render():
    asyncio.run(_test_publish_render())


async def _test_publish_render():
    async with boot() as (app, pilot):
        await pilot.press("g")
        await pilot.pause(0.6)
        assert isinstance(app.screen, PublishScreen), type(app.screen)
        pl = app.screen.query_one("#pub-list")
        assert pl.option_count == 1, f"pub options: {pl.option_count}"
        opt = pl.get_option_at_index(0)
        assert opt.id == "__empty", opt.id


ALL_TESTS = [
    test_theme_palette, test_settings, test_discovery_and_chat,
    test_stress_run_summary, test_prompt_arena_run_summary,
    test_model_arena_run_summary, test_history_delete_count,
    test_publish_render,
]

if __name__ == "__main__":
    for t in ALL_TESTS:
        t()
        print(f"  ✓ {t.__name__}")
    print("PILOT OK — all checks passed")