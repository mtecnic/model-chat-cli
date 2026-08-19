# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Model Chat CLI is a **Textual-based TUI** for discovering and working with local AI models on your network. It auto-discovers servers running Ollama, LM Studio, vLLM, or any OpenAI-compatible API, and provides: chat with live streaming + thinking-mode support, a stress lab (throughput / token / sustained / tool-bench tests), two arenas (system-prompt battle, model-vs-model blind judged), unified result history, settings, and publishing of results to a local git repo pushed to GitHub.

## Development Commands

```bash
# Run the app
python main.py          # or ./run.sh (activates venv)

# Syntax check all modules
python -m py_compile *.py ui/*.py storage/*.py

# Headless UI pilot (drives all screens + runtime test paths via Textual's run_test)
venv/bin/python tests/test_ui_pilot.py     # standalone runner
venv/bin/python -m pytest tests/ -v        # pytest mode (install: pip install -r requirements-dev.txt)

# Live network scan check
python -c "import asyncio, scanner; print(asyncio.run(scanner.scan_network()))"
```

Dependencies (`requirements.txt`): `textual` (TUI), `rich`, `httpx`, `asyncio-throttle`. Dev deps (`requirements-dev.txt`): `pytest`.

## Architecture

### Application Flow

1. **Entry** (`main.py`): launches `ModelChatApp` (`ui/app.py`)
2. **Home** (`ui/home.py`): 9-item `OptionList` menu — Chat, Models & Servers, Stress Lab, Prompt Arena, Model Arena, History, Publish, Settings, Quit. `m` returns to Home from any screen.
3. **Discovery** (`ui/discovery.py`): scans network (or re-validates cached servers), lists models across servers; selection sets `app.model`
4. **Chat** (`ui/chat.py`): streaming assistant messages with thinking split (`think_parser.py`), TPS/TTFT metrics, system-prompt modal, per-message interrupt
5. **Stress** (`ui/stress.py`): 6 test modes with dynamic per-mode config `Select`s
6. **Prompt Arena** (`ui/prompt_arena.py`): 7 built-in system prompts battle on the same question; question banks include the tool-bench suite (`tool_bench.py`)
7. **Model Arena** (`ui/model_arena.py`): N models answer one prompt, blind pairwise judging by a judge model
8. **History** (`ui/history.py`): browses all saved results (search / open / delete), marks published
9. **Settings** (`ui/settings.py`): theme, scan, chat, output, and GitHub repo config
10. **Publish** (`ui/publish_screen.py`): pushes result md+json to a local git repo and commits + pushes

### Core Components

**ui/app.py** — `ModelChatApp`
- Registers the `modelchat` theme (`ui/theme.py`); loads `ConfigManager` (`config.py`) and `ResultStore` (`storage/exports.py`) at startup
- `app.servers`, `app.model`: current discovery state; screens read/write these
- On mount: runs `_initial_load` (validation/scan) and `_migrate_legacy` (one-time migration of old `chat__*.md` / `arena_*.md` files from CWD into `~/.model_chat/exports`)
- Screen names: `home`, `discovery`, `chat`, `stress`, `prompt_arena`, `model_arena`, `history`, `settings`, `publish`

**ui/base.py** — `BaseScreen`: all screens inherit; `compose()` yields `build_content()` + a shared `StatusBar` (`update()` renders model + hints). `StatusBar` is a `Static` subclass. Shared design tokens: `MUTED = "#8b9ab0"` (solid muted tone for informational rich text — replaces the 50%-alpha `dim` style) and `make_header(title, subtitle)` for the standard screen header (bold bright-cyan title + muted subtitle, with a `$panel` divider line in each screen's CSS).

**Engines (UI-agnostic, do not restructure)**:
- `scanner.py` — two-phase network discovery (TCP pre-scan → HTTP probe); `scan_network(progress_callback, extra_ips, concurrency, tcp_timeout, ports)`, `tcp_probe()`, `probe_server()`, `check_server_health()`, `quick_validate_cache()`, `save_cache()`/`load_cache()` (`~/.model_chat_cache.json`). Server dict: `{ip, port, url, type, models, status, response_time}`
- `client.py` — `ModelClient` over OpenAI-compatible + Ollama APIs; `chat_stream()` (SSE / NDJSON), `chat()`, `enable_thinking` passthrough; also `estimate_tokens()`
- `think_parser.py` — incremental thinking/content splitter (chunks → `parsed_chat_stream` → `split_thinking`)
- `stress_tester.py` — `StressTester`: throughput / token / sustained / tool-bench modes; `TestResult`/`TestStats` dataclasses; logs via `logger.py`
- `prompt_arena.py` — `PromptArena` round-robin judge tournaments; `SYSTEM_PROMPTS` (7 built-ins); `TEST_QUESTIONS`
- `tool_bench.py` — tool-call benchmark suite used by prompt arena + stress
- `publish.py` — `Publisher`: local git repo, branch/prefix config, `list_published()`, commit + push (no tokens stored)

**storage/exports.py** — `ResultStore`
- Unified results under `~/.model_chat/exports` (or `config.exports_root`): subdirs `chats/`, `stress/`, `prompt_arena/`, `model_arena/`
- Every result is a **markdown + JSON pair** (`.md` human report + `.json` raw data); `_write_pair()` is the only writer
- `save_chat()`, `save_stress()`, `save_prompt_arena()`, `save_model_arena()`; `list_results()` → `StoredResult` (path, title, kind, created); `migrate_legacy(cwd)` — one-time move of old export files
- `storage/reports.py` — markdown renderers + the `TEST_QUESTIONS` data contracts

**config.py** — `ConfigManager` at `~/.model_chat/config.json`; sections: `scan`, `chat`, `output`, `github` (`repo_path`, `branch`, `prefix`); `exports_root` points at the ResultStore dir

**logger.py** — dual console (INFO) + file (DEBUG) logging to `logs/*.log`; `log_request_error()`, `log_vllm_error()`, `log_test_summary()`

### Key Patterns

**Screen navigation**: `app.push_screen("name")` / `pop_screen()`; `m` → home. Each screen is a `BaseScreen` subclass with `build_content()` and its own CSS string.

**Async workers**: `self.run_worker(self._coroutine, name=..., exclusive=True)` runs **async** workers. Passing a plain sync function raises `WorkerError` — use `run_worker(fn, thread=True)` for sync work (only `_migrate_legacy` does this).

**Unified storage**: anything the user can re-open later is saved via `ResultStore` as md+json. Chat export is done through the store (`save_chat`); stress/arena results are saved when they finish. History reads from the store directly.

**Publishing**: `github.repo_path` is a **local git repo** (user pushes from inside it or it is already a clone); `Publisher` adds `results/<prefix>/...`, commits, and `git push`s. Nothing stores credentials.

**Server type detection**: branch on `server["type"]` (`"openai"` / `"ollama"`) for endpoint differences in `client.py`.

**Message history format**: `[{"role": "user|assistant|system", "content": "..."}]`

**Thinking mode**: chat has a `ctrl+t` toggle; when on, `enable_thinking` is passed to the client (OpenAI via `chat_template_kwargs`, Ollama via `options`); `think_parser` splits the stream into thinking/content for the bubble.

### Textual 6.4 API pitfalls (learned the hard way — check before adding UI)

1. **OptionList**: `add_option(Option(text, id="..."))` from `textual.widgets.option_list` — no `label`/`id` kwargs on `add_option`. To highlight: set `option_list.highlighted = <index>`, not `highlight_option()`.
2. **Select**: `Select([(label, value), ...], prompt="label", value=..., id=...)` — options are label/value tuples, and the label kwarg is `prompt`, not `label`.
3. **Text/Rich renderables cannot use theme colors** (`$accent`, `accent`, etc.) — `visualize()` runs them through a plain `rich.console.Console`. Use literal Rich styles: `bright_cyan`, `yellow`, `green`, `bright_red`, `dim`.
4. **Widget child management**: `remove_child`/`insert_child` do **not exist**. Use `await child.remove()` and `await self.mount(new_widget, before=sibling)`; `on_mount` must be `async` to await these.
5. **No `scroll_to_end()`** — use `scroll_end(animate=False, force=True)`.
6. **CSS**: `margin` accepts integer cells only (no `auto`, no `%`). Center with `align: center middle`; hide a widget with `display: none` (no `.show()`/`.hide()`).
7. **Theme colors in CSS** (`$primary`, `$accent`, `$text-muted`, …) are fine in CSS (resolved from the registered theme) but will error in standalone/`run_test` parses if the theme isn't registered — `ModelChatApp` registers it, so in-app it's fine.
8. **Static** has `update()`; a bare `Widget` does not — status/label widgets should subclass `Static`.
9. **Border titles**: `border-title: "x"` is **not** a CSS property — set `widget.border_title = "x"` in Python. CSS does support `border-title-color`, `border-title-style`, `border-title-align`.
10. **Border types**: `thin` is not a valid border type — use `solid`, `round`, `panel`, `dashed`, etc.
11. **Mounting children**: do not call `child.mount(...)` on a widget that isn't attached yet (`MountError`). Pass children to the constructor instead: `V(Static(...), Static(...), id="x")`.

### Chat keys

- `escape` interrupt (stop streaming), `ctrl+t` thinking toggle, `ctrl+l` back to menu
- System prompt via the modal (`sys-area` TextArea), applied to new turns

## Testing Model Integration

1. **Check server detection**: `probe_server()` should classify the API type correctly
2. **Inspect server dict**: `{type, url, models}` drive API selection
3. **Stream test**: send a message from Chat; the assistant bubble renders via `think_parser` + `Markdown`, with a stats line (tokens/TPS/TTFT)
4. **API endpoints**:
   - OpenAI: POST `/v1/chat/completions` (`stream: true`)
   - Ollama: POST `/api/chat` (`stream: true`)

## Common Modifications

**Adding a new server type**:
1. Add the port to `scanner.COMMON_PORTS`
2. Add detection in `scanner.check_endpoint()` / `probe_server()`
3. Implement `client.ModelClient._chat_stream_<type>()`

**Adding a new screen**:
1. Create `ui/<name>.py` with a `BaseScreen` subclass (`build_content()` + `CSS`)
2. Register its name in `ModelChatApp`'s screen registry in `ui/app.py`
3. Add a `MENU` entry in `ui/home.py`

**Adding a stress mode**: extend `StressTester` in `stress_tester.py` + the mode table / config `Select` builder in `ui/stress.py`

**UI customization**: theme in `ui/theme.py` (Textual `Theme`), per-screen CSS strings

## Prompt Arena & Model Arena

- **Prompt Arena** (`/` home menu → Prompt Arena): N system prompts (7 built-ins, toggleable) answer one question; pairwise judging by the same model; leaderboard of wins. Question banks: reasoning / tool-bench (`tool_bench.py`).
- **Model Arena**: pick ≥2 models + one prompt; responses are judged blindly in pairs; winner per question, aggregate stats.

Both persist full results (md + json) to `ResultStore` so History and Publish cover them.
