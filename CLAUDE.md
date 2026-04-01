# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Model Chat CLI is a Rich-based terminal interface for discovering and chatting with local AI models on your network. It auto-discovers servers running Ollama, LM Studio, or OpenAI-compatible APIs.

## Development Commands

### Running the Application

```bash
# Direct execution
python main.py

# Using the provided shell script (activates venv automatically)
./run.sh
```

### Environment Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Dependencies:
# - rich>=13.7.0 (TUI framework)
# - httpx>=0.26.0 (async HTTP client)
# - asyncio-throttle>=1.0.0 (rate limiting)
# - prompt-toolkit>=3.0.0 (input handling)
```

## Architecture

### Application Flow

1. **Entry Point** (`main.py`): `ModelChatCLI` class manages app state transitions (DISCOVERY → CHAT → STRESS_TEST → ARENA → QUIT)
2. **Discovery Phase** (`ui/discovery.py`): `DiscoveryView` scans network, validates cached servers, displays model table
3. **Chat Phase** (`ui/chat.py`): `ChatView` handles streaming responses with live updates, TPS tracking, system prompts
4. **Stress Test Phase** (`ui/stress_test.py`): `StressTestView` runs throughput/token/sustained load tests
5. **Arena Phase** (`ui/arena.py`): `ArenaView` runs prompt comparison tournaments between different system prompts

### Core Components

**scanner.py** - Network Discovery Logic
- `scan_network()`: Probes subnet IPs (x.x.x.1-255) on ports [11434, 1234, 5000, 8000, 8080] using semaphore-limited concurrency (100 max)
- `probe_server()`: Checks `/v1/models` (OpenAI) and `/api/tags` (Ollama) endpoints in parallel via `check_endpoint()`
- `check_server_health()`: Validates a server and measures latency in ms
- `quick_validate_cache()`: Re-validates cached servers, returns only healthy ones
- `save_cache()` / `load_cache()`: Persists discovered servers to `~/.model_chat_cache.json`
- Returns server dict: `{ip, port, url, type, models, status, response_time}`

**client.py** - Model Communication
- `ModelClient`: Abstraction over OpenAI-compatible and Ollama APIs
- `chat_stream()`: Dispatches to `_chat_stream_openai()` or `_chat_stream_ollama()` based on server type; accepts `enable_thinking` param for models that support reasoning chains (e.g. Qwen 3/3.5)
- `chat()`: Non-streaming wrapper that collects full response (used by stress tester and arena)
- SSE stream parsing: OpenAI uses `data: {json}` format; Ollama uses NDJSON
- Both use `aiter_bytes()` with 64-byte chunks for real-time streaming
- HTTP timeout: 60s via `httpx.AsyncClient`

**stress_tester.py** - Load Testing Engine
- `StressTester`: Three test modes - throughput (concurrent), token stress (varying lengths), sustained load (over time)
- Uses `TestResult` dataclass to track individual requests and `TestStats` for aggregates
- Logs to `logs/stress_test_*.log` via `logger.py`

**prompt_arena.py** - System Prompt Comparison Engine
- `PromptArena`: Tests multiple system prompts against each other on the same question
- `SYSTEM_PROMPTS`: 7 built-in prompts (Basic, CoT, AoT, DeepCoT, Failure-First, Methodical, Concise)
- Uses same model for generation AND judging (self-evaluation)
- Round-robin tournament: all prompts compete head-to-head
- Dataclasses: `PromptResponse`, `JudgeResult`, `ArenaMatchup`, `ArenaResult`, `ArenaStats`
- `TEST_QUESTIONS`: Categorized sample questions for multi-round battles

**ui/components.py** - Rich Renderables
- `create_model_table()`: Discovery screen model list
- `create_chat_message()`: Chat bubbles with markdown/syntax highlighting
- `render_markdown_with_code()`: Extracts code blocks for Syntax highlighting

**storage/history.py** - Chat History Persistence
- `ChatHistoryManager`: Save/load/search/delete conversations as JSON in `~/.model_chat_history/`
- Not currently wired into the UI (chat export in `ChatView` writes markdown directly)

**logger.py** - Centralized Logging
- `setup_logger()`: Configures dual console (INFO) + file (DEBUG) logging to `logs/stress_test_*.log`
- `log_request_error()`, `log_vllm_error()`, `log_test_summary()`: Structured error/summary helpers

**ui/theme.py** - Rich Theme
- Color scheme: cyan for models, blue for user messages, green for assistant, magenta for servers

### Key Patterns

**State Machine in main.py**: `AppState` enum controls view transitions; each view's `run()` returns control signal ("switch", "quit", "stress_test")

**Async/Await Throughout**: All network operations use `async`/`await` with `httpx.AsyncClient`

**Progress Callbacks**: Scanner and validators accept `async def progress_callback(current, total)` for Rich Progress updates

**Server Type Detection**: Code branches on `server["type"]` being "openai" or "ollama" for API endpoint differences

**Message History Format**: Standard chat format: `[{"role": "user|assistant|system", "content": "..."}]`

**Thinking Mode**: `ChatView` has a `thinking_enabled` toggle (default off). When enabled, passes `enable_thinking` to the client. OpenAI-compatible servers receive it via `chat_template_kwargs`; Ollama via `options`.

### Chat Commands

`/quit`, `/q`, `/switch`, `/clear`, `/export`, `/system`, `/think`, `/stress`, `/arena`, `/help`

## Testing Model Integration

When debugging model communication:

1. **Check server detection**: Verify `probe_server()` identifies correct API type
2. **Inspect server dict**: `{type, url, models}` fields drive API selection
3. **Test streaming**: `ChatView._stream_response()` uses Rich Live for incremental updates
4. **API endpoints**:
   - OpenAI: POST `/v1/chat/completions` with `stream: true`
   - Ollama: POST `/api/chat` with `stream: true`

## Common Modifications

**Adding new server types**:
1. Add port to `scanner.COMMON_PORTS`
2. Add detection endpoint in `scanner.check_endpoint()` / `probe_server()`
3. Implement `client.ModelClient._chat_stream_<type>()`

**UI customization**: Edit `ui/theme.py` for colors, `ui/components.py` for layouts

## Prompt Arena

The `/arena` command launches a prompt comparison tournament where different system prompts compete head-to-head.

### Arena Modes

1. **Single Question Tournament**: Enter one question, all 7 prompts generate responses, then the model judges each pair in round-robin style
2. **Multi-Round Battle**: Run multiple questions, aggregate statistics across rounds to find the best overall prompt
3. **Custom Prompt Setup**: Add or remove system prompts before competing

### Built-in System Prompts

| Key | Name | Description |
|-----|------|-------------|
| basic | Basic | Simple helpful assistant |
| cot | CoT | Chain-of-thought step-by-step reasoning |
| aot | AoT | Atom-of-thought decomposition into independent units |
| deep_cot | DeepCoT | Deep reasoning with error-checking and calibrated confidence |
| failure_first | Failure-First | Consider failure modes before solving |
| methodical | Methodical | Structured understand/reason/challenge/respond |
| concise | Concise | Maximum brevity without sacrificing accuracy |

### Tournament Flow

1. User enters a question (or selects sample questions)
2. All active prompts generate responses concurrently
3. Model judges each pair of responses (N*(N-1)/2 matchups for N prompts)
4. Wins are tallied, leaderboard is displayed with winner

### Key Files

- `prompt_arena.py`: Engine class, system prompts, judging logic
- `ui/arena.py`: Rich UI view, mode selection, live progress display
