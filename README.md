# Model Chat CLI

A terminal tool for discovering, chatting with, and benchmarking local AI models across your network. Auto-discovers Ollama, LM Studio, vLLM, and any OpenAI-compatible server, then lets you chat, stress test, or pit models against each other in a multi-model arena.

## Features

### Network Discovery
- Scans your local subnet for AI servers on common ports (11434, 1234, 5000, 8000, 8080)
- Detects Ollama (even with no models pulled), LM Studio, vLLM, and OpenAI-compatible APIs
- Caches discovered servers for instant reconnect
- Health checks with latency measurements

### Chat
- Streaming responses with tokens-per-second tracking
- System prompt support (set, edit, clear)
- Thinking/reasoning mode toggle (for models like Qwen 3/3.5)
- Conversation export to markdown
- Natural terminal scrolling (no screen clearing)

### Arena (`/arena`)

Compare up to 6 models side by side with three modes:

**Quick Compare** -- Single prompt, all models respond in parallel, displayed in a live grid.

**Battle** -- Multi-round manual evaluation:
- Enter prompts each round, all models stream simultaneously
- Vote for the best response after each round
- Running scoreboard tracks wins across rounds
- Blind mode shuffles model identities to eliminate bias, revealed at the end

**Tournament** -- Automated evaluation with a judge model:
- Pick any available model as the judge
- Choose from 5 built-in test suites (Reasoning, Coding, Creative, Instruction Following, Analysis) or enter custom prompts
- Suite-specific judging criteria (e.g., coding is scored on correctness, code quality, edge cases -- not "creativity")
- Optional repeats per prompt for variance testing
- Full leaderboard with average TPS and TTFT

All modes support blind mode, system prompts, TTFT tracking, and markdown export.

### Prompt Arena (`/promptarena`)

Compare different system prompts on the same model to find the best one for a given task. Includes 7 built-in prompts (Basic, CoT, AoT, DeepCoT, Failure-First, Methodical, Concise) with round-robin judging.

### Stress Testing (`/stress`)

Three load testing modes:
- **Throughput** -- Concurrent requests (5-50 simultaneous)
- **Token Stress** -- Increasing prompt lengths (500-10,000 tokens)
- **Sustained Load** -- Endurance testing over time (1 min to 24 hrs)

Live dashboard with per-request status, error log, and summary statistics.

## Supported Servers

| Server | Port | Detection |
|--------|------|-----------|
| Ollama | 11434 | `/api/tags` + `/api/version` fallback |
| LM Studio | 1234 | `/v1/models` |
| vLLM | 8000 | `/v1/models` |
| Any OpenAI-compatible | 5000, 8080 | `/v1/models` |

## Install

```bash
git clone https://github.com/mtecnic/model-chat-cli.git
cd model-chat-cli
pip install -r requirements.txt
```

Dependencies: `rich`, `httpx`, `asyncio-throttle`, `prompt-toolkit`

## Usage

```bash
python main.py
# or
./run.sh
```

### Chat Commands

| Command | Description |
|---------|-------------|
| `/quit`, `/q` | Exit |
| `/switch` | Switch model |
| `/clear` | Clear conversation |
| `/export` | Export to markdown |
| `/system` | View/edit system prompt |
| `/think` | Toggle reasoning mode |
| `/arena` | Multi-model arena |
| `/promptarena` | Prompt comparison tournament |
| `/stress` | Stress testing |
| `/help` | Show commands |

### Keyboard Shortcuts

- `Ctrl+D` -- Back to model selection
- `Ctrl+C` -- Quit (with confirmation)

## Architecture

```
model-chat-cli/
├── main.py              # State machine (discovery -> chat -> arena/stress)
├── scanner.py           # Network discovery, caching, health checks
├── client.py            # Model API client (OpenAI + Ollama streaming)
├── prompt_arena.py      # System prompt comparison engine
├── stress_tester.py     # Load testing engine
├── logger.py            # Centralized logging
├── storage/
│   └── history.py       # Chat history persistence
└── ui/
    ├── theme.py         # Semantic color theme
    ├── components.py    # Shared renderables + token estimation
    ├── discovery.py     # Server scan + model selection
    ├── chat.py          # Streaming chat interface
    ├── multi_arena.py   # Multi-model arena (battle/tournament/blind)
    ├── arena.py         # Prompt comparison UI
    └── stress_test.py   # Stress test dashboard
```

## Requirements

- Python 3.10+
- Terminal with color support
