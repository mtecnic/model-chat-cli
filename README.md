# Model Chat CLI

A terminal tool for discovering, chatting with, and benchmarking local AI models across your network. Auto-discovers Ollama, LM Studio, vLLM, and any OpenAI-compatible server, then lets you chat, stress test, or pit models against each other in a multi-model arena.

## Features

### Network Discovery
- Scans your local subnet for AI servers on common ports (11434, 1234, 5000, 8000, 8080)
- Detects Ollama (even with no models pulled), LM Studio, vLLM, and OpenAI-compatible APIs
- Caches discovered servers for instant reconnect
- Health checks with latency measurements

### Chat
- Streaming responses with **decode tokens-per-second** (timer starts on first token, so TTFT and queue time are excluded — true generation rate, not wall-clock)
- **TTFT (time-to-first-token)** displayed alongside throughput
- Thinking-token count shown when `/think` is enabled (e.g. `↳ 234 tok · 42 think · 5.2s · 45.0 t/s · 320ms ttft`)
- Italic styling for thinking content; normal styling for the actual response
- System prompt support (set, edit, clear)
- Thinking/reasoning mode toggle (for models like Qwen 3 / 3.5)
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

Six modes covering throughput, stability, realistic traffic patterns, and agentic tool-calling capability:

- **Throughput** -- Concurrent requests (5-50 simultaneous)
- **Token Stress** -- Increasing prompt lengths (500-10,000 tokens)
- **Sustained Load** -- Endurance testing over time (1 min to 24 hrs)
- **Consistency** -- Same prompt N times serially to isolate hardware-level noise (thermals, DVFS, drivers, kernel scheduling). Reports stddev + drift between first and second halves of the run.
- **Realistic User** -- Poisson-distributed session arrivals with multi-turn conversations. Each arrival becomes a session that runs K turns sequentially with growing context (history replay) and log-normal think time between turns. Three depth profiles: One-shot (pure population), Short (~3 turns), Long (~8 turns). Tests how the model behaves under believable aggregate load *and* deep individual sessions.
- **Tool Calling Benchmark** -- Agentic tool-calling tests with mock tools, full agent loop (model → tool calls → execution → tool results → repeat until final answer). See below.

Live dashboard with per-request status, error log, percentile latencies, variance, drift, and summary statistics.

#### Tool Calling Benchmark

Drives the model through a suite of agentic tasks requiring one or more tool calls. The harness implements a real agent loop (parallel tool-call support, conversation history, normalized OpenAI / Ollama tool format), executes mock tools deterministically, and feeds results back until the model produces a final answer or exhausts its iteration budget.

Six difficulty tiers:

| Tier | Tasks | What it tests |
|------|-------|---------------|
| **Quick**     | 7  | Smoke test -- single-tool baseline |
| **Full**      | 45 | Everything across all tiers |
| **Hard**      | 10 | Distractors, error recovery, multi-step planning, sequential dependencies, refusal calibration |
| **Brutal**    | 6  | Long-horizon orchestration, prompt-injection resistance, parallel-required arrival, arg-precision (dict-subset matching), unstated dependency chains |
| **Realistic** | 6  | Verbose JSON envelopes (extract values from noise), pagination (multi-page iteration with cursor tracking), transient failures with retry, strict ISO-639 args, 33-tool catalog with 15 noise distractors |
| **EXTREME**   | 8  | Multi-hop prompt injection (chained files), conflicting tool outputs (model must flag the disagreement), self-verification (compute twice via different decompositions), social-engineered exfiltration refusal, compositional dependency chains, arg-type precision (int vs string), refusal calibration on prompts that look tool-needing but aren't |

Mock tools include `calculator`, `get_weather`, `get_stock_price`, `read_file`, `list_files`, `db_query`, `translate`, `unit_convert`, `get_current_time`, `send_email`, plus distractor tools (`eval_math`, `weather_lookup`, `currency_convert`, etc.) that return errors hinting at the right tool. Realistic tier adds verbose JSON envelopes, pagination, `flaky_search` (rate-limited), `weather_secondary` (independent provider for cross-checks), and 15 deprecated/duplicate noise tools.

**Multidimensional scoring** -- each task is scored on:
- **Answer correctness**: numeric tolerance (commas/scientific normalized), word-boundary regex with synonym tuples
- **Tool use**: per-call argument validation with dict-subset matching (`filters: {country: "JP"}` constraint allows extra filter keys but requires the country filter)
- **Forbidden tools**: explicit per-task list, plus `expect_zero_tools` flag that auto-forbids all tools
- **Iteration / call budget**: `min_tool_calls` / `max_tool_calls` / `max_iterations`

**Tool name normalization** handles common namespace prefixes (`functions.calculator`, `default_api.db_query`, `tools::send_email`) so newer models don't fail on cosmetic format differences.

**Diagnostics** surface root cause for each failed task in a `Reason` column: missing required call, called forbidden tool, answer missing number/word, budget exceeded, malformed args, unknown tool name, empty response. A separate "Model Diagnostics" panel aggregates malformed-JSON / unknown-tool / empty-response counts so capability gaps can be distinguished from chat-template / serving issues.

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
├── stress_tester.py     # Load testing engine (throughput / sustained / consistency / realistic-user / tool-bench)
├── tool_bench.py        # Agentic tool-calling benchmark (mock tools, agent loop, scoring, 6 tiers)
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
