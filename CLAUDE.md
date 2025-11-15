# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Model Chat CLI is a Textual-based terminal interface for discovering and chatting with local AI models on your network. It auto-discovers servers running Ollama, LM Studio, or OpenAI-compatible APIs.

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
# - textual>=0.47.0 (TUI framework)
# - httpx>=0.26.0 (async HTTP client)
# - asyncio-throttle>=1.0.0 (rate limiting)
```

## Architecture

### Application Flow

1. **Entry Point** (`main.py`): Initializes `ModelChatApp` (Textual app) and pushes `DiscoveryScreen`
2. **Discovery Phase** (`DiscoveryScreen` in `ui/screens.py`):
   - Scans local network using `scanner.py`
   - Displays discovered models as `ModelCard` widgets
   - User selects model and transitions to `ChatScreen`
3. **Chat Phase** (`ChatScreen` in `ui/screens.py`):
   - Uses `ModelClient` from `client.py` to communicate with selected model
   - Streams responses in real-time
   - Maintains conversation history

### Core Components

**scanner.py** - Network Discovery Logic
- `scan_network()`: Concurrently probes all local IPs (subnet.1-255) on common ports (11434, 1234, 5000, 8000, 8080)
- `probe_server()`: Checks both OpenAI-compatible (`/v1/models`) and Ollama (`/api/tags`) endpoints
- `check_server_health()`: Validates server responsiveness and measures latency
- Returns server dict with: `{ip, port, url, type, models, status, response_time}`

**client.py** - Model Communication
- `ModelClient`: Abstraction over OpenAI-compatible and Ollama APIs
- `chat_stream()`: Dispatches to `_chat_stream_openai()` or `_chat_stream_ollama()` based on server type
- Both implementations parse SSE (Server-Sent Events) streams:
  - OpenAI: `data: {json}` format, terminates on `data: [DONE]`
  - Ollama: NDJSON format (newline-delimited JSON)

**ui/screens.py** - Textual Screens
- `DiscoveryScreen`: Manages scan lifecycle, displays `ModelCard` widgets in `ScrollableContainer`
- `ChatScreen`: Manages chat UI with `ChatMessage` bubbles, handles input/streaming
- Both use Textual bindings for keyboard shortcuts (Q/R in discovery, Escape/Ctrl+C in chat)

**ui/widgets.py** - Custom Widgets
- `ModelCard`: Displays model info with health status (✓/✗/•) and response time
- `ChatMessage`: Reactive widget for streaming message updates via `update_content()`
- `ScanProgress`: Shows scan progress with reactive `current`/`total` properties

**ui/theme.tcss** - Textual CSS
- Modern dark theme with blue/slate palette (#0f172a background, #60a5fa accents)
- Different styles for user messages (blue, right-aligned) vs assistant (green, left-aligned)
- Focus states for interactive elements (model cards, inputs)

### Key Patterns

**Async/Await Throughout**: All network operations use `async`/`await` with `httpx.AsyncClient`

**Textual Reactive Properties**: Widgets use `reactive()` for auto-refreshing UI (e.g., `ChatMessage.content`, `ScanProgress.current`)

**Screen Navigation**:
- `app.push_screen(ChatScreen)` - Navigate to chat
- `app.pop_screen()` - Return to discovery

**Server Type Detection**: Code branches on `server["type"]` being "openai" or "ollama" for API differences

**Message History Format**: Standard chat format: `[{"role": "user|assistant", "content": "..."}]`

## Testing Model Integration

When testing or debugging model communication:

1. **Check server detection**: Verify scanner finds your server in `DiscoveryScreen`
2. **Inspect server dict**: Contains `type`, `url`, `models` fields - ensure correct
3. **Test streaming**: `ChatMessage.update_content()` should be called incrementally for each chunk
4. **API endpoint differences**:
   - OpenAI: POST to `/v1/chat/completions` with `stream: true`
   - Ollama: POST to `/api/chat` with `stream: true`

## Common Modifications

**Adding new server types**:
1. Add port to `scanner.COMMON_PORTS`
2. Add detection endpoint in `scanner.probe_server()`
3. Implement `client.ModelClient._chat_stream_<type>()`

**UI customization**: Edit `ui/theme.tcss` for colors/layout - uses Textual CSS syntax

**Scan optimization**: Modify `scanner.COMMON_PORTS` or subnet range in `scan_network()` to reduce scope
