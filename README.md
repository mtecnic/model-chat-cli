# Model Chat CLI

A modern terminal interface for discovering and chatting with local AI models.

## Features

- **Auto-Discovery**: Automatically scans your local network for AI model servers
- **Multi-Format Support**: Works with both OpenAI-compatible APIs and Ollama
- **Modern UI**: 24-bit color gradients and contemporary design
- **Streaming Responses**: Real-time streaming chat responses
- **Health Checks**: Shows server status and response times

## Supported Servers

- Ollama (default port 11434)
- LM Studio (default port 1234)
- Any OpenAI-compatible server on ports 5000, 8000, 8080

## Installation

```bash
cd model-chat-cli
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

### Controls

**Discovery Screen:**
- Arrow keys: Navigate models
- Enter: Select a model to chat
- R: Rescan network
- Q: Quit

**Chat Screen:**
- Type message and press Enter to send
- Escape or Ctrl+C: Return to discovery screen

## Requirements

- Python 3.8+
- Modern terminal with 24-bit color support

## Configuration

Model Chat CLI supports environment variable configuration. All settings have sensible defaults.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_CHAT_COMMON_PORTS` | `11434,1234,5000,8000,8080` | Comma-separated list of ports to scan |
| `MODEL_CHAT_SCAN_TIMEOUT` | `30.0` | Network scan timeout in seconds |
| `MODEL_CHAT_SCAN_SEMAPHORE` | `50` | Max concurrent network probes |
| `MODEL_CHAT_CACHE_FILE` | `~/.model_chat_cache.json` | Server cache file path |
| `MODEL_CHAT_THEME_FILE` | `~/.model_chat_theme.json` | Theme preference file |
| `MODEL_CHAT_FAVORITES_FILE` | `~/.model_chat_favorites.json` | Favorites file path |
| `MODEL_CHAT_MAX_HISTORY` | `1000` | Max messages in chat history |
| `MODEL_CHAT_MAX_DISPLAY` | `100` | Max messages to display |
| `MODEL_CHAT_MAX_RETRIES` | `3` | Network retry attempts |
| `MODEL_CHAT_RETRY_BASE_DELAY` | `1.0` | Base retry delay in seconds |
| `MODEL_CHAT_RETRY_MAX_DELAY` | `8.0` | Max retry delay in seconds |
| `MODEL_CHAT_ENABLE_BANNER` | `true` | Show startup banner |
| `MODEL_CHAT_BANNER_FONT` | `slant` | ASCII art banner font |
| `MODEL_CHAT_DEFAULT_THEME` | `default` | Default color theme |
| `MODEL_CHAT_LOG_LEVEL` | `INFO` | Logging level |

### Example

```bash
# Customize scan timeout and ports
export MODEL_CHAT_SCAN_TIMEOUT=60.0
export MODEL_CHAT_COMMON_PORTS="11434,1234,8000"
python main.py
```

## Architecture

```
model-chat-cli/
├── main.py          # Application entry point
├── scanner.py       # Network scanning logic
├── client.py        # Model API client
├── ui/
│   ├── screens.py   # Textual screens
│   ├── widgets.py   # Custom widgets
│   └── theme.tcss   # Modern CSS theme
└── requirements.txt
```
