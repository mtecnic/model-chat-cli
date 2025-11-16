# Model Chat CLI

A modern terminal interface for discovering and chatting with local AI models.

## Features

- **Auto-Discovery**: Automatically scans your local network for AI model servers
- **Multi-Format Support**: Works with both OpenAI-compatible APIs, vllm and Ollama
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
