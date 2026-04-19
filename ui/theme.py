"""Theme configuration for the Model Chat CLI using Rich."""
from rich.theme import Theme

APP_THEME = Theme({
    # Accent — interactive elements, model names, titles
    "accent": "cyan",
    "accent.bold": "bold cyan",

    # Roles
    "role.user": "bold blue",
    "role.assistant": "bold green",

    # Metrics — numbers, performance data
    "metric": "bold yellow",
    "metric.label": "dim",

    # Status
    "status.ok": "green",
    "status.error": "bold red",
    "status.warn": "yellow",
    "status.info": "blue",
    "status.running": "cyan",

    # Chrome — structural/decorative elements
    "chrome": "dim",
    "chrome.header": "bold",
    "chrome.border": "dim",
    "chrome.muted": "dim italic",

    # Prompt styling
    "prompt": "bold cyan",

    # Model display
    "model.name": "bold cyan",
    "model.server": "dim",

    # Thinking mode
    "thinking": "dim italic",
})
