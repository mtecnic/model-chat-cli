"""Theme configuration for the Model Chat CLI using Rich."""
from rich.theme import Theme

# Color scheme based on the original design
APP_THEME = Theme({
    # Model-related styles
    "model.name": "bold cyan",
    "model.server": "dim magenta",
    "model.healthy": "green",
    "model.error": "red",
    "model.unknown": "yellow",

    # Chat message styles
    "chat.user": "bold blue",
    "chat.assistant": "bold green",
    "chat.system": "dim yellow",

    # UI elements
    "header": "bold white on blue",
    "footer": "dim white",
    "title": "bold cyan",
    "prompt": "bold cyan",

    # Status styles
    "error": "bold red",
    "success": "bold green",
    "warning": "bold yellow",
    "info": "bold blue",
    "dim": "dim white",
})
