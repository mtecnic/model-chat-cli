"""Theme configuration for the Model Chat CLI using Rich."""
from rich.theme import Theme
from pathlib import Path
import json

from config import config as app_config

# ============================================================================
# THEME PRESETS
# ============================================================================

THEMES = {
    "default": Theme({
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
    }),

    "dracula": Theme({
        # Dracula theme colors
        "model.name": "bold #ff79c6",  # Pink
        "model.server": "dim #8be9fd",  # Cyan
        "model.healthy": "#50fa7b",  # Green
        "model.error": "#ff5555",  # Red
        "model.unknown": "#f1fa8c",  # Yellow

        "chat.user": "bold #bd93f9",  # Purple
        "chat.assistant": "bold #50fa7b",  # Green
        "chat.system": "dim #f1fa8c",

        "header": "bold #f8f8f2 on #6272a4",
        "footer": "dim #6272a4",
        "title": "bold #ff79c6",
        "prompt": "bold #8be9fd",

        "error": "bold #ff5555",
        "success": "bold #50fa7b",
        "warning": "bold #ffb86c",  # Orange
        "info": "bold #8be9fd",
        "dim": "dim #6272a4",
    }),

    "nord": Theme({
        # Nord theme colors
        "model.name": "bold #88c0d0",  # Frost cyan
        "model.server": "dim #5e81ac",  # Frost blue
        "model.healthy": "#a3be8c",  # Aurora green
        "model.error": "#bf616a",  # Aurora red
        "model.unknown": "#ebcb8b",  # Aurora yellow

        "chat.user": "bold #81a1c1",  # Frost blue
        "chat.assistant": "bold #a3be8c",  # Aurora green
        "chat.system": "dim #d08770",  # Aurora orange

        "header": "bold #eceff4 on #4c566a",
        "footer": "dim #4c566a",
        "title": "bold #88c0d0",
        "prompt": "bold #5e81ac",

        "error": "bold #bf616a",
        "success": "bold #a3be8c",
        "warning": "bold #ebcb8b",
        "info": "bold #81a1c1",
        "dim": "dim #4c566a",
    }),

    "monokai": Theme({
        # Monokai theme colors
        "model.name": "bold #f92672",  # Pink
        "model.server": "dim #66d9ef",  # Cyan
        "model.healthy": "#a6e22e",  # Green
        "model.error": "#f92672",  # Pink/Red
        "model.unknown": "#e6db74",  # Yellow

        "chat.user": "bold #ae81ff",  # Purple
        "chat.assistant": "bold #a6e22e",  # Green
        "chat.system": "dim #fd971f",  # Orange

        "header": "bold #f8f8f2 on #75715e",
        "footer": "dim #75715e",
        "title": "bold #f92672",
        "prompt": "bold #66d9ef",

        "error": "bold #f92672",
        "success": "bold #a6e22e",
        "warning": "bold #fd971f",
        "info": "bold #66d9ef",
        "dim": "dim #75715e",
    }),

    "solarized": Theme({
        # Solarized Dark theme
        "model.name": "bold #2aa198",  # Cyan
        "model.server": "dim #268bd2",  # Blue
        "model.healthy": "#859900",  # Green
        "model.error": "#dc322f",  # Red
        "model.unknown": "#b58900",  # Yellow

        "chat.user": "bold #268bd2",  # Blue
        "chat.assistant": "bold #859900",  # Green
        "chat.system": "dim #cb4b16",  # Orange

        "header": "bold #fdf6e3 on #586e75",
        "footer": "dim #586e75",
        "title": "bold #2aa198",
        "prompt": "bold #268bd2",

        "error": "bold #dc322f",
        "success": "bold #859900",
        "warning": "bold #b58900",
        "info": "bold #2aa198",
        "dim": "dim #586e75",
    }),

    "cyberpunk": Theme({
        # Cyberpunk/Neon theme
        "model.name": "bold #ff00ff",  # Magenta
        "model.server": "dim #00ffff",  # Cyan
        "model.healthy": "#00ff00",  # Lime
        "model.error": "#ff0066",  # Hot pink
        "model.unknown": "#ffff00",  # Yellow

        "chat.user": "bold #ff00ff",  # Magenta
        "chat.assistant": "bold #00ff00",  # Lime
        "chat.system": "dim #ff9900",  # Orange

        "header": "bold #ffffff on #660066",
        "footer": "dim #660066",
        "title": "bold #00ffff",
        "prompt": "bold #ff00ff",

        "error": "bold #ff0066",
        "success": "bold #00ff00",
        "warning": "bold #ffff00",
        "info": "bold #00ffff",
        "dim": "dim #660066",
    }),
}

# Default theme
APP_THEME = THEMES["default"]


def load_theme() -> str:
    """Load saved theme preference.

    Returns:
        Theme name
    """
    try:
        if app_config.THEME_FILE.exists():
            with open(app_config.THEME_FILE, 'r') as f:
                theme_config = json.load(f)
                return theme_config.get("theme", "default")
    except Exception:
        pass
    return "default"


def save_theme(theme_name: str):
    """Save theme preference.

    Args:
        theme_name: Name of theme to save
    """
    try:
        with open(app_config.THEME_FILE, 'w') as f:
            json.dump({"theme": theme_name}, f)
    except Exception:
        pass


def get_theme(theme_name: str = None) -> Theme:
    """Get theme by name.

    Args:
        theme_name: Theme name, or None for saved preference

    Returns:
        Rich Theme object
    """
    if theme_name is None:
        theme_name = load_theme()

    return THEMES.get(theme_name, THEMES["default"])


def list_themes() -> list:
    """Get list of available theme names.

    Returns:
        List of theme names
    """
    return list(THEMES.keys())
