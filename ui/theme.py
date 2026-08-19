"""Textual theme for the Model Chat CLI."""
from textual.theme import Theme


def modelchat_theme() -> Theme:
    """Flat dark theme with classic cyan/blue/green/yellow accents."""
    return Theme(
        name="modelchat",
        primary="#00a8cc",
        secondary="#4a9eff",
        accent="#00a8cc",
        success="#2fae6b",
        error="#e5484d",
        warning="#d9a441",
        dark=True,
        foreground="#d0d0d0",
        background="#121212",
        surface="#1a1a1a",
        panel="#242424",
    )