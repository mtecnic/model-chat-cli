"""Textual theme for the Model Chat CLI."""
from textual.theme import Theme


def modelchat_theme() -> Theme:
    """Deep-navy custom theme: layered surfaces and solid muted text."""
    return Theme(
        name="modelchat",
        primary="#22d3ee",
        secondary="#818cf8",
        accent="#38bdf8",
        success="#34d399",
        error="#f87171",
        warning="#fbbf24",
        dark=True,
        foreground="#dce3ec",
        background="#0b1220",
        surface="#121b2a",
        panel="#1a2637",
        variables={"text-muted": "#8b9ab0"},
    )