"""Visual effects and animations for Model Chat CLI."""
import time
import asyncio
from typing import Optional, List
from rich.text import Text
from rich.panel import Panel
from rich.console import Console
from rich.align import Align


# ============================================================================
# SPARKLINE CHARTS
# ============================================================================

def create_sparkline(data: List[float], width: int = 20, height: int = 8) -> str:
    """Create an ASCII sparkline chart from data points.

    Args:
        data: List of numeric values
        width: Number of data points to show
        height: Character height (uses unicode block chars)

    Returns:
        Sparkline string using unicode block characters
    """
    if not data:
        return "▁" * width

    # Use last N points
    points = data[-width:] if len(data) > width else data

    # Pad if needed
    while len(points) < width:
        points = [0] + points

    # Normalize to 0-8 range (8 unicode block levels)
    min_val = min(points) if points else 0
    max_val = max(points) if points else 1
    range_val = max_val - min_val if max_val != min_val else 1

    # Unicode block characters (8 levels)
    blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

    sparkline = ""
    for val in points:
        normalized = (val - min_val) / range_val
        index = min(int(normalized * 7), 7)
        sparkline += blocks[index]

    return sparkline


# ============================================================================
# GRADIENT TEXT
# ============================================================================

def create_gradient_text(text: str, colors: List[str]) -> Text:
    """Create text with gradient color effect.

    Args:
        text: Text to colorize
        colors: List of color names (e.g., ['cyan', 'blue', 'magenta'])

    Returns:
        Rich Text object with gradient
    """
    if not text or len(colors) < 2:
        return Text(text)

    result = Text()
    text_len = len(text)
    color_count = len(colors)

    for i, char in enumerate(text):
        # Calculate which color to use based on position
        progress = i / max(text_len - 1, 1)
        color_index = int(progress * (color_count - 1))
        color = colors[color_index]
        result.append(char, style=color)

    return result


# ============================================================================
# ANIMATED BANNER
# ============================================================================

def create_banner(text: str = "MODEL CHAT CLI", font: str = "slant") -> Text:
    """Create an ASCII art banner with gradient.

    Args:
        text: Text to display
        font: PyFiglet font name

    Returns:
        Rich Text with gradient banner
    """
    try:
        import pyfiglet
        ascii_art = pyfiglet.figlet_format(text, font=font)
    except ImportError:
        # Fallback if pyfiglet not available
        ascii_art = f"\n  {text}  \n"

    # Apply gradient (cyan -> blue -> magenta)
    lines = ascii_art.split('\n')
    result = Text()

    colors = ['cyan', 'bright_cyan', 'blue', 'bright_blue', 'magenta']

    for i, line in enumerate(lines):
        color_idx = int((i / max(len(lines) - 1, 1)) * (len(colors) - 1))
        result.append(line + '\n', style=colors[color_idx])

    return result


async def animate_banner_fade(console: Console, banner: Text, duration: float = 1.0):
    """Animate banner with fade-in effect.

    Args:
        console: Rich Console instance
        banner: Banner text to animate
        duration: Animation duration in seconds
    """
    steps = 10
    step_duration = duration / steps

    for i in range(steps + 1):
        opacity = i / steps
        # Simulate opacity by adjusting brightness
        if opacity < 0.3:
            style = "dim"
        elif opacity < 0.7:
            style = "not dim"
        else:
            style = "bold"

        console.clear()
        console.print(banner, style=style, justify="center")
        await asyncio.sleep(step_duration)


# ============================================================================
# PARTICLE EFFECTS
# ============================================================================

def create_particle_burst(char: str = "✨", count: int = 20) -> str:
    """Create a particle burst effect.

    Args:
        char: Character to use for particles
        count: Number of particles

    Returns:
        String with scattered particles
    """
    particles = ['∘', '•', '·', '✨', '⋆', '✦', '★']
    import random

    result = " ".join(random.choice(particles) for _ in range(count))
    return result


def create_confetti() -> Text:
    """Create confetti effect for celebrations.

    Returns:
        Rich Text with colorful confetti
    """
    confetti_chars = ['🎊', '🎉', '✨', '⭐', '🌟', '💫', '🎆', '🎇']
    colors = ['red', 'yellow', 'green', 'blue', 'magenta', 'cyan']

    import random
    result = Text()

    for _ in range(30):
        char = random.choice(confetti_chars)
        color = random.choice(colors)
        result.append(char + " ", style=color)

    return result


# ============================================================================
# 3D DEPTH EFFECTS
# ============================================================================

def create_3d_panel(content, title: str = "", border_style: str = "cyan") -> Panel:
    """Create a panel with 3D depth effect using shadows.

    Args:
        content: Panel content
        title: Panel title
        border_style: Border color

    Returns:
        Panel with 3D effect
    """
    # Use double-line box characters for depth
    return Panel(
        content,
        title=title,
        border_style=border_style,
        box=DoubleShadowBox(),
        padding=(1, 2)
    )


class DoubleShadowBox:
    """Custom box type with shadow effect."""

    def __init__(self):
        # Double-line characters for main box
        self.top = "╔"
        self.top_divider = "╦"
        self.bottom = "╚"
        self.bottom_divider = "╩"
        self.mid_divider = "╬"
        self.top_left = "╔"
        self.top_right = "╗"
        self.bottom_left = "╚"
        self.bottom_right = "╝"
        self.mid_left = "╠"
        self.mid_right = "╣"
        self.vertical = "║"
        self.horizontal = "═"


# ============================================================================
# GLASSMORPHISM PANELS
# ============================================================================

def create_glass_panel(content, title: str = "", accent_color: str = "cyan") -> Panel:
    """Create a glassmorphism-style panel.

    Args:
        content: Panel content
        title: Panel title
        accent_color: Accent color for borders

    Returns:
        Panel with glassmorphism effect
    """
    # Use subtle box characters
    from rich.box import ROUNDED

    # Create frosted effect with dim background
    return Panel(
        content,
        title=f"[{accent_color}]{title}[/{accent_color}]" if title else "",
        border_style=f"dim {accent_color}",
        box=ROUNDED,
        style="on #1a1b26",  # Subtle dark background
        padding=(0, 1)
    )


# ============================================================================
# PROGRESS BAR EFFECTS
# ============================================================================

def create_rainbow_progress_bar(progress: float, width: int = 40) -> Text:
    """Create a rainbow gradient progress bar.

    Args:
        progress: Progress value (0.0 to 1.0)
        width: Bar width in characters

    Returns:
        Rich Text with rainbow progress bar
    """
    filled = int(progress * width)
    bar = Text()

    # Rainbow colors
    colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta']

    for i in range(width):
        if i < filled:
            color_idx = int((i / width) * (len(colors) - 1))
            bar.append("█", style=colors[color_idx])
        else:
            bar.append("░", style="dim")

    return bar


# ============================================================================
# TYPEWRITER EFFECT
# ============================================================================

async def typewriter_effect(console: Console, text: str, speed: float = 0.03):
    """Display text with typewriter effect.

    Args:
        console: Rich Console instance
        text: Text to display
        speed: Delay between characters in seconds
    """
    from rich.live import Live
    from rich.text import Text

    display_text = Text()

    with Live(display_text, console=console, refresh_per_second=30) as live:
        for char in text:
            display_text.append(char)
            live.update(display_text)
            await asyncio.sleep(speed)


# ============================================================================
# DYNAMIC STATUS BAR
# ============================================================================

def create_status_bar(
    server: str = "",
    tps: float = 0,
    messages: int = 0,
    health: str = "good"
) -> Text:
    """Create a dynamic status bar with icons.

    Args:
        server: Server address
        tps: Tokens per second
        messages: Message count
        health: Connection health (good/medium/poor)

    Returns:
        Rich Text status bar
    """
    bar = Text()

    # Connection health icon
    health_icons = {
        "good": ("🟢", "green"),
        "medium": ("🟡", "yellow"),
        "poor": ("🔴", "red")
    }
    icon, color = health_icons.get(health, ("⚪", "dim"))

    bar.append(icon + " ", style=color)

    if server:
        bar.append("🌐 ", style="dim")
        bar.append(server, style="cyan")
        bar.append(" │ ", style="dim")

    if tps > 0:
        bar.append("⚡ ", style="yellow")
        bar.append(f"{tps:.0f} tok/s", style="bold yellow")
        bar.append(" │ ", style="dim")

    if messages > 0:
        bar.append("💬 ", style="blue")
        bar.append(f"{messages} msgs", style="cyan")
        bar.append(" │ ", style="dim")

    # Current time
    import datetime
    bar.append("🕐 ", style="dim")
    bar.append(datetime.datetime.now().strftime("%H:%M"), style="dim")

    return bar


# ============================================================================
# SMOOTH TRANSITIONS
# ============================================================================

async def fade_transition(console: Console, from_content=None, to_content=None, duration: float = 0.5):
    """Create a fade transition between two contents.

    Args:
        console: Rich Console instance
        from_content: Content to fade from
        to_content: Content to fade to
        duration: Transition duration in seconds
    """
    steps = 5
    step_duration = duration / steps

    # Fade out
    for i in range(steps, -1, -1):
        console.clear()
        if from_content:
            opacity_style = "dim" if i < steps / 2 else "not dim"
            console.print(from_content, style=opacity_style)
        await asyncio.sleep(step_duration)

    # Fade in
    for i in range(steps + 1):
        console.clear()
        if to_content:
            opacity_style = "dim" if i < steps / 2 else "bold"
            console.print(to_content, style=opacity_style)
        await asyncio.sleep(step_duration)
