"""Reusable Rich components for the Model Chat CLI."""
from typing import Callable, Optional, Sequence

from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class CellGrid:
    """Width/height-aware grid of small cells for live dashboards.

    Sizes itself from the space Rich gives it at render time, shows only the
    newest items that fit, and never wraps, so a dashboard keeps a fixed shape
    at any terminal size. `render_cell(item)` returns the cell's Text (it is
    placed in a no-wrap Panel of height `cell_h`).
    """

    def __init__(self, items: Sequence, render_cell: Callable[[object], Text],
                 title: str = "Results", cell_min_w: int = 24, cell_h: int = 4,
                 empty_text: str = "Waiting for results..."):
        self.items = items
        self.render_cell = render_cell
        self.title = title
        self.cell_min_w = cell_min_w
        self.cell_h = cell_h
        self.empty_text = empty_text

    def _cell(self, item) -> Panel:
        text = self.render_cell(item)
        text.no_wrap = True
        text.overflow = "ellipsis"
        return Panel(text, height=self.cell_h, padding=(0, 1), border_style="chrome.border")

    def __rich_console__(self, console, options):
        width = options.max_width
        height = options.height or options.max_height or console.height
        cols = max(1, max(1, width - 2) // self.cell_min_w)
        rows = max(1, max(1, height - 2) // self.cell_h)
        shown = list(self.items)[-(cols * rows):]

        if not shown:
            body = Text(self.empty_text, style="chrome")
        else:
            body = Table.grid(expand=True)
            for _ in range(cols):
                body.add_column(ratio=1, no_wrap=True)
            for i in range(0, len(shown), cols):
                cells = [self._cell(r) for r in shown[i:i + cols]]
                cells += [Text("")] * (cols - len(cells))
                body.add_row(*cells)

        yield Panel(body, title=f"{self.title} ({len(shown)} of {len(self.items)})",
                    height=height, padding=(0, 0))


def _fmt_context(tokens) -> str:
    """Format a max context length compactly: 262144 -> 256K, 261888 -> 256K."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return ""
    if n >= 1024 * 1024:
        m = n / (1024 * 1024)
        return f"{m:.0f}M" if m >= 10 else f"{m:.1f}M"
    if n >= 1024:
        return f"{n / 1024:.0f}K"
    return str(n)


def _model_label(model) -> str:
    """Display name for a model entry (plain string or {name, max_context})."""
    if isinstance(model, dict):
        return model.get("name", "unknown")
    return model


def create_model_table(servers: list, width: Optional[int] = None) -> Table:
    """Create a table displaying discovered models.

    Args:
        servers: List of server dictionaries with models
        width: Terminal width; caps the Model column so the others keep their size

    Returns:
        Rich Table with all discovered models

    Every column except Model has a fixed width, so only the model name
    shrinks (with an ellipsis) on a narrow terminal. (Column `ratio`s would
    squeeze the small columns to nothing instead.)
    """
    fixed = {"#": 3, "Server": 21, "Ctx": 5, "Type": 7, "": 1, "Latency": 8}
    # 7 columns x 2 padding + 8 border characters
    chrome_width = sum(fixed.values()) + 7 * 2 + 8
    model_max = max(12, width - chrome_width) if width else None

    table = Table(title="[accent.bold]Discovered Models[/accent.bold]", show_header=True)
    table.add_column("#", style="chrome", width=3, justify="right", no_wrap=True)
    table.add_column("Model", style="model.name", no_wrap=True, overflow="ellipsis",
                     min_width=12, max_width=model_max)
    table.add_column("Server", style="model.server", no_wrap=True, justify="right", width=21)
    table.add_column("Ctx", style="chrome", justify="right", no_wrap=True, width=5)
    table.add_column("Type", style="chrome", justify="center", no_wrap=True, width=7)
    table.add_column("", justify="center", no_wrap=True, width=1)
    table.add_column("Latency", justify="right", no_wrap=True, width=8)

    idx = 1
    for server in servers:
        for model in server.get("models", []):
            status = server.get("status", "unknown")
            status_symbol = {
                "healthy": "[status.ok]\u2713[/status.ok]",
                "error": "[status.error]\u2717[/status.error]",
                "unknown": "[status.warn]\u2022[/status.warn]",
            }.get(status, "\u2022")

            response_time = server.get("response_time", 0)
            latency = f"{response_time:.0f}ms" if response_time else "-"

            server_addr = f"{server['ip']}:{server['port']}"
            server_type = server['type'].upper()

            if isinstance(model, dict) and model.get("max_context"):
                ctx = _fmt_context(model["max_context"])
            else:
                ctx = "\u2022"

            table.add_row(
                str(idx),
                _model_label(model),
                server_addr,
                ctx,
                server_type,
                status_symbol,
                latency
            )
            idx += 1

    return table


def format_stats_line(
    tokens: int,
    elapsed: float,
    tps: float,
    ttft: float | None = None,
    think_tokens: int | None = None,
) -> Text:
    """Render compact stats after a response:
    ↳ 234 tok · 42 think · 5.2s · 45.0 t/s · 320ms ttft
    """
    t = Text()
    t.append("  \u21b3 ", style="chrome")
    t.append(str(tokens), style="metric")
    t.append(" tok", style="metric.label")
    if think_tokens is not None:
        t.append(" \u00b7 ", style="chrome")
        t.append(str(think_tokens), style="metric")
        t.append(" think", style="metric.label")
    t.append(" \u00b7 ", style="chrome")
    t.append(f"{elapsed:.1f}s", style="metric")
    t.append(" \u00b7 ", style="chrome")
    t.append(f"{tps:.1f}", style="metric")
    t.append(" t/s", style="metric.label")
    if ttft is not None and ttft > 0:
        t.append(" \u00b7 ", style="chrome")
        if ttft < 1:
            t.append(f"{ttft * 1000:.0f}ms", style="metric")
        else:
            t.append(f"{ttft:.2f}s", style="metric")
        t.append(" ttft", style="metric.label")
    return t


def format_menu_item(number: str, label: str, description: str = "") -> str:
    """Render a numbered menu choice consistently."""
    line = f"  [accent]{number}.[/accent] {label}"
    if description:
        line += f"\n     [chrome.muted]{description}[/chrome.muted]"
    return line


def estimate_tokens(text: str) -> int:
    """Estimate token count in a single pass, unicode-aware.

    Handles CJK (each char ~ 1 token), emoji grapheme clusters (~ 2-3 tokens
    per visible emoji regardless of codepoint count), Latin words (~ 1.3 tokens),
    and skips zero-width / format characters.
    """
    import unicodedata

    if not text:
        return 0

    tokens = 0.0
    newlines = 0
    in_latin_word = False
    in_emoji_seq = False  # tracks ZWJ / modifier sequences as one unit

    def _flush_latin():
        nonlocal tokens, in_latin_word
        if in_latin_word:
            tokens += 1.3
            in_latin_word = False

    def _is_emoji_component(c, cp):
        """Check if codepoint is part of an emoji sequence."""
        cat = unicodedata.category(c)
        # Symbol-Other (most emoji), Modifier-Symbol, skin tone modifiers,
        # regional indicators, variation selectors, keycap combining
        if cat in ('So', 'Sk'):
            return True
        if 0x1F3FB <= cp <= 0x1F3FF:  # skin tone modifiers
            return True
        if 0x1F1E0 <= cp <= 0x1F1FF:  # regional indicator symbols (flags)
            return True
        if cp in (0xFE0E, 0xFE0F):  # variation selectors
            return True
        if 0xE0020 <= cp <= 0xE007F:  # tag characters (flag subdivision)
            return True
        if 0x20E3 == cp:  # combining enclosing keycap
            return True
        return False

    for c in text:
        cp = ord(c)

        # Zero-width joiners / format chars — extend current emoji sequence
        if cp in (0x200D, 0x200B, 0x200C, 0xFEFF, 0x00AD) or unicodedata.category(c) == 'Cf':
            # ZWJ keeps the emoji sequence alive; others are just skipped
            continue

        if _is_emoji_component(c, cp):
            _flush_latin()
            if not in_emoji_seq:
                in_emoji_seq = True
                # Will be counted when sequence ends
            continue

        # Non-emoji char: close any open emoji sequence
        if in_emoji_seq:
            tokens += 2.5  # one visible emoji cluster ~ 2-3 BPE tokens
            in_emoji_seq = False

        if c == '\n':
            newlines += 1
            _flush_latin()

        elif c.isspace():
            _flush_latin()

        elif '\u4E00' <= c <= '\u9FFF' or '\u3400' <= c <= '\u4DBF' or \
             '\uF900' <= c <= '\uFAFF' or '\U00020000' <= c <= '\U0002A6DF':
            _flush_latin()
            tokens += 1

        elif '\u3040' <= c <= '\u30FF' or '\u31F0' <= c <= '\u31FF':
            _flush_latin()
            tokens += 1

        elif '\uAC00' <= c <= '\uD7AF':
            _flush_latin()
            tokens += 1

        elif c.isalnum():
            in_latin_word = True

        else:
            _flush_latin()
            tokens += 0.5

    # Flush trailing state
    _flush_latin()
    if in_emoji_seq:
        tokens += 2.5

    return max(1, int(tokens + newlines))
