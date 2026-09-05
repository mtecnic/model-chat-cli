"""Reusable Rich components for the Model Chat CLI."""
from rich.table import Table
from rich.text import Text


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


def create_model_table(servers: list) -> Table:
    """Create a table displaying discovered models.

    Args:
        servers: List of server dictionaries with models

    Returns:
        Rich Table with all discovered models

    The Model and Server columns shrink first when the terminal is narrow:
    the server column is right-aligned with overflow="fold", so the last
    characters of the address (the host's last octet and the port) stay
    visible when the row can't fit fully.
    """
    table = Table(title="[accent.bold]Discovered Models[/accent.bold]", show_header=True)
    table.add_column("#", style="chrome", width=3, justify="right")
    table.add_column("Model", style="model.name", no_wrap=True, overflow="fold", min_width=10, ratio=3)
    table.add_column("Server", style="model.server", no_wrap=True, overflow="fold",
                     justify="right", min_width=18, ratio=2)
    table.add_column("Ctx", style="chrome", justify="right", no_wrap=True)
    table.add_column("Type", style="chrome", justify="center", no_wrap=True)
    table.add_column("Status", justify="center", width=1)
    table.add_column("Latency", justify="right", width=7)

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
            latency = f"{response_time}ms" if response_time else "N/A"

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
