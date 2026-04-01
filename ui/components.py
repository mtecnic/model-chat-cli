"""Reusable Rich components for the Model Chat CLI."""
from rich.table import Table
from rich.text import Text


def create_model_table(servers: list) -> Table:
    """Create a table displaying discovered models.

    Args:
        servers: List of server dictionaries with models

    Returns:
        Rich Table with all discovered models
    """
    table = Table(title="[accent.bold]Discovered Models[/accent.bold]", show_header=True)
    table.add_column("#", style="chrome", width=4, justify="right")
    table.add_column("Model", style="model.name", no_wrap=True)
    table.add_column("Server", style="model.server")
    table.add_column("Type", style="chrome", justify="center")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Latency", justify="right", width=10)

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

            table.add_row(
                str(idx),
                model,
                server_addr,
                server_type,
                status_symbol,
                latency
            )
            idx += 1

    return table


def format_stats_line(tokens: int, elapsed: float, tps: float) -> Text:
    """Render compact stats after a response: ↳ 234 tok · 5.2s · 45.0 t/s"""
    t = Text()
    t.append("  \u21b3 ", style="chrome")
    t.append(str(tokens), style="metric")
    t.append(" tok ", style="metric.label")
    t.append("\u00b7 ", style="chrome")
    t.append(f"{elapsed:.1f}s", style="metric")
    t.append(" \u00b7 ", style="chrome")
    t.append(f"{tps:.1f}", style="metric")
    t.append(" t/s", style="metric.label")
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
