"""Reusable Rich components for the Model Chat CLI."""
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.console import Group
import re


def create_model_table(servers: list) -> Table:
    """Create a table displaying discovered models.

    Args:
        servers: List of server dictionaries with models

    Returns:
        Rich Table with all discovered models
    """
    table = Table(title="[bold cyan]Discovered Models[/bold cyan]", show_header=True)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Model", style="model.name", no_wrap=True)
    table.add_column("Server", style="model.server")
    table.add_column("Type", style="dim", justify="center")
    table.add_column("Status", justify="center", width=8)
    table.add_column("Latency", justify="right", width=10)

    idx = 1
    for server in servers:
        for model in server.get("models", []):
            status = server.get("status", "unknown")
            status_symbol = {
                "healthy": "[model.healthy]✓[/model.healthy]",
                "error": "[model.error]✗[/model.error]",
                "unknown": "[model.unknown]•[/model.unknown]",
            }.get(status, "•")

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


def create_chat_message(role: str, content: str, highlight_code: bool = True) -> Panel:
    """Create a Rich Panel for a chat message with optional code highlighting.

    Args:
        role: Message role ("user" or "assistant")
        content: Message content
        highlight_code: Whether to highlight code blocks

    Returns:
        Rich Panel formatted as a chat message
    """
    style = "chat.user" if role == "user" else "chat.assistant"
    border = "blue" if role == "user" else "green"

    # For assistant messages, render as markdown with code highlighting
    if role == "assistant" and highlight_code:
        renderable = render_markdown_with_code(content)
    else:
        renderable = content

    return Panel(
        renderable,
        title=f"[{style}]{role.upper()}[/{style}]",
        border_style=border,
        padding=(0, 1)
    )


def render_markdown_with_code(content: str):
    """Render markdown content with syntax-highlighted code blocks.

    Args:
        content: Markdown content potentially containing code blocks

    Returns:
        Rich renderable (Markdown or Group with Syntax)
    """
    # Check if there are code blocks
    code_block_pattern = r'```(\w+)?\n(.*?)```'
    matches = list(re.finditer(code_block_pattern, content, re.DOTALL))

    if not matches:
        # No code blocks, just render as markdown
        return Markdown(content)

    # Split content into parts and render code blocks with syntax highlighting
    renderables = []
    last_end = 0

    for match in matches:
        # Add markdown before code block
        if match.start() > last_end:
            pre_content = content[last_end:match.start()].strip()
            if pre_content:
                renderables.append(Markdown(pre_content))

        # Add syntax-highlighted code block
        language = match.group(1) or "text"
        code = match.group(2).strip()
        renderables.append(Syntax(code, language, theme="monokai", line_numbers=False))

        last_end = match.end()

    # Add remaining content after last code block
    if last_end < len(content):
        post_content = content[last_end:].strip()
        if post_content:
            renderables.append(Markdown(post_content))

    return Group(*renderables) if len(renderables) > 1 else renderables[0]


def create_header(model: str, server: str, avg_tps: float = 0.0) -> Panel:
    """Create a header panel showing current model and server.

    Args:
        model: Model name
        server: Server address
        avg_tps: Average tokens per second (optional)

    Returns:
        Rich Panel formatted as header
    """
    content = f"[bold cyan]{model}[/bold cyan] @ [dim]{server}[/dim]"
    if avg_tps > 0:
        content += f"  |  [bold yellow]{avg_tps:.1f}[/bold yellow] [dim]tokens/sec[/dim]"
    return Panel(content, style="header", padding=(0, 1))


def create_footer(message: str) -> Panel:
    """Create a footer panel with help text or status.

    Args:
        message: Footer message

    Returns:
        Rich Panel formatted as footer
    """
    return Panel(message, style="footer", padding=(0, 1))


def create_typing_indicator() -> Panel:
    """Create a typing indicator panel for when assistant is responding.

    Returns:
        Rich Panel showing typing indicator
    """
    return Panel(
        "[dim italic]...[/dim italic]",
        title="[chat.assistant]ASSISTANT[/chat.assistant]",
        border_style="dim green",
        padding=(0, 1)
    )
