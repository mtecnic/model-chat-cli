"""Discovery screen - scan the network and pick a model."""
from __future__ import annotations

from rich.text import Text
from textual import work
from textual.widgets.option_list import Option
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Input, OptionList, ProgressBar, Static

import scanner
from ui.base import BaseScreen, MUTED, make_header


class DiscoveryScreen(BaseScreen):
    DEFAULT_CSS = """
    DiscoveryScreen #disc-header {
        height: 3;
        content-align: center middle;
        border-bottom: solid $panel;
    }
    DiscoveryScreen #scan-status {
        height: 1;
        content-align: center middle;
        color: $text-muted;
    }
    DiscoveryScreen #scan-progress {
        display: none;
        width: 60%;
        align: center middle;
    }
    DiscoveryScreen Horizontal {
        height: 1;
        width: 100%;
    }
    DiscoveryScreen Input {
        width: 40%;
        height: 3;
        margin: 0 0 0 4;
    }
    DiscoveryScreen OptionList {
        height: 1fr;
    }
    DiscoveryScreen #disc-hints {
        height: 3;
        content-align: center middle;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("s", "rescan", "Re-scan", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._all_options: list[tuple[str, str, str, str]] = []  # (server_ip, port, model, server_type)
        self._scanning = False

    def build_content(self) -> ComposeResult:
        yield Static("", id="disc-header")
        yield Static("", id="scan-status")
        yield ProgressBar(id="scan-progress", show_eta=False)
        with Vertical(id="disc-list"):
            yield OptionList(id="models")
        yield Input(placeholder="filter models…", id="filter")
        yield Static("", id="disc-hints")

    def on_mount(self) -> None:
        app = self.app
        self.query_one("#disc-header", Static).update(
            make_header("Models & Servers",
                        f"{app.config.get('scan.concurrency', 500)}-way TCP pre-scan"))
        self.query_one("#disc-hints", Static).update(
            "enter: select model   /: filter   s: re-scan   esc: back"
        )
        self.run_worker(self._initial_load, name="discover", exclusive=True)

    # ------------------------------------------------------------------ #

    async def _initial_load(self) -> None:
        app = self.app
        cached = scanner.load_cache()
        if cached:
            self._set_status(f"validating {len(cached)} cached server(s)…")
            live = await scanner.quick_validate_cache(cached)
            if live:
                self._fill(live)
                self._set_status(f"{len(live)} server(s) live — press s to re-scan")
                return
        await self._full_scan()

    async def _full_scan(self) -> None:
        if self._scanning:
            return
        self._scanning = True
        app = self.app
        self._set_status("scanning local network…")
        bar = self.query_one("#scan-progress", ProgressBar)
        bar.styles.display = "block"
        bar.update(total=0)
        bar.border_subtitle = "TCP pre-scan…"

        extra = app.config.get("scan.extra_ips", "")
        if isinstance(extra, list):
            extra = ", ".join(str(x) for x in extra)
        ips = [ip.strip() for ip in str(extra).split(",") if ip.strip()]

        async def cb(current: int, total: int, phase: str = "") -> None:
            bar.update(total=total or 0, progress=current)
            if phase:
                bar.border_subtitle = f"{phase}  {current}/{total}"

        try:
            servers = await scanner.scan_network(
                progress_callback=cb,
                extra_ips=ips,
                concurrency=int(app.config.get("scan.concurrency", 500)),
                tcp_timeout=float(app.config.get("scan.tcp_timeout", 0.5)),
            )
        finally:
            bar.styles.display = "none"

        self._scanning = False
        if servers:
            scanner.save_cache(servers)
            self._fill(servers)
            self._set_status(f"found {len(servers)} server(s) / "
                             f"{sum(len(s.get('models', [])) for s in servers)} model(s)")
        else:
            self._set_status("no servers found — try the scan settings (s)")

    def action_rescan(self) -> None:
        self.run_worker(self._full_scan, name="rescan")

    def action_back(self) -> None:
        self.app.pop_screen()

    # ------------------------------------------------------------------ #

    def _set_status(self, msg: str) -> None:
        self.query_one("#scan-status", Static).update(Text(msg, style=MUTED))

    def _fill(self, servers: list[dict]) -> None:
        self.app.servers = servers
        self._all_options = []
        lst = self.query_one("#models", OptionList)
        lst.clear_options()
        for s in servers:
            for m in s.get("models", []):
                name = m if isinstance(m, str) else (m.get("name") or "?")
                self._all_options.append((s["ip"], str(s["port"]), name, s.get("type", "")))
                text = Text.assemble(
                    (name, "bright_cyan"),
                    (f"   {s['ip']}:{s['port']}  ·  {s.get('type', '?')}", MUTED),
                )
                lst.add_option(Option(text, id=f"{s['ip']}:{s['port']}:{name}"))
        if self._all_options:
            lst.highlighted = 0
        if not self.query_one("#filter", Input).value:
            self.query_one("#filter", Input).focused = True

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        app = self.app
        try:
            ip, port, model = str(event.option_id).split(":")
        except ValueError:
            return
        server = next((s for s in app.servers
                       if s["ip"] == ip and str(s["port"]) == port), None)
        if not server:
            return
        app.server = server
        app.model = model
        self.app.pop_screen()
        self.app.push_screen("chat")

    def on_input_changed(self, event: Input.Changed) -> None:
        q = event.value.strip().lower()
        lst = self.query_one("#models", OptionList)
        lst.clear_options()
        for (ip, port, model, stype) in self._all_options:
            if q and q not in model.lower() and q not in f"{ip}:{port}".lower():
                continue
            text = Text.assemble(
                (model, "bright_cyan"),
                (f"   {ip}:{port}  ·  {stype}", MUTED),
            )
            lst.add_option(Option(text, id=f"{ip}:{port}:{model}"))
        if lst.option_count:
            lst.highlighted = 0
