"""Unified result storage.

All results (chats, stress reports, arena results) live in one place:

    <exports root>
    ├── chats/    chat__<model>__<timestamp>.md / .json
    ├── stress/   stress__<mode>__<model>__<timestamp>.md / .json
    └── arena/    prompt_arena__<timestamp>.md / .json
                  model_arena__<timestamp>.md / .json

Every result is written as markdown (human) plus a JSON sidecar (machine).
"""
import datetime
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional


APP_NAME = "Model Chat"
APP_VERSION = "2.0.0"


def slugify(text: str, max_len: int = 48) -> str:
    """Make a string safe for use in a filename."""
    text = text.strip().replace("/", "__").replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.-]", "", text)
    return (text[:max_len] or "model")


def timestamp(now: Optional[datetime.datetime] = None) -> str:
    return (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")


@dataclass
class StoredResult:
    """Metadata for a result on disk (a .md with a .json sidecar)."""
    path: Path
    category: str      # "chats" | "stress" | "arena"
    kind: str          # "chat" | "stress" | "prompt_arena" | "model_arena"
    title: str
    created: str       # ISO timestamp, "" if unknown
    size: int = 0
    published: bool = False


class ResultStore:
    """Read/write results under a single exports root."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()
        for category in ("chats", "stress", "arena"):
            (self.root / category).mkdir(parents=True, exist_ok=True)

    # ── Writers ──────────────────────────────────────────────────────────

    def _write_pair(self, category: str, stem: str, markdown: str, json_data: dict) -> Path:
        """Write markdown + json sidecar, return the markdown path."""
        target = self.root / category / f"{stem}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown)
        json_path = target.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
        return target

    def save_chat(
        self,
        model: str,
        server: str,
        system_prompt: str,
        messages: List[dict],
        stats: Optional[dict] = None,
        markdown: str = "",
    ) -> Path:
        """Persist a chat transcript. `markdown` may be pre-rendered."""
        ts = timestamp()
        stem = f"chat__{slugify(model)}__{ts}"
        data = {
            "type": "chat",
            "app": f"{APP_NAME} {APP_VERSION}",
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "server": server,
            "system_prompt": system_prompt,
            "stats": stats or {},
            "messages": messages,
        }
        if not markdown:
            from storage.reports import chat_markdown
            markdown = chat_markdown(model, server, system_prompt, messages)
        return self._write_pair("chats", stem, markdown, data)

    def save_stress(
        self,
        mode: str,
        model: str,
        server: str,
        stats: Any,
        results: List[Any],
        markdown: str = "",
        extra: Optional[dict] = None,
    ) -> Path:
        """Persist a stress test report (stats + per-request results)."""
        ts = timestamp()
        stem = f"stress__{slugify(mode)}__{slugify(model)}__{ts}"
        data = {
            "type": "stress",
            "app": f"{APP_NAME} {APP_VERSION}",
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "mode": mode,
            "model": model,
            "server": server,
            "stats": _dataclass_dict(stats),
            "results": [_dataclass_dict(r) for r in results],
            "extra": extra or {},
        }
        if not markdown:
            from storage.reports import stress_markdown
            markdown = stress_markdown(mode, model, server, stats, results, extra or {})
        return self._write_pair("stress", stem, markdown, data)

    def save_prompt_arena(
        self,
        model: str,
        server: str,
        data: dict,
        markdown: str = "",
    ) -> Path:
        """Persist a prompt arena result."""
        ts = timestamp()
        stem = f"prompt_arena__{slugify(model)}__{ts}"
        envelope = {
            "type": "prompt_arena",
            "app": f"{APP_NAME} {APP_VERSION}",
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "server": server,
            "data": data,
        }
        if not markdown:
            from storage.reports import prompt_arena_markdown
            markdown = prompt_arena_markdown(model, server, data)
        return self._write_pair("arena", stem, markdown, envelope)

    def save_model_arena(
        self,
        models: List[str],
        data: dict,
        markdown: str = "",
    ) -> Path:
        """Persist a multi-model arena result."""
        ts = timestamp()
        slug = slugify("_vs_".join(models[:3]) + (f"_x{len(models)}" if len(models) > 3 else ""))
        stem = f"model_arena__{slug}__{ts}"
        envelope = {
            "type": "model_arena",
            "app": f"{APP_NAME} {APP_VERSION}",
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "models": models,
            "data": data,
        }
        if not markdown:
            from storage.reports import model_arena_markdown
            markdown = model_arena_markdown(models, data)
        return self._write_pair("arena", stem, markdown, envelope)

    # ── Readers ──────────────────────────────────────────────────────────

    def list_results(self) -> List[StoredResult]:
        """List all stored results, newest first."""
        results: List[StoredResult] = []
        for md in sorted(self.root.glob("*/*.md"), reverse=True):
            data = self._read_json(md.with_suffix(".json"))
            if data is None:
                continue
            category = md.parent.name
            kind = {
                "chats": "chat",
                "stress": "stress",
                "arena": "arena",
            }.get(category, category)
            if category == "arena":
                kind = data.get("type", "arena")
            title = self._title(kind, data, md)
            results.append(StoredResult(
                path=md,
                category=category,
                kind=kind,
                title=title,
                created=data.get("created", ""),
                size=md.stat().st_size,
            ))
        return results

    def result_paths(self) -> List[str]:
        """Relative paths (under the exports root) of all markdown results."""
        return sorted(str(p.relative_to(self.root)) for p in self.root.glob("*/*.md"))

    def read_result(self, path: Path) -> Optional[dict]:
        return self._read_json(Path(path).with_suffix(".json"))

    def _title(self, kind: str, data: dict, md: Path) -> str:
        if kind == "chat":
            return f"Chat: {data.get('model', '?')}"
        if kind == "stress":
            return f"Stress {data.get('mode', '?')}: {data.get('model', '?')}"
        if kind == "prompt_arena":
            return f"Prompt arena: {data.get('model', '?')}"
        if kind == "model_arena":
            models = data.get("models", [])
            return f"Model arena: {', '.join(models[:3])}{'…' if len(models) > 3 else ''}"
        return md.stem

    @staticmethod
    def _read_json(path: Path) -> Optional[dict]:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    # ── Legacy migration ─────────────────────────────────────────────────

    def migrate_legacy(self, cwd: str | Path, dry_run: bool = False) -> List[str]:
        """Move orphaned chat__*.md / arena_*.md files from a working directory
        into the store. Returns human-readable report lines."""
        cwd = Path(cwd)
        moved = []
        if not cwd.is_dir():
            return moved
        for path in sorted(cwd.glob("chat__[!#]*.md")):
            rel = self._legacy_rel(path, "chats")
            target = self.root / "chats" / path.name
            if not dry_run:
                shutil.move(str(path), str(target))
            moved.append(f"{path.name} -> chats/")
        for path in sorted(cwd.glob("arena_*.md")):
            target = self.root / "arena" / path.name
            if not dry_run:
                shutil.move(str(path), str(target))
            moved.append(f"{path.name} -> arena/")
        return moved

    @staticmethod
    def _legacy_rel(path: Path, category: str) -> str:
        return f"{category}/{path.name}"


def _dataclass_dict(obj: Any) -> dict:
    """Convert a dataclass (or dict) to a JSON-friendly dict."""
    import dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}
