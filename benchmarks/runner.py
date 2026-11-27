"""Run a benchmark against a model and save the results."""
import asyncio
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

import httpx

from benchmarks.base import Benchmark, BenchItem, BenchSummary, ItemResult
from benchmarks.datasets import BENCH_HOME, load_dataset
from client import DEFAULT_TIMEOUT, ModelClient, ModelServerError
from think_parser import strip_thinking

THINKING_TOKEN_FACTOR = 4

ResultCB = Optional[Callable[[ItemResult], Awaitable[None]]]


def sample_items(bench: Benchmark, rows: List[dict], n: Optional[int], seed: int) -> List[BenchItem]:
    """Deterministic sample of n items (None or n >= len -> all, in dataset order)."""
    indices = list(range(len(rows)))
    if n is not None and n < len(rows):
        indices = sorted(random.Random(seed).sample(indices, n))
    return [bench.to_item(rows[i], i) for i in indices]


async def load_items(bench: Benchmark, n: Optional[int], seed: int,
                     progress=None) -> List[BenchItem]:
    rows = await load_dataset(bench.key, bench.hf_dataset, bench.hf_config,
                              bench.hf_split, progress)
    return sample_items(bench, rows, n, seed)


async def run_bench(bench: Benchmark, client: ModelClient, items: List[BenchItem],
                    results: List[ItemResult], concurrency: int = 4,
                    enable_thinking: Optional[bool] = None,
                    callback: ResultCB = None,
                    max_tokens: Optional[int] = None) -> List[ItemResult]:
    """Ask every item and grade it. Fills `results` in place (one per item) so a
    caller that cancels this coroutine still has everything that finished.

    max_tokens defaults to bench.max_tokens (x4 with thinking on, since
    reasoning models spend most of their budget before answering)."""
    if max_tokens is None:
        max_tokens = bench.max_tokens * (THINKING_TOKEN_FACTOR if enable_thinking else 1)
    results.clear()
    results.extend(ItemResult(index=i, item_id=it.id, gold=it.gold) for i, it in enumerate(items))
    sem = asyncio.Semaphore(max(1, concurrency))
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, limits=limits) as http:
        async def one(i: int, item: BenchItem):
            r = results[i]
            async with sem:
                r.status = "running"
                if callback:
                    await callback(r)
                start = time.monotonic()
                try:
                    msgs = bench.build_messages(item)
                    reply, metrics = await client.chat_with_metrics(
                        None, history=msgs, max_tokens=max_tokens,
                        enable_thinking=enable_thinking, http_client=http)
                    r.latency = time.monotonic() - start
                    r.ttft = metrics.ttft
                    r.completion_tokens = metrics.completion_tokens
                    r.reply = reply
                    answer = strip_thinking(reply)
                    if not answer.strip():
                        r.status, r.error = "error", "empty reply"
                    else:
                        # HumanEval grading runs a subprocess: keep the event loop free
                        ok, r.extracted = await asyncio.to_thread(bench.grade, item, answer)
                        r.status = "correct" if ok else "wrong"
                except asyncio.CancelledError:
                    r.status = "pending"
                    raise
                except Exception as e:
                    r.latency = time.monotonic() - start
                    msg = str(e) if isinstance(e, ModelServerError) else f"{type(e).__name__}: {e}"
                    r.status, r.error = "error", msg[:200]
            if callback:
                await callback(r)

        await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    return results


# --------------------------------------------------------------------------- #
# Result files
# --------------------------------------------------------------------------- #
def results_dir() -> Path:
    return BENCH_HOME / "results"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:80] or "model"


def save_run(summary: BenchSummary, results: List[ItemResult], config: dict) -> Path:
    d = results_dir()
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    path = d / f"{ts:%Y%m%d_%H%M%S}_{summary.bench}_{_slug(summary.model)}.json"
    data = {"saved_at": ts.isoformat(timespec="seconds"), "config": config,
            "summary": summary.to_dict(),
            "items": [r.to_dict() for r in results if r.finished]}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def previous_runs(bench_key: str, model: str, limit: int = 5,
                  exclude: Optional[Path] = None) -> List[dict]:
    """Newest saved runs for this bench+model: [{saved_at, config, summary}]."""
    d = results_dir()
    if not d.exists():
        return []
    runs = []
    for path in sorted(d.glob(f"*_{bench_key}_*.json"), reverse=True):
        if exclude is not None and path == exclude:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("summary", {}).get("model") != model:
            continue
        runs.append({k: data.get(k) for k in ("saved_at", "config", "summary")})
        if len(runs) >= limit:
            break
    return runs
