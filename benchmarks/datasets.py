"""Download benchmark datasets from Hugging Face and cache them locally.

Uses the public datasets-server rows API (JSON, 100 rows per page), so no
`datasets`/`pyarrow` dependency is needed. Each dataset is cached as JSONL in
~/.model_chat_bench/data/<key>.jsonl; a cache hit makes no network calls.
"""
import asyncio
import json
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

import httpx

ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100
FETCH_CONCURRENCY = 8
MAX_ATTEMPTS = 6

BENCH_HOME = Path.home() / ".model_chat_bench"

ProgressCB = Optional[Callable[[int, int], Awaitable[None]]]


class DatasetError(Exception):
    """The dataset could not be downloaded."""


def data_dir() -> Path:
    return BENCH_HOME / "data"


def cache_path(key: str) -> Path:
    return data_dir() / f"{key}.jsonl"


def load_cached(key: str) -> Optional[List[dict]]:
    path = cache_path(key)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None  # corrupt cache: re-download


def _save_cache(key: str, rows: List[dict]) -> None:
    path = cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)  # atomic: an interrupted download never leaves a partial cache


async def _get_page(client: httpx.AsyncClient, params: dict,
                    sleep=asyncio.sleep) -> dict:
    """GET one page, backing off on 429 / 5xx."""
    delay = 1.0
    last = ""
    for _ in range(MAX_ATTEMPTS):
        try:
            resp = await client.get(ROWS_URL, params=params)
        except httpx.RequestError as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if resp.status_code == 200:
                return resp.json()
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code != 429 and resp.status_code < 500:
                raise DatasetError(last)
            retry_after = resp.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                delay = max(delay, float(retry_after))
        await sleep(delay)
        delay = min(delay * 2, 30.0)
    raise DatasetError(f"gave up after {MAX_ATTEMPTS} attempts ({last})")


async def fetch_rows(dataset: str, config: str, split: str,
                     progress: ProgressCB = None,
                     http_client: Optional[httpx.AsyncClient] = None,
                     sleep=asyncio.sleep) -> List[dict]:
    """Download every row of a dataset split, in order."""
    base = {"dataset": dataset, "config": config, "split": split}
    own = http_client is None
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    try:
        first = await _get_page(client, {**base, "offset": 0, "length": PAGE_SIZE}, sleep)
        total = int(first.get("num_rows_total") or len(first.get("rows", [])))
        pages = {0: first}
        offsets = list(range(PAGE_SIZE, total, PAGE_SIZE))
        done = 1
        n_pages = 1 + len(offsets)
        if progress:
            await progress(done, n_pages)

        sem = asyncio.Semaphore(FETCH_CONCURRENCY)

        async def one(offset: int):
            nonlocal done
            async with sem:
                pages[offset] = await _get_page(
                    client, {**base, "offset": offset, "length": PAGE_SIZE}, sleep)
            done += 1
            if progress:
                await progress(done, n_pages)

        await asyncio.gather(*(one(o) for o in offsets))
    finally:
        if own:
            await client.aclose()

    rows = []
    for offset in sorted(pages):
        rows.extend(r["row"] for r in pages[offset].get("rows", []))
    if not rows:
        raise DatasetError(f"{dataset} returned no rows")
    return rows


async def load_dataset(key: str, dataset: str, config: str, split: str,
                       progress: ProgressCB = None,
                       http_client: Optional[httpx.AsyncClient] = None) -> List[dict]:
    """Return the dataset rows, downloading and caching on first use."""
    rows = load_cached(key)
    if rows is not None:
        return rows
    rows = await fetch_rows(dataset, config, split, progress, http_client)
    _save_cache(key, rows)
    return rows
