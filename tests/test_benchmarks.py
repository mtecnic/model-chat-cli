"""Benchmark extraction, grading, dataset paging, sandbox and runner."""
import asyncio
import json

import httpx
import pytest

from benchmarks import ARCChallenge, GSM8K, HumanEval, MMLUPro
from benchmarks import datasets as ds
from benchmarks.base import BenchSummary, ItemResult, wilson_interval
from benchmarks.extract import extract_choice, extract_final_number, normalize_number
from benchmarks.runner import run_bench, sample_items
from benchmarks.sandbox import run_python


# ----------------------------------------------------------------- extraction
@pytest.mark.parametrize("text,want", [
    ("$1,234.50", "1234.5"), ("18", "18"), ("3.0", "3"), ("-7.", "-7"), ("abc", None),
])
def test_normalize_number(text, want):
    assert normalize_number(text) == want


@pytest.mark.parametrize("reply,want", [
    ("9 eggs * $2 = $18.\n#### 18", "18"),
    ("So she makes 2 then 5... #### $1,200", "1200"),
    ("The answer is 42.", "42"),
    ("Final: \\boxed{7}", "7"),
    ("first 3, then 4, total 12", "12"),
    ("no numbers here", None),
])
def test_extract_final_number(reply, want):
    assert extract_final_number(reply) == want


@pytest.mark.parametrize("reply,want", [
    ("Let's think. B looks wrong. The answer is (C).", "C"),
    ("the answer is **D**", "D"),
    ("Answer: J", "J"),
    ("I pick (E) because ...", "E"),
    ("A is tempting but the answer is (B). Not A.", "B"),
    ("nothing useful", None),
])
def test_extract_choice(reply, want):
    assert extract_choice(reply, "ABCDEFGHIJ") == want


# ----------------------------------------------------------------- grading
def test_gsm8k_item_and_grade():
    b = GSM8K()
    item = b.to_item({"question": "q?", "answer": "work\n#### 1,800"}, 3)
    assert item.gold == "1800" and item.id == "gsm8k/3"
    assert b.grade(item, "steps...\n#### 1800") == (True, "1800")
    assert b.grade(item, "#### 1799")[0] is False


def test_mmlu_pro_item_and_grade():
    b = MMLUPro()
    row = {"question_id": 5, "question": "Q", "options": ["a", "b", "c"], "answer": "B",
           "category": "law"}
    item = b.to_item(row, 0)
    assert "C. c" in item.prompt and "law" in item.prompt
    assert item.meta["letters"] == "ABC"
    assert b.grade(item, "The answer is (B)") == (True, "B")
    # Letter outside this question's options is not extracted
    assert b.grade(item, "The answer is (H)")[0] is False


def test_arc_numeric_labels():
    b = ARCChallenge()
    row = {"id": "x", "question": "Q", "answerKey": "2",
           "choices": {"label": ["1", "2", "3", "4"], "text": ["a", "b", "c", "d"]}}
    item = b.to_item(row, 0)
    assert item.gold == "B" and "B. b" in item.prompt
    assert b.grade(item, "The answer is (B).")[0]


HE_ROW = {
    "task_id": "HumanEval/0", "entry_point": "add",
    "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
    "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
}


def test_humaneval_pass_and_fail():
    b = HumanEval()
    item = b.to_item(HE_ROW, 0)
    good = "Here:\n```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    body_only = "```python\n    return a + b\n```"
    assert b.grade(item, good)[0] is True
    assert b.grade(item, bad)[0] is False
    assert b.grade(item, body_only)[0] is True


def test_sandbox_timeout_and_error():
    assert run_python("while True: pass\n", timeout=1) == (False, "timeout")
    ok, why = run_python("raise ValueError('boom')\n")
    assert not ok and "ValueError" in why
    assert run_python("print('hi')\n") == (True, None)


# ----------------------------------------------------------------- stats
def test_wilson_and_summary():
    lo, hi = wilson_interval(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson_interval(0, 0) == (0.0, 0.0)
    results = [ItemResult(0, "a", "correct"), ItemResult(1, "b", "wrong"),
               ItemResult(2, "c", "error"), ItemResult(3, "d", "pending")]
    s = BenchSummary.from_results("x", "m", 4, results, 1.0, stopped_early=True)
    assert (s.finished, s.correct, s.errors) == (3, 1, 1)
    assert s.accuracy == pytest.approx(1 / 3)


def test_sampling_is_deterministic():
    rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(100)]
    a = sample_items(GSM8K(), rows, 10, seed=7)
    b = sample_items(GSM8K(), rows, 10, seed=7)
    assert [i.id for i in a] == [i.id for i in b]
    assert len(sample_items(GSM8K(), rows, None, seed=7)) == 100


# ----------------------------------------------------------------- dataset paging
def test_fetch_rows_pages_and_retries_429():
    total = 250
    calls = {"429": 0}

    def handler(request: httpx.Request):
        offset = int(request.url.params["offset"])
        if offset == 100 and calls["429"] == 0:
            calls["429"] += 1
            return httpx.Response(429, text="slow down")
        rows = [{"row": {"i": i}} for i in range(offset, min(offset + 100, total))]
        return httpx.Response(200, json={"num_rows_total": total, "rows": rows})

    async def no_sleep(_):
        pass

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            return await ds.fetch_rows("d", "c", "s", http_client=c, sleep=no_sleep)

    rows = asyncio.run(go())
    assert [r["i"] for r in rows] == list(range(total))
    assert calls["429"] == 1


def test_fetch_rows_404_raises():
    def handler(request):
        return httpx.Response(404, text="no such dataset")

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
            await ds.fetch_rows("d", "c", "s", http_client=c)

    with pytest.raises(ds.DatasetError):
        asyncio.run(go())


def test_dataset_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "BENCH_HOME", tmp_path)
    assert ds.load_cached("k") is None
    ds._save_cache("k", [{"a": 1}, {"a": "é"}])
    assert ds.load_cached("k") == [{"a": 1}, {"a": "é"}]


# ----------------------------------------------------------------- runner
class _FakeClient:
    """Answers GSM8K items: correct for even ids, wrong for odd, error on id 3."""

    async def chat_with_metrics(self, message, history=None, max_tokens=None,
                                enable_thinking=None, http_client=None):
        from client import ChatMetrics
        q = history[-1]["content"]
        n = int(q.rsplit("q", 1)[-1])
        if n == 3:
            raise RuntimeError("server exploded")
        m = ChatMetrics(completion_tokens=10)
        answer = n if n % 2 == 0 else n + 1
        return f"<think>hmm {n + 5}</think>work\n#### {answer}", m


def test_run_bench_scores_and_strips_thinking():
    rows = [{"question": f"q{i}", "answer": f"#### {i}"} for i in range(6)]
    bench = GSM8K()
    items = sample_items(bench, rows, None, 0)
    results = []
    seen = []

    async def cb(r):
        seen.append(r.status)

    asyncio.run(run_bench(bench, _FakeClient(), items, results, concurrency=2, callback=cb))
    statuses = [r.status for r in results]
    assert statuses == ["correct", "wrong", "correct", "error", "correct", "wrong"]
    assert "running" in seen
    assert results[3].error.startswith("RuntimeError")
