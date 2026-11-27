"""Shared types for accuracy benchmarks."""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BenchItem:
    """One benchmark question."""
    id: str
    prompt: str
    gold: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ItemResult:
    """Outcome of one question. status: pending | running | correct | wrong | error."""
    index: int
    item_id: str
    status: str = "pending"
    extracted: Optional[str] = None
    gold: Any = None
    reply: str = ""
    error: str = ""
    completion_tokens: int = 0
    latency: float = 0.0
    ttft: float = 0.0

    @property
    def finished(self) -> bool:
        return self.status in ("correct", "wrong", "error")

    def to_dict(self) -> dict:
        return {
            "index": self.index, "id": self.item_id, "status": self.status,
            "extracted": self.extracted, "gold": self.gold, "reply": self.reply,
            "error": self.error, "completion_tokens": self.completion_tokens,
            "latency": round(self.latency, 3), "ttft": round(self.ttft, 3),
        }


def wilson_interval(correct: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion (well-behaved at 0% / 100%)."""
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class BenchSummary:
    """Aggregate over finished items. Errors count as wrong in accuracy."""
    bench: str
    model: str
    n_requested: int
    finished: int = 0
    correct: int = 0
    wrong: int = 0
    errors: int = 0
    avg_completion_tokens: float = 0.0
    avg_latency: float = 0.0
    wall_time: float = 0.0
    stopped_early: bool = False

    @property
    def accuracy(self) -> float:
        return self.correct / self.finished if self.finished else 0.0

    @property
    def ci(self) -> Tuple[float, float]:
        return wilson_interval(self.correct, self.finished)

    @classmethod
    def from_results(cls, bench: str, model: str, n_requested: int,
                     results: List[ItemResult], wall_time: float,
                     stopped_early: bool = False) -> "BenchSummary":
        done = [r for r in results if r.finished]
        ok = [r for r in done if r.status != "error"]
        return cls(
            bench=bench, model=model, n_requested=n_requested,
            finished=len(done),
            correct=sum(r.status == "correct" for r in done),
            wrong=sum(r.status == "wrong" for r in done),
            errors=sum(r.status == "error" for r in done),
            avg_completion_tokens=(sum(r.completion_tokens for r in ok) / len(ok)) if ok else 0.0,
            avg_latency=(sum(r.latency for r in ok) / len(ok)) if ok else 0.0,
            wall_time=wall_time, stopped_early=stopped_early,
        )

    def to_dict(self) -> dict:
        lo, hi = self.ci
        return {
            "bench": self.bench, "model": self.model, "n_requested": self.n_requested,
            "finished": self.finished, "correct": self.correct, "wrong": self.wrong,
            "errors": self.errors, "accuracy": self.accuracy, "ci_low": lo, "ci_high": hi,
            "avg_completion_tokens": self.avg_completion_tokens,
            "avg_latency": self.avg_latency, "wall_time": self.wall_time,
            "stopped_early": self.stopped_early,
        }


class Benchmark:
    """Base class. Subclasses define the dataset source, prompt and grader."""

    key: str = ""            # short id, used for cache/result file names
    name: str = ""           # display name
    description: str = ""
    # Hugging Face datasets-server coordinates
    hf_dataset: str = ""
    hf_config: str = "default"
    hf_split: str = "test"
    max_tokens: int = 2048   # generation budget per question (CoT needs room)
    executes_code: bool = False

    def to_item(self, row: dict, index: int) -> BenchItem:
        raise NotImplementedError

    def build_messages(self, item: BenchItem) -> List[dict]:
        return [{"role": "user", "content": item.prompt}]

    def grade(self, item: BenchItem, reply: str) -> Tuple[bool, Optional[str]]:
        """Return (correct, extracted_answer). `reply` has <think> blocks removed."""
        raise NotImplementedError
