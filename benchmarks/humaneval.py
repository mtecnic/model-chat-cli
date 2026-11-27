"""HumanEval: 164 Python functions, graded pass@1 by running the unit tests.

Grading EXECUTES MODEL-WRITTEN CODE on this machine (in a subprocess with
limits, see sandbox.py). The UI asks for confirmation before running it.
"""
import re

from benchmarks.base import Benchmark, BenchItem
from benchmarks.sandbox import run_python

PROMPT = (
    "Complete the following Python function. Reply with the complete function "
    "(including the signature and any imports it needs) in a single ```python code block.\n\n"
    "```python\n{prompt}```"
)

_FENCE = re.compile(r"```(?:python|py)?[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(reply: str, entry_point: str) -> str:
    """Pick the code block that defines the entry point (else the longest one)."""
    blocks = _FENCE.findall(reply)
    if not blocks:
        # Unterminated fence (ran out of tokens) or bare code
        start = reply.find("```")
        text = reply[start:].split("\n", 1)[1] if start >= 0 and "\n" in reply[start:] else reply
        blocks = [text]
    defining = [b for b in blocks if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", b)]
    return max(defining or blocks, key=len)


class HumanEval(Benchmark):
    key = "humaneval"
    name = "HumanEval"
    description = "164 Python coding problems, pass@1 (runs generated code locally)"
    hf_dataset = "openai/openai_humaneval"
    hf_config = "openai_humaneval"
    hf_split = "test"
    max_tokens = 2048
    executes_code = True

    def to_item(self, row: dict, index: int) -> BenchItem:
        return BenchItem(id=row["task_id"], prompt=PROMPT.format(prompt=row["prompt"]),
                         gold="pass",
                         meta={"prompt": row["prompt"], "test": row["test"],
                               "entry_point": row["entry_point"]})

    def build_program(self, item: BenchItem, reply: str) -> str:
        code = extract_code(reply, item.meta["entry_point"])
        # If the model returned only the body, prepend the original signature
        if not re.search(rf"def\s+{re.escape(item.meta['entry_point'])}\s*\(", code):
            code = item.meta["prompt"] + code
        else:
            # Keep the prompt's imports/helpers available, then the model's version wins
            code = item.meta["prompt"] + "    pass\n\n" + code
        return (f"{code}\n\n{item.meta['test']}\n\n"
                f"check({item.meta['entry_point']})\n")

    def grade(self, item, reply):
        ok, detail = run_python(self.build_program(item, reply))
        return ok, "pass" if ok else (detail or "fail")
