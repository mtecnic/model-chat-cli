"""GSM8K: grade-school math word problems, exact numeric match."""
from benchmarks.base import Benchmark, BenchItem
from benchmarks.extract import extract_final_number, normalize_number

PROMPT = (
    "Solve the following math problem step by step. "
    "At the end, write the final answer as a number on its own line in the form "
    "\"#### <number>\".\n\n{question}"
)


class GSM8K(Benchmark):
    key = "gsm8k"
    name = "GSM8K"
    description = "1319 grade-school math word problems (exact numeric answer)"
    hf_dataset = "openai/gsm8k"
    hf_config = "main"
    hf_split = "test"
    max_tokens = 1536

    def to_item(self, row: dict, index: int) -> BenchItem:
        gold = normalize_number(row["answer"].split("####")[-1])
        return BenchItem(id=f"gsm8k/{index}", prompt=PROMPT.format(question=row["question"]),
                         gold=gold)

    def grade(self, item, reply):
        got = extract_final_number(reply)
        return got is not None and got == item.gold, got
