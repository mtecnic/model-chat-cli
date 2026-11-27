"""MMLU-Pro: 10-option multiple choice across 14 subjects (zero-shot CoT)."""
from benchmarks.base import Benchmark, BenchItem
from benchmarks.extract import extract_choice

LETTERS = "ABCDEFGHIJ"

PROMPT = (
    "The following is a multiple choice question about {subject}. "
    "Think step by step and then finish your answer with \"The answer is (X)\" "
    "where X is the correct letter choice.\n\n"
    "Question: {question}\nOptions:\n{options}\n"
)


class MMLUPro(Benchmark):
    key = "mmlu_pro"
    name = "MMLU-Pro"
    description = "12032 hard 10-option multiple-choice questions, 14 subjects"
    hf_dataset = "TIGER-Lab/MMLU-Pro"
    hf_config = "default"
    hf_split = "test"
    max_tokens = 2048

    def to_item(self, row: dict, index: int) -> BenchItem:
        options = "\n".join(f"{LETTERS[i]}. {opt}" for i, opt in enumerate(row["options"]))
        prompt = PROMPT.format(subject=row.get("category", "general knowledge"),
                               question=row["question"], options=options)
        return BenchItem(id=f"mmlu_pro/{row.get('question_id', index)}", prompt=prompt,
                         gold=row["answer"].strip().upper(),
                         meta={"category": row.get("category", ""),
                               "letters": LETTERS[:len(row["options"])]})

    def grade(self, item, reply):
        got = extract_choice(reply, item.meta.get("letters", LETTERS))
        return got == item.gold, got
