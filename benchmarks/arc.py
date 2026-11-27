"""ARC-Challenge: grade-school science multiple choice (hard subset)."""
from benchmarks.base import Benchmark, BenchItem
from benchmarks.extract import extract_choice

PROMPT = (
    "Answer the following science question. Think briefly, then finish with "
    "\"The answer is (X)\" where X is the letter of the correct choice.\n\n"
    "Question: {question}\nChoices:\n{choices}\n"
)

# A few items label choices 1-4 instead of A-D
_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}


class ARCChallenge(Benchmark):
    key = "arc_challenge"
    name = "ARC-Challenge"
    description = "1172 hard grade-school science multiple-choice questions"
    hf_dataset = "allenai/ai2_arc"
    hf_config = "ARC-Challenge"
    hf_split = "test"
    max_tokens = 1024

    def to_item(self, row: dict, index: int) -> BenchItem:
        labels = [_NUM_TO_LETTER.get(l, l) for l in row["choices"]["label"]]
        choices = "\n".join(f"{l}. {t}" for l, t in zip(labels, row["choices"]["text"]))
        gold = _NUM_TO_LETTER.get(row["answerKey"], row["answerKey"])
        return BenchItem(id=f"arc/{row.get('id', index)}",
                         prompt=PROMPT.format(question=row["question"], choices=choices),
                         gold=gold, meta={"letters": "".join(labels)})

    def grade(self, item, reply):
        got = extract_choice(reply, item.meta["letters"])
        return got == item.gold, got
