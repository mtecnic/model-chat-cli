"""Answer extraction helpers shared by the benchmarks."""
import re
from typing import Optional

_NUM = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"


def normalize_number(text: str) -> Optional[str]:
    """'$1,234.50' -> '1234.5', '-7' -> '-7', '3.0' -> '3'. None if not a number."""
    s = text.replace(",", "").replace("$", "").replace(" ", "").strip().rstrip(".")
    try:
        value = float(s)
    except ValueError:
        return None
    if value == int(value):
        return str(int(value))
    return repr(value).rstrip("0").rstrip(".")


def extract_final_number(reply: str) -> Optional[str]:
    """GSM8K-style: the number after '####', then 'answer is N', then \\boxed{N},
    then the last number in the reply."""
    for pattern in (rf"####\s*({_NUM})",
                    rf"answer\s*(?:is|:)\s*\**\s*({_NUM})",
                    rf"\\boxed\{{\s*({_NUM})\s*\}}"):
        matches = re.findall(pattern, reply, flags=re.IGNORECASE)
        if matches:
            return normalize_number(matches[-1])
    numbers = re.findall(_NUM, reply)
    return normalize_number(numbers[-1]) if numbers else None


def extract_choice(reply: str, letters: str) -> Optional[str]:
    """Multiple-choice letter, following the MMLU-Pro reference extraction:
    'answer is (X)' -> 'Answer: X' -> the last standalone letter."""
    cls = f"[{letters}]"
    for pattern in (rf"answer\s+is\s*:?\s*\(?\**\s*({cls})\b",
                    rf"answer\s*:\s*\(?\**\s*({cls})\b",
                    rf"\\boxed\{{\s*(?:\\text\{{)?\s*({cls})\b"):
        matches = re.findall(pattern, reply, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    # Fallback: last letter standing alone, e.g. "(C)" or "C." on its own
    matches = re.findall(rf"(?:^|[\s(\[*])({cls})(?=[\s).\]*:]|$)", reply)
    return matches[-1].upper() if matches else None
