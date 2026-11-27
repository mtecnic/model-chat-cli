"""Accuracy benchmarks (GSM8K, MMLU-Pro, ARC-Challenge, HumanEval).

Datasets download from Hugging Face on first use and are cached under
~/.model_chat_bench/data/. See runner.run_bench() for the entry point.
"""
from benchmarks.base import Benchmark, BenchItem, ItemResult, BenchSummary
from benchmarks.gsm8k import GSM8K
from benchmarks.mmlu_pro import MMLUPro
from benchmarks.arc import ARCChallenge
from benchmarks.humaneval import HumanEval

# Display order in the /bench menu
ALL_BENCHMARKS = [GSM8K, MMLUPro, ARCChallenge, HumanEval]

__all__ = ["Benchmark", "BenchItem", "ItemResult", "BenchSummary",
           "GSM8K", "MMLUPro", "ARCChallenge", "HumanEval", "ALL_BENCHMARKS"]
