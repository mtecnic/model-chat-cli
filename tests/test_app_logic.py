"""Tests for think parsing, scanner model parsing, history and arena judging."""
import asyncio

from prompt_arena import JudgeResult, PromptArena, PromptResponse
from scanner import parse_models
from storage.history import ChatHistoryManager
from think_parser import ChunkType, ThinkTagParser


def test_think_parser_split_tags_across_chunks():
    p = ThinkTagParser()
    out = []
    for piece in ["<thi", "nk>reason", "ing</th", "ink>answer"]:
        out += p.feed(piece)
    out += p.flush()
    thinking = "".join(c.text for c in out if c.chunk_type == ChunkType.THINKING)
    content = "".join(c.text for c in out if c.chunk_type == ChunkType.CONTENT)
    assert (thinking, content) == ("reasoning", "answer")


def test_think_parser_start_thinking_without_close_flushes_as_content():
    p = ThinkTagParser(start_thinking=True)
    out = p.feed("just an answer") + p.flush()
    # Mid-stream text is (provisionally) thinking; the chat view reclassifies it
    assert "".join(c.text for c in out) == "just an answer"


def test_parse_models_openai_and_ollama():
    oa = parse_models("openai", {"data": [{"id": "qwen", "max_model_len": 32768},
                                          {"id": "llama", "meta": {"n_ctx": 8192}}]})
    assert oa == [{"name": "qwen", "max_context": 32768}, {"name": "llama", "max_context": 8192}]
    ol = parse_models("ollama", {"models": [{"name": "llama3:8b"}]})
    assert ol == [{"name": "llama3:8b", "max_context": None}]
    assert parse_models("openai", None) == []


def test_history_round_trip(tmp_path):
    mgr = ChatHistoryManager(root=str(tmp_path))
    msgs = [{"role": "user", "content": "first   question\nmore"},
            {"role": "assistant", "content": "answer", "thinking": "t"}]
    path = mgr.save_conversation("m/x:1", "1.2.3.4:11434", msgs, system_prompt="be brief")
    data = mgr.load_conversation(path)
    assert data["messages"] == msgs and data["system_prompt"] == "be brief"
    [summary] = mgr.list_summaries()
    assert summary["model"] == "m/x:1" and summary["preview"] == "first question more"


def _resp(key):
    return PromptResponse(prompt_key=key, prompt_name=key.upper(), question="q",
                          response=f"answer from {key}", status="success")


def _arena_with_verdicts(verdicts):
    """PromptArena whose _judge_once returns scripted (winner, score_first, score_second)."""
    arena = PromptArena({"url": "http://x", "type": "openai", "ip": "x", "port": 1}, "m")
    calls = []

    async def fake_judge_once(first, second, semaphore=None):
        calls.append((first.prompt_key, second.prompt_key))
        winner, sa, sb = verdicts[(first.prompt_key, second.prompt_key)]
        return JudgeResult(prompt_a=first.prompt_key, prompt_b=second.prompt_key,
                           winner=winner, score_a=sa, score_b=sb, confidence="high")

    arena._judge_once = fake_judge_once
    return arena, calls


def test_judge_pair_consistent_verdict_wins():
    # "a" wins in both orders (A first time, B when swapped)
    arena, calls = _arena_with_verdicts({("a", "b"): ("A", 8, 5), ("b", "a"): ("B", 6, 9)})
    jr = asyncio.run(arena.judge_pair(_resp("a"), _resp("b")))
    assert jr.winner == "A"
    assert (jr.score_a, jr.score_b) == (8.5, 5.5)
    assert sorted(calls) == [("a", "b"), ("b", "a")]


def test_judge_pair_position_bias_is_tie():
    # Judge always prefers whichever response is shown first
    arena, _ = _arena_with_verdicts({("a", "b"): ("A", 8, 5), ("b", "a"): ("A", 8, 5)})
    jr = asyncio.run(arena.judge_pair(_resp("a"), _resp("b")))
    assert jr.winner == "TIE" and jr.confidence == "low"


def test_judge_template_hides_prompt_names():
    from prompt_arena import JUDGE_TEMPLATE
    assert "{prompt_a}" not in JUDGE_TEMPLATE and "{prompt_b}" not in JUDGE_TEMPLATE
