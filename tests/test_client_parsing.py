"""Offline tests for the streaming parsers and request/retry logic in client.py."""
import asyncio
import json

import httpx
import pytest

from client import (
    CONTENT, REASONING, ChatMetrics, ModelClient, ModelServerError,
    _normalize_tool_calls, iter_ollama_events, iter_openai_events, wrap_think,
)


def run(coro):
    return asyncio.run(coro)


async def chunked(data: bytes, size: int = 7):
    """Yield bytes in small awkward pieces, like a slow network."""
    for i in range(0, len(data), size):
        yield data[i:i + size]


async def collect(gen):
    return [x async for x in gen]


def sse(*events) -> bytes:
    out = b""
    for e in events:
        payload = e if isinstance(e, str) else json.dumps(e)
        out += f"data: {payload}\n\n".encode()
    return out


def delta(**kw):
    return {"choices": [{"delta": kw}]}


# ---------------------------------------------------------------- OpenAI SSE

def test_openai_reasoning_then_content_and_usage():
    body = sse(
        delta(reasoning_content="let me "),
        delta(reasoning="think"),           # NVIDIA-style field
        delta(content="Hello"),
        delta(content=" world"),
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 9}},
        "[DONE]",
    )
    m = ChatMetrics()
    events = run(collect(iter_openai_events(chunked(body), m, 0.0)))
    assert events == [(REASONING, "let me "), (REASONING, "think"),
                      (CONTENT, "Hello"), (CONTENT, " world")]
    assert (m.prompt_tokens, m.completion_tokens) == (5, 9)
    assert m.ttft > 0


def test_openai_stops_at_done_and_skips_junk():
    body = b": keepalive\n\ndata: not-json\n\n" + sse(delta(content="a"), "[DONE]", delta(content="IGNORED"))
    events = run(collect(iter_openai_events(chunked(body, 3), ChatMetrics(), 0.0)))
    assert events == [(CONTENT, "a")]


def test_openai_in_stream_error_raises():
    body = sse(delta(content="partial"), {"error": {"message": "CUDA out of memory"}})
    with pytest.raises(ModelServerError, match="CUDA out of memory"):
        run(collect(iter_openai_events(chunked(body), ChatMetrics(), 0.0)))


def test_openai_multibyte_split_across_chunks():
    body = sse(delta(content="héllo – ✓"))
    events = run(collect(iter_openai_events(chunked(body, 1), ChatMetrics(), 0.0)))
    assert events == [(CONTENT, "héllo – ✓")]


# -------------------------------------------------------------- Ollama NDJSON

def ndjson(*objs) -> bytes:
    return b"".join(json.dumps(o).encode() + b"\n" for o in objs)


def test_ollama_thinking_content_and_metrics():
    body = ndjson(
        {"message": {"role": "assistant", "content": "", "thinking": "hmm"}, "done": False},
        {"message": {"role": "assistant", "content": "Hi"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True,
         "eval_count": 12, "prompt_eval_count": 3, "eval_duration": 2_000_000_000,
         "prompt_eval_duration": 100},
    )
    m = ChatMetrics()
    events = run(collect(iter_ollama_events(chunked(body), m, 0.0)))
    assert events == [(REASONING, "hmm"), (CONTENT, "Hi")]
    assert (m.completion_tokens, m.prompt_tokens, m.eval_duration_ns) == (12, 3, 2_000_000_000)


def test_ollama_error_line_raises():
    body = ndjson({"error": "model 'nope' not found"})
    with pytest.raises(ModelServerError, match="not found"):
        run(collect(iter_ollama_events(chunked(body), ChatMetrics(), 0.0)))


def test_ollama_trailing_line_without_newline():
    body = json.dumps({"message": {"content": "end"}, "done": True}).encode()
    events = run(collect(iter_ollama_events(chunked(body), ChatMetrics(), 0.0)))
    assert events == [(CONTENT, "end")]


# ------------------------------------------------------------------ wrap_think

async def aiter(items):
    for i in items:
        yield i


def test_wrap_think_tags():
    out = run(collect(wrap_think(aiter([(REASONING, "a"), (REASONING, "b"), (CONTENT, "c")]))))
    assert "".join(out) == "<think>ab</think>c"
    out = run(collect(wrap_think(aiter([(REASONING, "only")]))))
    assert "".join(out) == "<think>only</think>"


# ---------------------------------------------------- request building / retry

def make_client(server_type: str, handler) -> tuple:
    """ModelClient + an httpx client backed by a MockTransport handler."""
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    server = {"url": "http://test", "type": server_type, "ip": "test", "port": 1}
    return ModelClient(server, "m"), http


def test_openai_retries_only_when_stream_options_rejected():
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        if "stream_options" in payload:
            return httpx.Response(400, json={"error": {"message": "Unknown field stream_options"}})
        return httpx.Response(200, content=sse(delta(content="ok"), "[DONE]"))

    client, http = make_client("openai", handler)
    text, _ = run(client.chat_with_metrics("hi", http_client=http))
    assert text == "ok"
    assert len(seen) == 2 and "stream_options" not in seen[1]


def test_openai_404_raises_without_retry():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(404, json={"error": {"message": "The model `m` does not exist."}})

    client, http = make_client("openai", handler)
    with pytest.raises(ModelServerError) as err:
        run(client.chat_with_metrics("hi", http_client=http))
    assert err.value.status == 404 and "does not exist" in str(err.value)
    assert len(calls) == 1


def test_network_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("refused")

    client, http = make_client("openai", handler)
    with pytest.raises(httpx.ConnectError):
        run(client.chat_with_metrics("hi", http_client=http))
    assert len(calls) == 1


def test_ollama_think_flag_and_unsupported_retry():
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        if payload.get("think"):
            return httpx.Response(400, json={"error": "\"llama3\" does not support thinking"})
        return httpx.Response(200, content=ndjson({"message": {"content": "plain"}, "done": True}))

    client, http = make_client("ollama", handler)
    text, _ = run(client.chat_with_metrics("hi", enable_thinking=True, http_client=http))
    assert text == "plain"
    assert seen[0]["think"] is True and "options" not in seen[0]
    assert "think" not in seen[1]


def test_ollama_http_error_surfaces_message():
    def handler(request):
        return httpx.Response(404, json={"error": "model 'x' not found, try pulling it first"})

    client, http = make_client("ollama", handler)
    with pytest.raises(ModelServerError, match="try pulling it first"):
        run(client.chat_with_metrics("hi", http_client=http))


def test_ollama_max_tokens_option():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, content=ndjson({"message": {"content": "x"}, "done": True}))

    client, http = make_client("ollama", handler)
    run(client.chat_with_metrics("hi", max_tokens=5, http_client=http))
    assert seen[0]["options"] == {"num_predict": 5}


def test_normalize_tool_calls():
    out = _normalize_tool_calls([
        {"id": "a", "function": {"name": "f", "arguments": '{"x": 1}'}},
        {"function": {"name": "g", "arguments": {"y": 2}}},
        {"function": {"name": "h", "arguments": "not json"}},
        {"function": {"name": "i", "arguments": "[1, 2]"}},
    ])
    assert out[0] == {"id": "a", "name": "f", "arguments": {"x": 1}, "malformed_args": False}
    assert out[1]["id"] == "call_1" and out[1]["arguments"] == {"y": 2}
    assert out[2]["malformed_args"] and out[3]["malformed_args"]
