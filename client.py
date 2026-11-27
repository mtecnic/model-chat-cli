"""API client for communicating with AI model servers.

All streaming goes through one pipeline:

    HTTP bytes -> _iter_lines -> iter_openai_events / iter_ollama_events
               -> ("reasoning" | "content", text) events
               -> wrap_think -> text with <think>...</think> around reasoning

The rest of the app (think_parser.py) consumes the <think>-tagged text.
"""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, AsyncIterator, Dict, Optional, Tuple
import httpx
import json
import time


# One timeout policy for every request. `read` is the max gap between bytes,
# so it must cover long prompt prefill on big models before the first token.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0)

REASONING = "reasoning"
CONTENT = "content"


class ModelServerError(Exception):
    """The model server returned an error (HTTP status or in-stream error)."""

    def __init__(self, message: str, status: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class ChatMetrics:
    """Metrics extracted from an API response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_eval_duration_ns: int = 0   # Ollama-specific prefill timing
    eval_duration_ns: int = 0          # Ollama-specific decode timing
    ttft: float = 0.0                  # Time to first token of any kind (seconds)


@asynccontextmanager
async def _http_session(http_client: Optional[httpx.AsyncClient]):
    """Yield an httpx.AsyncClient: the caller-provided one, or a one-shot.

    When the caller supplies a client, we must NOT close it (connection pool
    must survive across requests). When we create our own, close it on exit.
    """
    if http_client is not None:
        yield http_client
    else:
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            yield client


# --------------------------------------------------------------------- #
# Error helpers
# --------------------------------------------------------------------- #
def _error_text(err) -> str:
    """Pull a readable message out of an error payload (str or dict)."""
    if isinstance(err, dict):
        return str(err.get("message") or err.get("error") or err.get("detail") or err)
    return str(err)


def _http_error(status: int, body: str) -> ModelServerError:
    """Build a ModelServerError from an HTTP error response body."""
    detail = body.strip()
    try:
        data = json.loads(body)
        if isinstance(data, dict):
            detail = _error_text(data.get("error") or data.get("detail") or data.get("message") or data)
    except (json.JSONDecodeError, ValueError):
        pass
    if len(detail) > 300:
        detail = detail[:300] + "..."
    return ModelServerError(f"HTTP {status}: {detail or 'no details'}", status=status, body=body)


# --------------------------------------------------------------------- #
# Stream parsing (pure: takes an async byte iterator, testable offline)
# --------------------------------------------------------------------- #
async def _iter_lines(byte_iter: AsyncIterator[bytes]) -> AsyncGenerator[str, None]:
    """Split a byte stream into decoded, stripped lines (incl. a trailing partial)."""
    buffer = b""
    async for chunk in byte_iter:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            yield line.decode("utf-8", errors="ignore").strip()
    if buffer.strip():
        yield buffer.decode("utf-8", errors="ignore").strip()


def _mark_first_token(metrics: ChatMetrics, request_start: float) -> None:
    if metrics.ttft == 0.0:
        metrics.ttft = max(time.monotonic() - request_start, 1e-6)


async def iter_openai_events(
    byte_iter: AsyncIterator[bytes], metrics: ChatMetrics, request_start: float
) -> AsyncGenerator[Tuple[str, str], None]:
    """Parse an OpenAI-compatible SSE stream into (kind, text) events."""
    async for line in _iter_lines(byte_iter):
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            raise ModelServerError(_error_text(chunk["error"]))

        # Final usage chunk (stream_options.include_usage) has no choices
        usage = chunk.get("usage")
        if usage:
            metrics.prompt_tokens = usage.get("prompt_tokens") or 0
            metrics.completion_tokens = usage.get("completion_tokens") or 0

        choices = chunk.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta") or {}

        # Reasoning arrives as `reasoning_content` (vLLM --reasoning-parser,
        # HF-style) or `reasoning` (NVIDIA/TensorRT-LLM serving).
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if reasoning:
            _mark_first_token(metrics, request_start)
            yield REASONING, reasoning

        content = delta.get("content") or ""
        if content:
            _mark_first_token(metrics, request_start)
            yield CONTENT, content


async def iter_ollama_events(
    byte_iter: AsyncIterator[bytes], metrics: ChatMetrics, request_start: float
) -> AsyncGenerator[Tuple[str, str], None]:
    """Parse an Ollama NDJSON stream into (kind, text) events."""
    async for line in _iter_lines(byte_iter):
        if not line:
            continue
        try:
            chunk = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            raise ModelServerError(_error_text(chunk["error"]))

        msg = chunk.get("message") or {}
        thinking = msg.get("thinking") or ""
        if thinking:
            _mark_first_token(metrics, request_start)
            yield REASONING, thinking
        content = msg.get("content") or ""
        if content:
            _mark_first_token(metrics, request_start)
            yield CONTENT, content

        if chunk.get("done"):
            metrics.completion_tokens = chunk.get("eval_count") or 0
            metrics.prompt_tokens = chunk.get("prompt_eval_count") or 0
            metrics.eval_duration_ns = chunk.get("eval_duration") or 0
            metrics.prompt_eval_duration_ns = chunk.get("prompt_eval_duration") or 0
            return


async def wrap_think(events: AsyncIterator[Tuple[str, str]]) -> AsyncGenerator[str, None]:
    """Turn (kind, text) events into text with <think>...</think> around reasoning."""
    thinking_active = False
    async for kind, text in events:
        if kind == REASONING:
            if not thinking_active:
                thinking_active = True
                yield "<think>"
            yield text
        else:
            if thinking_active:
                thinking_active = False
                yield "</think>"
            yield text
    if thinking_active:
        yield "</think>"


def _parse_tool_args(args_raw) -> Tuple[dict, bool]:
    """Normalize tool-call arguments to a dict. Returns (args, malformed)."""
    if isinstance(args_raw, dict):
        return args_raw, False
    if args_raw is None:
        return {}, False
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except Exception:
            return {}, True
        return (args, False) if isinstance(args, dict) else ({}, True)
    return {}, True


def _normalize_tool_calls(raw_calls: list) -> list:
    """Normalize server tool_calls to [{id, name, arguments, malformed_args}]."""
    normalized = []
    for i, tc in enumerate(raw_calls or []):
        fn = tc.get("function") or {}
        args, malformed = _parse_tool_args(fn.get("arguments"))
        normalized.append({
            "id": tc.get("id") or f"call_{i}",
            "name": fn.get("name") or "",
            "arguments": args,
            "malformed_args": malformed,
        })
    return normalized


class ModelClient:
    """Client for interacting with AI model servers."""

    def __init__(self, server: Dict, model: str):
        self.server = server
        self.model = model
        self.url = server["url"]
        self.server_type = server["type"]
        self.last_metrics: ChatMetrics = ChatMetrics()  # populated by chat_stream

    # ----------------------------------------------------------------- #
    # Streaming core
    # ----------------------------------------------------------------- #
    @staticmethod
    def _build_messages(message: Optional[str], history: Optional[list]) -> list:
        messages = list(history or [])
        if message:
            messages.append({"role": "user", "content": message})
        return messages

    def _build_request(self, messages: list, enable_thinking: Optional[bool],
                       max_tokens: Optional[int]):
        """Return (url, payload, parser, retry_rule) for this server type.

        retry_rule(payload, error_body) -> bool: if the server rejected an
        optional field, strip it from payload and return True to retry once.
        """
        if self.server_type == "openai":
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            # Thinking toggle for models that support it (e.g. Qwen 3/3.5 on vLLM)
            if enable_thinking is not None:
                payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

            def retry_rule(p: dict, body: str) -> bool:
                # Only retry when the error actually names the field — a 404 is
                # usually an unknown model and must surface, not be retried.
                if "stream_options" in p and "stream_options" in body:
                    p.pop("stream_options")
                    return True
                return False

            return f"{self.url}/v1/chat/completions", payload, iter_openai_events, retry_rule

        # Ollama
        payload = {"model": self.model, "messages": messages, "stream": True}
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}
        # Ollama's thinking switch is the top-level `think` field
        if enable_thinking is not None:
            payload["think"] = bool(enable_thinking)

        def retry_rule(p: dict, body: str) -> bool:
            # Non-thinking models reject `think` ("... does not support thinking")
            if "think" in p and "think" in body.lower():
                p.pop("think")
                return True
            return False

        return f"{self.url}/api/chat", payload, iter_ollama_events, retry_rule

    async def _stream_events(
        self, messages: list, metrics: ChatMetrics,
        enable_thinking: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """Stream (kind, text) events. Raises ModelServerError / httpx errors.

        Retries at most once, and only when the server rejected an optional
        field before streaming began. Network errors are never retried.
        """
        url, payload, parser, retry_rule = self._build_request(messages, enable_thinking, max_tokens)
        request_start = time.monotonic()

        async with _http_session(http_client) as client:
            for attempt in range(2):
                async with client.stream("POST", url, json=payload, timeout=DEFAULT_TIMEOUT) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode("utf-8", errors="ignore")
                        if attempt == 0 and response.status_code < 500 and retry_rule(payload, body):
                            continue
                        raise _http_error(response.status_code, body)
                    async for event in parser(response.aiter_bytes(), metrics, request_start):
                        yield event
                    return

    async def chat_stream(
        self, message: Optional[str], history: Optional[list] = None,
        enable_thinking: Optional[bool] = None
    ) -> AsyncGenerator[str, None]:
        """Stream a reply as text, with reasoning wrapped in <think> tags.

        Populates self.last_metrics as the stream progresses.
        """
        self.last_metrics = ChatMetrics()
        events = self._stream_events(
            self._build_messages(message, history), self.last_metrics,
            enable_thinking=enable_thinking,
        )
        async for text in wrap_think(events):
            yield text

    async def chat(
        self, message: Optional[str], history: Optional[list] = None,
        enable_thinking: Optional[bool] = None
    ) -> str:
        """Send a chat message and return the full response."""
        parts = []
        async for chunk in self.chat_stream(message, history, enable_thinking):
            parts.append(chunk)
        return "".join(parts)

    async def chat_with_metrics(
        self, message: Optional[str], history: Optional[list] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ) -> Tuple[str, ChatMetrics]:
        """Send a chat message and return the full response with API metrics.

        Streams internally to capture TTFT; token counts come from the API
        (OpenAI usage / Ollama eval_count) and stay 0 when not reported.
        Safe for concurrent use: metrics are per call, not self.last_metrics.

        If `http_client` is provided, it is reused (keepalive / connection
        pool preserved across calls). Otherwise a one-shot client is created.
        """
        metrics = ChatMetrics()
        events = self._stream_events(
            self._build_messages(message, history), metrics,
            enable_thinking=enable_thinking, max_tokens=max_tokens,
            http_client=http_client,
        )
        parts = []
        async for text in wrap_think(events):
            parts.append(text)
        return "".join(parts), metrics

    # ----------------------------------------------------------------- #
    # Tool calling (non-streaming) — supports agentic loops
    # ----------------------------------------------------------------- #
    def make_assistant_tool_msg(self, content: str, tool_calls: list) -> dict:
        """Build an assistant message containing tool_calls in server-native format.

        ``tool_calls`` is the normalized list returned by ``chat_with_tools``:
        ``[{"id": str, "name": str, "arguments": dict}, ...]``.
        """
        if self.server_type == "openai":
            return {
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {
                        "id": tc.get("id") or f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments") or {}),
                        },
                    }
                    for i, tc in enumerate(tool_calls)
                ],
            }
        # Ollama
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {"function": {"name": tc["name"], "arguments": tc.get("arguments") or {}}}
                for tc in tool_calls
            ],
        }

    def make_tool_result_msg(self, tool_call_id: str, result: str) -> dict:
        """Build a tool-result message in server-native format."""
        if self.server_type == "openai":
            return {
                "role": "tool",
                "tool_call_id": tool_call_id or "call_0",
                "content": str(result),
            }
        return {"role": "tool", "content": str(result)}

    async def chat_with_tools(
        self,
        messages: list,
        tools: list,
        max_tokens: Optional[int] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> Tuple[str, list, ChatMetrics]:
        """Non-streaming chat with tool support.

        Returns ``(content, tool_calls, metrics)``. ``tool_calls`` is normalized to
        ``[{"id": str, "name": str, "arguments": dict}, ...]`` regardless of server.
        ``content`` may be empty when the assistant only emits tool calls.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }
        if self.server_type == "openai":
            url = f"{self.url}/v1/chat/completions"
            payload["tool_choice"] = "auto"
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
        else:
            url = f"{self.url}/api/chat"
            if max_tokens is not None:
                payload["options"] = {"num_predict": max_tokens}

        request_start = time.monotonic()
        async with _http_session(http_client) as client:
            r = await client.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
            if r.status_code >= 400:
                raise _http_error(r.status_code, r.text)
            body = r.json()

        metrics = ChatMetrics()
        metrics.ttft = time.monotonic() - request_start  # non-streaming → TTFT == full

        if self.server_type == "openai":
            usage = body.get("usage") or {}
            metrics.prompt_tokens = usage.get("prompt_tokens") or 0
            metrics.completion_tokens = usage.get("completion_tokens") or 0
            msg = (body.get("choices") or [{}])[0].get("message") or {}
        else:
            if body.get("error"):
                raise ModelServerError(_error_text(body["error"]))
            metrics.completion_tokens = body.get("eval_count") or 0
            metrics.prompt_tokens = body.get("prompt_eval_count") or 0
            metrics.eval_duration_ns = body.get("eval_duration") or 0
            metrics.prompt_eval_duration_ns = body.get("prompt_eval_duration") or 0
            msg = body.get("message") or {}

        content = msg.get("content") or ""
        return content, _normalize_tool_calls(msg.get("tool_calls")), metrics
