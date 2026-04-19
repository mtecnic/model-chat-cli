"""API client for communicating with AI model servers."""
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, Optional, Tuple
import httpx
import json
import time


@asynccontextmanager
async def _http_session(http_client: Optional[httpx.AsyncClient]):
    """Yield an httpx.AsyncClient: the caller-provided one, or a one-shot.

    When the caller supplies a client, we must NOT close it (connection pool
    must survive across requests). When we create our own, close it on exit.
    """
    if http_client is not None:
        yield http_client
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            yield client


@dataclass
class ChatMetrics:
    """Metrics extracted from an API response."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_eval_duration_ns: int = 0   # Ollama-specific prefill timing
    eval_duration_ns: int = 0          # Ollama-specific decode timing
    ttft: float = 0.0                  # Time to first content token (seconds)


class ModelClient:
    """Client for interacting with AI model servers."""

    def __init__(self, server: Dict, model: str):
        self.server = server
        self.model = model
        self.url = server["url"]
        self.server_type = server["type"]
        self.last_metrics: ChatMetrics = ChatMetrics()  # populated by chat_stream

    async def chat_stream(
        self, message: str, history: Optional[list] = None,
        enable_thinking: Optional[bool] = None
    ) -> AsyncGenerator[str, None]:
        """Send a chat message and stream the response."""
        if self.server_type == "openai":
            async for chunk in self._chat_stream_openai(message, history, enable_thinking):
                yield chunk
        else:  # ollama
            async for chunk in self._chat_stream_ollama(message, history, enable_thinking):
                yield chunk

    async def _chat_stream_openai(
        self, message: str, history: Optional[list] = None,
        enable_thinking: Optional[bool] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat using OpenAI-compatible API."""
        # Create a copy of history to avoid mutating the original
        messages = (history or []).copy()
        if message:
            messages.append({"role": "user", "content": message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        # Pass thinking toggle for models that support it (e.g. Qwen 3/3.5 on vLLM)
        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.url}/v1/chat/completions",
                json=payload,
            ) as response:
                # Use aiter_bytes to avoid line buffering for real-time streaming
                buffer = b""
                thinking_active = False
                async for chunk_bytes in response.aiter_bytes(chunk_size=64):
                    buffer += chunk_bytes

                    # Process all complete lines in buffer
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode('utf-8', errors='ignore').strip()

                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                if thinking_active:
                                    yield "</think>"
                                return

                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})

                                # Handle reasoning_content (vLLM/Qwen thinking field)
                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    if not thinking_active:
                                        yield "<think>"
                                        thinking_active = True
                                    yield reasoning

                                content = delta.get("content", "")
                                if content:
                                    if thinking_active:
                                        yield "</think>"
                                        thinking_active = False
                                    yield content
                            except json.JSONDecodeError:
                                continue

    async def _chat_stream_ollama(
        self, message: str, history: Optional[list] = None,
        enable_thinking: Optional[bool] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat using Ollama API. Populates self.last_metrics on completion."""
        # Create a copy of history to avoid mutating the original
        messages = (history or []).copy()
        if message:
            messages.append({"role": "user", "content": message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        # Pass thinking toggle for Ollama models that support it
        if enable_thinking is not None:
            payload["options"] = payload.get("options", {})
            payload["options"]["enable_thinking"] = enable_thinking

        self.last_metrics = ChatMetrics()

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{self.url}/api/chat",
                json=payload,
            ) as response:
                # Use aiter_bytes to avoid line buffering for real-time streaming
                buffer = b""
                async for chunk_bytes in response.aiter_bytes(chunk_size=64):
                    buffer += chunk_bytes

                    # Process all complete lines in buffer (Ollama sends NDJSON)
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode('utf-8', errors='ignore').strip()

                        if line:
                            try:
                                chunk = json.loads(line)

                                # Capture metrics from the final done chunk
                                if chunk.get("done"):
                                    self.last_metrics.completion_tokens = chunk.get("eval_count", 0)
                                    self.last_metrics.prompt_tokens = chunk.get("prompt_eval_count", 0)
                                    self.last_metrics.eval_duration_ns = chunk.get("eval_duration", 0)
                                    self.last_metrics.prompt_eval_duration_ns = chunk.get("prompt_eval_duration", 0)
                                    break

                                content = chunk.get("message", {}).get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue

    async def chat(
        self, message: str, history: Optional[list] = None,
        enable_thinking: Optional[bool] = None
    ) -> str:
        """Send a chat message and return the full response."""
        full_response = ""
        async for chunk in self.chat_stream(message, history, enable_thinking):
            full_response += chunk
        return full_response

    async def chat_with_metrics(
        self, message: str, history: Optional[list] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ) -> Tuple[str, ChatMetrics]:
        """Send a chat message and return the full response with API metrics.

        Streams internally to capture TTFT, then extracts real token counts
        from the API response metadata (Ollama eval_count / OpenAI usage).
        Falls back gracefully when the server doesn't report metrics.

        If `http_client` is provided, it is reused (keepalive / connection
        pool preserved across calls). Otherwise a one-shot client is created.
        """
        if self.server_type == "openai":
            return await self._chat_metrics_openai(message, history, max_tokens, enable_thinking, http_client)
        else:
            return await self._chat_metrics_ollama(message, history, max_tokens, enable_thinking, http_client)

    async def _chat_metrics_openai(
        self, message: str, history: Optional[list] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ) -> Tuple[str, ChatMetrics]:
        """Stream via OpenAI-compatible API, extracting usage metrics."""
        messages = (history or []).copy()
        if message:
            messages.append({"role": "user", "content": message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        metrics = ChatMetrics()
        full_response = ""
        request_start = time.monotonic()
        first_content = True
        thinking_active = False

        try:
            async with _http_session(http_client) as client:
                async with client.stream(
                    "POST",
                    f"{self.url}/v1/chat/completions",
                    json=payload,
                    timeout=300.0,
                ) as response:
                    # Check for server error (stream_options not supported)
                    if response.status_code >= 400:
                        raise httpx.HTTPStatusError(
                            f"HTTP {response.status_code}",
                            request=response.request, response=response
                        )

                    buffer = b""
                    async for chunk_bytes in response.aiter_bytes(chunk_size=64):
                        buffer += chunk_bytes

                        while b"\n" in buffer:
                            line_bytes, buffer = buffer.split(b"\n", 1)
                            line = line_bytes.decode('utf-8', errors='ignore').strip()

                            if not line.startswith("data: "):
                                continue

                            data = line[6:]
                            if data == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data)

                                # Extract usage from final chunk if present
                                usage = chunk.get("usage")
                                if usage:
                                    metrics.prompt_tokens = usage.get("prompt_tokens", 0)
                                    metrics.completion_tokens = usage.get("completion_tokens", 0)

                                # Usage-only chunks have "choices": [] — skip delta parsing
                                choices = chunk.get("choices", [])
                                if not choices:
                                    continue
                                delta = choices[0].get("delta", {})

                                reasoning = delta.get("reasoning_content", "")
                                if reasoning:
                                    if not thinking_active:
                                        thinking_active = True
                                        full_response += "<think>"
                                    full_response += reasoning

                                content = delta.get("content", "")
                                if content:
                                    if thinking_active:
                                        thinking_active = False
                                        full_response += "</think>"
                                    if first_content:
                                        metrics.ttft = time.monotonic() - request_start
                                        first_content = False
                                    full_response += content
                            except json.JSONDecodeError:
                                continue

        except (httpx.HTTPStatusError, httpx.RequestError):
            # stream_options not supported — retry without it
            return await self._chat_metrics_openai_fallback(
                message, history, max_tokens, enable_thinking, http_client
            )

        if thinking_active:
            full_response += "</think>"

        return full_response, metrics

    async def _chat_metrics_openai_fallback(
        self, message: str, history: Optional[list] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ) -> Tuple[str, ChatMetrics]:
        """Fallback for servers that reject stream_options."""
        messages = (history or []).copy()
        if message:
            messages.append({"role": "user", "content": message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        metrics = ChatMetrics()
        full_response = ""
        request_start = time.monotonic()
        first_content = True
        thinking_active = False

        async with _http_session(http_client) as client:
            async with client.stream(
                "POST",
                f"{self.url}/v1/chat/completions",
                json=payload,
                timeout=300.0,
            ) as response:
                buffer = b""
                async for chunk_bytes in response.aiter_bytes(chunk_size=64):
                    buffer += chunk_bytes

                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode('utf-8', errors='ignore').strip()

                        if not line.startswith("data: "):
                            continue

                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})

                            reasoning = delta.get("reasoning_content", "")
                            if reasoning:
                                if not thinking_active:
                                    thinking_active = True
                                    full_response += "<think>"
                                full_response += reasoning

                            content = delta.get("content", "")
                            if content:
                                if thinking_active:
                                    thinking_active = False
                                    full_response += "</think>"
                                if first_content:
                                    metrics.ttft = time.monotonic() - request_start
                                    first_content = False
                                full_response += content
                        except json.JSONDecodeError:
                            continue

        if thinking_active:
            full_response += "</think>"

        # No usage data available — metrics.completion_tokens stays 0, caller falls back
        return full_response, metrics

    async def _chat_metrics_ollama(
        self, message: str, history: Optional[list] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        http_client: Optional[httpx.AsyncClient] = None
    ) -> Tuple[str, ChatMetrics]:
        """Stream via Ollama API, extracting eval_count / eval_duration."""
        messages = (history or []).copy()
        if message:
            messages.append({"role": "user", "content": message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        options = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if enable_thinking is not None:
            options["enable_thinking"] = enable_thinking
        if options:
            payload["options"] = options

        metrics = ChatMetrics()
        full_response = ""
        request_start = time.monotonic()
        first_content = True

        async with _http_session(http_client) as client:
            async with client.stream(
                "POST",
                f"{self.url}/api/chat",
                json=payload,
                timeout=300.0,
            ) as response:
                buffer = b""
                async for chunk_bytes in response.aiter_bytes(chunk_size=64):
                    buffer += chunk_bytes

                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode('utf-8', errors='ignore').strip()

                        if not line:
                            continue

                        try:
                            chunk = json.loads(line)

                            # Final chunk with done=true has timing/count data
                            if chunk.get("done"):
                                metrics.completion_tokens = chunk.get("eval_count", 0)
                                metrics.prompt_tokens = chunk.get("prompt_eval_count", 0)
                                metrics.eval_duration_ns = chunk.get("eval_duration", 0)
                                metrics.prompt_eval_duration_ns = chunk.get("prompt_eval_duration", 0)
                                break

                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                if first_content:
                                    metrics.ttft = time.monotonic() - request_start
                                    first_content = False
                                full_response += content
                        except json.JSONDecodeError:
                            continue

        return full_response, metrics
