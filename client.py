"""API client for communicating with AI model servers."""
from typing import AsyncGenerator, Dict, Optional
import httpx
import json


class ModelClient:
    """Client for interacting with AI model servers."""

    def __init__(self, server: Dict, model: str):
        self.server = server
        self.model = model
        self.url = server["url"]
        self.server_type = server["type"]

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
                async for chunk_bytes in response.aiter_bytes(chunk_size=64):
                    buffer += chunk_bytes

                    # Process all complete lines in buffer
                    while b"\n" in buffer:
                        line_bytes, buffer = buffer.split(b"\n", 1)
                        line = line_bytes.decode('utf-8', errors='ignore').strip()

                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                return

                            try:
                                chunk = json.loads(data)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue

    async def _chat_stream_ollama(
        self, message: str, history: Optional[list] = None,
        enable_thinking: Optional[bool] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat using Ollama API."""
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
                                content = chunk.get("message", {}).get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue

    async def chat(self, message: str, history: Optional[list] = None) -> str:
        """Send a chat message and return the full response."""
        full_response = ""
        async for chunk in self.chat_stream(message, history):
            full_response += chunk
        return full_response
