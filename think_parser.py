"""Parser for <think>...</think> tags in model responses.

Handles both streaming (chunk-by-chunk with tags split across boundaries)
and non-streaming (complete string) use cases.
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator


class ChunkType(Enum):
    THINKING = "thinking"
    CONTENT = "content"


@dataclass
class ParsedChunk:
    """A chunk of streaming text with its type classification."""
    text: str
    chunk_type: ChunkType


@dataclass
class ParsedResponse:
    """A fully parsed response split into thinking and content portions."""
    thinking: str
    content: str
    raw: str


class ThinkTagParser:
    """Stateful parser that classifies streaming chunks as thinking or content.

    Handles <think>/<​/think> tags split across arbitrary chunk boundaries.
    Maximum holdback is 7 characters (len("</think>") - 1).
    """

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self, start_thinking: bool = False):
        """Initialize parser.

        Args:
            start_thinking: If True, assume the model is already thinking
                (for models/templates where <think> is a prompt prefix, not
                part of the generated output). Parser starts in THINKING state
                and waits for </think> to transition to CONTENT.
        """
        self._inside_think: bool = start_thinking
        self._start_thinking: bool = start_thinking
        self._saw_close_tag: bool = False
        self._buffer: str = ""

    def feed(self, chunk: str) -> list[ParsedChunk]:
        """Feed a raw chunk and return zero or more typed chunks.

        May return empty list if the chunk is buffered awaiting tag completion.
        May return multiple chunks if a tag boundary falls within the chunk.
        """
        self._buffer += chunk
        return self._process_buffer()

    def flush(self) -> list[ParsedChunk]:
        """Flush remaining buffer at end of stream."""
        result = []
        if self._buffer:
            # If start_thinking was set but we never saw </think>,
            # the model didn't actually think — emit as content
            if self._inside_think and self._start_thinking and not self._saw_close_tag:
                result.append(ParsedChunk(text=self._buffer, chunk_type=ChunkType.CONTENT))
            else:
                chunk_type = ChunkType.THINKING if self._inside_think else ChunkType.CONTENT
                result.append(ParsedChunk(text=self._buffer, chunk_type=chunk_type))
            self._buffer = ""
        return result

    def _process_buffer(self) -> list[ParsedChunk]:
        """Scan buffer for tags, emit typed chunks, handle partial tags."""
        results = []

        while self._buffer:
            if self._inside_think:
                tag = self.CLOSE_TAG
                pos = self._buffer.find(tag)

                # Strip redundant <think> when already in thinking mode
                open_pos = self._buffer.find(self.OPEN_TAG)
                if open_pos >= 0 and (pos < 0 or open_pos < pos):
                    before = self._buffer[:open_pos]
                    if before:
                        results.append(ParsedChunk(text=before, chunk_type=ChunkType.THINKING))
                    self._buffer = self._buffer[open_pos + len(self.OPEN_TAG):]
                    continue
            else:
                # In CONTENT mode, look for both <think> and orphaned </think>
                open_pos = self._buffer.find(self.OPEN_TAG)
                close_pos = self._buffer.find(self.CLOSE_TAG)

                if close_pos >= 0 and (open_pos < 0 or close_pos < open_pos):
                    # Orphaned </think> — strip it silently
                    self._saw_close_tag = True
                    before = self._buffer[:close_pos]
                    if before:
                        results.append(ParsedChunk(text=before, chunk_type=ChunkType.CONTENT))
                    self._buffer = self._buffer[close_pos + len(self.CLOSE_TAG):]
                    continue

                tag = self.OPEN_TAG
                pos = open_pos

            if pos >= 0:
                # Found a complete tag
                before = self._buffer[:pos]
                if before:
                    chunk_type = ChunkType.THINKING if self._inside_think else ChunkType.CONTENT
                    results.append(ParsedChunk(text=before, chunk_type=chunk_type))

                # Toggle state and consume the tag
                if self._inside_think:
                    self._saw_close_tag = True
                self._inside_think = not self._inside_think
                self._buffer = self._buffer[pos + len(tag):]
                # Continue scanning remainder
            else:
                # No complete tag — check for partial tag at the tail
                # Check both tags when in CONTENT mode
                holdback = self._partial_tag_length(self._buffer, tag)
                if not self._inside_think:
                    holdback = max(holdback,
                                   self._partial_tag_length(self._buffer, self.CLOSE_TAG))
                if holdback > 0:
                    emittable = self._buffer[:-holdback]
                    if emittable:
                        chunk_type = ChunkType.THINKING if self._inside_think else ChunkType.CONTENT
                        results.append(ParsedChunk(text=emittable, chunk_type=chunk_type))
                    self._buffer = self._buffer[-holdback:]
                else:
                    # No partial tag — emit everything
                    chunk_type = ChunkType.THINKING if self._inside_think else ChunkType.CONTENT
                    results.append(ParsedChunk(text=self._buffer, chunk_type=chunk_type))
                    self._buffer = ""
                break

        return results

    @staticmethod
    def _partial_tag_length(text: str, tag: str) -> int:
        """Return length of the longest suffix of text that is a prefix of tag.
        Returns 0 if no such suffix exists."""
        max_check = min(len(text), len(tag) - 1)
        for length in range(max_check, 0, -1):
            if text[-length:] == tag[:length]:
                return length
        return 0


async def parsed_chat_stream(
    raw_stream: AsyncGenerator[str, None],
    start_thinking: bool = False
) -> AsyncGenerator[ParsedChunk, None]:
    """Wrap a raw chat stream, yielding typed ParsedChunk objects.

    Args:
        raw_stream: Raw string chunks from chat_stream().
        start_thinking: If True, assume model is already in thinking mode
            (for models where <think> is a template prefix, not generated).
    """
    parser = ThinkTagParser(start_thinking=start_thinking)
    async for raw_chunk in raw_stream:
        for parsed in parser.feed(raw_chunk):
            yield parsed
    for parsed in parser.flush():
        yield parsed


_THINK_PATTERN = re.compile(r'<think>(.*?)</think>', re.DOTALL)


def split_thinking(text: str) -> ParsedResponse:
    """Split a complete response string into thinking and content portions.

    Handles: no think block, think at start, think in middle,
    multiple think blocks, unclosed think tags, and the common case
    where <think> is a template prefix (only </think> in the output).
    """
    thinking_parts = []
    content_parts = []

    # Check for </think> without preceding <think> (template-prefixed thinking)
    first_close = text.find('</think>')
    first_open = text.find('<think>')
    if first_close >= 0 and (first_open < 0 or first_close < first_open):
        # Everything before </think> is thinking, rest is content
        thinking_parts.append(text[:first_close])
        remainder = text[first_close + len('</think>'):]
        # Process remainder for any additional <think>...</think> blocks
        for match in _THINK_PATTERN.finditer(remainder):
            thinking_parts.append(match.group(1))
        # Content is remainder with all think blocks stripped
        content = _THINK_PATTERN.sub('', remainder).strip()
        content = content.replace('</think>', '')
        return ParsedResponse(
            thinking="\n".join(p.strip() for p in thinking_parts if p.strip()),
            content=content,
            raw=text
        )

    last_end = 0
    for match in _THINK_PATTERN.finditer(text):
        before = text[last_end:match.start()]
        if before:
            content_parts.append(before)
        thinking_parts.append(match.group(1))
        last_end = match.end()

    # Content after last think block
    remaining = text[last_end:]

    # Handle unclosed <think> at the end
    unclosed_pos = remaining.find('<think>')
    if unclosed_pos >= 0:
        content_parts.append(remaining[:unclosed_pos])
        thinking_parts.append(remaining[unclosed_pos + len('<think>'):])
    else:
        content_parts.append(remaining)

    content = "".join(content_parts).strip()
    # Strip orphaned </think> tags (closing tag with no opener)
    content = content.replace("</think>", "")

    return ParsedResponse(
        thinking="\n".join(p.strip() for p in thinking_parts if p.strip()),
        content=content.strip(),
        raw=text
    )


def strip_thinking(text: str) -> str:
    """Remove all <think>...</think> blocks from text."""
    return split_thinking(text).content
