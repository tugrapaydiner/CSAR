"""Sentence-aware 50% overlapping chunker."""

from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import tiktoken
except ImportError:  # pragma: no cover - exercised only when dependency is absent.
    tiktoken = None


_SENTENCE_PATTERN = re.compile(r"\S.+?(?:[.!?]+(?=\s|$)|$)", re.DOTALL)


@dataclass(frozen=True)
class HalfBlock:
    """A contiguous sequence of complete sentences."""

    index: int
    sentences: tuple[str, ...]
    token_count: int

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass(frozen=True)
class Chunk:
    """A chunk covering one or two adjacent half-blocks."""

    index: int
    half_blocks: tuple[HalfBlock, ...]
    token_count: int

    @property
    def half_block_indices(self) -> tuple[int, ...]:
        return tuple(block.index for block in self.half_blocks)

    @property
    def text(self) -> str:
        return " ".join(block.text for block in self.half_blocks)


class OverlappingChunker:
    """Build chunks of two half-blocks, advanced by one half-block."""

    def __init__(self, half_block_size: int, *, model: str = "gpt-4o") -> None:
        if half_block_size <= 0:
            raise ValueError("half_block_size must be positive")
        if tiktoken is None:
            raise ImportError(
                "tiktoken is required for OverlappingChunker token counts. "
                "Install it with `pip install tiktoken`."
            )

        self.half_block_size = half_block_size
        self.model = model
        self.encoder = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def split_sentences(self, text: str) -> list[str]:
        stripped = text.strip()
        if not stripped:
            return []
        return [" ".join(match.group(0).split()) for match in _SENTENCE_PATTERN.finditer(stripped)]

    def build_half_blocks(self, text: str) -> list[HalfBlock]:
        sentences = self.split_sentences(text)
        blocks: list[HalfBlock] = []
        pending: list[str] = []

        def make_block(block_sentences: list[str]) -> HalfBlock:
            return HalfBlock(
                index=len(blocks),
                sentences=tuple(block_sentences),
                token_count=self.count_tokens(" ".join(block_sentences)),
            )

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if sentence_tokens > self.half_block_size:
                if pending:
                    blocks.append(make_block(pending))
                    pending = []

                blocks.append(
                    HalfBlock(
                        index=len(blocks),
                        sentences=(sentence,),
                        token_count=sentence_tokens,
                    )
                )
                continue

            candidate = [*pending, sentence]
            candidate_tokens = self.count_tokens(" ".join(candidate))
            if pending and candidate_tokens > self.half_block_size:
                blocks.append(make_block(pending))
                pending = []

            pending.append(sentence)

        if pending:
            blocks.append(make_block(pending))

        return blocks

    def chunk(self, text: str) -> list[Chunk]:
        half_blocks = self.build_half_blocks(text)
        if not half_blocks:
            return []
        if len(half_blocks) == 1:
            block = half_blocks[0]
            return [Chunk(index=0, half_blocks=(block,), token_count=block.token_count)]

        chunks: list[Chunk] = []
        for index in range(len(half_blocks) - 1):
            chunk_blocks = (half_blocks[index], half_blocks[index + 1])
            chunk_text = " ".join(block.text for block in chunk_blocks)
            chunks.append(
                Chunk(
                    index=index,
                    half_blocks=chunk_blocks,
                    token_count=self.count_tokens(chunk_text),
                )
            )
        return chunks
