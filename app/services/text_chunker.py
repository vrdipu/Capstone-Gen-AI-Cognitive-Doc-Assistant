from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    word_count: int
    metadata: dict[str, Any]


class TextChunkingService:
    def __init__(self) -> None:
        settings = get_settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(self, text: str, document_id: str, filename: str, file_type: str) -> list[TextChunk]:
        chunks = [chunk.strip() for chunk in self.splitter.split_text(text) if chunk.strip()]
        result: list[TextChunk] = []
        cursor = 0
        for index, chunk in enumerate(chunks):
            start = max(0, text.find(chunk[:80], cursor))
            if start == -1:
                start = cursor
            end = start + len(chunk)
            cursor = max(start + 1, end - get_settings().chunk_overlap)
            digest = hashlib.sha256(f"{document_id}:{index}:{chunk}".encode("utf-8")).hexdigest()[:16]
            result.append(
                TextChunk(
                    chunk_id=f"{document_id}:{index}:{digest}",
                    document_id=document_id,
                    text=chunk,
                    chunk_index=index,
                    start_char=start,
                    end_char=end,
                    word_count=len(chunk.split()),
                    metadata={"filename": filename, "file_type": file_type},
                )
            )
        return result
