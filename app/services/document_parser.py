from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.ingestion import ParsedDocument, parse_uploaded_file


class DocumentIngestionService:
    """File-path ingestion facade matching the development guide."""

    def ingest(self, file_path: str) -> dict[str, Any]:
        document = parse_uploaded_file(file_path)
        return {
            "text": document.content,
            "metadata": {
                **document.metadata,
                "word_count": len(document.content.split()),
                "char_count": len(document.content),
                "filename": Path(file_path).name,
            },
        }


def parse_document(file_path: str) -> ParsedDocument:
    return parse_uploaded_file(file_path)
