from __future__ import annotations

import re
from pathlib import Path

from app.core.config import get_settings


class FileValidator:
    @staticmethod
    def validate_filename(filename: str) -> tuple[bool, str]:
        if not filename or not filename.strip():
            return False, "Filename is required"
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension not in get_settings().allowed_extensions_list:
            return False, f"Unsupported file type: {extension}"
        return True, "OK"

    @staticmethod
    def validate_size(size_bytes: int) -> tuple[bool, str]:
        if size_bytes <= 0:
            return False, "Uploaded file is empty"
        if size_bytes > get_settings().max_file_size_bytes:
            return False, f"File exceeds {get_settings().max_file_size_mb} MB limit"
        return True, "OK"


class QuestionValidator:
    XSS_PATTERN = re.compile(r"<\s*script|javascript:|onerror\s*=", re.IGNORECASE)

    @classmethod
    def validate(cls, question: str) -> tuple[bool, str]:
        if not question or len(question.strip()) < 3:
            return False, "Question must contain at least 3 characters"
        if len(question) > 1000:
            return False, "Question must be 1000 characters or fewer"
        if cls.XSS_PATTERN.search(question):
            return False, "Question contains unsafe markup"
        return True, "OK"
