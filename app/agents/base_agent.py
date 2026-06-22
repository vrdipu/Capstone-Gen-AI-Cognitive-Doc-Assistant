from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentResult:
    success: bool
    output: Any
    error: str | None = None


class Agent(Protocol):
    name: str

    def process(self, payload: dict[str, Any]) -> AgentResult:
        ...
