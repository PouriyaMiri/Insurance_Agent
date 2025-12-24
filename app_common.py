from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List
from rich.console import Console
from rich.panel import Panel

from dialogue import SessionState
from rag import RAGIndex

console = Console()

EXIT_PHRASES: List[str] = [
    "hang up",
    "goodbye",
    "bye",
    "exit",
    "quit",
    "stop",
    "end",
    "terminate",
    "close",
    "disconnect",
]


@dataclass(frozen=True)
class AgentContext:
    rag: RAGIndex
    state: SessionState


def build_context(docs_path: str = "./docs") -> AgentContext:
    rag = RAGIndex()
    rag.build_from_folder(docs_path)
    state = SessionState()
    return AgentContext(rag=rag, state=state)


def print_agent(text: str, title: str = "Agent") -> str:
    cleaned = (text or "").strip()
    if cleaned:
        console.print(Panel(cleaned, title=title))
    return cleaned


def is_exit_phrase(user_text: str, phrases: Iterable[str] = EXIT_PHRASES) -> bool:
    t = (user_text or "").strip().lower()
    return t in set(p.lower() for p in phrases)
