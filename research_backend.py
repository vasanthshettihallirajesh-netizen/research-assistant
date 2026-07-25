"""
research_backend.py — abstract interface for source-fetching backends.

The API currently expects sources to be logged manually (via
POST /topics/{id}/source). This module defines the interface a real
backend (web search, internal document search, etc.) would implement
to plug directly into the pipeline instead — so swapping in real
automated research later doesn't require touching the API or DB layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SourceResult:
    url: Optional[str]
    title: Optional[str]
    excerpt: str
    stance: str  # "supports" | "contradicts" | "neutral"
    reliability_note: Optional[str] = None


class ResearchBackend(ABC):
    """Implement this against a real search/retrieval tool to make the
    pipeline actually fetch sources instead of requiring manual entry."""

    @abstractmethod
    def search(self, sub_question: str, max_results: int = 5) -> List[SourceResult]:
        """Given a sub-question, return candidate sources with a stance
        judgment relative to the sub-question."""
        raise NotImplementedError


class ManualEntryBackend(ResearchBackend):
    """Default no-op backend: returns nothing, since sources are expected
    to be logged manually via the API. Exists so the interface always has
    a concrete, safe default implementation."""

    def search(self, sub_question: str, max_results: int = 5) -> List[SourceResult]:
        return []


class StaticFixtureBackend(ResearchBackend):
    """Useful for tests/demos: returns pre-defined canned results instead
    of hitting a real network call. Register fixtures by sub-question
    text (case-insensitive substring match)."""

    def __init__(self):
        self._fixtures = {}

    def register(self, sub_question_substring: str, results: List[SourceResult]):
        self._fixtures[sub_question_substring.lower()] = results

    def search(self, sub_question: str, max_results: int = 5) -> List[SourceResult]:
        q_lower = sub_question.lower()
        for key, results in self._fixtures.items():
            if key in q_lower:
                return results[:max_results]
        return []
