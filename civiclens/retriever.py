"""Deterministic local retrieval over original catalogue records."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_WORD = re.compile(r"\w+", re.UNICODE)


def _flatten_text(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        result = []
        for item in value.values():
            result.extend(_flatten_text(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_text(item))
        return result
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return [str(value)]
    return []


def retrieve_relevant_schemes(
    query: str,
    user_profile: Mapping[str, Any] | None,
    schemes: Sequence[Mapping[str, Any]],
    top_k: int = 5,
) -> list[Mapping[str, Any]]:
    """Rank query-relevant records while returning them unchanged."""
    if top_k <= 0:
        return []
    terms = set(_WORD.findall(query.casefold())) if isinstance(query, str) else set()
    profile = user_profile if isinstance(user_profile, Mapping) else {}
    ranked = []
    for index, scheme in enumerate(schemes):
        fields = _flatten_text(scheme)
        searchable = " ".join(fields).casefold()
        name = str(scheme.get("name", "")).casefold()
        name_terms = set(_WORD.findall(name))
        score = sum(searchable.count(term) for term in sorted(terms))
        if terms and name_terms and terms <= name_terms:
            score += 100
        elif terms and name_terms and name_terms <= terms:
            score += 100
        jurisdiction = scheme.get("jurisdiction")
        state = profile.get("state")
        if isinstance(state, str) and isinstance(jurisdiction, str) and state.strip().casefold() == jurisdiction.strip().casefold():
            score += 5
        ranked.append((score, index, scheme))
    # Empty and unrelated queries remain deterministic and useful by preserving
    # catalogue order; exact and overlapping text still rank above that baseline.
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [scheme for _, _, scheme in ranked[:top_k]]


class Retriever:
    """Compatibility wrapper for the function-based retrieval API."""

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def retrieve(self, query, user_profile=None, top_k=5):
        return retrieve_relevant_schemes(query, user_profile, self.data_loader.load_schemes(), top_k)
