"""End-to-end, safety-conscious scheme discovery pipeline."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any

from . import llm_client
from .action_planner import build_portfolio_action_plan
from .engine import evaluate_all
from .retriever import retrieve_relevant_schemes

PROFILE_FIELDS = (
    "state",
    "age",
    "education_level",
    "annual_family_income_inr",
    "institution_type",
    "course_type",
    "academic_year",
    "first_generation_student",
)
MAX_HISTORY_MESSAGES = 10
_URL = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)


def extract_profile_updates(user_message, current_profile, recent_history, **kwargs):
    """Call the LLM extraction boundary; kept patchable for isolated tests."""
    return llm_client.extract_profile_updates(user_message, current_profile, recent_history, **kwargs)


def generate_grounded_response(
    user_message,
    recent_history,
    profile,
    retrieved_schemes,
    evaluations,
    **kwargs,
):
    """Call the LLM answer boundary; kept patchable for isolated tests."""
    return llm_client.generate_grounded_response(
        user_message, recent_history, profile, retrieved_schemes, evaluations, **kwargs
    )


def _valid_profile_value(field: str, value: Any) -> bool:
    if value is None:
        return True
    if field == "age":
        return type(value) is int and 0 <= value <= 120
    if field == "annual_family_income_inr":
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and value >= 0
            and (not isinstance(value, float) or __import__("math").isfinite(value))
        )
    if field == "first_generation_student":
        return type(value) is bool
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 200


def _normalise_profile(profile: Any) -> dict[str, Any]:
    incoming = profile if isinstance(profile, Mapping) else {}
    return {
        field: incoming.get(field) if _valid_profile_value(field, incoming.get(field)) else None
        for field in PROFILE_FIELDS
    }


def _bounded_history(history: Any) -> list[dict[str, str]]:
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        return []
    safe = []
    for item in history:
        if isinstance(item, Mapping) and item.get("role") in ("user", "assistant") and isinstance(item.get("content"), str):
            safe.append({"role": item["role"], "content": item["content"][:2000]})
    return safe[-MAX_HISTORY_MESSAGES:]


def _safe_relations(source_schemes, catalogue):
    known = {
        scheme.get("scheme_id"): scheme
        for scheme in catalogue
        if isinstance(scheme, Mapping) and isinstance(scheme.get("scheme_id"), str)
    }
    relations = []
    seen = set()
    for scheme in source_schemes:
        rows = scheme.get("relationships", [])
        if not isinstance(rows, list):
            continue
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            target_id = raw.get("target_scheme_id")
            if target_id not in known:
                continue
            key = (scheme.get("scheme_id"), target_id)
            if key in seen:
                continue
            seen.add(key)
            relation = dict(raw)
            if relation.get("verification_status") != "verified":
                relation["relation_type"] = "unknown"
            relations.append(relation)
    return relations


def _unavailable_result(profile, message):
    return {
        "answer": message,
        "error": message,
        "updated_profile": profile,
        "profile_updates": {},
        "evaluations": [],
        "source_schemes": [],
        "action_plan": [],
        "relationships": [],
        "suggested_questions": [],
    }


def respond_to_message(
    user_message: str,
    history: Sequence[Mapping[str, Any]] | None,
    profile: Mapping[str, Any] | None,
    schemes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Extract explicit profile facts, evaluate catalogue matches, and answer.

    The model is allowed to propose text and source IDs only. Scheme records,
    profile updates, eligibility results, links, and action plans are rebuilt or
    selected deterministically by Python.
    """
    if not isinstance(user_message, str) or not user_message.strip():
        raise ValueError("A non-empty user message is required.")
    safe_profile = _normalise_profile(profile)
    safe_history = _bounded_history(history)
    catalogue = [copy.deepcopy(scheme) for scheme in schemes if isinstance(scheme, Mapping)]

    try:
        raw_updates = extract_profile_updates(
            user_message, copy.deepcopy(safe_profile), copy.deepcopy(safe_history)
        )
    except Exception:
        raw_updates = {}
    updates = {}
    if isinstance(raw_updates, Mapping):
        for field in PROFILE_FIELDS:
            value = raw_updates.get(field)
            if value is not None and _valid_profile_value(field, value):
                if safe_profile.get(field) != value:
                    updates[field] = value
                    safe_profile[field] = value

    retrieved = retrieve_relevant_schemes(user_message, safe_profile, catalogue)
    evaluations = evaluate_all(safe_profile, retrieved)
    try:
        generated = generate_grounded_response(
            user_message,
            copy.deepcopy(safe_history),
            copy.deepcopy(safe_profile),
            copy.deepcopy(retrieved),
            copy.deepcopy(evaluations),
        )
    except Exception:
        return _unavailable_result(
            safe_profile,
            "The AI response is unavailable. Check the OpenAI configuration and try again.",
        )

    if not isinstance(generated, Mapping):
        return _unavailable_result(safe_profile, "The AI response was unavailable or malformed.")
    answer = generated.get("answer")
    raw_ids = generated.get("source_scheme_ids")
    raw_questions = generated.get("suggested_questions")
    if not isinstance(answer, str) or not answer.strip() or not isinstance(raw_ids, list) or not isinstance(raw_questions, list):
        return _unavailable_result(safe_profile, "The AI response was unavailable or malformed.")

    supplied = {scheme.get("scheme_id"): scheme for scheme in retrieved if isinstance(scheme.get("scheme_id"), str)}
    selected_ids = []
    for candidate in raw_ids:
        if isinstance(candidate, str) and candidate in supplied and candidate not in selected_ids:
            selected_ids.append(candidate)
    source_schemes = [supplied[item] for item in selected_ids]
    selected_evaluations = evaluate_all(
        safe_profile,
        source_schemes,
    )
    action_plan = build_portfolio_action_plan(selected_evaluations, source_schemes, safe_profile)
    safe_answer = _URL.sub("", answer).strip()
    suggested_questions = [
        question.strip()[:200]
        for question in raw_questions
        if isinstance(question, str) and question.strip()
    ][:4]
    result = {
        "answer": safe_answer,
        "updated_profile": safe_profile,
        "profile_updates": updates,
        "evaluations": selected_evaluations,
        "source_schemes": source_schemes,
        "action_plan": action_plan,
        "relationships": _safe_relations(source_schemes, catalogue),
        "suggested_questions": suggested_questions,
    }
    return result
