"""OpenAI Responses API boundary for CivicLens."""

from __future__ import annotations

import json
import os
from typing import Any

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


def _request_json(instructions: str, payload: dict[str, Any]) -> Any:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip()
    if not api_key or not model:
        raise RuntimeError("OpenAI configuration is unavailable; set OPENAI_API_KEY and OPENAI_MODEL.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI client dependency is unavailable.") from exc
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
        text={"format": {"type": "json_object"}},
    )
    text = getattr(response, "output_text", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("The language model returned no structured response.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("The language model returned invalid structured output.") from exc


def extract_profile_updates(user_message, current_profile, recent_history, **kwargs):
    """Extract only explicit user facts into the fixed profile schema."""
    return _request_json(
        "Extract facts explicitly stated about the user in the latest message. "
        "Do not infer, convert, or copy facts about another person. Return a JSON object "
        "with exactly these profile keys; use null when not newly stated: " + ", ".join(PROFILE_FIELDS),
        {
            "message": user_message,
            "current_profile": current_profile,
            "recent_history": recent_history,
        },
    )


def generate_grounded_response(
    user_message,
    recent_history,
    profile,
    retrieved_schemes,
    evaluations,
    **kwargs,
):
    """Generate an answer using only the supplied catalogue and rule results."""
    return _request_json(
        "Answer the student's question concisely using only the supplied scheme records "
        "and deterministic evaluations. Never change eligibility or verification results, "
        "promise an award, invent application details, or provide URLs. Return JSON with "
        "answer (string), source_scheme_ids (array of supplied scheme_id values), and "
        "suggested_questions (array of short strings).",
        {
            "message": user_message,
            "recent_history": recent_history,
            "profile": profile,
            "retrieved_schemes": retrieved_schemes,
            "evaluations": evaluations,
        },
    )
