"""CivicLens — Your Public Benefits Navigator (Streamlit front end).

Run with:
    streamlit run app.py

This module owns only the user interface. It does not evaluate eligibility,
retrieve schemes or call the LLM itself: it loads the catalogue with
``civiclens.data_loader.load_schemes`` and sends each message through
``civiclens.chat_service.respond_to_message``, then renders what comes back.

Design rules followed here:
* Chat history and the profile live only in ``st.session_state``.
* Unknown profile values stay ``None`` (blank input is never 0 / False).
* Links shown to the user come only from catalogue-derived records that the
  backend returned (and are re-checked to be http/https). Links inside the
  model's free text are stripped.
* If the backend fails, no answer is invented; history and profile are kept.
"""

from __future__ import annotations

import copy
import logging
import math
import os
import re
from contextlib import contextmanager
from typing import Any, Iterator

import streamlit as st

APP_TITLE = "CivicLens — Your Public Benefits Navigator"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="expanded",
)

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional at runtime
    load_dotenv = None  # type: ignore[assignment]

try:
    from civiclens.chat_service import respond_to_message
    from civiclens.data_loader import load_schemes
except Exception as _import_exc:  # noqa: BLE001 - show a friendly start-up error
    st.title(APP_TITLE)
    st.error(
        "CivicLens could not start because its backend modules failed to load "
        f"({type(_import_exc).__name__}). Run the app from the project root "
        "(the folder that contains `app.py` and the `civiclens/` package) with "
        "`streamlit run app.py`, and make sure the backend files are present."
    )
    st.stop()

LOGGER = logging.getLogger("civiclens.app")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROFILE_KEYS = (
    "state",
    "age",
    "education_level",
    "annual_family_income_inr",
    "institution_type",
    "course_type",
    "academic_year",
    "first_generation_student",
)

PROFILE_LABELS = {
    "state": "State",
    "age": "Age (years)",
    "education_level": "Education level",
    "annual_family_income_inr": "Annual family income (INR)",
    "institution_type": "Institution type",
    "course_type": "Course type",
    "academic_year": "Academic year",
    "first_generation_student": "First-generation student",
}

WIDGET_KEYS = {
    "state": "pf_state",
    "age": "pf_age",
    "education_level": "pf_education_level",
    "annual_family_income_inr": "pf_income",
    "institution_type": "pf_institution_type",
    "course_type": "pf_course_type",
    "academic_year": "pf_academic_year",
    "first_generation_student": "pf_first_gen",
}

FIRST_GEN_LABELS = {None: "Not specified", True: "Yes", False: "No"}
FIRST_GEN_VALUES = {label: value for value, label in FIRST_GEN_LABELS.items()}

HISTORY_LIMIT = 8  # messages sent to the backend as context
HISTORY_CHARS = 2000  # per-message cap on context sent to the backend
MAX_PROMPT_CHARS = 2000
MAX_TEXT_FIELD_CHARS = 100
AGE_MAX = 120
INCOME_MAX = 1_000_000_000_000
MAX_SUGGESTIONS = 4

LLM_ENV_VARS = ("OPENAI_API_KEY", "OPENAI_MODEL")

STARTER_PROMPTS = (
    "What kinds of education support can you help me explore?",
    "What details should I tell you so you can check scheme conditions?",
    "Which conditions might I still need to confirm for these schemes?",
    "Which documents should I check before applying to education schemes?",
)

# Ordering required by the project contract (display only; statuses come from
# the backend rule engine and are never recomputed here).
STATUS_ORDER = (
    "potential_match",
    "needs_more_information",
    "unlikely_match",
    "not_verified",
)
STATUS_LABELS = {
    "potential_match": ("✔", "Potential match"),
    "needs_more_information": ("?", "More information needed"),
    "unlikely_match": ("✖", "Unlikely match"),
    "not_verified": ("⚠", "Not verified"),
}
STATUS_FALLBACK = ("?", "Status unavailable")

CONDITION_LABELS = {
    "satisfied": ("✔", "Satisfied"),
    "not_satisfied": ("✖", "Not satisfied"),
    "unknown": ("?", "Unknown"),
}

VERIFICATION_TEXT = {
    "verified": "Verified",
    "needs_review": "Needs review — not verified",
    "demo_only": "Demo only — not real programme data",
}

ACTION_TYPE_LABELS = {
    "answer_missing_information": "Provide information",
    "verify_conditions": "Verify conditions",
    "prepare_document": "Prepare document",
    "complete_application_step": "Application step",
    "open_official_source": "Open official source",
}

ESTABLISHED_RELATIONS = {
    "compatible": "Compatible",
    "restricted": "Restricted",
    "prerequisite": "Prerequisite",
    "complementary": "Complementary",
}

DEFAULT_DISCLAIMER = (
    "Preliminary screening only. This is not an official eligibility decision "
    "or a guarantee of any benefit. Confirm everything on the official source."
)

# --------------------------------------------------------------------------- #
# Text safety helpers
# --------------------------------------------------------------------------- #

_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>$~|#])")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_REF_DEF = re.compile(r"^[ ]{0,3}\[[^\]]+\]:[ \t]*\S.*$", re.MULTILINE)
_ANGLE_URL = re.compile(r"<\s*(?:https?://|www\.|mailto:)[^>]*>", re.IGNORECASE)
_BARE_URL = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
_AADHAAR_LIKE = re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)")
_CARD_LIKE = re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)")


def _esc(text: Any) -> str:
    """Escape Markdown/LaTeX control characters so data renders as plain text."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def _short(text: str, limit: int = 300) -> str:
    """Collapse whitespace and truncate to ``limit`` characters."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _sanitize_model_text(text: str) -> tuple[str, bool]:
    """Remove links/images from model-written text.

    Returns ``(clean_text, links_were_removed)``. Displayed links must come
    from validated catalogue records, never from model output.
    """
    original = text
    cleaned = _MD_IMAGE.sub("", original)
    cleaned = _MD_LINK.sub(lambda m: m.group(1), cleaned)
    cleaned = _MD_REF_DEF.sub("", cleaned)
    cleaned = _ANGLE_URL.sub("(link removed)", cleaned)
    cleaned = _BARE_URL.sub("(link removed)", cleaned)
    removed = cleaned != original
    cleaned = cleaned.replace("$", "\\$")
    return cleaned.strip(), removed


def _looks_sensitive(text: str) -> bool:
    """Heuristic: 12-digit (Aadhaar-like) or 16-digit (card-like) numbers."""
    return bool(_AADHAAR_LIKE.search(text) or _CARD_LIKE.search(text))


def _safe_url(url: Any) -> str | None:
    """Return ``url`` only if it is a plain http(s) URL, else ``None``."""
    if not isinstance(url, str):
        return None
    candidate = url.strip()
    if not candidate or re.search(r"\s", candidate):
        return None
    if not re.match(r"^https?://[^/\s]+", candidate, re.IGNORECASE):
        return None
    return candidate


def _link(label: str, url: Any) -> str:
    """Markdown link for a safe URL, or an empty string."""
    safe = _safe_url(url)
    if safe is None:
        return ""
    encoded = (
        safe.replace("(", "%28")
        .replace(")", "%29")
        .replace("<", "%3C")
        .replace(">", "%3E")
        .replace('"', "%22")
        .replace("`", "%60")
    )
    return f"[{_esc(label)}]({encoded})"


def _fmt_value(value: Any) -> str:
    """Human-readable form of a profile/rule value."""
    if value is None:
        return "not provided"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, (int, float)):
        return f"{value:,}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt_value(v) for v in value) + "]"
    return str(value)


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _as_dict_list(value: Any) -> list[dict]:
    return [item for item in _as_list(value) if isinstance(item, dict)]


# --------------------------------------------------------------------------- #
# Profile validation and session helpers
# --------------------------------------------------------------------------- #


def _empty_profile() -> dict[str, Any]:
    return {key: None for key in PROFILE_KEYS}


def _number_to_text(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _valid_value(key: str, value: Any) -> bool:
    """Schema check for one profile value (bool is never a number)."""
    if value is None:
        return True
    if key == "age":
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0
    if key == "annual_family_income_inr":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        )
    if key == "first_generation_student":
        return isinstance(value, bool)
    return isinstance(value, str)


def _sanitize_profile(candidate: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    """Keep only schema-valid values from ``candidate``; else use ``fallback``."""
    result: dict[str, Any] = {}
    for key in PROFILE_KEYS:
        if isinstance(candidate, dict):
            value = candidate.get(key, fallback.get(key))
        else:
            value = fallback.get(key)
        if isinstance(value, str):
            value = value.strip() or None
        result[key] = value if _valid_value(key, value) else fallback.get(key)
    return result


def _set_profile(profile: dict[str, Any]) -> None:
    """Store the profile and mirror it into the sidebar widget state.

    Must run before the sidebar widgets are created in a script run (or inside
    a widget callback); ``main`` renders the sidebar last for that reason.
    """
    ss = st.session_state
    clean = {key: profile.get(key) for key in PROFILE_KEYS}
    ss["profile"] = clean
    for key in PROFILE_KEYS:
        widget_key = WIDGET_KEYS[key]
        value = clean[key]
        if key == "first_generation_student":
            ss[widget_key] = FIRST_GEN_LABELS[value]
        elif value is None:
            ss[widget_key] = ""
        elif key in ("age", "annual_family_income_inr"):
            ss[widget_key] = _number_to_text(value)
        else:
            ss[widget_key] = str(value)


def _parse_text_field(raw: Any, label: str) -> tuple[str | None, str | None]:
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not text:
        return None, None
    if len(text) > MAX_TEXT_FIELD_CHARS:
        return None, f"{label}: please keep it under {MAX_TEXT_FIELD_CHARS} characters."
    if _looks_sensitive(text):
        return None, f"{label}: this looks like an ID or card number. Please remove it."
    return text, None


def _parse_age(raw: Any) -> tuple[int | None, str | None]:
    text = str(raw or "").strip()
    if not text:
        return None, None
    if not re.fullmatch(r"\d{1,3}", text):
        return None, "Age: enter a whole number of years (such as 20), or leave it blank."
    age = int(text)
    if age > AGE_MAX:
        return None, f"Age: please enter a value between 0 and {AGE_MAX}, or leave it blank."
    return age, None


def _parse_income(raw: Any) -> tuple[int | float | None, str | None]:
    text = str(raw or "").strip()
    if not text:
        return None, None
    cleaned = re.sub(r"(?i)(₹|rs\.?|inr)", "", text).replace(",", "").replace(" ", "")
    if not re.fullmatch(r"\d+(\.\d+)?", cleaned):
        return None, (
            "Annual family income: enter a non-negative number in rupees "
            "(commas are fine), or leave it blank if unknown."
        )
    value: int | float = float(cleaned) if "." in cleaned else int(cleaned)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None, "Annual family income: please enter a finite number."
        if value.is_integer():
            value = int(value)
    if value > INCOME_MAX:
        return None, "Annual family income: that number is too large. Please check it."
    return value, None


# --------------------------------------------------------------------------- #
# Session state, callbacks
# --------------------------------------------------------------------------- #


def _init_state() -> None:
    ss = st.session_state
    if "messages" not in ss:
        ss["messages"] = []
    if "profile" not in ss:
        _set_profile(_empty_profile())
    ss.setdefault("profile_feedback", None)
    ss.setdefault("queued_prompt", None)
    ss.setdefault("catalogue", None)


def _queue_prompt(text: str) -> None:
    """Callback: send ``text`` as the next user message."""
    st.session_state["queued_prompt"] = text


def _reload_catalogue() -> None:
    st.session_state["catalogue"] = None


def _clear_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state["queued_prompt"] = None


def _clear_profile() -> None:
    _set_profile(_empty_profile())
    st.session_state["profile_feedback"] = {
        "kind": "success",
        "text": "Profile cleared. All fields are now unknown.",
    }


def _reset_everything() -> None:
    _clear_conversation()
    _set_profile(_empty_profile())
    st.session_state["profile_feedback"] = None


def _retry_last() -> None:
    """Callback: drop the failed turn and send the same message again."""
    messages = st.session_state["messages"]
    if (
        len(messages) >= 2
        and messages[-1].get("kind") == "error"
        and messages[-2].get("failed")
    ):
        prompt = messages[-2]["content"]
        del messages[-2:]
        st.session_state["queued_prompt"] = prompt


def _save_profile_from_form() -> None:
    """Callback for the profile form: validate, then store or report errors."""
    ss = st.session_state
    errors: list[str] = []
    new_profile: dict[str, Any] = {}

    for key in ("state", "education_level", "institution_type", "course_type", "academic_year"):
        value, error = _parse_text_field(ss.get(WIDGET_KEYS[key]), PROFILE_LABELS[key])
        new_profile[key] = value
        if error:
            errors.append(error)

    age, error = _parse_age(ss.get(WIDGET_KEYS["age"]))
    new_profile["age"] = age
    if error:
        errors.append(error)

    income, error = _parse_income(ss.get(WIDGET_KEYS["annual_family_income_inr"]))
    new_profile["annual_family_income_inr"] = income
    if error:
        errors.append(error)

    new_profile["first_generation_student"] = FIRST_GEN_VALUES.get(
        ss.get(WIDGET_KEYS["first_generation_student"], "Not specified")
    )

    if errors:
        ss["profile_feedback"] = {"kind": "error", "errors": errors}
        return
    _set_profile(new_profile)
    ss["profile_feedback"] = {
        "kind": "success",
        "text": (
            "Profile saved. Earlier replies keep the results they were "
            "generated with; use “Re-check my matches” to refresh."
        ),
    }


# --------------------------------------------------------------------------- #
# Catalogue and environment
# --------------------------------------------------------------------------- #


def _load_catalogue() -> dict[str, Any]:
    """Load the scheme catalogue; never raises."""
    try:
        schemes = load_schemes()
    except FileNotFoundError:
        return {"schemes": [], "error": "The catalogue file data/schemes.json was not found."}
    except Exception as exc:  # noqa: BLE001 - surfaced as a friendly message
        LOGGER.warning("Catalogue load failed: %s", type(exc).__name__)
        return {
            "schemes": [],
            "error": (
                "The catalogue could not be read or failed validation "
                f"({type(exc).__name__}): {_short(str(exc))}"
            ),
        }
    if not isinstance(schemes, list) or any(not isinstance(s, dict) for s in schemes):
        return {"schemes": [], "error": "The catalogue is malformed (expected a list of scheme records)."}
    if not schemes:
        return {"schemes": [], "error": "The catalogue is empty, so there are no schemes to check."}
    return {"schemes": schemes, "error": None}


def _get_catalogue() -> dict[str, Any]:
    ss = st.session_state
    if ss.get("catalogue") is None:
        ss["catalogue"] = _load_catalogue()
    return ss["catalogue"]


def _missing_llm_config() -> list[str]:
    return [name for name in LLM_ENV_VARS if not os.environ.get(name, "").strip()]


def _catalogue_hints(schemes: list[dict]) -> dict[str, str]:
    """Wording used in catalogue conditions, shown as optional input hints."""
    found: dict[str, set[str]] = {key: set() for key in PROFILE_KEYS}
    for scheme in schemes:
        for rule in _as_list(scheme.get("eligibility_rules")):
            if not isinstance(rule, dict):
                continue
            field = rule.get("field")
            if field not in found or rule.get("operator") not in ("equals", "in", "contains"):
                continue
            value = rule.get("value")
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, str) and item.strip():
                    found[field].add(item.strip())
    return {key: ", ".join(sorted(vals)[:8]) for key, vals in found.items() if vals}


# --------------------------------------------------------------------------- #
# Backend call
# --------------------------------------------------------------------------- #


def _bounded_history(messages: list[dict]) -> list[dict[str, str]]:
    """Last few real conversation turns as plain role/content pairs."""
    turns = [
        {"role": m["role"], "content": str(m.get("content", ""))[:HISTORY_CHARS]}
        for m in messages
        if m.get("kind") in ("user", "answer") and not m.get("failed")
    ]
    return turns[-HISTORY_LIMIT:]


def _describe_failure(exc: Exception) -> str:
    """User-safe explanation of a backend exception (never shows its message)."""
    name = type(exc).__name__
    missing = _missing_llm_config()
    lowered = name.lower()
    if missing:
        reason = (
            "the AI model is not configured "
            f"({', '.join(missing)} not set)"
        )
    elif "ratelimit" in lowered:
        reason = "the AI service is rate-limiting requests; please wait a moment"
    elif "timeout" in lowered:
        reason = "the request to the AI service timed out"
    elif "connection" in lowered:
        reason = "the AI service could not be reached (check your internet connection)"
    elif "authentication" in lowered or "permission" in lowered:
        reason = "the AI service rejected the configured credentials or model"
    else:
        reason = "something went wrong while processing your message"
    return f"No assistant answer was generated: {reason}. (Error type: {name}.)"


def _error_text(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("message") or value.get("error") or ""
    text = _short(str(value))
    return text or "The assistant reported an error."


def _clean_suggestions(value: Any) -> list[str]:
    out: list[str] = []
    for item in _as_list(value):
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        if not text or len(text) > 200 or _BARE_URL.search(text) or text in out:
            continue
        out.append(text)
        if len(out) >= MAX_SUGGESTIONS:
            break
    return out


def _call_backend(prompt: str, history: list[dict[str, str]]) -> dict[str, Any]:
    """Call ``respond_to_message`` and normalise success/failure.

    Returns ``{"error": str}`` on failure or ``{"message": dict, "profile": dict}``.
    """
    ss = st.session_state
    previous_profile = dict(ss["profile"])
    try:
        result = respond_to_message(
            prompt,
            copy.deepcopy(history),
            dict(previous_profile),
            copy.deepcopy(ss["catalogue"]["schemes"]),
        )
    except Exception as exc:  # noqa: BLE001 - never expose tracebacks
        LOGGER.warning("respond_to_message failed: %s", type(exc).__name__)
        return {"error": _describe_failure(exc)}

    if not isinstance(result, dict):
        return {"error": "The assistant returned an unexpected response, so nothing was shown."}
    if result.get("error"):
        return {"error": _error_text(result["error"])}
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return {"error": "The assistant returned an empty answer, so nothing was shown."}

    new_profile = _sanitize_profile(result.get("updated_profile"), previous_profile)
    raw_updates = result.get("profile_updates")
    updates = {}
    if isinstance(raw_updates, dict):
        for key in PROFILE_KEYS:
            value = raw_updates.get(key)
            if value is not None and _valid_value(key, value) and previous_profile.get(key) != value:
                updates[key] = value

    message = {
        "role": "assistant",
        "kind": "answer",
        "content": answer,
        "payload": {
            "evaluations": _as_dict_list(result.get("evaluations")),
            "source_schemes": _as_dict_list(result.get("source_schemes")),
            "action_plan": _as_dict_list(result.get("action_plan")),
            "relationships": _as_dict_list(result.get("relationships")),
            "suggested_questions": _clean_suggestions(result.get("suggested_questions")),
            "profile_updates": updates,
        },
    }
    return {"message": message, "profile": new_profile}


# --------------------------------------------------------------------------- #
# Rendering: evidence
# --------------------------------------------------------------------------- #


def _verification(ev: dict, scheme: dict) -> Any:
    return ev.get("verification_status") or scheme.get("verification_status")


def _status_key(ev: dict, scheme: dict) -> str:
    """Display status. Anything not verified is always shown as not verified."""
    if _verification(ev, scheme) != "verified":
        return "not_verified"
    status = ev.get("status")
    return status if status in STATUS_LABELS else "unknown"


def _sort_evaluations(evals: list[dict], lookup: dict[str, dict]) -> list[dict]:
    rank = {status: i for i, status in enumerate(STATUS_ORDER)}

    def sort_key(ev: dict) -> int:
        scheme = lookup.get(ev.get("scheme_id"), {})
        return rank.get(_status_key(ev, scheme), len(rank))

    return sorted(evals, key=sort_key)  # stable: keeps backend order within a status


def _scheme_name(ev: dict, lookup: dict[str, dict]) -> str:
    scheme = lookup.get(ev.get("scheme_id"), {})
    return str(ev.get("scheme_name") or scheme.get("name") or ev.get("scheme_id") or "Unnamed scheme")


def _scheme_title(ev: dict, lookup: dict[str, dict]) -> str:
    scheme = lookup.get(ev.get("scheme_id"), {})
    glyph, label = STATUS_LABELS.get(_status_key(ev, scheme), STATUS_FALLBACK)
    return f"{glyph} {label} — {_esc(_scheme_name(ev, lookup))}"


@contextmanager
def _section(title: str, nested: bool) -> Iterator[None]:
    """An expander, or (when already inside an expander) a titled block."""
    if nested:
        st.markdown(f"**{title}**")
        with st.container():
            yield
        st.divider()
    else:
        with st.expander(title):
            yield


def _render_conditions(conditions: list[dict], unverified: bool) -> None:
    if unverified:
        st.caption(
            "Conditions from an unverified record are shown for transparency only. "
            "They are not a screening result."
        )
    lines: list[str] = []
    for cond in conditions:
        glyph, label = CONDITION_LABELS.get(cond.get("status"), ("?", "Unknown"))
        text = cond.get("label") or cond.get("rule_id") or "Condition"
        lines.append(f"- **{glyph} {label}** — {_esc(text)}")
        field = cond.get("field")
        if field in PROFILE_LABELS:
            lines.append(f"    - Profile field: {_esc(PROFILE_LABELS[field])}")
        lines.append(
            f"    - Required value: {_esc(_fmt_value(cond.get('expected_value')))}"
            f" · Your value: {_esc(_fmt_value(cond.get('actual_value')))}"
        )
        explanation = cond.get("explanation")
        if isinstance(explanation, str) and explanation.strip():
            lines.append(f"    - {_esc(explanation)}")
        source = _link("Condition source", cond.get("source_url"))
        if source:
            lines.append(f"    - {source}")
    st.markdown("**Condition trace**\n\n" + "\n".join(lines))


def _missing_label(item: Any) -> str:
    if isinstance(item, dict):
        item = item.get("label") or item.get("field") or ""
    text = str(item)
    return PROFILE_LABELS.get(text, text)


def _render_scheme_body(ev: dict, lookup: dict[str, dict]) -> None:
    scheme = lookup.get(ev.get("scheme_id"), {})
    ver = _verification(ev, scheme)
    key = _status_key(ev, scheme)
    glyph, label = STATUS_LABELS.get(key, STATUS_FALLBACK)

    checked = scheme.get("verified_on")
    checked_text = (
        f"checked {checked.strip()}"
        if isinstance(checked, str) and checked.strip()
        else "check date not recorded"
    )
    st.markdown(
        f"**Screening status:** {glyph} {label}  \n"
        f"**Record verification:** {_esc(VERIFICATION_TEXT.get(ver, 'Unknown — treat as not verified'))}"
        f" ({_esc(checked_text)})"
    )
    if ver == "demo_only":
        st.warning(
            "**Demo-only record.** This is sample data, not a real programme's "
            "rules. Do not rely on it for any decision."
        )
    elif ver != "verified":
        st.warning(
            "**Not verified.** This record has not been checked against an "
            "official source, so no eligibility result is shown. Check the "
            "official source before acting on anything here."
        )
    elif key == "potential_match":
        st.caption("A potential match is preliminary screening, not official eligibility.")

    for heading, field in (
        ("Summary", "summary"),
        ("Benefit (as described in the record)", "benefit_summary"),
        ("Jurisdiction", "jurisdiction"),
    ):
        value = scheme.get(field)
        if isinstance(value, str) and value.strip():
            st.markdown(f"**{heading}:** {_esc(value)}")

    reasons = [r for r in _as_list(ev.get("reasons")) if isinstance(r, str) and r.strip()]
    if reasons:
        st.markdown("**Why this status**\n\n" + "\n".join(f"- {_esc(r)}" for r in reasons))

    conditions = _as_dict_list(ev.get("conditions"))
    if conditions:
        _render_conditions(conditions, unverified=(ver != "verified"))

    missing = [_missing_label(m) for m in _as_list(ev.get("missing_information"))]
    missing = [m for m in missing if m.strip()]
    if missing:
        st.markdown(
            "**Missing information**\n\n" + "\n".join(f"- {_esc(m)}" for m in missing)
        )

    docs = _as_list(ev.get("document_checklist"))
    if docs:
        doc_lines: list[str] = []
        for doc in docs:
            if isinstance(doc, dict):
                required = doc.get("required")
                tag = "Required" if required is True else "Optional" if required is False else "Requirement not stated"
                line = f"- **{tag}:** {_esc(doc.get('name') or 'Unnamed document')}"
                link = _link("source", doc.get("source_url"))
                if link:
                    line += f" — {link}"
                doc_lines.append(line)
            elif isinstance(doc, str) and doc.strip():
                doc_lines.append(f"- {_esc(doc)}")
        if doc_lines:
            suffix = " (from an unverified record)" if ver != "verified" else ""
            st.markdown(f"**Documents listed in the record{suffix}**\n\n" + "\n".join(doc_lines))

    links = []
    official = _link("Official source", ev.get("official_source_url") or scheme.get("official_source_url"))
    if official:
        links.append(official)
    apply_link = _link(
        "Application page" if ver == "verified" else "Application page listed in the record (not verified)",
        ev.get("application_url") or scheme.get("application_url"),
    )
    if apply_link:
        links.append(apply_link)
    if links:
        st.markdown("**Links:** " + " · ".join(links))

    disclaimer = ev.get("disclaimer")
    st.caption(_esc(disclaimer) if isinstance(disclaimer, str) and disclaimer.strip() else DEFAULT_DISCLAIMER)


def _render_sources(source_schemes: list[dict]) -> None:
    lines: list[str] = []
    for scheme in source_schemes:
        name = scheme.get("name") or scheme.get("scheme_id") or "Unnamed scheme"
        ver = VERIFICATION_TEXT.get(scheme.get("verification_status"), "Unknown — treat as not verified")
        checked = scheme.get("verified_on")
        checked_text = (
            f"checked {checked.strip()}"
            if isinstance(checked, str) and checked.strip()
            else "check date not recorded"
        )
        link = _link("Official source", scheme.get("official_source_url")) or "no official link recorded"
        line = f"- **{_esc(name)}** — {_esc(ver)}; {_esc(checked_text)} — {link}"
        title = scheme.get("source_title")
        if isinstance(title, str) and title.strip():
            line += f" (_{_esc(title)}_)"
        lines.append(line)
    if lines:
        st.markdown("**Sources**\n\n" + "\n".join(lines))


def _render_action_plan(actions: list[dict], lookup: dict[str, dict]) -> None:
    st.caption(
        "Ordered by fixed rules from catalogue data and your profile. These are "
        "preparation steps, not a confirmed application procedure."
    )
    number_by_id = {
        a["action_id"]: i
        for i, a in enumerate(actions, 1)
        if isinstance(a.get("action_id"), str)
    }
    lines: list[str] = []
    for i, action in enumerate(actions, 1):
        atype = action.get("action_type")
        type_label = ACTION_TYPE_LABELS.get(atype, str(atype or "Step").replace("_", " ").capitalize())
        lines.append(f"{i}. **{_esc(action.get('title') or 'Untitled step')}** — {_esc(type_label)}")
        description = action.get("description")
        if isinstance(description, str) and description.strip():
            lines.append(f"     - {_esc(description)}")
        if action.get("priority") is not None:
            lines.append(f"     - Priority: {_esc(action['priority'])}")
        if isinstance(action.get("status"), str) and action["status"].strip():
            lines.append(f"     - Status: {_esc(action['status'].replace('_', ' '))}")
        deps = [number_by_id[d] for d in _as_list(action.get("depends_on")) if d in number_by_id]
        if deps:
            lines.append("     - Comes after step " + ", ".join(str(n) for n in deps))
        names = [
            str(lookup.get(sid, {}).get("name") or sid)
            for sid in _as_list(action.get("scheme_ids"))
            if isinstance(sid, str)
        ]
        if names:
            lines.append("     - Related to: " + _esc(", ".join(names)))
        source = _link("Source", action.get("source_url"))
        if source:
            lines.append(f"     - {source}")
    st.markdown("\n".join(lines))


def _classify_relationship(rel: dict) -> tuple[str, str]:
    """Return ``(kind, label)``; only recorded, verified evidence is 'verified'."""
    rtype = rel.get("relation_type")
    if rtype not in ESTABLISHED_RELATIONS:
        return "unknown", "Unknown — no established compatibility information"
    if rel.get("verification_status") == "verified" and _safe_url(rel.get("source_url")):
        return "verified", f"Verified relation: {ESTABLISHED_RELATIONS[rtype]}"
    return "unverified", "Unverified / needs review — treat as unknown"


def _render_relationships(relationships: list[dict], lookup: dict[str, dict]) -> None:
    st.caption(
        "Relationships come only from catalogue records. A missing relationship "
        "means unknown, not compatible. Two schemes both matching you does not "
        "mean you can combine them."
    )

    def name_of(scheme_id: Any) -> str:
        if isinstance(scheme_id, str) and scheme_id:
            return str(lookup.get(scheme_id, {}).get("name") or scheme_id)
        return ""

    lines: list[str] = []
    for rel in relationships:
        kind, label = _classify_relationship(rel)
        origin = ""
        for field in ("scheme_id", "source_scheme_id", "from_scheme_id"):
            origin = name_of(rel.get(field))
            if origin:
                break
        target = name_of(rel.get("target_scheme_id")) or "another scheme"
        pair = f"**{_esc(origin)}** and **{_esc(target)}**" if origin else f"Relationship with **{_esc(target)}**"
        lines.append(f"- {pair}: **{_esc(label)}**")
        if kind == "verified":
            evidence = _link(str(rel.get("source_title") or "Evidence"), rel.get("source_url"))
            checked = rel.get("verified_on")
            extra = f" (checked {_esc(checked)})" if isinstance(checked, str) and checked.strip() else ""
            lines.append(f"    - Evidence: {evidence}{extra}")
        else:
            lines.append("    - Do not assume these schemes can be combined. Check the official rules.")
        notes = rel.get("notes")
        if isinstance(notes, str) and notes.strip():
            lines.append(f"    - Notes: {_esc(notes)}")
    st.markdown("\n".join(lines))


def _render_evidence(
    evals: list[dict],
    actions: list[dict],
    relationships: list[dict],
    lookup: dict[str, dict],
    nested: bool,
) -> None:
    for ev in evals:
        with _section(_scheme_title(ev, lookup), nested):
            _render_scheme_body(ev, lookup)
    if actions:
        with _section(f"Suggested next steps ({len(actions)})", nested):
            _render_action_plan(actions, lookup)
    if relationships:
        with _section(f"Related schemes ({len(relationships)})", nested):
            _render_relationships(relationships, lookup)


def _render_payload(idx: int, payload: dict, latest: bool, catalogue_lookup: dict[str, dict]) -> None:
    evals = _as_dict_list(payload.get("evaluations"))
    sources = _as_dict_list(payload.get("source_schemes"))
    actions = _as_dict_list(payload.get("action_plan"))
    relationships = _as_dict_list(payload.get("relationships"))

    if not evals and not sources:
        st.caption("No catalogue schemes were relevant to this message.")
        return

    lookup = dict(catalogue_lookup)
    lookup.update({s["scheme_id"]: s for s in sources if isinstance(s.get("scheme_id"), str)})

    if sources:
        _render_sources(sources)

    evals = _sort_evaluations(evals, lookup)
    if latest:
        _render_evidence(evals, actions, relationships, lookup, nested=False)
    elif evals or actions or relationships:
        if evals:
            summary = "; ".join(
                f"{STATUS_LABELS.get(_status_key(ev, lookup.get(ev.get('scheme_id'), {})), STATUS_FALLBACK)[1]}: "
                f"{_scheme_name(ev, lookup)}"
                for ev in evals
            )
            st.caption("Schemes considered — " + _esc(summary))
        with st.expander("Evidence and next steps from this reply"):
            _render_evidence(evals, actions, relationships, lookup, nested=True)
    st.caption(DEFAULT_DISCLAIMER)


def _render_suggestions(idx: int, questions: list[str]) -> None:
    if not questions:
        return
    st.caption("Suggested follow-up questions")
    for j, question in enumerate(questions):
        st.button(_esc(question), key=f"sugg_{idx}_{j}", on_click=_queue_prompt, args=(question,))


# --------------------------------------------------------------------------- #
# Rendering: messages and page
# --------------------------------------------------------------------------- #


def _render_user_text(text: str) -> None:
    st.markdown(_esc(text).replace("\n", "  \n"))


def _render_profile_updates(updates: Any) -> None:
    if not isinstance(updates, dict) or not updates:
        return
    parts = [
        f"{PROFILE_LABELS[key]}: {_fmt_value(updates[key])}"
        for key in PROFILE_KEYS
        if key in updates
    ]
    st.caption(
        "Added to your profile from your message — " + _esc("; ".join(parts))
        + ". You can correct or clear any value in the sidebar."
    )


def _render_assistant_body(idx: int, msg: dict, latest: bool, lookup: dict[str, dict]) -> None:
    if msg.get("kind") == "error":
        st.error(_esc(msg.get("content", "")))
        if latest:
            st.button("Retry last message", key=f"retry_{idx}", on_click=_retry_last)
        return
    clean, removed = _sanitize_model_text(str(msg.get("content", "")))
    st.markdown(clean or "_(empty response)_")
    if removed:
        st.caption("Links in the AI-written text were removed. Use the source links below.")
    payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}
    _render_profile_updates(payload.get("profile_updates"))
    _render_payload(idx, payload, latest, lookup)
    if latest:
        _render_suggestions(idx, _clean_suggestions(payload.get("suggested_questions")))


def _render_message(idx: int, msg: dict, latest: bool, lookup: dict[str, dict]) -> None:
    role = "user" if msg.get("role") == "user" else "assistant"
    with st.chat_message(role):
        if role == "user":
            _render_user_text(str(msg.get("content", "")))
        else:
            _render_assistant_body(idx, msg, latest, lookup)


def _process_prompt(prompt: str, lookup: dict[str, dict]) -> None:
    """Handle one new user message end to end (rendered inline)."""
    ss = st.session_state
    ss["profile_feedback"] = None
    text = prompt.strip()
    if not text:
        return
    if len(text) > MAX_PROMPT_CHARS:
        st.warning(f"That message is too long (limit {MAX_PROMPT_CHARS} characters). Please shorten it.")
        return
    if _looks_sensitive(text):
        st.warning(
            "Your message looks like it contains a 12- or 16-digit ID or card "
            "number (for example an Aadhaar number). It was **not** sent or "
            "saved. Please remove it and try again."
        )
        return

    history = _bounded_history(ss["messages"])
    user_msg: dict[str, Any] = {"role": "user", "kind": "user", "content": text}
    ss["messages"].append(user_msg)
    with st.chat_message("user"):
        _render_user_text(text)

    with st.chat_message("assistant"):
        with st.spinner("Checking the catalogue against your profile…"):
            outcome = _call_backend(text, history)
        if "error" in outcome:
            user_msg["failed"] = True
            error_msg = {"role": "assistant", "kind": "error", "content": outcome["error"]}
            ss["messages"].append(error_msg)
            _render_assistant_body(len(ss["messages"]) - 1, error_msg, True, lookup)
            return
        _set_profile(outcome["profile"])
        ss["messages"].append(outcome["message"])
        _render_assistant_body(len(ss["messages"]) - 1, outcome["message"], True, lookup)


def _render_header() -> None:
    st.title(APP_TITLE)
    st.caption("A prototype assistant that helps students in Tamil Nadu explore public education benefits.")
    st.info(
        "**Prototype — preliminary screening only.** CivicLens is not a government "
        "authority and does not decide official eligibility. A “potential match” "
        "is not a guarantee that any benefit will be awarded. Always confirm on "
        "the official source."
    )
    st.warning(
        "**Privacy:** your messages and relevant profile details are sent to the "
        "configured external LLM API (OpenAI) to understand and explain results; "
        "that provider's own data policies apply. This app keeps chat and profile "
        "only in this browser session. **Do not enter Aadhaar numbers, OTPs, "
        "passwords, bank details or identity documents.**"
    )


def _render_config_notice() -> None:
    missing = _missing_llm_config()
    if missing:
        names = ", ".join(f"`{name}`" for name in missing)
        st.warning(
            f"**AI assistant not configured.** Missing: {names}. Add them to a "
            "local `.env` file (see `.env.example`) or your environment, then "
            "restart the app. Until then CivicLens cannot write replies, and it "
            "will not fake one."
        )


def _render_welcome() -> None:
    with st.chat_message("assistant"):
        st.markdown(
            "**Hello — I'm CivicLens, a prototype benefits navigator for students "
            "in Tamil Nadu.**\n\n"
            "Tell me about your situation in your own words, such as your state, "
            "age, course and family income. I use only the facts you state. I "
            "check them against the scheme records in this app's catalogue, then "
            "explain what looks like a potential match, what looks unlikely, and "
            "what I still need to know, with sources and a checklist of next steps.\n\n"
            "**What I am not:** a government authority. Fixed rules, not AI "
            "opinion, decide each screening result, and a “potential match” is a "
            "preliminary screen, never official eligibility or a promise of any "
            "benefit. Please don't share Aadhaar numbers, OTPs, passwords or bank details."
        )


def _render_starters(disabled: bool) -> None:
    st.caption("Try a starter question")
    for i, text in enumerate(STARTER_PROMPTS):
        st.button(text, key=f"starter_{i}", on_click=_queue_prompt, args=(text,), disabled=disabled)


def _render_sidebar(catalogue: dict[str, Any]) -> None:
    ss = st.session_state
    schemes = catalogue["schemes"]
    hints = _catalogue_hints(schemes)
    with st.sidebar:
        st.header("Your profile")
        st.caption(
            "Optional. Share only what you're comfortable with; blank means "
            "unknown. Chat can add or change facts, but only you can clear them here."
        )

        def hint(key: str) -> str:
            base = "Leave blank if unknown."
            return f"{base} Wording used in scheme conditions: {hints[key]}." if key in hints else base

        with st.form("profile_form"):
            st.text_input(PROFILE_LABELS["state"], key=WIDGET_KEYS["state"], help=hint("state"))
            st.text_input(PROFILE_LABELS["age"], key=WIDGET_KEYS["age"], help="Whole years. Leave blank if unknown.")
            st.text_input(
                PROFILE_LABELS["education_level"],
                key=WIDGET_KEYS["education_level"],
                help=hint("education_level"),
            )
            st.text_input(
                PROFILE_LABELS["annual_family_income_inr"],
                key=WIDGET_KEYS["annual_family_income_inr"],
                help="Rupees per year, digits only (commas are fine). Blank is unknown, not zero.",
            )
            st.text_input(
                PROFILE_LABELS["institution_type"],
                key=WIDGET_KEYS["institution_type"],
                help=hint("institution_type"),
            )
            st.text_input(
                PROFILE_LABELS["course_type"], key=WIDGET_KEYS["course_type"], help=hint("course_type")
            )
            st.text_input(
                PROFILE_LABELS["academic_year"],
                key=WIDGET_KEYS["academic_year"],
                help=hint("academic_year"),
            )
            st.selectbox(
                PROFILE_LABELS["first_generation_student"],
                options=list(FIRST_GEN_LABELS.values()),
                key=WIDGET_KEYS["first_generation_student"],
                help="“Not specified” stays unknown. It is not treated as No.",
            )
            st.form_submit_button("Save profile", on_click=_save_profile_from_form)

        feedback = ss.get("profile_feedback")
        if isinstance(feedback, dict):
            if feedback.get("kind") == "success":
                st.success(feedback.get("text", "Saved."))
            else:
                for error in feedback.get("errors", []):
                    st.error(error)

        st.button(
            "Re-check my matches",
            on_click=_queue_prompt,
            args=("Please re-check which schemes I may match, using my current profile.",),
            disabled=bool(catalogue["error"]),
            help="Sends a message so the results reflect your saved profile.",
        )
        st.button("Clear profile", on_click=_clear_profile, help="Sets every profile field back to unknown.")
        st.button("Clear conversation", on_click=_clear_conversation, help="Keeps your profile.")
        st.button("Reset everything", on_click=_reset_everything, help="Clears the conversation and the profile.")

        if not catalogue["error"]:
            counts: dict[Any, int] = {}
            for scheme in schemes:
                status = scheme.get("verification_status")
                counts[status] = counts.get(status, 0) + 1
            st.caption(
                f"Catalogue: {len(schemes)} record(s) — "
                f"{counts.get('verified', 0)} verified, "
                f"{counts.get('needs_review', 0)} needs review, "
                f"{counts.get('demo_only', 0)} demo only."
            )
        st.button("Reload catalogue", on_click=_reload_catalogue)

        with st.expander("About, privacy and limits"):
            st.markdown(
                "- **Not official.** CivicLens is a prototype. It does not speak "
                "for any government body and cannot confirm eligibility.\n"
                "- **How results are produced.** An AI model reads your message "
                "and writes explanations. Fixed Python rules alone decide each "
                "scheme's status and the order of next steps.\n"
                "- **Verification.** Only records marked *Verified* can produce a "
                "potential match. *Needs review* and *Demo only* records are labelled.\n"
                "- **Privacy.** Messages and relevant profile details go to the "
                "configured external LLM API. This app keeps chat and profile in "
                "this browser session and does not save them to disk.\n"
                "- **Never enter** Aadhaar numbers, OTPs, passwords, bank "
                "details or identity documents."
            )


def main() -> None:
    """Render the CivicLens page."""
    if load_dotenv is not None:
        try:
            load_dotenv()
        except Exception:  # noqa: BLE001 - a bad .env must not break the page
            LOGGER.warning("Could not read .env file")

    _init_state()
    catalogue = _get_catalogue()
    catalogue_lookup = {
        s["scheme_id"]: s for s in catalogue["schemes"] if isinstance(s.get("scheme_id"), str)
    }

    _render_header()
    _render_config_notice()
    chat_disabled = bool(catalogue["error"])
    if chat_disabled:
        st.error(
            f"**Scheme catalogue unavailable.** {_esc(catalogue['error'])} Chat is "
            "paused because there are no scheme records to check."
        )
        st.button("Retry loading catalogue", on_click=_reload_catalogue)

    typed = st.chat_input(
        "Describe your situation or ask a question (no Aadhaar, OTPs or passwords)",
        disabled=chat_disabled,
    )
    queued = st.session_state.get("queued_prompt")
    st.session_state["queued_prompt"] = None
    to_process = typed or (None if chat_disabled else queued)

    messages = st.session_state["messages"]
    _render_welcome()
    if not messages and to_process is None:
        _render_starters(chat_disabled)
    last_index = len(messages) - 1
    for idx, msg in enumerate(messages):
        _render_message(idx, msg, latest=(idx == last_index and to_process is None), lookup=catalogue_lookup)

    if to_process:
        _process_prompt(to_process, catalogue_lookup)

    # Sidebar last: profile widget state may be updated by the reply above.
    _render_sidebar(catalogue)


main()
