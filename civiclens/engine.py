"""Deterministic, fail-closed evaluation of scheme eligibility rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .data_loader import DEFAULT_SCHEMES_PATH, NUMERIC_FIELDS, BOOLEAN_FIELDS, STRING_FIELDS, is_real_number, load_schemes

DISCLAIMER = (
    "Preliminary screening only. This is not an official eligibility decision "
    "or a guarantee of any benefit. Confirm everything on the official source."
)
STATUS_ORDER = ("potential_match", "needs_more_information", "unlikely_match", "not_verified")


def _condition_status(rule: Mapping[str, Any], profile: Mapping[str, Any]) -> tuple[str, Any]:
    field = rule.get("field")
    operator = rule.get("operator")
    expected = rule.get("value")
    actual = profile.get(field) if isinstance(field, str) else None

    if field not in NUMERIC_FIELDS | BOOLEAN_FIELDS | STRING_FIELDS:
        return "unknown", None
    if field in NUMERIC_FIELDS and not is_real_number(actual):
        return "unknown", None
    if field in BOOLEAN_FIELDS and not isinstance(actual, bool):
        return "unknown", None
    if field in STRING_FIELDS and (not isinstance(actual, str) or not actual.strip()):
        return "unknown", None

    try:
        if operator == "equals":
            if field in NUMERIC_FIELDS and not is_real_number(expected):
                return "unknown", None
            if field in BOOLEAN_FIELDS and not isinstance(expected, bool):
                return "unknown", None
            if field in STRING_FIELDS and (not isinstance(expected, str) or not expected.strip()):
                return "unknown", None
            passed = actual == expected if field in NUMERIC_FIELDS | BOOLEAN_FIELDS else actual.strip().casefold() == expected.strip().casefold()
        elif operator == "in":
            if not isinstance(expected, list) or not expected:
                return "unknown", None
            if field in NUMERIC_FIELDS and any(not is_real_number(item) for item in expected):
                return "unknown", None
            if field in BOOLEAN_FIELDS and any(not isinstance(item, bool) for item in expected):
                return "unknown", None
            if field in STRING_FIELDS and any(not isinstance(item, str) for item in expected):
                return "unknown", None
            if field in STRING_FIELDS:
                passed = actual.strip().casefold() in {item.strip().casefold() for item in expected}
            else:
                passed = actual in expected
        elif operator == "lte" and field in NUMERIC_FIELDS and is_real_number(expected):
            passed = actual <= expected
        elif operator == "gte" and field in NUMERIC_FIELDS and is_real_number(expected):
            passed = actual >= expected
        elif operator == "between" and field in NUMERIC_FIELDS and isinstance(expected, (list, tuple)) and len(expected) == 2 and all(is_real_number(item) for item in expected):
            passed = expected[0] <= actual <= expected[1]
        elif operator == "contains" and field in STRING_FIELDS and isinstance(expected, str) and expected.strip():
            passed = expected.strip().casefold() in actual.casefold()
        else:
            return "unknown", None
    except (TypeError, ValueError):
        return "unknown", None
    return ("satisfied" if passed else "not_satisfied"), actual


def evaluate_scheme(profile: Mapping[str, Any], scheme: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate one scheme without coercion or executing data-driven rules."""
    profile = profile if isinstance(profile, Mapping) else {}
    scheme = scheme if isinstance(scheme, Mapping) else {}
    raw_rules = scheme.get("eligibility_rules")
    rules = raw_rules if isinstance(raw_rules, list) else []
    conditions = []
    missing = []
    malformed = not isinstance(raw_rules, list)
    for index, item in enumerate(rules):
        if not isinstance(item, Mapping):
            malformed = True
            continue
        rule = item
        rule_id = rule.get("rule_id", f"rule_{index + 1}")
        field = rule.get("field")
        label = rule.get("label", str(rule_id))
        status, actual = _condition_status(rule, profile)
        required = rule.get("required", True)
        core_keys_valid = all(key in rule for key in ("rule_id", "field", "operator", "value", "label", "source_url", "required"))
        if not core_keys_valid or not isinstance(required, bool):
            status, actual = "unknown", None
            malformed = True
        condition = {
            "rule_id": rule_id,
            "label": label,
            "field": field,
            "status": status,
            "actual_value": actual,
            "expected_value": rule.get("value"),
            "source_url": rule.get("source_url") if isinstance(rule.get("source_url"), str) else "",
            "explanation": (
                f"{label}: {status.replace('_', ' ')}."
                if isinstance(label, str)
                else f"Eligibility condition {index + 1}: {status.replace('_', ' ')}."
            ),
        }
        conditions.append(condition)
        if required is True and status == "unknown":
            missing.append({"field": field, "label": label})

    verification = scheme.get("verification_status")
    valid_evidence = bool(scheme.get("official_source_url")) and any(
        isinstance(rule, Mapping) and bool(rule.get("source_url")) for rule in rules
    )
    required_conditions = [
        condition for condition, rule in zip(conditions, rules)
        if isinstance(rule, Mapping) and rule.get("required", True) is True
    ]
    reasons = []
    if verification != "verified":
        status = "not_verified"
        reasons.append("This scheme record has not been verified.")
    elif malformed:
        status = "needs_more_information"
        reasons.append("Some eligibility rule data is incomplete or invalid.")
    elif not valid_evidence:
        status = "needs_more_information"
        reasons.append("Source evidence is incomplete, so a positive screening result is withheld.")
    elif not required_conditions:
        status = "needs_more_information"
        reasons.append("No required eligibility conditions are available to evaluate.")
    elif any(condition["status"] == "not_satisfied" for condition in required_conditions):
        status = "unlikely_match"
        reasons.append("At least one required condition is not satisfied by the available profile.")
    elif any(condition["status"] == "unknown" for condition in required_conditions):
        status = "needs_more_information"
        reasons.append("More information is needed to assess every required condition.")
    else:
        status = "potential_match"
        reasons.append("All recorded required conditions are satisfied; confirm details with the official source.")

    return {
        "scheme_id": scheme.get("scheme_id"),
        "scheme_name": scheme.get("name"),
        "status": status,
        "verification_status": verification if isinstance(verification, str) else "unknown",
        "conditions": conditions,
        "missing_information": missing,
        "reasons": reasons,
        "official_source_url": scheme.get("official_source_url") if isinstance(scheme.get("official_source_url"), str) else "",
        "application_url": scheme.get("application_url") if isinstance(scheme.get("application_url"), str) else "",
        "document_checklist": list(scheme.get("documents", [])) if isinstance(scheme.get("documents"), list) else [],
        "disclaimer": DISCLAIMER,
    }


def evaluate_all(profile: Mapping[str, Any], schemes: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate all input records and order them by screening status."""
    results = [evaluate_scheme(profile, scheme) for scheme in schemes]
    rank = {status: index for index, status in enumerate(STATUS_ORDER)}
    return sorted(results, key=lambda result: (rank.get(result["status"], len(rank)), str(result.get("scheme_id") or "")))


class Engine:
    """Application-level interface for loading and searching schemes."""

    def __init__(self, data_dir=None):
        if data_dir is None:
            self.schemes_path = DEFAULT_SCHEMES_PATH
        else:
            path = Path(data_dir)
            self.schemes_path = path / "schemes.json" if path.is_dir() or not path.suffix else path

    def load_schemes(self) -> list[dict]:
        return load_schemes(self.schemes_path)

    def search_schemes(self, query, user_profile=None):
        from .retriever import retrieve_relevant_schemes
        return retrieve_relevant_schemes(query, user_profile or {}, self.load_schemes())
