"""Tests for ``civiclens.engine`` (``evaluate_scheme`` / ``evaluate_all``).

The engine is deterministic Python, so it is tested for real here: no mocks, no
network and no API credentials. Every scheme below is a SYNTHETIC fixture and does
not describe a real programme.

The common contract requires the engine to *fail closed*. For malformed or
unsupported rule data the contract allows either an error or an explicit
``unknown`` condition, so those tests accept both and only forbid a pass.
"""

from __future__ import annotations

import copy
import itertools
import json
import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from civiclens.engine import evaluate_all, evaluate_scheme  # noqa: E402

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

STATUS_ORDER = ["potential_match", "needs_more_information", "unlikely_match", "not_verified"]
CONDITION_STATUSES = {"satisfied", "not_satisfied", "unknown"}

CONDITION_KEYS = {
    "rule_id",
    "label",
    "field",
    "status",
    "actual_value",
    "expected_value",
    "source_url",
    "explanation",
}
EVALUATION_KEYS = {
    "scheme_id",
    "scheme_name",
    "status",
    "verification_status",
    "conditions",
    "missing_information",
    "reasons",
    "official_source_url",
    "application_url",
    "document_checklist",
    "disclaimer",
}

SYNTHETIC_URL = "https://example.org/synthetic"


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Fail any test whose code under test tries to open a network connection."""
    attempts = []

    def _blocked(*args, **kwargs):
        attempts.append(args)
        raise OSError("Network access is blocked in unit tests")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    yield
    assert not attempts, f"Unit test attempted network access: {attempts}"


def make_profile(**overrides):
    profile = {field: None for field in PROFILE_FIELDS}
    profile.update(overrides)
    return profile


def make_rule(rule_id, field, operator, value, *, required=True, source_url=None, label=None):
    return {
        "rule_id": rule_id,
        "field": field,
        "operator": operator,
        "value": value,
        "label": label or f"Synthetic condition {rule_id}",
        "source_url": f"{SYNTHETIC_URL}/rules/{rule_id}" if source_url is None else source_url,
        "required": required,
    }


def make_scheme(scheme_id="synthetic_a", rules=None, **overrides):
    scheme = {
        "scheme_id": scheme_id,
        "name": f"Synthetic Scheme {scheme_id}",
        "category": "education",
        "summary": "Synthetic summary used only for testing.",
        "benefit_summary": "Synthetic benefit used only for testing.",
        "jurisdiction": "Synthetic Test Jurisdiction",
        "eligibility_rules": rules if rules is not None else [make_rule("state_rule", "state", "equals", "Tamil Nadu")],
        "documents": [],
        "application_url": f"{SYNTHETIC_URL}/apply/{scheme_id}",
        "official_source_url": f"{SYNTHETIC_URL}/guidelines/{scheme_id}",
        "source_title": "Synthetic guideline document",
        "verified_on": "2025-01-15",
        "verification_status": "verified",
        "notes": "Synthetic record; not real data.",
    }
    scheme.update(overrides)
    return scheme


def three_rule_scheme(scheme_id="synthetic_three", **overrides):
    rules = [
        make_rule("state_rule", "state", "equals", "Tamil Nadu"),
        make_rule("income_rule", "annual_family_income_inr", "lte", 300000),
        make_rule("age_rule", "age", "between", [17, 25]),
    ]
    return make_scheme(scheme_id, rules=rules, **overrides)


QUALIFYING_PROFILE = make_profile(state="Tamil Nadu", annual_family_income_inr=250000, age=19)


def condition_for(result, rule_id):
    matches = [condition for condition in result["conditions"] if condition["rule_id"] == rule_id]
    assert len(matches) == 1, f"expected exactly one condition for {rule_id!r}, got {matches}"
    return matches[0]


def safe_evaluate(profile, scheme):
    """Return ``(result, error)``; exactly one of the two is ``None``."""
    try:
        return evaluate_scheme(profile, scheme), None
    except Exception as error:  # noqa: BLE001 - the contract allows an explicit error
        return None, error


def assert_never_a_pass(profile, scheme, bad_rule_id=None):
    """Malformed data may raise or be reported as unknown, but must never pass."""
    result, error = safe_evaluate(profile, scheme)
    if error is not None:
        return  # an explicit error is an acceptable fail-closed outcome
    assert result["status"] != "potential_match", result
    if bad_rule_id is not None:
        for condition in result["conditions"]:
            if condition["rule_id"] == bad_rule_id:
                assert condition["status"] == "unknown", (
                    "malformed rule data must be an explicit unknown, never a pass or a made-up failure"
                )


def document_name(item):
    return item if isinstance(item, str) else item.get("name")


def dumped(value):
    return json.dumps(value, ensure_ascii=False, default=str).lower()


# --------------------------------------------------------------------------- #
# Result shape
# --------------------------------------------------------------------------- #
def test_evaluation_contains_every_contract_key():
    result = evaluate_scheme(QUALIFYING_PROFILE, three_rule_scheme())

    assert EVALUATION_KEYS <= set(result)
    assert result["status"] in STATUS_ORDER
    assert isinstance(result["conditions"], list)
    assert isinstance(result["missing_information"], list)
    assert isinstance(result["reasons"], list) and result["reasons"]
    assert all(isinstance(reason, str) and reason.strip() for reason in result["reasons"])
    assert isinstance(result["document_checklist"], list)
    for condition in result["conditions"]:
        assert CONDITION_KEYS <= set(condition)
        assert condition["status"] in CONDITION_STATUSES
        assert isinstance(condition["explanation"], str) and condition["explanation"].strip()


def test_evaluation_is_plain_json_serialisable_data():
    result = evaluate_scheme(QUALIFYING_PROFILE, three_rule_scheme())
    assert json.loads(json.dumps(result))["scheme_id"] == "synthetic_three"


def test_one_condition_per_rule_in_rule_order():
    scheme = three_rule_scheme()
    result = evaluate_scheme(QUALIFYING_PROFILE, scheme)
    assert [c["rule_id"] for c in result["conditions"]] == ["state_rule", "income_rule", "age_rule"]
    for condition, rule in zip(result["conditions"], scheme["eligibility_rules"]):
        assert condition["field"] == rule["field"]
        assert condition["label"] == rule["label"]


# --------------------------------------------------------------------------- #
# Every supported operator: satisfied / not satisfied / unknown
# --------------------------------------------------------------------------- #
OPERATOR_CASES = [
    # id, field, operator, rule value, profile value, expected condition status
    ("equals-text-match", "state", "equals", "Tamil Nadu", "Tamil Nadu", "satisfied"),
    ("equals-text-mismatch", "state", "equals", "Tamil Nadu", "Karnataka", "not_satisfied"),
    ("equals-text-unknown", "state", "equals", "Tamil Nadu", None, "unknown"),
    ("equals-number-match", "age", "equals", 18, 18, "satisfied"),
    ("equals-number-mismatch", "age", "equals", 18, 19, "not_satisfied"),
    ("equals-bool-true-match", "first_generation_student", "equals", True, True, "satisfied"),
    ("equals-bool-true-mismatch", "first_generation_student", "equals", True, False, "not_satisfied"),
    ("equals-bool-false-match", "first_generation_student", "equals", False, False, "satisfied"),
    ("equals-bool-unknown", "first_generation_student", "equals", True, None, "unknown"),
    ("in-match", "institution_type", "in", ["government", "government_aided"], "government_aided", "satisfied"),
    ("in-mismatch", "institution_type", "in", ["government", "government_aided"], "private", "not_satisfied"),
    ("in-unknown", "institution_type", "in", ["government", "government_aided"], None, "unknown"),
    ("in-number-match", "age", "in", [18, 19, 20], 19, "satisfied"),
    ("lte-below", "annual_family_income_inr", "lte", 300000, 250000, "satisfied"),
    ("lte-boundary", "annual_family_income_inr", "lte", 300000, 300000, "satisfied"),
    ("lte-above", "annual_family_income_inr", "lte", 300000, 300001, "not_satisfied"),
    ("lte-float", "annual_family_income_inr", "lte", 300000, 299999.5, "satisfied"),
    ("lte-zero", "annual_family_income_inr", "lte", 300000, 0, "satisfied"),
    ("lte-unknown", "annual_family_income_inr", "lte", 300000, None, "unknown"),
    ("gte-above", "age", "gte", 18, 19, "satisfied"),
    ("gte-boundary", "age", "gte", 18, 18, "satisfied"),
    ("gte-below", "age", "gte", 18, 17, "not_satisfied"),
    ("gte-unknown", "age", "gte", 18, None, "unknown"),
    ("between-inside", "age", "between", [17, 25], 20, "satisfied"),
    ("between-lower-bound", "age", "between", [17, 25], 17, "satisfied"),
    ("between-upper-bound", "age", "between", [17, 25], 25, "satisfied"),
    ("between-below", "age", "between", [17, 25], 16, "not_satisfied"),
    ("between-above", "age", "between", [17, 25], 26, "not_satisfied"),
    ("between-unknown", "age", "between", [17, 25], None, "unknown"),
    (
        "contains-match",
        "course_type",
        "contains",
        "Engineering",
        "B.E. Engineering (Computer Science)",
        "satisfied",
    ),
    ("contains-mismatch", "course_type", "contains", "Engineering", "Arts and Science", "not_satisfied"),
    ("contains-unknown", "course_type", "contains", "Engineering", None, "unknown"),
]

OVERALL_FOR_CONDITION = {
    "satisfied": "potential_match",
    "not_satisfied": "unlikely_match",
    "unknown": "needs_more_information",
}


@pytest.mark.parametrize(
    "field, operator, rule_value, profile_value, expected",
    [case[1:] for case in OPERATOR_CASES],
    ids=[case[0] for case in OPERATOR_CASES],
)
def test_supported_operator_condition_status(field, operator, rule_value, profile_value, expected):
    scheme = make_scheme(rules=[make_rule("only_rule", field, operator, rule_value)])
    profile = make_profile(**{field: profile_value})

    result = evaluate_scheme(profile, scheme)
    condition = condition_for(result, "only_rule")

    assert condition["status"] == expected
    assert condition["expected_value"] == rule_value
    if expected == "unknown":
        assert condition["actual_value"] is None
    else:
        assert condition["actual_value"] == profile_value
    assert result["status"] == OVERALL_FOR_CONDITION[expected]


# --------------------------------------------------------------------------- #
# Numeric / Boolean type safety (never coerce)
# --------------------------------------------------------------------------- #
UNSAFE_CASES = [
    # id, field, operator, rule value, profile value that must NOT satisfy the rule
    ("true-is-not-age-1-gte", "age", "gte", 1, True),
    ("true-is-not-age-1-lte", "age", "lte", 5, True),
    ("true-is-not-age-1-equals", "age", "equals", 1, True),
    ("true-is-not-age-1-between", "age", "between", [0, 2], True),
    ("true-is-not-age-1-in", "age", "in", [1, 2], True),
    ("false-is-not-zero-income", "annual_family_income_inr", "lte", 300000, False),
    ("false-is-not-zero-income-equals", "annual_family_income_inr", "equals", 0, False),
    ("one-is-not-true", "first_generation_student", "equals", True, 1),
    ("zero-is-not-false", "first_generation_student", "equals", False, 0),
    ("text-true-is-not-true", "first_generation_student", "equals", True, "true"),
    ("text-yes-is-not-true", "first_generation_student", "equals", True, "yes"),
    ("rule-true-is-not-one", "age", "equals", True, 1),
    ("numeric-string-age", "age", "gte", 18, "20"),
    ("numeric-string-income", "annual_family_income_inr", "lte", 300000, "250000"),
    ("nan-income", "annual_family_income_inr", "lte", 300000, float("nan")),
    ("negative-infinity-income", "annual_family_income_inr", "lte", 300000, float("-inf")),
    ("positive-infinity-income", "annual_family_income_inr", "gte", 1, float("inf")),
    ("list-is-not-text", "state", "equals", "Tamil Nadu", ["Tamil Nadu"]),
    ("number-is-not-text-contains", "course_type", "contains", "1", 123),
]


@pytest.mark.parametrize(
    "field, operator, rule_value, profile_value",
    [case[1:] for case in UNSAFE_CASES],
    ids=[case[0] for case in UNSAFE_CASES],
)
def test_unsafe_coercion_never_satisfies_a_condition(field, operator, rule_value, profile_value):
    scheme = make_scheme(rules=[make_rule("only_rule", field, operator, rule_value)])
    profile = make_profile(**{field: profile_value})

    result, error = safe_evaluate(profile, scheme)
    if error is not None:
        return  # failing loudly on an invalid profile value is acceptable
    assert condition_for(result, "only_rule")["status"] != "satisfied"
    assert result["status"] != "potential_match"


def test_unknown_boolean_is_not_treated_as_false_or_true():
    scheme = make_scheme(rules=[make_rule("fg", "first_generation_student", "equals", True)])
    result = evaluate_scheme(make_profile(first_generation_student=None), scheme)
    assert condition_for(result, "fg")["status"] == "unknown"
    assert result["status"] == "needs_more_information"


def test_unknown_income_is_not_treated_as_zero():
    scheme = make_scheme(rules=[make_rule("income", "annual_family_income_inr", "lte", 300000)])
    result = evaluate_scheme(make_profile(annual_family_income_inr=None), scheme)
    assert condition_for(result, "income")["status"] == "unknown"
    assert result["status"] != "potential_match"


# --------------------------------------------------------------------------- #
# Malformed or unsupported rule data never passes
# --------------------------------------------------------------------------- #
MALFORMED_RULES = {
    "unsupported-operator": {"operator": "regex", "value": "Tamil.*"},
    "code-as-operator": {"operator": "__import__('os').getcwd()", "value": 1},
    "unsupported-field": {"field": "aadhaar_number", "operator": "equals", "value": "x"},
    "between-one-item": {"field": "age", "operator": "between", "value": [18]},
    "between-three-items": {"field": "age", "operator": "between", "value": [18, 20, 25]},
    "between-scalar": {"field": "age", "operator": "between", "value": 18},
    "between-text-bounds": {"field": "age", "operator": "between", "value": ["a", "z"]},
    "between-boolean-bounds": {"field": "age", "operator": "between", "value": [False, True]},
    "lte-text-value": {"field": "age", "operator": "lte", "value": "abc"},
    "gte-boolean-value": {"field": "age", "operator": "gte", "value": True},
    "in-scalar-value": {"field": "institution_type", "operator": "in", "value": "government"},
}


@pytest.mark.parametrize("overrides", list(MALFORMED_RULES.values()), ids=list(MALFORMED_RULES))
def test_malformed_rule_never_produces_a_pass(overrides):
    profile = make_profile(
        state="Tamil Nadu",
        age=20,
        annual_family_income_inr=100000,
        institution_type="gov",
        course_type="Engineering",
    )
    bad_rule = {**make_rule("bad_rule", "age", "equals", 20), **overrides}
    good_rule = make_rule("good_rule", "state", "equals", "Tamil Nadu")
    scheme = make_scheme(rules=[good_rule, bad_rule])

    assert_never_a_pass(profile, scheme, bad_rule_id="bad_rule")


@pytest.mark.parametrize(
    "broken_key",
    ["field", "operator", "value"],
)
def test_rule_missing_a_core_key_never_produces_a_pass(broken_key):
    bad_rule = make_rule("bad_rule", "age", "gte", 18)
    del bad_rule[broken_key]
    scheme = make_scheme(rules=[make_rule("good_rule", "state", "equals", "Tamil Nadu"), bad_rule])
    assert_never_a_pass(make_profile(state="Tamil Nadu", age=30), scheme)


@pytest.mark.parametrize("bad_rule", ["age >= 18", None, 5, ["age", "gte", 18]], ids=["string", "null", "number", "list"])
def test_non_object_rule_never_produces_a_pass(bad_rule):
    scheme = make_scheme(rules=[make_rule("good_rule", "state", "equals", "Tamil Nadu"), bad_rule])
    assert_never_a_pass(make_profile(state="Tamil Nadu", age=30), scheme)


@pytest.mark.parametrize("bad_rules", [None, "age >= 18", {"rule_id": "x"}, 5], ids=["null", "string", "object", "number"])
def test_non_list_eligibility_rules_never_produce_a_pass(bad_rules):
    scheme = make_scheme(eligibility_rules=bad_rules)
    assert_never_a_pass(QUALIFYING_PROFILE, scheme)


def test_rule_text_is_never_executed(tmp_path):
    marker = tmp_path / "executed.txt"
    payload = f"__import__('pathlib').Path(r'{marker}').write_text('pwned')"
    rules = [
        make_rule("code_value", "state", "equals", payload),
        {**make_rule("code_operator", "age", "equals", 1), "operator": payload},
        make_rule("code_label", "age", "gte", 1, label=payload),
    ]
    scheme = make_scheme(rules=rules)

    safe_evaluate(make_profile(state="Tamil Nadu", age=20), scheme)

    assert not marker.exists(), "rule text must never be executed as code"


# --------------------------------------------------------------------------- #
# Overall status rules
# --------------------------------------------------------------------------- #
def test_all_required_conditions_satisfied_gives_potential_match():
    result = evaluate_scheme(QUALIFYING_PROFILE, three_rule_scheme())

    assert result["status"] == "potential_match"
    assert all(condition["status"] == "satisfied" for condition in result["conditions"])
    assert result["missing_information"] == []


@pytest.mark.parametrize(
    "profile",
    [
        make_profile(state="Karnataka", annual_family_income_inr=250000, age=19),
        make_profile(state="Tamil Nadu", annual_family_income_inr=450000, age=19),
        make_profile(state="Tamil Nadu", annual_family_income_inr=250000, age=40),
    ],
    ids=["state-fails", "income-fails", "age-fails"],
)
def test_one_failed_required_condition_gives_unlikely_match(profile):
    result = evaluate_scheme(profile, three_rule_scheme())
    assert result["status"] == "unlikely_match"
    assert any(condition["status"] == "not_satisfied" for condition in result["conditions"])


def test_known_failure_takes_precedence_over_unknown_conditions():
    profile = make_profile(state="Karnataka", annual_family_income_inr=None, age=None)
    result = evaluate_scheme(profile, three_rule_scheme())
    assert result["status"] == "unlikely_match"


def test_unknown_required_condition_without_failures_needs_more_information():
    profile = make_profile(state="Tamil Nadu", annual_family_income_inr=None, age=19)
    result = evaluate_scheme(profile, three_rule_scheme())

    assert result["status"] == "needs_more_information"
    assert result["missing_information"], "missing information must be listed"
    assert "annual_family_income_inr" in dumped(result["missing_information"]) or (
        "synthetic condition income_rule" in dumped(result["missing_information"])
    )


def test_missing_information_covers_every_unknown_required_field():
    result = evaluate_scheme(make_profile(), three_rule_scheme())
    text = dumped(result["missing_information"])
    for field, label in (
        ("state", "synthetic condition state_rule"),
        ("annual_family_income_inr", "synthetic condition income_rule"),
        ("age", "synthetic condition age_rule"),
    ):
        assert field in text or label in text, f"{field} should be reported as missing"
    assert result["status"] == "needs_more_information"


@pytest.mark.parametrize("profile", [{}, {"age": 19}, {"state": "Tamil Nadu"}], ids=["empty", "age-only", "state-only"])
def test_profile_with_absent_keys_is_treated_as_unknown(profile):
    result = evaluate_scheme(profile, three_rule_scheme())
    assert result["status"] == "needs_more_information"


def test_optional_conditions_do_not_block_or_fail_a_potential_match():
    rules = [
        make_rule("required_state", "state", "equals", "Tamil Nadu"),
        make_rule("optional_failed", "age", "gte", 60, required=False),
        make_rule("optional_unknown", "course_type", "contains", "Engineering", required=False),
    ]
    scheme = make_scheme(rules=rules)

    result = evaluate_scheme(make_profile(state="Tamil Nadu", age=20), scheme)

    assert result["status"] == "potential_match"
    assert condition_for(result, "optional_failed")["status"] == "not_satisfied"
    assert condition_for(result, "optional_unknown")["status"] == "unknown"


def test_every_required_criterion_must_be_satisfied_for_a_match():
    rules = [make_rule(f"rule_{index}", "age", "gte", 10 + index) for index in range(5)]
    scheme = make_scheme(rules=rules)

    assert evaluate_scheme(make_profile(age=14), scheme)["status"] == "potential_match"
    assert evaluate_scheme(make_profile(age=13), scheme)["status"] == "unlikely_match"


# --------------------------------------------------------------------------- #
# Verification status handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", ["needs_review", "demo_only"])
def test_unverified_scheme_is_not_verified_even_when_every_condition_passes(status):
    scheme = three_rule_scheme(verification_status=status)

    result = evaluate_scheme(QUALIFYING_PROFILE, scheme)

    assert result["status"] == "not_verified"
    assert result["verification_status"] == status


@pytest.mark.parametrize("status", ["needs_review", "demo_only"])
def test_unverified_scheme_status_wins_over_failed_and_unknown_conditions(status):
    scheme = three_rule_scheme(verification_status=status)
    assert evaluate_scheme(make_profile(state="Karnataka"), scheme)["status"] == "not_verified"
    assert evaluate_scheme(make_profile(), scheme)["status"] == "not_verified"


@pytest.mark.parametrize("status", ["approved", "", None, 5, "Verified "], ids=["approved", "empty", "null", "number", "padded"])
def test_unrecognised_verification_status_fails_closed(status):
    scheme = three_rule_scheme(verification_status=status)
    result, error = safe_evaluate(QUALIFYING_PROFILE, scheme)
    if error is None:
        assert result["status"] == "not_verified"


def test_missing_verification_status_key_fails_closed():
    scheme = three_rule_scheme()
    del scheme["verification_status"]
    result, error = safe_evaluate(QUALIFYING_PROFILE, scheme)
    if error is None:
        assert result["status"] != "potential_match"


def test_unverified_results_still_carry_every_contract_key():
    result = evaluate_scheme(QUALIFYING_PROFILE, three_rule_scheme(verification_status="demo_only"))
    assert EVALUATION_KEYS <= set(result)
    assert isinstance(result["conditions"], list)
    assert result["disclaimer"].strip()


def test_verified_record_without_any_evidence_never_produces_a_positive_match():
    rules = [
        make_rule("state_rule", "state", "equals", "Tamil Nadu", source_url=""),
        make_rule("income_rule", "annual_family_income_inr", "lte", 300000, source_url=""),
    ]
    scheme = make_scheme(
        rules=rules,
        official_source_url="",
        application_url="",
        source_title="",
        verified_on="",
    )
    result, error = safe_evaluate(QUALIFYING_PROFILE, scheme)
    if error is None:
        assert result["status"] != "potential_match"


# --------------------------------------------------------------------------- #
# No meaningful criteria
# --------------------------------------------------------------------------- #
def test_verified_scheme_with_no_rules_is_never_a_potential_match():
    result = evaluate_scheme(QUALIFYING_PROFILE, make_scheme(eligibility_rules=[]))
    assert result["status"] != "potential_match"
    assert result["reasons"], "the reason for withholding a positive result must be explained"


def test_scheme_whose_rules_are_all_optional_is_never_a_potential_match():
    rules = [make_rule("optional_only", "state", "equals", "Tamil Nadu", required=False)]
    result = evaluate_scheme(make_profile(state="Tamil Nadu"), make_scheme(rules=rules))
    assert result["status"] != "potential_match"


# --------------------------------------------------------------------------- #
# Evidence, provenance and explanations
# --------------------------------------------------------------------------- #
def test_rule_level_source_urls_and_explanations_are_retained():
    rules = [
        make_rule("state_rule", "state", "equals", "Tamil Nadu", source_url=f"{SYNTHETIC_URL}/rule-a"),
        make_rule("income_rule", "annual_family_income_inr", "lte", 300000, source_url=f"{SYNTHETIC_URL}/rule-b"),
        make_rule("age_rule", "age", "gte", 60, source_url=f"{SYNTHETIC_URL}/rule-c"),
    ]
    result = evaluate_scheme(
        make_profile(state="Tamil Nadu", annual_family_income_inr=None, age=20),
        make_scheme(rules=rules),
    )

    assert condition_for(result, "state_rule")["source_url"] == f"{SYNTHETIC_URL}/rule-a"
    assert condition_for(result, "income_rule")["source_url"] == f"{SYNTHETIC_URL}/rule-b"
    assert condition_for(result, "age_rule")["source_url"] == f"{SYNTHETIC_URL}/rule-c"
    for condition in result["conditions"]:  # satisfied, unknown and not_satisfied all explained
        assert isinstance(condition["explanation"], str) and condition["explanation"].strip()


def test_scheme_level_provenance_is_retained():
    scheme = three_rule_scheme()
    result = evaluate_scheme(QUALIFYING_PROFILE, scheme)

    assert result["scheme_id"] == scheme["scheme_id"]
    assert result["scheme_name"] == scheme["name"]
    assert result["verification_status"] == "verified"
    assert result["official_source_url"] == scheme["official_source_url"]
    assert result["application_url"] == scheme["application_url"]


def test_document_checklist_lists_every_scheme_document():
    documents = [
        {"name": "Synthetic Income Certificate", "required": True, "source_url": f"{SYNTHETIC_URL}/docs/1"},
        {"name": "Synthetic Bonafide Certificate", "required": True, "source_url": f"{SYNTHETIC_URL}/docs/2"},
        {"name": "Synthetic Passport Photo", "required": False, "source_url": ""},
    ]
    result = evaluate_scheme(QUALIFYING_PROFILE, three_rule_scheme(documents=documents))

    names = [document_name(item) for item in result["document_checklist"]]
    assert names == [document["name"] for document in documents]


def test_potential_match_is_not_presented_as_guaranteed_approval():
    result = evaluate_scheme(QUALIFYING_PROFILE, three_rule_scheme())

    assert result["status"] == "potential_match"
    disclaimer = result["disclaimer"].lower()
    assert "not" in disclaimer
    assert any(word in disclaimer for word in ("guarantee", "official", "preliminary", "screening"))
    claims = dumped(result["reasons"]) + dumped([c["explanation"] for c in result["conditions"]])
    affirmative_claims = (
        "you are eligible",
        "you are entitled",
        "you qualify",
        "you will receive",
        "you will be awarded",
        "have been approved",
        "will be approved",
        "is guaranteed",
    )
    for forbidden in affirmative_claims:
        assert forbidden not in claims


@pytest.mark.parametrize(
    "profile, status",
    [
        (QUALIFYING_PROFILE, "verified"),
        (make_profile(), "verified"),
        (make_profile(state="Karnataka"), "verified"),
        (QUALIFYING_PROFILE, "demo_only"),
    ],
    ids=["match", "unknown", "unlikely", "unverified"],
)
def test_disclaimer_is_present_for_every_outcome(profile, status):
    result = evaluate_scheme(profile, three_rule_scheme(verification_status=status))
    assert isinstance(result["disclaimer"], str) and result["disclaimer"].strip()


def test_evaluation_does_not_mutate_its_inputs():
    profile = copy.deepcopy(QUALIFYING_PROFILE)
    scheme = three_rule_scheme(documents=[{"name": "Doc", "required": True, "source_url": ""}])
    profile_before, scheme_before = copy.deepcopy(profile), copy.deepcopy(scheme)

    evaluate_scheme(profile, scheme)

    assert profile == profile_before
    assert scheme == scheme_before


# --------------------------------------------------------------------------- #
# evaluate_all: completeness and deterministic ordering
# --------------------------------------------------------------------------- #
def mixed_catalogue():
    return [
        three_rule_scheme("synthetic_match"),
        make_scheme(
            "synthetic_needs_info",
            rules=[make_rule("course_rule", "course_type", "contains", "Engineering")],
        ),
        make_scheme(
            "synthetic_unlikely",
            rules=[make_rule("state_rule", "state", "equals", "Elsewhere")],
        ),
        three_rule_scheme("synthetic_unverified", verification_status="needs_review"),
    ]


def test_evaluate_all_returns_one_result_per_scheme_sorted_by_status():
    results = evaluate_all(QUALIFYING_PROFILE, mixed_catalogue())

    assert [r["status"] for r in results] == STATUS_ORDER
    assert [r["scheme_id"] for r in results] == [
        "synthetic_match",
        "synthetic_needs_info",
        "synthetic_unlikely",
        "synthetic_unverified",
    ]


def test_evaluate_all_order_does_not_depend_on_input_order():
    catalogue = mixed_catalogue()
    for permutation in itertools.permutations(catalogue):
        results = evaluate_all(QUALIFYING_PROFILE, list(permutation))
        assert [r["status"] for r in results] == STATUS_ORDER


def test_evaluate_all_is_repeatable_and_consistent_with_evaluate_scheme():
    catalogue = mixed_catalogue()

    first = evaluate_all(QUALIFYING_PROFILE, catalogue)
    second = evaluate_all(QUALIFYING_PROFILE, catalogue)

    assert first == second
    assert len({r["scheme_id"] for r in first}) == len(catalogue)
    for scheme in catalogue:
        direct = evaluate_scheme(QUALIFYING_PROFILE, scheme)
        assert direct in first


def test_evaluate_all_keeps_unverified_schemes_visible_and_last():
    results = evaluate_all(QUALIFYING_PROFILE, mixed_catalogue())
    assert results[-1]["status"] == "not_verified"
    assert results[-1]["scheme_id"] == "synthetic_unverified"


def test_evaluate_all_on_empty_catalogue_is_empty():
    assert evaluate_all(QUALIFYING_PROFILE, []) == []


def test_evaluate_all_does_not_mutate_inputs():
    profile = copy.deepcopy(QUALIFYING_PROFILE)
    catalogue = mixed_catalogue()
    profile_before, catalogue_before = copy.deepcopy(profile), copy.deepcopy(catalogue)

    evaluate_all(profile, catalogue)

    assert profile == profile_before
    assert catalogue == catalogue_before
