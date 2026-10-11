"""Tests for ``civiclens.data_loader.load_schemes``.

All scheme records in this file are SYNTHETIC test fixtures. They never describe
a real programme, and no test here needs network access or API credentials.

The common contract fixes the *shape* of the catalogue but not the exception
class raised for bad data. These tests therefore require a *clear* error (an
exception with a useful message that is not an accidental crash such as
``KeyError`` or ``AttributeError``) rather than one specific class.
"""

from __future__ import annotations

import json
import socket
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from civiclens.data_loader import load_schemes  # noqa: E402

DEFAULT_CATALOGUE = REPO_ROOT / "data" / "schemes.json"

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

REQUIRED_SCHEME_KEYS = (
    "scheme_id",
    "name",
    "category",
    "summary",
    "benefit_summary",
    "jurisdiction",
    "eligibility_rules",
    "documents",
    "application_url",
    "official_source_url",
    "source_title",
    "verified_on",
    "verification_status",
    "notes",
)

ALLOWED_VERIFICATION_STATUSES = {"verified", "needs_review", "demo_only"}
RELATION_TYPES = ["compatible", "restricted", "prerequisite", "complementary", "unknown"]

SYNTHETIC_URL = "https://example.org/synthetic"

# Errors that indicate an unvalidated access rather than a deliberate,
# descriptive validation failure.
ACCIDENTAL_ERRORS = (
    AttributeError,
    KeyError,
    IndexError,
    NameError,
    ImportError,
    ZeroDivisionError,
    RecursionError,
)


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


@contextmanager
def expect_clear_error(*hints):
    """Require a deliberate, descriptive error.

    ``hints`` are optional case-insensitive fragments; the message must contain
    at least one of them so that the person fixing the data can find the problem.
    """
    with pytest.raises(Exception) as caught:
        yield
    error = caught.value
    assert not isinstance(error, ACCIDENTAL_ERRORS), (
        f"Expected a clear validation error, got an accidental crash: {error!r}"
    )
    message = str(error)
    assert message.strip(), "Validation errors must carry a non-empty message"
    if hints:
        lowered = message.lower()
        assert any(hint.lower() in lowered for hint in hints), (
            f"Error message should identify the problem (one of {hints}); got: {message!r}"
        )


def make_rule(rule_id="income_limit", field="annual_family_income_inr", operator="lte", value=300000, **overrides):
    rule = {
        "rule_id": rule_id,
        "field": field,
        "operator": operator,
        "value": value,
        "label": f"Synthetic condition on {field}",
        "source_url": f"{SYNTHETIC_URL}/rules",
        "required": True,
    }
    rule.update(overrides)
    return rule


def make_document(name="Synthetic Certificate", **overrides):
    document = {"name": name, "required": True, "source_url": f"{SYNTHETIC_URL}/documents"}
    document.update(overrides)
    return document


def make_step(step_id="step_one", **overrides):
    step = {
        "step_id": step_id,
        "title": f"Synthetic task {step_id}",
        "description": "A synthetic task used only in tests.",
        "depends_on": [],
        "source_url": f"{SYNTHETIC_URL}/steps",
        "required": True,
    }
    step.update(overrides)
    return step


def make_relationship(target="synthetic_b", relation_type="unknown", **overrides):
    relationship = {
        "target_scheme_id": target,
        "relation_type": relation_type,
        "source_url": "",
        "source_title": "",
        "verified_on": "",
        "verification_status": "needs_review",
        "notes": "No authoritative compatibility evidence available",
    }
    relationship.update(overrides)
    return relationship


def make_scheme(scheme_id="synthetic_a", **overrides):
    scheme = {
        "scheme_id": scheme_id,
        "name": f"Synthetic Scheme {scheme_id}",
        "category": "education",
        "summary": "Synthetic summary used only for testing.",
        "benefit_summary": "Synthetic benefit description used only for testing.",
        "jurisdiction": "Synthetic Test Jurisdiction",
        "eligibility_rules": [make_rule()],
        "documents": [],
        "application_url": f"{SYNTHETIC_URL}/apply",
        "official_source_url": f"{SYNTHETIC_URL}/guidelines",
        "source_title": "Synthetic guideline document",
        "verified_on": "2025-01-15",
        "verification_status": "verified",
        "notes": "Synthetic record; not real data.",
    }
    scheme.update(overrides)
    return scheme


def write_catalogue(tmp_path, payload, name="schemes.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_records(tmp_path, *schemes):
    return load_schemes(write_catalogue(tmp_path, list(schemes)))


def assert_load_fails(tmp_path, payload, *hints):
    path = write_catalogue(tmp_path, payload)
    with expect_clear_error(*hints):
        load_schemes(path)


# --------------------------------------------------------------------------- #
# Valid loading
# --------------------------------------------------------------------------- #
def test_loads_valid_json_array_preserving_every_field(tmp_path):
    originals = [
        make_scheme("synthetic_a"),
        make_scheme("synthetic_b", name="Synthetic Beta Scheme", verification_status="needs_review"),
    ]

    loaded = load_records(tmp_path, *originals)

    assert isinstance(loaded, list)
    assert len(loaded) == 2
    by_id = {record["scheme_id"]: record for record in loaded}
    assert set(by_id) == {"synthetic_a", "synthetic_b"}
    for original in originals:
        record = by_id[original["scheme_id"]]
        assert isinstance(record, dict)
        for key, value in original.items():
            assert record[key] == value, f"field {key!r} was altered by the loader"


def test_every_required_key_is_present_on_loaded_records(tmp_path):
    (record,) = load_records(tmp_path, make_scheme())
    for key in REQUIRED_SCHEME_KEYS:
        assert key in record


def test_custom_path_accepts_str_and_pathlib(tmp_path):
    path = write_catalogue(tmp_path, [make_scheme()])
    assert load_schemes(path) == load_schemes(str(path))


def test_utf8_text_round_trips(tmp_path):
    scheme = make_scheme(name="சோதனைத் திட்டம் (synthetic)", jurisdiction="தமிழ்நாடு (synthetic)")
    (record,) = load_records(tmp_path, scheme)
    assert record["name"] == scheme["name"]
    assert record["jurisdiction"] == scheme["jurisdiction"]


def test_loader_does_not_promote_unverified_statuses(tmp_path):
    loaded = load_records(
        tmp_path,
        make_scheme("synthetic_review", verification_status="needs_review"),
        make_scheme("synthetic_demo", verification_status="demo_only"),
    )
    statuses = {record["scheme_id"]: record["verification_status"] for record in loaded}
    assert statuses == {"synthetic_review": "needs_review", "synthetic_demo": "demo_only"}


@pytest.mark.parametrize("value", [[17, 25], [17.5, 25.5], [0, 0]], ids=["ints", "floats", "equal-bounds"])
def test_between_with_two_numbers_is_valid(tmp_path, value):
    rule = make_rule("age_band", "age", "between", value)
    (record,) = load_records(tmp_path, make_scheme(eligibility_rules=[rule]))
    assert record["eligibility_rules"][0]["value"] == value


@pytest.mark.parametrize(
    "field, operator, value",
    [
        ("state", "equals", "Synthetic Test State"),
        ("institution_type", "in", ["government", "government_aided"]),
        ("annual_family_income_inr", "lte", 250000),
        ("age", "gte", 18),
        ("age", "between", [17, 25]),
        ("course_type", "contains", "Engineering"),
        ("first_generation_student", "equals", True),
    ],
)
def test_every_supported_operator_and_a_matching_field_loads(tmp_path, field, operator, value):
    rule = make_rule("rule_x", field, operator, value)
    (record,) = load_records(tmp_path, make_scheme(eligibility_rules=[rule]))
    assert record["eligibility_rules"][0]["operator"] == operator


@pytest.mark.parametrize("field", PROFILE_FIELDS)
def test_every_profile_field_is_accepted_in_rules(tmp_path, field):
    rule = make_rule("rule_x", field, "equals", "synthetic")
    (record,) = load_records(tmp_path, make_scheme(eligibility_rules=[rule]))
    assert record["eligibility_rules"][0]["field"] == field


def test_valid_documents_including_reuse_group_load(tmp_path):
    documents = [
        make_document("Synthetic Income Proof", reuse_group="synthetic_income_proof"),
        make_document("Synthetic ID Copy", reuse_group=""),
        make_document("Synthetic Photo", required=False),
    ]
    (record,) = load_records(tmp_path, make_scheme(documents=documents))
    assert record["documents"] == documents


def test_valid_application_steps_with_dependencies_load(tmp_path):
    steps = [
        make_step("collect"),
        make_step("register", depends_on=["collect"]),
        make_step("submit", depends_on=["collect", "register"]),
    ]
    (record,) = load_records(tmp_path, make_scheme(application_steps=steps))
    assert [step["step_id"] for step in record["application_steps"]] == ["collect", "register", "submit"]


def test_empty_application_steps_list_is_valid(tmp_path):
    (record,) = load_records(tmp_path, make_scheme(application_steps=[]))
    assert record["application_steps"] == []


@pytest.mark.parametrize("relation_type", RELATION_TYPES)
def test_every_relation_type_loads_with_evidence(tmp_path, relation_type):
    relationship = make_relationship(
        "synthetic_b",
        relation_type,
        source_url=f"{SYNTHETIC_URL}/relationship-notice",
        source_title="Synthetic relationship notice",
        verified_on="2025-02-01",
        verification_status="verified",
        notes="Synthetic evidence note.",
    )
    records = load_records(
        tmp_path,
        make_scheme("synthetic_a", relationships=[relationship]),
        make_scheme("synthetic_b"),
    )
    loaded_a = next(r for r in records if r["scheme_id"] == "synthetic_a")
    assert loaded_a["relationships"][0]["relation_type"] == relation_type


def test_unverified_relationship_status_is_not_promoted(tmp_path):
    relationship = make_relationship("synthetic_b", "compatible", verification_status="needs_review")
    records = load_records(
        tmp_path,
        make_scheme("synthetic_a", relationships=[relationship]),
        make_scheme("synthetic_b"),
    )
    loaded_a = next(r for r in records if r["scheme_id"] == "synthetic_a")
    assert loaded_a["relationships"][0]["verification_status"] == "needs_review"


# --------------------------------------------------------------------------- #
# Default catalogue resolution
# --------------------------------------------------------------------------- #
def test_default_path_does_not_depend_on_current_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # an empty directory with no data/ folder

    from_none = load_schemes()
    from_keyword = load_schemes(path=None)

    assert isinstance(from_none, list) and from_none, "default catalogue must not be empty"
    assert from_none == from_keyword
    assert from_none == load_schemes(DEFAULT_CATALOGUE)


def test_shipped_catalogue_file_is_a_utf8_json_array():
    payload = json.loads(DEFAULT_CATALOGUE.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload, "data/schemes.json must contain at least one record"


def test_shipped_catalogue_passes_validation_and_has_unique_ids():
    records = load_schemes()
    ids = [record["scheme_id"] for record in records]
    assert len(ids) == len(set(ids))
    for record in records:
        for key in REQUIRED_SCHEME_KEYS:
            assert key in record
        assert record["verification_status"] in ALLOWED_VERIFICATION_STATUSES


def test_shipped_verified_records_carry_checkable_provenance():
    """A 'verified' record must point at a source and a real, non-future check date."""
    for record in load_schemes():
        if record["verification_status"] != "verified":
            continue
        scheme_id = record["scheme_id"]
        assert record["official_source_url"].startswith(("http://", "https://")), scheme_id
        assert record["source_title"].strip(), scheme_id
        checked_on = date.fromisoformat(record["verified_on"])  # raises on a malformed date
        assert checked_on <= date.today(), f"{scheme_id}: verified_on is in the future"
        assert record["eligibility_rules"], f"{scheme_id}: verified record has no eligibility rules"


# --------------------------------------------------------------------------- #
# File- and root-level failures
# --------------------------------------------------------------------------- #
def test_missing_file_raises_clear_error(tmp_path):
    with expect_clear_error("nope.json", "not found", "no such file", "missing", "exist"):
        load_schemes(tmp_path / "nope.json")


def test_directory_instead_of_file_raises_clear_error(tmp_path):
    with expect_clear_error():
        load_schemes(tmp_path)


@pytest.mark.parametrize(
    "text",
    ["{not json", "", "[{\"scheme_id\": ", "[1, 2,]", "<html>not json</html>"],
    ids=["garbage", "empty-file", "truncated", "trailing-comma", "html"],
)
def test_malformed_json_raises_clear_error(tmp_path, text):
    path = tmp_path / "schemes.json"
    path.write_text(text, encoding="utf-8")
    with expect_clear_error():
        load_schemes(path)


@pytest.mark.parametrize(
    "payload",
    [{}, {"schemes": []}, "text", 42, None, True],
    ids=["empty-object", "wrapped-object", "string", "number", "null", "boolean"],
)
def test_non_array_root_is_rejected(tmp_path, payload):
    assert_load_fails(tmp_path, payload)


@pytest.mark.parametrize(
    "record",
    [1, "scheme", None, [], [make_scheme()]],
    ids=["number", "string", "null", "empty-list", "nested-list"],
)
def test_non_object_records_are_rejected(tmp_path, record):
    assert_load_fails(tmp_path, [record])


# --------------------------------------------------------------------------- #
# Required scheme fields and types
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("missing_key", REQUIRED_SCHEME_KEYS)
def test_missing_required_field_is_rejected(tmp_path, missing_key):
    scheme = make_scheme()
    del scheme[missing_key]
    assert_load_fails(tmp_path, [scheme], missing_key)


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("scheme_id", 123),
        ("scheme_id", ""),
        ("scheme_id", None),
        ("name", None),
        ("name", 5),
        ("category", 5),
        ("summary", ["not", "a", "string"]),
        ("benefit_summary", 1.5),
        ("jurisdiction", None),
        ("eligibility_rules", {}),
        ("eligibility_rules", "none"),
        ("eligibility_rules", None),
        ("documents", "passport"),
        ("documents", {}),
        ("documents", None),
        ("application_url", None),
        ("official_source_url", 7),
        ("source_title", None),
        ("verified_on", 20250115),
        ("notes", None),
        ("verification_status", 5),
    ],
)
def test_wrong_field_type_is_rejected(tmp_path, field, bad_value):
    assert_load_fails(tmp_path, [make_scheme(**{field: bad_value})], field)


def test_extension_fields_do_not_replace_required_fields(tmp_path):
    scheme = make_scheme(application_steps=[make_step()], relationships=[])
    del scheme["summary"]
    assert_load_fails(tmp_path, [scheme], "summary")


def test_duplicate_scheme_ids_are_rejected(tmp_path):
    payload = [make_scheme("synthetic_dup"), make_scheme("synthetic_dup", name="Another record")]
    assert_load_fails(tmp_path, payload, "synthetic_dup", "duplicate")


@pytest.mark.parametrize("bad_status", ["approved", "", "pending", "official", None, True])
def test_invalid_verification_status_is_rejected(tmp_path, bad_status):
    assert_load_fails(
        tmp_path,
        [make_scheme(verification_status=bad_status)],
        "verification_status",
        "status",
    )


# --------------------------------------------------------------------------- #
# Eligibility rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_field", ["aadhaar_number", "bank_account", "income", "__class__", ""])
def test_unsupported_rule_field_is_rejected(tmp_path, bad_field):
    rule = make_rule("rule_x", bad_field, "equals", "x")
    assert_load_fails(tmp_path, [make_scheme(eligibility_rules=[rule])], "field", bad_field or "field")


@pytest.mark.parametrize(
    "bad_operator",
    ["regex", "like", "exec", "not_equals", "eval", "==", ">=", "__import__('os').system('echo x')", ""],
)
def test_unsupported_rule_operator_is_rejected(tmp_path, bad_operator):
    rule = make_rule("rule_x", "age", bad_operator, 18)
    assert_load_fails(tmp_path, [make_scheme(eligibility_rules=[rule])], "operator", bad_operator or "operator")


@pytest.mark.parametrize(
    "bad_value",
    [18, "18-25", [18], [], [18, 20, 25], ["a", "z"], [True, False], [None, 5], {"min": 1, "max": 2}, None],
    ids=[
        "scalar",
        "string-range",
        "one-item",
        "empty-list",
        "three-items",
        "non-numeric",
        "booleans",
        "null-bound",
        "object",
        "null",
    ],
)
def test_malformed_between_value_is_rejected(tmp_path, bad_value):
    rule = make_rule("age_band", "age", "between", bad_value)
    assert_load_fails(
        tmp_path,
        [make_scheme(eligibility_rules=[rule])],
        "between",
        "value",
        "two",
        "range",
        "bound",
    )


@pytest.mark.parametrize("missing_key", ["rule_id", "field", "operator", "value", "label"])
def test_rule_missing_core_key_is_rejected(tmp_path, missing_key):
    rule = make_rule()
    del rule[missing_key]
    assert_load_fails(tmp_path, [make_scheme(eligibility_rules=[rule])], missing_key)


@pytest.mark.parametrize(
    "overrides",
    [{"required": "yes"}, {"required": 1}, {"required": None}, {"source_url": None}, {"source_url": 5}, {"label": None}],
    ids=["required-string", "required-int", "required-null", "url-null", "url-int", "label-null"],
)
def test_rule_with_wrong_types_is_rejected(tmp_path, overrides):
    rule = make_rule(**overrides)
    assert_load_fails(tmp_path, [make_scheme(eligibility_rules=[rule])])


@pytest.mark.parametrize("bad_rule", ["age >= 18", 5, None, ["age", "gte", 18]], ids=["string", "number", "null", "list"])
def test_non_object_rule_is_rejected(tmp_path, bad_rule):
    assert_load_fails(tmp_path, [make_scheme(eligibility_rules=[bad_rule])])


def test_duplicate_rule_ids_within_one_scheme_are_rejected(tmp_path):
    rules = [make_rule("same_id"), make_rule("same_id", "age", "gte", 18)]
    assert_load_fails(tmp_path, [make_scheme(eligibility_rules=rules)], "same_id", "duplicate", "rule_id")


def test_same_rule_id_in_different_schemes_is_allowed(tmp_path):
    records = load_records(
        tmp_path,
        make_scheme("synthetic_a", eligibility_rules=[make_rule("income_limit")]),
        make_scheme("synthetic_b", eligibility_rules=[make_rule("income_limit")]),
    )
    assert len(records) == 2


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_document",
    [
        {"required": True, "source_url": ""},
        {"name": "", "required": True, "source_url": ""},
        {"name": 5, "required": True, "source_url": ""},
        {"name": "Doc", "required": "yes", "source_url": ""},
        {"name": "Doc", "required": True, "source_url": None},
        {"name": "Doc", "required": True, "source_url": "", "reuse_group": 5},
        {"name": "Doc", "required": True, "source_url": "", "reuse_group": ["a"]},
        "Income certificate",
        None,
    ],
    ids=[
        "missing-name",
        "empty-name",
        "numeric-name",
        "required-string",
        "url-null",
        "reuse-group-int",
        "reuse-group-list",
        "bare-string",
        "null",
    ],
)
def test_invalid_document_records_are_rejected(tmp_path, bad_document):
    assert_load_fails(tmp_path, [make_scheme(documents=[bad_document])])


# --------------------------------------------------------------------------- #
# Application steps
# --------------------------------------------------------------------------- #
def test_application_steps_must_be_a_list(tmp_path):
    assert_load_fails(tmp_path, [make_scheme(application_steps={"step_id": "x"})], "application_steps")


@pytest.mark.parametrize(
    "bad_step",
    [
        {"title": "No id", "description": "", "depends_on": [], "source_url": "", "required": True},
        {"step_id": 5, "title": "Numeric id", "description": "", "depends_on": [], "source_url": "", "required": True},
        {"step_id": "", "title": "Empty id", "description": "", "depends_on": [], "source_url": "", "required": True},
        {"step_id": "s", "description": "", "depends_on": [], "source_url": "", "required": True},
        {"step_id": "s", "title": "T", "description": "", "depends_on": "other", "source_url": "", "required": True},
        {"step_id": "s", "title": "T", "description": "", "depends_on": [5], "source_url": "", "required": True},
        {"step_id": "s", "title": "T", "description": "", "depends_on": [], "source_url": "", "required": "yes"},
        "just a string",
        None,
    ],
    ids=[
        "missing-step-id",
        "numeric-step-id",
        "empty-step-id",
        "missing-title",
        "depends-on-string",
        "depends-on-non-string",
        "required-string",
        "bare-string",
        "null",
    ],
)
def test_invalid_application_step_records_are_rejected(tmp_path, bad_step):
    assert_load_fails(tmp_path, [make_scheme(application_steps=[bad_step])])


def test_duplicate_step_ids_within_a_scheme_are_rejected(tmp_path):
    steps = [make_step("same"), make_step("same")]
    assert_load_fails(tmp_path, [make_scheme(application_steps=steps)], "same", "duplicate", "step_id")


def test_dependency_on_unknown_step_is_rejected(tmp_path):
    steps = [make_step("a"), make_step("b", depends_on=["ghost_step"])]
    assert_load_fails(tmp_path, [make_scheme(application_steps=steps)], "ghost_step", "depends_on", "unknown")


def test_dependency_must_reference_the_same_scheme_not_another_scheme(tmp_path):
    other = make_scheme("synthetic_b", application_steps=[make_step("remote_step")])
    this = make_scheme("synthetic_a", application_steps=[make_step("local", depends_on=["remote_step"])])
    assert_load_fails(tmp_path, [this, other], "remote_step", "depends_on", "unknown")


# --------------------------------------------------------------------------- #
# Relationships
# --------------------------------------------------------------------------- #
def test_relationship_to_unknown_scheme_is_rejected(tmp_path):
    scheme = make_scheme("synthetic_a", relationships=[make_relationship("ghost_scheme")])
    assert_load_fails(tmp_path, [scheme], "ghost_scheme", "target", "unknown")


@pytest.mark.parametrize("bad_type", ["friends", "COMPATIBLE-ish", "", None, 5])
def test_invalid_relation_type_is_rejected(tmp_path, bad_type):
    relationship = make_relationship("synthetic_b", relation_type=bad_type)
    payload = [make_scheme("synthetic_a", relationships=[relationship]), make_scheme("synthetic_b")]
    assert_load_fails(tmp_path, payload, "relation_type", "relation")


def test_invalid_relationship_verification_status_is_rejected(tmp_path):
    relationship = make_relationship("synthetic_b", verification_status="approved")
    payload = [make_scheme("synthetic_a", relationships=[relationship]), make_scheme("synthetic_b")]
    assert_load_fails(tmp_path, payload, "verification_status", "status")


def test_relationship_missing_target_is_rejected(tmp_path):
    relationship = make_relationship("synthetic_b")
    del relationship["target_scheme_id"]
    payload = [make_scheme("synthetic_a", relationships=[relationship]), make_scheme("synthetic_b")]
    assert_load_fails(tmp_path, payload, "target_scheme_id", "target")


@pytest.mark.parametrize("bad_relationship", ["synthetic_b", None, 5], ids=["string", "null", "number"])
def test_non_object_relationship_is_rejected(tmp_path, bad_relationship):
    payload = [make_scheme("synthetic_a", relationships=[bad_relationship]), make_scheme("synthetic_b")]
    assert_load_fails(tmp_path, payload)


def test_relationships_must_be_a_list(tmp_path):
    payload = [make_scheme("synthetic_a", relationships={"target_scheme_id": "synthetic_b"}), make_scheme("synthetic_b")]
    assert_load_fails(tmp_path, payload, "relationships")
