"""Tests for ``civiclens.retriever.retrieve_relevant_schemes``.

The retriever is lightweight, local and deterministic, so it is tested for real:
no mocks, no network and no API credentials. All scheme records are SYNTHETIC
fixtures built from invented names so that term matching can be verified without
referring to any real programme.

The contract does not say what an empty or non-matching query must return, so
those tests only require a safe, deterministic list of original catalogue
records (never an exception and never an invented record).
"""

from __future__ import annotations

import copy
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from civiclens.engine import evaluate_all, evaluate_scheme  # noqa: E402
from civiclens.retriever import retrieve_relevant_schemes  # noqa: E402

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


def make_scheme(scheme_id, name, category, summary, benefit_summary, jurisdiction, rule_label, **overrides):
    scheme = {
        "scheme_id": scheme_id,
        "name": name,
        "category": category,
        "summary": summary,
        "benefit_summary": benefit_summary,
        "jurisdiction": jurisdiction,
        "eligibility_rules": [
            {
                "rule_id": f"{scheme_id}_rule",
                "field": "course_type",
                "operator": "contains",
                "value": "Engineering",
                "label": rule_label,
                "source_url": f"{SYNTHETIC_URL}/{scheme_id}/rules",
                "required": True,
            }
        ],
        "documents": [{"name": f"{name} form", "required": True, "source_url": f"{SYNTHETIC_URL}/{scheme_id}/docs"}],
        "application_url": f"{SYNTHETIC_URL}/{scheme_id}/apply",
        "official_source_url": f"{SYNTHETIC_URL}/{scheme_id}/guidelines",
        "source_title": f"{name} guidelines (synthetic)",
        "verified_on": "2025-01-15",
        "verification_status": "verified",
        "notes": "Synthetic record; not real data.",
    }
    scheme.update(overrides)
    return scheme


def catalogue():
    return [
        make_scheme(
            "alpha",
            "Alpha Merit Scholarship",
            "education",
            "Merit scholarship supporting tuition for undergraduate engineering students.",
            "Annual tuition reimbursement paid to the institution.",
            "Zephyria",
            "Student is enrolled in an undergraduate engineering course",
        ),
        make_scheme(
            "beta",
            "Beta Hostel Support",
            "housing",
            "Accommodation allowance for students living in hostels.",
            "Monthly hostel stipend.",
            "Zephyria",
            "Student resides in a recognised hostel",
        ),
        make_scheme(
            "gamma",
            "Gamma Skill Training Grant",
            "skills",
            "Funding for vocational certification courses.",
            "One-time training grant.",
            "Borealia",
            "Learner is enrolled in a polytechnic institution",
        ),
        make_scheme(
            "delta",
            "Delta Transport Concession",
            "transport",
            "Discounted bus travel pass for daily commuters.",
            "Concessional monthly pass.",
            "Borealia",
            "Applicant commutes daily to the institution",
        ),
    ]


def many_scholarships(count=8):
    return [
        make_scheme(
            f"scholarship_{index}",
            f"Scholarship Number{index}",
            "education",
            "A synthetic scholarship for students.",
            "Synthetic financial support.",
            "Zephyria",
            "Synthetic enrolment condition",
        )
        for index in range(count)
    ]


def ids(results):
    return [record["scheme_id"] for record in results]


def by_id(schemes):
    return {scheme["scheme_id"]: scheme for scheme in schemes}


def assert_only_original_records(results, schemes):
    originals = by_id(schemes)
    assert isinstance(results, list)
    assert len(ids(results)) == len(set(ids(results))), "a record must not be returned twice"
    for record in results:
        assert record["scheme_id"] in originals, f"invented record: {record.get('scheme_id')!r}"
        original = originals[record["scheme_id"]]
        for key, value in original.items():
            assert record[key] == value, f"field {key!r} of {record['scheme_id']} was altered"


# --------------------------------------------------------------------------- #
# Relevance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme_id", ["alpha", "beta", "gamma", "delta"])
def test_exact_scheme_name_ranks_that_scheme_first(scheme_id):
    schemes = catalogue()
    name = by_id(schemes)[scheme_id]["name"]

    results = retrieve_relevant_schemes(name, make_profile(), schemes)

    assert results, "an exact name query must return at least the named scheme"
    assert results[0]["scheme_id"] == scheme_id


def test_exact_name_match_is_case_insensitive():
    schemes = catalogue()
    results = retrieve_relevant_schemes("ALPHA MERIT SCHOLARSHIP", make_profile(), schemes)
    assert results[0]["scheme_id"] == "alpha"


@pytest.mark.parametrize(
    "query, expected_first",
    [
        ("hostel accommodation", "beta"),
        ("vocational certification", "gamma"),
        ("bus travel pass", "delta"),
        ("tuition", "alpha"),
        ("polytechnic", "gamma"),  # appears only in an eligibility label
        ("housing", "beta"),  # appears only in the category
        ("commutes", "delta"),  # appears only in an eligibility label
    ],
)
def test_related_terms_rank_the_matching_scheme_first(query, expected_first):
    schemes = catalogue()
    results = retrieve_relevant_schemes(query, make_profile(), schemes)
    assert results and results[0]["scheme_id"] == expected_first


def test_jurisdiction_terms_are_searched():
    schemes = catalogue()
    results = retrieve_relevant_schemes("Borealia", make_profile(), schemes, top_k=2)
    assert set(ids(results)) == {"gamma", "delta"}


def test_profile_context_prefers_the_scheme_that_matches_the_students_state():
    schemes = [
        make_scheme(
            "other_state",
            "Education Support One",
            "education",
            "Support for education costs.",
            "Financial support.",
            "Borealia",
            "Synthetic condition",
        ),
        make_scheme(
            "home_state",
            "Education Support Two",
            "education",
            "Support for education costs.",
            "Financial support.",
            "Zephyria",
            "Synthetic condition",
        ),
    ]

    results = retrieve_relevant_schemes("education support", make_profile(state="Zephyria"), schemes)

    assert ids(results)[0] == "home_state"


@pytest.mark.parametrize("profile", [make_profile(), {}], ids=["all-unknown", "empty-dict"])
def test_unknown_profile_values_are_handled(profile):
    results = retrieve_relevant_schemes("hostel", profile, catalogue())
    assert results[0]["scheme_id"] == "beta"


# --------------------------------------------------------------------------- #
# Determinism and ranking stability
# --------------------------------------------------------------------------- #
def test_repeated_calls_return_identical_rankings():
    schemes = many_scholarships()
    profile = make_profile(state="Zephyria", age=19)

    first = ids(retrieve_relevant_schemes("scholarship students", profile, schemes, top_k=8))
    for _ in range(5):
        assert ids(retrieve_relevant_schemes("scholarship students", profile, schemes, top_k=8)) == first


def test_smaller_top_k_is_a_prefix_of_the_full_ranking():
    schemes = many_scholarships()
    full = ids(retrieve_relevant_schemes("scholarship", make_profile(), schemes, top_k=8))
    for k in (1, 2, 3, 5):
        assert ids(retrieve_relevant_schemes("scholarship", make_profile(), schemes, top_k=k)) == full[:k]


HASH_SEED_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
from civiclens.retriever import retrieve_relevant_schemes
payload = json.load(sys.stdin)
output = {}
for query in payload["queries"]:
    results = retrieve_relevant_schemes(query, payload["profile"], payload["schemes"], top_k=10)
    output[query] = [record["scheme_id"] for record in results]
print(json.dumps(output))
"""


def test_ranking_does_not_depend_on_python_hash_randomisation():
    """Ties must be broken deterministically, not by set/dict hash order."""
    payload = {
        "schemes": many_scholarships(8) + catalogue(),
        "profile": make_profile(state="Zephyria"),
        "queries": ["scholarship", "support students", "hostel", "zzzz", ""],
    }
    outputs = []
    for seed in ("0", "1", "2", "12345"):
        completed = subprocess.run(
            [sys.executable, "-c", HASH_SEED_SCRIPT, str(REPO_ROOT)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(json.loads(completed.stdout))
    assert all(output == outputs[0] for output in outputs), "ranking changed with PYTHONHASHSEED"


# --------------------------------------------------------------------------- #
# Empty, odd and non-matching queries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("query", ["", "   ", "\n\t "], ids=["empty", "spaces", "whitespace"])
def test_empty_query_is_safe_and_deterministic(query):
    schemes = catalogue()
    profile = make_profile(state="Zephyria")

    first = retrieve_relevant_schemes(query, profile, schemes, top_k=3)
    second = retrieve_relevant_schemes(query, profile, schemes, top_k=3)

    assert len(first) <= 3
    assert_only_original_records(first, schemes)
    assert ids(first) == ids(second)


@pytest.mark.parametrize(
    "query",
    ["zzzzqqqq", "((((", "[a-z", "*", "\\", "?" * 50, "a" * 10_000, "கல்வி உதவித்தொகை", "<script>alert(1)</script>"],
    ids=["gibberish", "parens", "bracket", "star", "backslash", "question-marks", "very-long", "tamil", "markup"],
)
def test_unusual_queries_never_crash_or_invent_records(query):
    schemes = catalogue()
    results = retrieve_relevant_schemes(query, make_profile(), schemes)
    assert_only_original_records(results, schemes)
    assert len(results) <= 5


# --------------------------------------------------------------------------- #
# Catalogue and top_k boundaries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("query", ["", "scholarship", "anything at all"])
def test_empty_catalogue_returns_empty_list(query):
    assert retrieve_relevant_schemes(query, make_profile(), []) == []


@pytest.mark.parametrize("top_k", [0, -1, -100])
def test_non_positive_top_k_returns_nothing(top_k):
    assert retrieve_relevant_schemes("scholarship", make_profile(), many_scholarships(), top_k=top_k) == []


def test_top_k_one_returns_only_the_best_match():
    schemes = catalogue()
    results = retrieve_relevant_schemes("hostel accommodation", make_profile(), schemes, top_k=1)
    assert ids(results) == ["beta"]


def test_default_top_k_limits_results_to_five():
    results = retrieve_relevant_schemes("scholarship", make_profile(), many_scholarships(8))
    assert len(results) == 5


def test_top_k_larger_than_catalogue_returns_at_most_the_catalogue():
    schemes = catalogue()
    results = retrieve_relevant_schemes("scholarship hostel grant pass", make_profile(), schemes, top_k=1000)
    assert 0 < len(results) <= len(schemes)
    assert_only_original_records(results, schemes)


# --------------------------------------------------------------------------- #
# Provenance and integrity
# --------------------------------------------------------------------------- #
def test_returned_records_keep_all_original_metadata():
    schemes = catalogue()
    results = retrieve_relevant_schemes("Alpha Merit Scholarship", make_profile(), schemes)

    assert_only_original_records(results, schemes)
    alpha = results[0]
    assert alpha["official_source_url"] == f"{SYNTHETIC_URL}/alpha/guidelines"
    assert alpha["verified_on"] == "2025-01-15"
    assert alpha["verification_status"] == "verified"
    assert alpha["eligibility_rules"][0]["source_url"] == f"{SYNTHETIC_URL}/alpha/rules"
    assert alpha["documents"][0]["source_url"] == f"{SYNTHETIC_URL}/alpha/docs"


def test_retrieval_never_invents_records():
    schemes = catalogue()
    for query in ("scholarship", "hostel", "grant", "unrelated words entirely", ""):
        assert_only_original_records(retrieve_relevant_schemes(query, make_profile(), schemes), schemes)


def test_retrieval_does_not_mutate_inputs():
    schemes = catalogue()
    profile = make_profile(state="Zephyria", age=19)
    schemes_before, profile_before = copy.deepcopy(schemes), copy.deepcopy(profile)

    retrieve_relevant_schemes("scholarship hostel", profile, schemes)

    assert schemes == schemes_before
    assert profile == profile_before


# --------------------------------------------------------------------------- #
# Relevance never changes eligibility
# --------------------------------------------------------------------------- #
def test_an_exact_name_match_does_not_upgrade_an_unverified_scheme():
    unverified = make_scheme(
        "omega",
        "Omega Unverified Scholarship",
        "education",
        "Synthetic unverified scholarship.",
        "Synthetic benefit.",
        "Zephyria",
        "Student is enrolled in an undergraduate engineering course",
        verification_status="needs_review",
    )
    schemes = catalogue() + [unverified]
    profile = make_profile(course_type="Engineering")

    results = retrieve_relevant_schemes("Omega Unverified Scholarship", profile, schemes)

    assert results[0]["scheme_id"] == "omega"
    assert evaluate_scheme(profile, results[0])["status"] == "not_verified"


def test_a_scheme_the_student_fails_is_still_retrievable_and_still_unlikely():
    schemes = catalogue()
    profile = make_profile(course_type="Arts and Science")

    results = retrieve_relevant_schemes("Alpha Merit Scholarship", profile, schemes)

    assert results[0]["scheme_id"] == "alpha"
    assert evaluate_scheme(profile, results[0])["status"] == "unlikely_match"


def test_eligibility_statuses_do_not_depend_on_the_query_that_retrieved_a_scheme():
    schemes = catalogue()
    profile = make_profile(course_type="Engineering")
    expected = {result["scheme_id"]: result["status"] for result in evaluate_all(profile, schemes)}

    for query in ("scholarship", "hostel", "polytechnic", "bus pass", "", "zzzz"):
        for record in retrieve_relevant_schemes(query, profile, schemes, top_k=10):
            assert evaluate_scheme(profile, record)["status"] == expected[record["scheme_id"]]
