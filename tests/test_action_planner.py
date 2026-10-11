"""Tests for ``civiclens.action_planner``.

The planner is deterministic Python, so it is tested for real, together with the
real rule engine that produces its input evaluations. No mocks, no network and no
API credentials. All schemes are SYNTHETIC fixtures.

The contract returns the plan as an ordered list, so ordering assertions use list
positions. Where the contract leaves wording open (for example how a dependency
cycle is reported) the tests check for safe behaviour, not exact text.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
import socket
import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from civiclens.action_planner import build_action_plan, build_portfolio_action_plan  # noqa: E402
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

SYNTHETIC_URL = "https://example.org/synthetic"
REQUIRED_ACTION_KEYS = ("title", "description", "priority", "action_type", "source_url")
ACCIDENTAL_ERRORS = (AttributeError, KeyError, IndexError, NameError, ImportError, ZeroDivisionError, RecursionError)

FORBIDDEN_CLAIMS = (
    "you are eligible",
    "you are entitled",
    "you qualify",
    "apply now",
    "submit your application",
    "guaranteed",
    "will be approved",
    "have been approved",
    "confirmed eligible",
    "you will receive",
    "has been submitted",
    "successfully submitted",
)
DEADLINE_PATTERN = re.compile(r"\b(deadline|due|last date|closing date|apply by|submit by)\b[^.\n]{0,40}\d", re.IGNORECASE)
CYCLE_WORDS = ("cycle", "circular", "cyclic", "loop")


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


QUALIFYING = make_profile(state="Tamil Nadu", annual_family_income_inr=200000)


def make_rule(rule_id, field, operator, value, label=None):
    return {
        "rule_id": rule_id,
        "field": field,
        "operator": operator,
        "value": value,
        "label": label or f"Synthetic condition {rule_id}",
        "source_url": f"{SYNTHETIC_URL}/rules/{rule_id}",
        "required": True,
    }


def make_document(name, reuse_group=None, required=True, source_url=None):
    document = {
        "name": name,
        "required": required,
        "source_url": f"{SYNTHETIC_URL}/docs/{name.lower().replace(' ', '-')}" if source_url is None else source_url,
    }
    if reuse_group is not None:
        document["reuse_group"] = reuse_group
    return document


def make_step(step_id, title, depends_on=(), source_url=None, required=True):
    return {
        "step_id": step_id,
        "title": title,
        "description": "A synthetic task used only in tests.",
        "depends_on": list(depends_on),
        "source_url": f"{SYNTHETIC_URL}/steps/{step_id}" if source_url is None else source_url,
        "required": required,
    }


def make_scheme(scheme_id, rules=None, **overrides):
    scheme = {
        "scheme_id": scheme_id,
        "name": f"Synthetic Scheme {scheme_id}",
        "category": "education",
        "summary": "Synthetic summary used only for testing.",
        "benefit_summary": "Synthetic benefit used only for testing.",
        "jurisdiction": "Synthetic Test Jurisdiction",
        "eligibility_rules": rules if rules is not None else [make_rule(f"{scheme_id}_state", "state", "equals", "Tamil Nadu")],
        "documents": [],
        "application_url": f"{SYNTHETIC_URL}/{scheme_id}/apply",
        "official_source_url": f"{SYNTHETIC_URL}/{scheme_id}/guidelines",
        "source_title": "Synthetic guideline document",
        "verified_on": "2025-01-15",
        "verification_status": "verified",
        "notes": "Synthetic record; not real data.",
    }
    scheme.update(overrides)
    return scheme


def needs_income_scheme(scheme_id, **overrides):
    """A verified scheme whose only rule depends on income, which QUALIFYING lacks when blanked."""
    rule = make_rule(
        f"{scheme_id}_income",
        "annual_family_income_inr",
        "lte",
        300000,
        label="Annual family income is within the synthetic limit",
    )
    return make_scheme(scheme_id, rules=[rule], **overrides)


def portfolio(schemes, profile=QUALIFYING):
    evaluations = evaluate_all(profile, schemes)
    return build_portfolio_action_plan(evaluations, schemes, profile)


def single(scheme, profile=QUALIFYING):
    return build_action_plan(evaluate_scheme(profile, scheme), profile)


def of_type(plan, action_type):
    return [action for action in plan if action["action_type"] == action_type]


def text_of(action):
    return f"{action['title']} {action['description']}".lower()


def dumped(value):
    return json.dumps(value, ensure_ascii=False, default=str).lower()


def scheme_ids_of(action):
    return list(action.get("scheme_ids") or [])


def known_urls(schemes):
    urls = {""}
    for scheme in schemes:
        urls.update({scheme["official_source_url"], scheme["application_url"]})
        for rule in scheme["eligibility_rules"]:
            urls.add(rule["source_url"])
        for document in scheme["documents"]:
            urls.add(document["source_url"])
        for step in scheme.get("application_steps", []):
            urls.add(step["source_url"])
        for relationship in scheme.get("relationships", []):
            urls.add(relationship["source_url"])
    return urls


def step_positions(plan, tag):
    """Positions of step actions whose title (or, failing that, description) contains ``tag``."""
    steps = of_type(plan, "complete_application_step")
    by_title = [plan.index(action) for action in steps if tag.lower() in action["title"].lower()]
    if by_title:
        return by_title
    return [plan.index(action) for action in steps if tag.lower() in action["description"].lower()]


def single_step_position(plan, tag):
    positions = step_positions(plan, tag)
    assert len(positions) == 1, f"expected exactly one action for step {tag!r}, found {len(positions)}"
    return positions[0]


def run_with_timeout(function, *args, timeout=10.0):
    """Run ``function`` in a daemon thread so an infinite loop fails the test instead of hanging it."""
    outcome = {}

    def target():
        try:
            outcome["value"] = function(*args)
        except BaseException as error:  # noqa: BLE001 - re-raised below in the test thread
            outcome["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "planner did not terminate (possible infinite loop on a dependency cycle)"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def assert_valid_actions(plan):
    assert isinstance(plan, list)
    for action in plan:
        assert isinstance(action, dict)
        for key in REQUIRED_ACTION_KEYS:
            assert key in action, f"action is missing {key!r}: {action}"
        assert isinstance(action["title"], str) and action["title"].strip()
        assert isinstance(action["description"], str) and action["description"].strip()
        assert isinstance(action["action_type"], str) and action["action_type"].strip()
        assert isinstance(action["source_url"], str)
        assert action["priority"] is not None and not isinstance(action["priority"], bool)
        assert isinstance(action["priority"], (int, str))


def assert_no_unsupported_claims(plan):
    for action in plan:
        content = text_of(action) + " " + str(action.get("status", "")).lower()
        for phrase in FORBIDDEN_CLAIMS:
            assert phrase not in content, f"unsupported claim {phrase!r} in: {action}"
        assert not DEADLINE_PATTERN.search(content), f"invented deadline in: {action}"


# --------------------------------------------------------------------------- #
# Single-scheme plans
# --------------------------------------------------------------------------- #
def test_single_scheme_plan_for_a_potential_match_lists_documents_and_sources():
    scheme = make_scheme(
        "synthetic_a",
        documents=[make_document("Synthetic Income Certificate"), make_document("Synthetic Bonafide Letter")],
    )

    plan = single(scheme)

    assert plan, "a potential match should produce next steps"
    assert_valid_actions(plan)
    document_actions = of_type(plan, "prepare_document")
    for name in ("synthetic income certificate", "synthetic bonafide letter"):
        assert sum(name in text_of(action) for action in document_actions) == 1
    allowed = known_urls([scheme])
    assert all(action["source_url"] in allowed for action in plan)
    official_or_rule_urls = {scheme["official_source_url"]} | {r["source_url"] for r in scheme["eligibility_rules"]}
    assert any(action["source_url"] in official_or_rule_urls for action in plan), "authoritative links must be kept"


def test_single_scheme_plan_asks_for_missing_information_first():
    scheme = needs_income_scheme("synthetic_a", documents=[make_document("Synthetic Income Certificate")])
    profile = make_profile(state="Tamil Nadu")  # income unknown

    plan = single(scheme, profile)

    assert_valid_actions(plan)
    missing = of_type(plan, "answer_missing_information")
    assert missing, "unknown required information must become a task"
    assert any("income" in text_of(action) for action in missing)
    first_missing = plan.index(missing[0])
    for later in of_type(plan, "prepare_document") + of_type(plan, "complete_application_step"):
        assert plan.index(later) > first_missing


def test_conditions_are_checked_before_documents_are_prepared():
    scheme = make_scheme("synthetic_a", documents=[make_document("Synthetic Income Certificate")])

    plan = single(scheme)

    checks = of_type(plan, "verify_conditions")
    documents = of_type(plan, "prepare_document")
    if checks and documents:
        assert plan.index(checks[0]) < plan.index(documents[0])


def test_no_application_steps_are_given_for_a_scheme_the_student_does_not_match():
    scheme = make_scheme("synthetic_a", documents=[make_document("Synthetic Income Certificate")])
    plan = single(scheme, make_profile(state="Elsewhere"))

    assert_valid_actions(plan)
    assert of_type(plan, "complete_application_step") == []
    assert_no_unsupported_claims(plan)


# --------------------------------------------------------------------------- #
# Portfolio plans
# --------------------------------------------------------------------------- #
def test_portfolio_aggregates_actions_from_several_schemes():
    first = make_scheme("synthetic_a", documents=[make_document("Synthetic Income Certificate")])
    second = make_scheme("synthetic_b", documents=[make_document("Synthetic Bonafide Letter")])

    plan = portfolio([first, second])

    assert_valid_actions(plan)
    documents = of_type(plan, "prepare_document")
    assert any("income certificate" in text_of(action) for action in documents)
    assert any("bonafide letter" in text_of(action) for action in documents)
    covered = {scheme_id for action in documents for scheme_id in scheme_ids_of(action)}
    assert covered == {"synthetic_a", "synthetic_b"}


def test_missing_information_for_several_schemes_is_covered_in_the_plan():
    schemes = [needs_income_scheme("synthetic_a"), needs_income_scheme("synthetic_b")]
    profile = make_profile(state="Tamil Nadu")  # income unknown for both

    plan = portfolio(schemes, profile)

    missing = of_type(plan, "answer_missing_information")
    assert missing
    assert {scheme_id for action in missing for scheme_id in scheme_ids_of(action)} == {"synthetic_a", "synthetic_b"}
    assert all("income" in text_of(action) for action in missing)


def test_missing_information_precedes_that_schemes_other_tasks():
    scheme = needs_income_scheme(
        "synthetic_a",
        documents=[make_document("Synthetic Income Certificate")],
        application_steps=[make_step("only_step", "Create portal login (STEP-LOGIN)")],
    )
    profile = make_profile(state="Tamil Nadu")

    plan = portfolio([scheme], profile)

    missing = of_type(plan, "answer_missing_information")
    assert missing
    first_missing = plan.index(missing[0])
    for later in of_type(plan, "prepare_document") + of_type(plan, "complete_application_step"):
        assert plan.index(later) > first_missing


def test_empty_evaluations_give_an_empty_plan():
    assert build_portfolio_action_plan([], [], QUALIFYING) == []


def test_evaluation_for_a_scheme_missing_from_the_catalogue_is_handled_safely():
    scheme = make_scheme("synthetic_a", application_steps=[make_step("only_step", "Create portal login (STEP-LOGIN)")])
    evaluations = evaluate_all(QUALIFYING, [scheme])

    plan = build_portfolio_action_plan(evaluations, [], QUALIFYING)

    assert_valid_actions(plan)
    assert step_positions(plan, "STEP-LOGIN") == [], "steps must not be invented without catalogue data"


def test_portfolio_plan_is_deterministic():
    schemes = [
        make_scheme("synthetic_b", documents=[make_document("Synthetic Bonafide Letter")]),
        make_scheme("synthetic_a", documents=[make_document("Synthetic Income Certificate")]),
        needs_income_scheme("synthetic_c"),
    ]
    first = portfolio(schemes, make_profile(state="Tamil Nadu"))
    for _ in range(3):
        assert portfolio(schemes, make_profile(state="Tamil Nadu")) == first


def test_planning_does_not_mutate_its_inputs():
    schemes = [
        make_scheme(
            "synthetic_a",
            documents=[make_document("Synthetic Income Certificate", "grp")],
            application_steps=[make_step("one", "Step one (STEP-ONE)"), make_step("two", "Step two (STEP-TWO)", ["one"])],
        )
    ]
    profile = copy.deepcopy(QUALIFYING)
    evaluations = evaluate_all(profile, schemes)
    before = (copy.deepcopy(schemes), copy.deepcopy(profile), copy.deepcopy(evaluations))

    build_portfolio_action_plan(evaluations, schemes, profile)
    build_action_plan(evaluations[0], profile)

    assert (schemes, profile, evaluations) == before


def test_action_ids_are_unique_and_dependencies_refer_to_real_actions():
    schemes = [
        make_scheme(
            "synthetic_a",
            documents=[make_document("Synthetic Income Certificate")],
            application_steps=[make_step("one", "Step one (STEP-ONE)"), make_step("two", "Step two (STEP-TWO)", ["one"])],
        ),
        needs_income_scheme("synthetic_b"),
    ]
    plan = portfolio(schemes, make_profile(state="Tamil Nadu"))

    assert_valid_actions(plan)
    if all("action_id" in action for action in plan):
        action_ids = [action["action_id"] for action in plan]
        assert len(action_ids) == len(set(action_ids))
        for action in plan:
            for dependency in action.get("depends_on") or []:
                assert dependency in action_ids, f"{action['action_id']} depends on unknown action {dependency!r}"


# --------------------------------------------------------------------------- #
# Application-step prerequisites
# --------------------------------------------------------------------------- #
def chained_steps():
    """Declared in a deliberately scrambled order."""
    return [
        make_step("submit", "Submit application on portal (STEP-SUBMIT)", ["register", "copies"]),
        make_step("register", "Complete registration form (STEP-REGISTER)", ["login"]),
        make_step("copies", "Collect certified copies (STEP-COPIES)"),
        make_step("login", "Create portal login (STEP-LOGIN)"),
    ]


def assert_prerequisite_order(plan):
    login = single_step_position(plan, "STEP-LOGIN")
    register = single_step_position(plan, "STEP-REGISTER")
    copies = single_step_position(plan, "STEP-COPIES")
    submit = single_step_position(plan, "STEP-SUBMIT")
    assert login < register < submit
    assert copies < submit


def test_explicit_application_step_prerequisites_are_ordered_first():
    scheme = make_scheme("synthetic_a", application_steps=chained_steps())

    plan = portfolio([scheme])

    assert_valid_actions(plan)
    assert_prerequisite_order(plan)


def test_prerequisite_order_holds_for_every_declaration_order():
    steps = chained_steps()
    for permutation in itertools.permutations(steps):
        plan = portfolio([make_scheme("synthetic_a", application_steps=list(permutation))])
        assert_prerequisite_order(plan)


def test_step_actions_keep_their_own_source_links_and_scheme_ids():
    steps = chained_steps()
    scheme = make_scheme("synthetic_a", application_steps=steps)

    plan = portfolio([scheme])

    for step in steps:
        tag = step["title"].split("(")[-1].rstrip(")")
        action = plan[single_step_position(plan, tag)]
        assert action["source_url"] == step["source_url"]
        assert "synthetic_a" in scheme_ids_of(action)


def test_every_plan_url_comes_from_the_catalogue():
    schemes = [
        make_scheme(
            "synthetic_a",
            documents=[make_document("Synthetic Income Certificate", "grp")],
            application_steps=chained_steps(),
        ),
        make_scheme("synthetic_b", documents=[make_document("Synthetic Family Income Proof", "grp")]),
        needs_income_scheme("synthetic_c"),
    ]
    plan = portfolio(schemes, make_profile(state="Tamil Nadu"))

    allowed = known_urls(schemes)
    for action in plan:
        assert action["source_url"] in allowed, f"unknown URL in {action}"
        for scheme_id in scheme_ids_of(action):
            assert scheme_id in {"synthetic_a", "synthetic_b", "synthetic_c"}


# --------------------------------------------------------------------------- #
# Dependency cycles
# --------------------------------------------------------------------------- #
CYCLES = {
    "two-steps": [
        make_step("a", "Step A (STEP-CYC-A)", ["b"]),
        make_step("b", "Step B (STEP-CYC-B)", ["a"]),
    ],
    "three-steps": [
        make_step("a", "Step A (STEP-CYC-A)", ["c"]),
        make_step("b", "Step B (STEP-CYC-B)", ["a"]),
        make_step("c", "Step C (STEP-CYC-C)", ["b"]),
    ],
    "self-dependency": [make_step("a", "Step A (STEP-CYC-A)", ["a"])],
}


@pytest.mark.parametrize("steps", list(CYCLES.values()), ids=list(CYCLES))
def test_dependency_cycles_are_reported_safely_without_looping(steps):
    scheme = make_scheme(
        "synthetic_a",
        documents=[make_document("Synthetic Income Certificate")],
        application_steps=copy.deepcopy(steps),
    )
    evaluations = evaluate_all(QUALIFYING, [scheme])

    plan = run_with_timeout(build_portfolio_action_plan, evaluations, [scheme], QUALIFYING)

    assert_valid_actions(plan)
    for step in steps:
        tag = step["title"].split("(")[-1].rstrip(")")
        assert len(step_positions(plan, tag)) <= 1, "a step must not be scheduled more than once"
    assert any(word in dumped(plan) for word in CYCLE_WORDS), "the cycle must be reported, not silently dropped"
    assert any("income certificate" in text_of(action) for action in of_type(plan, "prepare_document"))


def test_dependency_on_an_undeclared_step_does_not_crash_the_planner():
    scheme = make_scheme(
        "synthetic_a",
        application_steps=[make_step("a", "Step A (STEP-DANGLING)", ["ghost_step"])],
    )
    evaluations = evaluate_all(QUALIFYING, [scheme])

    try:
        plan = run_with_timeout(build_portfolio_action_plan, evaluations, [scheme], QUALIFYING)
    except Exception as error:  # noqa: BLE001 - a clear, handled error is acceptable
        assert not isinstance(error, ACCIDENTAL_ERRORS), f"accidental crash: {error!r}"
        assert str(error).strip()
        return
    assert_valid_actions(plan)


# --------------------------------------------------------------------------- #
# Document deduplication (only through a shared, non-empty reuse_group)
# --------------------------------------------------------------------------- #
def two_scheme_plan(first_document, second_document):
    first = make_scheme("synthetic_a", documents=[first_document])
    second = make_scheme("synthetic_b", documents=[second_document])
    return portfolio([first, second]), [first, second]


def test_documents_with_the_same_non_empty_reuse_group_are_merged():
    plan, schemes = two_scheme_plan(
        make_document("Income Certificate (wording from scheme A)", "shared_income_proof"),
        make_document("Family Income Proof (wording from scheme B)", "shared_income_proof"),
    )

    documents = of_type(plan, "prepare_document")

    assert len(documents) == 1
    assert set(scheme_ids_of(documents[0])) == {"synthetic_a", "synthetic_b"}
    assert documents[0]["source_url"] in known_urls(schemes)


def test_only_the_shared_documents_are_merged():
    first = make_scheme(
        "synthetic_a",
        documents=[make_document("Income Proof A", "shared_income_proof"), make_document("Bonafide Letter A")],
    )
    second = make_scheme(
        "synthetic_b",
        documents=[make_document("Income Proof B", "shared_income_proof"), make_document("Bonafide Letter B")],
    )

    documents = of_type(portfolio([first, second]), "prepare_document")

    assert len(documents) == 3
    merged = [action for action in documents if len(set(scheme_ids_of(action))) == 2]
    assert len(merged) == 1


def test_documents_with_different_reuse_groups_stay_separate_even_with_identical_names():
    plan, _ = two_scheme_plan(
        make_document("Income Certificate", "income_group_a"),
        make_document("Income Certificate", "income_group_b"),
    )

    documents = of_type(plan, "prepare_document")

    assert len(documents) == 2
    assert sorted(tuple(scheme_ids_of(action)) for action in documents) == [("synthetic_a",), ("synthetic_b",)]


def test_documents_without_reuse_group_are_never_merged_by_name():
    plan, _ = two_scheme_plan(make_document("Income Certificate"), make_document("Income Certificate"))

    documents = of_type(plan, "prepare_document")

    assert len(documents) == 2
    assert sorted(tuple(scheme_ids_of(action)) for action in documents) == [("synthetic_a",), ("synthetic_b",)]


def test_empty_reuse_group_means_no_deduplication():
    plan, _ = two_scheme_plan(make_document("Income Certificate", ""), make_document("Income Certificate", ""))
    assert len(of_type(plan, "prepare_document")) == 2


def test_a_reuse_group_on_only_one_document_does_not_merge_it_with_a_same_named_document():
    plan, _ = two_scheme_plan(make_document("Income Certificate", "income_group_a"), make_document("Income Certificate"))
    assert len(of_type(plan, "prepare_document")) == 2


# --------------------------------------------------------------------------- #
# Unverified schemes and unsupported claims
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", ["needs_review", "demo_only"])
def test_unverified_schemes_get_no_confirmed_application_instructions(status):
    scheme = make_scheme(
        "synthetic_unverified",
        verification_status=status,
        documents=[make_document("Synthetic Income Certificate", "shared_group")],
        application_steps=[make_step("only_step", "Unverified step (STEP-UNVERIFIED)")],
    )
    assert evaluate_scheme(QUALIFYING, scheme)["status"] == "not_verified"

    plans = [single(scheme), portfolio([scheme])]

    for plan in plans:
        assert_valid_actions(plan)
        assert of_type(plan, "complete_application_step") == []
        assert "step-unverified" not in dumped(plan)
        assert_no_unsupported_claims(plan)


def test_unverified_scheme_is_kept_out_of_the_verified_schemes_tasks():
    verified = make_scheme(
        "synthetic_verified",
        documents=[make_document("Synthetic Income Certificate", "shared_group")],
        application_steps=[make_step("verified_step", "Verified step (STEP-VERIFIED)")],
    )
    unverified = make_scheme(
        "synthetic_unverified",
        verification_status="demo_only",
        documents=[make_document("Synthetic Income Certificate", "shared_group")],
        application_steps=[make_step("unverified_step", "Unverified step (STEP-UNVERIFIED)")],
    )

    plan = portfolio([verified, unverified])

    assert single_step_position(plan, "STEP-VERIFIED") >= 0
    assert step_positions(plan, "STEP-UNVERIFIED") == []
    for action in of_type(plan, "prepare_document") + of_type(plan, "complete_application_step"):
        assert "synthetic_unverified" not in scheme_ids_of(action)


def test_plans_never_invent_deadlines_or_submission_claims():
    schemes = [
        make_scheme(
            "synthetic_a",
            documents=[make_document("Synthetic Income Certificate", "grp")],
            application_steps=chained_steps(),
        ),
        make_scheme("synthetic_b", documents=[make_document("Synthetic Family Income Proof", "grp")]),
        needs_income_scheme("synthetic_c"),
        make_scheme("synthetic_d", verification_status="demo_only"),
    ]

    plan = portfolio(schemes, make_profile(state="Tamil Nadu"))

    assert plan
    assert_no_unsupported_claims(plan)
