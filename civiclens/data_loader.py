"""Load and validate the CivicLens scheme catalogue (``data/schemes.json``).

This module is the single place that knows the on-disk catalogue format. It
reads UTF-8 JSON, validates every record against the Common Project Contract
and returns the parsed records *unchanged*. It never touches the network,
never rewrites a URL, never marks a record as verified on its own and never
invents fallback records when loading fails.

It also owns the small set of shared constants (profile field names, supported
operators, ...) that the other backend modules import, so the whole backend
agrees on one definition.

Public API
----------
``load_schemes(path=None)``       read + validate the catalogue file.
``validate_catalogue(records)``   validate an already-parsed catalogue.
``validate_scheme_record(...)``   validate one record (no cross-references).
``validate_rule(rule)``           validate one eligibility rule.
``SchemeDataError``               raised for every catalogue problem.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

__all__ = [
    "BOOLEAN_FIELDS",
    "DEFAULT_SCHEMES_PATH",
    "NUMERIC_FIELDS",
    "PROFILE_FIELDS",
    "RELATION_TYPES",
    "STRING_FIELDS",
    "SUPPORTED_OPERATORS",
    "VERIFICATION_STATUSES",
    "SchemeDataError",
    "SchemeDataNotFoundError",
    "find_dependency_cycle",
    "is_real_number",
    "load_schemes",
    "validate_catalogue",
    "validate_rule",
    "validate_rule_operands",
    "validate_scheme_record",
]

# --------------------------------------------------------------------------
# Shared constants
# --------------------------------------------------------------------------

#: The eight canonical profile keys, in canonical order.
PROFILE_FIELDS: tuple[str, ...] = (
    "state",
    "age",
    "education_level",
    "annual_family_income_inr",
    "institution_type",
    "course_type",
    "academic_year",
    "first_generation_student",
)

#: Profile fields whose values are numbers.
NUMERIC_FIELDS: frozenset[str] = frozenset({"age", "annual_family_income_inr"})
#: Profile fields whose values are Booleans.
BOOLEAN_FIELDS: frozenset[str] = frozenset({"first_generation_student"})
#: Profile fields whose values are free-text strings.
STRING_FIELDS: frozenset[str] = frozenset(PROFILE_FIELDS) - NUMERIC_FIELDS - BOOLEAN_FIELDS

#: The only rule operators the engine understands.
SUPPORTED_OPERATORS: tuple[str, ...] = ("equals", "in", "lte", "gte", "between", "contains")

#: Allowed values of ``verification_status`` (scheme level and relationship level).
VERIFICATION_STATUSES: tuple[str, ...] = ("verified", "needs_review", "demo_only")

#: Allowed values of a relationship's ``relation_type``.
RELATION_TYPES: tuple[str, ...] = (
    "compatible",
    "restricted",
    "prerequisite",
    "complementary",
    "unknown",
)

#: Default catalogue location, resolved relative to this file (never the CWD).
DEFAULT_SCHEMES_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "schemes.json"

_REQUIRED_STRING_KEYS: tuple[str, ...] = (
    "scheme_id",
    "name",
    "category",
    "summary",
    "benefit_summary",
    "jurisdiction",
    "application_url",
    "official_source_url",
    "source_title",
    "verified_on",
    "notes",
)
_REQUIRED_SCHEME_KEYS: tuple[str, ...] = _REQUIRED_STRING_KEYS + (
    "eligibility_rules",
    "documents",
    "verification_status",
)
_RULE_KEYS: tuple[str, ...] = (
    "rule_id",
    "field",
    "operator",
    "value",
    "label",
    "source_url",
    "required",
)
_DOCUMENT_KEYS: tuple[str, ...] = ("name", "required", "source_url")
_STEP_KEYS: tuple[str, ...] = (
    "step_id",
    "title",
    "description",
    "depends_on",
    "source_url",
    "required",
)
_RELATIONSHIP_KEYS: tuple[str, ...] = (
    "target_scheme_id",
    "relation_type",
    "source_url",
    "source_title",
    "verified_on",
    "verification_status",
    "notes",
)


class SchemeDataError(ValueError):
    """Raised when the catalogue file or one of its records is invalid."""


class SchemeDataNotFoundError(SchemeDataError, FileNotFoundError):
    """Raised when the catalogue file does not exist."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def is_real_number(value: Any) -> bool:
    """Return True for finite ints/floats. ``bool`` is never a number here."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _check_missing_keys(record: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    missing = [key for key in keys if key not in record]
    if missing:
        raise SchemeDataError(f"{where}: missing required key(s): {', '.join(missing)}")


def _check_url(value: Any, where: str, *, allow_empty: bool = True) -> None:
    """Validate that ``value`` is an empty string or a clean http(s) URL.

    The URL is only inspected; it is never modified.
    """
    if not isinstance(value, str):
        raise SchemeDataError(f"{where} must be a string URL")
    if value == "":
        if allow_empty:
            return
        raise SchemeDataError(f"{where} must not be empty")
    if any(ch.isspace() for ch in value):
        raise SchemeDataError(f"{where} must not contain whitespace: {value!r}")
    try:
        parsed = urlparse(value)
    except ValueError:
        raise SchemeDataError(f"{where} is not a valid URL: {value!r}") from None
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise SchemeDataError(f"{where} must be an http(s) URL, got {value!r}")


def find_dependency_cycle(dependencies: Mapping[str, Sequence[str]]) -> list[str] | None:
    """Return one dependency cycle as a list of ids, or None if there is none.

    ``dependencies`` maps an id to the ids it depends on. References to ids
    that are not keys are ignored. The search is iterative (no recursion
    limit) and deterministic. A self-dependency is reported as ``[x, x]``.
    """
    state: dict[str, int] = {}  # 1 = on the current path, 2 = fully explored
    for start in sorted(dependencies):
        if state.get(start):
            continue
        state[start] = 1
        path = [start]
        stack = [(start, iter(sorted(dependencies[start])))]
        while stack:
            node, children = stack[-1]
            advanced = False
            for child in children:
                if child not in dependencies:
                    continue
                child_state = state.get(child, 0)
                if child_state == 1:
                    return path[path.index(child):] + [child]
                if child_state == 0:
                    state[child] = 1
                    path.append(child)
                    stack.append((child, iter(sorted(dependencies[child]))))
                    advanced = True
                    break
            if not advanced:
                state[node] = 2
                stack.pop()
                path.pop()
    return None


# --------------------------------------------------------------------------
# Rule validation (shared with the engine)
# --------------------------------------------------------------------------


def _valid_scalar_for_field(field: str, value: Any) -> bool:
    if field in NUMERIC_FIELDS:
        return is_real_number(value)
    if field in BOOLEAN_FIELDS:
        return isinstance(value, bool)
    return _non_empty_str(value)


def _type_name_for_field(field: str) -> str:
    if field in NUMERIC_FIELDS:
        return "a finite number"
    if field in BOOLEAN_FIELDS:
        return "a Boolean (true/false)"
    return "a non-empty string"


def validate_rule_operands(field: str, operator: str, value: Any, *, where: str = "rule") -> None:
    """Check that ``value`` is a legal operand for ``operator`` on ``field``.

    * ``equals``  - one scalar of the field's type.
    * ``in``      - a non-empty list of scalars of the field's type.
    * ``lte``/``gte`` - a finite number; numeric fields only.
    * ``between`` - a two-item list ``[low, high]`` of finite numbers with
      ``low <= high``; numeric fields only.
    * ``contains`` - a non-empty string; text fields only.

    Raises :class:`SchemeDataError` for anything else. Booleans are never
    accepted as numbers.
    """
    if operator == "equals":
        if not _valid_scalar_for_field(field, value):
            raise SchemeDataError(
                f"{where}: 'equals' value for '{field}' must be {_type_name_for_field(field)}"
            )
    elif operator == "in":
        if (
            not isinstance(value, list)
            or not value
            or not all(_valid_scalar_for_field(field, item) for item in value)
        ):
            raise SchemeDataError(
                f"{where}: 'in' value for '{field}' must be a non-empty list of "
                f"{_type_name_for_field(field)} items"
            )
    elif operator in ("lte", "gte"):
        if field not in NUMERIC_FIELDS:
            raise SchemeDataError(
                f"{where}: operator '{operator}' only applies to numeric fields "
                f"({', '.join(sorted(NUMERIC_FIELDS))}), not '{field}'"
            )
        if not is_real_number(value):
            raise SchemeDataError(f"{where}: '{operator}' value must be a finite number")
    elif operator == "between":
        if field not in NUMERIC_FIELDS:
            raise SchemeDataError(
                f"{where}: operator 'between' only applies to numeric fields "
                f"({', '.join(sorted(NUMERIC_FIELDS))}), not '{field}'"
            )
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
            or not all(is_real_number(item) for item in value)
        ):
            raise SchemeDataError(
                f"{where}: 'between' value must be a list of exactly two finite numbers"
            )
        if value[0] > value[1]:
            raise SchemeDataError(f"{where}: 'between' lower bound is greater than the upper bound")
    elif operator == "contains":
        if field not in STRING_FIELDS:
            raise SchemeDataError(
                f"{where}: operator 'contains' only applies to text fields, not '{field}'"
            )
        if not _non_empty_str(value):
            raise SchemeDataError(f"{where}: 'contains' value must be a non-empty string")
    else:
        raise SchemeDataError(
            f"{where}: unsupported operator {operator!r} "
            f"(supported: {', '.join(SUPPORTED_OPERATORS)})"
        )


def validate_rule(rule: Any, *, where: str = "rule") -> None:
    """Validate one eligibility rule. Raises :class:`SchemeDataError` if bad.

    Rule text is data only; nothing here (or anywhere in CivicLens) executes it.
    """
    if not isinstance(rule, Mapping):
        raise SchemeDataError(f"{where}: a rule must be a JSON object")
    _check_missing_keys(rule, _RULE_KEYS, where)
    if not _non_empty_str(rule["rule_id"]):
        raise SchemeDataError(f"{where}: 'rule_id' must be a non-empty string")
    field = rule["field"]
    if not isinstance(field, str) or field not in PROFILE_FIELDS:
        raise SchemeDataError(
            f"{where}: unsupported field {field!r} (supported: {', '.join(PROFILE_FIELDS)})"
        )
    operator = rule["operator"]
    if not isinstance(operator, str) or operator not in SUPPORTED_OPERATORS:
        raise SchemeDataError(
            f"{where}: unsupported operator {operator!r} "
            f"(supported: {', '.join(SUPPORTED_OPERATORS)})"
        )
    if not _non_empty_str(rule["label"]):
        raise SchemeDataError(f"{where}: 'label' must be a non-empty string")
    _check_url(rule["source_url"], f"{where}: 'source_url'")
    if not isinstance(rule["required"], bool):
        raise SchemeDataError(f"{where}: 'required' must be a Boolean")
    validate_rule_operands(field, operator, rule["value"], where=where)


# --------------------------------------------------------------------------
# Record validation
# --------------------------------------------------------------------------


def _validate_rules(rules: Any, where: str) -> None:
    if not isinstance(rules, list):
        raise SchemeDataError(f"{where}: 'eligibility_rules' must be a list")
    seen: set[str] = set()
    for index, rule in enumerate(rules):
        rule_where = f"{where} eligibility_rules[{index}]"
        validate_rule(rule, where=rule_where)
        rule_id = rule["rule_id"]
        if rule_id in seen:
            raise SchemeDataError(f"{rule_where}: duplicate rule_id {rule_id!r} within the scheme")
        seen.add(rule_id)


def _validate_documents(documents: Any, where: str) -> None:
    if not isinstance(documents, list):
        raise SchemeDataError(f"{where}: 'documents' must be a list")
    for index, doc in enumerate(documents):
        doc_where = f"{where} documents[{index}]"
        if not isinstance(doc, Mapping):
            raise SchemeDataError(f"{doc_where}: a document must be a JSON object")
        _check_missing_keys(doc, _DOCUMENT_KEYS, doc_where)
        if not _non_empty_str(doc["name"]):
            raise SchemeDataError(f"{doc_where}: 'name' must be a non-empty string")
        if not isinstance(doc["required"], bool):
            raise SchemeDataError(f"{doc_where}: 'required' must be a Boolean")
        _check_url(doc["source_url"], f"{doc_where}: 'source_url'")
        if "reuse_group" in doc and not isinstance(doc["reuse_group"], str):
            raise SchemeDataError(f"{doc_where}: 'reuse_group' must be a string when present")


def _validate_steps(steps: Any, where: str) -> None:
    if not isinstance(steps, list):
        raise SchemeDataError(f"{where}: 'application_steps' must be a list")
    dependencies: dict[str, list[str]] = {}
    for index, step in enumerate(steps):
        step_where = f"{where} application_steps[{index}]"
        if not isinstance(step, Mapping):
            raise SchemeDataError(f"{step_where}: a step must be a JSON object")
        _check_missing_keys(step, _STEP_KEYS, step_where)
        step_id = step["step_id"]
        if not _non_empty_str(step_id):
            raise SchemeDataError(f"{step_where}: 'step_id' must be a non-empty string")
        if step_id in dependencies:
            raise SchemeDataError(f"{step_where}: duplicate step_id {step_id!r} within the scheme")
        if not _non_empty_str(step["title"]):
            raise SchemeDataError(f"{step_where}: 'title' must be a non-empty string")
        if not isinstance(step["description"], str):
            raise SchemeDataError(f"{step_where}: 'description' must be a string")
        depends_on = step["depends_on"]
        if not isinstance(depends_on, list) or not all(_non_empty_str(d) for d in depends_on):
            raise SchemeDataError(f"{step_where}: 'depends_on' must be a list of step_id strings")
        if len(set(depends_on)) != len(depends_on):
            raise SchemeDataError(f"{step_where}: 'depends_on' contains duplicates")
        # Steps are claims about the application procedure, so each needs a source.
        _check_url(step["source_url"], f"{step_where}: 'source_url'", allow_empty=False)
        if not isinstance(step["required"], bool):
            raise SchemeDataError(f"{step_where}: 'required' must be a Boolean")
        dependencies[step_id] = list(depends_on)
    for step_id, depends_on in dependencies.items():
        for dep in depends_on:
            if dep not in dependencies:
                raise SchemeDataError(
                    f"{where}: step {step_id!r} depends on unknown step_id {dep!r}"
                )
    cycle = find_dependency_cycle(dependencies)
    if cycle is not None:
        raise SchemeDataError(
            f"{where}: application_steps contain a dependency cycle: {' -> '.join(cycle)}"
        )


def _validate_relationships(relationships: Any, where: str, scheme_id: str) -> None:
    if not isinstance(relationships, list):
        raise SchemeDataError(f"{where}: 'relationships' must be a list")
    seen_targets: set[str] = set()
    for index, rel in enumerate(relationships):
        rel_where = f"{where} relationships[{index}]"
        if not isinstance(rel, Mapping):
            raise SchemeDataError(f"{rel_where}: a relationship must be a JSON object")
        _check_missing_keys(rel, _RELATIONSHIP_KEYS, rel_where)
        target = rel["target_scheme_id"]
        if not _non_empty_str(target):
            raise SchemeDataError(f"{rel_where}: 'target_scheme_id' must be a non-empty string")
        if target == scheme_id:
            raise SchemeDataError(f"{rel_where}: a scheme cannot be related to itself")
        if target in seen_targets:
            raise SchemeDataError(f"{rel_where}: duplicate relationship to {target!r}")
        seen_targets.add(target)
        if rel["relation_type"] not in RELATION_TYPES:
            raise SchemeDataError(
                f"{rel_where}: invalid relation_type {rel['relation_type']!r} "
                f"(allowed: {', '.join(RELATION_TYPES)})"
            )
        _check_url(rel["source_url"], f"{rel_where}: 'source_url'")
        for key in ("source_title", "verified_on", "notes"):
            if not isinstance(rel[key], str):
                raise SchemeDataError(f"{rel_where}: '{key}' must be a string")
        if rel["verification_status"] not in VERIFICATION_STATUSES:
            raise SchemeDataError(
                f"{rel_where}: invalid verification_status {rel['verification_status']!r} "
                f"(allowed: {', '.join(VERIFICATION_STATUSES)})"
            )


def validate_scheme_record(record: Any, *, where: str = "scheme") -> None:
    """Validate one scheme record (everything except cross-record references).

    Raises :class:`SchemeDataError` with a descriptive message on any problem.
    Extra, unknown keys are tolerated and preserved.
    """
    if not isinstance(record, Mapping):
        raise SchemeDataError(f"{where}: a scheme record must be a JSON object")
    _check_missing_keys(record, _REQUIRED_SCHEME_KEYS, where)
    for key in _REQUIRED_STRING_KEYS:
        if not isinstance(record[key], str):
            raise SchemeDataError(f"{where}: '{key}' must be a string")
    scheme_id = record["scheme_id"]
    if not _non_empty_str(scheme_id):
        raise SchemeDataError(f"{where}: 'scheme_id' must be a non-empty string")
    if scheme_id != scheme_id.strip():
        raise SchemeDataError(f"{where}: 'scheme_id' must not have leading/trailing whitespace")
    if not _non_empty_str(record["name"]):
        raise SchemeDataError(f"{where}: 'name' must be a non-empty string")
    _check_url(record["application_url"], f"{where}: 'application_url'")
    _check_url(record["official_source_url"], f"{where}: 'official_source_url'")
    if record["verification_status"] not in VERIFICATION_STATUSES:
        raise SchemeDataError(
            f"{where}: invalid verification_status {record['verification_status']!r} "
            f"(allowed: {', '.join(VERIFICATION_STATUSES)})"
        )
    _validate_rules(record["eligibility_rules"], where)
    _validate_documents(record["documents"], where)
    if "application_steps" in record:
        _validate_steps(record["application_steps"], where)
    if "relationships" in record:
        _validate_relationships(record["relationships"], where, scheme_id)


def validate_catalogue(records: Any) -> list[dict]:
    """Validate a parsed catalogue and return it unchanged.

    Checks that the root is a list, that every record is valid, that
    ``scheme_id`` values are unique and that relationship targets exist.
    An empty list is a valid (empty) catalogue.
    """
    if not isinstance(records, list):
        raise SchemeDataError("The catalogue root must be a JSON array of scheme records")
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        hint = record.get("scheme_id") if isinstance(record, Mapping) else None
        where = f"scheme #{index + 1}" + (f" ({hint!r})" if isinstance(hint, str) and hint else "")
        validate_scheme_record(record, where=where)
        scheme_id = record["scheme_id"]
        if scheme_id in seen_ids:
            raise SchemeDataError(f"{where}: duplicate scheme_id {scheme_id!r}")
        seen_ids.add(scheme_id)
    for record in records:
        for rel in record.get("relationships", []):
            if rel["target_scheme_id"] not in seen_ids:
                raise SchemeDataError(
                    f"scheme {record['scheme_id']!r}: relationship target "
                    f"{rel['target_scheme_id']!r} is not in the catalogue"
                )
    return records


# --------------------------------------------------------------------------
# Public loader
# --------------------------------------------------------------------------


def _reject_json_constant(name: str) -> Any:
    raise ValueError(f"non-standard JSON constant {name!r} is not allowed")


def load_schemes(path: str | Path | None = None) -> list[dict]:
    """Load, validate and return the scheme catalogue.

    Args:
        path: Optional path to a catalogue file (used by tests). When omitted,
            ``data/schemes.json`` next to the ``civiclens`` package is used,
            independent of the current working directory.

    Returns:
        The list of scheme records exactly as stored in the file (an empty
        list is valid).

    Raises:
        SchemeDataNotFoundError: the file does not exist.
        SchemeDataError: the file cannot be read/decoded, is not valid JSON,
            is not a JSON array, or any record violates the contract.
    """
    file_path = Path(path) if path is not None else DEFAULT_SCHEMES_PATH
    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise SchemeDataNotFoundError(f"Scheme catalogue not found at {file_path}") from None
    except UnicodeDecodeError:
        raise SchemeDataError(f"{file_path.name} is not valid UTF-8 text") from None
    except OSError as exc:
        raise SchemeDataError(
            f"Could not read {file_path.name} ({exc.__class__.__name__})"
        ) from None
    try:
        data = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise SchemeDataError(
            f"{file_path.name} is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from None
    except ValueError as exc:
        raise SchemeDataError(f"{file_path.name} is not valid JSON: {exc}") from None
    return validate_catalogue(data)
