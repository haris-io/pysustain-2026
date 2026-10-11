"""Deterministic, evidence-linked next-step plans."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def _action(action_type, title, description, priority, source_url="", scheme_ids=(), action_id=None, depends_on=()):
    result = {
        "title": title,
        "description": description,
        "priority": priority,
        "action_type": action_type,
        "source_url": source_url if isinstance(source_url, str) else "",
        "scheme_ids": list(scheme_ids),
        "depends_on": list(depends_on),
    }
    if action_id is not None:
        result["action_id"] = action_id
    return result


def build_action_plan(evaluation: Mapping[str, Any], profile: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Plan safe profile, verification, and document actions for one result."""
    if not isinstance(evaluation, Mapping):
        return []
    scheme_id = evaluation.get("scheme_id")
    if not isinstance(scheme_id, str) or not scheme_id:
        return []
    status = evaluation.get("status")
    verified = evaluation.get("verification_status") == "verified"
    actions = []
    missing = evaluation.get("missing_information", [])
    if isinstance(missing, list):
        for item in missing:
            field = item.get("field") if isinstance(item, Mapping) else item
            label = item.get("label") if isinstance(item, Mapping) else None
            if not isinstance(field, str):
                continue
            readable = label if isinstance(label, str) and label.strip() else field.replace("_", " ")
            actions.append(_action(
                "answer_missing_information",
                f"Provide information about {readable}",
                f"Share this detail if you want a more complete screening for {evaluation.get('scheme_name') or scheme_id}.",
                1,
                evaluation.get("official_source_url", ""),
                [scheme_id],
            ))
    actions.append(_action(
        "verify_conditions",
        f"Review the recorded conditions for {evaluation.get('scheme_name') or scheme_id}",
        "Check the conditions and evidence on the official source; this screening is preliminary.",
        2,
        evaluation.get("official_source_url", ""),
        [scheme_id],
    ))
    if not verified:
        actions.append(_action(
            "open_official_source",
            f"Check the official source for {evaluation.get('scheme_name') or scheme_id}",
            "Confirm current details directly with the issuing authority; this record is not verified.",
            3,
            evaluation.get("official_source_url", ""),
            [scheme_id],
        ))
        return _assign_ids(actions, "single")

    if status == "potential_match":
        documents = evaluation.get("document_checklist", [])
        if isinstance(documents, list):
            for document in documents:
                name = document.get("name") if isinstance(document, Mapping) else document
                if isinstance(name, str) and name.strip():
                    actions.append(_action(
                        "prepare_document",
                        f"Check {name}",
                        f"Confirm whether {name} is required and current for this scheme.",
                        3,
                        document.get("source_url", "") if isinstance(document, Mapping) else evaluation.get("official_source_url", ""),
                        [scheme_id],
                    ))
        if isinstance(evaluation.get("application_url"), str) and evaluation["application_url"]:
            actions.append(_action(
                "open_official_source",
                f"Review the official application information for {evaluation.get('scheme_name') or scheme_id}",
                "Check current instructions with the issuing authority before taking action.",
                4,
                evaluation["application_url"],
                [scheme_id],
            ))
    return _assign_ids(actions, "single")


def _assign_ids(actions, prefix):
    used = set()
    for index, action in enumerate(actions, start=1):
        action_id = action.get("action_id")
        if not isinstance(action_id, str) or not action_id or action_id in used:
            action_id = f"{prefix}-{index}"
            while action_id in used:
                action_id = f"{action_id}-next"
            action["action_id"] = action_id
        used.add(action_id)
    return actions


def _ordered_steps(steps):
    if not isinstance(steps, list):
        return [], False
    by_id = {
        step.get("step_id"): step
        for step in steps
        if isinstance(step, Mapping) and isinstance(step.get("step_id"), str)
    }
    state = {}
    ordered = []
    cycle = False

    def visit(step_id, chain):
        nonlocal cycle
        if state.get(step_id) == 2:
            return
        if state.get(step_id) == 1:
            cycle = True
            return
        state[step_id] = 1
        step = by_id[step_id]
        deps = step.get("depends_on", [])
        if isinstance(deps, list):
            for dep in deps:
                if dep in by_id:
                    visit(dep, chain + [step_id])
                elif dep:
                    cycle = True
        state[step_id] = 2
        ordered.append(step)

    for step_id in by_id:
        visit(step_id, [])
    return ordered, cycle


def build_portfolio_action_plan(
    evaluations: Sequence[Mapping[str, Any]],
    schemes: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build one stable portfolio plan using only catalogue-backed actions."""
    if not evaluations:
        return []
    scheme_map = {
        scheme.get("scheme_id"): scheme
        for scheme in schemes
        if isinstance(scheme, Mapping) and isinstance(scheme.get("scheme_id"), str)
    }
    eval_map = {
        item.get("scheme_id"): item
        for item in evaluations
        if isinstance(item, Mapping) and isinstance(item.get("scheme_id"), str)
    }
    actions = []
    missing_actions = []
    verification_actions = []
    doc_groups = defaultdict(list)
    other_actions = []

    for scheme_id in sorted(eval_map):
        evaluation = eval_map[scheme_id]
        scheme = scheme_map.get(scheme_id)
        single_actions = build_action_plan(evaluation, profile)
        for action in single_actions:
            if action["action_type"] == "answer_missing_information":
                missing_actions.append(action)
            elif action["action_type"] == "verify_conditions":
                verification_actions.append(action)
            elif action["action_type"] == "prepare_document":
                document = next(
                    (doc for doc in (scheme or {}).get("documents", [])
                     if isinstance(doc, Mapping) and doc.get("name") in action["title"]),
                    None,
                )
                group = document.get("reuse_group") if isinstance(document, Mapping) else None
                if isinstance(group, str) and group.strip():
                    doc_groups[group].append(action)
                else:
                    other_actions.append(action)
            elif action["action_type"] == "open_official_source":
                other_actions.append(action)

        if (
            scheme is not None
            and evaluation.get("verification_status") == "verified"
            and evaluation.get("status") == "potential_match"
        ):
            steps = scheme.get("application_steps", [])
            ordered, has_cycle = _ordered_steps(steps)
            step_action_ids = {}
            if has_cycle:
                other_actions.append(_action(
                    "verify_conditions",
                    f"Review the application-step dependency cycle for {scheme.get('name', scheme_id)}",
                    "The catalogue lists cyclic or unresolved step dependencies; confirm the correct order with the official source.",
                    2,
                    scheme.get("official_source_url", ""),
                    [scheme_id],
                ))
                ordered = []
            for step in ordered:
                if step.get("required", True) is False:
                    continue
                step_id = step["step_id"]
                dep_ids = [step_action_ids[dep] for dep in step.get("depends_on", []) if dep in step_action_ids]
                action = _action(
                    "complete_application_step",
                    step.get("title") or "Review application step",
                    step.get("description") or "Check this step against the official instructions.",
                    4,
                    step.get("source_url", ""),
                    [scheme_id],
                    action_id=f"step-{scheme_id}-{step_id}",
                    depends_on=dep_ids,
                )
                step_action_ids[step_id] = action["action_id"]
                other_actions.append(action)

    actions.extend(missing_actions)
    actions.extend(verification_actions)
    for group in sorted(doc_groups):
        grouped = doc_groups[group]
        if len(grouped) == 1:
            other_actions.append(grouped[0])
            continue
        first = grouped[0]
        scheme_ids = sorted({sid for action in grouped for sid in action["scheme_ids"]})
        other_actions.append(_action(
            "prepare_document",
            first["title"],
            "This preparation item is shared by schemes " + ", ".join(scheme_ids) + ".",
            first["priority"],
            first["source_url"],
            scheme_ids,
        ))
    actions.extend(other_actions)
    return _assign_ids(actions, "portfolio")
