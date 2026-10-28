"""Deterministic subject-memory transitions for multi-turn research dialogue.

The LLM decides *which* action to take. This module decides deterministically
how that action changes the persisted research subject state.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


VALID_ACTIONS = {"INIT", "INHERIT", "APPEND", "REPLACE"}


def transition_subject_state(
    previous: Optional[Mapping[str, Any]],
    action: Optional[str],
    query: str,
    *,
    subject_entity: str = "",
    sub_subject: str = "",
    intent: str = "",
    evidence: Optional[List[Dict[str, Any]]] = None,
    constraints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Apply one explicit subject-memory transition.

    ``previous`` is the formal S_t state. The returned mapping is S_(t+1).
    Evidence and constraints are current-turn values; the evolution path is
    preserved because it describes the session's topic history.
    """
    old = dict(previous or {})
    old_subject = str(old.get("subject") or old.get("current_topic") or "").strip()
    old_sub = str(old.get("sub_subject") or old.get("sub_topic") or "").strip()
    entity = (subject_entity or "").strip()
    branch = (sub_subject or "").strip()
    normalized_action = (action or "INHERIT").strip().upper()
    normalized_action = {
        "INIT_LOCK": "INIT",
        "RESET": "REPLACE",
    }.get(normalized_action, normalized_action)
    if normalized_action not in VALID_ACTIONS:
        normalized_action = "INHERIT"

    path = list(old.get("evolution_path") or [])

    if normalized_action in {"INIT", "REPLACE"} and entity:
        subject = entity
        current_sub = branch
        status = "LOCKED"
        path = [entity]
    elif normalized_action == "APPEND" and entity:
        parts = [p.strip() for p in old_subject.split(" & ") if p.strip()]
        if entity not in parts:
            parts.append(entity)
        subject = " & ".join(parts) if parts else entity
        current_sub = branch or old_sub
        status = "EVOLVING"
        if not path or path[-1] != entity:
            path.append(entity)
    else:
        normalized_action = "INHERIT"
        subject = old_subject or entity
        current_sub = branch or old_sub
        status = old.get("status") or ("LOCKED" if subject else "INIT")

    if normalized_action == "INHERIT" and not subject and query.strip():
        # No invented topic: keep the subject empty until the router supplies
        # an entity. The query remains available to the caller unchanged.
        status = "INIT"

    return {
        "subject": subject,
        "sub_subject": current_sub,
        "intent": intent or old.get("intent", ""),
        "evidence": list(evidence or []),
        "constraints": list(constraints or []),
        "status": status,
        "evolution_path": path,
        "action": normalized_action,
    }
