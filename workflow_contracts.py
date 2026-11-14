"""Deterministic task handoff and delivery checks; no model calls or text rewriting.

These checks establish identity and provenance, never semantic truth or coverage.
"""

from copy import deepcopy
import json
import re
from urllib.parse import unquote, urlsplit, urlunsplit


TASK_FIELDS = (
    "domain", "intent", "task_segments", "negative_constraints", "time_constraint",
    "time_constraint_note", "framework_reference", "requires_long_form", "long_form_type",
)


def task_decision(state):
    """Each worker receives its own copy of the same accepted interpretation."""
    snapshot = state.get("task_snapshot") or {}
    return deepcopy(snapshot.get("decision") or state.get("intent_decision") or {})


def build_task_snapshot(original, visual_context, proposed, accepted, subject, alignment=None, aligned=None):
    """Record a model interpretation without promoting it to the user's literal request."""
    def differences(before, after):
        return {key: {"before": deepcopy(before.get(key)), "after": deepcopy(after.get(key))}
                for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
    aligned = accepted if aligned is None else aligned
    return {
        "version": 1, "original_user_query": original,
        "visual_context": visual_context or "", "active_subject": subject,
        "decision": deepcopy(accepted),
        "alignment_status": (alignment or {}).get("status", "not_reported"),
        "interpretation_log": [
            {"stage": "intent", "interpretation": deepcopy(proposed)},
            {"stage": "pre_search_alignment", "changes": differences(proposed, aligned),
             "grounds": deepcopy((alignment or {}).get("issues") or []),
             "grounds_type": "model_assessment", "semantic_correctness_verified": False},
            {"stage": "workflow_defaults", "changes": differences(aligned, accepted),
             "grounds_type": "system_policy"},
        ],
    }


def task_payload(state):
    snapshot = state.get("task_snapshot") or {}
    decision = task_decision(state)
    return {
        "original_user_query": snapshot.get("original_user_query", state.get("user_query", "")),
        "visual_context": snapshot.get("visual_context", state.get("extracted_visual_context") or ""),
        "active_subject": snapshot.get("active_subject", state.get("active_subject")),
        "active_doc": snapshot.get("active_doc", state.get("active_doc")),
        "interpreted_task": {key: decision.get(key) for key in TASK_FIELDS if decision.get(key) is not None},
    }


def task_prompt(state):
    return "本轮共享任务（原话为准，解释为模型判断；子任务服务于此，不另立目标）：\n" + json.dumps(
        task_payload(state), ensure_ascii=False, separators=(",", ":"))


def citation_integrity(answer, evidence):
    """Check numeric citation syntax outside code/math; do not infer factual support.

Bracketed array values can resemble citations, so all results are diagnostics,
not grounds for automatic rejection, supplementary search or answer rewriting.
"""
    text = re.sub(r"```[\s\S]*?```|`[^`\n]*`|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$\$[\s\S]*?\$\$|\$[^$\n]*\$", "", answer or "")
    refs = set()
    for match in re.finditer(r"(?<![!\\])\[(\d+(?:\s*[,，、–—-]\s*\d+)*)\](?!\s*[(:])", text):
        for group in re.split(r"\s*[,，、]\s*", match[1]):
            bounds = re.split(r"\s*[–—-]\s*", group)
            if len(bounds) == 2:
                start, end = map(int, bounds)
                # Reject huge numeric spans without allocating large ranges.
                if end < start or end - start > max(len(evidence), 1000):
                    refs.update((start, end))
                    continue
                refs.update(range(start, end + 1))
            else:
                refs.add(int(group))
    known = set(range(1, len(evidence) + 1))
    return {
        "checked": "numeric_citation_ids_only", "requires_semantic_review": True,
        "unknown_ids": sorted(refs - known),
        "excluded_ids": sorted(idx for idx in refs & known if evidence[idx - 1].get("excluded_as_irrelevant")),
        "unread_ids": sorted(idx for idx in refs & known if not any(evidence[idx - 1].get(k) for k in ("text", "summary"))),
    }


def append_answer(original, addition):
    """The model supplies only the appendix; the program owns the existing body."""
    addition = (addition or "").strip()
    if original and original in addition:
        addition = addition.replace(original, "").strip()
    return original + ("\n\n---\n\n### 补充内容\n\n" + addition if addition else ""), bool(addition)


def merge_text(existing, incoming):
    """Preserve complementary inspected text without repeating contained passages."""
    old, new = str(existing or "").strip(), str(incoming or "").strip()
    if not new or new in old:
        return old, ""
    if old and old in new:
        return new, new.replace(old, "", 1).strip()
    return (old + "\n" + new if old else new), new


def evidence_keys(item):
    """Source aliases, with page and repository-version boundaries preserved."""
    keys = set()
    doi = str(item.get("doi") or "").strip().casefold()
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", doi)
    if doi:
        keys.add("doi:" + unquote(doi))
    arxiv = str(item.get("arxiv_id") or "").strip()
    record_id = str(item.get("paper_id") or item.get("id") or "").strip()
    if record_id:
        if re.match(r"^(?:https?://(?:dx\.)?doi\.org/)?10\.", record_id, re.I):
            keys.add("doi:" + re.sub(r"^https?://(?:dx\.)?doi\.org/", "", record_id, flags=re.I).casefold())
        elif re.match(r"^\d{4}\.\d{4,5}(?:v\d+)?$", record_id):
            arxiv = arxiv or record_id
        else:
            keys.add("id:" + record_id.casefold())
    urls = [item.get(k) for k in ("url", "pdf_url", "github_url") if item.get(k)]
    for value in urls:
        url = urlsplit(str(value).strip())
        if url.hostname and url.hostname.casefold() in {"doi.org", "dx.doi.org"}:
            keys.add("doi:" + unquote(url.path.lstrip("/")).casefold())
        elif url.hostname and url.hostname.casefold() in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            arxiv = re.sub(r"^/(?:abs|pdf)/", "", url.path).removesuffix(".pdf")
        elif url.hostname:
            keys.add("url:" + urlunsplit((url.scheme.casefold(), url.netloc.casefold(), url.path.rstrip("/"), url.query, "")))
    if arxiv:
        arxiv = re.sub(r"^(?:https?://[^/]+/(?:abs|pdf)/|arxiv:)", "", arxiv, flags=re.I)
        keys.add("arxiv:" + re.sub(r"v\d+$", "", arxiv.removesuffix(".pdf")).casefold())
    if not keys:
        title = str(item.get("doc_name") or item.get("title") or "").strip().casefold()
        if title:
            keys.add("title:" + re.sub(r"\s+", " ", title))
    local = item.get("is_local_indexed") or item.get("doc_type") in {"pdf", "pptx", "docx", "local", "local_pdf", "document", "meeting_transcript"}
    location = (item.get("page_or_slide") if item.get("page_or_slide") is not None else item.get("location_label") or "") if local else ""
    boundary = (str(location).casefold(), str(item.get("version_ref") or ""))
    return {(key, *boundary) for key in keys}


def merge_evidence(existing, incoming):
    """Keep stable existing indices and return only newly inspected content as a delta."""
    evidence = deepcopy(existing or [])
    index = {key: idx for idx, item in enumerate(evidence) for key in evidence_keys(item)}
    additions = []
    for raw in incoming or []:
        if not isinstance(raw, dict):
            continue
        item = deepcopy(raw)
        keys = evidence_keys(item)
        idx = next((index[key] for key in keys if key in index), None)
        if idx is None:
            idx = len(evidence)
            evidence.append(item)
            additions.append({**item, "evidence_id": idx + 1})
        else:
            prior, delta = evidence[idx], {}
            for field in ("text", "summary"):
                merged, new = merge_text(prior.get(field), item.get(field))
                if field in prior or new:
                    prior[field] = merged
                if new:
                    delta[field] = new
            for field, value in item.items():
                if field not in {"text", "summary", "excluded_as_irrelevant"} and not prior.get(field) and value:
                    prior[field] = value
            if delta:
                # A previous exclusion cannot silently suppress newly inspected material.
                prior.pop("excluded_as_irrelevant", None)
                if item.get("evidence_level") and prior.get("evidence_level") in {
                        None, "unknown", "metadata", "bibliographic_metadata", "metadata_or_abstract",
                        "index_metadata_or_abstract", "bibliographic_metadata_or_abstract"}:
                    prior["evidence_level"] = item["evidence_level"]
                provenance = list(prior.get("provenance_records") or [])
                for source in (prior.get("provenance"), item.get("provenance")):
                    if isinstance(source, dict) and source and source not in provenance:
                        provenance.append(deepcopy(source))
                if provenance:
                    prior["provenance_records"] = provenance
                additions.append({**prior, "text": delta.get("text", ""), "summary": delta.get("summary", ""), "evidence_id": idx + 1})
        index.update({key: idx for key in keys | evidence_keys(evidence[idx])})
    return evidence, additions


def completion_checks(value):
    """Expose completion metadata without withholding the generated text."""
    reason = getattr(value, "finish_reason", None)
    return {"model_completion": {"finish_reason": reason, "complete": reason == "stop"}} if reason else {}
