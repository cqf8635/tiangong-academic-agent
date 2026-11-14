import json

from session_manager import ChatMessage, SessionManager


def test_shared_resources_are_session_scoped_and_merge_source_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("session_manager.SESSIONS_DIR", tmp_path)
    first = SessionManager.create_session(title="first")
    second = SessionManager.create_session(title="second")
    resource = {
        "doc_name": "A research paper",
        "doi": "10.1234/example",
        "doc_type": "crossref",
        "text": "bibliographic record",
        "evidence_level": "bibliographic_metadata_or_abstract",
        "provenance": {"source": "CrossRef", "url": "https://doi.org/10.1234/example"},
    }
    SessionManager.update_shared_resources(first.session_id, [resource])
    SessionManager.update_shared_resources(first.session_id, [{
        **resource,
        "doc_type": "local_pdf",
        "text": "local passage inspected from the uploaded PDF",
        "evidence_level": "local_snippet",
        "provenance": {"source": "local_pdf", "location": "p. 4"},
    }])

    a = SessionManager.get_session(first.session_id)
    b = SessionManager.get_session(second.session_id)
    assert len(a.shared_resources) == 1
    assert len(a.shared_resources[0]["provenance_records"]) == 2
    assert a.shared_resources[0]["evidence_level"] == "local_snippet"
    assert b.shared_resources == []
    assert len(a.shared_pool) == 1
    assert a.shared_pool[0]["kind"] == "evidence"
    assert "resource" not in a.shared_pool[0]  # shared index points to canonical evidence storage
    assert SessionManager.get_shared_pool(second.session_id) == []


def test_shared_pool_publishes_turn_state_and_enforces_storage_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("session_manager.SESSIONS_DIR", tmp_path)
    monkeypatch.setenv("SESSION_SHARED_POOL_MAX_ITEMS", "2")
    session = SessionManager.create_session(title="bounded")
    for idx in range(1, 4):
        message = ChatMessage(sender="agent", content=f"answer {idx}")
        SessionManager.add_message(session.session_id, message)
        SessionManager.publish_turn_to_shared_pool(
            session.session_id, idx, f"task {idx}", message,
            turn_state={"topic_state": {"current_topic": f"topic {idx}"}},
        )
    pool = SessionManager.get_shared_pool(session.session_id)
    assert len(pool) == 2
    assert {item["turn_idx"] for item in pool} == {2, 3}
    latest = next(item for item in pool if item["turn_idx"] == 3)
    assert latest["state"]["topic_state"]["current_topic"] == "topic 3"
    # Turn text is resolved from this session's message archive, not duplicated in the pool record.
    assert latest["content"] == "answer 3"


def test_clearing_a_session_also_clears_its_shared_memory(tmp_path, monkeypatch):
    monkeypatch.setattr("session_manager.SESSIONS_DIR", tmp_path)
    session = SessionManager.create_session(title="clearable")
    SessionManager.update_shared_resources(session.session_id, [{
        "doc_name": "Local paper", "text": "evidence", "doc_type": "local_pdf",
    }])
    cleared = SessionManager.clear_session_messages(session.session_id)
    assert cleared.shared_pool == []
    assert cleared.shared_resources == []
    assert cleared.topic_catalog == []
    assert SessionManager.get_shared_pool(session.session_id) == []


def test_pool_metadata_is_lazy_and_selected_turn_hydrates_linked_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("session_manager.SESSIONS_DIR", tmp_path)
    session = SessionManager.create_session(title="lazy")
    evidence = {"doc_name": "Paper", "doi": "10.9876/test", "text": "supporting evidence", "doc_type": "web"}
    SessionManager.update_shared_resources(session.session_id, [evidence])
    message = ChatMessage(sender="agent", content="prior conclusion", retrieved_evidence=[evidence])
    SessionManager.add_message(session.session_id, message)
    SessionManager.publish_turn_to_shared_pool(session.session_id, 1, "analyze paper", message)

    catalog = SessionManager.get_shared_pool(session.session_id, include_content=False)
    turn = next(item for item in catalog if item["kind"] == "turn_record")
    assert "content" not in turn
    assert "state" not in turn

    selected = SessionManager.get_shared_pool_items(session.session_id, [turn["item_id"]])
    selected_turn = next(item for item in selected if item["kind"] == "turn_record")
    selected_evidence = next(item for item in selected if item["kind"] == "evidence")
    assert selected_turn["content"] == "prior conclusion"
    assert selected_evidence["resource"]["text"] == "supporting evidence"


def test_candidate_records_remain_available_with_existing_shared_pool(tmp_path, monkeypatch):
    monkeypatch.setattr("session_manager.SESSIONS_DIR", tmp_path)
    session = SessionManager.create_session(title="candidates")
    SessionManager.update_shared_resources(session.session_id, [{"doc_name": "Existing", "text": "evidence"}])
    SessionManager.update_recent_candidates(session.session_id, [{
        "title": "Candidate", "paper_id": "paper-1", "summary": "candidate summary",
    }])
    pool = SessionManager.get_shared_pool(session.session_id, include_content=False)
    candidate = next(item for item in pool if item["kind"] == "candidate")
    selected = SessionManager.get_shared_pool_items(session.session_id, [candidate["item_id"]])
    assert selected[0]["candidate"]["title"] == "Candidate"


def test_new_explicit_version_cleans_old_cache_but_preserves_archived_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr("session_manager.SESSIONS_DIR", tmp_path)
    session = SessionManager.create_session(title="versions")
    old = {
        "doc_name": "Paper", "doi": "10.5555/versioned", "version": "v1",
        "text": "old version body", "doc_type": "web",
    }
    SessionManager.update_shared_resources(session.session_id, [old])
    message = ChatMessage(sender="agent", content="old conclusion", retrieved_evidence=[old])
    SessionManager.add_message(session.session_id, message)
    SessionManager.publish_turn_to_shared_pool(session.session_id, 1, "read v1", message)

    new = {**old, "version": "v2", "text": "new version body"}
    SessionManager.update_shared_resources(session.session_id, [new])
    current = SessionManager.get_session(session.session_id)
    old_resource = next(item for item in current.shared_resources if item.get("version") == "v1")
    assert "text" not in old_resource
    assert old_resource["superseded_by"] == SessionManager._resource_identity(new)

    old_item = next(item for item in current.shared_pool if item.get("resource_id") == old_resource["resource_id"])
    hydrated = SessionManager.get_shared_pool_items(session.session_id, [old_item["item_id"]])
    assert next(item for item in hydrated if item["kind"] == "evidence")["resource"]["text"] == "old version body"


def test_legacy_topic_catalog_migrates_into_shared_pool_without_duplicate_copy(tmp_path, monkeypatch):
    monkeypatch.setattr("session_manager.SESSIONS_DIR", tmp_path)
    session = SessionManager.create_session(title="legacy")
    session.topic_catalog = [{
        "topic_id": "T1", "topic_name": "topic", "action_type": "analyze",
        "task": "analyze topic", "entities": ["topic"],
        "key_findings": "legacy finding", "turn_idx": 1,
    }]
    SessionManager.save_session(session)
    pool = SessionManager.get_shared_pool(session.session_id, include_content=False)
    loaded = SessionManager.get_session(session.session_id)
    assert loaded.topic_catalog == []
    assert any(item.get("content") == "legacy finding" for item in loaded.shared_pool)
    assert all("content" not in item for item in pool)
    assert "legacy finding" in SessionManager.get_scoped_thematic_context(session.session_id, "topic")


def test_archived_messages_reference_duplicate_evidence_and_hydrate_it_on_read(tmp_path, monkeypatch):
    monkeypatch.setattr("session_manager.SESSIONS_DIR", tmp_path)
    session = SessionManager.create_session(title="dedup archive")
    evidence = {
        "doc_name": "Paper", "doi": "10.7777/archive", "text": "exact passage",
        "location_label": "p. 2", "doc_type": "local_pdf",
    }
    SessionManager.update_shared_resources(session.session_id, [evidence])
    message = ChatMessage(sender="agent", content="answer", retrieved_evidence=[evidence])
    SessionManager.add_message(session.session_id, message)

    raw = json.loads(SessionManager._file_path(session.session_id).read_text(encoding="utf-8"))
    assert raw["messages"][0]["retrieved_evidence"] == [{"resource_id": SessionManager._resource_identity(evidence)}]
    restored = SessionManager.get_session(session.session_id)
    assert restored.messages[0].retrieved_evidence[0]["text"] == "exact passage"


def test_archiving_does_not_replace_distinct_evidence_with_same_resource_id(tmp_path, monkeypatch):
    monkeypatch.setattr("session_manager.SESSIONS_DIR", tmp_path)
    session = SessionManager.create_session(title="evidence variants")
    cached = {"doc_name": "Paper", "doi": "10.8888/archive", "text": "fuller passage", "doc_type": "local_pdf"}
    SessionManager.update_shared_resources(session.session_id, [cached])
    snippet = {**cached, "text": "different excerpt", "location_label": "p. 9"}
    message = ChatMessage(sender="agent", content="answer", retrieved_evidence=[snippet])
    SessionManager.add_message(session.session_id, message)

    raw = json.loads(SessionManager._file_path(session.session_id).read_text(encoding="utf-8"))
    assert raw["messages"][0]["retrieved_evidence"][0]["text"] == "different excerpt"
