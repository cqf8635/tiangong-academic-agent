import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import online_paper_search_agent as mod
from session_manager import SessionManager, ConversationSession


def test_download_and_ingest_creates_sidecar_and_computes_sha256(monkeypatch):
    temp_dir = Path(tempfile.mkdtemp())
    monkeypatch.setattr(mod, "UPLOADS_DIR", temp_dir)

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self, _size=None):
            if not hasattr(self, "_done"):
                self._done = True
                return b"%PDF-1.5 test content for academic paper"
            return b""

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *_a, **_k: FakeResponse())

    agent = mod.OnlinePaperSearchAgent.__new__(mod.OnlinePaperSearchAgent)
    mock_rag = MagicMock()
    mock_rag.ingest_file.return_value = 5

    candidate = {
        "paper_id": "2401.99999",
        "title": "Quantum Mechanics and Relativity Unification",
        "pdf_url": "https://arxiv.org/pdf/2401.99999.pdf",
        "authors": ["Alice", "Bob"],
        "year": 2024,
        "summary": "Unified formulation."
    }

    res = agent.download_and_ingest(candidate, rag_engine=mock_rag)
    assert res["status"] == "success"
    assert res["chunks_indexed"] == 5
    assert not res["already_existed"]

    pdf_file = temp_dir / res["file_name"]
    assert pdf_file.exists()
    assert pdf_file.read_bytes().startswith(b"%PDF")

    json_file = temp_dir / f"{pdf_file.stem}.json"
    assert json_file.exists()
    meta = json.loads(json_file.read_text(encoding="utf-8"))
    assert meta["paper_id"] == "2401.99999"
    assert "sha256" in meta
    assert len(meta["sha256"]) == 64

    # 再次调用应复用缓存 (idempotent)
    res2 = agent.download_and_ingest(candidate, rag_engine=mock_rag)
    assert res2["status"] == "success"
    assert res2["already_existed"] is True


def test_session_manager_stores_recent_candidates():
    s = ConversationSession(session_id="test_sess_cand", title="测试会话")
    SessionManager.save_session(s)

    cands = [{"paper_id": "p1", "title": "Paper 1"}, {"paper_id": "p2", "title": "Paper 2"}]
    SessionManager.update_recent_candidates("test_sess_cand", cands)

    loaded = SessionManager.get_recent_candidates("test_sess_cand")
    assert len(loaded) == 2
    assert loaded[0]["title"] == "Paper 1"

    # 清理
    p = SessionManager._file_path("test_sess_cand")
    if p.exists():
        p.unlink()
