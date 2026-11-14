import base64
import io
import json

from online_paper_search_agent import OnlinePaperSearchAgent


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_github_search_records_readme_and_commit_without_claiming_code_validation(monkeypatch):
    responses = [
        json.dumps({"items": [{
            "full_name": "org/project", "description": "A research implementation",
            "stargazers_count": 42, "html_url": "https://github.com/org/project",
            "language": "Python", "default_branch": "main", "pushed_at": "2026-01-01T00:00:00Z",
        }]}).encode(),
        json.dumps({"content": base64.b64encode(b"# Project\nUsage instructions").decode()}).encode(),
        json.dumps([{"sha": "abc123", "commit": {"author": {"date": "2026-01-01T00:00:00Z"}}}]).encode(),
    ]
    monkeypatch.setattr("online_paper_search_agent.urllib.request.urlopen", lambda *args, **kwargs: _Response(responses.pop(0)))
    monkeypatch.setenv("GITHUB_DETAIL_FETCH_LIMIT", "1")
    agent = object.__new__(OnlinePaperSearchAgent)
    resources = agent.search_github("scientific implementation", max_results=1)
    assert len(resources) == 1
    assert resources[0].version_ref == "abc123"
    assert resources[0].version_ref_type == "commit"
    assert resources[0].evidence_level == "repository_metadata_and_readme"
    assert "Usage instructions" in resources[0].summary
    assert "source files not inspected or executed" in resources[0].content_scope
