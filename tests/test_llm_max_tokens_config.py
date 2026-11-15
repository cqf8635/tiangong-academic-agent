import ast
from pathlib import Path

from config_utils import get_llm_max_tokens


BASE_DIR = Path(__file__).resolve().parent.parent


def test_get_llm_max_tokens_defaults_and_env(monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    assert get_llm_max_tokens() == 12288
    assert get_llm_max_tokens(8192) == 8192

    monkeypatch.setenv("LLM_MAX_TOKENS", "16384")
    assert get_llm_max_tokens() == 16384

    monkeypatch.setenv("LLM_MAX_TOKENS", "0")
    assert get_llm_max_tokens() == 12288

    monkeypatch.setenv("LLM_MAX_TOKENS", "invalid")
    assert get_llm_max_tokens() == 12288


def test_no_low_hardcoded_max_tokens_in_core_modules():
    """Prevent accidental low hardcoded max_tokens (e.g. 4096) in long-form / JSON LLM calls."""
    targets = [
        "master_agent_workflow.py",
        "online_paper_search_agent.py",
        "retrieval_planner.py",
        "intent_recognition_agent.py",
        "meeting_summarizer.py",
    ]
    allowlist = {
        # Short 1-line academic keyword translation only
        ("online_paper_search_agent.py", "extract_academic_keywords"): 256,
    }

    violations = []
    for filename in targets:
        source = (BASE_DIR / filename).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filename)

        parent_func = {}

        class FuncVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_fn = None

            def visit_FunctionDef(self, node):
                prev = self.current_fn
                self.current_fn = node.name
                self.generic_visit(node)
                self.current_fn = prev

            def visit_Call(self, node):
                for kw in node.keywords:
                    if kw.arg == "max_tokens" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                        val = kw.value.value
                        if allowlist.get((filename, self.current_fn)) == val:
                            continue
                        if val <= 6144:
                            violations.append(f"{filename}:{node.lineno} ({self.current_fn}) max_tokens={val}")
                self.generic_visit(node)

        FuncVisitor().visit(tree)

    assert not violations, f"Found low hardcoded max_tokens: {violations}"
