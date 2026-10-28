import pytest
import numpy as np
from unittest.mock import MagicMock
from intent_recognition_agent import IntentRecognitionAgent
from master_agent_workflow import MasterAgentState, intent_routing_node, MasterEngineRegistry


def test_multiturn_state_driven_intent_flow(monkeypatch):
    """测试多轮对话中，意图解析直接从 LangGraph 状态字段感知研讨现场的端到端演进"""
    mock_rag = MagicMock()
    mock_rag.embedding_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    mock_rag.search.return_value = []
    monkeypatch.setattr(MasterEngineRegistry, "get_rag_engine", lambda: mock_rag)

    agent = IntentRecognitionAgent()

    # -------------------------------------------------------------
    # 第一轮：冷启动状态，无文献无主体 -> 必须触发全网检索 (need_online=True)
    # -------------------------------------------------------------
    state_turn1: MasterAgentState = {
        "user_query": "请检索近两年量子场论中关于非微扰方法、格点规范理论和量子模拟的代表性最新论文，说明主要问题、方法和贡献。",
        "active_subject": None,
        "sub_subject": None,
        "subject_evolution_log": [],
        "topic_state": {"current_topic": "", "sub_topic": "", "status": "INIT", "evolution_path": []},
        "history_summary": "",
        "active_doc": None,
        "recent_paper_candidates": [],
        "retrieved_evidence": [],
        "intent_decision": None,
        "target_agent": "",
        "final_answer": "",
        "status": "init"
    }

    decision1 = agent.parse_intent(state=state_turn1)
    assert decision1.need_online_paper_search is True
    assert decision1.subject_action == "INIT"
    print("\n✅ 第一轮通过: 冷启动无候选文献，准确感知并触发全网检索 (need_online=True)")

    # -------------------------------------------------------------
    # 第二轮：LangGraph 状态中已沉淀文献池 -> 用户筛选分类，严禁重新外搜 (need_online=False)
    # -------------------------------------------------------------
    candidates = [
        {
            "paper_id": "2608.30160",
            "title": "From Self-Dual to Physical CP^{N-1}",
            "authors": ["Author A", "Author B"],
            "year": 2026,
            "source": "arxiv",
            "summary": "Nonperturbative dynamics in 2D CP^{N-1} models and theta vacua on the lattice."
        },
        {
            "paper_id": "https://doi.org/10.1021/acsphotonics.3c00457",
            "title": "Roadmap for Optical Metasurfaces",
            "authors": ["Kuznetsov et al."],
            "year": 2024,
            "source": "openalex",
            "summary": "Review and perspective on nanophotonic optical metasurfaces."
        }
    ]

    state_turn2: MasterAgentState = {
        "user_query": "请从刚才的论文中筛选真正讨论强耦合规范场论的工作，并区分预印本、正式发表论文和综述。",
        "active_subject": "量子场论中的非微扰方法、格点规范理论与量子模拟",
        "sub_subject": "文献筛选",
        "subject_evolution_log": ["INIT -> 【量子场论中的非微扰方法、格点规范理论与量子模拟】"],
        "topic_state": {
            "current_topic": "量子场论中的非微扰方法、格点规范理论与量子模拟",
            "sub_topic": "文献筛选",
            "status": "LOCKED",
            "evolution_path": ["INIT -> 【量子场论中的非微扰方法、格点规范理论与量子模拟】"]
        },
        "history_summary": "第一轮确立了量子场论非微扰与格点规范理论主题，已检索候选文献。",
        "active_doc": None,
        "recent_paper_candidates": candidates,  # 👈 核心：状态字段中已存在就绪资产
        "retrieved_evidence": [],
        "intent_decision": None,
        "target_agent": "",
        "final_answer": "",
        "status": "init"
    }

    decision2 = agent.parse_intent(state=state_turn2)
    assert decision2.need_online_paper_search is False
    assert decision2.subject_action == "INHERIT"
    print(f"✅ 第二轮通过: 状态中已有文献池，大模型自然识别为内生操作 (need_online=False, 意图={decision2.intent})")

    # 验证在 intent_routing_node 下，文献自动无损装配到 evidence_chunks，且不重复外搜
    node_out2 = intent_routing_node(state_turn2)
    assert node_out2["intent_decision"]["need_online_paper_search"] is False
    assert len(node_out2["retrieved_evidence"]) == 2
    assert any("From Self-Dual" in c["text"] for c in node_out2["retrieved_evidence"])
    print("✅ 第二轮工作流通过: 证据链直接全量无损消费 LangGraph 状态中文献资产")

    # -------------------------------------------------------------
    # 第三轮：针对已锁定主题深入数学推导 -> 自动路由至 DerivationAgent (INHERIT)
    # -------------------------------------------------------------
    state_turn3: MasterAgentState = {
        "user_query": "针对格点规范作用量中的费米子倍增问题，具体怎么推导 Wilson 项如何消除虚假极点？",
        "active_subject": "量子场论中的非微扰方法、格点规范理论与量子模拟",
        "sub_subject": "Wilson费米子推导",
        "subject_evolution_log": state_turn2["subject_evolution_log"],
        "topic_state": state_turn2["topic_state"],
        "history_summary": "探讨了非微扰场论与格点规范理论，完成了候选文献初筛。",
        "active_doc": None,
        "recent_paper_candidates": candidates,
        "retrieved_evidence": [],
        "intent_decision": None,
        "target_agent": "",
        "final_answer": "",
        "status": "init"
    }

    decision3 = agent.parse_intent(state=state_turn3)
    assert decision3.intent == "stepwise_derivation"
    assert decision3.target_agent == "DerivationAgent"
    assert decision3.subject_action == "INHERIT"
    assert decision3.need_online_paper_search is False
    print("✅ 第三轮通过: 深入公式定理，准确继承主体并派发 DerivationAgent (need_online=False)")
