import pytest
from unittest.mock import MagicMock, patch
from master_agent_workflow import (
    MasterAgentState,
    general_tutor_node,
    evidence_synthesis_node,
    dialectical_inquiry_node,
    reflective_revision_node,
    academic_reviewer_node,
    author_revision_node,
    reviewer_decision_routing,
    build_master_workflow,
    MasterEngineRegistry,
)


def test_dynamic_specialist_spawning_and_collaboration(monkeypatch):
    """测试多学科交叉课题下，通用辅导节点能够动态派生特派专家子代理并合流，同时记录点对点总线日志"""
    mock_llm_client = MagicMock()
    mock_rag = MagicMock()
    mock_rag.search.return_value = []
    mock_paper_agent = MagicMock()
    mock_paper = MagicMock()
    mock_paper.title = "Lattice QCD on Quantum Computers"
    mock_paper.paper_id = "2301.0001"
    mock_paper.authors = ["Author A"]
    mock_paper.year = 2023
    mock_paper.summary = "Quantum simulation of lattice gauge theories."
    mock_paper.url = "https://arxiv.org/abs/2301.0001"
    mock_paper.model_dump.return_value = {
        "title": "Lattice QCD on Quantum Computers",
        "paper_id": "2301.0001",
        "authors": ["Author A"],
        "year": 2023,
        "summary": "Quantum simulation of lattice gauge theories.",
        "url": "https://arxiv.org/abs/2301.0001",
    }
    mock_paper_agent.search_arxiv.return_value = [mock_paper]
    
    # Mock LLM 返回专家研判与主研正文
    def mock_safe_completion(client, model, messages, **kwargs):
        content = messages[0]["content"] if messages and "content" in messages[0] else ""
        if "特派专家" in content:
            return "【学科底层机理】给出该学科算符约束与第一性原理边界条件。"
        return "【主研推演正文】综合各学科机理，建立全套形式化推导。"

    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: mock_llm_client)
    monkeypatch.setattr(MasterEngineRegistry, "get_rag_engine", lambda: mock_rag)
    monkeypatch.setattr(MasterEngineRegistry, "get_paper_search_agent", lambda: mock_paper_agent)
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", mock_safe_completion)

    state: MasterAgentState = {
        "user_query": "结合连续格点QCD与量子变分算法，如何设计非微扰夸克禁闭的量子线路模拟方案？",
        "active_subject": "格点规范理论与量子模拟",
        "sub_subject": None,
        "subject_evolution_log": [],
        "topic_state": None,
        "history_summary": "",
        "active_doc": None,
        "recent_paper_candidates": [],
        "retrieved_evidence": [],
        "intent_decision": {
            "intent": "research_design",
            "domain": "quantum_physics_and_computing",
            "is_cross_disciplinary": True,
            "disciplines": ["格点量子色动力学", "量子线路算法"],
            "sub_queries": [
                {"discipline": "格点量子色动力学", "query": "连续规范作用量在离散格点上的辛几何保持"},
                {"discipline": "量子线路算法", "query": "含噪声中等规模量子芯片上的哈密顿量拟设与变分编码"}
            ]
        },
        "target_agent": "StudyPlannerAgent",
        "final_answer": "",
        "tts_audio_file": None,
        "tts_voice": "zh-CN-XiaoyiNeural",
        "topic_catalog": [],
        "matched_topic": None,
        "pending_action": None,
        "peer_review_notes": None,
        "review_verdict": None,
        "review_scores_round1": None,
        "revision_count": 0,
        "agent_dialogue_history": [],
        "status": "init",
    }

    out = general_tutor_node(state)

    # 验证派生了 2 个领域的特派专家
    hist = out.get("agent_dialogue_history", [])
    assert len(hist) >= 3  # 2位专家交付 + 1次向审稿人提交初稿
    actions = [h["action"] for h in hist]
    assert actions.count("DELIVER_DISCIPLINARY_FOUNDATION") == 2
    assert out["status"] == "answered_by_tutor"
    assert out["final_answer"] != ""
    assert len(out["retrieved_evidence"]) == 1
    assert "Lattice QCD on Quantum Computers" in out["retrieved_evidence"][0]["doc_name"]
    assert len(out["recent_paper_candidates"]) == 1
    print("\n✅ 动态特派专家并行派生与总线合流及在线证据考据测试通过！")


def test_dialectical_inquiry_and_reflective_revision(monkeypatch):
    """测试多智能体学术博弈质询（去冗提纯、证据链核验）与主研团队基于研讨对话的深度反思重构闭环"""
    mock_llm_client = MagicMock()
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: mock_llm_client)

    inquiry_text = (
        "#### 🔬 学术博弈质询与去冗备忘 (Dialectical Inquiry Memorandum)\n\n"
        "1. 【相关性与去冗质询】：初稿前两段花费大量篇幅科普量子计算的历史与通识背景，与求解夸克禁闭的具体量子线路无直接因果，必须全量剔除；\n"
        "2. 【证据支撑性审查】：关于连续极限与热力学极限顺序的断言缺乏文献中的相变论证，未考虑非阿贝尔规范场的格点退化奇异点，需明确适用边界；\n"
        "3. 【核心机制反思追问】：请主研学者反思如何在无阿贝尔假设下确保算符因次量纲自洽，并给出严格的防御性边界说明。"
    )

    reflected_answer_text = (
        "### 1. 紧凑型格点规范理论量子变分线路构建\n"
        "（已直接切入核心控制算符，去除全部通识历史背景）\n"
        "设哈密顿量为 H_W = ...，算符严格定义在自然单位制下。[方案正文]...\n\n"
        "#### 💡 学术研讨反思与边界沉淀备忘\n"
        "- 剔除内容：彻底删除了量子计算发展史等无关冗余背景；\n"
        "- 边界修正：针对质询专员指出的格点奇异点，明确限定方案仅在弱耦合与非阿贝尔渐近自由区间成立，标记为【推测性假设】。"
    )

    # 阶段 1: 博弈质询席位执行去冗与证据链核验
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", lambda *args, **kwargs: inquiry_text)

    state_initial: MasterAgentState = {
        "user_query": "格点规范理论与量子变分方案",
        "intent_decision": {"intent": "research_design"},
        "target_agent": "StudyPlannerAgent",
        "final_answer": "包含冗余背景的初稿推演内容...",
        "retrieved_evidence": [{"title": "Lattice Gauge", "text": "Asymptotic freedom in non-abelian gauge theories"}],
        "dialectical_notes": None,
        "reflection_notes": None,
        "agent_dialogue_history": [],
        "status": "answered_by_tutor"
    }

    res_inquiry = dialectical_inquiry_node(state_initial)
    assert res_inquiry["status"] == "dialectical_inquiry_completed"
    assert "学术博弈质询与去冗备忘" in res_inquiry["dialectical_notes"]
    assert len(res_inquiry["agent_dialogue_history"]) == 1
    assert res_inquiry["agent_dialogue_history"][0]["action"] == "DIALECTICAL_INQUIRY_AND_PRUNING"
    print("\n✅ 第一阶段通过: 学术博弈质询专员成功对初稿执行去冗提纯与证据追问，写入研讨对话流！")

    # 阶段 2: 主研团队阅读对话流，执行深度认知反思并重构解答
    state_after_inquiry = {**state_initial, **res_inquiry}
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", lambda *args, **kwargs: reflected_answer_text)

    res_reflection = reflective_revision_node(state_after_inquiry)
    assert res_reflection["status"] == "reflective_revision_completed"
    assert "💡 学术研讨反思与边界沉淀备忘" in res_reflection["final_answer"]
    assert "剔除内容" in res_reflection["final_answer"]
    assert len(res_reflection["agent_dialogue_history"]) == 2
    assert res_reflection["agent_dialogue_history"][1]["action"] == "REFLECTIVE_RECONSTRUCTION"
    print("✅ 第二阶段通过: 主研学者消化对话流完成深度反思与证据对齐，产出经反思检验的高密度终稿！")


def test_multisource_online_evidence_injection_and_deduplication(monkeypatch):
    """测试多源检索 (arXiv + CrossRef + OpenAlex) 在通用研讨节点中的并发汇聚、内存去重与证据切片注入"""
    mock_llm_client = MagicMock()
    mock_rag = MagicMock()
    mock_rag.search.return_value = []
    mock_paper_agent = MagicMock()

    # 构造包含重复项的多源搜索结果
    mock_paper_agent.search_multisource.return_value = {
        "arxiv_papers": [
            {
                "title": "Quadruped Robot Locomotion via Reinforcement Learning",
                "paper_id": "2301.1111",
                "authors": ["Author RL"],
                "year": 2023,
                "summary": "Deep RL for robust quadruped gaits across rough terrain.",
                "url": "https://arxiv.org/abs/2301.1111",
                "source": "arxiv",
            },
            # 与 OpenAlex 论文重复的 arXiv 预印本版本 (标题微差但标准化后相同)
            {
                "title": "Quadruped Robot Locomotion via Reinforcement Learning!",
                "paper_id": "2301.1111-v2",
                "authors": ["Author RL"],
                "year": 2023,
                "summary": "Duplicate preprint version.",
                "url": "https://arxiv.org/abs/2301.1111-v2",
                "source": "arxiv",
            }
        ],
        "openalex_papers": [
            {
                "title": "Model Predictive Control for Legged Robots",
                "id": "https://openalex.org/W123456",
                "authors": ["Author MPC"],
                "year": 2022,
                "summary": "Convex MPC formulation for dynamic quadruped gaits.",
                "url": "https://doi.org/10.1109/TRO.2022.01",
                "doi": "10.1109/TRO.2022.01",
                "venue": "IEEE TRO",
                "source": "openalex",
            }
        ],
        "crossref_papers": [
            {
                "title": "Contact-Implicit Trajectory Optimization",
                "id": "10.1177/0278364921100",
                "authors": ["Author Traj"],
                "year": 2021,
                "summary": "Trajectory optimization through contacts without pre-specifying modes.",
                "url": "https://doi.org/10.1177/0278364921100",
                "doi": "10.1177/0278364921100",
                "source": "crossref",
            }
        ],
        "github_repos": [],
        "total_found": 4,
    }

    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: mock_llm_client)
    monkeypatch.setattr(MasterEngineRegistry, "get_rag_engine", lambda: mock_rag)
    monkeypatch.setattr(MasterEngineRegistry, "get_paper_search_agent", lambda: mock_paper_agent)
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", lambda *args, **kwargs: "四足机器人运动控制理论解答")

    state: MasterAgentState = {
        "user_query": "四足机器人的运动控制前沿有哪些主流方案？",
        "active_subject": "四足机器人控制",
        "sub_subject": None,
        "subject_evolution_log": [],
        "topic_state": None,
        "history_summary": "",
        "active_doc": None,
        "recent_paper_candidates": [],
        "retrieved_evidence": [],
        "intent_decision": {
            "intent": "academic_discussion",
            "domain": "robotics",
            "search_query": "四足机器人 运动控制",
            "online_search_query": "quadruped robot locomotion control",
        },
        "target_agent": "StudyPlannerAgent",
        "final_answer": "",
        "tts_audio_file": None,
        "tts_voice": "zh-CN-XiaoyiNeural",
        "topic_catalog": [],
        "matched_topic": None,
        "pending_action": None,
        "peer_review_notes": None,
        "review_verdict": None,
        "review_scores_round1": None,
        "revision_count": 0,
        "agent_dialogue_history": [],
        "status": "init",
    }

    out = general_tutor_node(state)

    # 4 篇输入中，1 篇重复被剔除，最终保留 3 篇唯一文献
    ev = out.get("retrieved_evidence", [])
    assert len(ev) == 3
    doc_names = [e["doc_name"] for e in ev]
    assert "Quadruped Robot Locomotion via Reinforcement Learning" in doc_names
    assert "Model Predictive Control for Legged Robots" in doc_names
    assert "Contact-Implicit Trajectory Optimization" in doc_names

    # 验证 location_label 包含了多源标识 (arXiv, OpenAlex, CrossRef)
    labels = [e["location_label"] for e in ev]
    assert any("arXiv" in lbl for lbl in labels)
    assert any("OpenAlex" in lbl for lbl in labels)
    assert any("CrossRef" in lbl for lbl in labels)

    # 验证审稿节点不会丢失 evidence
    rev_out = academic_reviewer_node(out)
    assert len(rev_out.get("retrieved_evidence", [])) == 3
    print("\n✅ 多源在线学术文献去重汇聚与证据链路贯通测试通过！")


def test_evidence_synthesis_node_and_full_academic_pipeline_convergence(monkeypatch):
    """测试多源资料整理归纳智能体（跨学科去噪、分类、机理骨架提炼）及全学术流程闭环贯通"""
    mock_llm_client = MagicMock()
    monkeypatch.setattr(MasterEngineRegistry, "get_llm_client", lambda: mock_llm_client)

    slate_content = (
        "### 📑 《多源学术资产与证据归纳备忘录》\n\n"
        "#### 1. 🔍 跨学科噪声过滤与有效性审查 (Noise Filtering)\n"
        "- 剔除噪声：《Urban Forestry & Microclimate》与拓扑量子计算无因果关联，确认为领域漂移噪声，已全量过滤；\n"
        "- 保留有效资料：《Topological Quantum Computation》、《Fractional Statistics in 2D》.\n\n"
        "#### 2. 📚 资料层级分类 (Taxonomy & Categorization)\n"
        "- 【经典专著/权威教材】：Nayak et al.《Non-Abelian Anyons and Topological Quantum Computation》\n"
        "- 【前沿预印本】：arXiv:2608.12345《Anyonic Braiding on Superconducting Quiddits》\n\n"
        "#### 3. 🔬 核心机理、方程与边界条件骨架 (Core Mechanisms & Boundary Slate)\n"
        "- 核心机制：拓扑非阿贝尔编织酉矩阵 U(B) = exp(i theta J)；\n"
        "- 适用边界与断裂带：仅在系统满足拓扑能隙 Delta > 0 且准粒子间距足够大时编织统计无误差，在有限温激发态下存在退相干断裂带。"
    )

    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", lambda *args, **kwargs: slate_content)

    state: MasterAgentState = {
        "user_query": "非阿贝尔任意子的拓扑量子计算方案与编织编解码机制",
        "active_subject": "拓扑量子计算",
        "sub_subject": None,
        "subject_evolution_log": [],
        "topic_state": None,
        "history_summary": "",
        "active_doc": None,
        "recent_paper_candidates": [
            {
                "paper_id": "2608.12345",
                "title": "Anyonic Braiding on Superconducting Quiddits",
                "authors": ["Alice", "Bob"],
                "year": 2026,
                "source": "arxiv",
                "summary": "Braiding operations on superconducting architecture."
            },
            {
                "paper_id": "forestry_001",
                "title": "Urban Forestry & Microclimate",
                "authors": ["Ecologist X"],
                "year": 2025,
                "source": "crossref",
                "summary": "Microclimate regulation by urban green parks."
            }
        ],
        "retrieved_evidence": [
            {
                "doc_name": "Non-Abelian Anyons Monograph",
                "location_label": "Chapter 4",
                "text": "Braiding operators form representation of the braid group B_n."
            }
        ],
        "intent_decision": {"intent": "research_design"},
        "target_agent": "StudyPlannerAgent",
        "final_answer": "初步关于非阿贝尔任意子编织的物理推导草案...",
        "tts_audio_file": None,
        "tts_voice": "zh-CN-XiaoyiNeural",
        "topic_catalog": [],
        "matched_topic": None,
        "pending_action": None,
        "synthesized_evidence_slate": None,
        "dialectical_notes": None,
        "reflection_notes": None,
        "peer_review_notes": None,
        "agent_dialogue_history": [],
        "status": "answered_by_planner",
    }

    # 1. 验证整理归纳智能体独立节点执行
    synth_res = evidence_synthesis_node(state)
    assert synth_res["status"] == "evidence_synthesis_completed"
    assert "多源学术资产与证据归纳备忘录" in synth_res["synthesized_evidence_slate"]
    assert "跨学科噪声过滤与有效性审查" in synth_res["synthesized_evidence_slate"]
    assert any(log["action"] == "EVIDENCE_CLEANSING_AND_SYNTHESIS" for log in synth_res["agent_dialogue_history"])
    print("\n✅ 阶段 1: 学术证据整理归纳智能体完成多源资料去噪、分级分类及机制提炼！")

    # 2. 验证博弈质询节点优先摄入整理归纳备忘录
    state_with_slate = {**state, **synth_res}
    inquiry_mock = "#### 🔬 学术博弈质询与去冗备忘\n1. 针对整理归纳备忘录中指出的有限温激发态退相干断裂带，主研方案未给出保护边界！"
    monkeypatch.setattr("master_agent_workflow.safe_chat_completion", lambda *args, **kwargs: inquiry_mock)

    inquiry_res = dialectical_inquiry_node(state_with_slate)
    assert inquiry_res["status"] == "dialectical_inquiry_completed"
    print("✅ 阶段 2: 博弈质询席位成功基于整理归纳备忘录发起靶向质询！")

    # 3. 验证主图 LangGraph 拓扑结构：所有学术管道汇流到 evidence_synthesis
    app = build_master_workflow()
    graph_dict = app.get_graph().to_json()
    nodes = [n["id"] for n in graph_dict["nodes"]]
    assert "evidence_synthesis" in nodes
    assert "dialectical_inquiry" in nodes
    assert "reflective_revision" in nodes

    # 检查边连接：各学术节点均流向 evidence_synthesis
    edges = [(e["source"], e["target"]) for e in graph_dict["edges"]]
    assert ("online_paper_search_pipeline", "evidence_synthesis") in edges
    assert ("document_rag_pipeline", "evidence_synthesis") in edges
    assert ("meeting_summary_pipeline", "evidence_synthesis") in edges
    assert ("cross_modal_pipeline", "evidence_synthesis") in edges
    assert ("general_tutor_pipeline", "evidence_synthesis") in edges
    assert ("evidence_synthesis", "dialectical_inquiry") in edges
    assert ("dialectical_inquiry", "reflective_revision") in edges
    assert ("reflective_revision", "tts_synthesis") in edges
    assert ("system_control_pipeline", "tts_synthesis") in edges
    print("✅ 阶段 3: LangGraph 全学术管道汇流整理归纳与思辨反思拓扑结构验证 100% 正确！")
