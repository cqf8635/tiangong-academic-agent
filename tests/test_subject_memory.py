from subject_memory import transition_subject_state


def test_subject_memory_lifecycle():
    empty = {"subject": "", "sub_subject": "", "status": "INIT", "evolution_path": []}

    first = transition_subject_state(empty, "INIT", "推导 VQE", subject_entity="VQE", intent="stepwise_derivation")
    assert first["subject"] == "VQE"
    assert first["status"] == "LOCKED"
    assert first["evolution_path"] == ["VQE"]

    inherited = transition_subject_state(first, "INHERIT", "它为什么会遇到贫瘠高原？", sub_subject="贫瘠高原")
    assert inherited["subject"] == "VQE"
    assert inherited["sub_subject"] == "贫瘠高原"
    assert inherited["action"] == "INHERIT"

    appended = transition_subject_state(inherited, "APPEND", "加入局域代价函数", subject_entity="局域代价函数")
    assert appended["subject"] == "VQE & 局域代价函数"
    assert appended["status"] == "EVOLVING"
    assert appended["evolution_path"] == ["VQE", "局域代价函数"]

    replaced = transition_subject_state(appended, "REPLACE", "换个话题推导狄拉克方程", subject_entity="狄拉克方程")
    assert replaced["subject"] == "狄拉克方程"
    assert replaced["evolution_path"] == ["狄拉克方程"]
    assert replaced["status"] == "LOCKED"


def test_append_is_idempotent():
    state = {"subject": "VQE & 局域代价函数", "evolution_path": ["VQE", "局域代价函数"], "status": "EVOLVING"}
    result = transition_subject_state(state, "APPEND", "再次讨论局域代价函数", subject_entity="局域代价函数")
    assert result["subject"] == "VQE & 局域代价函数"
    assert result["evolution_path"] == ["VQE", "局域代价函数"]
