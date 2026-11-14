"""Application controls with explicit execution results; no language-model calls."""

import re


def execute_system_control(action, parameters, *, session_id, voice_manager, session_manager):
    parameters = parameters if isinstance(parameters, dict) else {}
    if action == "start_recording":
        mode = parameters.get("mode") or "mic"
        if mode not in {"mic", "system"}:
            return {"status": "not_executed", "message": "未识别录音音源，请指定麦克风或系统声音。"}
        loop = voice_manager.event_loop
        if loop is None or loop.is_closed():
            return {"status": "not_executed", "message": "录音服务尚未就绪，本轮未开启录音。"}
        return voice_manager.start_voice_agent(loop, mode=mode)
    if action in {"stop_recording", "pause_recording"}:
        return voice_manager.stop_voice_agent()
    if action == "change_voice":
        voice = str(parameters.get("voice") or "").strip()
        voice = {"晓伊": "zh-CN-XiaoyiNeural", "晓晓": "zh-CN-XiaoxiaoNeural",
                 "晓臻": "zh-TW-HsiaoChenNeural", "云健": "zh-CN-YunjianNeural",
                 "云扬": "zh-CN-YunyangNeural"}.get(voice, voice)
        if not re.fullmatch(r"[a-z]{2,3}-[A-Z]{2}-[A-Za-z0-9]+Neural", voice):
            return {"status": "not_executed", "message": "未取得具体音色名称或有效音色 ID，本轮未切换音色。"}
        session = session_manager.get_session(session_id) if session_id else None
        if session is None:
            return {"status": "not_executed", "message": "未找到当前会话，本轮未保存音色设置。"}
        session.tts_voice = voice
        session_manager.save_session(session)
        return {"status": "completed", "message": f"当前会话朗读音色已切换为 {voice}。", "tts_voice": voice}
    if action == "clear_session":
        # Clearing checkpoints inside a running graph would be undone by its
        # next checkpoint. The API completes this after graph execution.
        return {"status": "pending", "message": "正在清空当前会话。"}
    return {"status": "not_executed", "message": "该控制动作尚无对应执行器，本轮未执行系统操作。"}
