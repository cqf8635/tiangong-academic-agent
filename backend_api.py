"""FastAPI backend for the research multi-agent workflow and web frontend.

Provides:
- REST API for Chat, Document Ingest, Meeting Summaries, Audio Stream.
- WebSocket for live VAD audio events & status broadcasting.
- Static file serving for the research assistant web UI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uvicorn.error")

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from document_rag_engine import UniversalDocumentRAGEngine
from intent_recognition_agent import IntentRecognitionAgent
from master_agent_workflow import MasterAgentState, build_master_workflow, register_stream_callback
from meeting_summarizer import MeetingSummarizer
from online_paper_search_agent import OnlinePaperSearchAgent

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

app = FastAPI(title="Research Multi-Agent System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
RECORDS_DIR = BASE_DIR / "records"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RECORDS_DIR.mkdir(parents=True, exist_ok=True)

# 懒加载单例组件 (共享全局单例，避免重复初始化)
rag_engine = UniversalDocumentRAGEngine()
meeting_summarizer = MeetingSummarizer()
paper_searcher = OnlinePaperSearchAgent()
from radar_sync_engine import RadarSyncEngine
radar_sync_engine = RadarSyncEngine(paper_searcher)

from master_agent_workflow import MasterEngineRegistry
MasterEngineRegistry.rag_engine = rag_engine
MasterEngineRegistry.meeting_summarizer = meeting_summarizer
MasterEngineRegistry.paper_searcher = paper_searcher
master_app = build_master_workflow()


@app.on_event("startup")
async def on_app_startup():
    """应用启动挂钩：自动拉起天工雷达后台静默轮询与断点续航同步引擎"""
    radar_sync_engine.start_scheduler()


@app.on_event("shutdown")
async def on_app_shutdown():
    radar_sync_engine.stop_scheduler()


from session_manager import ChatMessage, ConversationSession, SessionManager

# ==========================================
# 1. 领域会话频道管理 (Domain Workspaces & Sessions)
# ==========================================
class CreateSessionRequest(BaseModel):
    title: str
    domain: str = "general"


from config_utils import get_default_llm_model, get_llm_timeout

@app.get("/api/config")
async def get_system_config():
    """获取系统运行期动态配置（如当前激活的大模型名称）"""
    return {
        "model_name": get_default_llm_model(),
        "llm_timeout": get_llm_timeout(100.0),
    }


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": SessionManager.list_sessions()}


@app.post("/api/sessions/create")
async def create_session(req: CreateSessionRequest):
    session = SessionManager.create_session(title=req.title, domain=req.domain)
    return {"session": session.model_dump()}


@app.get("/api/sessions/{session_id}")
async def get_session_detail(session_id: str):
    session = SessionManager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session.model_dump()}


class RenameSessionRequest(BaseModel):
    title: str


@app.put("/api/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameSessionRequest):
    session = SessionManager.rename_session(session_id, req.title)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session.model_dump()}


@app.post("/api/sessions/{session_id}/clear")
async def clear_session_messages(session_id: str):
    session = SessionManager.clear_session_messages(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session": session.model_dump()}


@app.post("/api/sessions/clear_all")
async def clear_all_sessions():
    init_s = SessionManager.clear_all_sessions()
    return {"initial_session": init_s.model_dump()}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    ok = SessionManager.delete_session(session_id)
    return {"success": ok}


# ==========================================
# 2. 聊天与问答接口 (支持多会话隔离)
# ==========================================
class ChatRequest(BaseModel):
    query: str
    image_data: Optional[str] = None  # Base64 附带截图或图片 (可为空)
    session_id: Optional[str] = None
    context_hint: Optional[str] = None
    voice: Optional[str] = "zh-CN-XiaoyiNeural"


def _execute_chat_workflow(req: ChatRequest) -> Dict[str, Any]:
    session_id = req.session_id
    if not session_id or session_id == "null":
        new_sess = SessionManager.create_session(title=req.query.strip()[:24])
        session_id = new_sess.session_id

    # 获取该会话的长期研讨摘要、聚焦研讨文献与主题生命周期状态机
    session = SessionManager.get_session(session_id)
    history_summary = ""
    active_doc = SessionManager.get_active_doc(session_id)
    topic_state = SessionManager.get_topic_state(session_id)
    
    # 外科手术式在历史中锁定最合适的主题定论 (0ms 极速寻址，拒绝全量地毯式盲搜)
    matched_topic = SessionManager.match_best_historical_topic(session_id, query=req.query)
    history_summary = SessionManager.get_scoped_thematic_context(session_id, query=req.query)

    # 保存用户发送的消息到指定领域频道 (包含附带截图)
    user_msg = ChatMessage(sender="user", content=req.query, image_url=req.image_data)
    SessionManager.add_message(session_id, user_msg)

    recent_candidates = SessionManager.get_recent_candidates(session_id)
    initial_state: MasterAgentState = {
        "user_query": req.query,
        "user_image": req.image_data,
        "intent_decision": None,
        "retrieved_evidence": [],
        "final_answer": "",
        "target_agent": "",
        "tts_audio_file": None,
        "tts_voice": req.voice or "zh-CN-XiaoyiNeural",
        "history_summary": history_summary,
        "active_doc": active_doc,
        "topic_state": topic_state,
        "recent_paper_candidates": recent_candidates,
        "topic_catalog": session.topic_catalog if session else [],
        "matched_topic": matched_topic,
        "pending_action": None,
        "synthesized_evidence_slate": None,
        "dialectical_notes": None,
        "reflection_notes": None,
        "peer_review_notes": None,
        "review_verdict": None,
        "review_scores_round1": None,
        "revision_count": 0,
        "agent_dialogue_history": [],
        "status": "init",
    }

    # 执行 LangGraph 状态图 (原生挂载 thread_id 记忆会话检查点)
    try:
        config = {"configurable": {"thread_id": session_id}}
        result_state = master_app.invoke(initial_state, config=config)
    except Exception as e:
        print(f"❌ [Master Workflow Error]: {e}")
        err_msg = str(e)
        if "503" in err_msg or "Service temporarily unavailable" in err_msg:
            fallback_text = "⚠️ **【搭档提示】上游大模型 API 出现暂时性繁忙排队 (HTTP 503 Service Unavailable)**\n\n上游服务暂时过载，请稍候点击下方快捷词条或重新发送问题即可继续研讨。"
        elif "429" in err_msg or "Rate limit" in err_msg:
            fallback_text = "⚠️ **【搭档提示】模型调用频次已达限制 (HTTP 429 Rate Limit)**\n\n请稍候 3~5 秒后再次提问。"
        elif "524" in err_msg:
            fallback_text = "⚠️ **【搭档提示】上游代理响应超时 (HTTP 524 Gateway Timeout)**\n\n上游代理排队较慢，请稍后重试。"
        else:
            fallback_text = f"⚠️ **【搭档提示】大模型接口调用异常**: {err_msg}\n\n建议稍后重新发送提问。"

        result_state = {
            "intent_decision": None,
            "target_agent": "ErrorFallbackAgent",
            "topic_state": topic_state,
            "retrieved_evidence": [],
            "final_answer": fallback_text,
            "tts_audio_file": None,
            "status": "error_fallback",
        }

    # 持久化同步主题生命周期状态机与前序学术候选池
    if result_state.get("topic_state"):
        SessionManager.update_topic_state(session_id, result_state["topic_state"])

    if result_state.get("recent_paper_candidates"):
        SessionManager.update_recent_candidates(session_id, result_state["recent_paper_candidates"])

    if result_state.get("active_doc"):
        s_obj = SessionManager.get_session(session_id)
        if s_obj:
            s_obj.active_doc = result_state["active_doc"]
            SessionManager.save_session(s_obj)

    audio_url = None
    if result_state.get("tts_audio_file") and os.path.exists(result_state["tts_audio_file"]):
        audio_name = Path(result_state["tts_audio_file"]).name
        audio_url = f"/api/audio/{audio_name}"

    from document_rag_engine import extract_answer_mindmap
    final_ans = result_state.get("final_answer", "")
    mindmap_md = extract_answer_mindmap(final_ans, req.query)

    # 保存智能体回复到该领域频道
    agent_msg = ChatMessage(
        sender="agent",
        content=final_ans,
        intent_decision=result_state.get("intent_decision"),
        target_agent=result_state.get("target_agent"),
        retrieved_evidence=result_state.get("retrieved_evidence", []),
        mindmap_markdown=mindmap_md,
        audio_url=audio_url,
        agent_dialogue_history=result_state.get("agent_dialogue_history", []),
        synthesized_evidence_slate=result_state.get("synthesized_evidence_slate"),
        dialectical_notes=result_state.get("dialectical_notes") or result_state.get("peer_review_notes"),
    )
    SessionManager.add_message(session_id, agent_msg)
    
    # 自动归档本轮主题档案 (无损提炼核心定论，供后续轮次外科手术式调取)
    s_now = SessionManager.get_session(session_id)
    turn_num = len([m for m in (s_now.messages if s_now else []) if m.sender == "agent"])
    SessionManager.record_turn_topic(session_id, turn_num, req.query, agent_msg)

    return {
        "session_id": session_id,
        "query": req.query,
        "history_summary": history_summary,
        "topic_state": result_state.get("topic_state"),
        "intent_decision": result_state.get("intent_decision"),
        "target_agent": result_state.get("target_agent"),
        "retrieved_evidence": result_state.get("retrieved_evidence", []),
        "final_answer": final_ans,
        "mindmap_markdown": mindmap_md,
        "audio_url": audio_url,
        "active_doc": result_state.get("active_doc") or active_doc,
        "matched_topic": matched_topic,
        "pending_action": result_state.get("pending_action"),
        "synthesized_evidence_slate": result_state.get("synthesized_evidence_slate"),
        "dialectical_notes": result_state.get("dialectical_notes") or result_state.get("peer_review_notes"),
        "reflection_notes": result_state.get("reflection_notes"),
        "peer_review_notes": result_state.get("peer_review_notes"),
        "review_verdict": result_state.get("review_verdict"),
        "review_scores_round1": result_state.get("review_scores_round1"),
        "agent_dialogue_history": result_state.get("agent_dialogue_history", []),
        "status": result_state.get("status"),
    }


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    return _execute_chat_workflow(req)


@app.post("/api/chat/stream")
async def chat_stream_endpoint(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def token_handler(token: str):
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "token", "content": token})

    async def run_workflow():
        try:
            register_stream_callback(token_handler)
            result = await asyncio.to_thread(_execute_chat_workflow, req)
            register_stream_callback(None)
            await queue.put({"type": "done", "data": result})
        except Exception as e:
            register_stream_callback(None)
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(None)

    asyncio.create_task(run_workflow())

    async def sse_generator():
        yield f"data: {json.dumps({'type': 'start', 'query': req.query}, ensure_ascii=False)}\n\n"
        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )



# ==========================================
# 2. 文档上传与知识库构建 (PDF / Word / PPT)
# ==========================================
@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    
    dest_path = UPLOAD_DIR / file.filename
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 自动解析切块入库 (依托全格式解析与最大宽容度兜底)
    try:
        chunks_count = rag_engine.ingest_file(dest_path)
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=400, detail=f"文件解析入库失败 ({ext}): {e}")

    return {
        "filename": file.filename,
        "doc_type": ext.lstrip(".") or "txt",
        "chunks_count": chunks_count,
        "status": "ingested",
    }


@app.get("/api/documents")
async def list_documents():
    docs = []
    # 过滤掉元数据 json、临时文件以及隐藏文件，仅呈现正规文献
    ignored_suffixes = {".json", ".tmp", ".bak", ".part", ".crdownload"}
    for f in UPLOAD_DIR.glob("*.*"):
        if f.is_file() and f.suffix.lower() not in ignored_suffixes and not f.name.startswith("."):
            docs.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "modified_at": f.stat().st_mtime,
                "type": f.suffix.lstrip(".").lower()
            })
    docs.sort(key=lambda x: x["modified_at"], reverse=True)
    return {"documents": docs}


@app.get("/api/documents/{doc_name}/mindmap")
async def get_doc_mindmap(doc_name: str):
    """获取指定文献的深层细分全书/全篇经卷思维导图"""
    mindmap_md = rag_engine.get_document_mindmap(doc_name)
    return {"doc_name": doc_name, "mindmap_markdown": mindmap_md}


@app.get("/api/documents/{doc_name}/download")
async def download_document(doc_name: str):
    """下载或直接阅读本地琅嬛书阁中已入库的文献原卷 (PDF / Office / Text)"""
    target_file = UPLOAD_DIR / doc_name
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail="文献物理文件不存在")
    
    suffix = target_file.suffix.lower()
    media_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/markdown; charset=utf-8",
    }
    media_type = media_map.get(suffix, "application/octet-stream")
    return FileResponse(
        path=target_file,
        filename=target_file.name,
        media_type=media_type,
    )


@app.delete("/api/documents/{doc_name}")
async def delete_document(doc_name: str):
    """彻底删除指定文献，同时清除 ChromaDB 向量库中对应切片与磁盘物理文件"""
    # 1. 从 ChromaDB 向量库彻底删除切片
    deleted_from_db = rag_engine.delete_document(doc_name)

    # 2. 从 uploads/ 目录删除物理文件
    target_file = UPLOAD_DIR / doc_name
    deleted_from_disk = False
    if target_file.exists():
        target_file.unlink(missing_ok=True)
        deleted_from_disk = True

    return {
        "doc_name": doc_name,
        "deleted_from_db": deleted_from_db,
        "deleted_from_disk": deleted_from_disk,
        "success": True,
    }


# ==========================================
# 3. 会议记录与纪要 (Meeting & Action Items)
# ==========================================
@app.get("/api/meetings/latest")
async def get_latest_meeting():
    jsonl_files = list(RECORDS_DIR.glob("record_*.jsonl"))
    if not jsonl_files:
        return {"has_meeting": False, "report": None, "raw_records": []}

    jsonl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_file = jsonl_files[0]

    # 读取最近 30 条发言流水
    raw_lines = []
    with open(latest_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    raw_lines.append(json.loads(line.strip()))
                except Exception:
                    pass

    return {
        "has_meeting": True,
        "filename": latest_file.name,
        "total_speeches": len(raw_lines),
        "recent_speeches": raw_lines[-20:],
    }


@app.post("/api/meetings/summarize")
async def trigger_meeting_summary():
    jsonl_files = list(RECORDS_DIR.glob("record_*.jsonl"))
    if not jsonl_files:
        raise HTTPException(status_code=404, detail="No meeting records found")

    jsonl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    report = meeting_summarizer.summarize_session(jsonl_files[0])
    return {"report": report.model_dump()}


# ==========================================
# 3.1 Obsidian 知识库与 Canvas 白板一键导出
# ==========================================
class ObsidianExportRequest(BaseModel):
    session_id: str
    export_canvas: bool = True
    custom_vault_path: Optional[str] = None


@app.post("/api/export/obsidian")
async def export_to_obsidian(req: ObsidianExportRequest):
    """将当前会话全量导出为 Obsidian 双链笔记 (.md) 与原生白板 (.canvas)"""
    from obsidian_exporter import ObsidianKnowledgeExporter
    vault_p = Path(req.custom_vault_path) if req.custom_vault_path else None
    exporter = ObsidianKnowledgeExporter(vault_path=vault_p)
    result = exporter.export_session(session_id=req.session_id, export_canvas=req.export_canvas)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result


# ==========================================
# 4. 实时语音感知与时间轴流式推送 (Live Voice Stream & WebSocket)
# ==========================================
class VoiceSessionManager:
    def __init__(self):
        self.is_running = False
        self.agent_thread = None
        self.connected_websockets: List[WebSocket] = []

    async def broadcast_speech(self, entry: Dict[str, Any]):
        dead_sockets = []
        for ws in self.connected_websockets:
            try:
                await ws.send_json(entry)
            except Exception:
                dead_sockets.append(ws)
        for ws in dead_sockets:
            if ws in self.connected_websockets:
                self.connected_websockets.remove(ws)

    def start_voice_agent(self, loop, mode: str = "mic"):
        if self.is_running:
            return
        self.is_running = True
        mode = mode if mode in {"mic", "system"} else "mic"

        def run_worker():
            try:
                from voice_langgraph_agent import VoicePerceptionAgent
                agent = VoicePerceptionAgent(source="default", mode=mode)
                
                # 回调推送到 WebSocket
                orig_clean_and_format = agent._clean_and_format
                def hooked_clean_and_format(state):
                    res = orig_clean_and_format(state)
                    entry = {
                        "type": "speech",
                        "index": state["current_index"],
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "text": state["cleaned_text"],
                        "source": state.get("active_source", "mic")
                    }
                    asyncio.run_coroutine_threadsafe(self.broadcast_speech(entry), loop)
                    return res
                
                agent._clean_and_format = hooked_clean_and_format
                app_agent = agent.build_graph()
                
                init_state = {
                    "is_listening": True,
                    "audio_buffer": [],
                    "current_index": 0,
                    "last_speech_time": 0.0,
                    "session_file": "",
                    "latest_raw_text": "",
                    "cleaned_text": "",
                    "should_terminate": False,
                    "active_source": "mic"
                }
                app_agent.invoke(init_state)
            except Exception as e:
                print(f"[Voice Worker Error]: {e}")
            finally:
                self.is_running = False

        import threading
        self.agent_thread = threading.Thread(target=run_worker, daemon=True)
        self.agent_thread.start()

    def stop_voice_agent(self):
        self.is_running = False


voice_manager = VoiceSessionManager()


@app.websocket("/ws/voice_stream")
async def websocket_voice_endpoint(websocket: WebSocket):
    await websocket.accept()
    voice_manager.connected_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in voice_manager.connected_websockets:
            voice_manager.connected_websockets.remove(websocket)


class VoiceStartRequest(BaseModel):
    mode: str = "mic"  # mic: 麦克风；system: 系统声音/视频会议回环


@app.post("/api/voice/start")
async def start_voice_perception(req: Optional[VoiceStartRequest] = None):
    loop = asyncio.get_event_loop()
    voice_manager.start_voice_agent(loop, mode=(req.mode if req else "mic"))
    return {"status": "started", "is_recording": True}


@app.post("/api/voice/stop")
async def stop_voice_perception():
    voice_manager.stop_voice_agent()
    return {"status": "stopped", "is_recording": False}


# ==========================================
# 5. 音频资源播放与动态语音合成接口
# ==========================================
class TTSSynthesizeRequest(BaseModel):
    text: str
    voice: Optional[str] = "zh-CN-XiaoyiNeural"


@app.post("/api/tts/synthesize")
async def synthesize_speech_on_demand(req: TTSSynthesizeRequest):
    """按需动态合成指定文本的导师语音并返回专属音频路径"""
    from master_agent_workflow import tts_synthesis_node
    res = tts_synthesis_node({"final_answer": req.text, "tts_voice": req.voice or "zh-CN-XiaoyiNeural"})
    if res.get("tts_audio_file") and os.path.exists(res["tts_audio_file"]):
        audio_name = Path(res["tts_audio_file"]).name
        return {"audio_url": f"/api/audio/{audio_name}", "status": "success"}
    return {"audio_url": None, "status": "failed"}


@app.get("/api/audio/{filename}")
async def get_tts_audio_by_name(filename: str):
    """获取指定唯一哈希的导师语音音频文件，彻底消除跨对话声音串音"""
    if filename == "latest":
        mp3_path = BASE_DIR / "tts_response.mp3"
    else:
        mp3_path = BASE_DIR / "audio_cache" / filename
        if not mp3_path.exists():
            mp3_path = BASE_DIR / filename

    if not mp3_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(mp3_path, media_type="audio/mpeg")

# ==========================================
# 5. 天工雷达 (TianGong Academic Radar)
# ==========================================
class RadarSearchRequest(BaseModel):
    query: Optional[str] = None
    major: Optional[str] = "PHYSICS"
    sub_categories: Optional[List[str]] = None
    search_in: str = "all"  # all | title | authors | abstract
    sort_by: str = "submittedDate"  # submittedDate | relevance
    start: int = 0
    max_results: int = 15


@app.get("/api/radar/taxonomy")
async def get_radar_taxonomy():
    """获取权威 8 大领域、各学科群与 150+ 细分子学科分类全景字典"""
    from academic_taxonomy import ACADEMIC_TAXONOMY
    return {"taxonomy": ACADEMIC_TAXONOMY}


@app.post("/api/radar/search")
async def radar_search(req: RadarSearchRequest):
    """精准多学科雷达检索，提供子学科分布直方图与高精度文献卡片数据 (支持本地快照离线容灾)"""
    # 1. 尝试从本地持久化快照中先读取（如果未输入特定搜索词，且是请求第 1 页默认流）
    sub_key = req.sub_categories[0] if (req.sub_categories and len(req.sub_categories) == 1) else "all"
    is_default_browse = (not req.query or not req.query.strip()) and req.start == 0 and req.sort_by == "submittedDate"

    try:
        res = paper_searcher.search_radar(
            query=req.query,
            major=req.major,
            sub_categories=req.sub_categories,
            search_in=req.search_in,
            sort_by=req.sort_by,
            start=req.start,
            max_results=req.max_results,
        )
        if res and res.get("papers") and is_default_browse and not res.get("is_cached_snapshot"):
            # 自动沉淀为本地快照
            radar_sync_engine.save_snapshot(req.major or "CS", sub_key, res)

        # 🌟 若在线检索无返回或超时受限，自动由本地持久化快照平滑接管 (零报错、零空白、零延时)
        if not (res and res.get("papers")):
            cached_snapshot = radar_sync_engine.read_snapshot(req.major or "CS", sub_key)
            if cached_snapshot:
                cached_snapshot["is_cached_snapshot"] = True
                return {"status": "success", "data": cached_snapshot, "is_cached_snapshot": True}

        return {"status": "success", "data": res}
    except Exception as e:
        import traceback
        traceback.print_exc()
        # 2. 如果网络波动/超时，自动回退到本地持久化快照
        cached_snapshot = radar_sync_engine.read_snapshot(req.major or "CS", sub_key)
        if cached_snapshot:
            cached_snapshot["is_cached_snapshot"] = True
            return {"status": "success", "data": cached_snapshot, "is_cached_snapshot": True}
        raise HTTPException(status_code=500, detail=f"雷达检索异常: {str(e)}")


class RadarIngestRequest(BaseModel):
    paper_id: str
    title: str
    pdf_url: str
    authors: Optional[List[str]] = []
    year: Optional[int] = None
    summary: Optional[str] = ""


@app.post("/api/radar/ingest")
async def radar_ingest_paper(req: RadarIngestRequest):
    """将雷达发现的文献直接下载并向量化嵌入本地琅嬛书阁知识库 (BGE-M3/BAAI)，异步线程池调度避免阻塞事件循环"""
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        ingest_res = await loop.run_in_executor(
            None,
            lambda: paper_searcher.download_and_ingest(
                paper={
                    "paper_id": req.paper_id,
                    "title": req.title,
                    "pdf_url": req.pdf_url,
                    "authors": req.authors,
                    "year": req.year,
                    "summary": req.summary,
                },
                rag_engine=rag_engine,
            )
        )
        return {"status": "success", "result": ingest_res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文献入库失败: {str(e)}")


class RadarTranslateRequest(BaseModel):
    title: str
    text: Optional[str] = ""
    abstract: Optional[str] = ""


@app.post("/api/radar/translate")
async def radar_translate(req: RadarTranslateRequest):
    """为雷达论文提供专业学术级中英双璧精译、核心创新提要与摘要翻译"""
    try:
        from openai import OpenAI
        from config_utils import get_default_llm_model
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        model = get_default_llm_model()
        if not api_key:
            raise ValueError("未配置 API Key")
        client = OpenAI(api_key=api_key, base_url=base_url)

        paper_abstract = (req.abstract or req.text or "").strip()
        prompt = f"""你是一位严谨资深的学术学者与翻译专家。请针对以下 arXiv 论文进行中文学术提炼与精译：
论文英文标题：{req.title}
论文英文摘要：{paper_abstract}

请严格按以下 JSON 格式输出，不要附加任何其他多余文本：
{{
  "title_zh": "精准流畅的中文学术规范标题",
  "thesis_zh": "核心创新与突破提要（用1~2句专业凝练概括核心方法、解决的关键问题与创新贡献）",
  "abstract_zh": "中文学术摘要精译（通顺严谨，保留专业学术术语与数学符号格式）"
}}"""

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            timeout=25,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ==========================================
# 5.1 网站图标与科研搭档静态形象服务
# ==========================================
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    icon_path = STATIC_DIR / "img" / "partner_ruoxi_scholarly_1.jpg"
    if icon_path.exists():
        return FileResponse(icon_path, media_type="image/jpeg")
    return Response(content="", media_type="image/x-icon", status_code=204)


# ==========================================
# 6. 挂载前端静态文件
# ==========================================
# 静态文件挂载
img_extracted_dir = STATIC_DIR / "extracted_images"
img_extracted_dir.mkdir(parents=True, exist_ok=True)
app.mount("/extracted_images", StaticFiles(directory=str(img_extracted_dir)), name="extracted_images")
app.mount("/css", StaticFiles(directory=STATIC_DIR / "css"), name="css")
app.mount("/js", StaticFiles(directory=STATIC_DIR / "js"), name="js")
app.mount("/img", StaticFiles(directory=STATIC_DIR / "img"), name="img")

if (STATIC_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def run_server():
    import uvicorn
    print("\n" + "=" * 70)
    print("🚀 【科研学习与会议协同多智能体系统】启动成功！")
    print("🌐 访问网址: http://localhost:8008")
    print("=" * 70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="info")


if __name__ == "__main__":
    run_server()
