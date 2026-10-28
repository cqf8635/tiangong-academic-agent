"""Voice Perception Agent built with LangGraph.

Architecture:
- StateGraph:
    [START]
       │
       ▼
    [listen_vad] (自动声音检测与静音断句)
       │ (检测到有效说话且静音停顿)
       ▼
    [transcribe] (本地 Whisper-large-v3 转录)
       │
       ▼
    [clean_and_format] (语义清洗与文本规整)
       │
       ▼
    [route_to_brain] (准备交由下游意图/知识库智能体)
       │
       ▼
     [END]
"""

from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import torch
import os
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from config_utils import get_default_monitor_source, resolve_whisper_path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

MODEL_DIR = resolve_whisper_path(BASE_DIR)
DEFAULT_MONITOR = get_default_monitor_source()


# ==========================================
# 1. 智能体状态定义 (Agent State Schema)
# ==========================================
class VoiceAgentState(TypedDict):
    mode: str                         # "mic" (麦克风) 或 "system" (视频扬声器内录)
    threshold: float                  # VAD 音量触发阈值
    silence_timeout: float            # 静音截断时长（秒）
    audio_bytes: Optional[bytes]      # 录制捕获的音频数据
    raw_transcription: str            # Whisper 识别的原始文本
    final_text: str                   # 清洗整理后的确定文本
    source_type: str                  # 识别源 ("user_speech" / "video_audio")
    status: str                       # 状态: "listening" | "success" | "no_speech" | "error"
    error_message: Optional[str]      # 异常信息


# ==========================================
# 2. Whisper 本地模型单例管理
# ==========================================
class WhisperEngine:
    _instance = None
    _pipe = None

    @classmethod
    def get_pipeline(cls, model_dir: Path = MODEL_DIR):
        if cls._pipe is None:
            use_cuda = torch.cuda.is_available()
            dtype = torch.float16 if use_cuda else torch.float32

            print(f"🧠 [WhisperEngine] 加载本地模型: {model_dir.name} (GPU CUDA: {use_cuda})...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_dir,
                dtype=dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            if use_cuda:
                model.to("cuda:0")

            processor = AutoProcessor.from_pretrained(model_dir)
            cls._pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=dtype,
                device=0 if use_cuda else -1,
            )
        return cls._pipe


# ==========================================
# 3. LangGraph 节点函数实现 (Nodes)
# ==========================================
def get_audio_source(mode: str) -> str:
    if mode == "system":
        try:
            res = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True, check=True)
            for line in res.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and "monitor" in parts[1] and "Speaker" in parts[1]:
                    return parts[1]
        except Exception:
            pass
        return DEFAULT_MONITOR
    return "default"


def calc_rms(chunk_bytes: bytes) -> float:
    count = len(chunk_bytes) // 2
    if count == 0:
        return 0.0
    shorts = struct.unpack(f"<{count}h", chunk_bytes)
    return math.sqrt(sum(s * s for s in shorts) / count)


# 常见 Whisper 静音/底噪幻觉黑名单（YouTube 片尾字幕语料残余）
HALLUCINATION_PATTERNS = [
    "thank you for watching",
    "thanks for watching",
    "thank you.",
    "thank you",
    "ご視聴ありがとうございました",
    "ご視聴ありがとう",
    "チャンネル登録",
    "subtitles by",
    "字幕由",
    "请不吝赐教",
    "谢谢大家",
]


def listen_vad_node(state: VoiceAgentState) -> Dict[str, Any]:
    """节点 1: 自动声音活动检测与自适应环境降噪静音断句"""
    mode = state.get("mode", "mic")
    silence_timeout = state.get("silence_timeout", 1.3)

    source = get_audio_source(mode)
    rate = 16000
    chunk_size = int(rate * 2 * 0.1)  # 100ms
    silence_chunks_needed = int(silence_timeout / 0.1)

    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        "-f", "pulse", "-i", source,
        "-f", "s16le", "-ar", str(rate), "-ac", "1",
        "-",
    ]

    print(f"\n🟢 [ListenVAD] 正在监听: {'🎬 视频声音(扬声器)' if mode == 'system' else '🎤 麦克风人声'} ...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio_buffer = bytearray()
    is_speaking = False
    silence_counter = 0
    active_speech_chunks = 0

    # 动态环境噪音校准（初始 200ms）
    noise_samples = []
    for _ in range(2):
        chunk = proc.stdout.read(chunk_size)
        if chunk:
            noise_samples.append(calc_rms(chunk))
    ambient_noise = max(noise_samples) if noise_samples else 400.0
    # 动态动态阈值：至少高于底噪 2.5 倍，底限 1100
    dynamic_threshold = max(state.get("threshold", 1100.0), ambient_noise * 2.5, 1100.0)

    try:
        while True:
            raw_chunk = proc.stdout.read(chunk_size)
            if not raw_chunk:
                break

            rms = calc_rms(raw_chunk)

            if rms > dynamic_threshold:
                if not is_speaking:
                    is_speaking = True
                    sys.stdout.write("🎙️  [检测到说话，正在收音...] ")
                    sys.stdout.flush()

                audio_buffer.extend(raw_chunk)
                active_speech_chunks += 1
                silence_counter = 0
            elif is_speaking:
                audio_buffer.extend(raw_chunk)
                silence_counter += 1
                if silence_counter >= silence_chunks_needed:
                    sys.stdout.write(" ⏹️ [说话结束，触发转录]\n")
                    sys.stdout.flush()
                    break

        proc.terminate()
        proc.wait()

        # 有效说话判定：总时长 >= 0.8s 且实际高能量语音片段 >= 3帧(300ms)，避免呼吸/底噪误触发
        if len(audio_buffer) >= rate * 2 * 0.8 and active_speech_chunks >= 3:
            return {
                "audio_bytes": bytes(audio_buffer),
                "status": "speech_detected",
                "source_type": "video_audio" if mode == "system" else "user_speech",
            }
        else:
            if is_speaking:
                sys.stdout.write("\r" + " " * 45 + "\r")
                sys.stdout.flush()
            return {
                "audio_bytes": None,
                "status": "no_speech",
                "final_text": "",
            }

    except Exception as e:
        proc.terminate()
        return {
            "audio_bytes": None,
            "status": "error",
            "error_message": str(e),
        }


def transcribe_node(state: VoiceAgentState) -> Dict[str, Any]:
    """节点 2: 调用本地 Whisper-large-v3 语音转录"""
    if state["status"] != "speech_detected" or not state.get("audio_bytes"):
        return {"raw_transcription": ""}

    audio_bytes = state["audio_bytes"]
    pipe = WhisperEngine.get_pipeline()

    # 转换为 float32 numpy 数组
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    try:
        res = pipe(
            samples,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True,
        )
        text = res.get("text", "").strip()
        return {
            "raw_transcription": text,
            "status": "transcribed" if text else "empty_transcription",
        }
    except Exception as e:
        return {
            "raw_transcription": "",
            "status": "error",
            "error_message": f"Whisper推理失败: {e}",
        }


def clean_and_format_node(state: VoiceAgentState) -> Dict[str, Any]:
    """节点 3: 语义清洗、去幻觉与语音控制指令识别"""
    raw_text = state.get("raw_transcription", "").strip()
    if not raw_text:
        return {"final_text": "", "status": "completed"}

    # 1. 过滤 Whisper 常见的静音/杂音幻觉模式
    lower_text = raw_text.lower().strip(". !?,，。？！")
    for pattern in HALLUCINATION_PATTERNS:
        if pattern in lower_text or lower_text == pattern:
            return {"final_text": "", "status": "filtered_hallucination"}

    # 2. 文本格式规整
    cleaned = raw_text.strip(" ,，。？！")

    # 3. 语音控制指令检测：完全由大模型语义识别 (voice_system_control)，零关键词硬编码
    try:
        from intent_recognition_agent import IntentRecognitionAgent
        intent_agent = IntentRecognitionAgent()
        decision = intent_agent.parse_intent(cleaned)
        if decision.intent == "voice_system_control" or decision.target_agent == "SystemControlAgent":
            return {
                "final_text": cleaned,
                "status": "shutdown",
            }
    except Exception:
        pass

    return {
        "final_text": cleaned,
        "status": "completed",
    }


def route_decision(state: VoiceAgentState) -> str:
    """条件路由分支判断"""
    status = state.get("status", "")
    if status == "speech_detected":
        return "transcribe"
    return "end"


# ==========================================
# 4. 构建 LangGraph 状态图 (Build Graph)
# ==========================================
def build_voice_agent_graph():
    graph = StateGraph(VoiceAgentState)

    # 添加节点
    graph.add_node("listen_vad", listen_vad_node)
    graph.add_node("transcribe", transcribe_node)
    graph.add_node("clean_and_format", clean_and_format_node)

    # 添加边与路由
    graph.add_edge(START, "listen_vad")
    graph.add_conditional_edges(
        "listen_vad",
        route_decision,
        {
            "transcribe": "transcribe",
            "end": END,
        },
    )
    graph.add_edge("transcribe", "clean_and_format")
    graph.add_edge("clean_and_format", END)

    return graph.compile()


# ==========================================
# 5. 智能体封装入口 (Agent Runner)
# ==========================================
class LangGraphVoiceAgent:
    def __init__(
        self,
        mode: str = "mic",
        threshold: float = 600.0,
        silence_timeout: float = 1.3,
        records_dir: Path = Path(__file__).resolve().parent / "records",
    ):
        self.mode = mode
        self.threshold = threshold
        self.silence_timeout = silence_timeout
        self.records_dir = records_dir
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.app = build_voice_agent_graph()

    def listen_once(self) -> VoiceAgentState:
        """执行单轮听取与转录图流程"""
        initial_state: VoiceAgentState = {
            "mode": self.mode,
            "threshold": self.threshold,
            "silence_timeout": self.silence_timeout,
            "audio_bytes": None,
            "raw_transcription": "",
            "final_text": "",
            "source_type": "user_speech",
            "status": "listening",
            "error_message": None,
        }
        return self.app.invoke(initial_state)

    def run_continuous_loop(self, on_message_callback=None):
        """持续循环监听并实时持久化保存为结构化 JSONL 与 Markdown 纪要流水"""
        import json
        from datetime import datetime

        session_start_time = datetime.now()
        timestamp_str = session_start_time.strftime("%Y%m%d_%H%M%S")
        jsonl_path = self.records_dir / f"record_{timestamp_str}.jsonl"
        md_path = self.records_dir / f"record_{timestamp_str}.md"

        # 初始化 Markdown 头部
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 语音识别与会议记录流水 ({timestamp_str})\n\n")
            f.write(f"- **开始时间**: {session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **音频通道**: {'视频声音(扬声器内录)' if self.mode == 'system' else '麦克风人声'}\n\n")
            f.write("## 逐字记录流水\n\n")

        print(f"\n🚀 [LangGraphVoiceAgent] 智能体已启动 (LangGraph 编排驱动)")
        print(f"📁 实时记录已绑定: {jsonl_path.name}")
        print("💡 提示: 说话完毕停顿自动断句；随时说【录音结束】或【结束对话】即可语音关闭智能体\n")

        sentence_index = 0

        try:
            while True:
                result = self.listen_once()
                final_text = result.get("final_text")
                status = result.get("status")

                if final_text:
                    sentence_index += 1
                    current_time_str = datetime.now().strftime("%H:%M:%S")
                    source = result.get("source_type", "voice")
                    source_label = "视频内录" if source == "video_audio" else "麦克风"

                    # 1. 终端打印
                    print(f"🎯 [{sentence_index}] 【{current_time_str}】[{source_label}]:")
                    print(f"   >>> {final_text}\n")

                    # 2. 实时追加持久化到 JSONL (便于后续大模型分段总结与 RAG 检索)
                    record_item = {
                        "index": sentence_index,
                        "timestamp": current_time_str,
                        "source": source,
                        "text": final_text,
                    }
                    with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
                        f_jsonl.write(json.dumps(record_item, ensure_ascii=False) + "\n")
                        f_jsonl.flush()

                    # 3. 实时追加到 Markdown 纪要
                    with open(md_path, "a", encoding="utf-8") as f_md:
                        f_md.write(f"- `[{current_time_str}]` **{source_label}**: {final_text}\n")
                        f_md.flush()

                    if on_message_callback:
                        on_message_callback(final_text, source)

                # 检测到语音退出指令
                if status == "shutdown":
                    print(f"🛑 [语音控制] 检测到结束指令: 【{final_text}】")
                    break

                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n👋 LangGraph 语音智能体已手动安全停止。")

        # 打印会话统计与存储位置
        session_end_time = datetime.now()
        duration_sec = int((session_end_time - session_start_time).total_seconds())
        print("\n" + "=" * 55)
        print("📊 【本次会话记录已全部持久化安全落盘】")
        print(f"⏱️  总录制时长: {duration_sec // 60} 分 {duration_sec % 60} 秒")
        print(f"📝 累计记录有效句子: {sentence_index} 条")
        print(f"💾 JSONL 结构化数据: {jsonl_path}")
        print(f"📄 Markdown 格式文本: {md_path}")
        print("=" * 55 + "\n")


class _VoicePerceptionGraphRunner:
    """适配器：将 VoicePerceptionAgent 的持续 VAD 监听与 WebSocket 钩子无缝桥接至 backend_api"""

    def __init__(self, agent: "VoicePerceptionAgent"):
        self.agent = agent

    def invoke(self, init_state: Dict[str, Any]) -> Dict[str, Any]:
        return self.agent._run_backend_loop(init_state)


class VoicePerceptionAgent(LangGraphVoiceAgent):
    """兼容 backend_api 的实时语音感知与会议流水转录智能体"""

    def __init__(
        self,
        source: str = "default",
        mode: str = "mic",
        threshold: float = 600.0,
        silence_timeout: float = 1.3,
        records_dir: Optional[Path] = None,
    ):
        effective_mode = "system" if ("monitor" in str(source) or mode == "system") else "mic"
        super().__init__(
            mode=effective_mode,
            threshold=threshold,
            silence_timeout=silence_timeout,
            records_dir=records_dir or (Path(__file__).resolve().parent / "records"),
        )
        self.source = source or "default"

    def _is_still_running(self) -> bool:
        mod = sys.modules.get("backend_api")
        if mod and hasattr(mod, "voice_manager"):
            return bool(getattr(mod.voice_manager, "is_running", False))
        return True

    def _clean_and_format(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """清洗转录文本并原地更新 state，以触发 backend_api 的 WebSocket 广播钩子"""
        raw_text = (state.get("latest_raw_text") or state.get("raw_transcription") or "").strip()
        sub_res = clean_and_format_node({"raw_transcription": raw_text})
        cleaned = (sub_res.get("final_text") or "").strip()
        state["cleaned_text"] = cleaned
        if cleaned:
            state["current_index"] = int(state.get("current_index") or 0) + 1
        if sub_res.get("status") == "shutdown":
            state["should_terminate"] = True
        return state

    def build_graph(self) -> _VoicePerceptionGraphRunner:
        return _VoicePerceptionGraphRunner(self)

    def _listen_chunked_vad(self) -> Optional[bytes]:
        """支持 100ms 级响应停止信号的 VAD 录音监听"""
        source = get_audio_source(self.mode) if self.source == "default" else self.source
        rate = 16000
        chunk_size = int(rate * 2 * 0.1)
        silence_chunks_needed = int(self.silence_timeout / 0.1)

        cmd = [
            "ffmpeg", "-loglevel", "quiet",
            "-f", "pulse", "-i", source,
            "-f", "s16le", "-ar", str(rate), "-ac", "1",
            "-",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        audio_buffer = bytearray()
        is_speaking = False
        silence_counter = 0
        active_speech_chunks = 0

        try:
            noise_samples = []
            for _ in range(2):
                if not self._is_still_running():
                    return None
                chunk = proc.stdout.read(chunk_size) if proc.stdout else b""
                if chunk:
                    noise_samples.append(calc_rms(chunk))
            ambient_noise = max(noise_samples) if noise_samples else 400.0
            dynamic_threshold = max(self.threshold, ambient_noise * 2.5, 1100.0)

            while self._is_still_running():
                raw_chunk = proc.stdout.read(chunk_size) if proc.stdout else b""
                if not raw_chunk:
                    break
                rms = calc_rms(raw_chunk)
                if rms > dynamic_threshold:
                    is_speaking = True
                    audio_buffer.extend(raw_chunk)
                    active_speech_chunks += 1
                    silence_counter = 0
                elif is_speaking:
                    audio_buffer.extend(raw_chunk)
                    silence_counter += 1
                    if silence_counter >= silence_chunks_needed:
                        break
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                pass

        if len(audio_buffer) >= rate * 2 * 0.8 and active_speech_chunks >= 3:
            return bytes(audio_buffer)
        return None

    def _run_backend_loop(self, state: Dict[str, Any]) -> Dict[str, Any]:
        import json
        from datetime import datetime

        session_start_time = datetime.now()
        timestamp_str = session_start_time.strftime("%Y%m%d_%H%M%S")
        jsonl_path = self.records_dir / f"record_{timestamp_str}.jsonl"
        md_path = self.records_dir / f"record_{timestamp_str}.md"
        state["session_file"] = str(jsonl_path)

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# 语音识别与会议记录流水 ({timestamp_str})\n\n")
            f.write(f"- **开始时间**: {session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **音频通道**: {'视频声音(扬声器内录)' if self.mode == 'system' else '麦克风人声'}\n\n")
            f.write("## 逐字记录流水\n\n")

        print(f"🎙️ [VoicePerceptionAgent] 已开启后台实时语音感知与会议流水记录: {jsonl_path.name}")
        while self._is_still_running() and not state.get("should_terminate"):
            audio_bytes = self._listen_chunked_vad()
            if not audio_bytes:
                if not self._is_still_running():
                    break
                continue

            t_res = transcribe_node({"status": "speech_detected", "audio_bytes": audio_bytes})
            raw_text = (t_res.get("raw_transcription") or "").strip()
            if not raw_text:
                continue

            state["latest_raw_text"] = raw_text
            state["active_source"] = "system" if self.mode == "system" else "mic"
            # 调用（可能已被 backend_api 注入 WebSocket 广播钩子的）_clean_and_format
            self._clean_and_format(state)
            cleaned = (state.get("cleaned_text") or "").strip()
            if cleaned:
                now_str = datetime.now().strftime("%H:%M:%S")
                record_item = {
                    "index": state.get("current_index", 1),
                    "timestamp": now_str,
                    "source": state.get("active_source", "mic"),
                    "text": cleaned,
                }
                with open(jsonl_path, "a", encoding="utf-8") as f_jsonl:
                    f_jsonl.write(json.dumps(record_item, ensure_ascii=False) + "\n")
                with open(md_path, "a", encoding="utf-8") as f_md:
                    f_md.write(f"- `[{now_str}]` **{record_item['source']}**: {cleaned}\n")

        print("🛑 [VoicePerceptionAgent] 语音感知已安全停止。")
        return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于 LangGraph 的语音感知与持久化智能体")
    parser.add_argument("--mode", choices=["mic", "system"], default="mic", help="mic: 麦克风, system: 视频扬声器内录")
    parser.add_argument("--threshold", type=float, default=600.0, help="声音能量触发阈值")
    parser.add_argument("--silence", type=float, default=1.3, help="静音自动截断时长（秒）")
    args = parser.parse_args()

    agent = LangGraphVoiceAgent(
        mode=args.mode,
        threshold=args.threshold,
        silence_timeout=args.silence,
    )
    agent.run_continuous_loop()
