"""Comprehensive Voice Input & Output Test Script.

Tests:
1. Voice Output: Speaker / Audio Output Test (generates a test tone & plays audio).
2. Voice Input: Microphone recording.
3. Voice Playback: Replays the recorded audio through speakers.
4. Voice Recognition: Uses local Whisper-large-v3 model for STT.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config_utils import resolve_whisper_path
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

MODEL_DIR = resolve_whisper_path(BASE_DIR)
RECORDED_WAV = BASE_DIR / "voice_test.wav"
BEEP_WAV = BASE_DIR / "beep_test.wav"


def generate_beep_wav(filename: Path, duration_sec: float = 0.8, freq: float = 523.25, sample_rate: int = 16000) -> None:
    """Generate a clean test sine wave beep sound."""
    num_samples = int(duration_sec * sample_rate)
    with wave.open(str(filename), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(num_samples):
            # Apply fade in/out envelope to avoid clicks
            envelope = min(1.0, i / 500, (num_samples - i) / 500)
            value = int(32767 * 0.4 * envelope * math.sin(2.0 * math.pi * freq * i / sample_rate))
            frames.extend(struct.pack("<h", value))
        wf.writeframes(frames)


def play_audio(wav_path: Path) -> bool:
    """Play audio using system tools (paplay, aplay, or ffplay)."""
    if not wav_path.is_file():
        print(f"[错误] 音频文件不存在: {wav_path}")
        return False

    players = [
        ("paplay", ["paplay", str(wav_path)]),
        ("aplay", ["aplay", "-q", str(wav_path)]),
        ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(wav_path)]),
    ]

    for name, cmd in players:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, check=True)
                return True
            except Exception as e:
                print(f"[警告] 使用 {name} 播放失败: {e}")
                continue

    # Fallback to sounddevice/soundfile if available
    try:
        import sounddevice as sd
        import soundfile as sf
        data, fs = sf.read(str(wav_path))
        sd.play(data, fs)
        sd.wait()
        return True
    except Exception as e:
        print(f"[错误] 播放音频失败，没有可用的播放器或组件: {e}")
        return False


def get_audio_source(mode: str = "mic") -> str:
    """获取 PulseAudio/PipeWire 录音通道（支持麦克风或视频/会议系统声音内录）"""
    if mode in ("system", "video", "meeting"):
        # 1. 优先获取当前系统正在使用的 Default Sink 的 monitor (扬声器/耳机内录通道)
        try:
            sink_res = subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True, timeout=1.0)
            def_sink = sink_res.stdout.strip()
            if def_sink:
                return f"{def_sink}.monitor"
        except Exception:
            pass
        # 2. 从 sources 列表中查找 monitor
        try:
            res = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True, timeout=1.0)
            for line in res.stdout.strip().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2 and "monitor" in parts[1]:
                    return parts[1]
        except Exception:
            pass
        return "default.monitor"
    return "default"


def record_audio(wav_path: Path, seconds: int = 0, sample_rate: int = 16000, mode: str = "mic") -> bool:
    """Record audio using PulseAudio/PipeWire from microphone or system/meeting monitor with manual Enter/Ctrl+C support."""
    source = get_audio_source(mode)
    is_sys = mode in ("system", "video", "meeting")
    source_name = "🎬 视频/会议声音(系统扬声器内录)" if is_sys else "🎤 麦克风人声"

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y",
            "-nostdin",  # 核心关键: 彻底禁止 ffmpeg 争夺终端标准输入，确保回车键 100% 毫秒级响应
            "-f", "pulse",
            "-i", source,
            "-ac", "1",
            "-ar", str(sample_rate),
        ]
        if seconds > 0:
            cmd.extend(["-t", str(seconds)])
        cmd.append(str(wav_path))

        hint = "请在电脑上播放视频或保持会议发声" if is_sys else "请对准麦克风说话"
        duration_desc = f"上限 {seconds} 秒 (随时按回车提前结束)" if seconds > 0 else "无限时长 (随时按回车结束)"

        print(f"\n🎙️ [正在录制: {source_name}]")
        print(f"👉 提示: {hint}")
        print(f"⏱️ 录音模式: 【{duration_desc}】")
        print("💡 结束方式: 讲完或播完后，在键盘轻按【回车键 Enter】或【Ctrl+C】即可立即停止并转录！\n")

        # stdin 彻底指向 DEVNULL，杜绝任何外部子进程劫持终端输入
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        start_t = time.time()

        try:
            import select
            has_select = hasattr(select, "select")
        except ImportError:
            has_select = False

        try:
            while proc.poll() is None:
                elapsed = int(time.time() - start_t)
                m, s = divmod(elapsed, 60)
                sys.stdout.write(f"\r⏳ [正在收音: {m:02d}:{s:02d}] · 随时按 [回车 Enter] 结束录制... ")
                sys.stdout.flush()

                if has_select:
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.3)
                    if rlist:
                        sys.stdin.readline()
                        break
                else:
                    time.sleep(0.3)
        except (KeyboardInterrupt, EOFError):
            print("\n⏹️ [收到终止指令]")

        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=2.0)
            except Exception:
                proc.terminate()
                proc.wait()

        print("\n✅ 录音已结束，正在转入本地高精度 Whisper 模型进行识别...")
        return wav_path.exists() and wav_path.stat().st_size > 44

    # 2. 备用 arecord (仅麦克风固定时长)
    sec_fixed = seconds if seconds > 0 else 10
    arecord = shutil.which("arecord")
    if arecord and not is_sys:
        for dev in ["pulse", "default"]:
            cmd = [
                arecord,
                "-q",
                "-D", dev,
                "-f", "S16_LE",
                "-r", str(sample_rate),
                "-c", "1",
                "-d", str(sec_fixed),
                str(wav_path),
            ]
            try:
                print(f"🎤 [正在录音 (arecord -D {dev})...] 录音 {sec_fixed} 秒:")
                subprocess.run(cmd, check=True)
                print("✅ 录音完成！")
                return True
            except subprocess.CalledProcessError:
                continue

    return False


def transcribe_audio(wav_path: Path, model_dir: Path, language: str = "auto") -> str:
    """Transcribe audio prioritizing OpenAI official whisper library (Solution 1)."""
    # -------------------------------------------------------------------------
    # 方案 1: OpenAI 官方原生库 (最强降温回退、内置重复压缩率检测、完美支持中英混合与长篇研讨)
    # -------------------------------------------------------------------------
    try:
        import gc
        import os
        import torch
        import whisper

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # 动态解析模型下载存储位置 (优先保存在项目同级 models 目录，如 /home/cqf/文档/models，零硬编码)
        download_root = str(model_dir.parent) if model_dir.parent.exists() else None

        print("\n" + "=" * 60)
        print("💡 【执行方案 1】: 调用 OpenAI 官方原生 Whisper 引擎")
        print(f"📁 模型加载目录: {download_root or '~/.cache/whisper'}")
        print(f"⚡ 计算设备: {device.upper()} (GPU FP16 极致显存优化)")
        print("=" * 60)

        # 核心显存保护: 先在 CPU 完成反序列化与字典释放，彻底避开 GPU 显存双倍 Checkpoint 复制陷阱 (杜绝暴涨至 7.43GB)
        model = whisper.load_model("large-v3", device="cpu", download_root=download_root)
        if device == "cuda":
            model = model.to("cuda")
            gc.collect()
            torch.cuda.empty_cache()

        lang_code = "zh" if language in ("zh", "chinese") else (None if language == "auto" else language)
        lang_desc = "中文 / 中英混合" if lang_code == "zh" else (lang_code or "auto (自适应识别)")
        print(f"🔍 正在进行官方原生语音转录 (语言模式: {lang_desc}，支持长篇滑动窗口)...")

        # 官方原生推理：自动处理音频重采样、长音频分段、VAD切片与中英文混合
        result = model.transcribe(
            str(wav_path),
            language=lang_code,
            task="transcribe",
            fp16=(device == "cuda"),
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False, # 防止长音频长篇研讨出现重复循环
            temperature=0.0,
        )

        segments = result.get("segments", [])
        if len(segments) > 1:
            print("\n📜 [长音频学术时间轴详细分段]:")
            for seg in segments:
                start_m, start_s = divmod(int(seg.get("start", 0)), 60)
                end_m, end_s = divmod(int(seg.get("end", 0)), 60)
                seg_text = seg.get("text", "").strip()
                if seg_text:
                    print(f"   [{start_m:02d}:{start_s:02d} -> {end_m:02d}:{end_s:02d}] {seg_text}")

        text = result.get("text", "").strip()
        print(f"\n📋 OpenAI 官方原生完整识别结果:\n'{text}'")
        if text:
            return text
    except ImportError:
        print("\n💡 [提示] 未安装 openai-whisper 原生库，可通过以下命令一键安装:")
        print("   pip install -U openai-whisper\n")
    except Exception as e:
        print(f"💡 OpenAI 官方原生库运行提示: {e}")

    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="语音输入与语音输出整合测试脚本 (支持长篇视频研读、在线会议与麦克风长音频)")
    parser.add_argument("--file", type=Path, default=None, help="指定已有视频/音频文件直接识别 (如 .mp4, .mkv, .wav, .mp3)")
    parser.add_argument("--mode", type=str, default="system", choices=["mic", "system", "meeting", "video"], help="音频来源: system/meeting/video(视频/会议系统声音内录, 默认), mic(麦克风)")
    parser.add_argument("--seconds", type=int, default=0, help="录音时长 (秒，默认 0 表示无限时长随时按回车停止，亦可指定如 30、60)")
    parser.add_argument("--language", type=str, default="auto", choices=["auto", "zh", "en"], help="识别语言模式: auto(自适应中英文), zh(强制中文), en(强制英文)")
    parser.add_argument("--playback", action="store_true", help="是否在识别前完整回放刚才录入的声音")
    parser.add_argument("--model", type=Path, default=MODEL_DIR, help="本地 Whisper 模型路径")
    parser.add_argument("--skip-beep", action="store_true", help="跳过发声测试提示音")
    args = parser.parse_args()

    # 如果直接指定了 --file，跳过录音和蜂鸣音，直接识别指定音频/视频
    if args.file:
        target_file = Path(args.file)
        if not target_file.is_absolute():
            target_file = (BASE_DIR / target_file).resolve()
        print("=" * 60)
        print(f"      🎙️  Whisper 长篇音视频文件直接识别: {target_file.name}")
        print("=" * 60)
        text = transcribe_audio(target_file, args.model, language=args.language)
        print("\n" + "=" * 60)
        print("🎉 测试结果汇总:")
        print(f"📝 识别文字: {text if text else '[未识别到有效文字]'}")
        print("=" * 60)
        return 0

    mode_name = "🎬 视频/会议声音(系统扬声器内录)" if args.mode in ("system", "meeting", "video") else "🎤 麦克风人声"
    print("=" * 60)
    print(f"      🎙️  长语音测试 ({mode_name}) & 语音识别")
    print("=" * 60)

    # 1. 语音输出测试 (仅麦克风模式且未禁用时播放提示音，内录模式跳过避免污染视频声音)
    if not args.skip_beep and args.mode == "mic":
        print("\n--- [第 1 步] 🔊 语音输出测试（播放提示音） ---")
        print("正在播放一段测试提示音，请确认扬声器/耳机是否能听到声音...")
        generate_beep_wav(BEEP_WAV, duration_sec=0.8, freq=523.25)
        play_audio(BEEP_WAV)
        print("✅ 提示音播放完毕。")

    # 2. 语音输入测试 (长录制，随时按回车结束)
    print(f"\n--- [第 2 步] 🎙️ 语音输入录制 ({mode_name}) ---")
    if not record_audio(RECORDED_WAV, seconds=args.seconds, mode=args.mode):
        print(f"❌ 录音失败，请检查音源权限与连接。")
        return 1

    # 3. 录音回放测试 (仅当指定 --playback 时回放，长篇录音默认不强制回放避免浪费时间)
    if args.playback:
        print("\n--- [第 3 步] 🔊 录音回放测试 ---")
        print("正在通过扬声器回放刚才录入的声音...")
        play_audio(RECORDED_WAV)
        print("✅ 回放完毕。")

    # 4. Whisper 语音识别测试 (STT Inference)
    print("\n--- [第 4 步] 🧠 Whisper-large-v3 长音频转文字识别 ---")
    text = transcribe_audio(RECORDED_WAV, args.model, language=args.language)

    print("\n" + "=" * 60)
    print("🎉 测试结果汇总:")
    print(f"📝 完整识别文字: {text if text else '[未识别到有效文字]'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
