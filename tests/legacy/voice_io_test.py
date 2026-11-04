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
import os
import shutil
import struct
import subprocess
import sys
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


def record_audio(wav_path: Path, seconds: int = 5, sample_rate: int = 16000) -> bool:
    """Record audio from microphone using PulseAudio/PipeWire default source."""
    # 1. 优先使用 ffmpeg 抓取 pulse/pipewire 系统默认麦克风（杜绝 arecord 声卡静音通道陷阱）
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y",
            "-f", "pulse",
            "-i", "default",
            "-t", str(seconds),
            "-ac", "1",
            "-ar", str(sample_rate),
            str(wav_path),
        ]
        try:
            print(f"🎤 [正在录音...] 请对准麦克风说话（倒计时 {seconds} 秒）:")
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print("✅ 录音完成！")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[警告] ffmpeg 录音异常: {exc}，尝试备用方式...")

    # 2. 备用 arecord (指定 pulse 通道)
    arecord = shutil.which("arecord")
    if arecord:
        for dev in ["pulse", "default"]:
            cmd = [
                arecord,
                "-q",
                "-D", dev,
                "-f", "S16_LE",
                "-r", str(sample_rate),
                "-c", "1",
                "-d", str(seconds),
                str(wav_path),
            ]
            try:
                print(f"🎤 [正在录音 (arecord -D {dev})...] 请对准麦克风说话（倒计时 {seconds} 秒）:")
                subprocess.run(cmd, check=True)
                print("✅ 录音完成！")
                return True
            except subprocess.CalledProcessError:
                continue

    # 3. 备用 sounddevice
    try:
        import sounddevice as sd
        import soundfile as sf
        print(f"🎤 [正在录音 (sounddevice)...] 请对准麦克风说话（倒计时 {seconds} 秒）:")
        recording = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()
        sf.write(str(wav_path), recording, sample_rate)
        print("✅ 录音完成！")
        return True
    except Exception as e:
        print(f"[错误] 麦克风录音失败: {e}")
        return False


def transcribe_audio(wav_path: Path, model_dir: Path, language: str = "auto") -> str:
    """Transcribe audio prioritizing OpenAI official whisper library (Solution 1)."""
    # -------------------------------------------------------------------------
    # 方案 1: OpenAI 官方原生库 (最强降温回退、内置重复压缩率检测、完美支持中英混合)
    # -------------------------------------------------------------------------
    try:
        import os
        import whisper
        import torch

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.empty_cache()

        # 动态解析模型下载存储位置 (优先保存在项目同级 models 目录，如 /home/cqf/文档/models，零硬编码)
        download_root = str(model_dir.parent) if model_dir.parent.exists() else None

        print("\n" + "=" * 60)
        print("💡 【执行方案 1】: 调用 OpenAI 官方原生 Whisper 引擎")
        print(f"📁 模型加载目录: {download_root or '~/.cache/whisper'}")
        print(f"⚡ 计算设备: {device.upper()} (GPU FP16 极致显存优化)")
        print("=" * 60)

        # 先在 CPU 完成初始化，随后安全迁移至 GPU，内部由 transcribe 自动托管 fp16
        model = whisper.load_model("large-v3", device="cpu", download_root=download_root)
        if device == "cuda":
            model = model.to("cuda")
            torch.cuda.empty_cache()

        lang_code = "zh" if language in ("zh", "chinese") else (None if language == "auto" else language)
        lang_desc = "中文 / 中英混合" if lang_code == "zh" else (lang_code or "auto (自动识别)")
        print(f"🔍 正在进行官方原生语音转录 (语言模式: {lang_desc})...")

        # 官方原生推理：自动处理音频重采样、VAD切片与中英文混合
        result = model.transcribe(
            str(wav_path),
            language=lang_code,
            task="transcribe",
            fp16=(device == "cuda"),
        )

        text = result.get("text", "").strip()
        print(f"📋 OpenAI 官方原生识别结果: '{text}'")
        if text:
            return text
    except ImportError:
        print("\n💡 [提示] 未安装 openai-whisper 原生库，可通过以下命令一键安装:")
        print("   pip install -U openai-whisper\n")
    except Exception as e:
        print(f"💡 OpenAI 官方原生库运行提示: {e}")

    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="语音输入与语音输出整合测试脚本")
    parser.add_argument("--file", type=Path, default=None, help="指定已有音频文件直接识别")
    parser.add_argument("--language", type=str, default="auto", choices=["auto", "zh", "en"], help="识别语言模式: auto(自适应中英文), zh(强制中文), en(强制英文)")
    parser.add_argument("--seconds", type=int, default=5, help="录音时长（默认: 5秒）")
    parser.add_argument("--model", type=Path, default=MODEL_DIR, help="本地 Whisper 模型路径")
    parser.add_argument("--language", type=str, default="auto", choices=["auto", "zh", "en"], help="识别语言模式: auto(自适应中英文), zh(强制中文), en(强制英文)")
    parser.add_argument("--skip-beep", action="store_true", help="跳过发声测试提示音")
    args = parser.parse_args()

    # 如果直接指定了 --file，跳过录音和蜂鸣音，直接识别指定音频
    if args.file:
        target_file = Path(args.file)
        if not target_file.is_absolute():
            target_file = (BASE_DIR / target_file).resolve()
        print("=" * 60)
        print(f"      🎙️  Whisper 语音文件直接识别: {target_file.name}")
        print("=" * 60)
        text = transcribe_audio(target_file, args.model, language=args.language)
        print("\n" + "=" * 60)
        print("🎉 测试结果汇总:")
        print(f"📝 识别文字: {text if text else '[未识别到有效文字]'}")
        print("=" * 60)
        return 0

    print("=" * 60)
    print("      🎙️  语音输入 & 🔊 语音输出 综合测试")
    print("=" * 60)

    # 1. 语音输出测试 (Speaker / Audio Output Test)
    if not args.skip_beep:
        print("\n--- [第 1 步] 🔊 语音输出测试（播放提示音） ---")
        print("正在播放一段测试提示音，请确认扬声器/耳机是否能听到声音...")
        generate_beep_wav(BEEP_WAV, duration_sec=0.8, freq=523.25)
        play_audio(BEEP_WAV)
        print("✅ 提示音播放完毕。")

    # 2. 语音输入测试 (Microphone Recording)
    print("\n--- [第 2 步] 🎙️ 语音输入测试（麦克风录音） ---")
    if not record_audio(RECORDED_WAV, seconds=args.seconds):
        print("❌ 麦克风录音失败，请检查麦克风权限与设备连接。")
        return 1

    # 3. 录音回放测试 (Play Back Recorded Voice)
    print("\n--- [第 3 步] 🔊 录音回放测试 ---")
    print("正在通过扬声器回放刚才录入的声音...")
    play_audio(RECORDED_WAV)
    print("✅ 回放完毕。")

    # 4. Whisper 语音识别测试 (STT Inference)
    print("\n--- [第 4 步] 🧠 Whisper-large-v3 语音转文字识别 ---")
    text = transcribe_audio(RECORDED_WAV, args.model, language=args.language)

    print("\n" + "=" * 60)
    print("🎉 测试结果汇总:")
    print(f"📝 识别文字: {text if text else '[未识别到有效文字]'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
