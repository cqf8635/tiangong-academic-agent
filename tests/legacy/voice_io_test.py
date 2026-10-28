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

from config_utils import resolve_whisper_path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

MODEL_DIR = resolve_whisper_path(BASE_DIR)
RECORDED_WAV = Path("voice_test.wav")
BEEP_WAV = Path("beep_test.wav")


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
    """Record audio from microphone."""
    arecord = shutil.which("arecord")
    if arecord:
        cmd = [
            arecord,
            "-q",
            "-f", "S16_LE",
            "-r", str(sample_rate),
            "-c", "1",
            "-d", str(seconds),
            str(wav_path),
        ]
        try:
            print(f"🎤 [正在录音...] 请对准麦克风说话（倒计时 {seconds} 秒）:")
            subprocess.run(cmd, check=True)
            print("✅ 录音完成！")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[错误] arecord 录音失败: {exc}")
            return False

    # Fallback to sounddevice
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
        print(f"[错误] 录音失败: {e}")
        return False


def transcribe_audio(wav_path: Path, model_dir: Path) -> str:
    """Transcribe audio using local Whisper model."""
    if not (model_dir / "config.json").is_file() or not (model_dir / "model.safetensors").is_file():
        print(f"[错误] Whisper 模型文件不完整: {model_dir}")
        return ""

    print(f"\n🧠 正在加载本地 Whisper 模型: {model_dir.name} ...")
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as e:
        print(f"[错误] 缺少 transformers / torch 依赖: {e}")
        return ""

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32

    print(f"⚡ 计算设备: {'NVIDIA CUDA (GPU: ' + torch.cuda.get_device_name(0) + ')' if use_cuda else 'CPU'}")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    if use_cuda:
        model.to("cuda:0")

    processor = AutoProcessor.from_pretrained(model_dir)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )

    print("🔍 正在进行语音识别...")
    result = pipe(
        str(wav_path),
        generate_kwargs={"language": "zh", "task": "transcribe"},
    )
    return result.get("text", "").strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="语音输入与语音输出整合测试脚本")
    parser.add_argument("--seconds", type=int, default=5, help="录音时长（默认: 5秒）")
    parser.add_argument("--model", type=Path, default=MODEL_DIR, help="本地 Whisper 模型路径")
    parser.add_argument("--skip-beep", action="store_true", help="跳过发声测试提示音")
    args = parser.parse_args()

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
    text = transcribe_audio(RECORDED_WAV, args.model)

    print("\n" + "=" * 60)
    print("🎉 测试结果汇总:")
    print(f"📝 识别文字: {text if text else '[未识别到有效文字]'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
