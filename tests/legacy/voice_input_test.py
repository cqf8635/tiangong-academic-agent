"""Minimal local microphone -> Whisper transcription test."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Record microphone audio and transcribe it locally.")
    parser.add_argument("--seconds", type=float, default=6.0, help="Recording duration (default: 6 seconds)")
    parser.add_argument("--model", default="large-v3", help="Whisper model name or local path")
    parser.add_argument("--output", default="voice_test.wav", help="Temporary WAV output path")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds 必须大于 0")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        print(f"缺少依赖: {exc.name}")
        print("请在 torchStd 环境执行: python -m pip install faster-whisper")
        return 2

    sample_rate = 16_000
    arecord = shutil.which("arecord")
    if not arecord:
        print("找不到 arecord，无法录音。")
        return 1

    wav_path = Path(args.output).resolve()
    print(f"将在 {args.seconds:g} 秒内录音，请现在讲话...")
    command = [
        arecord,
        "-q",
        "-f", "S16_LE",
        "-r", str(sample_rate),
        "-c", "1",
        "-d", str(round(args.seconds)),
        str(wav_path),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"录音失败（arecord 返回 {exc.returncode}）。")
        print("可以先执行 `arecord -l` 查看麦克风设备。")
        return 1

    print(f"音频已保存: {wav_path}")
    print(f"正在加载模型 {args.model} ...")

    try:
        model = WhisperModel(args.model, device="cuda", compute_type="int8_float16")
    except Exception as exc:
        print(f"CUDA 模式加载失败: {exc}")
        print("改用 CPU int8 模式重试...")
        model = WhisperModel(args.model, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        str(wav_path),
        language="zh",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True,
    )
    text = "".join(segment.text for segment in segments).strip()
    print(f"\n识别语言: {info.language} (概率 {info.language_probability:.2f})")
    print(f"识别结果: {text or '[没有识别到语音]'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
