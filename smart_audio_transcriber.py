"""Multi-source audio capture and transcription tool.

Supports:
1. System playback / Video audio capture (Monitor loopback).
2. Microphone voice capture.
3. Mixed capture (Video sound + Microphone).
4. Direct video/audio file input.

Uses local Whisper model: models/whisper-large-v3
"""

from __future__ import annotations

import argparse
import subprocess
import shutil
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from config_utils import get_default_monitor_source, resolve_whisper_path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

MODEL_DIR = resolve_whisper_path(BASE_DIR)
DEFAULT_OUTPUT_WAV = BASE_DIR / "captured_audio.wav"


def record_stream(
    mode: str = "system",
    duration_sec: int = 10,
    output_file: Path = DEFAULT_OUTPUT_WAV
) -> bool:
    """Record audio from specified stream using ffmpeg pulse capture."""
    monitor_source = get_default_monitor_source()
    mic_source = "default"

    if mode == "system":
        print(f"🎬 [正在捕获视频播放声音 (扬声器内录)] 时长: {duration_sec} 秒...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "pulse",
            "-i", monitor_source,
            "-t", str(duration_sec),
            "-ac", "1",
            "-ar", "16000",
            str(output_file),
        ]
    elif mode == "mic":
        print(f"🎤 [正在录制麦克风交流] 请对准麦克风说话，时长: {duration_sec} 秒...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "pulse",
            "-i", mic_source,
            "-t", str(duration_sec),
            "-ac", "1",
            "-ar", "16000",
            str(output_file),
        ]
    elif mode == "mixed":
        print(f"🎧 [正在混音录制 (视频声音 + 麦克风说话)] 时长: {duration_sec} 秒...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "pulse", "-i", monitor_source,
            "-f", "pulse", "-i", mic_source,
            "-filter_complex", "amix=inputs=2:duration=first",
            "-t", str(duration_sec),
            "-ac", "1",
            "-ar", "16000",
            str(output_file),
        ]
    else:
        print(f"[错误] 不支持的模式: {mode}")
        return False

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"✅ 录制完成，音频已保存至: {output_file}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[错误] 录制失败: {exc}")
        return False


def extract_audio_from_file(media_file: Path, output_file: Path = DEFAULT_OUTPUT_WAV) -> bool:
    """Extract audio from local video/audio file."""
    print(f"📁 [正在解析媒体文件音轨] 文件: {media_file.name} ...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(media_file),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_file),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"✅ 音频提取完成: {output_file}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[错误] 提取文件音轨失败: {exc}")
        return False


def load_whisper_pipeline(model_dir: Path = MODEL_DIR):
    """Load local Whisper large-v3 model using OpenAI official whisper library."""
    import torch
    import whisper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    download_root = str(model_dir.parent) if model_dir.parent.exists() else None

    print(f"🧠 加载本地 Whisper 模型: {model_dir.name} (计算设备: {device.upper()})...")
    model = whisper.load_model("large-v3", device="cpu", download_root=download_root)
    if device == "cuda":
        model = model.half().to("cuda")
        torch.cuda.empty_cache()
    return model


def transcribe(model, wav_file: Path = DEFAULT_OUTPUT_WAV, lang: str = "auto") -> str:
    """Run transcription supporting both short and arbitrary long audio (>30s)."""
    import torch

    lang_code = "zh" if lang in ("zh", "chinese") else (None if lang == "auto" else lang)
    lang_desc = "中文 / 中英混合" if lang_code == "zh" else (lang_code or "auto (自适应识别)")
    print(f"🎙️ 正在进行语音识别转文本 (语言模式: {lang_desc})...")

    use_cuda = torch.cuda.is_available()
    result = model.transcribe(
        str(wav_file),
        language=lang_code,
        task="transcribe",
        fp16=use_cuda,
    )
    return result.get("text", "").strip()


def main():
    parser = argparse.ArgumentParser(description="多音源语音捕获与 Whisper 本地高精度识别工具")
    parser.add_argument(
        "--mode",
        choices=["system", "mic", "mixed"],
        default="system",
        help="录音音源: system(播放的视频声音), mic(麦克风人声), mixed(双通道混音)",
    )
    parser.add_argument("--seconds", type=int, default=10, help="录制时长（秒，默认10秒）")
    parser.add_argument("--file", type=str, default=None, help="可选：直接传入本地已有音视频文件进行识别")
    parser.add_argument(
        "--lang",
        choices=["auto", "zh", "en"],
        default="auto",
        help="语种模式: auto(自动检测中英混说), zh(强制中文简体), en(强制英文模式，彻底杜绝音译乱码)",
    )
    args = parser.parse_args()

    # 1. 准备音频
    if args.file:
        media_path = Path(args.file)
        if not media_path.exists():
            print(f"[错误] 指定的文件不存在: {media_path}")
            sys.exit(1)
        ok = extract_audio_from_file(media_path, DEFAULT_OUTPUT_WAV)
    else:
        ok = record_stream(mode=args.mode, duration_sec=args.seconds, output_file=DEFAULT_OUTPUT_WAV)

    if not ok:
        sys.exit(1)

    # 2. 加载模型并识别
    pipe = load_whisper_pipeline(MODEL_DIR)
    text = transcribe(pipe, DEFAULT_OUTPUT_WAV, lang=args.lang)

    print("\n" + "=" * 40)
    print("🎯 【识别转文本结果】:")
    print(text.strip())
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
