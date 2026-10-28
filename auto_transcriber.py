"""Real-time Streaming & Auto-VAD Transcriber using local Whisper-large-v3.

Features:
1. Auto Voice Activity Detection (VAD): Automatically detects when speech starts/stops.
2. Real-time Live Streaming Output: Prints words in real-time as speech progresses.
3. Multi-source: Supports both Microphone and System/Video Playback loopback.
4. Auto-language detection: Seamlessly handles Chinese, English, and mixed speech.
"""

from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from config_utils import get_default_monitor_source, resolve_whisper_path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

MODEL_DIR = resolve_whisper_path(BASE_DIR)
SPEAKER_MONITOR = get_default_monitor_source()
TEMP_WAV = BASE_DIR / "auto_temp.wav"


def load_whisper_pipeline(model_dir: Path = MODEL_DIR):
    """Load local Whisper large-v3 model."""
    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32

    print(f"🧠 正在加载本地 Whisper 模型: {model_dir.name} (计算设备: {'NVIDIA CUDA (GPU)' if use_cuda else 'CPU'})...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        dtype=dtype,
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
        dtype=dtype,
        device=device,
    )
    return pipe


def calculate_rms(chunk_bytes: bytes) -> float:
    """Calculate volume energy (RMS) of audio chunk."""
    count = len(chunk_bytes) // 2
    if count == 0:
        return 0.0
    shorts = struct.unpack(f"<{count}h", chunk_bytes)
    sum_squares = sum(s * s for s in shorts)
    return math.sqrt(sum_squares / count)


def live_stream_transcribe(
    pipe,
    mode: str = "mic",
    threshold: float = 600.0,
    silence_timeout: float = 1.4,
    stream_interval: float = 0.6,
):
    """
    Live stream VAD transcription loop with real-time token rendering.
    - threshold: RMS energy threshold to trigger speech detection
    - silence_timeout: seconds of silence to finalize utterance
    - stream_interval: seconds between live partial transcript updates
    """
    source = get_default_monitor_source() if mode == "system" else "default"
    rate = 16000
    chunk_duration = 0.1  # 100ms
    chunk_size = int(rate * 2 * chunk_duration)
    silence_chunks_needed = int(silence_timeout / chunk_duration)

    print("\n" + "=" * 55)
    print(f"🟢 【实时流式语音识别已就绪】")
    print(f"👉 监听通道: {'🎬 电脑视频声音 (扬声器内录)' if mode == 'system' else '🎤 麦克风人声'}")
    print(f"💡 特性: 自动检测说话、实时字幕流式上屏、停顿自动断句落盘")
    print("👉 按 Ctrl + C 可安全退出")
    print("=" * 55 + "\n")

    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        "-f", "pulse", "-i", source,
        "-f", "s16le", "-ar", str(rate), "-ac", "1",
        "-",
    ]

    import numpy as np
    import wave

    while True:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        audio_buffer = bytearray()
        is_speaking = False
        silence_counter = 0
        last_stream_time = time.time()
        current_display_text = ""

        try:
            while True:
                raw_chunk = proc.stdout.read(chunk_size)
                if not raw_chunk:
                    break

                rms = calculate_rms(raw_chunk)
                now = time.time()

                if rms > threshold:
                    if not is_speaking:
                        is_speaking = True
                        current_display_text = ""
                        sys.stdout.write("🎙️  [实时聆听中] ")
                        sys.stdout.flush()

                    audio_buffer.extend(raw_chunk)
                    silence_counter = 0

                    # 达到流式刷新间隔且积累了足够音频时，进行轻量级流式预览
                    if (now - last_stream_time >= stream_interval) and (len(audio_buffer) >= rate * 2 * 0.8):
                        last_stream_time = now
                        # 将当前 buffer 转为 float32 numpy 格式喂给 pipeline
                        samples = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        try:
                            partial_res = pipe(samples, generate_kwargs={"task": "transcribe"})
                            partial_text = partial_res.get("text", "").strip()
                            if partial_text:
                                current_display_text = partial_text
                                sys.stdout.write(f"\r🎙️  [实时字幕] {current_display_text}   ")
                                sys.stdout.flush()
                        except Exception:
                            pass

                elif is_speaking:
                    audio_buffer.extend(raw_chunk)
                    silence_counter += 1

                    # 连续静音判定为该句结束
                    if silence_counter >= silence_chunks_needed:
                        break

            proc.terminate()
            proc.wait()

            # 最终完整决议并打印确定句
            if len(audio_buffer) >= rate * 2 * 0.6:
                samples = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                try:
                    final_res = pipe(samples, generate_kwargs={"task": "transcribe"}, return_timestamps=True)
                    final_text = final_res.get("text", "").strip()
                    if final_text:
                        # 覆盖当前行并输出最终确定句子
                        sys.stdout.write(f"\r✅ 【确定识别】: {final_text}\n\n")
                        sys.stdout.flush()
                except Exception as e:
                    sys.stdout.write(f"\r[转录错误]: {e}\n\n")
                    sys.stdout.flush()
            else:
                # 杂音或时长过短，清除行
                if is_speaking:
                    sys.stdout.write("\r" + " " * 60 + "\r")
                    sys.stdout.flush()

        except KeyboardInterrupt:
            proc.terminate()
            print("\n\n👋 实时监听已安全停止。")
            break


def main():
    parser = argparse.ArgumentParser(description="实时流式语音识别与自动 VAD 工具")
    parser.add_argument(
        "--mode",
        choices=["mic", "system"],
        default="mic",
        help="输入源: mic(麦克风人声), system(播放的视频声音内录)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=600.0,
        help="声音灵敏度阈值 (默认600，噪音环境下可调高到 800-1000)",
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=1.3,
        help="停顿切句等待时间 (秒，默认1.3秒自动断句出最终文本)",
    )
    args = parser.parse_args()

    pipe = load_whisper_pipeline(MODEL_DIR)
    live_stream_transcribe(
        pipe,
        mode=args.mode,
        threshold=args.threshold,
        silence_timeout=args.silence,
    )


if __name__ == "__main__":
    main()
