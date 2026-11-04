"""Self-contained STT verification script for Whisper-large-v3."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

BASE_DIR = Path(__file__).resolve().parent
WAV_PATH = BASE_DIR / "voice_test.wav"
MODEL_DIR = Path("/home/cqf/文档/models/whisper-large-v3")


def run_transcription():
    if not WAV_PATH.is_file():
        print(f"[错误] 找不到录音文件: {WAV_PATH}")
        return

    print("=" * 60)
    print(f"🎙️  Whisper-large-v3 离线测试: {WAV_PATH.name}")
    print("=" * 60)

    # 1. 读取音频并重采样至 16000Hz 单声道
    waveform, sr = torchaudio.load(str(WAV_PATH))
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    audio_data = waveform.squeeze().numpy()

    # 峰值归一化 (防止弱音量底噪被截断)
    max_amp = float(np.max(np.abs(audio_data)))
    if max_amp > 0.001:
        audio_data = (audio_data / max_amp) * 0.95

    duration = len(audio_data) / 16000.0
    print(f"📊 音频时长: {duration:.2f} 秒, 峰值增益后最大振幅: {np.max(np.abs(audio_data)):.2f}")

    # 2. 加载模型与处理器
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32
    device = "cuda:0" if use_cuda else "cpu"
    print(f"⚡ 计算设备: {device} ({'GPU CUDA' if use_cuda else 'CPU'})")

    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(MODEL_DIR),
        dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)

    # 3. 提取特征
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=dtype)

    # 4. 显式锁定中文转写 Prompt，并加入防死循环惩罚 (repetition_penalty & no_repeat_ngram)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")

    print("🔍 正在执行转录（破除循环，精准捕获语音内容）...")
    with torch.inference_mode():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            max_new_tokens=128,
        )

    decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    text = (decoded[0] if decoded else "").strip()

    print("\n" + "=" * 60)
    print("🎉 最终识别结果:")
    print(f"📝 识别文字: {text if text else '[未识别到有效文字]'}")
    print("=" * 60)


if __name__ == "__main__":
    run_transcription()
