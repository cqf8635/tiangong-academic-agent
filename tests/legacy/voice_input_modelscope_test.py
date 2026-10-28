"""Record microphone audio and transcribe it with a local ModelScope Whisper model."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

from config_utils import resolve_whisper_path

DEFAULT_MODEL = resolve_whisper_path(BASE_DIR)


def main() -> int:
    parser = argparse.ArgumentParser(description="Local ModelScope Whisper microphone test")
    parser.add_argument("--seconds", type=int, default=6, help="Recording duration in seconds")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Local ModelScope model directory")
    parser.add_argument("--output", type=Path, default=Path("voice_test.wav"), help="Recorded WAV path")
    args = parser.parse_args()

    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")
    if not (args.model / "config.json").is_file() or not (args.model / "model.safetensors").is_file():
        print(f"模型尚未下载完成: {args.model.resolve()}")
        print(
            "继续下载: modelscope download AI-ModelScope/whisper-large-v3 "
            "--local-dir models/whisper-large-v3 "
            "--include '*.json' '*.txt' 'model.safetensors'"
        )
        return 2

    arecord = shutil.which("arecord")
    if not arecord:
        print("找不到 arecord，无法访问麦克风。")
        return 1

    output_path = args.output.resolve()
    print(f"将在 {args.seconds} 秒内录音，请现在讲话...")
    record = [arecord, "-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(args.seconds), str(output_path)]
    try:
        subprocess.run(record, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"录音失败（arecord 返回 {exc.returncode}）。")
        return 1

    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    except ImportError as exc:
        print(f"缺少依赖: {exc.name}")
        return 2

    use_cuda = torch.cuda.is_available()
    device = 0 if use_cuda else -1
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"加载模型: {args.model.resolve()} ({'CUDA' if use_cuda else 'CPU'})")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    if use_cuda:
        model.to("cuda:0")
    processor = AutoProcessor.from_pretrained(args.model)
    transcribe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )
    result = transcribe(str(output_path), generate_kwargs={"language": "zh", "task": "transcribe"})
    print(f"识别结果: {result['text'].strip() or '[没有识别到语音]'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
