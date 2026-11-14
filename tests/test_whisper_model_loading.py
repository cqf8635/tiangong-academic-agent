import sys
from types import SimpleNamespace

import pytest
import torch

from voice_langgraph_agent import WhisperEngine


def test_native_whisper_loader_uses_configured_pt_checkpoint(monkeypatch, tmp_path):
    model_path = tmp_path / "large-v3.pt"
    model_path.write_bytes(b"checkpoint")
    captured = {}

    class FakeModel:
        def to(self, device):
            return self

    fake_whisper = SimpleNamespace(
        load_model=lambda name, **kwargs: (captured.update(name=name, kwargs=kwargs) or FakeModel())
    )
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    monkeypatch.setattr("voice_langgraph_agent.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(WhisperEngine, "_whisper_native_model", None)

    loaded = WhisperEngine.get_whisper_native_model(model_path)

    assert isinstance(loaded, FakeModel)
    assert captured["name"] == str(model_path)
    assert captured["kwargs"]["device"] == "cpu"


@pytest.mark.parametrize("out_of_memory", [False, True])
def test_native_whisper_prefers_cuda_and_recovers_from_actual_oom(
    monkeypatch, tmp_path, out_of_memory
):
    model_path = tmp_path / "large-v3.pt"
    model_path.write_bytes(b"checkpoint")
    calls = []

    class FakeModel:
        def half(self):
            calls.append("half")
            return self

        def to(self, device):
            calls.append(device)
            if out_of_memory:
                raise torch.cuda.OutOfMemoryError("test GPU is full")
            return self

        def cpu(self):
            calls.append("cpu")
            return self

        def float(self):
            calls.append("float")
            return self

    model = FakeModel()

    def load_model(name, **kwargs):
        assert kwargs["device"] == "cpu"
        calls.append("checkpoint")
        return model

    def reject_memory_estimate(*args, **kwargs):
        pytest.fail("CUDA loading must be attempted without a free-memory threshold")

    monkeypatch.setitem(sys.modules, "whisper", SimpleNamespace(load_model=load_model))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", reject_memory_estimate)
    monkeypatch.setattr(WhisperEngine, "_whisper_native_model", None)

    assert WhisperEngine.get_whisper_native_model(model_path) is model
    assert calls == ["checkpoint", "half", "cuda"] + (
        ["cpu", "float"] if out_of_memory else []
    )
    assert WhisperEngine.get_whisper_native_model(model_path) is model
    assert calls.count("checkpoint") == 1


def test_huggingface_directory_skips_native_checkpoint_loader(tmp_path, monkeypatch):
    snapshot = tmp_path / "whisper-large-v3"
    snapshot.mkdir()
    monkeypatch.setattr(WhisperEngine, "_whisper_native_model", None)

    assert WhisperEngine.get_whisper_native_model(snapshot) is None


def test_transformers_loader_rejects_single_pt_file(tmp_path):
    model_path = tmp_path / "large-v3.pt"
    model_path.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="single .pt file"):
        WhisperEngine.get_model_and_processor(model_path)
