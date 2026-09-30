"""GPU selection must never silently fall back to CPU (streaming_rag/gpu.py)."""
import onnxruntime as ort
import pytest

from streaming_rag import gpu


class _Session:
    def __init__(self, providers):
        self._p = providers

    def get_providers(self):
        return self._p


def test_forced_cpu_never_uses_cuda(monkeypatch):
    monkeypatch.setenv("STREAMING_RAG_ORT_PROVIDER", "cpu")
    assert gpu.ort_use_cuda() is False


def test_invalid_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("STREAMING_RAG_ORT_PROVIDER", "gpu0")
    with pytest.raises(ValueError):
        gpu.ort_use_cuda()


def test_required_cuda_without_cuda_build_raises(monkeypatch):
    monkeypatch.setenv("STREAMING_RAG_ORT_PROVIDER", "cuda")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    with pytest.raises(gpu.GpuUnavailable):
        gpu.ort_use_cuda()


def test_auto_without_any_nvidia_gpu_uses_cpu(monkeypatch):
    monkeypatch.setenv("STREAMING_RAG_ORT_PROVIDER", "auto")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(gpu, "nvidia_gpu_present", lambda: False)
    assert gpu.ort_use_cuda() is False


def test_auto_with_nvidia_gpu_but_cpu_only_runtime_raises(monkeypatch):
    """GPU present -> CPU neural inference is never a silent option."""
    monkeypatch.setenv("STREAMING_RAG_ORT_PROVIDER", "auto")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(gpu, "nvidia_gpu_present", lambda: True)
    with pytest.raises(gpu.GpuUnavailable):
        gpu.ort_use_cuda()


def test_auto_on_cuda_build_uses_cuda(monkeypatch):
    monkeypatch.setenv("STREAMING_RAG_ORT_PROVIDER", "auto")
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(gpu, "add_cuda_dlls", lambda: [])
    assert gpu.ort_use_cuda() is True


def test_session_that_fell_back_to_cpu_is_refused():
    with pytest.raises(gpu.GpuUnavailable):
        gpu.check_ort_session(_Session(["CPUExecutionProvider"]), "m.onnx")
    gpu.check_ort_session(_Session(["CUDAExecutionProvider", "CPUExecutionProvider"]), "m.onnx")


def test_asr_forced_cpu(monkeypatch):
    monkeypatch.setenv("STREAMING_RAG_ASR_DEVICE", "cpu")
    assert gpu.asr_device() == "cpu"


def test_spacy_survives_a_running_cuda_model():
    """Regression: with torch installed (GPU env), spaCy's first import after a CUDA
    model had run failed (cuDNN/cuBLAS build clash) and sentence splitting silently
    fell back to a regex splitter that segments differently. Fresh interpreter:
    import order is the thing under test."""
    import subprocess
    import sys
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        pytest.skip("needs onnxruntime-gpu")
    code = (
        "import os, sys; sys.path.insert(0, os.getcwd()); os.environ['STREAMING_RAG_ORT_PROVIDER'] = 'cuda'\n"
        "from streaming_rag.retrieval import neural, text\n"
        "neural.get_cross_encoder().score('q', ['a passage'])\n"
        "n = len(text.split_sentences('Dr. Smith moved to St. Louis in 1999. He left.'))\n"
        "print('RESULT', text._spacy_unavailable, n)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert "RESULT False 2" in out.stdout, out.stdout + out.stderr[-500:]


def test_missing_fused_graph_falls_back_to_the_original_graph_on_the_gpu_not_the_cpu(monkeypatch, tmp_path, caplog):
    import logging

    from streaming_rag.retrieval import neural
    monkeypatch.setattr(neural, "FUSED_DIR", tmp_path)                 # nothing built there
    monkeypatch.setattr(gpu, "ort_use_cuda", lambda: True)             # a CUDA machine
    monkeypatch.setenv("STREAMING_RAG_ORT_FUSED", "auto")
    with caplog.at_level(logging.WARNING, logger="streaming_rag.neural"):
        assert neural._onnx_path("ce", "original.onnx") == "original.onnx"
    assert "not built" in caplog.text
    monkeypatch.setenv("STREAMING_RAG_ORT_FUSED", "1")                 # explicitly required -> loud
    with pytest.raises(neural.ModelNotAvailable):
        neural._onnx_path("ce", "original.onnx")
    monkeypatch.setenv("STREAMING_RAG_ORT_FUSED", "0")                 # explicitly off
    assert neural._onnx_path("ce", "original.onnx") == "original.onnx"
