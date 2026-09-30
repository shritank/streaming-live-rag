"""GPU selection shared by the ONNX Runtime models (retrieval/neural.py) and
the Whisper front end (asr/whisper.py).

STREAMING_RAG_ORT_PROVIDER / STREAMING_RAG_ASR_DEVICE:
  auto (default)  CUDA whenever the machine has an NVIDIA GPU - mandatory
                  then: if the installed runtime cannot use it, raise
                  GpuUnavailable instead of running neural inference on the
                  CPU. Only a machine without any NVIDIA GPU (e.g. the CPU
                  Docker image of requirements.lock) runs on the CPU.
  cuda            require the GPU; raise if it cannot be used.
  cpu             force CPU (explicit opt-out only; the reference
                  configuration of the earlier CPU reports).

Never a silent fallback: once CUDA is chosen, a session that does not bind
the CUDA provider raises GpuUnavailable instead of quietly running on CPU.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

log = logging.getLogger("streaming_rag.gpu")
_dlls_added = False


class GpuUnavailable(RuntimeError):
    pass


def nvidia_gpu_present() -> bool:
    """An NVIDIA driver is installed (nvidia-smi ships with it)."""
    return shutil.which("nvidia-smi") is not None


def _import_spacy_before_cuda() -> None:
    """spaCy (sentence splitting) imports thinc, which imports torch whenever torch
    is installed (it is, in the GPU environment). Torch and ONNX Runtime ship
    different builds of cuDNN/cuBLAS: once a CUDA model has loaded its copy,
    torch's import fails (0xc0000139) and spaCy - with the sentence boundaries
    every claim depends on - silently falls back to a regex splitter ("Dr. Smith
    ... St. Louis" -> 4 sentences instead of 3). Import it first, before any CUDA
    library is loaded (tests/test_gpu.py::test_spacy_survives_a_running_cuda_model)."""
    from .retrieval.text import _get_spacy_sentencizer
    _get_spacy_sentencizer()


def add_cuda_dlls() -> list[str]:
    """Windows: put the CUDA 12 / cuDNN 9 runtime DLLs on the search path,
    from the pip `nvidia-*` wheels or, failing that, a CUDA torch wheel."""
    global _dlls_added
    _import_spacy_before_cuda()
    if _dlls_added or sys.platform != "win32":
        return []
    dirs: list[str] = []
    try:
        import nvidia
        dirs = [str(b) for root in nvidia.__path__ for b in sorted(Path(root).glob("*/bin"))]
    except ImportError:
        try:
            import torch
            dirs = [str(Path(torch.__file__).parent / "lib")]
        except ImportError:
            pass
    for d in dirs:
        os.add_dll_directory(d)
    if dirs:  # CTranslate2 resolves cuBLAS/cuDNN through PATH
        os.environ["PATH"] = os.pathsep.join(dirs + [os.environ["PATH"]])  # OS variable, always set on Windows
    _dlls_added = True
    return dirs


def _mode(var: str) -> str:
    mode = os.environ.get(var, "auto").strip().lower()
    if mode not in ("auto", "cuda", "cpu"):
        raise ValueError(f"{var}={mode!r}: expected auto, cuda or cpu")
    return mode


def ort_use_cuda() -> bool:
    mode = _mode("STREAMING_RAG_ORT_PROVIDER")
    if mode == "cpu":
        return False
    import onnxruntime as ort
    if "CUDAExecutionProvider" in ort.get_available_providers():
        add_cuda_dlls()
        return True
    if mode == "cuda" or nvidia_gpu_present():
        raise GpuUnavailable("an NVIDIA GPU is required for neural inference but this onnxruntime build has no "
                             "CUDAExecutionProvider (install onnxruntime-gpu; STREAMING_RAG_ORT_PROVIDER=cpu "
                             "is the only explicit CPU opt-out)")
    return False


def check_ort_session(session, path: str) -> None:
    bound = session.get_providers()
    if not bound or bound[0] != "CUDAExecutionProvider":
        raise GpuUnavailable(f"{path}: CUDA requested but the session bound {bound} "
                             "(CUDA/cuDNN libraries missing?) - refusing to fall back to CPU")


def asr_device() -> str:
    mode = _mode("STREAMING_RAG_ASR_DEVICE")
    if mode == "cpu":
        return "cpu"
    add_cuda_dlls()
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
    except Exception:
        n = 0
    if n > 0:
        return "cuda"
    if mode == "cuda" or nvidia_gpu_present():
        raise GpuUnavailable("an NVIDIA GPU is required for Whisper but CTranslate2 sees no CUDA device "
                             "(STREAMING_RAG_ASR_DEVICE=cpu is the only explicit CPU opt-out)")
    return "cpu"
