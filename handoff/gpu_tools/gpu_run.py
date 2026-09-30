"""Run an eval module with the isolated onnxruntime-gpu build (CUDA provider).

    python gpu_run.py eval.run_quality --scenarios ... [args]

Production environment untouched: the GPU build lives in iso/pylib_gpu and
the CUDA 12 DLLs are borrowed from the installed torch wheel.
"""
import os
import runpy
import sys
from pathlib import Path

import torch

os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
sys.path.insert(0, str(Path(__file__).parent / "pylib_gpu"))
os.environ["STREAMING_RAG_ORT_PROVIDER"] = "cuda"
import onnxruntime  # noqa: E402
print("onnxruntime", onnxruntime.__version__, onnxruntime.get_available_providers(), flush=True)
module = sys.argv[1]
sys.argv = [module] + sys.argv[2:]
runpy.run_module(module, run_name="__main__", alter_sys=True)
