"""Run a module or script inside .venv-gpu with the CUDA 12 / cuDNN 9 DLLs
from the pip `nvidia-*` wheels on the DLL search path (the RTX 6000 Ada
workstation has no torch and no CUDA toolkit; the old MX250 laptop borrowed
these DLLs from a torch wheel instead - see gpu_run.py).

    .venv-gpu\\Scripts\\python handoff\\gpu_tools\\gpu_env.py [--ort-cuda] -m eval.run_quality ...
    .venv-gpu\\Scripts\\python handoff\\gpu_tools\\gpu_env.py [--ort-cuda] path\\to\\script.py args...

--ort-cuda sets STREAMING_RAG_ORT_PROVIDER=cuda (the RAG models' opt-in
switch); without it only Whisper/CTranslate2 can use the GPU.
"""
import os
import runpy
import sys
from pathlib import Path


def add_cuda_dlls() -> list[str]:
    import nvidia
    dirs = []
    for root in nvidia.__path__:
        for b in sorted(Path(root).glob("*/bin")):
            os.add_dll_directory(str(b))
            dirs.append(str(b))
    os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])  # CTranslate2 uses LoadLibrary/PATH
    return dirs


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--ort-cuda":
        os.environ["STREAMING_RAG_ORT_PROVIDER"] = "cuda"
        args = args[1:]
    add_cuda_dlls()
    sys.path.insert(0, os.getcwd())
    if args[0] == "-m":
        sys.argv = [args[1]] + args[2:]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = args
        sys.path.insert(0, str(Path(args[0]).resolve().parent))
        runpy.run_path(args[0], run_name="__main__")


if __name__ == "__main__":
    main()
