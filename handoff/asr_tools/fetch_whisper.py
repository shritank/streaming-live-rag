"""Fetch the pinned faster-whisper models into asr_tools/whisper/<name>/ and
verify every file against the Hub's LFS SHA-256 at that revision.

    python fetch_whisper.py tiny.en base.en small.en
"""
import hashlib
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ISO = Path(__file__).parent


def main() -> int:
    api = HfApi()
    for name in sys.argv[1:]:
        rev = (ISO / f"whisper_{name}_REVISION").read_text().strip()
        repo = f"Systran/faster-whisper-{name}"
        src = Path(snapshot_download(repo, revision=rev))
        dst = ISO / "whisper" / name
        dst.mkdir(parents=True, exist_ok=True)
        info = api.model_info(repo, revision=rev, files_metadata=True)
        for s in info.siblings:
            f = src / s.rfilename
            shutil.copy2(f, dst / s.rfilename)
            if s.lfs:
                h = hashlib.sha256(f.read_bytes()).hexdigest()
                assert h == s.lfs.sha256, f"{repo} {s.rfilename}: {h} != {s.lfs.sha256}"
        (dst / "REVISION").write_text(rev + "\n")
        print(f"{repo}@{rev[:8]} -> {dst} (verified)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
