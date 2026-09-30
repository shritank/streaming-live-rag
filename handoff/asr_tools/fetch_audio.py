"""Download HeySQuAD validation audio for the dev and diagnostic scenario
questions only. test2 (sealed) ids are refused outright.

    python fetch_audio.py REPO_ROOT OUT_DIR
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

REVISION = "2a9aff6032a76185b98eff703144a2a896313e49"
API = ("https://datasets-server.huggingface.co/rows?dataset=yijingwu/HeySQuAD_human"
       "&config=default&split=validation&offset={off}&length=100")


def ids_of(repo: Path, split: str) -> set[str]:
    return {p.stem.split("_", 2)[2] for p in (repo / "eval" / f"scenarios_heysquad_{split}_clean").glob("*.json")}


def get(url: str, tries: int = 6) -> bytes:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:  # network here is flaky; back off and retry
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))
            print("retry", i + 1, type(e).__name__, flush=True)


def main() -> int:
    repo, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    wanted = ids_of(repo, "dev") | ids_of(repo, "diag")
    sealed = ids_of(repo, "test2")
    assert not (wanted & sealed), "dev/diag overlap test2 - refusing"
    print(f"wanted {len(wanted)} ids (dev+diag); sealed test2 ids excluded: {len(sealed)}", flush=True)
    have = {p.stem for p in out.glob("*.wav")}
    manifest = {}
    for off in range(0, 4158, 100):
        page = json.loads(get(API.format(off=off)))
        for row in page["rows"]:
            r = row["row"]
            rid = r["id"]
            if rid in sealed or rid not in wanted:
                continue
            manifest[rid] = {"row_idx": row["row_idx"], "question": r["question"],
                             "provided_transcription": r["transcription"]}
            if rid in have:
                continue
            (out / f"{rid}.wav").write_bytes(get(r["audio"][0]["src"]))
        print(f"offset {off}: {len(manifest)}/{len(wanted)}", flush=True)
    (out / "manifest.json").write_text(json.dumps({"dataset": "yijingwu/HeySQuAD_human", "revision": REVISION,
                                                   "split": "validation", "items": manifest}, indent=1))
    missing = wanted - set(manifest)
    print(f"done: {len(manifest)} clips, missing {len(missing)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
