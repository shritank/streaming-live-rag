from __future__ import annotations
import asyncio, json, os, shutil, subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "eval" / "results" / "improve_final_dev.json"
SCENARIO_DIR = ROOT / "eval" / "scenarios_real_dev"
OUT = ROOT / "eval" / "results" / "answer_recall_diagnostics.json"
AI_OUT = ROOT / "eval" / "results" / "answer_recall_ai_analysis.md"

def plain(x: Any, depth=0):
    if depth > 8:
        return repr(x)
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if is_dataclass(x):
        return plain(asdict(x), depth + 1)
    if isinstance(x, dict):
        return {str(k): plain(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [plain(v, depth + 1) for v in x]
    if hasattr(x, "model_dump"):
        try: return plain(x.model_dump(), depth + 1)
        except Exception: pass
    if hasattr(x, "__dict__"):
        try: return plain(vars(x), depth + 1)
        except Exception: pass
    return repr(x)

def load_failures():
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    out = []
    for key, ok in report.get("items", {}).get("answer", {}).items():
        if not ok:
            stem, item = key.rsplit("|", 1)
            out.append((stem, item))
    return out

def ask_ollama(diag):
    ollama = shutil.which("ollama")
    if not ollama:
        return None
    model = os.environ.get("RECALL_LLM_MODEL", "qwen2.5:7b")
    prompt = """Analyze these answer-recall failures in a streaming RAG system.
Classify each affected scenario as DECOMPOSITION, RETRIEVAL,
EVIDENCE_SELECTION, SYNTHESIS, CITATION, ABSTENTION, or OTHER.
Then identify recurring causes and propose at most 3 small, testable
experiments. Do not claim an improvement until the evaluator confirms it.
Prioritize answer recall while preserving citation/claim precision and latency.

DIAGNOSTIC:
""" + json.dumps(diag, indent=2, ensure_ascii=False)
    try:
        p = subprocess.run([ollama, "run", model], input=prompt,
                           text=True, capture_output=True, timeout=600)
        if p.returncode == 0:
            return p.stdout.strip()
        return "Ollama error: " + p.stderr.strip()
    except Exception as e:
        return "Ollama invocation error: " + repr(e)

async def main():
    if not REPORT.exists():
        print("ERROR: missing", REPORT); return 2
    failures = load_failures()
    stems = sorted({s for s, _ in failures})
    print(f"Found {len(failures)} failures across {len(stems)} scenarios.")

    from streaming_rag.config import load_config
    from eval.runner import run_scenario_detailed

    config = load_config()
    config.corpus_dir = str(ROOT / "data" / "corpus")
    wanted = set(failures)
    diag = {
        "source_report": str(REPORT),
        "failure_count": len(failures),
        "affected_scenarios": stems,
        "failures": []
    }

    for i, stem in enumerate(stems, 1):
        print(f"[{i}/{len(stems)}] {stem}")
        path = SCENARIO_DIR / f"{stem}.json"
        if not path.exists():
            diag["failures"].append({"scenario": stem, "error": str(path)})
            continue
        try:
            scenario = json.loads(path.read_text(encoding="utf-8"))
            results, trace, answers = await run_scenario_detailed(
                str(path), config, time_scale=8.0
            )
            diag["failures"].append({
                "scenario": stem,
                "failed_items": [f"{s}|{v}" for s, v in wanted if s == stem],
                "ground_truth": plain(scenario.get("ground_truth", {})),
                "results": plain(results),
                "answers": plain(answers),
                "trace": plain(trace)
            })
        except Exception as e:
            diag["failures"].append({"scenario": stem, "error": repr(e)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nDiagnostic report:", OUT)

    ai = ask_ollama(diag)
    if ai:
        AI_OUT.write_text("# Local-model answer-recall analysis\n\n" + ai + "\n",
                          encoding="utf-8")
        print("Local-model analysis:", AI_OUT)
    else:
        print("Ollama not found; diagnostic extraction completed without an LLM.")
    return 0

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
