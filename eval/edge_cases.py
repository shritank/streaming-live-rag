"""python -m eval.edge_cases [--scenarios DIR] [--corpus-dir DIR] [--out reports/edge_cases.md]

Collects the worst-scoring turns per gate into a Markdown report, so the
benchmark report can analyse >= 3 concrete failures with a root cause and a
mitigation, per project context §13.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from streaming_rag.config import load_config
from streaming_rag.telemetry.trace import assemble_turns

from .gates import score_g4
from .loader import discover_scenarios, load_ground_truth
from .runner import run_scenario


async def collect(scenarios: list[Path], corpus_dir: str) -> list[dict]:
    config = load_config()
    config.corpus_dir = corpus_dir
    findings = []
    for path in scenarios:
        gt = load_ground_truth(path)
        turn_results, trace = await run_scenario(str(path), config, impl="real", time_scale=8)
        g4 = score_g4(trace, gt)
        if g4.fabricated:
            findings.append({"scenario": path.name, "gate": "G4", "issue": "fabricated citations",
                              "detail": g4.fabricated})
        for turn in assemble_turns(trace):
            for e in turn.of("error"):
                findings.append({"scenario": path.name, "gate": "robustness",
                                  "issue": e.get("error_type"), "detail": e.get("message")})
        for tr in turn_results:
            if tr.get("uncertainty") and tr.get("reason") not in ("degraded",):
                findings.append({"scenario": path.name, "gate": "G4", "issue": "uncertainty flagged",
                                  "detail": tr["uncertainty"]})
    return findings


def render_markdown(findings: list[dict]) -> str:
    lines = ["# Edge-case failure log", "",
             "Auto-collected from the dev scenario suite. Each row is a turn where",
             "grounding, robustness or the corpus's coverage broke down.", ""]
    if not findings:
        lines.append("No edge cases found in the current suite.")
        return "\n".join(lines)
    lines.append("| scenario | gate | issue | detail |")
    lines.append("|---|---|---|---|")
    for f in findings[:30]:
        detail = str(f["detail"])[:100].replace("|", "\\|")
        lines.append(f"| {f['scenario']} | {f['gate']} | {f['issue']} | {detail} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", default="eval/scenarios")
    parser.add_argument("--corpus-dir", default="fixtures/dev_corpus")
    parser.add_argument("--out", default="reports/edge_cases.md")
    args = parser.parse_args(argv)

    scenarios = discover_scenarios(args.scenarios)
    findings = asyncio.run(collect(scenarios, args.corpus_dir))
    md = render_markdown(findings)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {len(findings)} findings to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
