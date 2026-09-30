"""Markdown tables for the final report, straight from the saved result files (no hand-copying).

    python handoff/final_audit/make_tables.py latency | modes | compare | ablations | gates_table | quality | overhead

Reads eval/results/final_audit/. Paired differences use eval.compare_runs.compare (cluster bootstrap, 4000
resamples, clustered by scenario); '*' marks a 95 % CI that excludes 0.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, ".")
from eval.compare_runs import compare  # noqa: E402

R = Path("eval/results/final_audit")


def load(name):
    return json.loads((R / f"{name}.json").read_text(encoding="utf-8"))


def pct(rep, key):
    row = rep["quality"][key]
    return f"{row['rate']:.1%}" if row["n"] else "n/a"


def latency():
    boundaries = [("Controller decision, per transcript chunk", "controller.decision_ms"),
                  ("Final controller pass at utterance end", "engine.final_decision_ms"),
                  ("Answer assembly at utterance end", "engine.synthesis_ms"),
                  ("Answer telemetry / bookkeeping", "engine.answer_telemetry_ms"),
                  ("**Post-speech: utterance_end -> answer emitted (all turns)**", "engine.post_speech_ms"),
                  ("  of which waiting for in-flight retrieval / claim selection", "engine.retrieval_wait_ms")]
    context = [("Retrieval per sub-query (runs during speech)", "retrieval.subquery_ms"),
               ("  e5 query encode", "retrieval.e5_encode_ms"), ("  BM25", "retrieval.bm25_ms"),
               ("  cross-encoder rerank of 20 passages", "ce.rerank_ms"),
               ("Claim selection + refusal gate (prefetch, during speech)", "synthesis.prefetch_ms")]
    for split, name in (("DEV", "lat_dev"), ("DIAG", "lat_diag")):
        rep = load(name)
        st = rep["stages"]
        spec = rep.get("speculation", {})
        print(f"\n**{split}** - {rep['n_scenarios']} scenarios, real time (1x), providers "
              f"{sorted(set(rep['execution_providers'].values()))}, speculative final-pass retrievals "
              f"launched {spec.get('launched')}, reused {spec.get('reused')}\n")
        print("| Boundary | n | p50 | p95 | p99 | max (ms) | p99 < 5 ms |")
        print("|---|---|---|---|---|---|---|")
        for label, key in boundaries:
            v = st.get(key)
            if v:
                print(f"| {label} | {v['n']} | {v['p50']:.2f} | {v['p95']:.2f} | {v['p99']:.2f} | {v['max']:.2f} | "
                      f"{'**yes**' if v['p99'] < 5 else 'no'} |")
        print("| *context (not a < 5 ms target)* | | | | | | |")
        for label, key in context:
            v = st.get(key)
            if v:
                print(f"| {label} | {v['n']} | {v['p50']:.1f} | {v['p95']:.1f} | {v['p99']:.1f} | {v['max']:.1f} | - |")
        over = rep.get("turns_over_5ms", [])
        print(f"\nTurns with post-speech > 5 ms: {len(over)} of {st['engine.post_speech_ms']['n']}"
              + ("".join(f"; {t['scenario']} {t['post_speech_ms']} ms" for t in over[:5]) if over else "."))


def modes():
    reps = {m: load(f"mode_{m}_diag") for m in ("streaming", "deferred", "baseline")}
    print("| Engine mode (DIAG) | Answer | Citation | Claim precision | Abstention | Exact decomp. |")
    print("|---|---|---|---|---|---|")
    for m, r in reps.items():
        print(f"| {m} | {pct(r, 'answer')} | {pct(r, 'citation')} | {pct(r, 'claim_correct')} | {pct(r, 'abstention')} | "
              f"{pct(r, 'split_exact')} |")
    for a, b in (("deferred", "streaming"), ("baseline", "streaming")):
        print("")
        print(f"Paired, {a} -> {b}:")
        for row in compare(reps[a], reps[b]):
            if row["baseline"] is None:
                continue
            star = "*" if row["significant"] else ""
            print(f"- {row['metric']}: {row['baseline']:.1%} -> {row['candidate']:.1%}, diff {row['diff']:+.1%} "
                  f"[{row['ci'][0]:+.1%}, {row['ci'][1]:+.1%}]{star} (n={row['n_paired']})")


def compare_log():
    print("```")
    print((R / "compare_diag.log").read_text(encoding="utf-8", errors="replace").replace("\r", "").strip())
    print("```")


def gates_table():
    rows = [("Legacy synthetic (fixtures/dev_corpus), 8x", "gates_legacy_ts8_reps3"),
            ("Real corpus, 15 scenarios, 8x", "gates_real_corpus_ts8_reps3"),
            ("**DEV**, 60 scenarios, **real time (1x)**", "gates_real_dev_ts1_reps3"),
            ("**DIAG**, 60 scenarios, **real time (1x)**", "gates_real_diag_ts1_reps3")]
    print("| Suite (`eval.run_all --reps 3`) | Scenarios | G2 early retrieval (false triggers) | G3 multi-intent | "
          "G4 grounding (fabricated) | G5 refinement | G6 telemetry | Verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for label, name in rows:
        t = (R / f"{name}.log").read_text(encoding="utf-8", errors="replace").replace("\r", "")

        def grab(key):
            m = re.search(key + r"[^:\n]*:\s*([^\n]*)", t)
            return m.group(1).strip() if m else "?"
        n = re.search(r"scenarios run:\s*(\d+)", t)
        v = re.search(r"VERDICT:\s*(\w+)", t)
        print(f"| {label} | {n.group(1) if n else '?'} | {grab('G2')} | {grab('G3')} | {grab('G4')} | {grab('G5')} | "
              f"{grab('G6')} | **{v.group(1) if v else '?'}** |")


def ablations():
    configs = [("baseline (defaults)", "baseline"), ("retrieval.mode=sparse (BM25 only)", "sparse"),
               ("retrieval.mode=dense (e5 only)", "dense"), ("no cross-encoder rerank", "norerank"),
               ("no refusal gate", "nogate"), ("lexical claim selector (also no gate)", "lexsel"),
               ("speculative final pass off", "nospec")]
    for split in ("dev", "diag"):
        base = load(f"abl_{split}_baseline")
        print(f"\n**{split.upper()}** (paired against the default configuration; '*' = 95 % CI excludes 0)\n")
        print("| Configuration | Answer | d answer [95 % CI] | Citation | Claim precision | d claim precision | Abstention |")
        print("|---|---|---|---|---|---|---|")
        for label, key in configs:
            r = load(f"abl_{split}_{key}")
            rows = {x["metric"]: x for x in compare(base, r)}

            def d(metric):
                x = rows[metric]
                if key == "baseline" or x["baseline"] is None:
                    return "-"
                return f"{x['diff']:+.1%} [{x['ci'][0]:+.1%}, {x['ci'][1]:+.1%}]{'*' if x['significant'] else ''}"
            print(f"| {label} | {pct(r, 'answer')} | {d('answer')} | {pct(r, 'citation')} | {pct(r, 'claim_correct')} | "
                  f"{d('claim_correct')} | {pct(r, 'abstention')} |")
    print("\n**Refusal gate on the HeySQuAD sets** (typed questions; ~50 % are hard SQuAD2-style unanswerable ones)\n")
    print("| Set | Configuration | Answer | Citation | Correct abstention | d answer | d abstention |")
    print("|---|---|---|---|---|---|---|")
    for split, label in (("dev", "DEV (279 + 279 Q)"), ("diag", "DIAG (219 + 219 Q)")):
        base, off = load(f"spoken_{split}_typed"), load(f"spoken_{split}_nogate")
        rows = {x["metric"]: x for x in compare(base, off)}
        d = lambda m: f"{rows[m]['diff']:+.1%} [{rows[m]['ci'][0]:+.1%}, {rows[m]['ci'][1]:+.1%}]{'*' if rows[m]['significant'] else ''}"
        print(f"| {label} | learned gate (default) | {pct(base, 'answer')} | {pct(base, 'citation')} | {pct(base, 'abstention')} | - | - |")
        print(f"| {label} | no gate | {pct(off, 'answer')} | {pct(off, 'citation')} | {pct(off, 'abstention')} | {d('answer')} | {d('abstention')} |")


def quality():
    print("| Set | Answer recall | Citation recall | Claim precision | Abstention | Exact decomp. | Over-split |")
    print("|---|---|---|---|---|---|---|")
    for label, name in (("Text DEV (102 Q)", "abl_dev_baseline"), ("Text DIAG (98 Q)", "abl_diag_baseline"),
                        ("HeySQuAD DEV typed (279 answerable + 279 unanswerable)", "spoken_dev_typed"),
                        ("HeySQuAD DIAG typed (219 + 219)", "spoken_diag_typed")):
        r = load(name)
        q = r["quality"]

        def cell(key):
            row = q[key]
            return f"{row['rate']:.1%} [{row['ci'][0]:.1%}, {row['ci'][1]:.1%}]" if row["n"] else "n/a"
        print(f"| {label} | {cell('answer')} | {cell('citation')} | {cell('claim_correct')} | {cell('abstention')} | "
              f"{cell('split_exact')} | {cell('oversplit')} |")


def overhead():
    rep = json.loads((R / "telemetry_overhead.json").read_text(encoding="utf-8"))
    print(f"Turns per sink: {rep['turns_per_sink']} ({rep['scenarios']} DEV scenarios x {rep['rounds']} interleaved rounds, "
          f"time-scale {rep['time_scale']:g}); external clock: utterance_end enqueued -> turn_result dequeued.")
    print("")
    print("| Sink | p50 | p95 | p99 | max (ms) | p95 vs null | emit() cost |")
    print("|---|---|---|---|---|---|---|")
    for k in ("null", "buffered", "jsonl"):
        v = rep[k]
        rel = "-" if k == "null" else f"{rep[k + '_vs_null_p95_pct']:+.1f} %"
        print(f"| {k} | {v['p50']:.2f} | {v['p95']:.2f} | {v['p99']:.2f} | {v['max']:.2f} | {rel} | {rep['emit_cost_us'][k]:.2f} us |")


if __name__ == "__main__":
    {"latency": latency, "modes": modes, "compare": compare_log, "ablations": ablations, "gates_table": gates_table,
     "quality": quality, "overhead": overhead}[sys.argv[1]]()
