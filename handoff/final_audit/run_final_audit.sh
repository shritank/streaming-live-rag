#!/usr/bin/env bash
# Final Theme 4 audit measurements for the SHIPPED configuration (rerank_pool = 20).
# Scope: streaming transcript -> RAG -> grounded, cited answer (audio out of scope).
# GPU-only neural inference: STREAMING_RAG_ORT_PROVIDER=cuda makes CUDA mandatory (no CPU fallback).
# Sealed test2 is NOT referenced anywhere in this script.
#
#   Phase A  timing-sensitive, sequential, machine otherwise idle
#   Phase B  answer-only, parallel (official gates --reps 3, ablations, typed spoken-question quality)
#
# Results: eval/results/final_audit/ (the earlier rerank_pool = 5 measurements that led to the pool-20
# decision are kept in eval/results/final_audit_pool5/).  Run from the repository root:
#   bash handoff/final_audit/run_final_audit.sh
export STREAMING_RAG_ORT_PROVIDER=cuda STREAMING_RAG_ASR_DEVICE=cuda PYTHONIOENCODING=utf-8
P=./.venv-gpu/Scripts/python.exe
R=eval/results/final_audit
mkdir -p "$R"
stamp() { echo "$(date +%H:%M:%S) $*" >> "$R/progress.log"; }
: > "$R/progress.log"

# ---------------------------------------------------------------- Phase A (exclusive machine)
stamp "A1 latency profile DEV (real time, 1x)"
$P -m eval.latency_profile --scenarios eval/scenarios_real_dev --corpus-dir data/corpus --time-scale 1 \
   --out "$R/lat_dev.json" > "$R/lat_dev.log" 2>/dev/null
stamp "A2 latency profile DIAG (real time, 1x)"
$P -m eval.latency_profile --scenarios eval/scenarios_real_test --corpus-dir data/corpus_test --time-scale 1 \
   --out "$R/lat_diag.json" > "$R/lat_diag.log" 2>/dev/null
stamp "A3 eval.compare streaming / deferred / baseline on DIAG (1x, paired turns)"
$P -m eval.compare --modes streaming,deferred,baseline --scenarios eval/scenarios_real_test --corpus-dir data/corpus_test \
   --time-scale 1 > "$R/compare_diag.log" 2>/dev/null
stamp "A4 telemetry overhead (Task 4 section 4.3): null vs buffered vs jsonl sink"
$P handoff/final_audit/telemetry_overhead.py eval/scenarios_real_dev data/corpus 4 2 > "$R/telemetry_overhead.log" 2>/dev/null
stamp "PHASE A DONE"

# ---------------------------------------------------------------- Phase B (parallel, answers only)
q() {  # q NAME SCENARIOS CORPUS [--set ...]
  local name=$1 sc=$2 co=$3; shift 3
  $P -m eval.run_quality --scenarios "$sc" --corpus-dir "$co" "$@" --out "$R/$name.json" > "$R/$name.log" 2>/dev/null
  stamp "B done $name"
}
jobGatesDev()  { $P -m eval.run_all --scenarios eval/scenarios_real_dev  --corpus-dir data/corpus      --time-scale 1 --reps 3 > "$R/gates_real_dev_ts1_reps3.log" 2>/dev/null;  stamp "B done gates real_dev ts1 reps3"; }
jobGatesDiag() { $P -m eval.run_all --scenarios eval/scenarios_real_test --corpus-dir data/corpus_test --time-scale 1 --reps 3 > "$R/gates_real_diag_ts1_reps3.log" 2>/dev/null; stamp "B done gates real_diag ts1 reps3"; }
jobGatesSmall() {
  $P -m eval.run_all --scenarios eval/scenarios --time-scale 8 --reps 3 > "$R/gates_legacy_ts8_reps3.log" 2>/dev/null; stamp "B done gates legacy ts8 reps3"
  $P -m eval.run_all --scenarios eval/scenarios_real_corpus --corpus-dir data/corpus --time-scale 8 --reps 3 > "$R/gates_real_corpus_ts8_reps3.log" 2>/dev/null; stamp "B done gates real_corpus ts8 reps3"
}
jobAblations() {
  for split in dev diag; do
    if [ $split = dev ]; then sc=eval/scenarios_real_dev; co=data/corpus; else sc=eval/scenarios_real_test; co=data/corpus_test; fi
    q abl_${split}_baseline $sc $co
    q abl_${split}_sparse   $sc $co --set retrieval.mode=sparse
    q abl_${split}_dense    $sc $co --set retrieval.mode=dense
    q abl_${split}_norerank $sc $co --set retrieval.reranker=none
    q abl_${split}_nogate   $sc $co --set synthesis.refusal_gate=none
    q abl_${split}_lexsel   $sc $co --set synthesis.selector=lexical
    q abl_${split}_nospec   $sc $co --set engine.speculative_final=false
  done
}
jobSpoken() {   # HeySQuAD typed: ~50 % hard unanswerable questions (the text sets have only 5 easy ones)
  q spoken_dev_typed   eval/scenarios_heysquad_dev_clean  data/corpus
  q spoken_diag_typed  eval/scenarios_heysquad_diag_clean data/corpus_test
  q spoken_dev_nogate  eval/scenarios_heysquad_dev_clean  data/corpus      --set synthesis.refusal_gate=none
  q spoken_diag_nogate eval/scenarios_heysquad_diag_clean data/corpus_test --set synthesis.refusal_gate=none
}
jobModes() {    # gold-referenced quality of the three engine modes (timing-independent; latency is in Phase A)
  for m in streaming deferred baseline; do q mode_${m}_diag eval/scenarios_real_test data/corpus_test --set engine.mode=$m; done
  for m in deferred baseline; do q mode_${m}_dev eval/scenarios_real_dev data/corpus --set engine.mode=$m; done
}
jobAnalysis() {
  $P -m eval.error_analysis --scenarios eval/scenarios_real_dev  --corpus-dir data/corpus      --out "$R/ea_dev.json"  > /dev/null 2>&1; stamp "B done error analysis dev"
  $P -m eval.error_analysis --scenarios eval/scenarios_real_test --corpus-dir data/corpus_test --out "$R/ea_diag.json" > /dev/null 2>&1; stamp "B done error analysis diag"
  for split in "dev data/corpus" "diag data/corpus_test"; do set -- $split
    $P -m eval.retrieval_eval --corpus-dir $2 --sample 1000 --seed 0 --config "hybrid+CE(default)" \
       --config "sparse_BM25_only:retrieval.mode=sparse" --config "dense_e5_only:retrieval.mode=dense" \
       --config "hybrid_no_rerank:retrieval.reranker=none" --config "rerank_pool_5:retrieval.rerank_pool=5" \
       > "$R/retrieval_ablation_$1.txt" 2>/dev/null; stamp "B done retrieval ablation $1"
  done
}
stamp "B start"
jobGatesDev & jobGatesDiag & jobGatesSmall & jobAblations & jobSpoken & jobModes & jobAnalysis & wait
stamp "PHASE B DONE"
echo ALL_DONE >> "$R/progress.log"
