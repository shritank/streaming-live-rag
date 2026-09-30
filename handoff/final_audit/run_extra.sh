#!/usr/bin/env bash
# Extra answer-only runs of the final audit (GPU only; no timing is measured here):
#   * refusal gate on / off on the HeySQuAD sets (they contain hard SQuAD2-style unanswerable questions;
#     the text scenario sets contain only 5 easy off-topic ones, so they cannot show the gate's value)
#   * rerank pool 20 on the HeySQuAD sets (confirms the text-set result on 279 / 219 questions)
# Run from the repository root.
export STREAMING_RAG_ORT_PROVIDER=cuda STREAMING_RAG_ASR_DEVICE=cuda
P=./.venv-gpu/Scripts/python.exe
R=eval/results/final_audit
stamp() { echo "$(date +%H:%M:%S) $*" >> "$R/progress.log"; }
q() { local name=$1 sc=$2 co=$3; shift 3
  $P -m eval.run_quality --scenarios "$sc" --corpus-dir "$co" "$@" --out "$R/$name.json" > "$R/$name.log" 2>/dev/null
  stamp "X done $name"; }
( q spoken_dev_nogate  eval/scenarios_heysquad_dev_clean  data/corpus      --set synthesis.refusal_gate=none
  q spoken_diag_nogate eval/scenarios_heysquad_diag_clean data/corpus_test --set synthesis.refusal_gate=none ) &
( q spoken_dev_pool20  eval/scenarios_heysquad_dev_clean  data/corpus      --set retrieval.rerank_pool=20
  q spoken_diag_pool20 eval/scenarios_heysquad_diag_clean data/corpus_test --set retrieval.rerank_pool=20 ) &
wait
stamp "EXTRA DONE"
