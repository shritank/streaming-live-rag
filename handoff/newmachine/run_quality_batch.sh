#!/usr/bin/env bash
# Parallel quality runs (answers only - no latency measured here). GPU-only neural inference.
export STREAMING_RAG_ORT_PROVIDER=cuda STREAMING_RAG_ASR_DEVICE=cuda
P=./.venv-gpu/Scripts/python.exe; L=handoff/newmachine
q() { $P -m eval.run_quality --scenarios "$1" --corpus-dir "$2" --out "$L/$3.json" > "$L/$3.log" 2>/dev/null; echo "done $3"; }
job1() {  # diag streaming ASR (quality only; its timing is under load and not reported) -> restream -> quality
  $P handoff/gpu_tools/gpu_env.py handoff/asr_tools/stream_asr.py base.en cuda int8 $L/stream_diag_2s64.json \
     --min-first 2 --max-tokens 64 --ids handoff/asr_tools/diag_ids.txt > $L/stream_diag.log 2>/dev/null
  ./.venv/Scripts/python.exe -m eval.heysquad --restream eval/scenarios_heysquad_diag_clean $L/stream_diag_2s64.json $L/scenarios_heysquad_diag_wbase_stream_ada
  q $L/scenarios_heysquad_diag_wbase_stream_ada data/corpus_test Q_heydiag_wstream
}
job2() { q $L/scenarios_heysquad_dev_wbase_ada data/corpus Q_heydev_woffline
         q $L/scenarios_heysquad_dev_wbase_stream_ada data/corpus Q_heydev_wstream
         q $L/scen_audio100_nohyp data/corpus Q_audio100_nohyp
         q $L/scen_audio100_hyp data/corpus Q_audio100_hyp; }
job3() { q eval/scenarios_heysquad_diag_clean data/corpus_test Q_heydiag_clean
         q eval/scenarios_heysquad_diag_asr data/corpus_test Q_heydiag_asr
         q $L/scenarios_heysquad_diag_wbase_ada data/corpus_test Q_heydiag_woffline; }
job4() { q eval/scenarios_real_test data/corpus_test Q_text_diag
         q eval/scenarios_heysquad_dev_asr data/corpus Q_heydev_asr; }
job1 & job2 & job3 & job4 & wait
echo ALL_DONE
