

---

# New workstation pass (29 Sep 2026): RTX 6000 Ada — GPU-only neural inference

Everything below is **NEW MACHINE** unless labelled OLD MACHINE. Labels as before: **DEV** (used to
decide), **DIAG** (held-out confirmation), **PRELIMINARY**. Sealed test2 was **not** run, read for
tuning, or downloaded (no test2 audio exists on this machine).

## W0 — Machine, environment, GPU verification [FACT]

| | OLD MACHINE | NEW MACHINE |
|---|---|---|
| CPU | Intel i5-10210U, 4 C / 8 T, 15 W | Intel Xeon w9-3475X, 36 C / 72 T |
| RAM | ~8 GB (often < 1.6 GB free) | 255 GB (236 GB free) |
| GPU | NVIDIA MX250, 2 GB | NVIDIA RTX 6000 Ada Generation, 48 GB, compute capability 8.9 |
| Driver / CUDA | — | 573.44 / CUDA 12.8 (WDDM) |
| OS | Windows | Windows 11 Pro for Workstations 10.0.26200 |
| Python | LibreOffice-bundled 3.11 | 3.11.9 (python.org) |

The machine had no Python, git or CUDA user-space libraries; installed with the user's approval.
Two environments: `.venv` = `requirements.lock` exactly (CPU `onnxruntime 1.20.1`; kept only for the
CPU/Docker reproducibility path) and `.venv-gpu` = the same pins with `onnxruntime-gpu 1.20.0` (the
version the old machine used), `nvidia-cudnn-cu12 9.26`, `nvidia-cublas-cu12 12.9`, CUDA runtime 12.9,
`faster-whisper 1.2.1`, `ctranslate2 4.8.2`, `onnx 1.23` and (at the user's request) `torch
2.14.0+cu126` — torch is used only by ONNX Runtime's offline graph optimiser, never at run time.
Models re-fetched at the pinned revisions and SHA-256-verified (e5 / MiniLM-L6 CE / Whisper
tiny,base,small.en); the migrated SQuAD2 reader ONNX matches its recorded hash `6e9e62e6…`.
HeySQuAD dev+diag audio (996 clips, dataset revision `2a9aff60`) re-downloaded by `fetch_audio.py`,
which refuses test2 ids.

Pitfall found and fixed: `faster-whisper` pulls in CPU `onnxruntime`, which shares the `onnxruntime/`
package directory with `onnxruntime-gpu` and silently replaces it — both were removed and only
`onnxruntime-gpu` reinstalled.

**GPU verification (controlled test, product code path, defaults only)** —
`handoff/gpu_tools/verify_product_gpu.py`, output `handoff/newmachine/gpu_verify_product.json`:
every session built by `neural.get_e5_encoder / get_cross_encoder / get_reader` bound
`CUDAExecutionProvider`; Whisper (`StreamingWhisper(device="auto")`) loaded on `cuda`; GPU memory
0.8 → 2.6 GB during the test; answers correct. ONNX Runtime's per-operator profiler shows where
kernel time goes (5 × 192 tokens): e5 100 % CUDA; cross-encoder and reader see below (W5).

**GPU made mandatory** (`streaming_rag/gpu.py`, used by `retrieval/neural.py` and `asr/whisper.py`):
`STREAMING_RAG_ORT_PROVIDER` / `STREAMING_RAG_ASR_DEVICE` default `auto` = CUDA whenever an NVIDIA
GPU is present; if the installed runtime cannot use it, `GpuUnavailable` is raised instead of running
neural inference on the CPU. A session that does not bind CUDA first is refused. `cpu` is only an
explicit opt-out (the CPU Docker image has no GPU and is unaffected). Every eval output records
`execution_providers`. Tests: `tests/test_gpu.py` (8).

## W1 — Baseline reproduction, OLD vs NEW [DEV]

Same code, same configuration, same data.

Quality is hardware-independent for the RAG models: dev text answers **bit-identical** to the old
machine's saved run on all 60 scenarios (new CPU and new GPU alike): answer 62.7 %, citation 67.6 %,
claim precision 84.5 %, abstention 5/5, exact decomposition 100 %, G2–G6 100 %.

**Correction to earlier results (old-hardware timeouts):** the old run of HeySQuAD dev clean
(`r8_final_heydev_clean`, 70.6 %) differs from the new machine (71.3 %) on exactly 2 of 558
scenarios, and both are *timeouts on the old laptop* ("internal error or timeout prevented a
grounded answer"); the new machine answers both. The ±0.7 point "non-determinism" noted in E0 was
this. Timeout answers appear in old result files `r8_final_heydev_clean` (2), `r8_final_heydev_asr`
(2), `r8_final_heydiag_asr` (1), `r3_nogate_heydev_clean` (1) and `SEALED_heysquad_test2_asr` (1 —
read only, not re-run); the final sealed files `SEALED_final_*` contain none.

Timings, OLD vs NEW (same protocol as the old measurement in each row):

| Component | OLD MACHINE | NEW MACHINE | Speed-up |
|---|---|---|---|
| Whisper tiny.en, GPU int8, 40 clips, ms/clip | 306 | 192 | 1.6× |
| Whisper base.en, GPU int8 | 475 | 210 | 2.3× |
| Whisper small.en, GPU int8 | 1,193 | 281 | 4.2× |
| Whisper base.en, CPU int8, 6 threads | 1,405 | 943 | 1.5× |
| Cross-encoder rerank, 5 passages, p50 (real-time pipeline) | 126 ms (CPU) | 89–147 ms (CPU) / **14 ms (GPU)** | 9× (GPU vs old) |
| Claim selection p50 | 170 ms (CPU) | 75–144 (CPU) / **24 ms (GPU)** | 7× |
| Retrieval per sub-query p50 | 150 ms (CPU) | 115–181 (CPU) / **51 ms (GPU)** | 3× |
| Test suite (190 tests) | ~65 s | 14 s | 4.6× |
| HeySQuAD dev clean quality run (558 scen.) wall | 25,419 s (contended) | 461 s | indicative only |
| Dev text quality run (60 scen., time-scale 8) wall | 122.8 s | 86 s | bounded by replay time |

CPU run-to-run variance on the Xeon is large (identical code, idle machine: rerank p50 89 vs 147 ms in
two runs) — another reason all neural work now runs on the GPU.

## W2 — Streaming ASR: 1 s vs 2 s minimum buffer, and the full-dev rerun

The 100-clip comparison was completed on the OLD MACHINE before the migration (recomputed here from
the raw per-decode logs, nearest-rank percentiles, nothing excluded):

| OLD MACHINE, 100 dev clips, base.en MX250 int8 | identical to offline | decode p50 / p95 / p99 / max | ASR post-speech p50 / p95 / p99 / max |
|---|---|---|---|
| 1 s first decode | 83/100 | 531 / 3,964 / 11,298 / 16,969 ms | 1,101 / 12,203 / 15,736 / 18,260 ms |
| 2 s first decode | 91/100 | 547 / 1,420 / 6,572 / 15,008 | 889 / 9,220 / 14,949 / 15,101 |
| **2 s + 64 tokens (adopted)** | 91/100 | 548 / 1,210 / 2,631 / 3,846 | 767 / 2,711 / 4,121 / 7,177 |

The full 558-clip run with the adopted settings (stopped at 300/558 on the old machine, no output)
was re-run **on the NEW MACHINE** (RTX 6000 Ada, base.en int8), 737 s wall:

| NEW MACHINE, 2 s + 64 tokens | clips | identical to offline | decodes n | decode p50 / p95 / p99 / max | decodes > 1 s / > 3 s | ASR post-speech p50 / p95 / p99 / max |
|---|---|---|---|---|---|---|
| all dev | 558 | 499 (89.4 %) | 3,032 | 205 / 432 / 1,279 / 2,364 ms | 54 / 0 | 236 / 479 / 1,240 / 1,713 ms |
| same 100 clips as above | 100 | 93 | — | 204 / 415 / 1,019 / 2,364 | 6 / 0 | 235 / 501 / 1,445 / 1,508 |

The remaining > 1 s decodes are runaway generations stopped by the 64-token cap (~1 s of beam-5
decoding at the cap), not stalls.

## W3 — Whisper output is NOT hardware-independent [FACT]

Offline base.en on the RTX 6000 Ada vs the old MX250, identical model revision, settings and audio:
**70 of 996 transcripts differ** (CTranslate2's int8 GPU kernels differ between Pascal sm_61 and Ada
sm_89). Some differences remove old errors (a "Okay. Okay. Okay…" hallucination loop is gone),
others add new ones. Aggregate WER is equal or slightly better:

| base.en offline | dev WER | dev content-word err. | dev exact | diag WER | diag content-word err. |
|---|---|---|---|---|---|
| OLD MACHINE (MX250) | 9.36 % | 8.83 % | 338/558 | 8.33 % | 8.18 % |
| NEW MACHINE (Ada) | 9.23 % | 8.83 % | 340/558 | 7.58 % | 8.22 % |

Consequence: Whisper-based answer-quality numbers (E4–E6) must be re-measured on this machine, not
carried over (context.md had assumed hardware independence).
