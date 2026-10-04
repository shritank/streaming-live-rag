.PHONY: install install-gpu fuse test corpus eval eval-real eval-gpu compare demo ablate docker

install:
	pip install -r requirements.lock

install-gpu:            # NVIDIA machine: GPU-only neural inference (see requirements-gpu.txt for the caveats)
	python -m pip install -r requirements-gpu.txt
	python -m pip uninstall -y onnxruntime onnxruntime-gpu
	python -m pip install --no-deps onnxruntime-gpu==1.20.0

fuse:                   # one-time CUDA graph build (optional, ~2x faster model calls)
	python -m streaming_rag.retrieval.neural --fuse

corpus:
	python -m fixtures.build_dev_corpus --seed 1 --out fixtures/dev_corpus
	python -m data.build_squad_corpus --n-articles 14 --out data/corpus

test:
	pytest tests/ -q

eval:
	python -m eval.run_all --scenarios eval/scenarios --time-scale 8 --reps 1 --corpus-dir fixtures/dev_corpus

eval-real:
	python -m eval.run_all --scenarios eval/scenarios_real_corpus --time-scale 8 --reps 1 --corpus-dir data/corpus

eval-gpu:               # the official procedure on the GPU: real-time replay, 3 reps, median
	python -m eval.run_all --scenarios eval/scenarios_real_dev --corpus-dir data/corpus --time-scale 1 --reps 3

compare:                # streaming vs deferred vs baseline, real pipeline, real time (final report 5.10)
	python -m eval.compare --modes streaming,deferred,baseline --scenarios eval/scenarios_real_test --corpus-dir data/corpus_test --time-scale 1

demo:                   # the <= 5 minute demonstration (~90 s of it is real-time replay)
	python -m streaming_rag.cli replay eval/scenarios/dev_001_multi_intent.json --corpus-dir fixtures/dev_corpus --time-scale 1 --warmup --telemetry demo1.jsonl
	python -m streaming_rag.telemetry.report demo1.jsonl
	python -m streaming_rag.cli evidence "cancellation policy plan customer workshop pune" --corpus-dir fixtures/dev_corpus
	python -m streaming_rag.cli replay eval/scenarios/dev_002_late_detail.json --corpus-dir fixtures/dev_corpus --time-scale 1 --warmup --telemetry demo2.jsonl
	python -m streaming_rag.telemetry.report demo2.jsonl

ablate:
	python -m eval.ablate --scenarios eval/scenarios_real_dev --corpus-dir data/corpus

docker:
	docker compose up --build
