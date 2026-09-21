.PHONY: install test corpus eval eval-real compare ablate docker

install:
	pip install -r requirements.lock

corpus:
	python -m fixtures.build_dev_corpus --seed 1 --out fixtures/dev_corpus
	python -m data.build_squad_corpus --n-articles 14 --out data/corpus

test:
	pytest tests/ -q

eval:
	python -m eval.run_all --scenarios eval/scenarios --time-scale 8 --reps 1 --corpus-dir fixtures/dev_corpus

eval-real:
	python -m eval.run_all --scenarios eval/scenarios_real_corpus --time-scale 8 --reps 1 --corpus-dir data/corpus

compare:
	python -m eval.compare --modes streaming,baseline --scenarios eval/scenarios --corpus-dir fixtures/dev_corpus --simulated-latency-ms 300

ablate:
	python -m eval.ablate --scenarios eval/scenarios --corpus-dir fixtures/dev_corpus

docker:
	docker compose up --build
