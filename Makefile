# `make benchmark DATASET=rel-f1` runs every method over every seed on a RelBench task, then writes
# benchmarks/results/RESULTS.md from the result files (issue #65; the published table is the
# hand-written benchmarks/RESULTS.md, issue #66). It needs `pip install -r
# benchmarks/requirements.txt` and downloads the dataset on first use. Method d (strong model) is
# skipped, and says so, unless MLPILOT_BENCH_STRONG_MODEL=provider:model and that provider's API key
# are set; method c uses MLPILOT_BENCH_CHEAP_MODEL the same way and runs on the offline stub without.
PYTHON ?= python
DATASET ?= rel-f1
TASK ?= driver-dnf
METHODS ?= a b c d
SEEDS ?= 0 1 2
BENCH = $(PYTHON) benchmarks/relbench/run.py

.PHONY: benchmark
benchmark:
	@fail=0; \
	for m in $(METHODS); do for s in $(SEEDS); do \
		$(BENCH) --dataset $(DATASET) --task $(TASK) --method $$m --seed $$s || fail=1; \
	done; done; \
	$(BENCH) --aggregate || fail=1; \
	exit $$fail

# `make benchmark-openml` is the single-table sanity benchmark (issue #67): LightGBM, MLPilot and
# AutoGluon on OpenML binary-classification datasets, every seed, then benchmarks/openml/RESULTS.md
# from the result files. It needs `pip install -r benchmarks/openml/requirements.txt` and downloads the
# datasets on first use. MLPilot runs on the offline stub (no feature is proposed) unless
# MLPILOT_BENCH_CHEAP_MODEL=provider:model and that provider's API key are set.
OPENML_DATASETS ?= 37 31 29 1464 1494 1480 1489 44 1067 1487
OPENML_METHODS ?= lightgbm mlpilot autogluon
OPENML_SEEDS ?= 0 1 2
AG_SECONDS ?= 90
AG_PRESET ?= high_quality
OPENML = $(PYTHON) benchmarks/openml/openml_run.py

.PHONY: benchmark-openml
benchmark-openml:
	@fail=0; \
	for d in $(OPENML_DATASETS); do for m in $(OPENML_METHODS); do for s in $(OPENML_SEEDS); do \
		$(OPENML) --dataset $$d --method $$m --seed $$s --ag-seconds $(AG_SECONDS) --ag-preset $(AG_PRESET) || fail=1; \
	done; done; done; \
	$(OPENML) --aggregate || fail=1; \
	exit $$fail
