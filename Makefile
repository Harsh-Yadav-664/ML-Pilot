# `make benchmark DATASET=rel-f1` runs every method over every seed on a RelBench task, then writes
# benchmarks/RESULTS.md from the result files (issue #65). It needs `pip install -r
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
