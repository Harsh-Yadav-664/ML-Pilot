# Benchmark results

One RelBench task so far: **rel-f1 / driver-dnf** (RelBench's label for whether a driver does not
finish a race in the window after the cutoff; metric ROC-AUC, higher is better). Labels, split and
metric are RelBench's own. Nothing here says anything about other tasks.

On this task MLPilot does not beat the published numbers, and its feature search (method b) does
not beat LightGBM on the entity table alone. Both are in the table below.

## Ours

Run by the [Benchmark workflow](../.github/workflows/benchmark.yml) on `main`, commit `f0b996d`:
**[CI run 38138813383](https://github.com/Harsh-Yadav-664/ML-Pilot/actions/runs/38138813383)**
(GitHub-hosted `ubuntu-latest` runner, RelBench 3.0.2, LightGBM 4.7.0, Python 3.13.16, about 6
minutes for everything). The table below is copied from that run's log, where `run.py --aggregate`
prints it from the per-run result files. Test ROC-AUC is RelBench's own evaluator on the official
test labels, read once per run; mean ± sample standard deviation over seeds 0, 1, 2.

Reproduce (the same commands the workflow runs):

```bash
pip install -r benchmarks/requirements.txt
make benchmark DATASET=rel-f1 TASK=driver-dnf METHODS="a b" SEEDS="0 1 2"
# or, in GitHub: Actions > Benchmark > Run workflow (defaults are rel-f1, driver-dnf)
```

`make benchmark` writes its table to `benchmarks/results/RESULTS.md` and one JSON per run next to it.
Those files are git-ignored; this file is the published copy. The CI run also uploads them as the
`relbench-results` artifact (GitHub keeps artifacts for a limited time).

```text
| Method | Test ROC-AUC | Val | Seeds (test per seed) | Wall time | LLM |
|---|---|---|---|---|---|
| a: LightGBM on the entity table's columns | 0.6148 ± 0.0014 | 0.5225 ± 0.0057 | 3 (0: 0.6144, 1: 0.6138, 2: 0.6164) | 2 s |    |
| b: DFS features + LightGBM | 0.6137 ± 0.0165 | 0.7956 ± 0.0090 | 3 (0: 0.6081, 1: 0.6323, 2: 0.6008) | 35 s |    |
| c: MLPilot agent, cheap model | 0.6137 ± 0.0165 | 0.7956 ± 0.0090 | 3 (0: 0.6081, 1: 0.6323, 2: 0.6008) | 37 s |   no LLM answered (offline stub): no feature was proposed |
| d: MLPilot agent, strong model | skipped: no strong model: set MLPILOT_BENCH_STRONG_MODEL (provider:model) and that provider's key | | | | |
```

Method notes:

- **a** trains LightGBM on the columns of the entity table (drivers) only: no history, and ids,
  timestamps and high-cardinality text columns are not used.
- **b** is MLPilot's baseline: deep feature synthesis over the related tables, each feature read
  through the point-in-time guard so a row only sees data from before its own cutoff, then
  LightGBM. No LLM.
- **c is not an agent result.** The workflow ran without a model, so the offline stub answered, no
  feature was proposed, and c is b run again: the identical numbers are the proof of that. Method c
  needs `cheap_model` (`provider:model`) and that provider's API key as a repository secret.
- **d was skipped** (no `strong_model`, no key). It is not in the README table.
- Cost and tokens: a and b call no LLM, so there is no LLM cost. Cost for c and d is **not
  measured**; the harness records the gateway's estimate (it counts words, so it is not a bill)
  only when a model answers.
- Val is the same evaluator on the validation rows. MLPilot stops its model early on them, so val is
  not a held-out number. Val is much higher than test for b (0.7956 against 0.6137); we have not
  investigated why.
- The split is RelBench's own temporal split: validation rows from 2005-01-01, test rows from
  2010-01-01 (11411 train, 566 validation and 702 test rows in the log).

## Where MLPilot loses or ties

- **b against a: a tie, with a lot of noise.** 0.6137 ± 0.0165 against 0.6148 ± 0.0014. The feature
  search did not help on test for driver-dnf. Its three seeds range from 0.6008 to 0.6323, so one
  seed alone could have shown a gain or a loss.
- **Against every quoted number below, ours is lower**, including the quoted LightGBM (0.6856). Our
  method a is the same kind of model as that LightGBM and scores about 0.07 less. We have not
  checked how the paper's LightGBM was set up, so we cannot say why. Do not read the gap between
  ours and quoted as a like-for-like comparison.
- **The agent with a real model is untested here**, so there is no result for the part of MLPilot
  that is supposed to add value.

## Quoted (not run by us)

These are the authors' numbers for the same task, read from their papers. They use the authors'
own setups, models, seeds, compute and (for the LLM systems) LLM and budgets. We did not re-run
anything. Papers print AUROC multiplied by 100; here it is divided by 100 and the printed value is
in brackets.

| Method | Test ROC-AUC | Source |
|---|---|---|
| LightGBM (RelBench baseline) | 0.6856 (68.56); 68.56 ± 3.89 over 5 runs | [RelBench paper, arXiv 2407.20060](https://arxiv.org/abs/2407.20060), Table 3 and Table 6 (appendix B.1), driver-dnf Test |
| RDL (GraphSAGE model of the RelBench paper) | 0.7262 (72.62); 72.62 ± 0.27 over 5 runs | same paper, Table 3 and Table 6, driver-dnf Test. [RelAgent](https://arxiv.org/abs/2605.07840) Table 2 prints the same 72.62 under the name "GraphSAGE" |
| RelAgent (LLM agent that writes SQL features, GPT-5.2) | 0.7834 (78.34) | [RelAgent, arXiv 2605.07840](https://arxiv.org/abs/2605.07840), Table 2, dnf column. Five search rollouts, the one with the best validation score is reported |
| KumoRFM-2 | 0.8459 (84.59) | [KumoRFM-2, arXiv 2604.12596](https://arxiv.org/abs/2604.12596), Table 3, dnf column. We did not check which mode (few-shot or fine-tuned) the cell is. [RelAgent](https://arxiv.org/abs/2605.07840) Table 2 prints the same 84.59 as "KumoRFM-v2" |
| RelBench v2 | not verified | [RelBench v2, arXiv 2602.12606](https://arxiv.org/abs/2602.12606): no driver-dnf number found in the text we read |

How these were read: from the arXiv HTML/PDF text of each paper on 2026-10-11, the dnf column of the
named table. Two papers (RelAgent and KumoRFM-2) print the same LightGBM 68.56, GraphSAGE/RDL 72.62
and KumoRFM 84.59, which is a cross-check on the first two. Check the cell against the PDF before
you quote it elsewhere.
