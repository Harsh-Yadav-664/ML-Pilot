# Quick start: your own PostgreSQL

This takes about ten minutes and ends with an evidence report for one prediction question on your database. Nothing is written to your database at any step.

You need: Python 3.13 or newer, a PostgreSQL database you can reach, and someone who may create a login role on it (your DBA, or you).

## 1. Create a read-only login role

MLPilot should read through a role that can `SELECT` the tables you choose and nothing else. The pilot kit has a script for that:

```bash
psql "postgresql://dba@db.example.com:5432/shop" \
     -v pilot_password='choose-a-long-random-password' \
     -v pilot_tables='public.customers, public.orders, public.order_items' \
     -f docs/pilot/readonly_role.sql
```

The script's header lists every option, and how to lock or drop the role afterwards. [What MLPilot reads, stores and shows a language model](pilot/data_handling.md) is the page to give your security team.

## 2. Install and connect

```bash
git clone https://github.com/Harsh-Yadav-664/ML-Pilot && cd ML-Pilot
python scripts/build_ui.py && python -m pip install build && python -m build --wheel --outdir dist backend
pip install dist/mlpilot-*.whl      # once a release is on PyPI: pip install mlpilot
export DB_PW='the password from step 1'
mlpilot connect --name shop --dialect postgres --host db.example.com --port 5432 \
  --database shop --user mlpilot_pilot --password-env DB_PW
mlpilot schema
```

`connect` tests the login and warns if the role could write. `schema` lists the tables, the event-time column MLPilot found in each, and the relationships (declared foreign keys plus inferred ones). If a time column is wrong, fix it in the web UI (`mlpilot ui`, Connect page) before going on: every feature is cut off by it.

## 3. Ask a question and read the task

```bash
mlpilot task draft --question "Which customers will stop ordering in the next 30 days?"
mlpilot task confirm
```

`draft` prints the task as sentences: who is predicted (the entity), what counts as a positive (the target), the horizon, and the cutoff dates, with the number of positives and the base rate per cutoff and the feasibility checks. Read it. If it is not the question you meant, change it in the task editor of the web UI, or write the YAML by hand ([task spec reference](task_spec.md)). `confirm` locks it.

## 4. Run, then read the report

```bash
mlpilot run
mlpilot report --out report.md
```

`run` builds the training table (each row only sees data from before its cutoff), trains a baseline, then tries features one at a time and keeps one only if it helps on later time periods. The test period is scored once, at the end. Without a language-model key the offline stub answers and the run says so: you then get the automatic baseline features only. To use a model, set one provider key (for example `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; the list is in `backend/.env.example`), or point at a local model as described in [data handling](pilot/data_handling.md#3-what-a-language-model-is-shown). At the default privacy level the model sees table and column names and aggregate statistics, never cell values.

`mlpilot run --max-rounds 5 --max-seconds 600 --max-cost-usd 1` caps the search.

## 5. Take the result with you

```bash
mlpilot export          # a zip: features as SQL and dbt, the model, task, report, score.py
mlpilot score -o ranked.csv   # rank the entities eligible now, with reasons
```

`score.py` in the bundle scores from your database without MLPilot installed; `python score.py --verify` recomputes the validation metric as a check.

## If something fails

Every failure is recorded with a message: `mlpilot run` prints the error and the run is stored as `failed`, and the web UI shows the run's event log. A blocked feasibility check (too few positives, a base rate near 0 or 1) stops the run with the numbers; `--override` runs anyway and records that you did. How the safeguards work: [safety model](safety.md).
