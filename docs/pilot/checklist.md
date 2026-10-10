# Pilot checklist

One page, three stages. Tick a box only when you have seen it, not when someone said it was done. Where a step is something MLPilot checks for you, the line says where to look. The data questions a customer asks are answered in [`data_handling.md`](data_handling.md); the role is created with [`readonly_role.sql`](readonly_role.sql).

## Before: agree the question and the access

- [ ] **The question, in one sentence, with its three parts:** who (the entity: customer, account, ...), what counts as the outcome (for example "places no order"), and how long to wait for it (the horizon: 30 days). Example: "Which customers who ordered in the last 90 days will place no order in the next 30 days?"
- [ ] **The outcome is recorded in the database.** Someone has pointed at the table and column where it can be seen after the fact. If the outcome is not stored (a cancellation that only a sales rep knows about), stop: the model has nothing to learn from.
- [ ] **There is enough history.** Several cutoff dates' worth of history plus the horizon, and at least 50 past cases of the outcome in the training period and 20 each in the validation and test periods; below that MLPilot blocks the run (see "Feasibility is ok or understood" below). Ask now.
- [ ] **A success metric the business agreed to, in business words, before any result.** For example: "of the 10% of customers the model ranks highest, at least 3 in 10 should really stop ordering (the usual rate is 1 in 10)", or "the call list of 500 should hold twice as many leavers as 500 random customers". Write down the number that means "worth continuing" and the number that means "not worth it". The report gives precision, recall and lift at the top 1%, 5% and 10% of the ranking, so use one of those.
- [ ] **What the business will do with the answer** (a call list, an offer, a review), and roughly how many entities they can act on. This sets which top-N cut matters.
- [ ] **A read-only database role.** The customer's DBA runs `readonly_role.sql` with the list of tables MLPilot may read. Check: the DBA sent the output of the last two queries of the script (the role's settings and the granted tables) and the table list is the one you agreed. Not "all tables".
- [ ] **Columns MLPilot should not see.** Personal data (email, name, phone, address) is either left out of the grants (column-level `GRANT SELECT (...)`, see the comment in the script) or marked "never send" in MLPilot. Write the list down.
- [ ] **A privacy level chosen and written down.** The default sends names, types and aggregate statistics (including real min and max values) to the language model, never rows. Level 1 sends names and types only. A local model sends nothing outside the customer's network. See section 3 of [`data_handling.md`](data_handling.md). If the customer's answer is "nothing may leave", use a local model and say so in the report.
- [ ] **Where MLPilot runs and who can reach it.** On a machine the customer approves, reachable only by the people who need it (it listens on `127.0.0.1` by default). The disk is encrypted, because the project folder holds a copy of the data.
- [ ] **Network path to the database** (firewall, `pg_hba.conf`, TLS) arranged by the customer. Prefer a read replica.
- [ ] **Who deletes what, and when.** Agree the date by which the copy of the data is deleted (see "After").

## During: build and sign off the task

- [ ] **Connect with the new role** (Connect page). The connection test must say the role cannot write. If it shows the red "this role can write" warning, stop and fix the role.
- [ ] **The schema graph is right.** Look at the tables, the relationships (declared and inferred) and the event-time column of each table. Fix wrong ones with the overrides. Mark tables that hold the current state only (a `status` table with just `updated_at`) because they leak the future.
- [ ] **Ask the question, or build the task by hand** (Ask a prediction question / Task editor). Read the task back "in words" with the customer. Check the entity, who is eligible, the outcome, the horizon and the cutoff dates.
- [ ] **A person at the customer signs off the task spec.** Confirm the task only after they have read the words and the label counts per cutoff. A confirmed task is never changed; editing it saves a new version.
- [ ] **Feasibility is ok or understood.** Open the feasibility result in the task preview. "Block" (fewer than 50 positives in training, or fewer than 20 in validation or test) means do not start; get more history or a wider question. "Warn" goes into the cover note. An override is recorded in the run, so use it only with a reason written down.
- [ ] **Start the run** with a budget you agreed (cost, time, number of proposals) and the approval mode you want. With "approve each feature", someone who knows the data reads each proposal and its SQL.
- [ ] **Watch for leakage.** The run view and the report list tables that look like the current state and columns that may be overwritten later. A feature that predicts the outcome suspiciously well is a reason to look, not to celebrate.

## After: report, handover, deletion

- [ ] **Read the evidence report together** (run view > report, or `GET /runs/{run}/report?format=html`). Go through, in order: the Answer (test column), the Data section (what period, which tables, which rows were left out), Validation, the features kept and the ones not kept, Leakage and safety, Limitations.
- [ ] **Compare the test numbers with the success metric agreed before.** Say plainly whether the bar was met. Do not change the bar afterwards.
- [ ] **Write the cover note** from [`report_template.md`](report_template.md), with numbers copied from the report, not from memory.
- [ ] **Export handed over** (`GET /runs/{run}/export`): features as SQL and as a dbt project, the model file, the task, the manifest, the offline report and `score.py`. Someone at the customer has run `score.py --verify` against their database and the validation PR-AUC matches the report. The bundle holds the model the validation numbers belong to.
- [ ] **A scoring date is agreed** if they want a list now (Score panel in the run view): the cutoff, and who receives the CSV, which holds one row per entity.
- [ ] **What was not done is written down:** no monitoring for drift, no promise that the model still holds after the data changes, scores are rankings not probabilities (the report says so too).
- [ ] **Revoke the role.** The DBA runs `readonly_role.sql` with `-v revoke=drop` (or `revoke=lock` to pause) in every database where it was granted, and confirms the role no longer logs in.
- [ ] **Delete the data.** Delete the saved connection in MLPilot, then the project folder `backend/data/projects/<project id>/` (snapshot, prompts, scores, models), and, if no other project uses it, `backend/mlpilot.db`. Deleting the project in the API does not remove the folder; see section 2 of [`data_handling.md`](data_handling.md). Record the date and who did it.
- [ ] **Ask the customer for one line of feedback** on what they would act on, and whether they would pay for a second question.
