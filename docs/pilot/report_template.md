# Cover note for the MLPilot evidence report

The evidence report ([`GET /runs/{run}/report?format=html`](../features.md#evidence-report-60), or the button in the run view) holds the proof: the task, the data, the split, every feature and its SQL, the leakage checks, the limitations and the cost. It is written for a person who checks. This note is for the person who decides. It is one page, in business words, and it points to the report for everything it says.

**How to fill it in**

- Copy every number from the report's Summary table, **Test** column, and write the report section next to it. The Validation columns are higher than what to expect on new data, because the run used them to choose features; do not quote them as the result.
- If the report's Answer says "not available", send the note with section 2 reading "The model could not be scored on the test period" and the reason from the report. Do not fill the gap with the validation number.
- Do not change the bar that was agreed before the run (checklist: "success metric"). Say whether it was met.
- Delete the instruction lines (in italics) and any section that does not apply, but keep sections 4 and 5.

---

## <Customer>: <question in one sentence>, pilot result

Prepared by <name>, <date>. Run <run id> in the report; data as of <date of the data version>.

### 1. The question we answered

*Copy "Question" from the report's Summary.*

We looked at <who: for example, every customer who ordered in the 90 days before a given date> and asked: <what: for example, will this customer place no order in the next 30 days?> The answer is a ranking: every <customer> gets a position from most to least likely.

### 2. What we found

*Use the Test column. Write the numbers the agreed bar was about.*

On a period the model never saw while it was being built (<test period start> to <end>, <n rows> <customers> scored):

- On average, <base rate>% of <customers> <did the outcome>. That is what you would hit by picking at random.
- Among the 10% the model ranks highest, <precision at 10%>% <did the outcome>: <lift at 10%> times the random rate. (Report: Summary, "Precision" and "Lift" rows.)
- Those 10% contain <recall at 10%>% of everyone who <did the outcome>. (Report: Summary, "Recall" row.)
- *If the call list is smaller, add the 1% or 5% rows.*

**Against the bar we agreed:** you said <the agreed number>. The result is <above / below / close to> it, because <one honest sentence>.

*Only if the customer asked for money terms, and only with their own figures:* If you contact the top <N> and each saved <customer> is worth <value, supplied by you> and a contact costs <cost, supplied by you>, the arithmetic is <show it>. These two figures are your assumptions, not results of the model.

### 3. How we know it is not a fluke or a leak

- **Tested on later dates.** The model learned from earlier periods, was checked on a later one, and was scored on the latest one once, after every decision was made. No row used information from after its own cutoff date. (Report: Task, Validation.)
- **Each feature earned its place.** The model started from <n> automatic features. We tried <n proposed> more ideas; <n kept> improved the result by more than the margin and were kept, the rest were not. (Report: Features.) The most useful ones, in plain words: <the report's description of the top kept features>.
- **Leakage checks.** <Copy the Leakage and safety result: no table or column flagged, or what was flagged and what you did.>
- **Read-only access.** <Copy the report's "Database role" line: checked, and it cannot write.> (Report: Leakage and safety.)

### 4. What this does not tell you

*Keep all of these. Add the facts the report lists under "From the facts of this run".*

- It ranks; its scores were not checked for calibration, so a score is not a probability. A score of 0.8 is not an 80% chance.
- It shows patterns in past data, not causes. It does not say why a <customer> is at risk, and acting on the ranking may not change the outcome.
- It was built on data up to <date>. Nothing monitors it: if your business or your data changes, it can stop working without any warning.
- It answers only this question: a different horizon, outcome or group needs a new run.
- *Include if the report says so:* the test period had only about <n> positive cases, so the test numbers are noisy.
- *Include if the report says so:* the rate of the outcome moved from <low>% to <high>% between dates; a model that ranks well at one rate may rank differently at another.
- *Choose one:* "<Name, role> at <customer> confirmed the task on <date>." or "It has not been reviewed by a person at <customer>."

### 5. What we used, and where it went

- **Read from your database:** <list of tables>, with a read-only role created by your team. Rows after <cutoff> were not used for training.
- **What a language model saw:** <privacy level in words: only table and column names and types / plus summary statistics such as counts, averages, minimum and maximum values / ...>. It was not shown your rows. <Provider and model, or "a model running inside your network".> (Report: Cost and reproducibility lists every call; the privacy setting is in the Data section.)
- **Kept by us:** <a copy of those tables up to the cutoff, the prompts, the model> on <machine>, until <deletion date>. We will confirm the deletion in writing.
- Details and the tests behind each statement: [data handling](data_handling.md).

### 6. What you can do with it

- **Use the list now:** a ranked list of <customers> as of <date> is attached (<file>, <n> rows, with the three strongest reasons per row). It is only as current as that date.
- **Run it yourselves:** the export bundle has the features as SQL and as a dbt project, the model file, and a scoring script that reads your database read-only. When we ran its check on <date> on <whose machine>, it reproduced the validation figure in the report (<validation PR-AUC from the report>). The bundle holds the model the validation numbers belong to.
- **Next step, if you want one:** <for example: a second question on the same data; a month of tracking who was contacted and what happened; weekly scoring>.

### 7. What we need from you

- <Decision: continue / stop / change the question, by date.>
- <Confirmation that the database role was revoked on date.>

---

Attached: evidence report (HTML), <ranked list>, <export bundle>.
