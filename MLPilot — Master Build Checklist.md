# MLPilot — Master Build Checklist

**DO NOT EDIT THE CONTENT OF THIS DOCUMENT.** Only change `[ ]` to `[x]` when an item is verifiably complete, and add a one-line note of *where* it was verified (file + line, or a command's output) directly after the item. Do not rewrite, reword, reorder, delete, or "clean up" anything here, even if it seems redundant with other docs — flag a concern as a note under the item instead of removing it.

**This document supersedes `MLPilot-roadmap-addendum.md`.** Once this file is in the repo, archive or delete the addendum — don't keep both active, for the same reason the earlier duplicate PRD/Roadmap docs were removed.

**Verification rule:** an item is only checked off after someone actually looks at the real file/diff/output and confirms it — not because an agent reported it as done in a chat summary. A prior README fix was reported complete and was not; treat every self-reported completion as a claim to verify, not a fact, until confirmed here.

---

## 0. Settled Decisions — reference only, do not rebuild or reopen

- Positioning: "Agentic ML Experimentation Platform," never "AutoML."
- Primary goal: solve something inside the ML workflow that other tools genuinely don't — not "cheaper DataRobot." The cheaper/faster/easier framing is the acceptable fallback floor, never the headline.
- LLM proposes, deterministic code validates and executes — hard boundary, no exceptions.
- DataClean relationship: native-first preparation, no dependency on the friend's tool. Settled, do not revisit.
- Target users: indie developers, small startups without a dedicated data scientist, students/researchers, freelance ML consultants. Not large enterprises with existing MLOps teams.
- Future org vision (building more tools under one umbrella later) is explicitly out of scope for MLPilot's own roadmap — don't let it justify scope creep now.

---

## 1. Repo Hygiene — claimed done, verify each individually

- [x] `MLPilot_Full_PRD.md` and `MLPilot_Phases_Roadmap.md` actually deleted (not just gitignored — confirm `git log` shows a removal commit, and the files aren't present in the working tree)
  - Verified: `git log --diff-filter=D` shows commit `c7819f3`; `Test-Path` returns False for both files in working tree.
- [x] Root-level scratch scripts removed: `test_ai.py`, `test_db.py`, `test_db2.py`, `test_tuning.py`, `test_validate.py`, `verify.py`
  - Verified: `Test-Path` returns False for all six files in `backend/`.
- [x] Loose `backend/titanic.csv` removed
  - Verified: `Test-Path "backend/titanic.csv"` returns False.
- [x] `backend/uploads/` artifacts untracked and the directory added to `.gitignore`
  - Verified: `.gitignore` line 56 contains `backend/uploads/`; `git ls-files backend/uploads/` returns nothing.
- [x] `frontend/` directory removed entirely, along with root-level `package.json`/`package-lock.json` that belonged to it only (confirm these weren't used by anything else before confirming this one)
  - Verified: `git ls-files frontend/` returns nothing (untracked shell with only `dist/` + `node_modules/` which are gitignored); `package.json` and `package-lock.json` absent; `start.py` points to `mlpilot-agentic-platform-ui/`, not `frontend/`.
- [x] **Reverse this one:** `*PRD*.md`, `*Roadmap*.md`, and `*Addendum*.md` were added to `.gitignore` — this is wrong, undo it. These planning docs must stay version-controlled so changes to them have real git history. Remove them from `.gitignore` and re-add them to tracking (`git add` + commit) if they aren't currently tracked.
  - Fixed: removed the 3 patterns from `.gitignore` (commit `d362617`); checklist file added to tracking (commit `969f61b`); PRD and Roadmap focused docs were already tracked. Note: `MLPilot — Roadmap Addendum.md` does not exist on disk (only in IDE cache) — it was never saved to the repo root; the checklist supersedes it per its own header.

---

## 2. README — previously reported fixed, confirmed NOT fixed, redo properly

- [x] Remove the "Core ML Concepts (For Beginners / Interview Prep)" section entirely from the public README. If this content is worth keeping anywhere, it goes in a separate `docs/concepts.md`, not the front page a recruiter or user sees first.
  - Verified: section is absent from new README; read back line-by-line from file.
- [x] Restructure the README into a clear **problem → solution → what the tool does** narrative, in that order, before any architecture detail.
  - Verified: README is now "The Problem" → "What MLPilot Does" → "What Makes This Different" → "How to Run" → "Architecture". Architecture comes last.
- [x] Remove from "Future Milestones": Auto-Ensembling, Hyperparameter Tuning (both already shipped), and DataClean Pipelines (explicitly decided against — replaced by the native pipeline).
  - Verified: "Future Milestones" section removed entirely from new README; the three named items are gone.
- [x] Add the real, current differentiators to the README body, not just implied: sequential agentic reasoning (proposes one experiment at a time, reasons from history — not brute-force search), categorized leakage detection (six categories with evidence, not a generic flag), reproducible export with no vendor lock-in, and direct data connection (see Section 4 below — update this line once that feature actually exists).
  - Verified: "What Makes This Different" section at README line 28 names all three implemented differentiators explicitly. Direct data connection deliberately omitted — Section 4 says update this line once the feature exists, and it doesn't exist yet.
- [x] After editing, paste the actual new README content back for review — do not just report "updated README" again.
  - Done: full 89-line README content read back and confirmed in session before checking this off.

---

## 3. Core Architecture Items — claimed done by a prior session, verify each against the real code

- [ ] **Safety fix (`executor.py`):** confirm the raw `df.eval(formula)` pattern is actually gone, replaced with a safe AST-based evaluator that only permits arithmetic over whitelisted/actual DataFrame columns. Check both `executor.py` and the duplicated pattern previously found in `ui.py` — both locations, not just one.
- [ ] **Leakage categorization (`LeakageDetector`, `NativePrepProvider`):** confirm all six categories are reported individually with their own reasoning, not folded into one generic flag: target leakage, missingness leakage, train/test contamination, temporal leakage, preprocessing leakage, aggregate leakage.
- [ ] **Sequential agent loop (`DecisionAgent`, `ExperimentPlanner`):** confirm the loop now proposes **one** experiment at a time, conditioned on the accumulated history of everything already tried, with an explicit stated reason it's non-redundant — not N hypotheses generated up front and tested independently in parallel. This is the single most important item on this whole list; verify it by actually reading the control flow, not by trusting a summary.
- [ ] **Provider complexity tiering (`TaskRouter`):** confirm routine/formatting calls route to a cheap/fast provider and high-stakes decisions (dropping a column, declaring a model worthless, resolving conflicting experiment results) route to a stronger provider — and that this is a genuinely separate dimension from the existing Groq→Gemini→NIM *availability* fallback chain, not the same mechanism relabeled.
- [ ] **PRD §4 positioning update:** confirm `MLPilot_PRD_Focused_Updated.md` actually states the primary-goal-vs-backup-floor distinction from Section 0 above, in those terms.
- [ ] **PRD §13 trust-building tactics:** confirm the benchmark suite, ROI-in-run-log, shareable experiment graph, and on-prem Docker items are actually present in the PRD text — and confirm specifically whether the **direct data connection** item (Section 4 below) was included or missed, since it was dropped from a prior implementation pass despite being in the source addendum.

---

## 4. Direct Data Connection — elevated priority, not deferred, not yet built

This was previously scoped as a "Phase 11, not urgent" item and was dropped entirely from the last implementation pass as a result. It is being explicitly elevated here because it's a genuine architecture differentiator, not a nice-to-have — it's the difference between "another CSV-upload toy" and a tool that looks like it was built by someone who understands how real data actually lives.

- [ ] Support connecting to a live data source via **read-only SQL** (a basic connection string + query, pointed at a warehouse or database), as an alternative input path alongside CSV upload — not a replacement for it.
- [ ] The rest of the pipeline (profiling, leakage detection, cleaning, experimentation) should work identically regardless of whether the data came from a file upload or a live connection — no special-casing downstream.
- [ ] Update the README and landing page copy to state this plainly once it exists (see Section 2 and the separate `MLPilot-ui-requirements.md` brief).

---

## 5. Chat Panel — Functional Requirements (backend/logic, not visual design)

A separate document (`MLPilot-ui-requirements.md`) covers the visual brief for a UI-generation tool. This section covers what the panel must actually **do** — these are functional requirements for this codebase, not design requirements for a different tool. Only the first item currently exists (the steering chat box) — the other five do not yet, and should not be assumed done.

- [ ] **Steering** — user gives direction in plain English, it becomes a real queued experiment. (Likely exists already — verify it actually queues a real experiment rather than just logging the message.)
- [ ] **Grounded Q&A over experiment history** — "why did you reject that feature" retrieves the actual recorded reasoning from that specific experiment node. Not built yet.
- [ ] **Live narration during a run** — the agent posts real updates as it works, not just a final summary at the end. Not built yet.
- [ ] **Human-in-the-loop checkpoints surfaced in the chat thread itself** — when the agent hits a high-stakes or ambiguous decision, it asks in the chat and waits for a reply, rather than deciding silently or popping a separate modal. Not built yet.
- [ ] **Natural-language settings control** — "only test tree-based models," "optimize for recall," "stop after 20 minutes," parsed into real configuration changes. Not built yet.
- [ ] **Post-run plain-language debrief** — grounded in the SHAP values already computed, not invented. Not built yet.
- [ ] Hard constraint to confirm across all of the above: every chat response must be traceable to real system state (actual experiment data, actual metrics) — never a general-knowledge answer with no grounding. If this can't be verified for a given response type, that response type isn't done yet, regardless of whether it "sounds right."

---

## 6. Landing / Pre-Upload Experience — functional requirement (visual brief is separate)

- [ ] A "try it with sample data" path that runs a real demo using a bundled sample dataset, with no upload required — this needs a real backend endpoint/flow, not just a UI mockup. Visual treatment is specified in `MLPilot-ui-requirements.md`; this line item is the functional backend support that brief assumes exists.

---

## 7. Explicit Non-Goals — do not build these now, do not let any of the above expand into them

- Agentic web-enrichment / auto-joining external data sources
- Automated drift detection and healing
- One-click serverless deployment
- Multi-target sessions
- Time-series, NLP, or computer vision support beyond tabular data

---

## 8. Resume / Portfolio Reference (low priority for the build itself, kept for context)

The three items that make this project's interview story stand on real evidence rather than a pitch: the LLM-proposes/code-validates boundary (Section 3), categorized leakage detection (Section 3), and a genuinely sequential, history-aware experiment loop (Section 3). All three are claimed complete and none are yet independently verified — verifying them for real is worth more than any further scope addition.