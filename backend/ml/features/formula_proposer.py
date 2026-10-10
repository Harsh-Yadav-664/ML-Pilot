"""An LLM proposes one formula at a time for a single-table task; code checks it (#151).

The proposer of the run loop (``ml.agents.run_loop``) for a table with a target column, as
``FeatureProposer`` is for a relational task. It gives the loop the same thing, a
``ProposalRecord``: a feature that passed every check and is computed on all rows, or a rejection
with the stage that stopped it. Whether a computed feature helps is decided by the loop's paired
rule, never here and never by the model.

The model's answer is data. The formula is parsed and evaluated by the one safe evaluator
(``ml.features.safe_eval``, a whitelisted AST walk: no ``eval``, no ``exec``), in this order, and
the first failure ends the proposal:

1. **schema**: the answer has a name and a formula, the name is a new identifier;
2. **duplicate**: the same formula was already tried in this run;
3. **guard**: the formula passes the whitelist and uses columns the table has;
4. **execution**: it computes on the rows, is not constant on the training rows and is not an
   exact copy of a numeric column in use.

Formulas can use the columns of the prepared table and the features already adopted by the
champion, in the order they were added (the exported script rebuilds them in that order).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from pydantic import ValidationError

from ml.experiments.planner import ExperimentPlanner
from ml.features.baseline import BaselineResult, same_column
from ml.features.llm_sql import Attempt, Proposal, ProposalRecord, Stage, Status
from ml.features.safe_eval import InvalidFormula, evaluate, parse
from ml.features.table_baseline import TableTask

OBJECTIVE = "Maximize PR-AUC on held-out rows while preventing overfitting"
MAX_REASON = 300
HISTORY_ROWS = 12


def formula_key(formula: str) -> str:
    return "".join(formula.split())


class FormulaProposer:
    """Proposes formula features for one table, remembering what was tried."""

    def __init__(
        self,
        *,
        planner: ExperimentPlanner,
        profile: Any,
        task: TableTask,
        baseline: BaselineResult,
    ) -> None:
        self.planner = planner
        self.profile = profile
        self.target_column = task.target_column
        self.train = task.split.train
        # prepared columns in their own dtypes, plus the features the champion adopts
        self.work = task.features.copy()
        self.numeric: dict[str, pd.Series] = {
            str(c): self.work[c]
            for c in self.work.columns
            if pd.api.types.is_numeric_dtype(self.work[c])
        }
        self.names: set[str] = {str(c) for c in self.work.columns}
        self.tried: set[str] = set()
        self.rejected: list[dict[str, str]] = []
        self.records: list[ProposalRecord] = []
        # what the planner is told of earlier rounds: the facts the loop decided, nothing else
        self.history: list[dict[str, Any]] = [
            {
                "name": "baseline",
                "formula": None,
                "decision": "baseline",
                "val_pr_auc": round(float(baseline.metrics.get("pr_auc", 0.0)), 4),
            }
        ]

    # -- the loop's proposer interface ----------------------------------------------------------
    async def propose(self, remaining: int = 1, hints: Sequence[str] = ()) -> ProposalRecord:
        """One feature: ask the planner, check the answer, compute it."""
        try:
            answer = await self.planner.generate_next_hypothesis(
                profile=self.profile,
                target_column=self.target_column,
                objective=OBJECTIVE,
                history=self.history[-HISTORY_ROWS:],
            )
        except Exception as e:  # noqa: BLE001 - recorded as a failed proposal; the loop goes on
            record = ProposalRecord("invalid", "parse", [f"the planner failed: {e}"])
            record.llm = [{"decision_mode": "fallback", "error": str(e)}]
            self.records.append(record)
            return record
        meta = answer.get("llm") or {"decision_mode": "llm"}
        if meta.get("decision_mode") == "fallback":
            record = ProposalRecord("no_llm", None, ["no LLM answered; the baseline stands"])
            record.llm = [meta]
            self.records.append(record)
            return record
        record = self._evaluate(answer)
        record.llm = [meta]
        self.records.append(record)
        return record

    def adopt(self, record: ProposalRecord, importance: float = 0.0) -> None:
        """The proposal joined the champion: later formulas may use it as a column."""
        p = record.proposal
        if p is None or record.values is None:
            raise ValueError("only a proposal that was computed can be adopted")
        self.work[p.name] = record.values.to_numpy()
        self.numeric[p.name] = self.work[p.name]
        self.names.add(p.name)
        self.history.append(
            {
                "name": p.name,
                "formula": p.formula,
                "decision": "keep",
                "share_of_gain": round(importance, 3),
            }
        )

    def reject_for_gain(self, record: ProposalRecord, reason: str, stage: str = "gain") -> None:
        """A proposal passed every check but was not kept: later prompts hear why."""
        p = record.proposal
        if p is not None:
            self._rejected(p, stage, reason)

    # -- the checks ------------------------------------------------------------------------------
    def _rejected(self, p: Proposal, stage: str, reason: str) -> None:
        reason = reason if len(reason) <= MAX_REASON else reason[: MAX_REASON - 1] + "…"
        self.rejected.append({"name": p.name, "stage": stage, "reason": reason})
        self.history.append(
            {"name": p.name, "formula": p.formula, "decision": "reject", "reason": reason}
        )

    def _reject(
        self,
        answer: dict[str, Any],
        proposal: Proposal | None,
        status: Status,
        stage: Stage,
        reasons: list[str],
    ) -> ProposalRecord:
        if proposal is not None:
            self._rejected(proposal, stage, "; ".join(reasons))
        attempt = Attempt(answer, stage, reasons)
        return ProposalRecord(status, stage, reasons, proposal, attempts=[attempt])

    def _evaluate(self, answer: dict[str, Any]) -> ProposalRecord:
        fields = {
            "name": answer.get("name"),
            "rationale": answer.get("reason") or answer.get("rationale") or "",
            "formula": answer.get("formula"),
        }
        try:
            p = Proposal.model_validate(fields)
        except ValidationError as e:
            reasons = [
                f"{'.'.join(str(x) for x in err['loc']) or 'answer'}: {err['msg']}"
                for err in e.errors()
            ]
            return self._reject(answer, None, "invalid", "parse", reasons)
        assert p.formula is not None
        if not p.name.isidentifier():
            return self._reject(
                answer,
                p,
                "rejected_guard",
                "schema",
                ["name: use letters, digits and underscores, starting with a letter"],
            )
        if p.name in self.names:
            return self._reject(
                answer,
                p,
                "rejected_duplicate",
                "duplicate",
                [f"name: {p.name!r} is already a column or a feature in use"],
            )
        key = formula_key(p.formula)
        if key in self.tried:
            return self._reject(
                answer, p, "rejected_duplicate", "duplicate", ["the same formula was already tried"]
            )
        self.tried.add(key)
        try:
            expr = parse(p.formula, self.work.columns)
        except InvalidFormula as e:
            return self._reject(answer, p, "rejected_guard", "guard", [e.reason])
        try:
            values = evaluate(expr, self.work)
        except InvalidFormula as e:
            return self._reject(answer, p, "rejected_guard", "execution", [e.reason])
        if values.iloc[self.train].nunique(dropna=False) <= 1:
            return self._reject(
                answer,
                p,
                "rejected_guard",
                "execution",
                ["the feature has one value on every training row"],
            )
        for name, other in self.numeric.items():
            # exact copies only: a correlated feature may still add something, the loop decides
            if same_column(values, other, threshold=1.0) == "equal":
                return self._reject(
                    answer,
                    p,
                    "rejected_duplicate",
                    "duplicate_after_execution",
                    [f"equal to the column {name}"],
                )
        attempt = Attempt(answer, None, [])
        return ProposalRecord("proposed", None, [], p, attempts=[attempt], values=values)
