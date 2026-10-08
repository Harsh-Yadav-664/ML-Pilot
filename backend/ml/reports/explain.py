"""Explanations of a run (#63): real TreeSHAP values, and answers that come from stored records.

SHAP: LightGBM computes exact TreeSHAP contributions itself (``pred_contrib=True``, the algorithm
of Lundberg et al. 2018), so no extra package is needed. The contributions of a row plus the
base value add up to the model's raw score; a test checks that.

Answers: retrieval first. A question is matched to records by a deterministic lookup (feature
names, a few intents). The records are the answer; a model may reword them, and only if every
number and every identifier of its wording is in the records. Otherwise the records are shown
as they are, with a note. Nothing matched means "I don't have a record of that in this run."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

NO_RECORD = "I don't have a record of that in this run."
MAX_SHAP_ROWS = 2000
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?")
IDENTIFIER = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:_{1,2}[A-Za-z0-9]+)+\b")


# TreeSHAP ---------------------------------------------------------------------------------------
def contributions(model: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(per-feature contributions, base value) in raw-score (log-odds) units, one row per input."""
    raw = np.asarray(model.booster_.predict(frame, pred_contrib=True))
    return raw[:, :-1], raw[:, -1]


def shap_summary(
    model: Any, frame: pd.DataFrame, rows: np.ndarray, seed: int = 42
) -> dict[str, Any]:
    """Global importance on a sample of ``rows``: the mean absolute contribution per feature."""
    rows = np.asarray(rows)
    if len(rows) > MAX_SHAP_ROWS:
        rows = np.sort(np.random.default_rng(seed).choice(rows, MAX_SHAP_ROWS, replace=False))
    values, base = contributions(model, frame.iloc[rows])
    mean_abs = np.abs(values).mean(axis=0)
    total = float(mean_abs.sum()) or 1.0
    return {
        "method": "TreeSHAP (LightGBM pred_contrib), raw-score units",
        "n_rows": len(rows),
        "base_value": float(base[0]) if len(base) else 0.0,
        "mean_abs": {str(c): float(v) for c, v in zip(frame.columns, mean_abs, strict=True)},
        "share": {str(c): float(v / total) for c, v in zip(frame.columns, mean_abs, strict=True)},
    }


def top_reasons(model: Any, frame: pd.DataFrame, k: int = 3) -> list[list[tuple[str, float]]]:
    """For each row, the ``k`` features that pushed its score most, with the signed push."""
    values, _ = contributions(model, frame)
    names = [str(c) for c in frame.columns]
    out: list[list[tuple[str, float]]] = []
    for row in values:
        order = np.argsort(-np.abs(row))[:k]
        out.append([(names[i], float(row[i])) for i in order])
    return out


# Records ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Record:
    id: str  # run, feature:<name>, shap
    facts: dict[str, Any]

    def lines(self) -> list[str]:
        return [f"{k}: {_show(v)}" for k, v in self.facts.items() if v not in (None, "", [], {})]

    def text(self) -> str:
        return f"[{self.id}] " + "; ".join(self.lines())


def _show(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, dict):
        return ", ".join(f"{k} {_show(x)}" for k, x in v.items())
    if isinstance(v, list):
        return ", ".join(_show(x) for x in v)
    return str(v)


def build_corpus(
    run: dict[str, Any], features: list[dict[str, Any]], shap: dict[str, Any] | None
) -> list[Record]:
    """The records a question can be answered from. Facts only: no SQL text, no cell values."""
    corpus = [Record("run", run)]
    for f in features:
        corpus.append(Record(f"feature:{f['name']}", {k: v for k, v in f.items() if k != "name"}))
    if shap:
        top = sorted(shap["mean_abs"].items(), key=lambda kv: -kv[1])[:10]
        corpus.append(
            Record(
                "shap",
                {
                    "method": shap["method"],
                    "rows explained": shap["n_rows"],
                    "most influential features (mean absolute SHAP)": dict(top),
                },
            )
        )
    return corpus


Intent = Literal["debrief", "metrics", "importance", "rejected", "accepted"]
INTENTS: dict[Intent, tuple[str, ...]] = {
    "debrief": ("debrief", "summary", "summarise", "summarize", "what happened", "overview"),
    "metrics": ("test", "score", "metric", "auc", "accuracy", "result", "how good", "performance"),
    "importance": ("shap", "important", "importance", "matter", "influence", "driver", "top"),
    "rejected": ("reject", "rejected", "why not", "dropped", "refused", "veto", "failed"),
    "accepted": ("accepted", "kept", "champion", "helped", "gain"),
}


def retrieve(question: str, corpus: list[Record]) -> list[Record]:
    """Deterministic lookup: features named in the question, then the intents it expresses."""
    q = question.lower()
    named = [r for r in corpus if r.id.startswith("feature:") and r.id[8:].lower() in q]
    chosen: list[Record] = list(named)
    byid = {r.id: r for r in corpus}
    wants = {i for i, words in INTENTS.items() if any(w in q for w in words)}
    if "debrief" in wants:
        wants |= {"metrics", "importance", "accepted"}
    if "metrics" in wants and "run" in byid:
        chosen.append(byid["run"])
    if "importance" in wants and "shap" in byid:
        chosen.append(byid["shap"])
    for intent, status in (("accepted", "accepted"), ("rejected", "rejected")):
        if intent in wants and not named:
            chosen += [
                r
                for r in corpus
                if r.id.startswith("feature:") and str(r.facts.get("status", "")).startswith(status)
            ][:8]
    unique: dict[str, Record] = {}
    for r in chosen:
        unique.setdefault(r.id, r)
    return list(unique.values())


def allowed_tokens(records: list[Record]) -> tuple[set[str], set[str]]:
    """(numbers, identifiers) that appear in the records, the only ones an answer may use."""
    text = " ".join(r.text() for r in records)
    return set(NUMBER.findall(text)), set(IDENTIFIER.findall(text))


def violations(answer: str, records: list[Record]) -> list[str]:
    """Numbers or identifiers in ``answer`` that the records do not contain."""
    numbers, identifiers = allowed_tokens(records)
    bad = [n for n in NUMBER.findall(answer) if n not in numbers]
    bad += [i for i in IDENTIFIER.findall(answer) if i not in identifiers]
    return sorted(set(bad))


@dataclass
class Answer:
    text: str
    records: list[str] = field(default_factory=list)
    mode: Literal["records", "llm", "fallback", "no_record"] = "records"
    note: str | None = None


def records_answer(records: list[Record]) -> str:
    return "\n".join(r.text() for r in records)
