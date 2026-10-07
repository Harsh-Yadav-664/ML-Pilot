"""The honest baseline (#55): DFS features, filtered, then LightGBM on the temporal split.

``build_baseline`` takes the label rows, the tables of a data version and the schema graph, and

1. generates candidate features (``ml.features.dfs``), best first;
2. computes each one by running its SQL on the tables with the label rows as ``__labels``
   (``ml.tasks.pit_verify.run_feature``), and keeps it unless, on the training rows only, it is
   constant, nearly constant (one value in more than 99% of rows) or a duplicate (equal to, or
   correlated above 0.99 with, a feature already kept). It stops at the cap (default 300);
3. runs the leakage scan on the attributes of the entity row, which are the one kind of feature
   the cutoff guard cannot vouch for (a column such as ``is_churned`` is filled in after the
   event it describes) and drops what it flags;
4. trains LightGBM on the training rows, stops early on the validation rows, and scores the
   validation rows.

The test rows are never read: the test set is scored once per run, at its end. This function
returns the validation score only. It is the bar that LLM-proposed features have to beat.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

from ml.data.schema_graph import SchemaGraph, type_kind
from ml.features.dfs import DEFAULT_MAX_FEATURES, TOP_CATEGORIES, Candidate, Skipped, generate
from ml.metrics.ranking import ranking_metrics
from ml.tasks.pit_guard import CUTOFF, ENTITY
from ml.tasks.pit_verify import run_feature
from ml.tasks.spec import TaskSpec
from ml.validation import leakage
from ml.validation.splits import TemporalSplit, TemporalSplitPlan, make_temporal_splits

NEAR_CONSTANT_SHARE = 0.99
DUPLICATE_CORRELATION = 0.99
SAMPLE_ROWS = 5000
MAX_CATEGORY_LEVELS = 50
SEED = 42
LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}
EARLY_STOPPING_ROUNDS = 50


class BaselineError(ValueError):
    """The baseline cannot be built: no usable label rows, or a split without both classes."""


@dataclass
class Dropped:
    name: str
    group: str
    reason: str  # constant, near_constant, duplicate, leakage_scan, over_cap
    detail: str = ""


@dataclass
class KeptFeature:
    candidate: Candidate
    importance: float  # share of the total gain of the model


@dataclass
class BaselineResult:
    features: list[KeptFeature]
    dropped: list[Dropped]
    skipped_tables: list[Skipped]
    candidates: int
    metrics: dict[str, float]
    split: dict[str, Any]
    params: dict[str, Any]
    seed: int
    engine: str
    engine_version: str
    seconds: float
    model: Any = field(repr=False, default=None)

    def dropped_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.dropped:
            counts[d.reason] = counts.get(d.reason, 0) + 1
        return counts


def flatten_graph(graph: SchemaGraph, entity_table: str) -> tuple[SchemaGraph, str]:
    """The graph with every table keyed by its bare name, which is how the tables of a data
    version are registered when a feature runs. Raises if two schemas share a table name."""
    names = [t.name for t in graph.tables]
    clash = sorted({n for n in names if names.count(n) > 1})
    if clash:
        raise BaselineError(f"table names are not unique across schemas: {clash}")
    rename = {t.key: t.name for t in graph.tables}
    if entity_table not in rename:
        raise BaselineError(f"the entity table {entity_table!r} is not in the schema graph")
    tables = [t.model_copy(update={"key": t.name}) for t in graph.tables]
    edges = [
        e.model_copy(update={"from_table": rename[e.from_table], "to_table": rename[e.to_table]})
        for e in graph.edges
        if e.from_table in rename and e.to_table in rename
    ]
    return graph.model_copy(update={"tables": tables, "edges": edges}), rename[entity_table]


def typed_times(tables: dict[str, pa.Table], graph: SchemaGraph) -> dict[str, pa.Table]:
    """Tables whose event-time column holds text (a SQLite snapshot keeps times as delivered)
    get a real timestamp column, so the features can compare it with the cutoff."""
    out = dict(tables)
    for table in graph.tables:
        data = out.get(table.name)
        column = table.time_column
        if data is None or column is None or column not in data.column_names:
            continue
        field = data.schema.field(column)
        if not (pa.types.is_string(field.type) or pa.types.is_large_string(field.type)):
            continue
        try:
            stamps = pd.to_datetime(data.column(column).to_pandas(), utc=True).dt.tz_localize(None)
        except (ValueError, TypeError) as e:
            raise BaselineError(
                f"{table.name}.{column} is the event time but holds text that is not a date: {e}"
            ) from None
        index = data.column_names.index(column)
        out[table.name] = data.set_column(index, column, pa.Array.from_pandas(stamps))
    return out


def restrict_graph(graph: SchemaGraph, keys: set[str]) -> tuple[SchemaGraph, list[Skipped]]:
    """The graph without the tables that are not in the data version, and why they are gone."""
    gone = [t.key for t in graph.tables if t.key not in keys]
    tables = [t for t in graph.tables if t.key in keys]
    edges = [e for e in graph.edges if e.from_table in keys and e.to_table in keys]
    skipped = [Skipped(k, "not in the data version") for k in gone]
    return graph.model_copy(update={"tables": tables, "edges": edges}), skipped


def top_values_from(
    tables: dict[str, pa.Table], graph: SchemaGraph
) -> dict[tuple[str, str], list[str]]:
    """The most frequent values of each low-cardinality text column. Used in SQL, never in a prompt."""
    out: dict[tuple[str, str], list[str]] = {}
    for table in graph.tables:
        data = tables.get(table.name)
        if data is None:
            continue
        names = {c.name for c in table.columns if type_kind(c.type) == "text"}
        for column in (n for n in data.column_names if n in names):
            counts = data.column(column).to_pandas().dropna().astype(str).value_counts()
            if 0 < len(counts) <= MAX_CATEGORY_LEVELS:
                out[(table.key, column)] = [str(v) for v in counts.index[:TOP_CATEGORIES]]
    return out


def _naive(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def _frame_column(values: pd.Series) -> pd.Series:
    """A feature as a model column: numbers as floats, booleans as 0/1, text as categories."""
    if pd.api.types.is_bool_dtype(values):
        return values.astype("float64")
    if pd.api.types.is_numeric_dtype(values):
        return values.astype("float64")
    return values.astype("string").astype("category")


def _same_column(a: pd.Series, b: pd.Series) -> str | None:
    if isinstance(a.dtype, pd.CategoricalDtype) or isinstance(b.dtype, pd.CategoricalDtype):
        equal = (
            a.astype("string").fillna("\0").to_numpy() == b.astype("string").fillna("\0").to_numpy()
        )
        return "equal" if bool(equal.all()) else None
    x, y = a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64")
    if np.array_equal(x, y, equal_nan=True):
        return "equal"
    both = ~(np.isnan(x) | np.isnan(y))
    if both.sum() < 30 or (np.isnan(x) != np.isnan(y)).any():
        return None
    if x[both].std() == 0 or y[both].std() == 0:
        return None
    r = float(np.corrcoef(x[both], y[both])[0, 1])
    return f"correlation {r:.3f}" if abs(r) > DUPLICATE_CORRELATION else None


def _degenerate(values: pd.Series) -> str | None:
    counts = values.astype("string").fillna("\0").value_counts()
    if len(counts) <= 1:
        return "constant"
    share = float(counts.iloc[0]) / float(counts.sum())
    if share > NEAR_CONSTANT_SHARE:
        return f"near_constant: {share:.1%} of training rows have one value"
    return None


def build_baseline(
    spec: TaskSpec,
    graph: SchemaGraph,
    tables: dict[str, pa.Table],
    labels: pd.DataFrame,
    plan: TemporalSplitPlan,
    *,
    max_features: int = DEFAULT_MAX_FEATURES,
    seed: int = SEED,
) -> BaselineResult:
    """DFS features and a LightGBM model on the temporal split; see the module docstring."""
    started = time.monotonic()
    need = {ENTITY, CUTOFF, "label", "label_window_end"} - set(labels.columns)
    if need:
        raise BaselineError(f"the label rows need the columns {sorted(need)}")
    frame = labels[[ENTITY, CUTOFF, "label", "label_window_end"]].copy()
    frame[CUTOFF] = _naive(frame[CUTOFF])
    frame["label_window_end"] = _naive(frame["label_window_end"])
    # the order of the rows from the database is not defined; a fixed order makes a run repeatable
    frame = frame.sort_values([CUTOFF, ENTITY], kind="stable").reset_index(drop=True)
    if frame.duplicated([ENTITY, CUTOFF]).any():
        raise BaselineError("the label rows have more than one row for an entity and cutoff")
    split = make_temporal_splits(frame[CUTOFF], frame["label_window_end"], plan)
    y = frame["label"].astype(int).to_numpy()
    for part, idx in (("training", split.train), ("validation", split.val)):
        if len(set(y[idx])) < 2:
            raise BaselineError(f"the {part} rows have only one class; there is nothing to learn")

    graph, entity_table = flatten_graph(graph, spec.entity.table)
    tables = typed_times(tables, graph)
    generated = generate(
        graph,
        entity_table,
        entity_created_at=spec.entity.created_at,
        top_values=top_values_from(tables, graph),
    )
    keys = frame[[ENTITY, CUTOFF]]
    index = pd.MultiIndex.from_frame(keys)
    rng = np.random.default_rng(seed)
    sample = np.sort(
        rng.choice(split.train, size=min(SAMPLE_ROWS, len(split.train)), replace=False)
    )
    kept: dict[str, pd.Series] = {}
    by_name: dict[str, Candidate] = {}
    dropped: list[Dropped] = []
    for cand in generated.candidates:
        if len(kept) >= max_features:
            dropped.append(Dropped(cand.name, cand.group, "over_cap", f"cap {max_features}"))
            continue
        result = run_feature(cand.sql, tables, keys, graph)
        values = result.set_index([ENTITY, CUTOFF])["value"].reindex(index)
        column = _frame_column(values.reset_index(drop=True))
        problem = _degenerate(column.iloc[split.train])
        if problem:
            reason, _, detail = problem.partition(": ")
            dropped.append(Dropped(cand.name, cand.group, reason, detail))
            continue
        twin = next(
            (
                (name, why)
                for name, other in kept.items()
                if (why := _same_column(column.iloc[sample], other.iloc[sample]))
            ),
            None,
        )
        if twin:
            dropped.append(Dropped(cand.name, cand.group, "duplicate", f"{twin[1]} with {twin[0]}"))
            continue
        kept[cand.name] = column
        by_name[cand.name] = cand

    _drop_leaky_attributes(spec, kept, by_name, dropped, frame["label"], split)
    if not kept:
        raise BaselineError("no feature survived the filters")

    features = pd.DataFrame(kept)
    model = lgb.LGBMClassifier(
        **LGBM_PARAMS, random_state=seed, deterministic=True, force_row_wise=True, verbose=-1
    )
    model.fit(
        features.iloc[split.train],
        y[split.train],
        eval_set=[(features.iloc[split.val], y[split.val])],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    score = np.asarray(model.predict_proba(features.iloc[split.val]))[:, 1]
    metrics = ranking_metrics(y[split.val], score, absolute_k=(100,))
    metrics["n_train"] = float(len(split.train))
    metrics["n_val"] = float(len(split.val))
    metrics["best_iteration"] = float(model.best_iteration_ or LGBM_PARAMS["n_estimators"])
    gain = model.booster_.feature_importance(importance_type="gain").astype("float64")
    total = float(gain.sum()) or 1.0
    ranked = sorted(zip(features.columns, gain / total, strict=True), key=lambda p: -p[1])
    return BaselineResult(
        features=[KeptFeature(by_name[name], float(share)) for name, share in ranked],
        dropped=dropped,
        skipped_tables=generated.skipped,
        candidates=len(generated.candidates),
        metrics={k: float(v) for k, v in metrics.items()},
        split=split.summary(),
        params={**LGBM_PARAMS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
        seed=seed,
        engine="lightgbm",
        engine_version=lgb.__version__,
        seconds=time.monotonic() - started,
        model=model,
    )


def _drop_leaky_attributes(
    spec: TaskSpec,
    kept: dict[str, pd.Series],
    by_name: dict[str, Candidate],
    dropped: list[Dropped],
    label: pd.Series,
    split: TemporalSplit,
) -> None:
    """Scan the entity-row attributes on the training rows; drop the columns the scan flags."""
    names = [n for n, c in by_name.items() if c.group == "attribute" and n in kept]
    if not names:
        return
    table = pd.DataFrame({n: kept[n].iloc[split.train].reset_index(drop=True) for n in names})
    for n in names:  # the scan wants plain columns
        if isinstance(table[n].dtype, pd.CategoricalDtype):
            table[n] = table[n].astype("string")
    table[spec.name] = label.iloc[split.train].reset_index(drop=True).to_numpy()
    findings = leakage.scan(
        table,
        spec.name,
        checks=(leakage.NameTokenCheck, leakage.TargetCopyCheck, leakage.SingleFeatureCheck),
    )
    flagged = leakage.flagged_columns(findings)
    for f in findings:
        if f.column in flagged and f.column in kept:
            cand = by_name.pop(f.column)
            del kept[f.column]
            dropped.append(Dropped(f.column, cand.group, "leakage_scan", f.explanation))
