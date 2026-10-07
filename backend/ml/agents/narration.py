"""Say what a run is doing, in words, from its events (#59).

``narrate`` fills a fixed sentence for each event type from the event's own payload. It never
rounds a number differently from the text it is shown in and never adds one: every number in the
sentence is a number of the payload (a test checks this for every template). A language model
may rephrase these sentences later, but the numbers always come from here.
"""

from __future__ import annotations

from typing import Any


def _pct(x: float) -> str:
    return f"{x:.3f}"


def narrate(type_: str, p: dict[str, Any]) -> str | None:
    """One sentence for an event, or None for events that are not worth a line (steps)."""
    fn = _TEMPLATES.get(type_)
    return fn(p) if fn else None


def _baseline(p: dict[str, Any]) -> str:
    return (
        f"Baseline: {p['features']} automatic features give a validation PR-AUC of "
        f"{_pct(p['val_pr_auc'])} against a base rate of {_pct(p['base_rate'])}."
    )


def _decision(p: dict[str, Any]) -> str:
    name = p.get("name")
    if p["status"] == "no_llm":
        return "No language model answered, so no features are proposed; the baseline stands."
    head = f"Round {p['round']}"
    if name is None:
        return f"{head}: the model gave no usable answer ({p['reasons'][0] if p.get('reasons') else 'unknown'})."
    if p["status"] != "proposed":
        why = p["reasons"][0] if p.get("reasons") else p["status"]
        return f"{head}: proposed `{name}`. Rejected at the {p['stage']} check: {why}"
    if p.get("vetoed"):
        return f"{head}: proposed `{name}`. Passed the checks; you vetoed it, so it was not tested."
    gain = p.get("gain")
    if gain is None:
        return f"{head}: proposed `{name}`."
    lo, hi = gain["ci95"]
    measured = (
        f"Gain {gain['mean']:+.4f} PR-AUC on the temporal folds (95% CI {lo:+.4f} to {hi:+.4f}), "
        f"margin {gain['margin']:.4f}."
    )
    if p["accepted"]:
        return (
            f"{head}: proposed `{name}`. Passed the guard and execution checks. {measured} "
            f"Accepted; the champion's validation PR-AUC is now {_pct(p['champion_val_pr_auc'])}."
        )
    return f"{head}: proposed `{name}`. Passed the guard and execution checks. {measured} Rejected: no gain."


def _checkpoint(p: dict[str, Any]) -> str:
    s = p["summary"]
    return (
        f"Waiting for you (round {p['round']}): approve or veto `{s['name']}` - {s['description'].rstrip('.')}. "
        f"If you do not answer in {p['timeout_seconds']:g} seconds it is {p['recommended']}d."
    )


def _answered(p: dict[str, Any]) -> str:
    how = "no answer in time, so the recommended action" if p["how"] == "timeout" else "your answer"
    return f"Round {p['round']}: {p['decision']} ({how})."


def _budget_stop(p: dict[str, Any]) -> str:
    limits, used = p["limits"], p["used"]
    if p["budget"] == "cost":
        return f"Stopped at the budget: the model cost ${used['cost_usd']:.4f} against a limit of ${limits['max_cost_usd']:.4f}."
    if p["budget"] == "time":
        return f"Stopped at the time limit: {used['seconds']:.0f} seconds used of {limits['max_seconds']:.0f}."
    return f"Stopped at the proposal limit: {used['proposals']} proposals of {limits['max_proposals']}."


def _finished(p: dict[str, Any]) -> str:
    test = p.get("test")
    tail = (
        f" Test rows, scored once: PR-AUC {_pct(test['pr_auc'])} against a base rate of {_pct(test['base_rate'])}."
        if test
        else f" {p.get('error') or 'The test rows were not scored.'}"
    )
    return f"Run finished ({p['status']}, {p['stop_reason']}).{tail}"


def _settings(p: dict[str, Any]) -> str:
    parts = [f"{k} {v['from']} -> {v['to']}" for k, v in p["changes"].items()]
    return "Settings changed: " + "; ".join(parts) + "."


def _suggestion(p: dict[str, Any]) -> str:
    return f"Round {p['round']}: using your suggestion: {p['text']}"


_TEMPLATES = {
    "baseline": _baseline,
    "feature_decision": _decision,
    "checkpoint": _checkpoint,
    "checkpoint_answered": _answered,
    "budget_stop": _budget_stop,
    "run_finished": _finished,
    "settings_changed": _settings,
    "suggestion": _suggestion,
}
