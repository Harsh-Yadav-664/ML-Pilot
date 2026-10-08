"""SHAP of a finished run and answers grounded in its stored records (#63)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

import app.db.session as db_session
from app.jobs import handlers
from app.services.explain_service import ExplainService
from ml.experiments.acceptance import DEFAULT_RULE
from ml.reports import explain
from tests.integration import test_baseline_run_api as base
from tests.integration.test_run_loop_api import GOOD, finished, runs_url
from tests.integration.test_tasks_api import take_snapshot
from tests.unit.test_feature_engine import Costly
from tests.unit.test_llm_sql import answer

connection = base.connection  # the fixture, re-exported


async def demo_run(
    client: httpx.AsyncClient, project_id: str, conn: str, monkeypatch: pytest.MonkeyPatch
) -> str:
    monkeypatch.setitem(DEFAULT_RULE, "min_gain", -1.0)
    monkeypatch.setitem(DEFAULT_RULE, "std_multiplier", -1e6)
    gateway = Costly([answer(k) for k in [*GOOD[:2], "leaky_sql"]], cost=0.01)
    monkeypatch.setattr(handlers, "make_gateway", lambda: gateway)
    version = await take_snapshot(client, project_id, conn, "2025-01-01T00:00:00Z", base.TABLES)
    _, run_id = await base.started_run(client, project_id, conn, version)
    started = await client.post(
        runs_url(project_id, f"/{run_id}/start"), json={"max_rounds": 3, "patience": 3}
    )
    job = await finished(client, project_id, started.json()["job_id"])
    assert job["status"] == "succeeded", job
    return run_id


class Wording:
    """A model that answers with a fixed text."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def complete(self, *_: Any) -> str:
        return self.text


async def test_shap_answers_and_the_no_record_reply_come_from_stored_records(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = await demo_run(client, project_id, connection, monkeypatch)

    shap = (await client.get(runs_url(project_id, f"/{run_id}/explain"))).json()
    assert shap["method"].startswith("TreeSHAP") and shap["n_rows"] > 100
    assert abs(sum(f["share"] for f in shap["features"]) - 1.0) < 1e-9
    assert shap["features"][0]["mean_abs"] >= shap["features"][-1]["mean_abs"] > -1e-12
    top = shap["features"][0]["feature"]

    feats = (await client.get(runs_url(project_id, f"/{run_id}/features"))).json()
    proposed = [f["name"] for f in feats["features"] if f["kind"] == "llm_sql"]
    assert proposed, feats

    async with db_session.AsyncSessionLocal() as db:
        corpus = await ExplainService(db).corpus(project_id, run_id)
    ids = {r.id for r in corpus}
    assert {"run", "shap", f"feature:{top}"} <= ids

    # debrief and Q&A through the API (the offline stub model): whatever comes back, every
    # number and name in it is in the records it cites
    for question in ("debrief", f"why was {proposed[0]} accepted or rejected?", "test score?"):
        body = (
            await client.post(runs_url(project_id, f"/{run_id}/ask"), json={"question": question})
        ).json()
        assert body["mode"] in ("llm", "fallback") and body["records"], body
        cited = [r for r in corpus if r.id in body["records"]]
        assert explain.violations(body["answer"], cited) == [], body
    debrief = (await client.get(runs_url(project_id, f"/{run_id}/debrief"))).json()
    assert (
        explain.violations(debrief["answer"], [r for r in corpus if r.id in debrief["records"]])
        == []
    )

    # a feature that was never proposed has no record
    unknown = (
        await client.post(
            runs_url(project_id, f"/{run_id}/ask"),
            json={"question": "What did weekend_magic_feature do?"},
        )
    ).json()
    assert unknown == {
        "answer": explain.NO_RECORD,
        "records": [],
        "mode": "no_record",
        "note": None,
    }


async def test_a_wording_with_a_number_that_is_not_in_the_records_is_replaced(
    client: httpx.AsyncClient,
    project_id: str,
    connection: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id = await demo_run(client, project_id, connection, monkeypatch)
    async with db_session.AsyncSessionLocal() as db:
        service = ExplainService(db)
        records = explain.retrieve(
            "what is the test score?", await service.corpus(project_id, run_id)
        )
        stored = explain.records_answer(records)
        invented = await service.ask(
            project_id,
            run_id,
            "what is the test score?",
            Wording("The test PR-AUC is 0.9731 [run]."),  # type: ignore[arg-type]
        )
        honest = await service.ask(
            project_id,
            run_id,
            "what is the test score?",
            Wording(stored.splitlines()[0]),  # type: ignore[arg-type]
        )
    assert invented.mode == "fallback" and "0.9731" in (invented.note or "")
    assert invented.answer == stored and "0.9731" not in invented.answer
    assert honest.mode == "llm" and honest.answer == stored.splitlines()[0]
