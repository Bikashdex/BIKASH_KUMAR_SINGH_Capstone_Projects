"""The decision log has to survive the process that wrote it, and it has to refuse nonsense."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import audit, main


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_DB_URL", f"sqlite:///{tmp_path / 'audit.db'}")
    return TestClient(main.app)


def test_decision_is_written_and_read_back(store):
    response = store.post(
        "/audit",
        json={
            "study_uid": "1.2.276.0.7230010.3.1.2.8323329.8799.1517874339.479989",
            "decision": "approved",
            "reason": "opacity confirmed in the right lower lobe",
            "probability": 0.8685,
            "threshold_used": 0.5107557773590088,
            "model_version": "unversioned",
        },
    )

    assert response.status_code == 201
    row = response.json()["recorded"]
    assert row["decision"] == "approved"
    assert row["probability"] == 0.8685
    assert row["reason"] == "opacity confirmed in the right lower lobe"
    assert row["recorded_at"].endswith("+00:00")

    read = store.get(f"/audit/{row['study_uid']}").json()
    assert [d["id"] for d in read["decisions"]] == [row["id"]]


def test_decisions_are_scoped_to_their_study(store):
    store.post("/audit", json={"study_uid": "study-a", "decision": "approved"})
    store.post("/audit", json={"study_uid": "study-b", "decision": "rejected"})

    assert [d["decision"] for d in store.get("/audit/study-a").json()["decisions"]] == ["approved"]
    assert [d["decision"] for d in store.get("/audit/study-b").json()["decisions"]] == ["rejected"]
    assert store.get("/audit/study-c").json()["decisions"] == []


def test_unknown_decision_is_rejected_before_it_reaches_the_table(store):
    response = store.post("/audit", json={"study_uid": "study-a", "decision": "signed off"})
    assert response.status_code == 422
    assert store.get("/audit/study-a").json()["decisions"] == []


def test_store_survives_a_new_connection(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'audit.db'}"
    monkeypatch.setenv("AUDIT_DB_URL", url)
    audit.record("study-a", "approved", reason="first")

    monkeypatch.delenv("AUDIT_DB_URL")
    reopened = audit.for_study("study-a", path=tmp_path / "audit.db")
    assert len(reopened) == 1
    assert reopened[0]["reason"] == "first"


def test_writer_refuses_a_bad_decision_directly(tmp_path):
    with pytest.raises(ValueError):
        audit.record("study-a", "maybe", path=tmp_path / "audit.db")
