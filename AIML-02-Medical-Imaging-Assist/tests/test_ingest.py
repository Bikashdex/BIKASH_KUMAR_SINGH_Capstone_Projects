"""The poller's half of the Orthanc chain, exercised against a stub that speaks Orthanc's REST shape.

A stub serves `/changes` and `/instances/{id}/file` in both the legacy and the 1.12+ merged format, and the assertions
below are about what the poller does with them — forward new instances, skip `Removed` ones, survive a fetch that
fails, resolve a patient-level change down to its instances, score each instance once per run, and advance the marker
so repeated polls do not replay the same batch.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from src.ingest import orthanc_poller


class _Stub:
    def __init__(self) -> None:
        self.predicts: list[bytes] = []
        self.paths: list[str] = []
        self.files = {"good": b"GOOD-DICOM", "bad": b""}
        self.changes_calls: list[int] = []


def _make_handler(state: _Stub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep pytest output clean
            return

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state.paths.append(self.path)
            if self.path.startswith("/changes"):
                last = int(self.path.split("last=")[1])
                state.changes_calls.append(last)
                if last >= 20:  # carry an Orthanc 1.12+ feed, which merges child changes into the patient
                    payload = {
                        "First": 21,
                        "Last": 21,
                        "Changes": [
                            {"ChangeType": "StablePatient", "ResourceType": "Patient", "ID": "p1", "Seq": 21},
                            {"ChangeType": "RemovedInstance", "ResourceType": "Instance", "ID": "deleted", "Seq": 19},
                        ],
                    }
                else:
                    payload = {
                        "Last": 7,
                        "Instances": [
                            {"ID": "good"},
                            {"ID": "deleted", "Removed": True},
                            {"ID": "bad"},
                        ],
                    }
                self._send(200, json.dumps(payload).encode(), "application/json")
                return
            if self.path == "/patients/p1/studies":
                self._send(200, json.dumps([{"ID": "s1", "Type": "Study"}]).encode(), "application/json")
                return
            if self.path == "/studies/s1/instances":
                self._send(200, json.dumps([{"ID": "good", "Type": "Instance"}]).encode(), "application/json")
                return
            if self.path.startswith("/instances/"):
                instance_id = self.path.split("/")[2]
                if instance_id == "bad":
                    self._send(500, b"orthanc error", "text/plain")
                    return
                self._send(200, state.files[instance_id], "application/dicom")
                return
            self._send(404, b"no", "text/plain")

        def do_POST(self):
            state.paths.append(self.path)
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            state.predicts.append(body)
            self._send(
                200,
                json.dumps(
                    {
                        "study_uid": "1.2.3",
                        "prediction": "opacity",
                        "probability": 0.87,
                        "threshold_used": 0.51,
                        "model_version": "test",
                        "inference_ms": 12.0,
                    }
                ).encode(),
                "application/json",
            )

    return Handler


@pytest.fixture()
def stub():
    state = _Stub()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield state, base
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_poll_once_forwards_new_instances_and_skips_removed(stub, monkeypatch):
    state, base = stub
    monkeypatch.setattr(orthanc_poller, "ORTHANC_URL", base)
    monkeypatch.setattr(orthanc_poller, "API_URL", base)

    with httpx.Client() as client:
        last = orthanc_poller.poll_once(client, 0)

    assert last == 7, "the marker must advance or the next poll replays the same batch"
    assert state.changes_calls == [0]
    assert state.paths == [
        "/changes?last=0",
        "/instances/good/file",
        "/predict",
        "/instances/bad/file",
    ], "the Removed instance must not be fetched, and 'bad' must not stop the batch"
    assert len(state.predicts) == 1
    assert b"GOOD-DICOM" in state.predicts[0]


def test_poll_once_uses_the_marker_it_was_given(stub, monkeypatch):
    state, base = stub
    monkeypatch.setattr(orthanc_poller, "ORTHANC_URL", base)
    monkeypatch.setattr(orthanc_poller, "API_URL", base)

    with httpx.Client() as client:
        orthanc_poller.poll_once(client, 7)

    assert state.changes_calls == [7]


def test_poll_once_resolves_merged_patient_changes_once(stub, monkeypatch):
    """Orthanc 1.12+ reports a new image as a patient change; the poller must walk down to the instance."""
    state, base = stub
    monkeypatch.setattr(orthanc_poller, "ORTHANC_URL", base)
    monkeypatch.setattr(orthanc_poller, "API_URL", base)

    with httpx.Client() as client:
        seen: set[str] = set()
        last = orthanc_poller.poll_once(client, 20, seen)
        orthanc_poller.poll_once(client, last, seen)  # the feed still lists the same patient

    assert last == 21
    assert "/patients/p1/studies" in state.paths
    assert "/studies/s1/instances" in state.paths
    assert "/instances/deleted/file" not in state.paths, "a removed instance must not be fetched"
    assert len(state.predicts) == 1, "a re-listed patient must not re-score instances already seen"
    assert b"GOOD-DICOM" in state.predicts[0]


def test_poll_once_lets_an_unreachable_server_raise(stub, monkeypatch):
    """main() catches this and retries; poll_once must not swallow it, or the loop spins silently."""
    monkeypatch.setattr(orthanc_poller, "ORTHANC_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(orthanc_poller, "API_URL", "http://127.0.0.1:9")

    with httpx.Client() as client, pytest.raises(httpx.HTTPError):
        orthanc_poller.poll_once(client, 0)
