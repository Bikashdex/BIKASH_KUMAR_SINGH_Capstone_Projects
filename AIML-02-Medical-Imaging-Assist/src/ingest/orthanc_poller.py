"""Poll Orthanc for newly-received studies and forward them to the inference API."""

from __future__ import annotations

import os
import time

import httpx

ORTHANC_URL = os.getenv("ORTHANC_URL", "http://localhost:8042")
API_URL = os.getenv("API_URL", "http://localhost:8000")
POLL_SECONDS = float(os.getenv("POLL_INTERVAL_SECONDS", "5"))
AUTH = None
if os.getenv("ORTHANC_USERNAME"):
    AUTH = (os.environ["ORTHANC_USERNAME"], os.environ["ORTHANC_PASSWORD"])


def fetch_dcm(client: httpx.Client, instance_id: str) -> bytes:
    response = client.get(f"{ORTHANC_URL}/instances/{instance_id}/file", auth=AUTH)
    # Without this a failed fetch returns Orthanc's error page as the "study" and the API scores it.
    response.raise_for_status()
    return response.content


def send_to_api(client: httpx.Client, instance_id: str, payload: bytes) -> dict | None:
    try:
        response = client.post(
            f"{API_URL}/predict",
            files={"file": (f"{instance_id}.dcm", payload, "application/dicom")},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as exc:
        print(f"[ingest] {instance_id} -> API error: {exc}", flush=True)
        return None


def _ids(items: list) -> list[str]:
    """Orthanc returns resource IDs either bare or as expanded objects, depending on version."""
    return [item if isinstance(item, str) else item["ID"] for item in items]


def instances_for_change(client: httpx.Client, change: dict) -> list[str]:
    """Resolve a /changes entry to the instances it covers.

    Orthanc >= 1.12 merges child changes into their parent, so a new image can
    arrive as a Patient/Study/Series change and never as an Instance change.
    """
    resource_type, resource_id = change.get("ResourceType"), change.get("ID")
    if resource_type == "Instance":
        return [resource_id]
    if resource_type == "Series":
        return _ids(client.get(f"{ORTHANC_URL}/series/{resource_id}/instances", auth=AUTH).json())
    if resource_type == "Study":
        return _ids(client.get(f"{ORTHANC_URL}/studies/{resource_id}/instances", auth=AUTH).json())
    if resource_type == "Patient":
        studies = _ids(client.get(f"{ORTHANC_URL}/patients/{resource_id}/studies", auth=AUTH).json())
        found: list[str] = []
        for study_id in studies:
            found += _ids(client.get(f"{ORTHANC_URL}/studies/{study_id}/instances", auth=AUTH).json())
        return found
    return []


def changed_instance_ids(client: httpx.Client, changes: dict) -> list[str]:
    """Accept both the legacy Instances/Deleted feed and the 1.12+ Changes feed."""
    found: list[str] = []
    for instance in changes.get("Instances", []):
        if not instance.get("Removed"):
            found.append(instance["ID"])
    for change in changes.get("Changes", []):
        if "Removed" in change.get("ChangeType", ""):
            continue
        found += instances_for_change(client, change)
    return found


def poll_once(client: httpx.Client, last: int, seen: set[str] | None = None) -> int:
    """Handle one /changes batch and return the marker to resume from."""
    changes = client.get(f"{ORTHANC_URL}/changes", params={"last": last}, auth=AUTH).json()
    last = changes.get("Last", last)
    for instance_id in changed_instance_ids(client, changes):
        # A merged parent change re-lists instances we already scored; score each one once. Only a successful
        # forward counts as seen, so an instance that arrived while the API was still booting is retried.
        if seen is not None and instance_id in seen:
            continue
        try:
            payload = fetch_dcm(client, instance_id)
        except httpx.HTTPError as exc:
            print(f"[ingest] {instance_id} -> fetch error: {exc}", flush=True)
            continue
        result = send_to_api(client, instance_id, payload)
        if result:
            print(
                f"[ingest] {instance_id} -> {result['prediction']} "
                f"p={result['probability']:.3f} ({result['inference_ms']}ms)",
                flush=True,
            )
            if seen is not None:
                seen.add(instance_id)
    return last


def main() -> None:
    print(f"[ingest] watching {ORTHANC_URL} every {POLL_SECONDS:.0f}s", flush=True)
    last = 0
    seen: set[str] = set()
    with httpx.Client() as client:
        while True:
            try:
                last = poll_once(client, last, seen)
            except httpx.HTTPError as exc:
                print(f"[ingest] Orthanc unreachable: {exc}", flush=True)
            if len(seen) > 10000:
                seen.clear()
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
