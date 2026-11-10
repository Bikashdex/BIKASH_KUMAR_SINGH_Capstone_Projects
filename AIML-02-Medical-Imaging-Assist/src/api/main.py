"""FastAPI inference service. Boots without weights so the container is demonstrable early."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.data.dicom_io import describe, read_study, to_rgb_uint8
from src.data.transforms import make_transform
from src.api import audit

DISCLAIMER = "For education only. Not for clinical use."
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/densenet121_lungopacity.pt")
MODEL_VERSION = os.getenv("MODEL_VERSION", "unversioned")

app = FastAPI(
    title="AIML-02 Medical Imaging Inference",
    version=MODEL_VERSION,
    description=f"{DISCLAIMER}",
)

_bundle: dict[str, object] | None = None


def load_bundle() -> dict[str, object] | None:
    global _bundle
    if _bundle is not None:
        return _bundle
    path = Path(MODEL_PATH)
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "model" in checkpoint:            # a pickled module, as old exports wrote it
        model = checkpoint["model"]
        size = 224
    else:                                # the graded artefact notebook 03 writes: weights + config
        from src.models.classifier import build_densenet121

        model = build_densenet121(num_classes=1, pretrained=False)
        model.load_state_dict(checkpoint["state_dict"])
        size = int(checkpoint.get("config", {}).get("train_size", 224))

    # The API must see exactly what training saw. Passing raw pixels to a model trained on
    # resize-235 -> center-crop-224 -> ImageNet normalisation returns confident nonsense.
    _bundle = {
        "model": model.to(device).eval(),
        "threshold": float(checkpoint.get("threshold", 0.5)),
        "device": device,
        "transform": make_transform(size, train=False),
        "epoch": checkpoint.get("epoch"),
        "input_size": size,
    }
    return _bundle


def _jsonable(value: object) -> object:
    """pydicom returns MultiValue for tags like PixelSpacing; JSON needs plain lists."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__iter__"):
        return [_jsonable(item) for item in value]
    return str(value)


class Prediction(BaseModel):
    study_uid: str
    prediction: str
    probability: float = Field(ge=0.0, le=1.0)
    threshold_used: float
    model_version: str
    inference_ms: float
    dicom_tags: dict[str, object] | None = None
    disclaimer: str = DISCLAIMER


class DecisionIn(BaseModel):
    study_uid: str = Field(min_length=1)
    decision: str = Field(pattern="^(approved|rejected)$")
    reason: str | None = None
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold_used: float | None = None
    model_version: str | None = None


@app.get("/health")
def health() -> dict[str, object]:
    bundle = load_bundle()
    return {
        "status": "ok",
        "model_loaded": bundle is not None,
        "device": bundle["device"] if bundle else None,
        "disclaimer": DISCLAIMER,
    }


@app.get("/model-info")
def model_info() -> dict[str, object]:
    return {
        "model": "densenet121",
        "version": MODEL_VERSION,
        "task": "binary lung-opacity detection on chest X-ray",
        "label_semantics": "1 = lung opacity present; 0 = normal OR another abnormality",
        "split_policy": "grouped split on patient_id (one image per group in RSNA stage_2)",
        "disclaimer": DISCLAIMER,
    }


@app.post("/audit", status_code=201)
def record_decision(entry: DecisionIn) -> dict[str, object]:
    row = audit.record(
        study_uid=entry.study_uid,
        decision=entry.decision,
        reason=entry.reason,
        probability=entry.probability,
        threshold_used=entry.threshold_used,
        model_version=entry.model_version,
    )
    return {"recorded": row, "disclaimer": DISCLAIMER}


@app.get("/audit/{study_uid}")
def decisions_for_study(study_uid: str) -> dict[str, object]:
    return {"study_uid": study_uid, "decisions": audit.for_study(study_uid), "disclaimer": DISCLAIMER}


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...), study_uid: str | None = None) -> Prediction:
    bundle = load_bundle()
    if bundle is None:
        raise HTTPException(status_code=503, detail=f"no model at {MODEL_PATH}; train and export first")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")

    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "study.dcm").suffix or ".dcm",
                                     delete=False) as handle:
        handle.write(raw)
        tmp = Path(handle.name)

    tags: dict[str, object] | None = None
    uid = study_uid or (file.filename or "unknown")
    try:
        ds = read_study(tmp)
        tags = describe(ds)
        uid = study_uid or str(tags.get("study_instance_uid") or tags.get("patient_id") or uid)
        array = to_rgb_uint8(ds)
    except Exception:
        from PIL import Image
        from io import BytesIO

        with Image.open(BytesIO(raw)) as img:
            array = np.asarray(img.convert("RGB"))
    finally:
        tmp.unlink(missing_ok=True)

    transform = bundle["transform"]
    tensor = transform(array).unsqueeze(0).to(bundle["device"])  # type: ignore[operator,union-attr]

    started = time.perf_counter()
    with torch.no_grad():
        logit = bundle["model"](tensor)  # type: ignore[operator]
        probability = float(torch.sigmoid(logit).item())
    elapsed_ms = (time.perf_counter() - started) * 1000

    threshold = float(bundle["threshold"])  # type: ignore[arg-type]
    return Prediction(
        study_uid=uid,
        prediction="opacity" if probability >= threshold else "no_opacity",
        probability=round(probability, 4),
        threshold_used=threshold,
        model_version=MODEL_VERSION,
        inference_ms=round(elapsed_ms, 2),
        dicom_tags=None if tags is None else {key: _jsonable(value) for key, value in tags.items()},
    )
# --- optional: serve the built React viewer from the same origin (free-tier deployment) ---
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_DIST = Path(os.getenv("FRONTEND_DIST", "deploy/dist"))
if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="viewer-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa(full_path: str):
        candidate = _DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_DIST / "index.html")
