"""The API must hand back JSON for whatever pydicom gives it, on any OS."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, SecondaryCaptureImageStorage

from src.api import main


class _FixedModel:
    def __call__(self, tensor):
        return torch.zeros(1, 1)


def _bundle() -> dict[str, object]:
    return {
        "model": _FixedModel(),
        "threshold": 0.6,
        "device": "cpu",
        "transform": lambda array: torch.zeros(3, 224, 224),
        "epoch": 1,
        "input_size": 224,
    }


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(main, "_bundle", _bundle())
    return TestClient(main.app)


def _dicom_bytes() -> bytes:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = "1.2.3.4"
    ds.PatientID = "synthetic"
    ds.StudyInstanceUID = "1.2.3.4.5"
    ds.Modality = "CR"
    ds.Rows = 8
    ds.Columns = 8
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    # MultiValue, not list: this is what broke JSON serialisation on the first live run.
    ds.PixelSpacing = ["0.143", "0.143"]
    ds.PixelData = np.zeros((8, 8), dtype=np.uint8).tobytes()
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_predict_dicom_serialises_multivalue_tags(client):
    response = client.post(
        "/predict", files={"file": ("study.dcm", _dicom_bytes(), "application/dicom")}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["study_uid"] == "1.2.3.4.5"
    assert body["prediction"] == "no_opacity"
    assert body["dicom_tags"]["pixel_spacing"] == [0.143, 0.143]


def test_predict_png_still_accepted(client):
    response = client.post("/predict", files={"file": ("study.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["study_uid"] == "study.png"
