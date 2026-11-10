"""Create a small synthetic RSNA-shaped dataset so the EDA notebook can be executed end to end."""

import csv
import random
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import SecondaryCaptureImageStorage, ExplicitVRLittleEndian

OUT = Path("synthetic_rsna")
IMAGES = OUT / "train_images"
IMAGES.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
random.seed(0)

N_PATIENTS = 40
rows_labels, rows_class, rows_s2 = [], [], []

for p in range(N_PATIENTS):
    patient = f"SINT{p:04d}"
    study = f"1.2.826.0.1.3680043.10.171.{p}.1"
    has_pneumonia = rng.random() < 0.35
    n_films = int(rng.integers(1, 6))
    rows_class.append(
        {
            "PatientID": patient,
            "Class": "Pneumonia" if has_pneumonia else "No Finding",
            "StudyInstanceUID": study,
        }
    )
    for f in range(n_films):
        series = f"1.2.826.0.1.3680043.10.171.{p}.{f + 1}"
        target = int(has_pneumonia and rng.random() < 0.8)
        rows_labels.append(
            {"SeriesInstanceUID": series, "StudyInstanceUID": study,
             "Label": "pneumonia" if target else "none", "Target": target}
        )
        rows_s2.append(
            {"PatientID": patient, "StudyInstanceUID": study, "SeriesInstanceUID": series,
             "Target": target, "x": 100, "y": 100, "width": 400, "height": 500}
        )

        arr = (rng.normal(1200, 300, (256, 256))).clip(0, 4095).astype(np.uint16)
        if target:
            ys, xs = np.ogrid[:256, :256]
            blob = ((ys - 150) ** 2 + (xs - 90) ** 2) < 45**2
            arr[blob] = np.clip(arr[blob] + 700, 0, 4095).astype(np.uint16)

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = series
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = meta
        ds.SOPClassUID = SecondaryCaptureImageStorage
        ds.SOPInstanceUID = series
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = series
        ds.PatientID = patient
        ds.Modality = "CR"
        ds.Rows, ds.Columns = 256, 256
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
        ds.PixelRepresentation = 0
        ds.RescaleIntercept, ds.RescaleSlope = 0, 1
        ds.WindowCenter, ds.WindowWidth = 1200, 2400
        ds.PatientPosition = random.choice(["PA", "AP"])
        ds.PixelData = arr.tobytes()
        ds["PixelData"].VR = "OB"
        pydicom.dcmwrite(IMAGES / f"{series}.dcm", ds, write_like_original=False)


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


write_csv("train_labels.csv", rows_labels)
write_csv("stage_2_detailed_class_info.csv", rows_class)
write_csv("stage_2_train_labels.csv", rows_s2)
print(f"wrote {len(rows_labels)} synthetic DICOM images for {N_PATIENTS} patients into {OUT}")
