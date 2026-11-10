"""Scan every DICOM header once and cache the tags the dataset card needs.

Reading 26k headers takes a couple of minutes, so the result is written to CSV and
the notebooks read the cache instead of the files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom

TAGS = (
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Modality",
    "PatientSex",
    "PatientAge",
    "BodyPartExamined",
    "ImageType",
    "Manufacturer",
    "StudyDate",
    "Rows",
    "Columns",
    "BitsAllocated",
    "PixelSpacing",
    "PatientPosition",
    "WindowCenter",
    "WindowWidth",
    "RescaleIntercept",
    "RescaleSlope",
)


def _one(path: Path) -> dict:
    row = {"dicom_path": str(path), "file_id": path.stem}
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        for tag in TAGS:
            value = getattr(ds, tag, None)
            row[tag] = ";".join(map(str, value)) if isinstance(value, list) else value
    except Exception as exc:  # a single unreadable file must not kill the scan
        row["error"] = repr(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path, help="folder of .dcm files")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    paths = sorted(args.images.rglob("*.dcm"))
    print(f"scanning {len(paths):,} files")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(_one, paths, chunksize=200), 1):
            rows.append(row)
            if i % 2000 == 0:
                print(f"  {i:,}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  shape={df.shape}  errors={int('error' in df and df['error'].notna().sum())}")


if __name__ == "__main__":
    main()
