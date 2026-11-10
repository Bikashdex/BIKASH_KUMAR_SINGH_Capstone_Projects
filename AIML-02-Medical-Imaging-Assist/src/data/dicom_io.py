"""Read chest X-ray DICOM files and apply windowing before a model ever sees them."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset

WINDOW_CENTRE_DEFAULT = 1024
WINDOW_WIDTH_DEFAULT = 4096


def read_study(path: str | Path) -> Dataset:
    return pydicom.dcmread(str(path), force=True)


def describe(ds: Dataset) -> dict[str, object]:
    """Pull the tags worth documenting in dataset_card.md."""
    return {
        "sop_class_uid": getattr(ds, "SOPClassUID", None),
        "patient_id": getattr(ds, "PatientID", None),
        "study_instance_uid": getattr(ds, "StudyInstanceUID", None),
        "modality": getattr(ds, "Modality", None),
        "rows": int(getattr(ds, "Rows", 0)),
        "columns": int(getattr(ds, "Columns", 0)),
        "bits_allocated": int(getattr(ds, "BitsAllocated", 0)),
        "photometric_interpretation": getattr(ds, "PhotometricInterpretation", None),
        "pixel_spacing": getattr(ds, "PixelSpacing", None),
        "window_center": getattr(ds, "WindowCenter", None),
        "window_width": getattr(ds, "WindowWidth", None),
        "patient_position": getattr(ds, "PatientPosition", None),
    }


def window_centre(ds: Dataset) -> float:
    centres = _numbers(_tag(ds, "WindowCenter"))
    return centres[0] if centres else WINDOW_CENTRE_DEFAULT


def window_width(ds: Dataset) -> float:
    widths = _numbers(_tag(ds, "WindowWidth"))
    return widths[0] if widths else WINDOW_WIDTH_DEFAULT


def _tag(ds: Dataset, keyword: str):
    """Read an optional tag. pydicom raises instead of returning None for a malformed value."""
    try:
        return getattr(ds, keyword, None)
    except (ValueError, TypeError):
        return None


def _numbers(value) -> list[float]:
    """Parse a DICOM value that may be absent, scalar, or a multi-value container.

    Real files omit these tags entirely, so absence has to fall through to the defaults.
    """
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.split("\\")
    else:
        try:
            parts = list(value)  # pydicom's multi-value container is not always a list subclass
        except TypeError:
            parts = [value]
    out = []
    for part in parts:
        try:
            out.append(float(part))
        except (TypeError, ValueError):
            continue
    return out


def to_uint8(ds: Dataset, centre: float | None = None, width: float | None = None) -> np.ndarray:
    """Apply DICOM windowing and return HxW uint8. Windowing is what makes a scan readable."""
    pixels = ds.pixel_array.astype(np.float64)
    rescale_intercept = _numbers(_tag(ds, "RescaleIntercept")) or [0.0]
    rescale_slope = _numbers(_tag(ds, "RescaleSlope")) or [1.0]
    pixels = pixels * rescale_slope[0] + rescale_intercept[0]

    # 8-bit secondary captures are already display-encoded. Applying a Hounsfield window to them
    # (the defaults below) squeezes the whole image into a handful of grey levels.
    if centre is None and width is None and not _has_window(ds) and _bits_allocated(ds) <= 8:
        return np.clip(pixels, 0.0, 255.0).astype(np.uint8)

    centre = window_centre(ds) if centre is None else centre
    width = window_width(ds) if width is None else width
    low = centre - 0.5 - (width - 1) / 2
    high = centre - 0.5 + (width - 1) / 2
    if high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)

    scaled = (pixels - low) / (high - low)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _has_window(ds: Dataset) -> bool:
    return _numbers(_tag(ds, "WindowCenter")) and _numbers(_tag(ds, "WindowWidth"))


def _bits_allocated(ds: Dataset) -> int:
    try:
        return int(getattr(ds, "BitsAllocated", 8))
    except (TypeError, ValueError):
        return 8


def to_rgb_uint8(ds: Dataset) -> np.ndarray:
    """Repeat a single-channel windowed image into HxWx3, the shape most CNN code expects."""
    grey = to_uint8(ds)
    return np.repeat(grey[..., None], 3, axis=2)
