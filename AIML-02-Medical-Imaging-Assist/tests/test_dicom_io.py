"""Windowing must survive files that omit the tags, which real chest X-ray exports do."""

from __future__ import annotations

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, SecondaryCaptureImageStorage

from src.data.dicom_io import (
    WINDOW_CENTRE_DEFAULT,
    WINDOW_WIDTH_DEFAULT,
    to_uint8,
    window_centre,
    window_width,
)


TAGS = {
    "centre": "WindowCenter",
    "width": "WindowWidth",
    "slope": "RescaleSlope",
    "intercept": "RescaleIntercept",
}


def _ds(bits: int = 16, **tags) -> Dataset:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = "1.2.3.4"
    ds.Rows = 4
    ds.Columns = 4
    ds.BitsAllocated = bits
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    dtype = "<u1" if bits <= 8 else "<u2"
    ds.PixelData = np.arange(16, dtype=dtype).tobytes()
    for key, value in tags.items():
        setattr(ds, TAGS[key], value)
    return ds


def test_missing_window_tags_fall_back_to_defaults():
    ds = _ds()
    assert not hasattr(ds, "WindowCenter")
    assert window_centre(ds) == WINDOW_CENTRE_DEFAULT
    assert window_width(ds) == WINDOW_WIDTH_DEFAULT


def test_multi_value_window_tag_takes_the_first_number():
    ds = _ds(centre="64\\128")
    assert window_centre(ds) == 64.0


def test_empty_window_tag_falls_back_to_default():
    # A present-but-blank tag behaves like a missing one. pydicom rejects non-numeric DS at
    # assignment time, so a junk value can never reach this reader from a parsed file.
    ds = _ds(width="")
    assert window_width(ds) == WINDOW_WIDTH_DEFAULT


def test_windowing_maps_the_range_onto_uint8():
    ds = _ds(centre=7.5, width=16)
    out = to_uint8(ds)
    assert out.dtype == np.uint8
    assert out.shape == (4, 4)
    assert out[0, 0] == 8 and out.max() == 255  # pixels 0..15 mapped onto low=-0.5, high=14.5


def test_rescale_is_applied_before_windowing():
    plain = _ds(centre=7.5, width=16)
    shifted = _ds(centre=7.5, width=16)
    shifted.RescaleSlope = "2"
    shifted.RescaleIntercept = "-8"
    assert not np.array_equal(to_uint8(plain), to_uint8(shifted))


@pytest.mark.parametrize("centre,width", [(1024, 4096), (40, 400), (0, 1)])
def test_windowing_stays_in_uint8_range(centre, width):
    out = to_uint8(_ds(centre=centre, width=width), centre=centre, width=width)
    assert out.min() >= 0 and out.max() <= 255


def test_8_bit_capture_without_window_tags_keeps_its_range():
    # This is what the RSNA files actually are. Windowing them with the Hounsfield defaults
    # collapsed 0..245 into 64..79 and destroyed the image.
    ds = _ds(bits=8)
    out = to_uint8(ds)
    assert out.min() == 0 and out.max() == 15


def test_16_bit_capture_without_window_tags_uses_the_default_window():
    ds = _ds(bits=16)
    out = to_uint8(ds)
    assert not np.array_equal(out, ds.pixel_array)
    assert int(out.max()) - int(out.min()) <= 2  # compressed near the centre of the default window


def test_explicit_window_still_applies_to_an_8_bit_capture():
    ds = _ds(bits=8)
    out = to_uint8(ds, centre=7.5, width=16)
    assert out.min() == 8 and out.max() == 255
