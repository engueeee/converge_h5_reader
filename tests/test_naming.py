from __future__ import annotations

import pytest

from converge_h5_reader import cad_from_filename, normalize_stream


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "STREAM_00"), (1, "STREAM_01"), (12, "STREAM_12"), ("1", "STREAM_01")],
)
def test_normalize_stream_from_index(value, expected):
    assert normalize_stream(value) == expected


@pytest.mark.parametrize("value", ["STREAM_00", "stream_00", "Stream_00"])
def test_normalize_stream_from_name(value):
    assert normalize_stream(value) == "STREAM_00"


def test_normalize_stream_rejects_garbage():
    # The legacy reader silently fell back to STREAM_00 here.
    with pytest.raises(ValueError, match="cannot interpret"):
        normalize_stream("CELL_CENTER_DATA")
    with pytest.raises(ValueError, match=">= 0"):
        normalize_stream(-1)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("post000061_-7.40757e+01.h5", -74.0757),
        ("post000224_+7.00004e+02.h5", 700.004),
        ("post000001_+0.00000e+00.h5", 0.0),
    ],
)
def test_cad_from_filename(name, expected):
    assert cad_from_filename(name) == pytest.approx(expected)


def test_cad_from_filename_non_matching():
    assert cad_from_filename("results.h5", strict=False) is None
    with pytest.raises(ValueError, match="naming convention"):
        cad_from_filename("results.h5")
