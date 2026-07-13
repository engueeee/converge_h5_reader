from __future__ import annotations

import pytest

from converge_h5_reader import ConvergeFile


@pytest.fixture
def boundaries(synthetic_h5):
    with ConvergeFile(synthetic_h5) as f:
        yield f.boundaries


def test_rows_are_decoded(boundaries):
    assert len(boundaries) == 4
    assert boundaries.names() == ("PISTON", "LINER", "CYLINDERHEAD", "EMPTY_BND")
    assert boundaries.ids() == (4, 14, 9, 28)
    assert all(isinstance(b.name, str) for b in boundaries)


def test_lookup_by_id_and_name(boundaries):
    assert boundaries["PISTON"].id == 4
    assert boundaries.by_id(14).name == "LINER"
    assert boundaries[9].name == "CYLINDERHEAD"  # int keys are ids, not positions
    assert boundaries.by_name("piston").id == 4


def test_unknown_lookup(boundaries):
    with pytest.raises(KeyError, match="no boundary named"):
        boundaries["NOPE"]
    with pytest.raises(KeyError, match="no boundary with id"):
        boundaries.by_id(999)


def test_stream_normalisation_and_filters(boundaries):
    # The file stores b"Stream_00"; we expose the canonical form.
    assert boundaries["PISTON"].stream == "STREAM_00"
    assert len(boundaries.for_stream(0)) == 3
    assert len(boundaries.for_stream("STREAM_01")) == 1
    assert len(boundaries.nonempty()) == 3
    assert boundaries["EMPTY_BND"].is_empty


def test_center_and_dataframe(boundaries):
    assert boundaries["PISTON"].center == pytest.approx((0.0, 0.0, -1.0))
    frame = boundaries.to_dataframe()
    assert list(frame.columns) == [
        "id",
        "name",
        "stream",
        "n_elements",
        "n_points",
        "center_x",
        "center_y",
        "center_z",
    ]
    assert frame.loc[frame["name"] == "LINER", "n_elements"].item() == 4
