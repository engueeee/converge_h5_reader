from __future__ import annotations

import h5py
import pytest

from converge_h5_reader import ClosedFileError, ConvergeFile, StreamNotFoundError


def test_metadata_is_scalarised(synthetic_h5):
    with ConvergeFile(synthetic_h5) as f:
        assert f.crank_angle == pytest.approx(-74.07573604)
        assert f.time == pytest.approx(-0.01543245)
        assert f.rpm == 800.0
        assert f.version == (5, 1, 1)
        assert isinstance(f.attrs["RPM"], float)


def test_cad_from_name_matches_the_root_attribute(synthetic_h5):
    with ConvergeFile(synthetic_h5) as f:
        assert f.cad_from_name == pytest.approx(f.crank_angle, abs=1e-4)


def test_streams(synthetic_h5):
    with ConvergeFile(synthetic_h5) as f:
        assert f.stream_names == ("STREAM_00", "STREAM_01")
        assert len(f) == 2
        assert f[0].n_cells == 8
        assert f["STREAM_01"].n_cells == 3
        assert f[0] is f.stream(0)  # streams are cached
        assert [s.name for s in f] == ["STREAM_00", "STREAM_01"]


def test_unknown_stream(synthetic_h5):
    with ConvergeFile(synthetic_h5) as f, pytest.raises(StreamNotFoundError, match="STREAM_09"):
        f.stream(9)


def test_close_makes_the_file_unusable(synthetic_h5):
    f = ConvergeFile(synthetic_h5)
    assert f.crank_angle == pytest.approx(-74.07573604)
    f.close()
    with pytest.raises(ClosedFileError):
        _ = f.h5


def test_open_reads_no_bulk_data(synthetic_h5, monkeypatch):
    """Opening a file and touching metadata must not read any cell array."""
    reads: list[str] = []
    original = h5py.Dataset.__getitem__

    def counting_getitem(self, key):
        if "CELL_CENTER_DATA" in self.name:
            reads.append(self.name)
        return original(self, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", counting_getitem)

    with ConvergeFile(synthetic_h5) as f:
        _ = f.attrs, f.stream_names, f.boundaries
        stream = f[0]
        _ = stream.n_cells, stream.variables, stream.attrs
        handle = stream.cells.dataset("T")  # a handle is not a read
        assert handle.shape == (8,)
        assert reads == []

        stream.cells["T"]
        assert len(reads) == 1
