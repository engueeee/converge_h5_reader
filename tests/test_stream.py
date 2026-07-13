from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import CELL_VARS, N_MESH_POLYGONS, N_MESH_VERTICES
from converge_h5_reader import ConvergeFile, MissingFieldError


@pytest.fixture
def stream(synthetic_h5):
    with ConvergeFile(synthetic_h5) as f:
        yield f[0]


def test_variables_come_from_variable_names(stream):
    # The declared order is not the sorted order; a reader that used
    # sorted(CELL_CENTER_DATA) would fail here.
    assert stream.variables == tuple(CELL_VARS)
    assert stream.variables != tuple(sorted(CELL_VARS))


def test_alias_access_preserves_dtype(stream):
    values = stream.cells["T"]
    assert values.dtype == np.float32
    np.testing.assert_allclose(values, stream.cells["TEMPERATURE"])
    np.testing.assert_allclose(values, 300.0 + 100.0 * np.arange(8))


def test_missing_field_lists_what_is_available(stream):
    with pytest.raises(MissingFieldError) as excinfo:
        stream.cells["NOPE"]
    assert "TEMPERATURE" in str(excinfo.value)


def test_contains_and_len(stream):
    assert "T" in stream.cells
    assert "TEMPERATURE" in stream.cells
    assert "NOPE" not in stream.cells
    assert len(stream.cells) == len(CELL_VARS)


def test_to_dataframe_columns_and_dtypes(stream):
    frame = stream.cells.to_dataframe(["T", "P"])
    assert list(frame.columns) == ["T", "P", "time", "cad"]
    assert frame["T"].dtype == np.float32
    assert len(frame) == 8
    assert frame["cad"].iloc[0] == pytest.approx(-74.07573604)


def test_to_dataframe_without_rename(stream):
    frame = stream.cells.to_dataframe(["T"], rename=False, add_time=False, add_cad=False)
    assert list(frame.columns) == ["T"]

    frame = stream.cells.to_dataframe(["TEMPERATURE"], rename=False, add_time=False, add_cad=False)
    assert list(frame.columns) == ["TEMPERATURE"]


def test_to_dataframe_all_fields(stream):
    frame = stream.cells.to_dataframe(add_time=False, add_cad=False)
    assert len(frame.columns) == len(CELL_VARS)


def test_to_dataframe_upcast(stream):
    frame = stream.cells.to_dataframe(["T"], dtype=np.float64)
    assert frame["T"].dtype == np.float64


def test_iter_chunks_matches_a_full_read(stream):
    chunks = list(stream.cells.iter_chunks(["T", "P"], chunk=3))
    assert [len(c) for c in chunks] == [3, 3, 2]

    joined = pd.concat(chunks, ignore_index=True)
    full = stream.cells.to_dataframe(["T", "P"], add_time=False, add_cad=False)
    pd.testing.assert_frame_equal(joined, full)


def test_iter_chunks_rejects_bad_chunk(stream):
    with pytest.raises(ValueError, match="chunk must be positive"):
        list(stream.cells.iter_chunks(["T"], chunk=0))


def test_shape_and_dtype_without_reading(stream):
    assert stream.cells.shape("T") == (8,)
    assert stream.cells.dtype("T") == np.float32


def test_vertices_and_connectivity(stream):
    vertices = stream.vertices()
    assert vertices.shape == (N_MESH_VERTICES, 3)
    assert stream.n_vertices == N_MESH_VERTICES

    conn = stream.connectivity
    assert conn.n_polygons == N_MESH_POLYGONS
    np.testing.assert_array_equal(conn.polygon(0), [0, 1, 2, 3])

    offsets, polygons = conn.cell_map()
    assert offsets.shape == (stream.n_cells + 1,)
    # Every hex owns 6 faces: 2 horizontal and 4 sides.
    for cell in range(stream.n_cells):
        assert len(polygons[offsets[cell] : offsets[cell + 1]]) == 6

    # Cells 0 and 1 share the horizontal quad at level 1.
    assert 1 in polygons[offsets[0] : offsets[1]]
    assert 1 in polygons[offsets[1] : offsets[2]]


def test_second_stream_has_no_mesh_but_reads_cells(synthetic_h5):
    with ConvergeFile(synthetic_h5) as f:
        assert f[1].n_cells == 3
        assert len(f[1].cells.to_dataframe(["T"])) == 3
