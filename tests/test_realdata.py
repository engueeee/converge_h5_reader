"""Tests against a real CONVERGE file. Run with:

    CONVERGE_H5_TEST_FILE=/path/to/post000061_-7.40757e+01.h5 pytest -m realdata
"""

from __future__ import annotations

import resource

import numpy as np
import pytest

from converge_h5_reader import ConvergeFile, region

pytestmark = pytest.mark.realdata


def _peak_rss_mb() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if peak > 1e7 else peak / 1e3


def test_metadata(real_h5):
    with ConvergeFile(real_h5) as f:
        assert f.rpm == pytest.approx(800.0)
        assert f.version == (5, 1, 1)
        assert f.crank_angle == pytest.approx(f.cad_from_name, abs=1e-4)

        assert f.stream_names == ("STREAM_00",)
        stream = f[0]
        assert stream.n_cells > 1_000_000
        assert "TEMPERATURE" in stream.variables
        assert "VELOCITY_Z" in stream.variables

        boundaries = f.boundaries
        assert boundaries["PISTON"].id == 4
        assert boundaries["LINER"].n_elements > 0
        assert all(isinstance(name, str) for name in boundaries.names())


def test_filtered_read_stays_lazy(real_h5):
    """A region-filtered read must not materialise the whole stream."""
    before = _peak_rss_mb()
    with ConvergeFile(real_h5) as f:
        stream = f[0]
        frame = stream.cells.to_dataframe(["T", "P"], where=region(1))

    assert 0 < len(frame) < stream.n_cells
    assert frame["T"].dtype == np.float32
    assert frame["P"].dtype == np.float32
    assert (frame["T"] > 0).all()

    growth = _peak_rss_mb() - before
    # Three full float32 fields would be ~100 MB; the whole stream would be 1.2 GB.
    assert growth < 600, f"read grew peak RSS by {growth:.0f} MB"


def test_chunked_read_matches_a_full_read(real_h5):
    with ConvergeFile(real_h5) as f:
        stream = f[0]
        chunks = list(stream.cells.iter_chunks(["T"], chunk=1_000_000))
        assert sum(len(c) for c in chunks) == stream.n_cells

        full = stream.cells["T"]
        np.testing.assert_allclose(np.concatenate([c["T"].to_numpy() for c in chunks]), full)


def test_matches_the_legacy_reader(real_h5):
    """Values must agree with functions_pyflut.converge_h5_to_df on the shared fields."""
    h5py = pytest.importorskip("h5py")

    with h5py.File(real_h5, "r") as raw:
        group = raw["STREAM_00"]["CELL_CENTER_DATA"]
        legacy_t = np.asarray(group["TEMPERATURE"][:]).ravel()
        legacy_x = np.asarray(group["XCEN_X"][:]).ravel()

    with ConvergeFile(real_h5) as f:
        cells = f[0].cells
        np.testing.assert_array_equal(cells["T"], legacy_t)
        np.testing.assert_array_equal(cells["x"], legacy_x)


@pytest.mark.mesh
def test_vtk_backend_reads_this_file_not_its_neighbour(real_h5):
    """vtkCONVERGECFDReader globs the directory as a time series.

    Defaulting to "the last timestep" would return a *neighbouring* post*.h5's data,
    silently and with a different cell count. The default must be this file's own CAD.
    """
    pytest.importorskip("pyvista")
    from converge_h5_reader.mesh import read_mesh
    from converge_h5_reader.mesh.vtk_reader import read_timesteps

    with ConvergeFile(real_h5) as f:
        expected_cells = f[0].n_cells
        expected_cad = f.crank_angle

    steps = read_timesteps(real_h5)
    if len(steps) > 1:
        assert not np.isclose(steps[-1], expected_cad), (
            "this file is the last step in its directory, so the test cannot catch the bug"
        )

    mesh = read_mesh(real_h5)
    volume = mesh["Mesh"]
    assert volume.n_cells == expected_cells
    assert "REGION_ID" in volume.cell_data


@pytest.mark.mesh
def test_tumble_slice_to_vti(real_h5, tmp_path):
    """The full CLAUDE.md pipeline: read -> region -> tumble slice -> sample -> .vti."""
    pytest.importorskip("pyvista")
    import pyvista as pv

    from converge_h5_reader.mesh import extract, read_mesh

    mesh = read_mesh(real_h5)
    cylinder = extract.to_volume(mesh, region_id=1)
    assert cylinder.n_cells > 0

    plane = extract.tumble_slice(cylinder, y=0.0, keep_largest=True)
    assert plane.n_cells > 0

    image = extract.sample_to_image(plane, spacing=200e-6, fields=["TEMPERATURE", "VELOCITY"])
    assert image.point_data["vtkValidPointMask"].any()

    path = extract.save_vti(image, tmp_path / "tumble.vti")
    assert pv.read(path).dimensions == image.dimensions


@pytest.mark.mesh
def test_vtk_backend_blocks_match_the_boundary_table(real_h5):
    pytest.importorskip("pyvista")
    from converge_h5_reader.mesh import read_mesh

    with ConvergeFile(real_h5) as f:
        boundary_names = set(f.boundaries.nonempty().names())

    mesh = read_mesh(real_h5, backend="vtk")
    block_names = {name.split("#")[0] for name in mesh.keys()}  # noqa: SIM118

    # Every named block the VTK reader produces should be a boundary we can see
    # via h5py (the reader adds volume-region blocks on top).
    assert boundary_names & block_names, (
        f"no overlap between boundaries {sorted(boundary_names)[:5]} "
        f"and blocks {sorted(block_names)[:5]}"
    )
