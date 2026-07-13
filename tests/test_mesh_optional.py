from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys

import pytest

import converge_h5_reader
from converge_h5_reader import MissingBackendError

HAS_PYVISTA = importlib.util.find_spec("pyvista") is not None
needs_mesh = pytest.mark.skipif(not HAS_PYVISTA, reason="needs the [mesh] extra")


def test_import_does_not_pull_in_vtk():
    """The core must be usable with no VTK/PyVista installed at all."""
    code = (
        "import sys; import converge_h5_reader; "
        "assert 'vtk' not in sys.modules, 'importing the package imported vtk'; "
        "assert 'pyvista' not in sys.modules, 'importing the package imported pyvista'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unknown_backend():
    with pytest.raises(MissingBackendError, match="unknown mesh backend"):
        converge_h5_reader.mesh.get_backend("nope")


def test_missing_backend_error_names_the_extra(monkeypatch):
    """When vtk cannot be imported, the error must say how to install it."""
    from converge_h5_reader import mesh

    monkeypatch.setattr(mesh, "_REGISTERED", {})

    real_import = importlib.import_module

    def failing_import(name, package=None):
        if "vtk_reader" in name:
            raise ImportError("No module named 'vtk'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", failing_import)

    with pytest.raises(MissingBackendError, match=r"\[mesh\]"):
        mesh.get_backend("vtk")


@needs_mesh
def test_backends_are_discoverable():
    assert "vtk" in converge_h5_reader.mesh.available_backends()


@needs_mesh
@pytest.mark.mesh
def test_polyhedral_backend_builds_the_synthetic_mesh(synthetic_h5):
    from conftest import N_MESH_CELLS, N_MESH_VERTICES

    grid = converge_h5_reader.mesh.read_mesh(
        synthetic_h5, backend="h5", stream=0, fields=["TEMPERATURE"]
    )
    assert grid.n_cells == N_MESH_CELLS
    assert grid.n_points == N_MESH_VERTICES
    assert grid.cell_data["TEMPERATURE"][0] == pytest.approx(300.0)
    assert grid.volume == pytest.approx(float(N_MESH_CELLS))  # unit cubes


@needs_mesh
@pytest.mark.mesh
def test_polyhedral_backend_honours_a_selection(synthetic_h5):
    from converge_h5_reader import Field

    grid = converge_h5_reader.mesh.read_mesh(
        synthetic_h5, backend="h5", stream=0, where=Field("TEMPERATURE") < 350
    )
    assert grid.n_cells == 1
