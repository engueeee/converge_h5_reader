from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pyvista") is None, reason="needs the [mesh] extra"
)


@pytest.fixture
def volume(synthetic_h5):
    """The synthetic 1x1x8 hex column, built through the h5 backend."""
    from converge_h5_reader.mesh import read_mesh

    return read_mesh(synthetic_h5, backend="h5", stream=0, fields=["TEMPERATURE", "REGION_ID"])


def test_missing_backend_message_covers_extract(monkeypatch):
    """Without pyvista, reaching for the extract helpers must name the extra."""
    import importlib

    import converge_h5_reader.mesh as mesh_pkg
    from converge_h5_reader import MissingBackendError

    real_import = importlib.import_module

    def failing(name, package=None):
        if name == ".extract":
            raise ImportError("No module named 'pyvista'")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", failing)

    with pytest.raises(MissingBackendError, match=r"\[mesh\]"):
        mesh_pkg.__getattr__("extract")


def test_tumble_slice(volume):
    from converge_h5_reader.mesh import extract

    # The column spans y in [0, 1]; cutting at y = 0.5 gives a 1 x 8 rectangle.
    cut = extract.tumble_slice(volume, y=0.5)
    assert cut.n_cells > 0
    ymin, ymax = cut.bounds[2], cut.bounds[3]
    assert ymin == pytest.approx(0.5) and ymax == pytest.approx(0.5)
    assert cut.area == pytest.approx(8.0)  # 1 wide x 8 tall


def test_axis_slice(volume):
    from converge_h5_reader.mesh import extract

    cut = extract.axis_slice(volume, axis="z", position=4.0)
    assert cut.bounds[4] == pytest.approx(4.0)
    assert cut.area == pytest.approx(1.0)  # the 1x1 cross-section

    with pytest.raises(ValueError, match="axis must be one of"):
        extract.axis_slice(volume, axis="w")


def test_empty_slice_is_an_error(volume):
    from converge_h5_reader.mesh import extract

    with pytest.raises(RuntimeError, match="empty slice"):
        extract.axis_slice(volume, axis="z", position=99.0)


def test_injector_slice_contains_the_axis(volume):
    from converge_h5_reader.mesh import extract

    # An injector firing down the z axis from the top of the column.
    cut = extract.injector_slice(volume, origin=(0.5, 0.5, 8.0), axis=(0.0, 0.0, 1.0))
    assert cut.n_cells > 0
    # The plane contains the axis, so it spans the full height.
    assert cut.bounds[5] - cut.bounds[4] == pytest.approx(8.0)


def test_injector_slice_rejects_a_non_perpendicular_normal(volume):
    from converge_h5_reader.mesh import extract

    with pytest.raises(ValueError, match="perpendicular"):
        extract.injector_slice(
            volume, origin=(0.5, 0.5, 0.0), axis=(0, 0, 1), normal=(0, 0, 1)
        )


def test_clip_box_keeps_the_inside(volume):
    from converge_h5_reader.mesh import extract

    # The column is 1 x 1 x 8; the lower half is exactly 4 volume units.
    lower = extract.clip_box(volume, (0.0, 1.0, 0.0, 1.0, 0.0, 4.0))
    assert lower.volume == pytest.approx(4.0, rel=1e-6)
    assert lower.bounds[5] == pytest.approx(4.0)

    upper = extract.clip_box(volume, (0.0, 1.0, 0.0, 1.0, 0.0, 4.0), invert=True)
    assert upper.volume == pytest.approx(4.0, rel=1e-6)  # the complementary half
    assert upper.bounds[5] == pytest.approx(8.0)


def test_clip_preserves_cell_data(volume):
    from converge_h5_reader.mesh import extract

    lower = extract.clip_box(volume, (0.0, 1.0, 0.0, 1.0, 0.0, 4.0))
    assert "TEMPERATURE" in lower.cell_data


def test_clip_sphere(volume):
    from converge_h5_reader.mesh import extract

    # Centred mid-column, big enough to reach the corners but not the ends.
    ball = extract.clip_sphere(volume, center=(0.5, 0.5, 4.0), radius=2.0)
    assert 0 < ball.volume < volume.volume


def test_clip_cylinder(volume):
    from converge_h5_reader.mesh import extract

    # Radius 0.9 encloses the 1x1 cross-section, so this clips in z only: half the column.
    tube = extract.clip_cylinder(
        volume, center=(0.5, 0.5, 2.0), direction=(0, 0, 1), radius=0.9, height=4.0
    )
    assert tube.volume == pytest.approx(4.0, rel=0.02)


def test_clip_that_selects_nothing_is_an_error(volume):
    from converge_h5_reader.mesh import extract

    with pytest.raises(RuntimeError, match="empty mesh"):
        extract.clip_box(volume, (10.0, 11.0, 10.0, 11.0, 10.0, 11.0))


def test_sample_to_image_and_save_vti(volume, tmp_path):
    import pyvista as pv

    from converge_h5_reader.mesh import extract

    image = extract.sample_to_image(volume, spacing=0.25, fields=["TEMPERATURE"])
    assert isinstance(image, pv.ImageData)
    assert image.dimensions == (5, 5, 33)  # 1 x 1 x 8 at 0.25
    assert set(image.point_data.keys()) == {"TEMPERATURE", "vtkValidPointMask"}
    assert image.point_data["vtkValidPointMask"].any()

    path = extract.save_vti(image, tmp_path / "volume")
    assert path.name == "volume.vti"
    assert pv.read(path).dimensions == image.dimensions


def test_plane_sampling_collapses_to_2d(volume, tmp_path):
    from converge_h5_reader.mesh import extract

    cut = extract.tumble_slice(volume, y=0.5)
    image = extract.sample_to_image(cut, spacing=0.25)
    # The Y extent is zero, so that axis collapses to a single point: a 2D .vti.
    assert image.dimensions[1] == 1

    saved = extract.save_vti(image, tmp_path / "tumble.vti")
    assert saved.exists()


def test_image_grid_validation():
    from converge_h5_reader.mesh import extract

    with pytest.raises(ValueError, match="6 values"):
        extract.image_grid((0, 1, 0, 1), spacing=0.1)
    with pytest.raises(ValueError, match="inverted"):
        extract.image_grid((1, 0, 0, 1, 0, 1), spacing=0.1)
    with pytest.raises(ValueError, match="positive"):
        extract.image_grid((0, 1, 0, 1, 0, 1), spacing=0.0)


def test_to_volume_selects_a_region(synthetic_h5):
    from converge_h5_reader.mesh import extract, read_mesh

    mesh = read_mesh(synthetic_h5, backend="h5", stream=0, fields=["REGION_ID"])
    # The fixture puts 4 of the 8 cells in region 1. This is a plain grid with no
    # PISTONHEAD, so the crevice filter has to be switched off.
    assert extract.to_volume(mesh, region_id=1, exclude_crevice=False).n_cells == 4
    assert extract.to_volume(mesh, exclude_crevice=False).n_cells == 8


def test_drop_below_z(volume):
    from converge_h5_reader.mesh import extract

    # Cell centres sit at z = 0.5, 1.5, ... 7.5, so a floor at 4.0 keeps the top four.
    assert extract.drop_below_z(volume, 4.0).n_cells == 4
    assert extract.drop_below_z(volume, 0.0).n_cells == 8

    with pytest.raises(RuntimeError, match="no cells lie above"):
        extract.drop_below_z(volume, 99.0)


def test_crevice_filter_uses_z_piston(volume):
    from converge_h5_reader.mesh import extract

    kept = extract.to_volume(volume, z_piston=4.0)
    assert kept.n_cells == 4
    assert extract.to_volume(volume, exclude_crevice=False).n_cells == 8


def test_crevice_filter_needs_a_piston_it_can_find(volume):
    from converge_h5_reader.mesh import extract

    # A plain grid with no PISTONHEAD boundary and no explicit z_piston.
    with pytest.raises(ValueError, match="z_piston is needed"):
        extract.to_volume(volume)


def test_piston_crown_z_reports_a_missing_boundary(volume):
    import pyvista as pv

    from converge_h5_reader.mesh import extract

    with pytest.raises(KeyError, match="PISTONHEAD"):
        extract.piston_crown_z(pv.MultiBlock({"Mesh": volume}))


def test_sample_to_image_raises_the_grid_floor(volume):
    from converge_h5_reader.mesh import extract

    full = extract.sample_to_image(volume, spacing=0.5)
    floored = extract.sample_to_image(volume, spacing=0.5, z_piston=4.0)

    assert floored.origin[2] == pytest.approx(4.0)
    assert floored.dimensions[2] < full.dimensions[2]

    with pytest.raises(ValueError, match="above the source's top"):
        extract.sample_to_image(volume, spacing=0.5, z_piston=99.0)
