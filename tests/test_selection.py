from __future__ import annotations

import numpy as np
import pytest

from converge_h5_reader import ConvergeFile, Field, region
from converge_h5_reader.selection import as_mask, read_masked


@pytest.fixture
def stream(synthetic_h5):
    with ConvergeFile(synthetic_h5) as f:
        yield f[0]


def test_region_selects_the_known_cells(stream):
    mask = region(1).evaluate(stream.cells)
    assert mask.sum() == 4

    frame = stream.cells.to_dataframe(["T"], where=region(1), add_time=False, add_cad=False)
    assert len(frame) == 4
    np.testing.assert_allclose(frame["T"], [300.0, 400.0, 500.0, 600.0])


def test_composite_predicates(stream):
    predicate = (Field("REGION_ID") == 1) & (Field("TEMPERATURE") > 450)
    values = stream.cells.read("T", where=predicate)
    np.testing.assert_allclose(values, [500.0, 600.0])

    either = (Field("REGION_ID") == 2) | (Field("TEMPERATURE") < 350)
    assert either.evaluate(stream.cells).sum() == 3  # cells 4, 5 and 0

    assert (~(Field("REGION_ID") == 1)).evaluate(stream.cells).sum() == 4


def test_boolean_array_as_mask(stream):
    mask = np.zeros(8, dtype=bool)
    mask[[1, 3]] = True
    np.testing.assert_allclose(stream.cells.read("T", where=mask), [400.0, 600.0])


def test_bad_masks_are_rejected(stream):
    with pytest.raises(ValueError, match="expected"):
        stream.cells.read("T", where=np.ones(3, dtype=bool))
    with pytest.raises(TypeError, match="boolean"):
        stream.cells.read("T", where=np.ones(8, dtype=np.int32))


def test_both_read_strategies_agree(stream):
    """The contiguous and point-selection branches must return identical data."""
    mask = as_mask(region(1), stream.cells, stream.n_cells)
    dataset = stream.cells.dataset("T")

    contiguous = read_masked(dataset, mask, threshold=0.0)  # forces ds[...][mask]
    point_select = read_masked(dataset, mask, threshold=1.0)  # forces ds[indices]
    np.testing.assert_array_equal(contiguous, point_select)


def test_empty_selection(stream):
    values = stream.cells.read("T", where=Field("REGION_ID") == 99)
    assert values.size == 0
    assert values.dtype == np.float32


def test_legacy_idreg_fallback_warns(legacy_region_h5):
    with ConvergeFile(legacy_region_h5) as f, pytest.warns(DeprecationWarning, match="IDREG"):
        assert region(1).evaluate(f[0].cells).sum() == 4
