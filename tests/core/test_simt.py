import numpy as np
import pytest
import quadrants as qd

import genesis as gs
from genesis.utils.misc import qd_to_numpy
from genesis.utils.simt import qd_block_scan


@qd.kernel
def _scan_128_lanes(
    values: qd.types.ndarray(dtype=qd.i32, ndim=1),
    prefixes: qd.types.ndarray(dtype=qd.i32, ndim=1),
    totals: qd.types.ndarray(dtype=qd.i32, ndim=1),
):
    for lane in range(128):
        prefix, total = qd_block_scan(values[lane])
        prefixes[lane] = prefix
        totals[lane] = total


def test_block_scan_keeps_each_32_lane_tile_independent():
    """Catch a full-wave scan crossing Genesis's logical 32-lane blocks."""
    if gs.backend == gs.cpu:
        pytest.skip("SIMT subgroup scan requires a GPU backend")

    values = qd.ndarray(dtype=qd.i32, shape=(128,))
    prefixes = qd.ndarray(dtype=qd.i32, shape=(128,))
    totals = qd.ndarray(dtype=qd.i32, shape=(128,))
    values.from_numpy(np.ones(128, dtype=np.int32))

    _scan_128_lanes(values, prefixes, totals)
    qd.sync()

    actual_prefixes = qd_to_numpy(prefixes)
    actual_totals = qd_to_numpy(totals)
    expected_prefixes = np.tile(np.arange(1, 33, dtype=np.int32), 4)
    expected_totals = np.full(128, 32, dtype=np.int32)

    np.testing.assert_array_equal(actual_prefixes, expected_prefixes)
    np.testing.assert_array_equal(actual_totals, expected_totals)
