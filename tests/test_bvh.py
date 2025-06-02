import genesis as gs
import pytest

from genesis.engine.bvh import AABB, LBVH


def test_aabb():
    aabb = AABB(min=gs.ti_vec3(0, 0, 0), max=gs.ti_vec3(1, 1, 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
