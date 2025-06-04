import taichi as ti
import torch
import numpy as np
import genesis as gs


import pytest
from .utils import assert_allclose


@pytest.fixture(scope="function")
def lbvh():
    """Fixture for a LBVH tree"""
    from genesis.engine.bvh import LBVH, AABB

    n_aabbs = 10
    n_batches = 1
    tree = LBVH(n_aabbs=n_aabbs, n_batches=n_batches)
    min = np.random.rand(n_aabbs, n_batches, 3).astype(np.float32)
    max = min + np.random.rand(n_aabbs, n_batches, 3).astype(np.float32)
    aabbs = AABB.field(
        shape=(n_aabbs, n_batches),
        needs_grad=False,
        layout=ti.Layout.SOA,
    )
    aabbs.min.from_numpy(min)
    aabbs.max.from_numpy(max)
    tree.build(aabbs)
    return tree, aabbs


def test_morton_code(lbvh):
    morton_codes = lbvh.morton_codes.to_numpy()
    # Check that the morton codes are sorted
    for i in range(1, morton_codes.shape[0]):
        assert (
            morton_codes[i] > morton_codes[i - 1]
        ), f"Morton codes are not sorted: {morton_codes[i]} < {morton_codes[i - 1]}"


@ti.kernel
def expand_bits(lbvh: ti.template(), x: ti.template(), expanded_x: ti.template()):
    n_x = x.shape[0]
    for i in range(n_x):
        expanded_x[i] = lbvh.expand_bits(x[i])


def test_expand_bits():
    # random integer
    x_np = np.random.randint(0, 1024, (10,), dtype=np.uint32)
    x_ti = ti.field(ti.uint32, shape=x_np.shape)
    x_ti.from_numpy(x_np)
    expanded_x_ti = ti.field(ti.uint32, shape=x_np.shape)
    # expand bits
    from genesis.engine.bvh import LBVH

    lbvh = LBVH(n_aabbs=10, n_batches=1)
    expand_bits(lbvh, x_ti, expanded_x_ti)
    expanded_x_np = expanded_x_ti.to_numpy()
    for i in range(x_np.shape[0]):
        str_x = f"{x_np[i]:010b}"
        str_expanded_x = f"{expanded_x_np[i]:030b}"
        # check that the expanded bits are correct
        assert str_expanded_x == "".join(
            f"00{bit}" for bit in str_x
        ), f"Expected {str_expanded_x}, got {''.join(f'00{bit}' for bit in str_x)}"


def test_build_tree(lbvh):
    tree, aabbs = lbvh

    nodes = tree.nodes.to_numpy()
    n_aabbs = tree.n_aabbs
    n_batches = tree.n_batches

    # Check parent-child relationships
    for i in range(n_aabbs * 2 - 1):
        for j in range(n_batches):
            parent = nodes["parent"][i, j]
            left = nodes["left"][i, j]
            right = nodes["right"][i, j]

            # Check that the parent is correct
            if i == 0:  # root node has no parent
                assert parent == -1

            else:
                assert (
                    nodes["left"][parent, j] == i or nodes["right"][parent, j] == i
                ), f"Node {i} in batch {j} has incorrect parent: {parent}"

            # Check that left and right children are correct
            if left != -1:
                assert (
                    nodes["parent"][left, j] == i
                ), f"Left child {left} of node {i} in batch {j} has incorrect parent: {nodes['parent'][left, j]}, expected {i}"
            if right != -1:
                assert (
                    nodes["parent"][right, j] == i
                ), f"Right child {right} of node {i} in batch {j} has incorrect parent: {nodes['parent'][right, j]}, expected {i}"

            if left != -1 and right != -1:
                # Check that the AABBs of the children are within the AABB of the parent
                parent_min = nodes["bound"]["min"][i, j]
                parent_max = nodes["bound"]["max"][i, j]
                left_min = nodes["bound"]["min"][left, j]
                left_max = nodes["bound"]["max"][left, j]
                right_min = nodes["bound"]["min"][right, j]
                right_max = nodes["bound"]["max"][right, j]

                parent_min_expected = np.minimum(left_min, right_min)
                parent_max_expected = np.maximum(left_max, right_max)
                assert_allclose(parent_min, parent_min_expected, rtol=1e-5, atol=1e-5)
                assert_allclose(parent_max, parent_max_expected, rtol=1e-5, atol=1e-5)


def test_query(lbvh):
    tree, aabbs = lbvh

    # Query the tree
    tree.query(aabbs)

    query_result_count = tree.query_result_count.to_numpy()
    query_result = tree.query_result.to_numpy()

    n_aabbs = tree.n_aabbs
    n_batches = tree.n_batches

    # Check that the query results are correct
    for i_b in range(n_batches):
        intersect = np.zeros((n_aabbs, n_aabbs), dtype=bool)
        for i_a in range(n_aabbs):
            count = query_result_count[i_a, i_b]
            for j in range(count):
                intersect[i_a, query_result[j, i_a, i_b]] = True

        for i_a in range(n_aabbs):
            for j_a in range(n_aabbs):
                if i_a == j_a:
                    assert intersect[i_a, j_a] == True, f"AABB {i_a} should intersect with itself"
                else:
                    assert (
                        intersect[i_a, j_a] == intersect[j_a, i_a]
                    ), f"AABBs {i_a} and {j_a} should have the same intersection result"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
