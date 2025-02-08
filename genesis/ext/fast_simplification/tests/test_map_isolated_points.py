import numpy as np

from fast_simplification import _map_isolated_points as map_isolated_points


def test_map_isolated_points():
    # Example 1
    #
    #      (1)
    #     / | \
    #  (0)  |  (2)-3
    #     \ | /  \
    #      (4)    6-9
    #       |
    #       5     8-7

    points = np.random.rand(10, 3)

    edges = np.array(
        [
            [0, 1],
            [0, 4],
            [1, 4],
            [1, 2],
            [2, 4],
            [2, 3],
            [2, 6],
            [6, 9],
            [4, 5],
            [8, 7],
        ],
        dtype=np.int64,
    )

    triangles = np.array(
        [
            [0, 1, 4],
            [1, 2, 4],
        ],
        dtype=np.int64,
    )

    target_mapping = np.array([0, 1, 2, 2, 4, 4, 2, 7, 8, 2], dtype=np.int64)

    target_merged_points = np.array([3, 5, 6, 9], dtype=np.int64)

    mapping, merged_points = map_isolated_points(points, edges, triangles)
    assert np.allclose(mapping, target_mapping)
    assert np.allclose(merged_points, target_merged_points)

    # Example 2
    #
    # (7)-(8)       3
    #   \  |
    #    \ |
    #     (0)-1-2-9-4-5-6

    points = np.random.rand(10, 3)

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 9],
            [9, 4],
            [4, 5],
            [5, 6],
        ],
        dtype=np.int64,
    )

    triangles = np.array(
        [
            [0, 7, 8],
        ]
    )

    target_mapping = np.array([0, 0, 0, 3, 0, 0, 0, 7, 8, 0], dtype=np.int64)

    target_merged_points = np.array([1, 2, 4, 5, 6, 9], dtype=np.int64)

    mapping, merged_points = map_isolated_points(points, edges, triangles)
    assert np.allclose(mapping, target_mapping)
    assert np.allclose(merged_points, target_merged_points)

    # Example 3
    #
    # (1)
    #  | \
    #  |  \
    # (2)-(0)-4-6
    #         | |
    #         3-5

    points = np.random.rand(7, 3)

    edges = np.array([[0, 1], [1, 2], [2, 0], [0, 4], [4, 6], [5, 6], [3, 5]], dtype=np.int64)

    triangles = np.array(
        [
            [0, 1, 2],
        ],
        dtype=np.int64,
    )

    target_mapping = np.array([0, 1, 2, 0, 0, 0, 0], dtype=np.int64)

    target_merged_points = np.array([3, 4, 5, 6], dtype=np.int64)

    mapping, merged_points = map_isolated_points(points, edges, triangles)
    assert np.allclose(mapping, target_mapping)
    assert np.allclose(merged_points, target_merged_points)

    ## Example 4
    #
    # (6)           (7)-(8)
    #  | \           |  /
    #  |  \          | /
    # (5)-(0)-1-2-3-(4)
    #
    # Here the situation is ambiguous. Does 2 merge into 0 or 4 ?
    # We consider 2 -> 4 and 2 -> 0 as valid solutions.

    points = np.random.rand(9, 3)

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
        ],
        dtype=np.int64,
    )

    triangles = np.array([[0, 5, 6], [4, 7, 8]], dtype=np.int64)

    target_mapping1 = np.array([0, 0, 0, 4, 4, 5, 6, 7, 8], dtype=np.int64)

    target_mapping2 = np.array([0, 0, 4, 4, 4, 5, 6, 7, 8], dtype=np.int64)

    target_merged_points = np.array([1, 2, 3], dtype=np.int64)

    mapping, merged_points = map_isolated_points(points, edges, triangles)
    assert np.allclose(mapping, target_mapping1) or np.allclose(mapping, target_mapping2)
    assert np.allclose(merged_points, target_merged_points)

    ## Example 5
    #
    # (1)            (7)-(8)-9
    #  | \            |  /
    #  |  \           | /
    # (2)-(3)-0  4-5-(6)

    points = np.random.rand(10, 3)
    edges = np.array(
        [
            [0, 3],
            [1, 3],
            [2, 3],
            [1, 2],
            [4, 5],
            [5, 6],
            [8, 9],
        ],
        dtype=np.int64,
    )
    triangles = np.array(
        [
            [1, 2, 3],
            [6, 7, 8],
        ],
        dtype=np.int64,
    )

    target_mapping = np.array([3, 1, 2, 3, 6, 6, 6, 7, 8, 8], dtype=np.int64)

    target_merged_points = np.array([0, 4, 5, 9], dtype=np.int64)

    mapping, merged_points = map_isolated_points(points, edges, triangles)
    assert np.allclose(mapping, target_mapping)
    assert np.allclose(merged_points, target_merged_points)

    ## Example 6
    #
    # 0-1-2

    points = np.random.rand(3, 3)

    edges = np.array(
        [
            [0, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )

    triangles = np.array([[]], dtype=np.int64)

    target_mapping = np.array([0, 1, 2], dtype=np.int64)

    target_merged_points = np.array([], dtype=np.int64)

    mapping, merged_points = map_isolated_points(points, edges, triangles)
    assert np.allclose(mapping, target_mapping)
    assert np.allclose(merged_points, target_merged_points)
