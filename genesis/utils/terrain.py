import numpy as np
import pickle
import os

import genesis as gs
from genesis.ext import trimesh
from genesis.ext.isaacgym import terrain_utils as isaacgym_terrain_utils
from genesis.options.morphs import Terrain

from .misc import get_assets_dir


def parse_terrain(morph: Terrain, surface):
    """
    Generate mesh (and height field) according to configurations passed by morph.

    ------------------------------------------------------------------------------------------------------
    If morph.height_field is not passed, generate the height field by
        n_subterrains    : Tuple[int, int]     = (3, 3)     # number of subterrains in x and y directions
        subterrain_types : Any                 = [
                ['flat_terrain', 'random_uniform_terrain', 'pyramid_sloped_terrain'],
                ['pyramid_sloped_terrain', 'discrete_obstacles_terrain', 'wave_terrain'],
                ['random_uniform_terrain', 'pyramid_stairs_terrain', 'pyramid_sloped_terrain'],
        ]                                                   # types of subterrains in x and y directions
    If morph.height_field is passed, (n_subterrains, subterrain_size, subterrain_types) will be ignored.
    ------------------------------------------------------------------------------------------------------

    Returns
    --------------------------
    vmesh        : Mesh
    mesh         : Mesh
    height_field : np.ndarray
    """

    if morph.from_stored is not None:
        terrain_dir = os.path.join(os.path.join(get_assets_dir(), f"terrain/{morph.name}"), morph.from_stored)
        os.makedirs(terrain_dir, exist_ok=True)

        tmesh = trimesh.load(os.path.join(terrain_dir, "tmesh.stl"))
        sdf_tmesh = trimesh.load(os.path.join(terrain_dir, "sdf_tmesh.stl"))
        with open(os.path.join(terrain_dir, "info.pkl"), "rb") as f:
            info = pickle.load(f)
            morph.horizontal_scale = info["horizontal_scale"]
            morph.vertical_scale = info["vertical_scale"]
            heightfield = info["height_field"]
    else:
        if morph.height_field is not None:
            heightfield = morph.height_field
        else:
            subterrain_rows = int(morph.subterrain_size[0] / morph.horizontal_scale)
            subterrain_cols = int(morph.subterrain_size[1] / morph.horizontal_scale)
            heightfield = np.zeros(
                np.array(morph.n_subterrains) * np.array([subterrain_rows, subterrain_cols]), dtype=np.int16
            )

            for i in range(morph.n_subterrains[0]):
                for j in range(morph.n_subterrains[1]):
                    subterrain_type = morph.subterrain_types[i][j]

                    new_subterrain = isaacgym_terrain_utils.SubTerrain(
                        width=subterrain_rows,
                        length=subterrain_cols,
                        vertical_scale=morph.vertical_scale,
                        horizontal_scale=morph.horizontal_scale,
                    )
                    if not morph.randomize:
                        saved_state = np.random.get_state()
                        np.random.seed(0)

                    if subterrain_type == "flat_terrain":
                        subterrain_height_field = np.zeros((subterrain_rows, subterrain_cols), dtype=np.int16)

                    elif subterrain_type == "fractal_terrain":
                        subterrain_height_field = fractal_terrain(new_subterrain, levels=8, scale=5.0).height_field_raw

                    elif subterrain_type == "random_uniform_terrain":
                        subterrain_height_field = isaacgym_terrain_utils.random_uniform_terrain(
                            new_subterrain,
                            min_height=-0.1,
                            max_height=0.1,
                            step=0.1,
                            downsampled_scale=0.5,
                        ).height_field_raw

                    elif subterrain_type == "sloped_terrain":
                        subterrain_height_field = isaacgym_terrain_utils.sloped_terrain(
                            new_subterrain,
                            slope=-0.5,
                        ).height_field_raw

                    elif subterrain_type == "pyramid_sloped_terrain":
                        subterrain_height_field = isaacgym_terrain_utils.pyramid_sloped_terrain(
                            new_subterrain,
                            slope=-0.1,
                        ).height_field_raw

                    elif subterrain_type == "discrete_obstacles_terrain":
                        subterrain_height_field = isaacgym_terrain_utils.discrete_obstacles_terrain(
                            new_subterrain,
                            max_height=0.05,
                            min_size=1.0,
                            max_size=5.0,
                            num_rects=20,
                        ).height_field_raw

                    elif subterrain_type == "wave_terrain":
                        subterrain_height_field = isaacgym_terrain_utils.wave_terrain(
                            new_subterrain,
                            num_waves=2.0,
                            amplitude=0.1,
                        ).height_field_raw

                    elif subterrain_type == "stairs_terrain":
                        subterrain_height_field = isaacgym_terrain_utils.stairs_terrain(
                            new_subterrain,
                            step_width=0.75,
                            step_height=-0.1,
                        ).height_field_raw

                    elif subterrain_type == "pyramid_stairs_terrain":
                        subterrain_height_field = isaacgym_terrain_utils.pyramid_stairs_terrain(
                            new_subterrain,
                            step_width=0.75,
                            step_height=-0.1,
                        ).height_field_raw

                    elif subterrain_type == "stepping_stones_terrain":
                        subterrain_height_field = isaacgym_terrain_utils.stepping_stones_terrain(
                            new_subterrain,
                            stone_size=1.0,
                            stone_distance=0.25,
                            max_height=0.2,
                            platform_size=0.0,
                        ).height_field_raw

                    else:
                        gs.raise_exception(f"Unsupported subterrain type: {subterrain_type}")

                    if not morph.randomize:
                        np.random.set_state(saved_state)

                    heightfield[
                        i * subterrain_rows : (i + 1) * subterrain_rows, j * subterrain_cols : (j + 1) * subterrain_cols
                    ] = subterrain_height_field

        tmesh, sdf_tmesh = convert_heightfield_to_watertight_trimesh(
            heightfield,
            horizontal_scale=morph.horizontal_scale,
            vertical_scale=morph.vertical_scale,
        )

        terrain_dir = os.path.join(get_assets_dir(), f"terrain/{morph.name}")
        os.makedirs(terrain_dir, exist_ok=True)

        tmesh.export(os.path.join(terrain_dir, "tmesh.stl"))
        sdf_tmesh.export(os.path.join(terrain_dir, "sdf_tmesh.stl"))
        with open(os.path.join(terrain_dir, "info.pkl"), "wb") as f:
            info = {
                "horizontal_scale": morph.horizontal_scale,
                "vertical_scale": morph.vertical_scale,
                "height_field": heightfield,
            }
            pickle.dump(info, f)

    vmesh = gs.Mesh.from_trimesh(mesh=tmesh, surface=surface)
    mesh = gs.Mesh.from_trimesh(
        mesh=tmesh,
        surface=gs.surfaces.Collision(),
        metadata={
            "horizontal_scale": morph.horizontal_scale,
            "sdf_mesh": sdf_tmesh,
            "height_field": heightfield,
        },
    )
    return vmesh, mesh, heightfield


def fractal_terrain(terrain, levels=8, scale=1.0):
    """
    Generates a fractal terrain

    Parameters
        terrain (SubTerrain): the terrain
        levels (int, optional): granurarity of the fractal terrain. Defaults to 8.
        scale (float, optional): scales vertical variation. Defaults to 1.0.
    """
    width = terrain.width
    length = terrain.length
    height = np.zeros((width, length))
    for level in range(1, levels + 1):
        step = 2 ** (levels - level)
        for y in range(0, width, step):
            y_skip = (1 + y // step) % 2
            for x in range(step * y_skip, length, step * (1 + y_skip)):
                x_skip = (1 + x // step) % 2
                xref = step * (1 - x_skip)
                yref = step * (1 - y_skip)
                mean = height[y - yref : y + yref + 1 : 2 * step, x - xref : x + xref + 1 : 2 * step].mean()
                variation = 2 ** (-level) * np.random.uniform(-1, 1)
                height[y, x] = mean + scale * variation

    height /= terrain.vertical_scale
    terrain.height_field_raw = height.astype(np.int16)
    return terrain


def convert_heightfield_to_watertight_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Adapted from Issac Gym's `convert_heightfield_to_trimesh` function.
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:
        assert False  # our sdf representation doesn't support steep slopes well

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[: num_rows - 1, :] += hf[1:num_rows, :] - hf[: num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[: num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, : num_cols - 1] += hf[:, 1:num_cols] - hf[:, : num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, : num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[: num_rows - 1, : num_cols - 1] += (
            hf[1:num_rows, 1:num_cols] - hf[: num_rows - 1, : num_cols - 1] > slope_threshold
        )
        move_corners[1:num_rows, 1:num_cols] -= (
            hf[: num_rows - 1, : num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        )
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices_top = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices_top[:, 0] = xx.flatten()
    vertices_top[:, 1] = yy.flatten()
    vertices_top[:, 2] = hf.flatten() * vertical_scale
    triangles_top = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles_top[start:stop:2, 0] = ind0
        triangles_top[start:stop:2, 1] = ind3
        triangles_top[start:stop:2, 2] = ind1
        triangles_top[start + 1 : stop : 2, 0] = ind0
        triangles_top[start + 1 : stop : 2, 1] = ind2
        triangles_top[start + 1 : stop : 2, 2] = ind3

    # bottom plane
    z_min = np.min(vertices_top[:, 2]) - 1.0

    vertices_bottom = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices_bottom[:, 0] = xx.flatten()
    vertices_bottom[:, 1] = yy.flatten()
    vertices_bottom[:, 2] = z_min
    triangles_bottom = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles_bottom[start:stop:2, 0] = ind0
        triangles_bottom[start:stop:2, 2] = ind3
        triangles_bottom[start:stop:2, 1] = ind1
        triangles_bottom[start + 1 : stop : 2, 0] = ind0
        triangles_bottom[start + 1 : stop : 2, 2] = ind2
        triangles_bottom[start + 1 : stop : 2, 1] = ind3
    triangles_bottom += num_rows * num_cols

    # side face
    triangles_side_0 = np.zeros([2 * (num_rows - 1), 3], dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = i * num_cols
        ind1 = (i + 1) * num_cols
        ind2 = ind0 + num_rows * num_cols
        ind3 = ind1 + num_rows * num_cols
        triangles_side_0[2 * i] = [ind0, ind2, ind1]
        triangles_side_0[2 * i + 1] = [ind1, ind2, ind3]

    triangles_side_1 = np.zeros([2 * (num_cols - 1), 3], dtype=np.uint32)
    for i in range(num_cols - 1):
        ind0 = i
        ind1 = i + 1
        ind2 = ind0 + num_rows * num_cols
        ind3 = ind1 + num_rows * num_cols
        triangles_side_1[2 * i] = [ind0, ind1, ind2]
        triangles_side_1[2 * i + 1] = [ind1, ind3, ind2]

    triangles_side_2 = np.zeros([2 * (num_rows - 1), 3], dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = i * num_cols + num_cols - 1
        ind1 = (i + 1) * num_cols + num_cols - 1
        ind2 = ind0 + num_rows * num_cols
        ind3 = ind1 + num_rows * num_cols
        triangles_side_2[2 * i] = [ind0, ind1, ind2]
        triangles_side_2[2 * i + 1] = [ind1, ind3, ind2]

    triangles_side_3 = np.zeros([2 * (num_cols - 1), 3], dtype=np.uint32)
    for i in range(num_cols - 1):
        ind0 = i + (num_rows - 1) * num_cols
        ind1 = i + 1 + (num_rows - 1) * num_cols
        ind2 = ind0 + num_rows * num_cols
        ind3 = ind1 + num_rows * num_cols
        triangles_side_3[2 * i] = [ind0, ind2, ind1]
        triangles_side_3[2 * i + 1] = [ind1, ind2, ind3]

    vertices = np.concatenate([vertices_top, vertices_bottom], axis=0)
    triangles = np.concatenate(
        [triangles_top, triangles_bottom, triangles_side_0, triangles_side_1, triangles_side_2, triangles_side_3],
        axis=0,
    )

    # This a uniformly-distributed full mesh, which gives faster sdf generation
    sdf_mesh = trimesh.Trimesh(vertices, triangles, process=False)
    # This is the mesh used for non-sdf purposes. It's losslessly simplified from the full mesh, to save memory cost for storing verts and faces
    mesh = sdf_mesh.simplify_quadric_decimation(face_count=0, lossless=True)

    return mesh, sdf_mesh
