import os
import pickle as pkl

import numpy as np

import genesis as gs
from genesis.ext import trimesh

from . import geom as gu
from . import mesh as mu


def box_to_elements(pos=(0, 0, 0), size=(1, 1, 1), tet_cfg=dict()):
    trimesh_obj = trimesh.creation.box(extents=size)
    trimesh_obj.vertices += np.array(pos)
    verts, elems = mu.tetrahedralize_mesh(trimesh_obj, tet_cfg)

    return verts, elems


def sphere_to_elements(pos=(0, 0, 0), radius=0.5, tet_cfg=dict()):
    trimesh_obj = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    trimesh_obj.vertices *= np.array(radius)
    trimesh_obj.vertices += np.array(pos)
    verts, elems = mu.tetrahedralize_mesh(trimesh_obj, tet_cfg)

    return verts, elems


def cylinder_to_elements():
    raise NotImplementedError


def mesh_to_elements(file, pos=(0, 0, 0), scale=1.0, tet_cfg=dict()):
    mesh = mu.load_mesh(file)
    mesh.vertices = mesh.vertices * scale

    tet_file_path = mu.get_tet_path(mesh.vertices, mesh.faces, tet_cfg)
    if not os.path.exists(tet_file_path):
        with gs.logger.timer(f"Tetrahedralization with configuration {tet_cfg} and generating `.tet` file:"):
            verts, elems = mu.tetrahedralize_mesh(mesh, tet_cfg)

            os.makedirs(os.path.dirname(tet_file_path), exist_ok=True)
            pkl.dump((verts, elems), open(tet_file_path, "wb"))
    else:
        gs.logger.debug("Tetrahedra (`.tet`) found in cache.")
        verts, elems = pkl.load(open(tet_file_path, "rb"))

    verts += np.array(pos)

    return verts, elems
