import os
import pickle as pkl
from itertools import chain
from typing import TYPE_CHECKING

import quadrants as ti
import igl
import numpy as np
import skimage
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.repr_base import RBC
from genesis.utils.misc import tensor_to_array, ti_to_torch, DeprecationError

if TYPE_CHECKING:
    from genesis.engine.materials.rigid import Rigid as RigidMaterial
    from genesis.engine.mesh import Mesh
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

    from .rigid_entity import RigidEntity
    from .rigid_link import RigidLink


NUM_VERTS_VISUAL_GEOM_AABB = 200


@ti.data_oriented
class RigidGeom(RBC):
    """
    A `RigidGeom` is the basic building block of a `RigidEntity` for collision checking. It is usually constructed from a single mesh. This can be accessed via `link.geoms`.
    """

    def __init__(
        self,
        link: "RigidLink",
        idx,
        cell_start: int,
        vert_start: int,
        face_start: int,
        edge_start: int,
        verts_state_start: int,
        mesh: "Mesh",
        type: gs.GEOM_TYPE,
        friction: float,
        sol_params,
        init_pos,
        init_quat,
        needs_coup: bool,
        contype,
        conaffinity,
        center_init=None,
        data=None,
    ):
        self._link: "RigidLink" = link
        self._entity: "RigidEntity" = link.entity
        self._material: "RigidMaterial" = link.entity.material
        self._solver: "RigidSolver" = link.entity.solver
        self._mesh: "Mesh" = mesh

        self._uid = gs.UID()
        self._idx = idx
        self._type: gs.GEOM_TYPE = type
        self._friction: float = friction
        self._sol_params = sol_params
        self._needs_coup: bool = needs_coup
        self._contype = int(contype)
        self._conaffinity = int(conaffinity)
        self._is_convex: bool = mesh.is_convex
        self._cell_start: int = cell_start
        self._vert_start: int = vert_start
        self._face_start: int = face_start
        self._edge_start: int = edge_start
        self._verts_state_start: int = verts_state_start

        self._coup_softness: float = self._material.coup_softness
        self._coup_friction: float = self._material.coup_friction
        self._coup_restitution: float = self._material.coup_restitution

        self._init_pos: np.ndarray = init_pos
        self._init_quat: np.ndarray = init_quat

        # For heterogeneous simulation: which environments this geom is active in (None = all envs)
        self.active_envs_mask: torch.Tensor | None = None
        self.active_envs_idx: np.ndarray | None = None

        self._init_verts = mesh.verts
        self._init_faces = mesh.faces
        self._init_edges = mesh.get_unique_edges()
        self._init_normals = mesh.normals
        self._uvs = mesh.uvs
        self._surface = mesh.surface
        self._metadata = mesh.metadata

        if center_init is None:
            self._init_center_pos = np.repeat(
                self._init_verts.mean(0, keepdims=True), repeats=self._init_verts.shape[0], axis=0
            )
        else:
            self._init_center_pos = np.array(center_init)
        self._data = np.zeros([7])
        if data is not None:
            self._data[: len(data)] = data

        # verts and faces for sdf genertaion
        if "sdf_mesh" in self._metadata:
            self._sdf_verts = np.ascontiguousarray(self._metadata["sdf_mesh"].vertices)
            self._sdf_faces = np.ascontiguousarray(self._metadata["sdf_mesh"].faces)
        else:
            self._sdf_verts = np.array(self._init_verts)
            self._sdf_faces = np.array(self._init_faces)

        if len(self._sdf_faces) > 50000:
            mesh_descr = f"({mesh.metadata['mesh_path']})" if "mesh_path" in mesh.metadata else ""
            gs.logger.warning(
                f"Beware that SDF pre-processing of mesh {mesh_descr} having more than 50000 vertices may take a very "
                "long time (>10min) and require large RAM allocation (>20Gb). Please either enable convexify or "
                "decimation. (see FileMorph options)"
            )

        # Compute adjacency graph
        tmesh = trimesh.Trimesh(vertices=self._init_verts, faces=self._init_faces, process=False)
        all_vert_neighbors_list = tmesh.vertex_neighbors
        assert self.n_verts == len(all_vert_neighbors_list)
        self.vert_neighbors = np.array(tuple(chain.from_iterable(all_vert_neighbors_list)), dtype=gs.np_int)
        self.vert_n_neighbors = np.array(tuple(map(len, all_vert_neighbors_list)), dtype=gs.np_int)
        self.vert_neighbor_start = np.array((0, *np.cumsum(self.vert_n_neighbors)[:-1]), dtype=gs.np_int)

        # NOTE: sdf size is from the center of the lower voxel cell to the center of the upper voxel cell
        # add padding. Adjust the cell size to keep resolution within bounds.
        padding_ratio = 0.2
        lower = self._init_verts.min(axis=0)
        upper = self._init_verts.max(axis=0)
        grid_size = (upper - lower).max() * padding_ratio + (upper - lower)
        self._sdf_cell_size = gs.EPS + np.clip(
            self._material.sdf_cell_size,
            grid_size.max() / (self._material.sdf_max_res - 1),
            grid_size.min() / max(self._material.sdf_min_res - 1, 2),
        )
        self._sdf_res = np.ceil(grid_size / self._sdf_cell_size).astype(gs.np_int) + 1
        self._sdf_grad_delta = 0.0 if self.type == gs.GEOM_TYPE.TERRAIN else self._sdf_cell_size * 1e-2
        self._is_preprocessed = False

    def _build(self):
        pass

    def _preprocess(self):
        # compute file name via hashing for caching
        self._gsd_path = mu.get_gsd_path(
            self._init_verts,
            self._init_faces,
            self._material.sdf_cell_size,
            self._material.sdf_min_res,
            self._material.sdf_max_res,
        )

        # loading pre-computed cache if available
        is_cached_loaded = False
        if os.path.exists(self._gsd_path):
            gs.logger.debug(f"Preprocessed file (`.gsd`) found in cache for geom idx {self._idx}.")
            try:
                with open(self._gsd_path, "rb") as file:
                    gsd_dict = pkl.load(file)
                is_cached_loaded = True
            except (EOFError, ModuleNotFoundError, pkl.UnpicklingError, TypeError, MemoryError):
                gs.logger.info("Ignoring corrupted cache.")

        if not is_cached_loaded:
            with gs.logger.timer(f"Preprocessing geom idx ~~<{self._idx}>~~."):
                ######## sdf ########
                lower = self._init_verts.min(axis=0)
                upper = self._init_verts.max(axis=0)
                center = (upper + lower) / 2.0

                # NOTE: sdf size is from the center of the lower voxel cell to the center of the upper voxel cell
                # add padding. Adjust the cell size to keep resolution within bounds.
                padding_ratio = 0.2
                grid_size = (upper - lower).max() * padding_ratio + (upper - lower)

                # round up to multiple of sdf_cell_size
                grid_size = (self._sdf_res - 1) * self._sdf_cell_size

                halfsize = grid_size / 2.0
                voxel_lower = center - halfsize
                voxel_upper = center + halfsize

                x = np.linspace(voxel_lower[0], voxel_upper[0], self._sdf_res[0])
                y = np.linspace(voxel_lower[1], voxel_upper[1], self._sdf_res[1])
                z = np.linspace(voxel_lower[2], voxel_upper[2], self._sdf_res[2])
                X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
                query_points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))

                sdf_val = self._compute_sd(query_points)
                sdf_closest_vert = self._compute_closest_verts(query_points)
                sdf_val = sdf_val.reshape(self._sdf_res)
                sdf_closest_vert = sdf_closest_vert.reshape(self._sdf_res)
                T_mesh_to_centered = np.eye(4)
                T_mesh_to_centered[:3, 3] = -center
                T_centered_to_sdf = np.eye(4)
                T_centered_to_sdf[:3, :3] *= (self._sdf_res - 1) / grid_size
                T_centered_to_sdf[:3, 3] = (self._sdf_res - 1) / 2
                T_mesh_to_sdf = T_centered_to_sdf @ T_mesh_to_centered

                ######## sdf gradient ########
                if self.type == gs.GEOM_TYPE.TERRAIN:  # terrain uses finite difference for sdf gradient computation
                    # dummy
                    sdf_grad = np.zeros((*self._sdf_res, 3), dtype=gs.np_float)
                else:
                    sdf_grad = self._compute_sd_grad(query_points, self._sdf_grad_delta).reshape((*self._sdf_res, 3))

                # caching
                gsd_dict = {
                    "sdf_val": sdf_val,
                    "sdf_grad": sdf_grad,
                    "sdf_max": np.max(sdf_val),
                    "sdf_closest_vert": sdf_closest_vert,
                    "T_mesh_to_sdf": T_mesh_to_sdf,
                }
                os.makedirs(os.path.dirname(self._gsd_path), exist_ok=True)
                with open(self._gsd_path, "wb") as file:
                    pkl.dump(gsd_dict, file)

        self._sdf_val = gsd_dict["sdf_val"]
        self._sdf_grad = gsd_dict["sdf_grad"]
        self._sdf_max = gsd_dict["sdf_max"]
        self._sdf_closest_vert = gsd_dict["sdf_closest_vert"]
        self._T_mesh_to_sdf = gsd_dict["T_mesh_to_sdf"]

        self._is_preprocessed = True

    def _compute_sd(self, query_points):
        signed_distance, *_ = igl.signed_distance(query_points, self._sdf_verts, self._sdf_faces)
        return signed_distance.astype(gs.np_float, copy=False)

    def _compute_closest_verts(self, query_points):
        _, closest_faces, *_ = igl.signed_distance(query_points, self._init_verts, self._init_faces)
        verts_ids = self._init_faces[closest_faces]
        verts_ids = verts_ids[
            np.arange(len(query_points), dtype=gs.np_int),
            np.argmin(np.linalg.norm(self._init_verts[verts_ids] - query_points[:, None, :], axis=-1), axis=-1),
        ]
        return verts_ids

    def _compute_sd_grad(self, query_points, delta=5e-4):
        ######## sdf gradient via finite differencing ########
        sd_val_xpos = self._compute_sd(query_points + np.array((delta, 0.0, 0.0), dtype=gs.np_float))
        sd_val_xneg = self._compute_sd(query_points - np.array((delta, 0.0, 0.0), dtype=gs.np_float))
        sd_val_ypos = self._compute_sd(query_points + np.array((0.0, delta, 0.0), dtype=gs.np_float))
        sd_val_yneg = self._compute_sd(query_points - np.array((0.0, delta, 0.0), dtype=gs.np_float))
        sd_val_zpos = self._compute_sd(query_points + np.array((0.0, 0.0, delta), dtype=gs.np_float))
        sd_val_zneg = self._compute_sd(query_points - np.array((0.0, 0.0, delta), dtype=gs.np_float))

        sd_grad_x = (sd_val_xpos - sd_val_xneg) / (2 * delta)
        sd_grad_y = (sd_val_ypos - sd_val_yneg) / (2 * delta)
        sd_grad_z = (sd_val_zpos - sd_val_zneg) / (2 * delta)
        sd_grad = np.stack([sd_grad_x, sd_grad_y, sd_grad_z], -1)
        return sd_grad

    def get_trimesh(self):
        """
        Get the geom's trimesh object.
        """
        return self._mesh.trimesh

    def get_sdf_trimesh(self, color=[1.0, 1.0, 0.6, 1.0]):
        """
        Reconstruct trimesh object from sdf.
        """
        if self.sdf_val.min() >= 0:
            gs.logger.warning("SDF is positive everywhere. Returning empty mesh.")
            return trimesh.Trimesh(vertices=[], faces=[])
        else:
            vertices, faces, _, _ = skimage.measure.marching_cubes(self.sdf_val, level=0)
            vertices = (np.linalg.inv(self.T_mesh_to_sdf) @ np.hstack([vertices, np.ones([len(vertices), 1])]).T)[:3].T
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(np.array([color]), [len(vertices), 1]))
            return mesh

    def visualize_sdf(
        self,
        pos=None,
        T=None,
        color=(1.0, 1.0, 0.3, 1.0),
        show_axis=False,
        axis_color=(1.0, 0.0, 0.0, 1.0),
        axis_length=0.3,
        show_boundary=False,
        boundary_color=(0.0, 1.0, 0.0, 0.2),
    ):
        """
        Visualizes the signed distance field (SDF) of the rigid geometry in the viewer.
        """
        if self._solver.scene.viewer is None:
            gs.raise_exception("Viewer is not available.")

        if self._solver.n_envs > 0:
            gs.raise_exception("Batched scene does not support sdf visualization.")

        sdf_mesh = self.get_sdf_trimesh(color)
        if T is None:
            if pos is None:
                T = gu.trans_quat_to_T(tensor_to_array(self.get_pos()), tensor_to_array(self.get_quat()))
            else:
                T = gu.trans_to_T(np.array(pos))
        else:
            T = np.array(T)
        self._solver.scene.draw_debug_mesh(sdf_mesh, T=T)

        if show_axis:
            axis_mesh = trimesh.creation.axis(origin_size=0.01, axis_radius=0.005, axis_length=axis_length)
            axis_mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(np.array([axis_color]), [len(axis_mesh.vertices), 1])
            )
            self._solver.scene.draw_debug_mesh(axis_mesh, T=T)

        if show_boundary:
            boundary_mesh = trimesh.creation.box(extents=[1, 1, 1])
            boundary_mesh.vertices = gu.inv_transform_by_T(
                (boundary_mesh.vertices + 0.5) * (self.sdf_res - 1), self.T_mesh_to_sdf
            )
            boundary_mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile(np.array([boundary_color]), [len(boundary_mesh.vertices), 1])
            )
            self._solver.scene.draw_debug_mesh(boundary_mesh, T=T)

    def set_friction(self, friction):
        """
        Set the friction coefficient of this geometry.
        """
        if friction < 0:
            gs.raise_exception("`friction` must be non-negative.")
        self._friction = friction

        if self._solver.is_built:
            self._solver.set_geom_friction(friction, self._idx)

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_pos(self, envs_idx=None):
        """
        Get the position of the geom in world frame.
        """
        tensor = ti_to_torch(self._solver.geoms_state.pos, envs_idx, self._idx, transpose=True, copy=True)[..., 0, :]
        return tensor[0] if self._solver.n_envs == 0 else tensor

    @gs.assert_built
    def get_quat(self, envs_idx=None):
        """
        Get the quaternion of the geom in world frame.
        """
        tensor = ti_to_torch(self._solver.geoms_state.quat, envs_idx, self._idx, transpose=True, copy=True)[..., 0, :]
        return tensor[0] if self._solver.n_envs == 0 else tensor

    @gs.assert_built
    def get_verts(self):
        """
        Get the vertices of the geom in world frame.
        """
        self._solver.update_verts_for_geoms(self._idx)

        verts_idx = slice(self.verts_state_start, self.verts_state_end)
        if self.is_fixed and not self._entity._batch_fixed_verts:
            tensor = ti_to_torch(self._solver.fixed_verts_state.pos, verts_idx, copy=True)
        else:
            tensor = ti_to_torch(self._solver.free_verts_state.pos, None, verts_idx, transpose=True, copy=True)
            if self._solver.n_envs == 0:
                tensor = tensor[0]
        return tensor

    @gs.assert_built
    def get_AABB(self):
        """
        Get the axis-aligned bounding box (AABB) of the geom in world frame.
        """
        verts = self.get_verts()
        return torch.stack((verts.min(dim=-2).values, verts.max(dim=-2).values), dim=-2)

    def set_sol_params(self, sol_params):
        """
        Set the solver parameters of this geometry.
        """
        if self._solver.is_built:
            self._solver.set_sol_params(sol_params, geoms_idx=self._idx, envs_idx=None)
        else:
            self._sol_params = sol_params

    @property
    def sol_params(self):
        """
        Get the solver parameters of this geometry.
        """
        if self._solver.is_built:
            return self._solver.get_sol_params(geoms_idx=self._idx, envs_idx=None)[0]
        return self._sol_params

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        Get the unique ID of the geom.
        """
        return self._uid

    @property
    def idx(self) -> int:
        """
        Get the global index of the geom in RigidSolver.
        """
        return self._idx

    @property
    def type(self) -> gs.GEOM_TYPE:
        """
        Get the type of the geom.
        """
        return self._type

    @property
    def friction(self):
        """
        Get the friction coefficient of the geom.
        """
        return self._friction

    @property
    def data(self):
        """
        Get the additional data of the geom.
        """
        return self._data

    @property
    def metadata(self):
        """
        Get the metadata of the geom.
        """
        return self._metadata

    @property
    def link(self) -> "RigidLink":
        """
        Get the link that the geom belongs to.
        """
        return self._link

    @property
    def entity(self) -> "RigidEntity":
        """
        Get the entity that the geom belongs to.
        """
        return self._entity

    @property
    def solver(self) -> "RigidSolver":
        """
        Get the solver that the geom belongs to.s
        """
        return self._solver

    @property
    def is_convex(self) -> bool:
        """
        Get whether the geom is convex.
        """
        return self._is_convex

    @property
    def mesh(self) -> "Mesh":
        return self._mesh

    @property
    def needs_coup(self) -> bool:
        """
        Get whether the geom needs coupling with other non-rigid entities.
        """
        return self._needs_coup

    @property
    def contype(self) -> int:
        """
        Get the contact type of the geometry for collision pair filtering.

        The two geoms are deemed "compatible" (i.e. collisions between them is allowed) if the 'contype' of one geom
        and the 'conaffinity' of the other geom have a common bit set to 1, i.e.
        `(geom1.contype & geom2.conaffinity) || (geom2.contype & geom1.conaffinity) == True`. This is a powerful
        mechanism borrowed from Open Dynamics Engine.
        """
        return self._contype

    @property
    def conaffinity(self) -> int:
        """
        Get the contact affinity of the geometry for collision pair filtering.

        See `contype` documentation for details.
        """
        return self._conaffinity

    @property
    def coup_softness(self) -> float:
        """
        Get the softness coefficient of the geom for coupling.
        """
        return self._coup_softness

    @property
    def coup_friction(self) -> float:
        """
        Get the friction coefficient of the geom for coupling.
        """
        return self._coup_friction

    @property
    def coup_restitution(self) -> float:
        """
        Get the restitution coefficient of the geom for coupling.
        """
        return self._coup_restitution

    @property
    def init_pos(self) -> np.ndarray:
        """
        Get the initial position of the geom.
        """
        return self._init_pos

    @property
    def init_quat(self) -> np.ndarray:
        """
        Get the initial quaternion of the geom.
        """
        return self._init_quat

    @property
    def init_verts(self):
        """
        Get the initial vertices of the geom.
        """
        return self._init_verts

    @property
    def init_faces(self):
        """
        Get the initial faces of the geom.
        """
        return self._init_faces

    @property
    def init_edges(self):
        """
        Get the initial edges of the geom.
        """
        return self._init_edges

    @property
    def init_normals(self):
        """
        Get the initial normals of the geom.
        """
        return self._init_normals

    @property
    def init_center_pos(self):
        """
        Get the initial center position of the geom.
        """
        return self._init_center_pos

    @property
    def uvs(self):
        """
        Get the UV coordinates of the geom.
        """
        return self._uvs

    @property
    def surface(self):
        """
        Get the surface object of the geom.
        """
        return self._surface

    @property
    def gsd_path(self):
        """
        Get the path to the preprocessed `.gsd` file.
        """
        return self._gsd_path

    @property
    def sdf_res(self):
        """
        Get the resolution of the geom's signed distance field (SDF).
        """
        return self._sdf_res

    @property
    def sdf_val(self):
        """
        Get the signed distance field (SDF) of the geom.
        """
        if not self._is_preprocessed:
            self._preprocess()
        return self._sdf_val

    @property
    def sdf_val_flattened(self):
        """
        Get the flattened signed distance field (SDF) of the geom.
        """
        return self.sdf_val.reshape((-1,))

    @property
    def sdf_grad(self):
        """
        Get the gradient of the geom's signed distance field (SDF).
        """
        if not self._is_preprocessed:
            self._preprocess()
        return self._sdf_grad

    @property
    def sdf_grad_flattened(self):
        """
        Get the flattened gradient of the geom's signed distance field (SDF).
        """
        return self.sdf_grad.reshape(-1, 3)

    @property
    def sdf_max(self):
        """
        Get the maximum value of the geom's signed distance field (SDF).
        """
        if not self._is_preprocessed:
            self._preprocess()
        return self._sdf_max

    @property
    def sdf_cell_size(self):
        """
        Get the cell size of the geom's signed distance field (SDF).
        """
        return self._sdf_cell_size

    @property
    def sdf_grad_delta(self):
        """
        Get the delta value for computing the gradient of the geom's signed distance field (SDF).
        """
        return self._sdf_grad_delta

    @property
    def sdf_closest_vert(self):
        """
        Get the closest vertex of each cell of the geom's signed distance field (SDF).
        """
        if not self._is_preprocessed:
            self._preprocess()
        return self._sdf_closest_vert

    @property
    def sdf_closest_vert_flattened(self):
        """
        Get the flattened closest vertex of each cell of the geom's signed distance field (SDF).
        """
        return self.sdf_closest_vert.reshape((-1,))

    @property
    def T_mesh_to_sdf(self):
        """
        Get the transformation matrix of the geom's mesh frame w.r.t its signed distance field (SDF) frame.
        """
        if not self._is_preprocessed:
            self._preprocess()
        return self._T_mesh_to_sdf

    @property
    def n_cells(self):
        """
        Number of cells in the geom's signed distance field (SDF).
        """
        return np.prod(self.sdf_res)

    @property
    def n_verts(self) -> int:
        """
        Number of vertices of the geom.
        """
        return len(self._init_verts)

    @property
    def n_faces(self) -> int:
        """
        Number of faces of the geom.
        """
        return len(self._init_faces)

    @property
    def n_edges(self):
        """
        Number of edges of the geom.
        """
        return len(self._init_edges)

    @property
    def cell_start(self):
        """
        Get the starting index of the cells of the signed distance field (SDF) in the rigid solver.
        """
        return self._cell_start

    @property
    def vert_start(self):
        """
        Get the starting index of the geom's vertices in the rigid solver.
        """
        return self._vert_start

    @property
    def face_start(self):
        """
        Get the starting index of the geom's faces in the rigid solver.
        """
        return self._face_start

    @property
    def edge_start(self):
        """
        Get the starting index of the geom's edges in the rigid solver.
        """
        return self._edge_start

    @property
    def verts_state_start(self):
        """
        Get the starting index of the geom's vertices in the rigid solver.
        """
        return self._verts_state_start

    @property
    def cell_end(self):
        """
        Get the ending index of the cells of the signed distance field (SDF) in the rigid solver.
        """
        return self.n_cells + self.cell_start

    @property
    def vert_end(self):
        """
        Get the ending index of the geom's vertices in the rigid solver.
        """
        return self.n_verts + self.vert_start

    @property
    def verts_state_end(self):
        """
        Get the ending index of the geom's vertices in the rigid solver.
        """
        return self.n_verts + self.verts_state_start

    @property
    def face_end(self):
        """
        Get the ending index of the geom's faces in the rigid solver.
        """
        return self.n_faces + self.face_start

    @property
    def edge_end(self):
        """
        Get the ending index of the geom's edges in the rigid solver.
        """
        return self.n_edges + self.edge_start

    @property
    def is_built(self):
        """
        Whether the rigid entity the geom belongs to is built.
        """
        return self.entity.is_built

    @property
    def is_free(self):
        raise DeprecationError("This property has been removed.")

    @property
    def is_fixed(self) -> bool:
        """
        Whether this geom is fixed in the world.
        """
        return self.link.is_fixed

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        return f"{self._repr_type()}: {self._uid}, idx: {self._idx} (from entity {self._entity.uid}, link {self._link.uid})"


@ti.data_oriented
class RigidVisGeom(RBC):
    """
    A `RigidVisGeom` is a counterpart of `RigidGeom`, but for visualization purposes. This can be accessed via `link.vis_geoms`.
    """

    def __init__(
        self,
        link,
        idx,
        vvert_start,
        vface_start,
        vmesh,
        init_pos,
        init_quat,
    ):
        self._link = link
        self._entity = link.entity
        self._material = link.entity.material
        self._solver = link.entity.solver
        self._vmesh = vmesh

        # Lazy-initialize low-res geometry because it is usually unused and may be slow to compute
        self._init_pos_tc = torch.from_numpy(init_pos).to(device=gs.device, dtype=gs.tc_float)
        self._init_quat_tc = torch.from_numpy(init_quat).to(device=gs.device, dtype=gs.tc_float)
        self._aabb_verts: torch.Tensor | None = None

        self._uid = gs.UID()
        self._idx = idx

        self._vvert_start = vvert_start
        self._vface_start = vface_start

        self._init_pos = init_pos
        self._init_quat = init_quat

        # For heterogeneous simulation: which environments this vgeom is active in (None = all envs)
        self.active_envs_mask: torch.Tensor | None = None
        self.active_envs_idx: np.ndarray | None = None

        self._init_vverts = vmesh.verts
        self._init_vfaces = vmesh.faces
        self._init_vnormals = vmesh.normals
        self._uvs = vmesh.uvs
        self._surface = vmesh.surface
        self._metadata = vmesh.metadata
        self._color = vmesh._color

    def _build(self):
        pass

    def get_trimesh(self):
        """
        Get trimesh object.
        """
        return self._vmesh.trimesh

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_pos(self, envs_idx=None):
        """
        Get the position of the geom in world frame.
        """
        tensor = ti_to_torch(self._solver.vgeoms_state.pos, envs_idx, self._idx, transpose=True, copy=True)[..., 0, :]
        return tensor[0] if self._solver.n_envs == 0 else tensor

    @gs.assert_built
    def get_quat(self, envs_idx=None):
        """
        Get the quaternion of the geom in world frame.
        """
        tensor = ti_to_torch(self._solver.vgeoms_state.quat, envs_idx, self._idx, transpose=True, copy=True)[..., 0, :]
        return tensor[0] if self._solver.n_envs == 0 else tensor

    @gs.assert_built
    def get_vAABB(self, envs_idx=None):
        """
        Get the axis-aligned bounding box (AABB) of the geom in world frame.

        This method computes the bounding box of the geometry after aggressive decimation of its convex hull. This is
        usually sufficiently accurate (<1mm), while significantly improving runtime speed and reducing memory footprint.
        """
        if self._aabb_verts is None:
            # Aggressiveness has been tuned to give sub-millimeter accuracy on Franka robot in random configurations
            aabb_mesh = self.vmesh.copy()
            aabb_mesh.convexify()
            aabb_mesh.decimate(decimate_face_num=NUM_VERTS_VISUAL_GEOM_AABB, decimate_aggressiveness=3, convexify=False)
            self._aabb_verts = torch.from_numpy(aabb_mesh.verts).to(dtype=gs.tc_float, device=gs.device)

        pos, quat = gu.transform_pos_quat_by_trans_quat(
            self._init_pos_tc, self._init_quat_tc, self.link.get_pos(envs_idx), self.link.get_quat(envs_idx)
        )
        vverts_pos = pos[..., None, :] + gu.transform_by_quat(self._aabb_verts, quat[..., None, :])
        return torch.stack((vverts_pos.min(dim=-2).values, vverts_pos.max(dim=-2).values), dim=-2)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        Get the unique ID of the vgeom.
        """
        return self._uid

    @property
    def idx(self):
        """
        Get the global index of the vgeom in RigidSolver.
        """
        return self._idx

    @property
    def link(self):
        """
        Get the link that the vgeom belongs to.
        """
        return self._link

    @property
    def entity(self):
        """
        Get the entity that the vgeom belongs to.
        """
        return self._entity

    @property
    def vmesh(self):
        return self._vmesh

    @property
    def solver(self):
        """
        Get the solver that the vgeom belongs to.
        """
        return self._solver

    @property
    def metadata(self):
        """
        Get the metadata of the vgeom.
        """
        return self._metadata

    @property
    def init_pos(self):
        """
        Get the initial position of the vgeom.
        """
        return self._init_pos

    @property
    def init_quat(self):
        """
        Get the initial quaternion of the vgeom.
        """
        return self._init_quat

    @property
    def init_vverts(self):
        """
        Get the initial vertices of the vgeom.
        """
        return self._init_vverts

    @property
    def init_vfaces(self):
        """
        Get the initial faces of the vgeom.
        """
        return self._init_vfaces

    @property
    def init_vnormals(self):
        """
        Get the initial normals of the vgeom.
        """
        return self._init_vnormals

    @property
    def uvs(self):
        """
        Get the UV coordinates of the vgeom.
        """
        return self._uvs

    @property
    def surface(self):
        """
        Get the surface object of the vgeom.
        """
        return self._surface

    @property
    def n_vverts(self):
        """
        Number of vertices of the vgeom.
        """
        return len(self._init_vverts)

    @property
    def n_vfaces(self):
        """
        Number of faces of the vgeom.
        """
        return len(self._init_vfaces)

    @property
    def vvert_start(self):
        """
        Get the starting index of the vgeom's vertices in the rigid solver.
        """
        return self._vvert_start

    @property
    def vface_start(self):
        """
        Get the starting index of the vgeom's faces in the rigid solver.
        """
        return self._vface_start

    @property
    def vvert_end(self):
        """
        Get the ending index of the vgeom's vertices in the rigid solver.
        """
        return self.n_vverts + self.vvert_start

    @property
    def vface_end(self):
        """
        Get the ending index of the vgeom's faces in the rigid solver.
        """
        return self.n_vfaces + self.vface_start

    @property
    def is_built(self):
        """
        Whether the rigid entity the vgeom belongs to is built.
        """
        return self.entity.is_built

    @property
    def is_free(self):
        raise DeprecationError("This property has been removed.")

    @property
    def is_fixed(self) -> bool:
        """
        Whether this vgeom is fixed in the world.
        """
        return self.link.is_fixed

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        return f"{self._repr_type()}: {self._uid}, idx: {self._idx} (from entity {self._entity.uid}, link {self._link.uid})"
