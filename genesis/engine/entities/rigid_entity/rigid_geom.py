import os
import pickle as pkl

import igl
import numpy as np
import skimage
import taichi as ti
import torch

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.ext import trimesh
from genesis.repr_base import RBC
from genesis.utils.misc import tensor_to_array


@ti.data_oriented
class RigidGeom(RBC):
    """
    A `RigidGeom` is the basic building block of a `RigidEntity` for collision checking. It is usually constructed from a single mesh. This can be accessed via `link.geoms`.
    """

    def __init__(
        self,
        link,
        idx,
        cell_start,
        vert_start,
        face_start,
        edge_start,
        verts_state_start,
        mesh,
        type,
        friction,
        sol_params,
        init_pos,
        init_quat,
        needs_coup,
        center_init=None,
        data=None,
    ):
        self._link = link
        self._entity = link.entity
        self._material = link.entity.material
        self._solver = link.entity.solver
        self._mesh = mesh

        self._uid = gs.UID()
        self._idx = idx
        self._type = type
        self._friction = friction
        self._sol_params = sol_params
        self._needs_coup = needs_coup
        self._is_convex = mesh.is_convex
        self._cell_start = cell_start
        self._vert_start = vert_start
        self._face_start = face_start
        self._edge_start = edge_start
        self._verts_state_start = verts_state_start

        self._coup_softness = self._material.coup_softness
        self._coup_friction = self._material.coup_friction
        self._coup_restitution = self._material.coup_restitution

        self._init_pos = init_pos
        self._init_quat = init_quat

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

        # collision mesh uses default color
        self._preprocess()

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
            except (EOFError, pkl.UnpicklingError):
                gs.logger.info("Ignoring corrupted cache.")

        if not is_cached_loaded:
            with gs.logger.timer(f"Preprocessing geom idx ~~<{self._idx}>~~."):
                ######## sdf ########
                lower = self._init_verts.min(axis=0)
                upper = self._init_verts.max(axis=0)
                center = (upper + lower) / 2.0

                # NOTE: sdf size is from the center of the lower voxel cell to the center of the upper voxel cell
                # add padding
                padding_ratio = 0.2
                grid_size = (upper - lower).max() * padding_ratio + (upper - lower)
                sdf_res = np.ceil(grid_size / self._material.sdf_cell_size).astype(int) + 1

                if sdf_res.max() > self._material.sdf_max_res:
                    sdf_res = np.ceil(sdf_res * self._material.sdf_max_res / sdf_res.max()).astype(int)
                    sdf_cell_size = grid_size.max() / (sdf_res.max() - 1)
                elif sdf_res.max() < self._material.sdf_min_res:
                    sdf_res = np.ceil(sdf_res * self._material.sdf_min_res / sdf_res.max()).astype(int)
                    sdf_cell_size = grid_size.max() / (sdf_res.max() - 1)
                else:
                    sdf_cell_size = self._material.sdf_cell_size
                sdf_res = np.clip(sdf_res, 3, None)

                # round up to multiple of sdf_cell_size
                grid_size = (sdf_res - 1) * sdf_cell_size

                halfsize = grid_size / 2.0
                voxel_lower = center - halfsize
                voxel_upper = center + halfsize

                x = np.linspace(voxel_lower[0], voxel_upper[0], sdf_res[0])
                y = np.linspace(voxel_lower[1], voxel_upper[1], sdf_res[1])
                z = np.linspace(voxel_lower[2], voxel_upper[2], sdf_res[2])
                X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
                query_points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))

                sdf_val = self._compute_sd(query_points)
                sdf_closest_vert = self._compute_closest_verts(query_points)
                sdf_val = sdf_val.reshape(sdf_res)
                sdf_closest_vert = sdf_closest_vert.reshape(sdf_res)
                T_mesh_to_centered = np.eye(4)
                T_mesh_to_centered[:3, 3] = -center
                T_centered_to_sdf = np.eye(4)
                T_centered_to_sdf[:3, :3] *= (sdf_res - 1) / grid_size
                T_centered_to_sdf[:3, 3] = (sdf_res - 1) / 2
                T_mesh_to_sdf = T_centered_to_sdf @ T_mesh_to_centered

                ######## sdf gradient ########
                if self.type == gs.GEOM_TYPE.TERRAIN:  # terrain uses finite difference for sdf gradient computation
                    # dummy
                    sdf_grad_delta = 0.0
                    sdf_grad = np.zeros([*sdf_res, 3])
                else:
                    sdf_grad_delta = sdf_cell_size * 1e-2
                    sdf_grad = self._compute_sd_grad(query_points, sdf_grad_delta).reshape([*sdf_res, 3])

                ######## adjacency graph ########
                all_vert_neighbors_list = trimesh.Trimesh(
                    vertices=self._init_verts, faces=self._init_faces, process=False
                ).vertex_neighbors
                assert self.n_verts == len(all_vert_neighbors_list)

                vert_neighbor_start = list()
                vert_n_neighbors = list()
                vert_neighbors = list()
                n = 0
                for vert_neighbors_list in all_vert_neighbors_list:
                    n_vert_neighbors = len(vert_neighbors_list)
                    vert_neighbors += vert_neighbors_list
                    vert_neighbor_start.append(n)
                    vert_n_neighbors.append(n_vert_neighbors)
                    n += n_vert_neighbors

                vert_neighbors = np.array(vert_neighbors)
                vert_n_neighbors = np.array(vert_n_neighbors)
                vert_neighbor_start = np.array(vert_neighbor_start)

                # caching
                gsd_dict = {
                    "sdf_res": sdf_res,
                    "sdf_val": sdf_val,
                    "sdf_grad": sdf_grad,
                    "sdf_max": np.max(sdf_val),
                    "sdf_grad_delta": sdf_grad_delta,
                    "sdf_cell_size": sdf_cell_size,
                    "sdf_closest_vert": sdf_closest_vert,
                    "T_mesh_to_sdf": T_mesh_to_sdf,
                    "vert_neighbors": vert_neighbors,
                    "vert_n_neighbors": vert_n_neighbors,
                    "vert_neighbor_start": vert_neighbor_start,
                }
                os.makedirs(os.path.dirname(self._gsd_path), exist_ok=True)
                with open(self._gsd_path, "wb") as file:
                    pkl.dump(gsd_dict, file)

        self._sdf_res = gsd_dict["sdf_res"]
        self._sdf_val = gsd_dict["sdf_val"]
        self._sdf_grad = gsd_dict["sdf_grad"]
        self._sdf_max = gsd_dict["sdf_max"]
        self._sdf_cell_size = gsd_dict["sdf_cell_size"]
        self._sdf_grad_delta = gsd_dict["sdf_grad_delta"]
        self._sdf_closest_vert = gsd_dict["sdf_closest_vert"]
        self._T_mesh_to_sdf = gsd_dict["T_mesh_to_sdf"]

        self.vert_neighbors = gsd_dict["vert_neighbors"]
        self.vert_n_neighbors = gsd_dict["vert_n_neighbors"]
        self.vert_neighbor_start = gsd_dict["vert_neighbor_start"]

    def _compute_sd(self, query_points):
        try:
            sd, _, _ = igl.signed_distance(query_points, self._sdf_verts, self._sdf_faces)
        except:
            sd, _, _, _ = igl.signed_distance(query_points, self._sdf_verts, self._sdf_faces)
        return sd

    def _compute_closest_verts(self, query_points):
        try:
            _, closest_faces, _ = igl.signed_distance(query_points, self._init_verts, self._init_faces)
        except:
            _, closest_faces, _, _ = igl.signed_distance(query_points, self._init_verts, self._init_faces)
        verts_ids = self._init_faces[closest_faces]
        verts_ids = verts_ids[
            np.arange(len(query_points)).astype(int),
            np.argmin(np.linalg.norm(self._init_verts[verts_ids] - query_points[:, None, :], axis=-1), axis=-1),
        ]
        return verts_ids

    def _compute_sd_grad(self, query_points, delta=5e-4):
        ######## sdf gradient via finite differencing ########
        sd_val_xpos = self._compute_sd(query_points + np.array([delta, 0.0, 0.0]))
        sd_val_xneg = self._compute_sd(query_points - np.array([delta, 0.0, 0.0]))
        sd_val_ypos = self._compute_sd(query_points + np.array([0.0, delta, 0.0]))
        sd_val_yneg = self._compute_sd(query_points - np.array([0.0, delta, 0.0]))
        sd_val_zpos = self._compute_sd(query_points + np.array([0.0, 0.0, delta]))
        sd_val_zneg = self._compute_sd(query_points - np.array([0.0, 0.0, delta]))

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

    def sdf_grad_world(self, pos_world, recompute=False):
        """
        sdf grad wrt world frame coordinate.
        """
        pos_mesh = gu.inv_transform_by_trans_quat(pos_world, self.get_pos(), self.get_quat())
        grad_mesh = self.sdf_grad_mesh(pos_mesh, recompute)
        grad_world = gu.transform_by_quat(grad_mesh, self.get_quat())
        return grad_world

    def sdf_grad_mesh(self, pos_mesh, recompute=False):
        """
        sdf grad wrt mesh frame coordinate.
        """
        if recompute:
            grad_mesh = self._compute_sd_grad(np.array([pos_mesh]), self.sdf_grad_delta)

        else:
            pos_sdf = gu.transform_by_T(pos_mesh, self.T_mesh_to_sdf)
            grad_sdf = self.sdf_grad_sdf(pos_sdf)
            grad_mesh = grad_sdf  # no rotation between mesh and sdf frame

        return grad_mesh

    def sdf_grad_sdf(self, pos_sdf):
        """
        sdf grad wrt sdf frame coordinate.
        """
        base = np.floor(pos_sdf)
        res = self.sdf_res

        if (base >= res - 1).any() or (base < 0).any():
            grad_sdf = np.array([np.nan, np.nan, np.nan])

        else:
            grad_sdf = np.zeros(3)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        offset = np.array([i, j, k])
                        voxel_pos = (base + offset).astype(int)
                        w_xyz = 1 - np.abs(pos_sdf - voxel_pos)
                        w = w_xyz[0] * w_xyz[1] * w_xyz[2]
                        grad_sdf += w * self.sdf_grad[voxel_pos[0], voxel_pos[1], voxel_pos[2]]

        return grad_sdf

    def sdf_world(self, pos_world, recompute=False):
        """
        sdf value from world coordinate
        """
        pos_mesh = gu.inv_transform_by_trans_quat(pos_world, self.get_pos(), self.get_quat())
        return self.sdf_mesh(pos_mesh, recompute)

    def sdf_mesh(self, pos_mesh, recompute=False):
        """
        sdf value from mesh coordinate
        """
        if recompute:
            return self._compute_sd(np.array([pos_mesh]))
        else:
            pos_sdf = gu.transform_by_T(pos_mesh, self.T_mesh_to_sdf)
            return self.sdf_sdf(pos_sdf)

    def sdf_sdf(self, pos_sdf):
        """
        sdf value wrt sdf frame coordinate.
        Note that the stored sdf magnitude is already w.r.t world frame.
        """
        base = np.floor(pos_sdf)
        res = self.sdf_res

        if (base >= res - 1).any() or (base < 0).any():
            signed_dist = np.inf

        else:
            signed_dist = 0.0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        offset = np.array([i, j, k])
                        voxel_pos = (base + offset).astype(int)
                        w_xyz = 1 - np.abs(pos_sdf - voxel_pos)
                        w = w_xyz[0] * w_xyz[1] * w_xyz[2]
                        signed_dist += w * self.sdf_val[voxel_pos[0], voxel_pos[1], voxel_pos[2]]

        return signed_dist

    def set_friction(self, friction):
        """
        Set the friction coefficient of the geom.
        """
        if friction < 0:
            gs.raise_exception("`friction` must be non-negative.")
        self._friction = friction

        if self.is_built:
            self._solver.set_geom_friction(friction, self._idx)

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_pos(self):
        """
        Get the position of the geom in world frame.
        """
        tensor = torch.empty(self._solver._batch_shape(3, True), dtype=gs.tc_float, device=gs.device)
        self._kernel_get_pos(tensor)
        if self._solver.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_pos(self, tensor: ti.types.ndarray()):
        for i, b in ti.ndrange(3, self._solver._B):
            tensor[b, i] = self._solver.geoms_state[self._idx, b].pos[i]

    @gs.assert_built
    def get_quat(self):
        """
        Get the quaternion of the geom in world frame.
        """
        tensor = torch.empty(self._solver._batch_shape(4, True), dtype=gs.tc_float, device=gs.device)
        self._kernel_get_quat(tensor)
        if self._solver.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_quat(self, tensor: ti.types.ndarray()):
        for i, b in ti.ndrange(4, self._solver._B):
            tensor[b, i] = self._solver.geoms_state[self._idx, b].quat[i]

    @gs.assert_built
    def get_verts(self):
        """
        Get the vertices of the geom in world frame.
        """
        if self.is_free:
            tensor = torch.empty(
                self._solver._batch_shape((self.n_verts, 3), True), dtype=gs.tc_float, device=gs.device
            )
            self._kernel_get_free_verts(tensor)
            if self._solver.n_envs == 0:
                tensor = tensor.squeeze(0)
        else:
            tensor = torch.empty((self.n_verts, 3), dtype=gs.tc_float, device=gs.device)
            self._kernel_get_fixed_verts(tensor)
        return tensor

    @ti.kernel
    def _kernel_get_free_verts(self, tensor: ti.types.ndarray()):
        for b in range(self._solver._B):
            self._solver._func_update_verts_for_geom(self._idx, b)

        for i, j, b in ti.ndrange(self.n_verts, 3, self._solver._B):
            idx_vert = i + self._verts_state_start
            tensor[b, i, j] = self._solver.free_verts_state[idx_vert, b].pos[j]

    @ti.kernel
    def _kernel_get_fixed_verts(self, tensor: ti.types.ndarray()):
        self._solver._func_update_verts_for_geom(self._idx, 0)

        for i, j in ti.ndrange(self.n_verts, 3):
            idx_vert = i + self._verts_state_start
            tensor[i, j] = self._solver.fixed_verts_state[idx_vert].pos[j]

    @gs.assert_built
    def get_AABB(self):
        """
        Get the axis-aligned bounding box (AABB) of the geom in world frame.
        """
        verts = self.get_verts()
        AABB = torch.concatenate(
            [verts.min(axis=-2, keepdim=True)[0], verts.max(axis=-2, keepdim=True)[0]],
            axis=-2,
        )
        return AABB

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
    def idx(self):
        """
        Get the global index of the geom in RigidSolver.
        """
        return self._idx

    @property
    def type(self):
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
    def sol_params(self):
        """
        Get the solver parameters of the geom.
        """
        return self._sol_params

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
    def link(self):
        """
        Get the link that the geom belongs to.
        """
        return self._link

    @property
    def entity(self):
        """
        Get the entity that the geom belongs to.
        """
        return self._entity

    @property
    def solver(self):
        """
        Get the solver that the geom belongs to.s
        """
        return self._solver

    @property
    def is_convex(self):
        """
        Get whether the geom is convex.
        """
        return self._is_convex

    @property
    def mesh(self):
        return self._mesh

    @property
    def needs_coup(self):
        """
        Get whether the geom needs coupling with other non-rigid entities.
        """
        return self._needs_coup

    @property
    def coup_softness(self):
        """
        Get the softness coefficient of the geom for coupling.
        """
        return self._coup_softness

    @property
    def coup_friction(self):
        """
        Get the friction coefficient of the geom for coupling.
        """
        return self._coup_friction

    @property
    def coup_restitution(self):
        """
        Get the restitution coefficient of the geom for coupling.
        """
        return self._coup_restitution

    @property
    def init_pos(self):
        """
        Get the initial position of the geom.
        """
        return self._init_pos

    @property
    def init_quat(self):
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
        return self._sdf_val

    @property
    def sdf_val_flattened(self):
        """
        Get the flattened signed distance field (SDF) of the geom.
        """
        return self._sdf_val.flatten()

    @property
    def sdf_grad(self):
        """
        Get the gradient of the geom's signed distance field (SDF).
        """
        return self._sdf_grad

    @property
    def sdf_grad_flattened(self):
        """
        Get the flattened gradient of the geom's signed distance field (SDF).
        """
        return self._sdf_grad.reshape(-1, 3)

    @property
    def sdf_max(self):
        """
        Get the maximum value of the geom's signed distance field (SDF).
        """
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
        return self._sdf_closest_vert

    @property
    def sdf_closest_vert_flattened(self):
        """
        Get the flattened closest vertex of each cell of the geom's signed distance field (SDF).
        """
        return self._sdf_closest_vert.flatten()

    @property
    def T_mesh_to_sdf(self):
        """
        Get the transformation matrix of the geom's mesh frame w.r.t its signed distance field (SDF) frame.
        """
        return self._T_mesh_to_sdf

    @property
    def n_cells(self):
        """
        Number of cells in the geom's signed distance field (SDF).
        """
        return np.prod(self._sdf_res)

    @property
    def n_verts(self):
        """
        Number of vertices of the geom.
        """
        return len(self._init_verts)

    @property
    def n_faces(self):
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
        """
        Whether the rigid entity the vgeom belongs to is free.
        """
        return self.entity.is_free

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

        self._uid = gs.UID()
        self._idx = idx

        self._vvert_start = vvert_start
        self._vface_start = vface_start

        self._init_pos = init_pos
        self._init_quat = init_quat

        self._init_vverts = vmesh.verts
        self._init_vfaces = vmesh.faces
        self._init_vnormals = vmesh.normals
        self._uvs = vmesh.uvs
        self._surface = vmesh.surface
        self._metadata = vmesh.metadata

    def _build(self):
        pass

    def get_trimesh(self):
        """
        Get trimesh object.
        """
        return self._vmesh.trimesh

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
        """
        Whether the rigid entity the vgeom belongs to is free.
        """
        return self.entity.is_free

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        return f"{self._repr_type()}: {self._uid}, idx: {self._idx} (from entity {self._entity.uid}, link {self._link.uid})"
