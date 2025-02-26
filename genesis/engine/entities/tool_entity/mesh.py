import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.mesh import (
    cleanup_mesh,
    compute_sdf_data,
    load_mesh,
    normalize_mesh,
)


@ti.data_oriented
class Mesh:
    def __init__(self, entity, material, morph):
        self.entity = entity
        self.material = material
        self.collision = material.collision
        self.sdf_res = material.sdf_res
        self.scale = morph.scale
        self.raw_file = morph.file

        self.load_file()
        self.init_transform()

    def load_file(self):
        # mesh
        self.process_mesh()
        self.raw_vertices = np.asarray(self.mesh.vertices, dtype=np.float32, order="C")
        self.raw_vertex_normals_np = np.asarray(self.mesh.vertex_normals, dtype=np.float32, order="C")
        self.faces_np = np.asarray(self.mesh.faces, dtype=np.int32, order="C").reshape((-1))

        self.n_vertices = len(self.raw_vertices)
        self.n_faces = len(self.faces_np)

    def process_mesh(self):
        # clean up mesh
        gs.logger.debug(f"Processing mesh: {self.raw_file}.")
        raw_mesh = load_mesh(self.raw_file)
        self.mesh = cleanup_mesh(normalize_mesh(raw_mesh))

        # generate sdf
        if self.collision:
            raw_mesh = load_mesh(self.raw_file)
            sdf_data = compute_sdf_data(cleanup_mesh(normalize_mesh(raw_mesh)), self.sdf_res)
            self.friction = self.material.friction
            self.sdf_voxels_np = sdf_data["voxels"].astype(gs.np_float, order="C", copy=False)
            self.sdf_res = self.sdf_voxels_np.shape[0]
            self.T_mesh_to_sdf_np = sdf_data["T_mesh_to_sdf"].astype(gs.np_float, order="C", copy=False)

    def init_transform(self):
        scale = np.array(self.scale, dtype=gs.np_float)

        # apply initial transforms (scale then quat then pos)
        T_init = gu.scale_to_T(scale)
        self.init_vertices_np = gu.transform_by_T(self.raw_vertices, T_init).astype(np.float32, order="C", copy=False)

        self.init_vertex_normals_np = self.raw_vertex_normals_np.astype(np.float32, order="C", copy=False)

        # init ti fields
        self.init_vertices = ti.Vector.field(3, dtype=gs.ti_float, shape=(self.n_vertices))
        self.init_vertex_normals = ti.Vector.field(3, dtype=gs.ti_float, shape=(self.n_vertices))
        self.faces = ti.field(dtype=gs.ti_int, shape=(self.n_faces))

        self.init_vertices.from_numpy(self.init_vertices_np)
        self.init_vertex_normals.from_numpy(self.init_vertex_normals_np)
        self.faces.from_numpy(self.faces_np)

        if self.collision:
            self.T_mesh_to_sdf_np = self.T_mesh_to_sdf_np @ np.linalg.inv(T_init)
            self.sdf_voxels = ti.field(dtype=gs.ti_float, shape=self.sdf_voxels_np.shape)
            self.T_mesh_to_sdf = ti.Matrix.field(4, 4, dtype=gs.ti_float, shape=())

            self.sdf_voxels.from_numpy(self.sdf_voxels_np)
            self.T_mesh_to_sdf.from_numpy(self.T_mesh_to_sdf_np)

        self.vertices = ti.Vector.field(3, dtype=gs.ti_float, shape=(self.n_vertices))
        self.vertex_normals = ti.Vector.field(3, dtype=gs.ti_float, shape=(self.n_vertices))

    @ti.kernel
    def update_vertices(self, f: ti.i32):
        for i in self.vertices:
            self.vertices[i] = gu.ti_transform_by_trans_quat(
                self.init_vertices[i], self.entity.pos[f], self.entity.quat[f]
            )
            self.vertex_normals[i] = gu.ti_transform_by_quat(self.init_vertex_normals[i], self.entity.quat[f])

    @ti.func
    def sdf(self, f, pos_world):
        # sdf value from world coordinate
        pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, self.entity.pos[f], self.entity.quat[f])
        pos_voxels = gu.ti_transform_by_T(pos_mesh, self.T_mesh_to_sdf[None])

        return self.sdf_(pos_voxels)

    @ti.func
    def sdf_(self, pos_voxels):
        # sdf value from voxels coordinate
        base = ti.floor(pos_voxels, gs.ti_int)
        signed_dist = ti.cast(0.0, gs.ti_float)
        if (base >= self.sdf_res - 1).any() or (base < 0).any():
            signed_dist = 1.0
        else:
            signed_dist = 0.0
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                voxel_pos = base + offset
                w_xyz = 1 - ti.abs(pos_voxels - voxel_pos)
                w = w_xyz[0] * w_xyz[1] * w_xyz[2]
                signed_dist += w * self.sdf_voxels[voxel_pos]

        return signed_dist

    @ti.func
    def normal(self, f, pos_world):
        # compute normal with finite difference
        pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, self.entity.pos[f], self.entity.quat[f])
        pos_voxels = gu.ti_transform_by_T(pos_mesh, self.T_mesh_to_sdf[None])
        normal_vec_voxels = self.normal_(pos_voxels)

        R_voxels_to_mesh = self.T_mesh_to_sdf[None][:3, :3].inverse()
        normal_vec_mesh = R_voxels_to_mesh @ normal_vec_voxels

        normal_vec_world = gu.ti_transform_by_quat(normal_vec_mesh, self.entity.quat[f])
        normal_vec_world = gu.ti_normalize(normal_vec_world)

        return normal_vec_world

    @ti.func
    def normal_(self, pos_voxels):
        # since we are in voxels frame, delta can be a relatively big value
        delta = ti.cast(1e-2, gs.ti_float)
        normal_vec = ti.Vector([0, 0, 0], dt=gs.ti_float)

        for i in ti.static(range(3)):
            inc = pos_voxels
            dec = pos_voxels
            inc[i] += delta
            dec[i] -= delta
            normal_vec[i] = (self.sdf_(inc) - self.sdf_(dec)) / (2 * delta)

        normal_vec = gu.ti_normalize(normal_vec)

        return normal_vec

    @ti.func
    def vel_collider(self, f, pos_world):
        pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, self.entity.pos[f], self.entity.quat[f])
        pos_world_new = gu.ti_transform_by_trans_quat(pos_mesh, self.entity.pos[f + 1], self.entity.quat[f + 1])
        vel_collider = (pos_world_new - pos_world) / self.entity.solver.substep_dt
        return vel_collider

    @ti.func
    def collide(self, f, pos_world, vel_mat):
        if ti.static(self.collision):
            signed_dist = self.sdf(f, pos_world)
            # bigger coup_softness implies that the coupling influence extends further away from the object.
            influence = ti.min(ti.exp(-signed_dist / max(gs.EPS, self.material.coup_softness)), 1)
            if signed_dist <= 0 or influence > 0.1:
                vel_collider = self.vel_collider(f, pos_world)

                # v w.r.t collider
                rel_v = vel_mat - vel_collider
                normal_vec = self.normal(f, pos_world)
                normal_component = rel_v.dot(normal_vec)

                if normal_component < 0:
                    # remove inward velocity
                    rel_v_t = rel_v - normal_component * normal_vec
                    rel_v_t_norm = rel_v_t.norm(gs.EPS)

                    # tangential component after friction (if friction exists)
                    rel_v_t_friction = (
                        rel_v_t / rel_v_t_norm * ti.max(0, rel_v_t_norm + normal_component * self.friction)
                    )

                    # tangential component after friction
                    flag = ti.cast(normal_component < 0, gs.ti_float)
                    rel_v_t = rel_v_t_friction * flag + rel_v_t * (1 - flag)
                    vel_mat = vel_collider + rel_v_t * influence + rel_v * (1 - influence)

        return vel_mat

    @ti.func
    def is_collide(self, f, pos_world):
        flag = 0
        if ti.static(self.collision):
            signed_dist = self.sdf(f, pos_world)
            if signed_dist <= 0:
                flag = 1

        return flag

    @ti.func
    def pbd_collide(self, f, pos_world, thickness, dt):
        return_pos_world = pos_world
        if ti.static(self.collision):
            pos_mesh = gu.ti_inv_transform_by_trans_quat(pos_world, self.entity.pos[f], self.entity.quat[f])
            pos_voxels = gu.ti_transform_by_T(pos_mesh, self.T_mesh_to_sdf[None])

            sdf_voxels = self.sdf_(pos_voxels)
            normal_vec_voxels = self.normal_(pos_voxels)
            normal_vec_voxels_ = gu.ti_normalize(normal_vec_voxels)

            vec_voxels = -sdf_voxels * normal_vec_voxels_
            R_voxels_to_mesh = self.T_mesh_to_sdf[None][:3, :3].inverse()
            vec_mesh = R_voxels_to_mesh @ vec_voxels
            vec_world = gu.ti_transform_by_quat(vec_mesh, self.entity.quat[f])
            vec_world_norm = vec_world.norm(gs.EPS)

            if sdf_voxels < 0:
                return_pos_world += vec_world / vec_world_norm * (vec_world_norm + thickness)

            if sdf_voxels > 0 and sdf_voxels < 1 and vec_world_norm < thickness:
                return_pos_world += vec_world / vec_world_norm * (vec_world_norm - thickness)

        return return_pos_world
