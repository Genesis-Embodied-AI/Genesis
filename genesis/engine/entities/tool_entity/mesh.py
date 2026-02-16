import quadrants as qd
import numpy as np
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.mesh import compute_sdf_data, load_mesh


@qd.data_oriented
class Mesh:
    def __init__(self, entity, material, morph):
        self.entity = entity
        self.material = material
        self.collision = material.collision
        self.sdf_res = material.sdf_res
        self.scale = morph.scale
        self.raw_file = morph.file

        self.load_file()
        self.init_fields()

    def load_file(self):
        self.process_mesh()
        self.raw_vertices = np.asarray(self.mesh.vertices, dtype=gs.np_float, order="C")
        self.raw_vertex_normals = np.asarray(self.mesh.vertex_normals, dtype=gs.np_float, order="C")
        self.faces_np = np.asarray(self.mesh.faces, dtype=gs.np_int, order="C").reshape((-1))

        # apply initial transforms (scale then quat then pos)
        scale = np.array(self.scale, dtype=gs.np_float)
        T_init = gu.scale_to_T(scale)
        self.raw_vertices = gu.transform_by_T(self.raw_vertices, T_init)
        if self.collision:
            self.T_mesh_to_sdf_np = self.T_mesh_to_sdf_np @ gu.inv_T(T_init)

        self.n_vertices = len(self.raw_vertices)
        self.n_faces = len(self.faces_np)

    def process_mesh(self):
        # Normalize mesh
        gs.logger.debug(f"Processing mesh: {self.raw_file}.")
        mesh_orig = load_mesh(self.raw_file)
        scale = np.linalg.norm(mesh_orig.extents, ord=float("inf"))
        center = np.mean(mesh_orig.bounds, axis=0)
        normalized_vertices = (mesh_orig.vertices - center) / scale
        self.mesh = trimesh.Trimesh(
            vertices=normalized_vertices,
            faces=mesh_orig.faces,
            vertex_normals=mesh_orig.vertex_normals,
            face_normals=mesh_orig.face_normals,
        )

        # generate sdf
        if self.collision:
            sdf_data = compute_sdf_data(self.mesh, self.sdf_res)
            self.friction = self.material.friction
            self.sdf_voxels_np = sdf_data["voxels"].astype(gs.np_float, order="C", copy=False)
            self.sdf_res = self.sdf_voxels_np.shape[0]
            self.T_mesh_to_sdf_np = sdf_data["T_mesh_to_sdf"].astype(gs.np_float, order="C", copy=False)

    def init_fields(self):
        # init qd fields
        self.init_vertices = qd.Vector.field(3, dtype=gs.qd_float, shape=(self.n_vertices))
        self.init_vertex_normals = qd.Vector.field(3, dtype=gs.qd_float, shape=(self.n_vertices))
        self.faces = qd.field(dtype=gs.qd_int, shape=(self.n_faces,))

        self.init_vertices.from_numpy(self.raw_vertices)
        self.init_vertex_normals.from_numpy(self.raw_vertex_normals)
        self.faces.from_numpy(self.faces_np)

        if self.collision:
            self.sdf_voxels = qd.field(dtype=gs.qd_float, shape=self.sdf_voxels_np.shape)
            self.T_mesh_to_sdf = qd.Matrix.field(4, 4, dtype=gs.qd_float, shape=())

            self.sdf_voxels.from_numpy(self.sdf_voxels_np)
            self.T_mesh_to_sdf.from_numpy(self.T_mesh_to_sdf_np)

        self.vertices = qd.Vector.field(3, dtype=gs.qd_float, shape=(self.n_vertices))
        self.vertex_normals = qd.Vector.field(3, dtype=gs.qd_float, shape=(self.n_vertices))

    @qd.func
    def sdf(self, f, pos_world, i_b):
        # sdf value from world coordinate
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, self.entity.pos[f, i_b], self.entity.quat[f, i_b])
        pos_voxels = gu.qd_transform_by_T(pos_mesh, self.T_mesh_to_sdf[None])

        return self.sdf_(pos_voxels)

    @qd.func
    def sdf_(self, pos_voxels):
        # sdf value from voxels coordinate
        base = qd.floor(pos_voxels, gs.qd_int)
        signed_dist = gs.qd_float(0.0)
        if (base >= self.sdf_res - 1).any() or (base < 0).any():
            signed_dist = 1.0
        else:
            signed_dist = 0.0
            for offset in qd.static(qd.grouped(qd.ndrange(2, 2, 2))):
                voxel_pos = base + offset
                w_xyz = 1 - qd.abs(pos_voxels - voxel_pos)
                w = w_xyz[0] * w_xyz[1] * w_xyz[2]
                signed_dist += w * self.sdf_voxels[voxel_pos]

        return signed_dist

    @qd.func
    def normal(self, f, pos_world, i_b):
        # compute normal with finite difference
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, self.entity.pos[f, i_b], self.entity.quat[f, i_b])
        pos_voxels = gu.qd_transform_by_T(pos_mesh, self.T_mesh_to_sdf[None])
        normal_vec_voxels = self.normal_(pos_voxels)

        R_voxels_to_mesh = self.T_mesh_to_sdf[None][:3, :3].inverse()
        normal_vec_mesh = R_voxels_to_mesh @ normal_vec_voxels

        normal_vec_world = gu.qd_transform_by_quat(normal_vec_mesh, self.entity.quat[f, i_b])
        normal_vec_world = gu.qd_normalize(normal_vec_world, gs.EPS)

        return normal_vec_world

    @qd.func
    def normal_(self, pos_voxels):
        # since we are in voxels frame, delta can be a relatively big value
        delta = gs.qd_float(1e-2)
        normal_vec = qd.Vector([0, 0, 0], dt=gs.qd_float)

        for i in qd.static(range(3)):
            inc = pos_voxels
            dec = pos_voxels
            inc[i] += delta
            dec[i] -= delta
            normal_vec[i] = (self.sdf_(inc) - self.sdf_(dec)) / (2 * delta)

        normal_vec = gu.qd_normalize(normal_vec, gs.EPS)

        return normal_vec

    @qd.func
    def vel_collider(self, f, pos_world, i_b):
        pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, self.entity.pos[f, i_b], self.entity.quat[f, i_b])
        pos_world_new = gu.qd_transform_by_trans_quat(
            pos_mesh, self.entity.pos[f + 1, i_b], self.entity.quat[f + 1, i_b]
        )
        vel_collider = (pos_world_new - pos_world) / self.entity.solver.substep_dt
        return vel_collider

    @qd.func
    def collide(self, f, pos_world, vel_mat, i_b):
        if qd.static(self.collision):
            signed_dist = self.sdf(f, pos_world, i_b)
            # bigger coup_softness implies that the coupling influence extends further away from the object.
            influence = qd.min(qd.exp(-signed_dist / max(gs.EPS, self.material.coup_softness)), 1)
            if signed_dist <= 0.0 or influence > 0.1:
                vel_collider = self.vel_collider(f, pos_world, i_b)

                # v w.r.t collider
                rel_v = vel_mat - vel_collider
                normal_vec = self.normal(f, pos_world, i_b)
                normal_component = rel_v.dot(normal_vec)

                if normal_component < 0.0:
                    # remove inward velocity
                    rel_v_t = rel_v - normal_component * normal_vec
                    rel_v_t_norm = rel_v_t.norm(gs.EPS)

                    # tangential component after friction (if friction exists)
                    rel_v_t_friction = (
                        rel_v_t / rel_v_t_norm * qd.max(0, rel_v_t_norm + normal_component * self.friction)
                    )

                    # tangential component after friction
                    # FIXME: This formula could be simplified since flag = 1.0 systematically.
                    flag = qd.cast(normal_component < 0.0, gs.qd_float)
                    rel_v_t = rel_v_t_friction * flag + rel_v_t * (1.0 - flag)
                    vel_mat = vel_collider + rel_v_t * influence + rel_v * (1.0 - influence)

        return vel_mat

    @qd.func
    def is_collide(self, f, pos_world):
        flag = 0
        if qd.static(self.collision):
            signed_dist = self.sdf(f, pos_world)
            if signed_dist <= 0:
                flag = 1

        return flag

    @qd.func
    def pbd_collide(self, f, pos_world, thickness, dt):
        return_pos_world = pos_world
        if qd.static(self.collision):
            pos_mesh = gu.qd_inv_transform_by_trans_quat(pos_world, self.entity.pos[f], self.entity.quat[f])
            pos_voxels = gu.qd_transform_by_T(pos_mesh, self.T_mesh_to_sdf[None])

            sdf_voxels = self.sdf_(pos_voxels)
            normal_vec_voxels = self.normal_(pos_voxels)
            normal_vec_voxels_ = gu.qd_normalize(normal_vec_voxels, gs.EPS)

            vec_voxels = -sdf_voxels * normal_vec_voxels_
            R_voxels_to_mesh = self.T_mesh_to_sdf[None][:3, :3].inverse()
            vec_mesh = R_voxels_to_mesh @ vec_voxels
            vec_world = gu.qd_transform_by_quat(vec_mesh, self.entity.quat[f])
            vec_world_norm = vec_world.norm(gs.EPS)

            if sdf_voxels < 0:
                return_pos_world += vec_world / vec_world_norm * (vec_world_norm + thickness)

            if sdf_voxels > 0 and sdf_voxels < 1 and vec_world_norm < thickness:
                return_pos_world += vec_world / vec_world_norm * (vec_world_norm - thickness)

        return return_pos_world
