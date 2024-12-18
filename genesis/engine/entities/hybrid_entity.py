import numpy as np
import taichi as ti

import genesis as gs
import genesis.utils.geom as gu
from genesis.ext import trimesh
from genesis.ext.urdfpy.urdf import URDF
from genesis.utils.hybrid import (
    check_graph,
    compute_graph_attribute,
    gel_graph_to_nx_graph,
    graph_to_tree,
    reduce_graph,
    skeletonization,
    trimesh_to_gelmesh,
)
from genesis.utils.mesh import (
    cleanup_mesh,
    load_mesh,
    normalize_mesh,
)

from .base_entity import Entity
from .mpm_entity import MPMEntity


@ti.data_oriented
class HybridEntity(Entity):
    def __init__(
        self,
        idx,
        scene,
        material,
        morph,
        surface,
    ):
        super().__init__(idx, scene, morph, None, material, surface)

        mat_rigid = material.mat_rigid
        mat_soft = material.mat_soft

        surface_rigid = gs.surfaces.Smooth(roughness=0.4)  # HACK hardcoded

        assert isinstance(mat_soft, gs.materials.MPM.Base)  # TODO: need FEM and PBD

        if isinstance(morph, gs.morphs.URDF):
            # set up rigid part
            morph.fixed = material.fixed  # NOTE: use hybrid material to determine this
            if material.use_default_coupling:
                gs.logger.info("Use default coupling in hybrid. Overwrite `needs_coup` in rigid material to True")
                mat_rigid._needs_coup = True
            else:
                gs.logger.info("Use default coupling in hybrid. Overwrite `needs_coup` in rigid material to False")
                mat_rigid._needs_coup = False

            part_rigid = scene.add_entity(
                material=mat_rigid,
                morph=morph,
                surface=surface_rigid,
            )

            # get rigid part in world coords
            augment_link_world_coords(part_rigid)

            # set soft parts based on rigid links
            if material._func_instantiate_soft_from_rigid is None:
                func_instantiate_soft_from_rigid = default_func_instantiate_soft_from_rigid
            else:
                func_instantiate_soft_from_rigid = material._func_instantiate_soft_from_rigid
            part_soft = func_instantiate_soft_from_rigid(
                scene=scene,
                part_rigid=part_rigid,
                material_soft=mat_soft,
                material_hybrid=material,
                surface=surface,
            )

        elif isinstance(morph, gs.morphs.Mesh):
            # instantiate soft part
            part_soft = scene.add_entity(
                material=mat_soft,
                morph=morph,
                surface=surface,
            )

            # load mesh in the same way as the soft entity
            mesh = load_mesh(morph.file)

            # instantiate rigid part
            if material._func_instantiate_rigid_from_soft is None:
                func_instantiate_rigid_from_soft = default_func_instantiate_rigid_from_soft
            else:
                func_instantiate_rigid_from_soft = material._func_instantiate_rigid_from_soft
            part_rigid = func_instantiate_rigid_from_soft(
                scene=scene,
                mesh=mesh,
                morph=morph,
                material_rigid=mat_rigid,
                material_hybrid=material,
                surface=surface_rigid,
            )

        else:
            raise ValueError(f"`morph` in hybrid entity should be either URDF or Mesh")

        if not material.use_default_coupling:
            # get rigid-soft association function
            if material._func_instantiate_rigid_soft_association is None:
                if isinstance(morph, gs.morphs.URDF):
                    func_instantiate_rigid_soft_association = default_func_instantiate_rigid_soft_association_from_rigid
                elif isinstance(morph, gs.morphs.Mesh):
                    func_instantiate_rigid_soft_association = default_func_instantiate_rigid_soft_association_from_soft
            else:
                func_instantiate_rigid_soft_association = material._func_instantiate_rigid_soft_association
            muscle_group, link_idcs, geom_idcs, trans_local_to_global, quat_local_to_global = (
                func_instantiate_rigid_soft_association(
                    part_rigid=part_rigid,
                    part_soft=part_soft,
                )
            )

            # set muscle group
            mat_soft._n_groups = len(link_idcs)
            self._muscle_group_cache = muscle_group

            # set up info in taichi field
            if isinstance(mat_soft, gs.materials.MPM.Base):
                part_soft_info = ti.types.struct(
                    link_idx=gs.ti_int,
                    geom_idx=gs.ti_int,
                    trans_local_to_global=gs.ti_vec3,
                    quat_local_to_global=gs.ti_vec4,
                ).field(shape=(mat_soft.n_groups,), needs_grad=False, layout=ti.Layout.SOA)
                part_soft_info.link_idx.from_numpy(np.array(link_idcs).astype(gs.np_int))
                part_soft_info.geom_idx.from_numpy(np.array(geom_idcs).astype(gs.np_int))
                part_soft_info.trans_local_to_global.from_numpy(np.array(trans_local_to_global).astype(gs.np_float))
                part_soft_info.quat_local_to_global.from_numpy(np.array(quat_local_to_global).astype(gs.np_float))

                part_soft_init_positions = ti.field(dtype=gs.ti_vec3, shape=(part_soft.init_particles.shape[0],))
                part_soft_init_positions.from_torch(gs.Tensor(part_soft.init_particles))

                self._part_soft_info = part_soft_info
                self._part_soft_init_positions = part_soft_init_positions
            else:
                raise ValueError(f"Cannot handle soft material {mat_soft}")

            # set coupling func
            def wrap_func(func, before=False):
                def wrapper(f):
                    if before:
                        self.update_soft_part(f)
                    func(f)
                    if not before:
                        self.update_soft_part(f)

                return wrapper

            if isinstance(mat_soft, gs.materials.MPM.Base):
                # NOTE: coupling operating at particle level and here we modify post_coupling, i.e., update particle state after g2p
                self._update_soft_part_at_pre_coupling = False
                if self._update_soft_part_at_pre_coupling:
                    part_soft.solver.substep_pre_coupling = wrap_func(
                        part_soft.solver.substep_pre_coupling, before=True
                    )
                else:
                    part_soft.solver.substep_post_coupling = wrap_func(
                        part_soft.solver.substep_post_coupling, before=False
                    )
            else:
                raise ValueError(f"Cannot handle soft material {mat_soft}")

        # set members
        self._mat_rigid = mat_rigid
        self._part_rigid = part_rigid
        self._solver_rigid = part_rigid.solver
        self._mat_soft = mat_soft
        self._part_soft = part_soft
        self._solver_soft = part_soft.solver

        # TODO: test with different dt
        assert self._solver_rigid.dt == self._solver_soft.dt, "Rigid and soft solver should have the same dt for now"

    # ------------------------------------------------------------------------------------
    # ----------------------------------- basic ops --------------------------------------
    # ------------------------------------------------------------------------------------

    def get_dofs_position(self, *args, **kwargs):
        return self._part_rigid.get_dofs_position(*args, **kwargs)

    def get_dofs_velocity(self, *args, **kwargs):
        return self._part_rigid.get_dofs_velocity(*args, **kwargs)

    def get_dofs_force(self, *args, **kwargs):
        return self._part_rigid.get_dofs_force(*args, **kwargs)

    def get_dofs_control_force(self, *args, **kwargs):
        return self._part_rigid.get_dofs_control_force(*args, **kwargs)

    def set_dofs_velocity(self, *args, **kwargs):
        self._part_rigid.set_dofs_velocity(*args, **kwargs)

    def set_dofs_force(self, *args, **kwargs):
        self._part_rigid.set_dofs_force(*args, **kwargs)

    def control_dofs_position(self, *args, **kwargs):
        return self._part_rigid.control_dofs_position(*args, **kwargs)

    def control_dofs_velocity(self, *args, **kwargs):
        self._part_rigid.control_dofs_velocity(*args, **kwargs)

    def control_dofs_force(self, *args, **kwargs):
        self._part_rigid.control_dofs_force(*args, **kwargs)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- instantiation ----------------------------------
    # ------------------------------------------------------------------------------------

    def build(self):
        # can only be called here (at sim build)
        if not self.material.use_default_coupling and self._muscle_group_cache is not None:
            self._part_soft.set_muscle(
                muscle_group=gs.tensor(self._muscle_group_cache.astype(gs.np_int)),
                # no muscle direction as the soft body is actuated by rigid parts
            )

    def update_soft_part(self, f):
        if isinstance(self._part_soft, MPMEntity):
            self._kernel_update_soft_part_mpm(f=f)
        else:
            raise NotImplementedError

    @ti.kernel
    def _kernel_update_soft_part_mpm(self, f: ti.i32):
        for i in range(self._part_soft.n_particles):
            if self._solver_soft.particles_ng[f, i].active:
                i_global = i + self._part_soft.particle_start
                i_b = 0  # batch index always the first
                f_ = f
                if ti.static(not self._update_soft_part_at_pre_coupling):
                    f_ = f + 1  # NOTE: this is after g2p and thus we use f + 1

                # get corresponding link
                group_idx = self._solver_soft.particles_info[i_global].muscle_group

                link_idx = self._part_soft_info[group_idx].link_idx
                geom_idx = self._part_soft_info[group_idx].geom_idx
                trans_local_to_global = self._part_soft_info[group_idx].trans_local_to_global
                quat_local_to_global = self._part_soft_info[group_idx].quat_local_to_global

                link = self._solver_rigid.links_state[link_idx, i_b]
                geom_info = self._solver_rigid.geoms_info[geom_idx]

                # compute new pos in minimal coordinate using rigid-bodied dynamics
                x_init_pos = self._part_soft_init_positions[i]
                x_init_local = gu.ti_inv_transform_by_trans_quat(
                    x_init_pos, trans_local_to_global, quat_local_to_global
                )
                scaled_pos = gu.ti_transform_by_quat(
                    gu.ti_inv_transform_by_quat(geom_info.pos, geom_info.quat),
                    geom_info.quat,
                )
                tx_pos, tx_quat = gu.ti_transform_pos_quat_by_trans_quat(
                    scaled_pos,
                    geom_info.quat,
                    link.pos,
                    link.quat,
                )
                new_x_pos = gu.ti_transform_by_trans_quat(
                    x_init_local,
                    trans=tx_pos,
                    quat=tx_quat,
                )

                # compute velocity in closed-loop setting
                dt = self._solver_soft.substep_dt
                dt_scale = (
                    self._solver_soft.substep_dt / self._solver_rigid.dt
                )  # NOTE: move soft part incrementally at soft solver's substeps
                x_pos = self._solver_soft.particles[f_, i_global].pos
                xd_vel = (new_x_pos - x_pos) / dt
                xd_vel *= dt_scale  # assume linear scaling between the timestep difference of soft/rigid solver

                vel_d = xd_vel - self._solver_soft.particles[f_, i_global].vel
                vel_d *= ti.exp(-self._solver_soft.dt * self.material.damping)

                # soft-to-rigid coupling
                dt_for_rigid_acc = (
                    self._solver_rigid.dt
                )  # NOTE: use rigid dt here as we are sorta doing integration within soft solver substep
                mass_real = self._solver_soft.particles_info[i_global].mass / self._solver_soft._p_vol_scale
                acc = vel_d / dt_for_rigid_acc
                frc_vel = mass_real * acc
                frc_ang = (x_pos - link.COM).cross(frc_vel)
                self._solver_rigid.links_state[link_idx, i_b].cfrc_ext_vel += frc_vel
                self._solver_rigid.links_state[link_idx, i_b].cfrc_ext_ang += frc_ang

                # rigid-to-soft coupling # NOTE: this may lead to unstable feedback loop
                self._solver_soft.particles[f_, i_global].vel += vel_d * self.material.soft_dv_coef

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def n_dofs(self) -> int:
        return self._part_rigid.n_dofs

    @property
    def fixed(self) -> bool:
        return self._part_rigid.morph.fixed

    @property
    def part_rigid(self):
        return self._part_rigid

    @property
    def part_soft(self):
        return self._part_soft

    @property
    def solver_rigid(self):
        return self._solver_rigid

    @property
    def solver_soft(self):
        return self._solver_soft


# ------------------------------------------------------------------------------------
# ------------------------------------- misc -----------------------------------------
# ------------------------------------------------------------------------------------


def augment_link_world_coords(part_rigid):
    ordered_links_idx = [link.idx for link in part_rigid.links]  # now links are pre-ordered
    for i, i_l_global in enumerate(ordered_links_idx):
        i_l = i_l_global - part_rigid.link_start  # NOTE: seems like by default link idx is global
        link = part_rigid.links[i_l]
        i_p = link.parent_idx

        parent_pos = np.zeros((3,))
        parent_quat = gu.identity_quat()

        if i_p != -1:
            link_p = part_rigid.links[i_p - part_rigid.link_start]  # NOTE: seems like by default link idx is global
            parent_pos = link_p.init_x_pos
            parent_quat = link_p.init_x_quat

        link_is_fixed = link.joint.type == gs.JOINT_TYPE.FIXED
        if link.joint.type == gs.JOINT_TYPE.FREE or (link_is_fixed and i_p == -1):
            link.init_x_pos = link.pos
            link.init_x_quat = link.quat
        else:
            link.init_x_pos, link.init_x_quat = gu.transform_pos_quat_by_trans_quat(
                link.pos, link.quat, parent_pos, parent_quat
            )
            link.init_x_pos, link.init_x_quat = gu.transform_pos_quat_by_trans_quat(
                link.joint.pos, link.joint.quat, link.init_x_pos, link.init_x_quat
            )


def _visualize_muscle_group(positions, muscle_group):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    colors = np.zeros((positions.shape[0], 3))
    for gii, group_id in enumerate(np.unique(muscle_group)):
        colors[muscle_group == group_id] = np.random.uniform(0, 1, (3,))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw([pcd])


# ------------------------------------------------------------------------------------
# ----------------------------- default instantiation --------------------------------
# ------------------------------------------------------------------------------------


def default_func_instantiate_soft_from_rigid(
    scene,
    part_rigid,
    material_soft,
    material_hybrid,
    surface,
):
    meshes = []
    trans_local_to_global = []
    euler_local_to_global = []
    for link in part_rigid.links:
        if len(link.geoms) < 1:  # no collision geom
            continue

        geom = link.geoms[0]  # NOTE: collision geom is always the prior one based on the URDF parser
        trans, quat = gu.transform_pos_quat_by_trans_quat(
            geom.init_pos, geom.init_quat, link.init_x_pos, link.init_x_quat
        )
        euler = gu.quat_to_xyz(quat)

        # can also do link.init_verts here and it seems to have more indices than geom.init_verts (but there is no idx_offset_vert)
        lower = geom.init_verts.min(axis=0)
        upper = geom.init_verts.max(axis=0)
        center = (upper + lower) / 2.0
        verts = geom.init_verts
        assert hasattr(geom, "init_normals")
        inner_mesh = trimesh.Trimesh(
            vertices=verts,  # NOTE: scale is already applied here
            faces=geom.init_faces,
            vertex_normals=geom.init_normals,
        )
        outer_verts = verts + inner_mesh.vertex_normals * material_hybrid.thickness
        outer_mesh = trimesh.Trimesh(
            vertices=outer_verts,
            faces=geom.init_faces,
        )
        # mesh = trimesh.boolean.difference([outer_mesh, inner_mesh]) # wrap around the rigid link
        mesh = outer_mesh  # HACK to avoid `ValueError: No backends available for boolean operations!`

        meshes.append(mesh)
        trans_local_to_global.append(trans)
        euler_local_to_global.append(euler)

    rm_cross_link_overlap_mesh = False
    if rm_cross_link_overlap_mesh:
        for i, mesh in enumerate(meshes[:-1]):  # remove cross-link overlapping area
            meshes[i] = trimesh.boolean.difference([mesh] + meshes[i + 1 :])

    part_soft = scene.add_entity(
        material=material_soft,
        morph=gs.morphs.MeshSet(
            files=meshes,
            poss=trans_local_to_global,
            eulers=euler_local_to_global,
            scale=1,  # scale is already handled in geom.init_verts
        ),
        surface=surface,
    )

    return part_soft


def default_func_instantiate_rigid_from_soft(
    scene,
    mesh,
    morph,
    material_rigid,
    material_hybrid,
    surface,
):
    # skeletonization
    gelmesh = trimesh_to_gelmesh(mesh)
    graph_gel = skeletonization(gelmesh)

    # convert to nxgraph
    graph_nx = gel_graph_to_nx_graph(graph_gel)
    check_graph(graph_nx)

    # compute graph attribute
    graph_pos = graph_gel.positions()
    compute_graph_attribute(graph_nx, graph_pos)

    # reduce nxgraph
    graph_nx_reduced = reduce_graph(graph_nx, straight_thresh=60)
    check_graph(graph_nx_reduced)

    # to URDF
    G = graph_nx_reduced
    G, src_node = graph_to_tree(G)
    urdf = URDF.from_nxgraph(G)

    # add rigid entity
    mesh_center = (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2.0
    offset = (graph_pos[src_node] - mesh_center) * morph.scale
    pos_rigid = morph.pos + offset.astype(gs.np_float)
    quat_rigid = morph.quat
    scale_rigid = morph.scale
    morph_rigid = gs.morphs.URDF(
        file=urdf,
        pos=pos_rigid,
        quat=quat_rigid,
        scale=scale_rigid,
        fixed=material_hybrid.fixed,
    )
    part_rigid = scene.add_entity(
        material=material_rigid,
        morph=morph_rigid,
        surface=surface,
    )

    return part_rigid


def default_func_instantiate_rigid_soft_association_from_rigid(
    part_rigid,
    part_soft,
):
    muscle_group = None  # instantiate soft from rigid already set muscle group using MeshSet

    link_idcs = []
    geom_idcs = []
    trans_local_to_global = []
    quat_local_to_global = []
    for link in part_rigid.links:
        if len(link.geoms) < 1:  # no collision geom
            continue

        geom = link.geoms[0]  # NOTE: collision geom is always the prior one based on the URDF parser
        trans, quat = gu.transform_pos_quat_by_trans_quat(
            geom.init_pos, geom.init_quat, link.init_x_pos, link.init_x_quat
        )

        link_idcs.append(link.idx)
        geom_idcs.append(geom.idx)
        trans_local_to_global.append(trans)
        quat_local_to_global.append(quat)

    return muscle_group, link_idcs, geom_idcs, trans_local_to_global, quat_local_to_global


def default_func_instantiate_rigid_soft_association_from_soft(
    part_rigid,
    part_soft,
):
    # compute distance between particle position and line segment of each link
    augment_link_world_coords(part_rigid)
    positions = part_soft.init_particles
    dist_to_links = []
    link_idcs = []
    geom_idcs = []
    trans_local_to_global = []
    quat_local_to_global = []
    for i, link in enumerate(part_rigid.links):
        geom = link.geoms[0]
        link_end_x_pos = (
            gu.transform_by_trans_quat(
                np.zeros((3,)),
                trans=geom.init_pos * 2,
                quat=geom.init_quat,
            )
            + link.init_x_pos
        )  # TODO: check if this is correct

        p0 = link.init_x_pos
        p1 = link_end_x_pos

        line_vec = p1 - p0  # NOTE: assume a link is a line segment
        line_length = np.linalg.norm(line_vec)

        positions_proj_on_line_t = (positions - p0) @ line_vec / (line_length**2)
        positions_proj_on_line = p0 + positions_proj_on_line_t[:, None] * line_vec

        dist_to_p0 = np.linalg.norm(positions - p0, axis=-1)
        dist_to_p1 = np.linalg.norm(positions - p1, axis=-1)
        dist_to_line = np.sqrt(dist_to_p0**2 - ((positions_proj_on_line_t[:, None] * line_vec) ** 2).sum(-1))

        dist_to_link = (
            dist_to_p0 * (positions_proj_on_line_t < 0).astype(float)
            + dist_to_p1 * (positions_proj_on_line_t > 1).astype(float)
            + dist_to_line * np.logical_and(positions_proj_on_line_t >= 0, positions_proj_on_line_t <= 1).astype(float)
        )

        trans, quat = gu.transform_pos_quat_by_trans_quat(
            geom.init_pos, geom.init_quat, link.init_x_pos, link.init_x_quat
        )

        dist_to_links.append(dist_to_link)
        link_idcs.append(link.idx)
        geom_idcs.append(geom.idx)
        trans_local_to_global.append(trans)
        quat_local_to_global.append(quat)

    # get muscle group
    dist_to_links = np.array(dist_to_links)
    muscle_group = dist_to_links.argmin(axis=0)

    return muscle_group, link_idcs, geom_idcs, trans_local_to_global, quat_local_to_global
