import numpy as np
import taichi as ti
import torch

import genesis as gs
from genesis.utils import geom as gu
from genesis.utils import linalg as lu
from genesis.utils import mesh as mu
from genesis.utils import mjcf as mju
from genesis.utils import terrain as tu
from genesis.utils import urdf as uu
from genesis.utils.misc import tensor_to_array

from ..base_entity import Entity
from .rigid_joint import RigidJoint
from .rigid_link import RigidLink


@ti.data_oriented
class RigidEntity(Entity):
    """
    Entity class in rigid body systems. One rigid entity can be a robot, a terrain, a floating rigid body, etc.
    """

    def __init__(
        self,
        scene,
        solver,
        material,
        morph,
        surface,
        idx,
        idx_in_solver,
        link_start=0,
        joint_start=0,
        q_start=0,
        dof_start=0,
        geom_start=0,
        cell_start=0,
        vert_start=0,
        face_start=0,
        edge_start=0,
        vgeom_start=0,
        vvert_start=0,
        vface_start=0,
        visualize_contact=False,
    ):
        super().__init__(idx, scene, morph, solver, material, surface)

        self._idx_in_solver = idx_in_solver
        self._link_start = link_start
        self._joint_start = joint_start
        self._q_start = q_start
        self._dof_start = dof_start
        self._geom_start = geom_start
        self._cell_start = cell_start
        self._vert_start = vert_start
        self._face_start = face_start
        self._edge_start = edge_start
        self._vgeom_start = vgeom_start
        self._vvert_start = vvert_start
        self._vface_start = vface_start

        self._visualize_contact = visualize_contact

        self._is_built = False

        self._load_model()

    def _load_model(self):
        self._links = gs.List()
        self._joints = gs.List()

        if isinstance(self._morph, gs.morphs.Mesh):
            self._load_mesh(self._morph, self._surface)

        elif isinstance(self._morph, gs.morphs.MJCF):
            self._load_MJCF(self._morph, self._surface)

        elif isinstance(self._morph, gs.morphs.URDF):
            self._load_URDF(self._morph, self._surface)

        elif isinstance(self._morph, gs.morphs.Drone):
            self._load_URDF(self._morph, self._surface)

        elif isinstance(self._morph, gs.morphs.Primitive):
            self._load_primitive(self._morph, self._surface)

        elif isinstance(self._morph, gs.morphs.Terrain):
            self._load_terrain(self._morph, self._surface)
        else:
            gs.raise_exception(f"Unsupported morph: {self._morph}.")

        self._requires_jac_and_IK = self._morph.requires_jac_and_IK

        self._update_child_idxs()

    def _update_child_idxs(self):
        for link in self._links:
            if link.parent_idx != -1:
                parent_link = self._links[link.parent_idx_local]
                if link.idx not in parent_link.child_idxs:
                    parent_link.child_idxs.append(link.idx)

    def _load_primitive(self, morph, surface):
        if morph.fixed:
            joint_type = gs.JOINT_TYPE.FIXED
            n_qs = 0
            n_dofs = 0
            init_qpos = np.zeros(0)

        else:
            joint_type = gs.JOINT_TYPE.FREE
            n_qs = 7
            n_dofs = 6
            init_qpos = np.concatenate([morph.pos, morph.quat])

        if isinstance(morph, gs.options.morphs.Box):
            extents = np.array(morph.size)
            tmesh = mu.create_box(extents=extents)
            geom_data = extents
            geom_type = gs.GEOM_TYPE.BOX
            link_name_prefix = "box"

        elif isinstance(morph, gs.options.morphs.Sphere):
            tmesh = mu.create_sphere(radius=morph.radius)
            geom_data = np.array([morph.radius])
            geom_type = gs.GEOM_TYPE.SPHERE
            link_name_prefix = "sphere"

        elif isinstance(morph, gs.options.morphs.Cylinder):
            tmesh = mu.create_cylinder(radius=morph.radius, height=morph.height)
            geom_data = None
            geom_type = gs.GEOM_TYPE.MESH
            link_name_prefix = "cylinder"

        elif isinstance(morph, gs.options.morphs.Plane):
            tmesh = mu.create_plane(normal=morph.normal)
            geom_data = np.array(morph.normal)
            geom_type = gs.GEOM_TYPE.PLANE
            link_name_prefix = "plane"

        else:
            gs.raise_exception("Unsupported primitive shape")

        vmesh = gs.Mesh.from_trimesh(tmesh, surface=surface)
        mesh = gs.Mesh.from_trimesh(tmesh, surface=gs.surfaces.Collision())

        link = self._add_link(
            name=f"{link_name_prefix}_baselink",
            pos=np.array(morph.pos),
            quat=np.array(morph.quat),
            inertial_pos=gu.zero_pos(),
            inertial_quat=gu.identity_quat(),
            inertial_i=None,
            inertial_mass=None,
            parent_idx=-1,
            invweight=None,
        )
        self._add_joint(
            name=f"{link_name_prefix}_baselink_joint",
            n_qs=n_qs,
            n_dofs=n_dofs,
            type=joint_type,
            pos=gu.zero_pos(),
            quat=gu.identity_quat(),
            dofs_motion_ang=gu.default_dofs_motion_ang(n_dofs),
            dofs_motion_vel=gu.default_dofs_motion_vel(n_dofs),
            dofs_limit=gu.default_dofs_limit(n_dofs),
            dofs_invweight=gu.default_dofs_invweight(n_dofs),
            dofs_stiffness=gu.default_dofs_stiffness(n_dofs),
            dofs_sol_params=gu.default_solver_params(n_dofs),
            dofs_damping=gu.free_dofs_damping(n_dofs),
            dofs_armature=gu.free_dofs_armature(n_dofs),
            dofs_kp=gu.default_dofs_kp(n_dofs),
            dofs_kv=gu.default_dofs_kv(n_dofs),
            dofs_force_range=gu.default_dofs_force_range(n_dofs),
            init_qpos=init_qpos,
        )

        # contains one visual geom (vgeom) and one collision geom (geom)
        if morph.visualization:
            link._add_vgeom(
                vmesh=vmesh,
                init_pos=gu.zero_pos(),
                init_quat=gu.identity_quat(),
            )
        if morph.collision:
            link._add_geom(
                mesh=mesh,
                init_pos=gu.zero_pos(),
                init_quat=gu.identity_quat(),
                type=geom_type,
                friction=gu.default_friction() if self.material.friction is None else self.material.friction,
                sol_params=gu.default_solver_params(n=1)[0],
                data=geom_data,
                needs_coup=self.material.needs_coup,
            )

    def _load_mesh(self, morph, surface):
        if morph.fixed:
            joint_type = gs.JOINT_TYPE.FIXED
            n_qs = 0
            n_dofs = 0
            init_qpos = np.zeros(0)

        else:
            joint_type = gs.JOINT_TYPE.FREE
            n_qs = 7
            n_dofs = 6
            init_qpos = np.concatenate([morph.pos, morph.quat])
        link_name = morph.file.split("/")[-1].replace(".", "_")

        link = self._add_link(
            name=f"{link_name}_baselink",
            pos=np.array(morph.pos),
            quat=np.array(morph.quat),
            inertial_pos=None,
            inertial_quat=gu.identity_quat(),
            inertial_i=None,
            inertial_mass=None,
            parent_idx=-1,
            invweight=None,
        )
        self._add_joint(
            name=f"{link_name}_baselink_joint",
            n_qs=n_qs,
            n_dofs=n_dofs,
            type=joint_type,
            pos=gu.zero_pos(),
            quat=gu.identity_quat(),
            dofs_motion_ang=gu.default_dofs_motion_ang(n_dofs),
            dofs_motion_vel=gu.default_dofs_motion_vel(n_dofs),
            dofs_limit=gu.default_dofs_limit(n_dofs),
            dofs_invweight=gu.default_dofs_invweight(n_dofs),
            dofs_stiffness=gu.default_dofs_stiffness(n_dofs),
            dofs_sol_params=gu.default_solver_params(n_dofs),
            dofs_damping=gu.free_dofs_damping(n_dofs),
            dofs_armature=gu.free_dofs_armature(n_dofs),
            dofs_kp=gu.default_dofs_kp(n_dofs),
            dofs_kv=gu.default_dofs_kv(n_dofs),
            dofs_force_range=gu.default_dofs_force_range(n_dofs),
            init_qpos=init_qpos,
        )

        vmeshes, meshes = mu.parse_visual_and_col_mesh(morph, surface)

        if morph.visualization:
            for vmesh in vmeshes:
                link._add_vgeom(
                    vmesh=vmesh,
                    init_pos=gu.zero_pos(),
                    init_quat=gu.identity_quat(),
                )

        if morph.collision:
            for mesh in meshes:
                link._add_geom(
                    mesh=mesh,
                    init_pos=gu.zero_pos(),
                    init_quat=gu.identity_quat(),
                    type=gs.GEOM_TYPE.MESH,
                    friction=gu.default_friction() if self.material.friction is None else self.material.friction,
                    sol_params=gu.default_solver_params(n=1)[0],
                    needs_coup=self.material.needs_coup,
                )

    def _load_terrain(self, morph, surface):
        link = self._add_link(
            name="baselink",
            pos=np.array(morph.pos),
            quat=np.array(morph.quat),
            inertial_pos=None,
            inertial_quat=gu.identity_quat(),
            inertial_i=None,
            inertial_mass=None,
            parent_idx=-1,
            invweight=None,
        )
        self._add_joint(
            name="joint_baselink",
            n_qs=0,
            n_dofs=0,
            type=gs.JOINT_TYPE.FIXED,
            pos=gu.zero_pos(),
            quat=gu.identity_quat(),
            dofs_motion_ang=gu.default_dofs_motion_ang(0),
            dofs_motion_vel=gu.default_dofs_motion_vel(0),
            dofs_limit=gu.default_dofs_limit(0),
            dofs_invweight=gu.default_dofs_invweight(0),
            dofs_stiffness=gu.default_dofs_stiffness(0),
            dofs_sol_params=gu.default_solver_params(0),
            dofs_damping=gu.free_dofs_damping(0),
            dofs_armature=gu.free_dofs_armature(0),
            dofs_kp=gu.default_dofs_kp(0),
            dofs_kv=gu.default_dofs_kv(0),
            dofs_force_range=gu.default_dofs_force_range(0),
            init_qpos=np.zeros(0),
        )

        vmesh, mesh, self.terrain_hf = tu.parse_terrain(morph, surface)
        self.terrain_scale = np.array([morph.horizontal_scale, morph.vertical_scale])

        if morph.visualization:
            link._add_vgeom(
                vmesh=vmesh,
                init_pos=gu.zero_pos(),
                init_quat=gu.identity_quat(),
            )

        if morph.collision:
            link._add_geom(
                mesh=mesh,
                init_pos=gu.zero_pos(),
                init_quat=gu.identity_quat(),
                type=gs.GEOM_TYPE.TERRAIN,
                friction=gu.default_friction() if self.material.friction is None else self.material.friction,
                sol_params=gu.default_solver_params(n=1)[0],
                needs_coup=self.material.needs_coup,
            )

    def _load_MJCF(self, morph, surface):
        mj = mju.parse_mjcf(morph.file)
        n_geoms = mj.ngeom
        n_links = mj.nbody - 1  # link 0 in mj is world

        links_g_info = [list() for _ in range(n_links)]
        world_g_info = []

        # assign geoms to link
        for i_g in range(n_geoms):
            if mj.geom_bodyid[i_g] < 0:
                continue

            # address geoms directly attached to worldbody (0)
            elif mj.geom_bodyid[i_g] == 0:
                g_info = mju.parse_geom(mj, i_g, morph.scale, morph.convexify, surface, morph.file)
                if g_info is not None:
                    world_g_info.append(g_info)

            else:
                g_info = mju.parse_geom(mj, i_g, morph.scale, morph.convexify, surface, morph.file)
                if g_info is not None:
                    link_idx = mj.geom_bodyid[i_g] - 1
                    links_g_info[link_idx].append(g_info)

        l_infos = []
        j_infos = []

        q_offset, dof_offset = 0, 0

        for i_l in range(n_links):
            l_info, j_info = mju.parse_link(mj, i_l + 1, q_offset, dof_offset, morph.scale)

            l_infos.append(l_info)
            j_infos.append(j_info)

            q_offset += j_info["n_qs"]
            dof_offset += j_info["n_dofs"]

        l_infos, j_infos, links_g_info = uu._order_links(l_infos, j_infos, links_g_info)
        for i_l in range(len(l_infos)):
            l_info = l_infos[i_l]
            j_info = j_infos[i_l]

            if l_info["parent_idx"] < 0:  # base link
                if morph.pos is not None:
                    l_info["pos"] = np.array(morph.pos)
                    gs.logger.warning("Overriding base link's pos with user provided value in morph.")
                if morph.quat is not None:
                    l_info["quat"] = np.array(morph.quat)
                    gs.logger.warning("Overriding base link's quat with user provided value in morph.")

                if j_info["type"] == gs.JOINT_TYPE.FREE:
                    # in this case, l_info['pos'] and l_info['quat'] are actually not used in solver, but this initial value will be reflected
                    j_info["init_qpos"] = np.concatenate([l_info["pos"], l_info["quat"]])

            self._add_by_info(l_info, j_info, links_g_info[i_l], morph, surface)

        if world_g_info:
            l_world_info, j_world_info = mju.parse_link(mj, 0, q_offset, dof_offset, morph.scale)
            if morph.pos is not None:
                l_world_info["pos"] = np.array(morph.pos)
                gs.logger.warning("Overriding base link's pos with user provided value in morph.")
            if morph.quat is not None:
                l_world_info["quat"] = np.array(morph.quat)
                gs.logger.warning("Overriding base link's quat with user provided value in morph.")
            self._add_by_info(l_world_info, j_world_info, world_g_info, morph, surface)

    def _load_URDF(self, morph, surface):
        l_infos, j_infos = uu.parse_urdf(morph, surface)

        for i_l in range(len(l_infos)):
            l_info = l_infos[i_l]
            j_info = j_infos[i_l]

            if l_info["parent_idx"] < 0:  # base link
                l_info["pos"] = np.array(morph.pos)
                l_info["quat"] = np.array(morph.quat)

                if j_info["type"] == gs.JOINT_TYPE.FREE:
                    # in this case, l_info['pos'] and l_info['quat'] are actually not used in solver, but this initial value will be reflected in init_qpos
                    j_info["init_qpos"] = np.concatenate([l_info["pos"], l_info["quat"]])

            self._add_by_info(l_info, j_info, l_info["g_infos"], morph, surface)

    def _add_by_info(self, l_info, j_info, g_infos, morph, surface):
        link = self._add_link(
            name=l_info["name"],
            pos=l_info["pos"],
            quat=l_info["quat"],
            inertial_pos=l_info["inertial_pos"] if not morph.recompute_inertia else None,
            inertial_quat=l_info["inertial_quat"],
            inertial_i=l_info["inertial_i"] if not morph.recompute_inertia else None,
            inertial_mass=l_info["inertial_mass"] if not morph.recompute_inertia else None,
            parent_idx=l_info["parent_idx"] + self._link_start if l_info["parent_idx"] >= 0 else l_info["parent_idx"],
            invweight=l_info["invweight"],
        )
        self._add_joint(
            name=j_info["name"],
            n_qs=j_info["n_qs"],
            n_dofs=j_info["n_dofs"],
            type=j_info["type"],
            pos=j_info["pos"],
            quat=j_info["quat"],
            dofs_motion_ang=j_info["dofs_motion_ang"],
            dofs_motion_vel=j_info["dofs_motion_vel"],
            dofs_limit=j_info["dofs_limit"],
            dofs_invweight=j_info["dofs_invweight"],
            dofs_stiffness=j_info["dofs_stiffness"],
            dofs_sol_params=j_info["dofs_sol_params"],
            dofs_damping=j_info["dofs_damping"],
            dofs_armature=j_info["dofs_armature"],
            dofs_kp=j_info["dofs_kp"],
            dofs_kv=j_info["dofs_kv"],
            dofs_force_range=j_info["dofs_force_range"],
            init_qpos=j_info["init_qpos"],
        )

        # geoms
        for g_info in g_infos:
            if g_info["is_col"] and morph.collision:
                link._add_geom(
                    mesh=g_info["mesh"],
                    init_pos=g_info["pos"],
                    init_quat=g_info["quat"],
                    type=g_info["type"],
                    friction=g_info["friction"] if self.material.friction is None else self.material.friction,
                    sol_params=g_info["sol_params"],
                    data=g_info["data"],
                    needs_coup=self.material.needs_coup,
                )
            if not g_info["is_col"] and morph.visualization:
                link._add_vgeom(
                    vmesh=g_info["mesh"],
                    init_pos=g_info["pos"],
                    init_quat=g_info["quat"],
                )

    def _build(self):
        assert self.n_links == self.n_joints
        for link in self._links:
            link._build()

        self._n_qs = self.n_qs
        self._n_dofs = self.n_dofs
        self._is_built = True

        self._geoms = self.geoms
        self._vgeoms = self.vgeoms

        self._init_jac_and_IK()

    def _init_jac_and_IK(self):
        if not self._requires_jac_and_IK:
            return

        if self.n_dofs == 0:
            return

        self._jacobian = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape((6, self.n_dofs)))

        # compute joint limit in q space
        q_limit_lower = []
        q_limit_upper = []
        for joint in self._joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                q_limit_lower.append(joint.dofs_limit[:3, 0])
                q_limit_lower.append(-np.ones(4))  # quaternion lower bound
                q_limit_upper.append(joint.dofs_limit[:3, 1])
                q_limit_upper.append(np.ones(4))  # quaternion upper bound
            elif joint.type == gs.JOINT_TYPE.FIXED:
                pass
            else:
                q_limit_lower.append(joint.dofs_limit[:, 0])
                q_limit_upper.append(joint.dofs_limit[:, 1])
        self.q_limit = np.array([np.concatenate(q_limit_lower), np.concatenate(q_limit_upper)])

        # for storing intermediate results
        self._IK_n_tgts = self._solver._options.IK_max_targets
        self._IK_error_dim = self._IK_n_tgts * 6
        self._IK_mat = ti.field(
            dtype=gs.ti_float, shape=self._solver._batch_shape((self._IK_error_dim, self._IK_error_dim))
        )
        self._IK_inv = ti.field(
            dtype=gs.ti_float, shape=self._solver._batch_shape((self._IK_error_dim, self._IK_error_dim))
        )
        self._IK_L = ti.field(
            dtype=gs.ti_float, shape=self._solver._batch_shape((self._IK_error_dim, self._IK_error_dim))
        )
        self._IK_U = ti.field(
            dtype=gs.ti_float, shape=self._solver._batch_shape((self._IK_error_dim, self._IK_error_dim))
        )
        self._IK_y = ti.field(
            dtype=gs.ti_float, shape=self._solver._batch_shape((self._IK_error_dim, self._IK_error_dim))
        )
        self._IK_qpos_orig = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.n_qs))
        self._IK_qpos_best = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.n_qs))
        self._IK_delta_qpos = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.n_dofs))
        self._IK_vec = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._IK_error_dim))
        self._IK_err_pose = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._IK_error_dim))
        self._IK_err_pose_best = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self._IK_error_dim))
        self._IK_jacobian = ti.field(
            dtype=gs.ti_float, shape=self._solver._batch_shape((self._IK_error_dim, self.n_dofs))
        )
        self._IK_jacobian_T = ti.field(
            dtype=gs.ti_float, shape=self._solver._batch_shape((self.n_dofs, self._IK_error_dim))
        )

    def _add_link(
        self,
        name,
        pos,
        quat,
        inertial_pos,
        inertial_quat,
        inertial_i,
        inertial_mass,
        parent_idx,
        invweight,
    ):
        link = RigidLink(
            entity=self,
            name=name,
            idx=self.n_links + self._link_start,
            geom_start=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            vgeom_start=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            pos=pos,
            quat=quat,
            inertial_pos=inertial_pos,
            inertial_quat=inertial_quat,
            inertial_i=inertial_i,
            inertial_mass=inertial_mass,
            parent_idx=parent_idx,
            invweight=invweight,
            visualize_contact=self.visualize_contact,
        )
        self._links.append(link)
        return link

    def _add_joint(
        self,
        name,
        n_qs,
        n_dofs,
        type,
        pos,
        quat,
        dofs_motion_ang,
        dofs_motion_vel,
        dofs_limit,
        dofs_invweight,
        dofs_stiffness,
        dofs_sol_params,
        dofs_damping,
        dofs_armature,
        dofs_kp,
        dofs_kv,
        dofs_force_range,
        init_qpos,
    ):
        joint = RigidJoint(
            entity=self,
            name=name,
            idx=self.n_joints + self._joint_start,
            q_start=self.n_qs + self._q_start,
            dof_start=self.n_dofs + self._dof_start,
            n_qs=n_qs,
            n_dofs=n_dofs,
            type=type,
            pos=pos,
            quat=quat,
            dofs_motion_ang=dofs_motion_ang,
            dofs_motion_vel=dofs_motion_vel,
            dofs_limit=dofs_limit,
            dofs_invweight=dofs_invweight,
            dofs_stiffness=dofs_stiffness,
            dofs_sol_params=dofs_sol_params,
            dofs_damping=dofs_damping,
            dofs_armature=dofs_armature,
            dofs_kp=dofs_kp,
            dofs_kv=dofs_kv,
            dofs_force_range=dofs_force_range,
            init_qpos=init_qpos,
        )
        self._joints.append(joint)
        return joint

    # ------------------------------------------------------------------------------------
    # --------------------------------- Jacobian & IK ------------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_jacobian(self, link):
        """
        Get the Jacobian matrix for a target link.

        Parameters
        ----------
        link : RigidLink
            The target link.

        Returns
        -------
        jacobian : torch.Tensor
            The Jacobian matrix of shape (n_envs, 6, entity.n_dofs) or (6, entity.n_dofs) if n_envs == 0.
        """
        if not self._requires_jac_and_IK:
            gs.raise_exception(
                "Inverse kinematics and jacobian are disabled for this entity. Set `morph.requires_jac_and_IK` to True if you need them."
            )

        if self.n_dofs == 0:
            gs.raise_exception("Entity has zero dofs.")

        self._kernel_get_jacobian(link.idx)

        jacobian = self._jacobian.to_torch(gs.device).permute(2, 0, 1)
        if self._solver.n_envs == 0:
            jacobian = jacobian.squeeze(0)

        return jacobian

    @ti.kernel
    def _kernel_get_jacobian(self, tgt_link_idx: ti.i32):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._solver._B):
            self._func_get_jacobian(
                tgt_link_idx,
                i_b,
                ti.Vector.one(gs.ti_int, 3),
                ti.Vector.one(gs.ti_int, 3),
            )

    @ti.func
    def _func_get_jacobian(self, tgt_link_idx, i_b, pos_mask, rot_mask):
        for i_row, i_d in ti.ndrange(6, self.n_dofs):
            self._jacobian[i_row, i_d, i_b] = 0.0

        tgt_link_pos = self._solver.links_state[tgt_link_idx, i_b].pos
        i_l = tgt_link_idx
        while i_l > -1:
            l_info = self._solver.links_info[i_l]
            l_state = self._solver.links_state[i_l, i_b]

            if l_info.joint_type == gs.JOINT_TYPE.FIXED:
                pass

            elif l_info.joint_type == gs.JOINT_TYPE.REVOLUTE:
                i_d = l_info.dof_start
                i_d_jac = i_d - self._dof_start
                rotation = gu.ti_transform_by_quat(self._solver.dofs_info[i_d].motion_ang, l_state.quat)
                translation = rotation.cross(tgt_link_pos - l_state.pos)

                self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]
                self._jacobian[3, i_d_jac, i_b] = rotation[0] * rot_mask[0]
                self._jacobian[4, i_d_jac, i_b] = rotation[1] * rot_mask[1]
                self._jacobian[5, i_d_jac, i_b] = rotation[2] * rot_mask[2]

            elif l_info.joint_type == gs.JOINT_TYPE.PRISMATIC:
                i_d = l_info.dof_start
                i_d_jac = i_d - self._dof_start
                translation = gu.ti_transform_by_quat(self._solver.dofs_info[i_d].motion_vel, l_state.quat)

                self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]

            elif l_info.joint_type == gs.JOINT_TYPE.FREE:
                # translation
                for i_d_ in range(3):
                    i_d = l_info.dof_start + i_d_
                    i_d_jac = i_d - self._dof_start

                    self._jacobian[i_d_, i_d_jac, i_b] = 1.0 * pos_mask[i_d_]

                # rotation
                for i_d_ in range(3):
                    i_d = l_info.dof_start + i_d_ + 3
                    i_d_jac = i_d - self._dof_start
                    rotation = self._solver.dofs_info[i_d].motion_ang
                    translation = rotation.cross(tgt_link_pos - l_state.pos)

                    self._jacobian[0, i_d_jac, i_b] = translation[0] * pos_mask[0]
                    self._jacobian[1, i_d_jac, i_b] = translation[1] * pos_mask[1]
                    self._jacobian[2, i_d_jac, i_b] = translation[2] * pos_mask[2]
                    self._jacobian[3, i_d_jac, i_b] = rotation[0] * rot_mask[0]
                    self._jacobian[4, i_d_jac, i_b] = rotation[1] * rot_mask[1]
                    self._jacobian[5, i_d_jac, i_b] = rotation[2] * rot_mask[2]

            i_l = l_info.parent_idx

    @gs.assert_built
    def inverse_kinematics(
        self,
        link,
        pos=None,
        quat=None,
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,  # 0.5 mm
        rot_tol=5e-3,  # 0.28 degree
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        return_error=False,
    ):
        """
        Compute inverse kinematics for a single target link.

        Parameters
        ----------
        link : RigidLink
            The link to be used as the end-effector.
        pos : None | array_like, shape (3,), optional
            The target position. If None, position error will not be considered. Defaults to None.
        quat : None | array_like, shape (4,), optional
            The target orientation. If None, orientation error will not be considered. Defaults to None.
        init_qpos : None | array_like, shape (n_dofs,), optional
            Initial qpos used for solving IK. If None, the current qpos will be used. Defaults to None.
        respect_joint_limit : bool, optional
            Whether to respect joint limits. Defaults to True.
        max_samples : int, optional
            Number of resample attempts. Defaults to 50.
        max_solver_iters : int, optional
            Maximum number of solver iterations per sample. Defaults to 20.
        damping : float, optional
            Damping for damped least squares. Defaults to 0.01.
        pos_tol : float, optional
            Position tolerance for normalized position error (in meter). Defaults to 1e-4.
        rot_tol : float, optional
            Rotation tolerance for normalized rotation vector error (in radian). Defaults to 1e-4.
        pos_mask : list, shape (3,), optional
            Mask for position error. Defaults to [True, True, True]. E.g.: If you only care about position along x and y, you can set it to [True, True, False].
        rot_mask : list, shape (3,), optional
            Mask for rotation axis alignment. Defaults to [True, True, True]. E.g.: If you only want the link's Z-axis to be aligned with the Z-axis in the given quat, you can set it to [False, False, True].
        max_step_size : float, optional
            Maximum step size in q space for each IK solver step. Defaults to 0.5.
        return_error : bool, optional
            Whether to return the final errorqpos. Defaults to False.

        Returns
        -------
        qpos : array_like, shape (n_dofs,) or (n_envs, n_dofs)
            Solver qpos (joint positions).
        (optional) error_pose : array_like, shape (6,) or (n_envs, 6)
            Pose error for each target. The 6-vector is [err_pos_x, err_pos_y, err_pos_z, err_rot_x, err_rot_y, err_rot_z]. Only returned if `return_error` is True.
        """
        if self._solver.n_envs > 0:
            if pos is not None:
                if pos.shape[0] != self._solver.n_envs:
                    gs.raise_exception("First dimension of `pos` must be equal to `scene.n_envs`.")
            if quat is not None:
                if quat.shape[0] != self._solver.n_envs:
                    gs.raise_exception("First dimension of `quat` must be equal to `scene.n_envs`.")

        ret = self.inverse_kinematics_multilink(
            links=[link],
            poss=[pos] if pos is not None else [],
            quats=[quat] if quat is not None else [],
            init_qpos=init_qpos,
            respect_joint_limit=respect_joint_limit,
            max_samples=max_samples,
            max_solver_iters=max_solver_iters,
            damping=damping,
            pos_tol=pos_tol,
            rot_tol=rot_tol,
            pos_mask=pos_mask,
            rot_mask=rot_mask,
            max_step_size=max_step_size,
            return_error=return_error,
        )

        if return_error:
            qpos, error_pose = ret
            error_pose = error_pose.squeeze(-2)  # 1 single link
            return qpos, error_pose

        else:
            return ret

    @gs.assert_built
    def inverse_kinematics_multilink(
        self,
        links,
        poss=[],
        quats=[],
        init_qpos=None,
        respect_joint_limit=True,
        max_samples=50,
        max_solver_iters=20,
        damping=0.01,
        pos_tol=5e-4,  # 0.5 mm
        rot_tol=5e-3,  # 0.28 degree
        pos_mask=[True, True, True],
        rot_mask=[True, True, True],
        max_step_size=0.5,
        return_error=False,
    ):
        """
        Compute inverse kinematics for  multiple target links.

        Parameters
        ----------
        links : list of RigidLink
            List of links to be used as the end-effectors.
        poss : list, optional
            List of target positions. If empty, position error will not be considered. Defaults to None.
        quats : list, optional
            List of target orientations. If empty, orientation error will not be considered. Defaults to None.
        init_qpos : array_like, shape (n_dofs,), optional
            Initial qpos used for solving IK. If None, the current qpos will be used. Defaults to None.
        respect_joint_limit : bool, optional
            Whether to respect joint limits. Defaults to True.
        max_samples : int, optional
            Number of resample attempts. Defaults to 50.
        max_solver_iters : int, optional
            Maximum number of solver iterations per sample. Defaults to 20.
        damping : float, optional
            Damping for damped least squares. Defaults to 0.01.
        pos_tol : float, optional
            Position tolerance for normalized position error (in meter). Defaults to 1e-4.
        rot_tol : float, optional
            Rotation tolerance for normalized rotation vector error (in radian). Defaults to 1e-4.
        pos_mask : list, shape (3,), optional
            Mask for position error. Defaults to [True, True, True]. E.g.: If you only care about position along x and y, you can set it to [True, True, False].
        rot_mask : list, shape (3,), optional
            Mask for rotation axis alignment. Defaults to [True, True, True]. E.g.: If you only want the link's Z-axis to be aligned with the Z-axis in the given quat, you can set it to [False, False, True].
        max_step_size : float, optional
            Maximum step size in q space for each IK solver step. Defaults to 0.5.
        return_error : bool, optional
            Whether to return the final errorqpos. Defaults to False.


        Returns
        -------
        qpos : array_like, shape (n_dofs,) or (n_envs, n_dofs)
            Solver qpos (joint positions).
        (optional) error_pose : array_like, shape (6,) or (n_envs, 6)
            Pose error for each target. The 6-vector is [err_pos_x, err_pos_y, err_pos_z, err_rot_x, err_rot_y, err_rot_z]. Only returned if `return_error` is True.
        """

        if not self._requires_jac_and_IK:
            gs.raise_exception(
                "Inverse kinematics and jacobian are disabled for this entity. Set `morph.requires_jac_and_IK` to True if you need them."
            )

        if self.n_dofs == 0:
            gs.raise_exception("Entity has zero dofs.")

        n_links = len(links)
        if n_links == 0:
            gs.raise_exception("Target link not provided.")

        if len(poss) == n_links:
            if self._solver.n_envs > 0:
                if poss[0].shape[0] != self._solver.n_envs:
                    gs.raise_exception("First dimension of elements in `poss` must be equal to scene.n_envs.")
        elif len(poss) == 0:
            if self._solver.n_envs == 0:
                poss = [gu.zero_pos()] * n_links
            else:
                poss = [self._solver._batch_array(gu.zero_pos(), True)] * n_links
            pos_mask = [False, False, False]
        else:
            gs.raise_exception("Accepting only `poss` with length equal to `links` or empty list.")

        if len(quats) == n_links:
            if self._solver.n_envs > 0:
                if quats[0].shape[0] != self._solver.n_envs:
                    gs.raise_exception("First dimension of elements in `quats` must be equal to scene.n_envs.")
        elif len(quats) == 0:
            if self._solver.n_envs == 0:
                quats = [gu.identity_quat()] * n_links
            else:
                quats = [self._solver._batch_array(gu.identity_quat(), True)] * n_links
            rot_mask = [False, False, False]
        else:
            gs.raise_exception("Accepting only `quats` with length equal to `links` or empty list.")

        if init_qpos is not None:
            init_qpos = torch.as_tensor(init_qpos, dtype=gs.tc_float)
            if init_qpos.shape[-1] != self.n_qs:
                gs.raise_exception(
                    f"Size of last dimension `init_qpos` does not match entity's `n_qs`: {init_qpos.shape[-1]} vs {self.n_qs}."
                )

            init_qpos = self._solver._process_dim(init_qpos)
            custom_init_qpos = True

        else:
            init_qpos = torch.empty((0, 0), dtype=gs.tc_float)  # B * n_qs, dummy
            custom_init_qpos = False

        # pos and rot mask
        pos_mask = torch.as_tensor(pos_mask, dtype=bool, device=gs.device)
        if len(pos_mask) != 3:
            gs.raise_exception("`pos_mask` must have length 3.")
        rot_mask = torch.as_tensor(rot_mask, dtype=bool, device=gs.device)
        if len(rot_mask) != 3:
            gs.raise_exception("`rot_mask` must have length 3.")
        if sum(rot_mask) == 1:
            rot_mask = ~rot_mask
        elif sum(rot_mask) == 2:
            gs.raise_exception("You can only align 0, 1 axis or all 3 axes.")
        else:
            pass  # nothing needs to change for 0 or 3 axes

        links_idx = torch.as_tensor([link.idx for link in links], dtype=gs.tc_int, device=gs.device)
        poss = torch.stack(
            [self._solver._process_dim(torch.as_tensor(pos, dtype=gs.tc_float, device=gs.device)) for pos in poss]
        )
        quats = torch.stack(
            [self._solver._process_dim(torch.as_tensor(quat, dtype=gs.tc_float, device=gs.device)) for quat in quats]
        )

        self._kernel_inverse_kinematics(
            links_idx,
            poss,
            quats,
            n_links,
            custom_init_qpos,
            init_qpos,
            max_samples,
            max_solver_iters,
            damping,
            pos_tol,
            rot_tol,
            pos_mask,
            rot_mask,
            max_step_size,
            respect_joint_limit,
        )
        qpos = self._IK_qpos_best.to_torch(gs.device).permute(1, 0)
        if self._solver.n_envs == 0:
            qpos = qpos.squeeze(0)

        if return_error:
            error_pose = (
                self._IK_err_pose_best.to_torch(gs.device)
                .reshape([self._IK_n_tgts, 6, -1])[:n_links]
                .permute([2, 0, 1])
            )
            if self._solver.n_envs == 0:
                error_pose = error_pose.squeeze(0)
            return qpos, error_pose

        else:
            return qpos

    @ti.kernel
    def _kernel_inverse_kinematics(
        self,
        links_idx: ti.types.ndarray(),
        poss: ti.types.ndarray(),
        quats: ti.types.ndarray(),
        n_links: ti.i32,
        custom_init_qpos: ti.i32,
        init_qpos: ti.types.ndarray(),
        max_samples: ti.i32,
        max_solver_iters: ti.i32,
        damping: ti.f32,
        pos_tol: ti.f32,
        rot_tol: ti.f32,
        pos_mask_: ti.types.ndarray(),
        rot_mask_: ti.types.ndarray(),
        max_step_size: ti.f32,
        respect_joint_limit: ti.i32,
    ):
        # convert to ti Vector
        pos_mask = ti.Vector([pos_mask_[0], pos_mask_[1], pos_mask_[2]], dt=gs.ti_float)
        rot_mask = ti.Vector([rot_mask_[0], rot_mask_[1], rot_mask_[2]], dt=gs.ti_float)
        n_error_dims = 6 * n_links

        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(self._solver._B):
            # save original qpos
            for i_q in range(self.n_qs):
                self._IK_qpos_orig[i_q, i_b] = self._solver.qpos[i_q + self._q_start, i_b]

            if custom_init_qpos:
                for i_q in range(self.n_qs):
                    self._solver.qpos[i_q + self._q_start, i_b] = init_qpos[i_b, i_q]

            for i_error in range(n_error_dims):
                self._IK_err_pose_best[i_error, i_b] = 1e4

            solved = False
            for i_sample in range(max_samples):
                for _ in range(max_solver_iters):
                    # run FK to update link states using current q
                    self._solver._func_forward_kinematics_entity(self._idx_in_solver, i_b)

                    # compute error
                    solved = True
                    for i_ee in range(n_links):
                        i_l_ee = links_idx[i_ee]

                        tgt_pos_i = ti.Vector([poss[i_ee, i_b, 0], poss[i_ee, i_b, 1], poss[i_ee, i_b, 2]])
                        err_pos_i = tgt_pos_i - self._solver.links_state[i_l_ee, i_b].pos
                        for k in range(3):
                            err_pos_i[k] *= pos_mask[k]
                        if err_pos_i.norm() > pos_tol:
                            solved = False

                        tgt_quat_i = ti.Vector(
                            [quats[i_ee, i_b, 0], quats[i_ee, i_b, 1], quats[i_ee, i_b, 2], quats[i_ee, i_b, 3]]
                        )
                        err_rot_i = gu.ti_quat_to_rotvec(
                            gu.ti_transform_quat_by_quat(
                                gu.ti_inv_quat(self._solver.links_state[i_l_ee, i_b].quat), tgt_quat_i
                            )
                        )
                        for k in range(3):
                            err_rot_i[k] *= rot_mask[k]
                        if err_rot_i.norm() > rot_tol:
                            solved = False

                        # put into multi-link error array
                        for k in range(3):
                            self._IK_err_pose[i_ee * 6 + k, i_b] = err_pos_i[k]
                            self._IK_err_pose[i_ee * 6 + k + 3, i_b] = err_rot_i[k]

                    if solved:
                        break

                    # compute multi-link jacobian
                    for i_ee in range(n_links):
                        # update jacobian for ee link
                        i_l_ee = links_idx[i_ee]
                        self._func_get_jacobian(i_l_ee, i_b, pos_mask, rot_mask)

                        # copy to multi-link jacobian
                        for i_error, i_dof in ti.ndrange(6, self.n_dofs):
                            i_row = i_ee * 6 + i_error
                            self._IK_jacobian[i_row, i_dof, i_b] = self._jacobian[i_error, i_dof, i_b]

                    # compute dq = jac.T @ inverse(jac @ jac.T + diag) @ error
                    lu.mat_transpose(self._IK_jacobian, self._IK_jacobian_T, n_error_dims, self.n_dofs, i_b)
                    lu.mat_mul(
                        self._IK_jacobian,
                        self._IK_jacobian_T,
                        self._IK_mat,
                        n_error_dims,
                        self.n_dofs,
                        n_error_dims,
                        i_b,
                    )
                    lu.mat_add_eye(self._IK_mat, damping**2, n_error_dims, i_b)
                    lu.mat_inverse(self._IK_mat, self._IK_L, self._IK_U, self._IK_y, self._IK_inv, n_error_dims, i_b)
                    lu.mat_mul_vec(self._IK_inv, self._IK_err_pose, self._IK_vec, n_error_dims, n_error_dims, i_b)
                    lu.mat_mul_vec(
                        self._IK_jacobian_T, self._IK_vec, self._IK_delta_qpos, self.n_dofs, n_error_dims, i_b
                    )
                    for i in range(self.n_dofs):
                        self._IK_delta_qpos[i, i_b] = ti.math.clamp(
                            self._IK_delta_qpos[i, i_b], -max_step_size, max_step_size
                        )

                    # update q
                    self._solver._func_integrate_dq_entity(
                        self._IK_delta_qpos, self._idx_in_solver, i_b, respect_joint_limit
                    )

                if not solved:
                    # re-compute final error if exited not due to solved
                    self._solver._func_forward_kinematics_entity(self._idx_in_solver, i_b)
                    solved = True
                    for i_ee in range(n_links):
                        i_l_ee = links_idx[i_ee]

                        tgt_pos_i = ti.Vector([poss[i_ee, i_b, 0], poss[i_ee, i_b, 1], poss[i_ee, i_b, 2]])
                        err_pos_i = tgt_pos_i - self._solver.links_state[i_l_ee, i_b].pos
                        for k in range(3):
                            err_pos_i[k] *= pos_mask[k]
                        if err_pos_i.norm() > pos_tol:
                            solved = False

                        tgt_quat_i = ti.Vector(
                            [quats[i_ee, i_b, 0], quats[i_ee, i_b, 1], quats[i_ee, i_b, 2], quats[i_ee, i_b, 3]]
                        )
                        err_rot_i = gu.ti_quat_to_rotvec(
                            gu.ti_transform_quat_by_quat(
                                gu.ti_inv_quat(self._solver.links_state[i_l_ee, i_b].quat), tgt_quat_i
                            )
                        )
                        for k in range(3):
                            err_rot_i[k] *= rot_mask[k]
                        if err_rot_i.norm() > rot_tol:
                            solved = False

                        # put into multi-link error array
                        for k in range(3):
                            self._IK_err_pose[i_ee * 6 + k, i_b] = err_pos_i[k]
                            self._IK_err_pose[i_ee * 6 + k + 3, i_b] = err_rot_i[k]

                if solved:
                    for i_q in range(self.n_qs):
                        self._IK_qpos_best[i_q, i_b] = self._solver.qpos[i_q + self._q_start, i_b]
                    for i_error in range(n_error_dims):
                        self._IK_err_pose_best[i_error, i_b] = self._IK_err_pose[i_error, i_b]

                else:
                    # copy to _IK_qpos if this sample is better
                    improved = True
                    for i_ee in range(n_links):
                        error_pos_i = ti.Vector([self._IK_err_pose[i_ee * 6 + i_error, i_b] for i_error in range(3)])
                        error_rot_i = ti.Vector([self._IK_err_pose[i_ee * 6 + i_error, i_b] for i_error in range(3, 6)])
                        error_pos_best = ti.Vector(
                            [self._IK_err_pose_best[i_ee * 6 + i_error, i_b] for i_error in range(3)]
                        )
                        error_rot_best = ti.Vector(
                            [self._IK_err_pose_best[i_ee * 6 + i_error, i_b] for i_error in range(3, 6)]
                        )
                        if error_pos_i.norm() > error_pos_best.norm() or error_rot_i.norm() > error_rot_best.norm():
                            improved = False
                            break

                    if improved:
                        for i_q in range(self.n_qs):
                            self._IK_qpos_best[i_q, i_b] = self._solver.qpos[i_q + self._q_start, i_b]
                        for i_error in range(n_error_dims):
                            self._IK_err_pose_best[i_error, i_b] = self._IK_err_pose[i_error, i_b]

                    # Resample init q
                    if respect_joint_limit and i_sample < max_samples - 1:
                        for i_l in range(self.link_start, self.link_end):
                            l_info = self._solver.links_info[i_l]
                            dof_info = self._solver.dofs_info[l_info.dof_start]
                            q_start = l_info.q_start

                            if l_info.joint_type == gs.JOINT_TYPE.FREE:
                                pass

                            elif (
                                l_info.joint_type == gs.JOINT_TYPE.REVOLUTE
                                or l_info.joint_type == gs.JOINT_TYPE.PRISMATIC
                            ):
                                if ti.math.isinf(dof_info.limit[0]) or ti.math.isinf(dof_info.limit[1]):
                                    pass
                                else:
                                    self._solver.qpos[q_start, i_b] = dof_info.limit[0] + ti.random() * (
                                        dof_info.limit[1] - dof_info.limit[0]
                                    )
                    else:
                        pass  # When respect_joint_limit=False, we can simply continue from the last solution

            # restore original qpos and link state
            for i_q in range(self.n_qs):
                self._solver.qpos[i_q + self._q_start, i_b] = self._IK_qpos_orig[i_q, i_b]
            self._solver._func_forward_kinematics_entity(self._idx_in_solver, i_b)

    # ------------------------------------------------------------------------------------
    # --------------------------------- motion planing -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def plan_path(
        self,
        qpos_goal,
        qpos_start=None,
        timeout=5.0,
        smooth_path=True,
        num_waypoints=100,
        ignore_collision=False,
        ignore_joint_limit=False,
        planner="RRTConnect",
    ):
        """
        Plan a path from `qpos_start` to `qpos_goal`.

        Parameters
        ----------
        qpos_goal : array_like
            The goal state.
        qpos_start : None | array_like, optional
            The start state. If None, the current state of the rigid entity will be used. Defaults to None.
        timeout : float, optional
            The maximum time (in seconds) allowed for the motion planning algorithm to find a solution. Defaults to 5.0.
        smooth_path : bool, optional
            Whether to smooth the path after finding a solution. Defaults to True.
        num_waypoints : int, optional
            The number of waypoints to interpolate the path. If None, no interpolation will be performed. Defaults to 100.
        ignore_collision : bool, optional
            Whether to ignore collision checking during motion planning. Defaults to False.
        ignore_joint_limit : bool, optional
            Whether to ignore joint limits during motion planning. Defaults to False.
        planner : str, optional
            The name of the motion planning algorithm to use. Supported planners: 'PRM', 'RRT', 'RRTConnect', 'RRTstar', 'EST', 'FMT', 'BITstar', 'ABITstar'. Defaults to 'RRTConnect'.

        Returns
        -------
        waypoints : list
            A list of waypoints representing the planned path. Each waypoint is an array storing the entity's qpos of a single time step.
        """

        ########## validate ##########
        try:
            from ompl import base as ob
            from ompl import geometric as og
            from ompl import util as ou

            ou.setLogLevel(ou.LOG_ERROR)
        except:
            gs.raise_exception(
                "Failed to import OMPL. Did you install? (For installation instructions, see https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html#optional-motion-planning)"
            )

        supported_planners = [
            "PRM",
            "RRT",
            "RRTConnect",
            "RRTstar",
            "EST",
            "FMT",
            "BITstar",
            "ABITstar",
        ]
        if planner not in supported_planners:
            gs.raise_exception(f"Planner {planner} is not supported. Supported planners: {supported_planners}.")

        if self._solver.n_envs > 0:
            gs.raise_exception("Motion planning is not supported for batched envs (yet).")

        if self.n_qs != self.n_dofs:
            gs.raise_exception("Motion planning is not yet supported for rigid entities with free joints.")

        if qpos_start is None:
            qpos_start = self.get_qpos()
        qpos_start = tensor_to_array(qpos_start)
        qpos_goal = tensor_to_array(qpos_goal)

        if qpos_start.shape != (self.n_qs,) or qpos_goal.shape != (self.n_qs,):
            gs.raise_exception("Invalid shape for `qpos_start` or `qpos_goal`.")

        ######### process joint limit ##########
        if ignore_joint_limit:
            q_limit_lower = np.full_like(self.q_limit[0], -1e6)
            q_limit_upper = np.full_like(self.q_limit[1], 1e6)
        else:
            q_limit_lower = self.q_limit[0]
            q_limit_upper = self.q_limit[1]

        if (qpos_start < q_limit_lower).any() or (qpos_start > q_limit_upper).any():
            gs.logger.warning(
                "`qpos_start` exceeds joint limit. Relaxing joint limit to contain `qpos_start` for planning."
            )
            q_limit_lower = np.minimum(q_limit_lower, qpos_start)
            q_limit_upper = np.maximum(q_limit_upper, qpos_start)

        if (qpos_goal < q_limit_lower).any() or (qpos_goal > q_limit_upper).any():
            gs.logger.warning(
                "`qpos_goal` exceeds joint limit. Relaxing joint limit to contain `qpos_goal` for planning."
            )
            q_limit_lower = np.minimum(q_limit_lower, qpos_goal)
            q_limit_upper = np.maximum(q_limit_upper, qpos_goal)

        ######### setup OMPL ##########
        space = ob.RealVectorStateSpace(self.n_qs)
        bounds = ob.RealVectorBounds(self.n_qs)

        for i_q in range(self.n_qs):
            bounds.setLow(i_q, q_limit_lower[i_q])
            bounds.setHigh(i_q, q_limit_upper[i_q])
        space.setBounds(bounds)
        ss = og.SimpleSetup(space)
        if ignore_collision:
            ss.setStateValidityChecker(ob.StateValidityCheckerFn(lambda state: True))
        else:
            ss.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_ompl_state_valid))
        ss.setPlanner(getattr(og, planner)(ss.getSpaceInformation()))

        state_start = ob.State(space)
        state_goal = ob.State(space)
        for i_q in range(self.n_qs):
            state_start[i_q] = float(qpos_start[i_q])
            state_goal[i_q] = float(qpos_goal[i_q])
        ss.setStartAndGoalStates(state_start, state_goal)

        ######### solve ##########
        qpos_cur = self.get_qpos()
        solved = ss.solve(timeout)
        waypoints = []
        if solved:
            gs.logger.info("Path solution found successfully.")
            path = ss.getSolutionPath()
            if smooth_path:
                ps = og.PathSimplifier(ss.getSpaceInformation())
                # simplify the path
                try:
                    ps.partialShortcutPath(path)
                    ps.ropeShortcutPath(path)
                except:
                    ps.shortcutPath(path)
                ps.smoothBSpline(path)

            if num_waypoints is not None:
                path.interpolate(num_waypoints)
            waypoints = self._ompl_states_to_tensor_list(path.getStates())
        else:
            gs.logger.warning("Path planning failed. Returning empty path.")

        ########## restore original state #########
        self.set_qpos(qpos_cur)

        return waypoints

    def _is_ompl_state_valid(self, state):
        self.set_qpos(self._ompl_state_to_tensor(state))
        collision_pairs = self.detect_collision()

        if len(collision_pairs) > 0:
            return False
        else:
            return True

    def _ompl_states_to_tensor_list(self, states):
        tensor_list = []
        for state in states:
            tensor_list.append(self._ompl_state_to_tensor(state))
        return tensor_list

    def _ompl_state_to_tensor(self, state):
        tensor = torch.empty(self.n_qs, dtype=gs.tc_float, device=gs.device)
        for i in range(self.n_qs):
            tensor[i] = state[i]
        return tensor

    # ------------------------------------------------------------------------------------
    # ---------------------------------- control & io ------------------------------------
    # ------------------------------------------------------------------------------------

    def get_joint(self, name=None, id=None):
        """
        Get a RigidJoint object by name or id.

        Parameters
        ----------
        name : str, optional
            The name of the joint. Defaults to None.
        id : str, optional
            The id of the joint. This can be a substring of the joint's id. Defaults to None.

        Returns
        -------
        joint : RigidJoint
            The joint object.
        """

        if name is not None:
            for joint in self._joints:
                if joint.name == name:
                    return joint
            gs.raise_exception(f"Joint not found for name: {name}.")

        elif id is not None:
            for joint in self._joints:
                if id in str(joint.id):
                    return joint
            gs.raise_exception(f"Joint not found for id: {id}.")

        else:
            gs.raise_exception("Neither `name` nor `id` is provided.")

    def get_link(self, name=None, id=None):
        """
        Get a RigidLink object by name or id.

        Parameters
        ----------
        name : str, optional
            The name of the link. Defaults to None.
        id : str, optional
            The id of the link. This can be a substring of the link's id. Defaults to None.

        Returns
        -------
        link : RigidLink
            The link object.
        """

        if name is not None:
            for link in self._links:
                if link.name == name:
                    return link
            gs.raise_exception(f"Link not found for name: {name}.")

        elif id is not None:
            for link in self._links:
                if id in str(link.id):
                    return link
            gs.raise_exception(f"Link not found for id: {id}.")

        else:
            gs.raise_exception("Neither `name` nor `id` is provided.")

    @gs.assert_built
    def get_pos(self, envs_idx=None):
        """
        Returns position of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        pos : torch.Tensor, shape (3,) or (n_envs, 3)
            The position of the entity's base link.
        """
        return self._solver.get_links_pos([self.base_link_idx], envs_idx).squeeze(-2)

    @gs.assert_built
    def get_quat(self, envs_idx=None):
        """
        Returns quaternion of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        quat : torch.Tensor, shape (4,) or (n_envs, 4)
            The quaternion of the entity's base link.
        """
        return self._solver.get_links_quat([self.base_link_idx], envs_idx).squeeze(-2)

    @gs.assert_built
    def get_vel(self, envs_idx=None):
        """
        Returns linear velocity of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        vel : torch.Tensor, shape (3,) or (n_envs, 3)
            The linear velocity of the entity's base link.
        """
        return self._solver.get_links_vel([self.base_link_idx], envs_idx).squeeze(-2)

    @gs.assert_built
    def get_ang(self, envs_idx=None):
        """
        Returns angular velocity of the entity's base link.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        ang : torch.Tensor, shape (3,) or (n_envs, 3)
            The angular velocity of the entity's base link.
        """
        return self._solver.get_links_ang([self.base_link_idx], envs_idx).squeeze(-2)

    @gs.assert_built
    def get_links_pos(self, envs_idx=None):
        """
        Returns position of all the entity's links.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        pos : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The position of all the entity's links.
        """
        return self._solver.get_links_pos(np.arange(self.link_start, self.link_end), envs_idx)

    @gs.assert_built
    def get_links_quat(self, envs_idx=None):
        """
        Returns quaternion of all the entity's links.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        quat : torch.Tensor, shape (n_links, 4) or (n_envs, n_links, 4)
            The quaternion of all the entity's links.
        """
        return self._solver.get_links_quat(np.arange(self.link_start, self.link_end), envs_idx)

    @gs.assert_built
    def get_links_vel(self, envs_idx=None):
        """
        Returns linear velocity of all the entity's links.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        vel : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear velocity of all the entity's links.
        """
        return self._solver.get_links_vel(np.arange(self.link_start, self.link_end), envs_idx)

    @gs.assert_built
    def get_links_ang(self, envs_idx=None):
        """
        Returns angular velocity of all the entity's links.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        ang : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The angular velocity of all the entity's links.
        """
        return self._solver.get_links_ang(np.arange(self.link_start, self.link_end), envs_idx)

    @gs.assert_built
    def set_pos(self, pos, zero_velocity=True, envs_idx=None):
        """
        Set position of the entity's base link.

        Parameters
        ----------
        pos : array_like
            The position to set.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a sudden change in entity pose.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """

        if self.base_link.is_fixed:
            gs.logger.warning("Base link is fixed. Overriding base link pose.")

        pos = torch.as_tensor(pos)
        if self._solver.n_envs == 0:
            if pos.ndim != 1:
                gs.raise_exception("`pos` must be a 1D tensor for unparallelized scene.")
        else:
            if pos.ndim != 2:
                gs.raise_exception("`pos` must be a 2D tensor for parallelized scene.")

        self._solver.set_links_pos(pos.unsqueeze(-2), [self.base_link_idx], envs_idx)

        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx)

    @gs.assert_built
    def set_quat(self, quat, zero_velocity=True, envs_idx=None):
        """
        Set quaternion of the entity's base link.

        Parameters
        ----------
        quat : array_like
            The quaternion to set.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a sudden change in entity pose.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """

        if self.base_link.is_fixed:
            gs.logger.warning("Base link is fixed. Overriding base link pose.")

        quat = torch.as_tensor(quat)
        if self._solver.n_envs == 0:
            if quat.ndim != 1:
                gs.raise_exception("`quat` must be a 1D tensor for unparallelized scene.")
        else:
            if quat.ndim != 2:
                gs.raise_exception("`quat` must be a 2D tensor for parallelized scene.")

        self._solver.set_links_quat(torch.as_tensor(quat).unsqueeze(-2), [self.base_link_idx], envs_idx)

        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx)

    @gs.assert_built
    def get_verts(self):
        """
        Get the all vertices of the entity (using collision geoms).

        Returns
        -------
        verts : torch.Tensor, shape (n_verts, 3) or (n_envs, n_verts, 3)
            The vertices of the entity (using collision geoms).
        """

        tensor = torch.empty(self._solver._batch_shape((self.n_verts, 3), True), dtype=gs.tc_float, device=gs.device)
        self._kernel_get_verts(tensor)
        if self._solver.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_verts(self, tensor: ti.types.ndarray()):
        for i_g_, i_b in ti.ndrange(self.n_geoms, self._solver._B):
            i_g = i_g_ + self._geom_start
            self._solver._func_update_verts_for_geom(i_g, i_b)

        for i, j, b in ti.ndrange(self.n_verts, 3, self._solver._B):
            idx_vert = i + self._vert_start
            tensor[b, i, j] = self._solver.verts_state[idx_vert, b].pos[j]

    @gs.assert_built
    def get_AABB(self):
        """
        Get the axis-aligned bounding box (AABB) of the entity (using collision geoms).

        Returns
        -------
        AABB : torch.Tensor, shape (2, 3) or (n_envs, 2, 3)
            The axis-aligned bounding box (AABB) of the entity (using collision geoms).
        """
        if self.n_geoms == 0:
            gs.raise_exception("Entity has no geoms.")

        verts = self.get_verts()
        AABB = torch.concatenate(
            [verts.min(axis=-2, keepdim=True)[0], verts.max(axis=-2, keepdim=True)[0]],
            axis=-2,
        )
        return AABB

    def _get_qs_idx_local(self, qs_idx_local=None):
        if qs_idx_local is None:
            qs_idx_local = torch.arange(self.n_qs, dtype=torch.int32, device=gs.device)
        else:
            qs_idx_local = torch.as_tensor(qs_idx_local, dtype=gs.tc_int)
            if (qs_idx_local < 0).any() or (qs_idx_local >= self.n_qs).any():
                gs.raise_exception("`qs_idx_local` exceeds valid range.")
        return qs_idx_local

    def _get_qs_idx(self, qs_idx_local=None):
        return self._get_qs_idx_local(qs_idx_local) + self._q_start

    def _get_dofs_idx_local(self, dofs_idx_local=None):
        if dofs_idx_local is None:
            dofs_idx_local = torch.arange(self.n_dofs, dtype=torch.int32, device=gs.device)
        else:
            dofs_idx_local = torch.as_tensor(dofs_idx_local, dtype=gs.tc_int, device=gs.device)
            if (dofs_idx_local < 0).any() or (dofs_idx_local >= self.n_dofs).any():
                gs.raise_exception("`dofs_idx_local` exceeds valid range.")
        return dofs_idx_local

    def _get_dofs_idx(self, dofs_idx_local=None):
        return self._get_dofs_idx_local(dofs_idx_local) + self._dof_start

    @gs.assert_built
    def set_qpos(self, qpos, qs_idx_local=None, zero_velocity=True, envs_idx=None):
        """
        Set the entity's qpos.

        Parameters
        ----------
        qpos : array_like
            The qpos to set.
        qs_idx_local : None | array_like, optional
            The indices of the qpos to set. If None, all qpos will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a sudden change in entity pose.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        self._solver.set_qpos(qpos, self._get_qs_idx(qs_idx_local), envs_idx)

        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx)

    @gs.assert_built
    def set_dofs_kp(self, kp, dofs_idx_local=None):
        """
        Set the entity's dofs' positional gains for the PD controller.

        Parameters
        ----------
        kp : array_like
            The positional gains to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        """

        self._solver.set_dofs_kp(kp, self._get_dofs_idx(dofs_idx_local))

    @gs.assert_built
    def set_dofs_kv(self, kv, dofs_idx_local=None):
        """
        Set the entity's dofs' velocity gains for the PD controller.

        Parameters
        ----------
        kv : array_like
            The velocity gains to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        """
        self._solver.set_dofs_kv(kv, self._get_dofs_idx(dofs_idx_local))

    @gs.assert_built
    def set_dofs_force_range(self, lower, upper, dofs_idx_local=None):
        """
        Set the entity's dofs' force range.

        Parameters
        ----------
        lower : array_like
            The lower bounds of the force range.
        upper : array_like
            The upper bounds of the force range.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        """

        self._solver.set_dofs_force_range(lower, upper, self._get_dofs_idx(dofs_idx_local))

    @gs.assert_built
    def set_dofs_velocity(self, velocity, dofs_idx_local=None, envs_idx=None):
        """
        Set the entity's dofs' velocity.

        Parameters
        ----------
        velocity : array_like
            The velocity to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        self._solver.set_dofs_velocity(velocity, self._get_dofs_idx(dofs_idx_local), envs_idx)

    @gs.assert_built
    def set_dofs_position(self, position, dofs_idx_local=None, zero_velocity=True, envs_idx=None):
        """
        Set the entity's dofs' position.

        Parameters
        ----------
        position : array_like
            The position to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a sudden change in entity pose.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        self._solver.set_dofs_position(position, self._get_dofs_idx(dofs_idx_local), envs_idx)

        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx)

    @gs.assert_built
    def control_dofs_force(self, force, dofs_idx_local=None, envs_idx=None):
        """
        Control the entity's dofs' motor force. This is used for force/torque control.

        Parameters
        ----------
        force : array_like
            The force to apply.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """

        self._solver.control_dofs_force(force, self._get_dofs_idx(dofs_idx_local), envs_idx)

    @gs.assert_built
    def control_dofs_velocity(self, velocity, dofs_idx_local=None, envs_idx=None):
        """
        Set the PD controller's target velocity for the entity's dofs. This is used for velocity control.

        Parameters
        ----------
        velocity : array_like
            The target velocity to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        self._solver.control_dofs_velocity(velocity, self._get_dofs_idx(dofs_idx_local), envs_idx)

    @gs.assert_built
    def control_dofs_position(self, position, dofs_idx_local=None, envs_idx=None):
        """
        Set the PD controller's target position for the entity's dofs. This is used for position control.

        Parameters
        ----------
        position : array_like
            The target position to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """

        self._solver.control_dofs_position(position, self._get_dofs_idx(dofs_idx_local), envs_idx)

    @gs.assert_built
    def get_qpos(self, qs_idx_local=None, envs_idx=None):
        """
        Get the entity's qpos.

        Parameters
        ----------
        qs_idx_local : None | array_like, optional
            The indices of the qpos to get. If None, all qpos will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        qpos : torch.Tensor, shape (n_qs,) or (n_envs, n_qs)
            The entity's qpos.
        """
        return self._solver.get_qpos(self._get_qs_idx(qs_idx_local), envs_idx)

    @gs.assert_built
    def get_dofs_control_force(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' internal control force, computed based on the position/velocity control command.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        control_force : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' internal control force.
        """
        return self._solver.get_dofs_control_force(self._get_dofs_idx(dofs_idx_local), envs_idx)

    @gs.assert_built
    def get_dofs_force(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' internal force at the current time step.

        Note
        ----
        Different from `get_dofs_control_force`, this function returns the actual internal force experienced by all the dofs at the current time step.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        force : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' force.
        """
        return self._solver.get_dofs_force(self._get_dofs_idx(dofs_idx_local), envs_idx)

    @gs.assert_built
    def get_dofs_velocity(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' velocity.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        velocity : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' velocity.
        """
        return self._solver.get_dofs_velocity(self._get_dofs_idx(dofs_idx_local), envs_idx)

    @gs.assert_built
    def get_dofs_position(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the entity's dofs' position.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        position : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The entity's dofs' position.
        """
        return self._solver.get_dofs_position(self._get_dofs_idx(dofs_idx_local), envs_idx)

    @gs.assert_built
    def get_dofs_kp(self, dofs_idx_local=None):
        """
        Get the positional gain (kp) for the entity's dofs used by the PD controller.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.

        Returns
        -------
        kp : torch.Tensor, shape (n_dofs,)
            The positional gain (kp) for the entity's dofs.
        """
        return self._solver.get_dofs_kp(self._get_dofs_idx(dofs_idx_local))

    @gs.assert_built
    def get_dofs_kv(self, dofs_idx_local=None):
        """
        Get the velocity gain (kv) for the entity's dofs used by the PD controller.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.

        Returns
        -------
        kv : torch.Tensor, shape (n_dofs,)
            The velocity gain (kv) for the entity's dofs.
        """
        return self._solver.get_dofs_kv(self._get_dofs_idx(dofs_idx_local))

    @gs.assert_built
    def get_dofs_force_range(self, dofs_idx_local=None):
        """
        Get the force range (min and max limits) for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_dofs,)
            The lower limit of the force range for the entity's dofs.
        upper_limit : torch.Tensor, shape (n_dofs,)
            The upper limit of the force range for the entity's dofs.
        """
        return self._solver.get_dofs_force_range(self._get_dofs_idx(dofs_idx_local))

    @gs.assert_built
    def get_dofs_limit(self, dofs_idx=None):
        """
        Get the positional limits (min and max) for the entity's dofs.

        Parameters
        ----------
        dofs_idx : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_dofs,)
            The lower limit of the positional limits for the entity's dofs.
        upper_limit : torch.Tensor, shape (n_dofs,)
            The upper limit of the positional limits for the entity's dofs.
        """
        return self._solver.get_dofs_limit(self._get_dofs_idx(dofs_idx))

    @gs.assert_built
    def zero_all_dofs_velocity(self, envs_idx=None):
        """
        Zero the velocity of all the entity's dofs.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """

        envs_idx = self._solver._get_envs_idx(envs_idx)
        if self._solver.n_envs == 0:
            qvel = torch.zeros(self.n_dofs, dtype=gs.tc_float, device=gs.device)
            self.set_dofs_velocity(qvel)
        else:
            qvel = torch.zeros(
                self._solver._batch_shape(self.n_dofs, True, B=len(envs_idx)), dtype=gs.tc_float, device=gs.device
            )
            self.set_dofs_velocity(qvel, envs_idx=envs_idx)

    @gs.assert_built
    def detect_collision(self, env_idx=0):
        """
        Detects collision for the entity. This only supports a single environment.

        Note
        ----
        This function re-detects real-time collision for the entity, so it doesn't rely on scene.step() and can be used for applications like motion planning, which doesn't require physical simulation during state sampling.

        Parameters
        ----------
        env_idx : int, optional
            The index of the environment. Defaults to 0.
        """

        all_collision_pairs = self._solver.detect_collision(env_idx)
        collision_pairs = all_collision_pairs[
            np.logical_and(
                all_collision_pairs >= self.geom_start,
                all_collision_pairs < self.geom_end,
            ).any(axis=1)
        ]
        return collision_pairs

    @gs.assert_built
    def get_contacts(self, with_entity=None):
        """
        Returns contact information computed during the most recent `scene.step()`.
        If `with_entity` is provided, only returns contact information involving the caller entity and the specified `with_entity`. Otherwise, returns all contact information involving the caller entity.

        The returned dict contains the following keys (a contact pair consists of two geoms: A and B):

        - 'geom_a'     : The global geom index of geom A in the contact pair. (actual geom object can be obtained by scene.rigid_solver.geoms[geom_a])
        - 'geom_b'     : The global geom index of geom B in the contact pair. (actual geom object can be obtained by scene.rigid_solver.geoms[geom_b])
        - 'link_a'     : The global link index of link A (that contains geom A) in the contact pair. (actual link object can be obtained by scene.rigid_solver.links[link_a])
        - 'link_b'     : The global link index of link B (that contains geom B) in the contact pair. (actual link object can be obtained by scene.rigid_solver.links[link_b])
        - 'position'   : The contact position in world frame.
        - 'force_a'    : The contact force applied to geom A.
        - 'force_b'    : The contact force applied to geom B.
        - 'valid_mask' : (Only when scene is parallelized) A boolean mask indicating whether the contact information is valid.

        The shape of each entry is (n_envs, n_contacts, ...) for scene with parallel envs, and (n_contacts, ...) for non-parallelized scene.

        Parameters
        ----------
        with_entity : RigidEntity, optional
            The entity to check contact with. Defaults to None.

        Returns
        -------
        contact_info : dict
            The contact information.
        """

        scene_contact_info = self._solver.collider.contact_data.to_numpy()
        n_contacts = self._solver.collider.n_contacts.to_numpy()

        valid_mask = np.logical_or(
            np.logical_and(
                scene_contact_info["geom_a"] >= self.geom_start,
                scene_contact_info["geom_a"] < self.geom_end,
            ),
            np.logical_and(
                scene_contact_info["geom_b"] >= self.geom_start,
                scene_contact_info["geom_b"] < self.geom_end,
            ),
        )
        if with_entity is not None:
            if self.idx == with_entity.idx:
                gs.raise_exception("`with_entity` cannot be the same as the caller entity.")

            valid_mask = np.logical_and(
                valid_mask,
                np.logical_or(
                    np.logical_and(
                        scene_contact_info["geom_a"] >= with_entity.geom_start,
                        scene_contact_info["geom_a"] < with_entity.geom_end,
                    ),
                    np.logical_and(
                        scene_contact_info["geom_b"] >= with_entity.geom_start,
                        scene_contact_info["geom_b"] < with_entity.geom_end,
                    ),
                ),
            )
        valid_mask = np.logical_and(valid_mask, np.arange(valid_mask.shape[0])[:, None] < n_contacts)

        if self._solver.n_envs == 0:
            valid_idx = np.where(valid_mask.squeeze())[0]
            contact_info = {
                "geom_a": scene_contact_info["geom_a"][valid_idx, 0],
                "geom_b": scene_contact_info["geom_b"][valid_idx, 0],
                "link_a": scene_contact_info["link_a"][valid_idx, 0],
                "link_b": scene_contact_info["link_b"][valid_idx, 0],
                "position": scene_contact_info["pos"][valid_idx, 0],
                "force_a": -scene_contact_info["force"][valid_idx, 0],
                "force_b": scene_contact_info["force"][valid_idx, 0],
            }
        else:
            max_env_collisions = np.max(n_contacts)
            contact_info = {
                "geom_a": scene_contact_info["geom_a"][:max_env_collisions].T,
                "geom_b": scene_contact_info["geom_b"][:max_env_collisions].T,
                "link_a": scene_contact_info["link_a"][:max_env_collisions].T,
                "link_b": scene_contact_info["link_b"][:max_env_collisions].T,
                "position": scene_contact_info["pos"][:max_env_collisions].transpose([1, 0, 2]),
                "force_a": -scene_contact_info["force"][:max_env_collisions].transpose([1, 0, 2]),
                "force_b": scene_contact_info["force"][:max_env_collisions].transpose([1, 0, 2]),
                "valid_mask": valid_mask[:max_env_collisions].T,
            }
        return contact_info

    def get_links_net_contact_force(self):
        """
        Returns net force applied on each links due to direct external contacts.

        Returns
        -------
        entity_links_force : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The net force applied on each links due to direct external contacts.
        """
        scene_links_force = (
            self._solver.links_state.contact_force.to_torch(gs.device).clone().detach().permute([1, 0, 2])
        )
        entity_links_force = scene_links_force[:, self.link_start : self.link_end]

        if self._solver.n_envs == 0:
            entity_links_force = entity_links_force.squeeze(0)

        return entity_links_force

    def set_friction_ratio(self, friction_ratio, link_indices, envs_idx=None):
        """
        Set the friction ratio of the geoms of the specified links.
        Parameters
        ----------
        friction_ratio : torch.Tensor, shape (n_envs, n_links)
            The friction ratio
        link_indices : array_like
            The indices of the links to set friction ratio.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        geom_indices = []
        for i in link_indices:
            for j in range(self._links[i].n_geoms):
                geom_indices.append(self._links[i]._geom_start + j)
        self._solver.set_geoms_friction_ratio(
            torch.cat(
                [
                    ratio.unsqueeze(-1).repeat(1, self._links[j].n_geoms)
                    for j, ratio in zip(link_indices, friction_ratio.unbind(-1))
                ],
                dim=-1,
            ),
            geom_indices,
            envs_idx,
        )

    def set_friction(self, friction):
        """
        Set the friction coefficient of all the links (and in turn, geoms) of the rigid entity.
        Note that for a pair of geoms in contact, the actual friction coefficient is set to be max(geom_a.friction, geom_b.friction), so you need to set for both geoms.

        Note
        ----
        In actual simulation, friction will be computed using `max(max(ga_info.friction, gb_info.friction), 1e-2)`; i.e. the minimum friction coefficient is 1e-2.

        Parameters
        ----------
        friction : float
            The friction coefficient to set.
        """

        if friction < 1e-2 or friction > 5.0:
            gs.raise_exception("`friction` must be in the range [1e-2, 5.0] for simulation stability.")

        for link in self._links:
            link.set_friction(friction)

    def set_mass_shift(self, mass_shift, link_indices, envs_idx=None):
        """
        Set the mass shift of specified links.
        Parameters
        ----------
        mass : torch.Tensor, shape (n_envs, n_links)
            The mass shift
        link_indices : array_like
            The indices of the links to set mass shift.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        for i in range(len(link_indices)):
            link_indices[i] += self._link_start
        self._solver.set_links_mass_shift(mass_shift, link_indices, envs_idx)

    def set_COM_shift(self, com_shift, link_indices, envs_idx=None):
        """
        Set the center of mass (COM) shift of specified links.
        Parameters
        ----------
        com : torch.Tensor, shape (n_envs, n_links, 3)
            The COM shift
        link_indices : array_like
            The indices of the links to set COM shift.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        for i in range(len(link_indices)):
            link_indices[i] += self._link_start
        self._solver.set_links_COM_shift(com_shift, link_indices, envs_idx)

    @gs.assert_built
    def get_mass(self):
        """
        Get the total mass of the entity in kg.

        Returns
        -------
        mass : float
            The total mass of the entity in kg.
        """
        mass = 0.0
        for link in self.links:
            mass += link.get_mass()
        return mass

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def visualize_contact(self):
        """Whether to visualize contact force."""
        return self._visualize_contact

    @property
    def init_qpos(self):
        """The initial qpos of the entity."""
        return np.concatenate([joint.init_qpos for joint in self._joints])

    @property
    def n_qs(self):
        """The number of `q` (generalized coordinates) of the entity."""
        if self._is_built:
            return self._n_qs
        else:
            return sum([joint.n_qs for joint in self._joints])

    @property
    def n_links(self):
        """The number of `RigidLink` in the entity."""
        return len(self._links)

    @property
    def n_joints(self):
        """The number of `RigidJoint` in the entity."""
        return len(self._joints)

    @property
    def n_dofs(self):
        """The number of degrees of freedom (DOFs) of the entity."""
        if self._is_built:
            return self._n_dofs
        else:
            return sum([joint.n_dofs for joint in self._joints])

    @property
    def n_geoms(self):
        """The number of `RigidGeom` in the entity."""
        return sum([link.n_geoms for link in self._links])

    @property
    def n_cells(self):
        """The number of sdf cells in the entity."""
        return sum([link.n_cells for link in self._links])

    @property
    def n_verts(self):
        """The number of vertices (from collision geom `RigidGeom`) in the entity."""
        return sum([link.n_verts for link in self._links])

    @property
    def n_faces(self):
        """The number of faces (from collision geom `RigidGeom`) in the entity."""
        return sum([link.n_faces for link in self._links])

    @property
    def n_edges(self):
        """The number of edges (from collision geom `RigidGeom`) in the entity."""
        return sum([link.n_edges for link in self._links])

    @property
    def n_vgeoms(self):
        """The number of vgeoms (visual geoms - `RigidVisGeom`) in the entity."""
        return sum([link.n_vgeoms for link in self._links])

    @property
    def n_vverts(self):
        """The number of vverts (visual vertices, from vgeoms) in the entity."""
        return sum([link.n_vverts for link in self._links])

    @property
    def n_vfaces(self):
        """The number of vfaces (visual faces, from vgeoms) in the entity."""
        return sum([link.n_vfaces for link in self._links])

    @property
    def geom_start(self):
        """The index of the entity's first RigidGeom in the scene."""
        return self._geom_start

    @property
    def geom_end(self):
        """The index of the entity's last RigidGeom in the scene *plus one*."""
        return self._geom_start + self.n_geoms

    @property
    def cell_start(self):
        """The start index the entity's sdf cells in the scene."""
        return self._cell_start

    @property
    def cell_end(self):
        """The end index the entity's sdf cells in the scene *plus one*."""
        return self._cell_start + self.n_cells

    @property
    def base_link_idx(self):
        """The index of the entity's base link in the scene."""
        return self._link_start

    @property
    def link_start(self):
        """The index of the entity's first RigidLink in the scene."""
        return self._link_start

    @property
    def link_end(self):
        """The index of the entity's last RigidLink in the scene *plus one*."""
        return self._link_start + self.n_links

    @property
    def dof_start(self):
        """The index of the entity's first degree of freedom (DOF) in the scene."""
        return self._dof_start

    @property
    def gravity_compensation(self):
        """Apply a force to compensate gravity. A value of 1 will make a zero-gravity behavior. Default to 0"""
        return self.material.gravity_compensation

    @property
    def dof_end(self):
        """The index of the entity's last degree of freedom (DOF) in the scene *plus one*."""
        return self._dof_start + self.n_dofs

    @property
    def vert_start(self):
        """The index of the entity's first `vert` (collision vertex) in the scene."""
        return self._vert_start

    @property
    def vvert_start(self):
        """The index of the entity's first `vvert` (visual vertex) in the scene."""
        return self._vvert_start

    @property
    def face_start(self):
        """The index of the entity's first `face` (collision face) in the scene."""
        return self._face_start

    @property
    def vface_start(self):
        """The index of the entity's first `vface` (visual face) in the scene."""
        return self._vface_start

    @property
    def edge_start(self):
        """The index of the entity's first `edge` (collision edge) in the scene."""
        return self._edge_start

    @property
    def q_start(self):
        """The index of the entity's first `q` (generalized coordinates) in the scene."""
        return self._q_start

    @property
    def q_end(self):
        """The index of the entity's last `q` (generalized coordinates) in the scene *plus one*."""
        return self._q_start + self.n_qs

    @property
    def geoms(self):
        """The list of collision geoms (`RigidGeom`) in the entity."""
        if self.is_built:
            return self._geoms
        else:
            geoms = gs.List()
            for link in self._links:
                geoms += link.geoms
            return geoms

    @property
    def vgeoms(self):
        """The list of visual geoms (`RigidVisGeom`) in the entity."""
        if self.is_built:
            return self._vgeoms
        else:
            vgeoms = gs.List()
            for link in self._links:
                vgeoms += link.vgeoms
            return vgeoms

    @property
    def links(self):
        """The list of links (`RigidLink`) in the entity."""
        return self._links

    @property
    def joints(self):
        """The list of joints (`RigidJoint`) in the entity."""
        return self._joints

    @property
    def base_link(self):
        """The base link of the entity"""
        return self._links[0]

    @property
    def base_joint(self):
        """The base joint of the entity"""
        return self._joints[0]
