from copy import copy
from itertools import chain
from typing import TYPE_CHECKING, Any

import gstaichi as ti
import numpy as np
import torch
import trimesh

import genesis as gs
from genesis.engine.materials.base import Material
from genesis.options.morphs import Morph
from genesis.options.surfaces import Surface
from genesis.utils import array_class
from genesis.utils import linalg as lu
from genesis.utils import geom as gu
from genesis.utils import mesh as mu
from genesis.utils import mjcf as mju
from genesis.utils import terrain as tu
from genesis.utils import urdf as uu
from genesis.utils.misc import DeprecationError

from ..base_entity import Entity
from .rigid_equality import RigidEquality
from .rigid_geom import RigidGeom
from .rigid_joint import RigidJoint
from .rigid_link import RigidLink
from .impl import (
    RigidEntityLoaderMixin,
    RigidEntityKinematicsMixin,
    RigidEntityAccessorMixin,
    compute_inertial_from_geom_infos,
    kernel_rigid_entity_inverse_kinematics,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


@ti.data_oriented
class RigidEntity(RigidEntityLoaderMixin, RigidEntityKinematicsMixin, RigidEntityAccessorMixin, Entity):
    """
    Entity class in rigid body systems. One rigid entity can be a robot, a terrain, a floating rigid body, etc.
    """

    # override typing
    _solver: "RigidSolver"

    def __init__(
        self,
        scene: "Scene",
        solver: "RigidSolver",
        material: Material,
        morph: Morph,
        surface: Surface,
        idx: int,
        idx_in_solver,
        link_start: int = 0,
        joint_start: int = 0,
        q_start=0,
        dof_start=0,
        geom_start=0,
        cell_start=0,
        vert_start=0,
        free_verts_state_start=0,
        fixed_verts_state_start=0,
        face_start=0,
        edge_start=0,
        vgeom_start=0,
        vvert_start=0,
        vface_start=0,
        equality_start=0,
        visualize_contact: bool = False,
        morph_heterogeneous: list[Morph] | None = None,
    ):
        super().__init__(idx, scene, morph, solver, material, surface)

        self._idx_in_solver = idx_in_solver
        self._link_start: int = link_start
        self._joint_start: int = joint_start
        self._q_start = q_start
        self._dof_start = dof_start
        self._geom_start = geom_start
        self._cell_start = cell_start
        self._vert_start = vert_start
        self._face_start = face_start
        self._edge_start = edge_start
        self._free_verts_state_start = free_verts_state_start
        self._fixed_verts_state_start = fixed_verts_state_start
        self._vgeom_start = vgeom_start
        self._vvert_start = vvert_start
        self._vface_start = vface_start
        self._equality_start = equality_start

        self._free_verts_idx_local = torch.tensor([], dtype=gs.tc_int, device=gs.device)
        self._fixed_verts_idx_local = torch.tensor([], dtype=gs.tc_int, device=gs.device)

        self._batch_fixed_verts = morph.batch_fixed_verts

        self._visualize_contact: bool = visualize_contact

        self._is_built: bool = False
        self._is_attached: bool = False

        # Heterogeneous simulation support (convert None to [] for consistency)
        self._morph_heterogeneous = morph_heterogeneous if morph_heterogeneous is not None else []
        self._enable_heterogeneous = bool(self._morph_heterogeneous)

        self._load_model()

        # Initialize target variables and checkpoint
        self._tgt_keys = ("pos", "quat", "qpos", "dofs_velocity")
        self._tgt = dict()
        self._tgt_buffer = list()
        self._ckpt = dict()
        self._update_tgt_while_set = self._solver._requires_grad

    def _update_tgt(self, key, value):
        # Set [self._tgt] value while keeping the insertion order between keys. When a new key is inserted or an existing
        # key is updated, the new element should be inserted at the end of the dict. This is because we need to keep
        # the insertion order to correctly pass the gradients in the backward pass.
        self._tgt.pop(key, None)
        self._tgt[key] = value

    def init_ckpt(self):
        pass

    def _build(self):
        for link in self._links:
            link._build()

        self._n_qs = self.n_qs
        self._n_dofs = self.n_dofs
        self._n_geoms = self.n_geoms
        self._geoms = self.geoms
        self._vgeoms = self.vgeoms
        self._is_built = True

        verts_start = 0
        free_verts_idx_local, fixed_verts_idx_local = [], []
        for link in self.links:
            verts_idx = torch.arange(verts_start, verts_start + link.n_verts, dtype=gs.tc_int, device=gs.device)
            if link.is_fixed and not self._batch_fixed_verts:
                fixed_verts_idx_local.append(verts_idx)
            else:
                free_verts_idx_local.append(verts_idx)
            verts_start += link.n_verts
        if free_verts_idx_local:
            self._free_verts_idx_local = torch.cat(free_verts_idx_local)
        if fixed_verts_idx_local:
            self._fixed_verts_idx_local = torch.cat(fixed_verts_idx_local)
        self._n_free_verts = len(self._free_verts_idx_local)
        self._n_fixed_verts = len(self._fixed_verts_idx_local)

        self._init_jac_and_IK()

    def _init_jac_and_IK(self):
        if not self._requires_jac_and_IK:
            return

        if self.n_dofs == 0:
            return

        self._jacobian = ti.field(dtype=gs.ti_float, shape=(6, self.n_dofs, self._solver._B))

        # compute joint limit in q space
        q_limit_lower = []
        q_limit_upper = []
        for joint in self.joints:
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
        self.q_limit = np.stack(
            (np.concatenate(q_limit_lower), np.concatenate(q_limit_upper)), axis=0, dtype=gs.np_float
        )

        # for storing intermediate results
        self._IK_n_tgts = self._solver._options.IK_max_targets
        self._IK_error_dim = self._IK_n_tgts * 6
        self._IK_mat = ti.field(dtype=gs.ti_float, shape=(self._IK_error_dim, self._IK_error_dim, self._solver._B))
        self._IK_inv = ti.field(dtype=gs.ti_float, shape=(self._IK_error_dim, self._IK_error_dim, self._solver._B))
        self._IK_L = ti.field(dtype=gs.ti_float, shape=(self._IK_error_dim, self._IK_error_dim, self._solver._B))
        self._IK_U = ti.field(dtype=gs.ti_float, shape=(self._IK_error_dim, self._IK_error_dim, self._solver._B))
        self._IK_y = ti.field(dtype=gs.ti_float, shape=(self._IK_error_dim, self._IK_error_dim, self._solver._B))
        self._IK_qpos_orig = ti.field(dtype=gs.ti_float, shape=(self.n_qs, self._solver._B))
        self._IK_qpos_best = ti.field(dtype=gs.ti_float, shape=(self.n_qs, self._solver._B))
        self._IK_delta_qpos = ti.field(dtype=gs.ti_float, shape=(self.n_dofs, self._solver._B))
        self._IK_vec = ti.field(dtype=gs.ti_float, shape=(self._IK_error_dim, self._solver._B))
        self._IK_err_pose = ti.field(dtype=gs.ti_float, shape=(self._IK_error_dim, self._solver._B))
        self._IK_err_pose_best = ti.field(dtype=gs.ti_float, shape=(self._IK_error_dim, self._solver._B))
        self._IK_jacobian = ti.field(dtype=gs.ti_float, shape=(self._IK_error_dim, self.n_dofs, self._solver._B))
        self._IK_jacobian_T = ti.field(dtype=gs.ti_float, shape=(self.n_dofs, self._IK_error_dim, self._solver._B))

    def _add_by_info(self, l_info, j_infos, g_infos, morph, surface):
        if len(j_infos) > 1 and any(j_info["type"] in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.FIXED) for j_info in j_infos):
            raise ValueError(
                "Compounding joints of types 'FREE' or 'FIXED' with any other joint on the same body not supported"
            )

        if isinstance(morph, gs.options.morphs.FileMorph) and morph.recompute_inertia:
            l_info.update(inertial_pos=None, inertial_quat=None, inertial_i=None, inertial_mass=None)

        parent_idx = l_info["parent_idx"]
        if parent_idx >= 0:
            parent_idx += self._link_start
        root_idx = l_info.get("root_idx")
        if root_idx is not None and root_idx >= 0:
            root_idx += self._link_start
        link_idx = self.n_links + self._link_start
        joint_start = self.n_joints + self._joint_start
        free_verts_start, fixed_verts_start = self._free_verts_state_start, self._fixed_verts_state_start
        for link in self.links:
            if link.is_fixed and not self._batch_fixed_verts:
                fixed_verts_start += link.n_verts
            else:
                free_verts_start += link.n_verts

        # Add parent joints
        joints = gs.List()
        self._joints.append(joints)
        for i_j_, j_info in enumerate(j_infos):
            n_dofs = j_info["n_dofs"]

            sol_params = np.array(j_info.get("sol_params", gu.default_solver_params()), copy=True)
            if (
                len(sol_params.shape) == 2
                and sol_params.shape[0] == 1
                and (sol_params[0][3] >= 1.0 or sol_params[0][2] >= sol_params[0][3])
            ):
                gs.logger.warning(
                    f"Joint {j_info['name']}'s sol_params {sol_params[0]} look not right, change to default."
                )
                sol_params = gu.default_solver_params()

            dofs_motion_ang = j_info.get("dofs_motion_ang")
            if dofs_motion_ang is None:
                if n_dofs == 6:
                    dofs_motion_ang = np.eye(6, 3, -3)
                elif n_dofs == 0:
                    dofs_motion_ang = np.zeros((0, 3))
                else:
                    assert False

            dofs_motion_vel = j_info.get("dofs_motion_vel")
            if dofs_motion_vel is None:
                if n_dofs == 6:
                    dofs_motion_vel = np.eye(6, 3)
                elif n_dofs == 0:
                    dofs_motion_vel = np.zeros((0, 3))
                else:
                    assert False

            joint = RigidJoint(
                entity=self,
                name=j_info["name"],
                idx=joint_start + i_j_,
                link_idx=link_idx,
                q_start=self.n_qs + self._q_start,
                dof_start=self.n_dofs + self._dof_start,
                n_qs=j_info["n_qs"],
                n_dofs=n_dofs,
                type=j_info["type"],
                pos=j_info.get("pos", gu.zero_pos()),
                quat=j_info.get("quat", gu.identity_quat()),
                init_qpos=j_info.get("init_qpos", np.zeros(n_dofs)),
                sol_params=sol_params,
                dofs_motion_ang=dofs_motion_ang,
                dofs_motion_vel=dofs_motion_vel,
                dofs_limit=j_info.get("dofs_limit", np.tile([[-np.inf, np.inf]], [n_dofs, 1])),
                dofs_invweight=j_info.get("dofs_invweight", np.zeros(n_dofs)),
                dofs_frictionloss=j_info.get("dofs_frictionloss", np.zeros(n_dofs)),
                dofs_stiffness=j_info.get("dofs_stiffness", np.zeros(n_dofs)),
                dofs_damping=j_info.get("dofs_damping", np.zeros(n_dofs)),
                dofs_armature=j_info.get("dofs_armature", np.zeros(n_dofs)),
                dofs_kp=j_info.get("dofs_kp", np.zeros(n_dofs)),
                dofs_kv=j_info.get("dofs_kv", np.zeros(n_dofs)),
                dofs_force_range=j_info.get("dofs_force_range", np.tile([[-np.inf, np.inf]], [n_dofs, 1])),
            )
            joints.append(joint)

        # Add child link
        link = RigidLink(
            entity=self,
            name=l_info["name"],
            idx=link_idx,
            joint_start=joint_start,
            n_joints=len(j_infos),
            geom_start=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            free_verts_state_start=free_verts_start,
            fixed_verts_state_start=fixed_verts_start,
            vgeom_start=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            pos=l_info["pos"],
            quat=l_info["quat"],
            inertial_pos=l_info.get("inertial_pos"),
            inertial_quat=l_info.get("inertial_quat"),
            inertial_i=l_info.get("inertial_i"),
            inertial_mass=l_info.get("inertial_mass"),
            parent_idx=parent_idx,
            root_idx=root_idx,
            invweight=l_info.get("invweight"),
            visualize_contact=self.visualize_contact,
        )
        self._links.append(link)

        # Separate collision from visual geometry for post-processing
        cg_infos, vg_infos = [], []
        for g_info in g_infos:
            is_col = g_info["contype"] or g_info["conaffinity"]
            if morph.collision and is_col:
                cg_infos.append(g_info)
            if morph.visualization and not is_col:
                vg_infos.append(g_info)

        # Post-process all collision meshes at once.
        # Destroying the original geometries should be avoided if possible as it will change the way objects
        # interact with the world due to only computing one contact point per convex geometry. The idea is to
        # check if each geometry can be convexified independently without resorting on convex decomposition.
        # If so, the original geometries are preserve. If not, then they are all merged as one. Following the
        # same approach as before, the resulting geometry is convexify without resorting on convex decomposition
        # if possible. Mergeing before falling back directly to convex decompositio is important as it gives one
        # last chance to avoid it. Moreover, it tends to reduce the final number of collision geometries. In
        # both cases, this improves runtime performance, numerical stability and compilation time.
        if isinstance(morph, gs.options.morphs.FileMorph):
            # Choose the appropriate convex decomposition error threshold depending on whether the link at hand
            # is associated with a robot.
            # The rational behind it is that performing convex decomposition for robots is mostly useless because
            # the non-physical part that is added to the original geometries to convexify them are generally inside
            # the mechanical structure and not interacting directly with the outer world. On top of that, not only
            # iy increases the memory footprint and compilation time, but also the simulation speed (marginally).
            if l_info["is_robot"]:
                decompose_error_threshold = morph.decompose_robot_error_threshold
            else:
                decompose_error_threshold = morph.decompose_object_error_threshold

            cg_infos = mu.postprocess_collision_geoms(
                cg_infos,
                morph.decimate,
                morph.decimate_face_num,
                morph.decimate_aggressiveness,
                morph.convexify,
                decompose_error_threshold,
                morph.coacd_options,
            )

        # Randomize collision mesh colors. The is especially useful to check convex decomposition.
        for g_info in cg_infos:
            mesh = g_info["mesh"]
            mesh.set_color((*np.random.rand(3), 0.7))

        # Add visual geometries
        for g_info in vg_infos:
            link._add_vgeom(
                vmesh=g_info["vmesh"],
                init_pos=g_info.get("pos", gu.zero_pos()),
                init_quat=g_info.get("quat", gu.identity_quat()),
            )

        # Add collision geometries
        for g_info in cg_infos:
            friction = self.material.friction
            if friction is None:
                friction = g_info.get("friction", gu.default_friction())
            link._add_geom(
                mesh=g_info["mesh"],
                init_pos=g_info.get("pos", gu.zero_pos()),
                init_quat=g_info.get("quat", gu.identity_quat()),
                type=g_info["type"],
                friction=friction,
                sol_params=g_info["sol_params"],
                data=g_info.get("data"),
                needs_coup=self.material.needs_coup,
                contype=g_info["contype"],
                conaffinity=g_info["conaffinity"],
            )

        return link, joints

    @staticmethod
    def _convert_g_infos_to_cg_infos_and_vg_infos(morph, g_infos, is_robot):
        """
        Separate collision from visual geometry and post-process collision meshes.
        Used for both normal loading and heterogeneous simulation.
        """
        cg_infos, vg_infos = [], []
        for g_info in g_infos:
            is_col = g_info["contype"] or g_info["conaffinity"]
            if morph.collision and is_col:
                cg_infos.append(g_info)
            if morph.visualization and not is_col:
                vg_infos.append(g_info)

        # Post-process all collision meshes at once
        if isinstance(morph, gs.options.morphs.FileMorph):
            if is_robot:
                decompose_error_threshold = morph.decompose_robot_error_threshold
            else:
                decompose_error_threshold = morph.decompose_object_error_threshold

            cg_infos = mu.postprocess_collision_geoms(
                cg_infos,
                morph.decimate,
                morph.decimate_face_num,
                morph.decimate_aggressiveness,
                morph.convexify,
                decompose_error_threshold,
                morph.coacd_options,
            )

        # Randomize collision mesh colors
        for g_info in cg_infos:
            mesh = g_info["mesh"]
            mesh.set_color((*np.random.rand(3), 0.7))

        return cg_infos, vg_infos

    def _add_equality(self, name, type, objs_name, data, sol_params):
        objs_id = []
        for obj_name in objs_name:
            if type == gs.EQUALITY_TYPE.CONNECT:
                obj_id = self.get_link(obj_name).idx
            elif type == gs.EQUALITY_TYPE.JOINT:
                obj_id = self.get_joint(obj_name).idx
            elif type == gs.EQUALITY_TYPE.WELD:
                obj_id = self.get_link(obj_name).idx
            else:
                gs.logger.warning(f"Equality type {type} not supported. Only CONNECT, JOINT, and WELD are supported.")
            objs_id.append(obj_id)

        equality = RigidEquality(
            entity=self,
            name=name,
            idx=self.n_equalities + self._equality_start,
            type=type,
            eq_obj1id=objs_id[0],
            eq_obj2id=objs_id[1],
            eq_data=data,
            sol_params=sol_params,
        )
        self._equalities.append(equality)
        return equality

    @gs.assert_unbuilt
    def attach(self, parent_entity, parent_link_name: str | None = None):
        """
        Merge two entities to act as single one, by attaching the base link of this entity as a child of a given link of
        another entity.

        Parameters
        ----------
        parent_entity : genesis.Entity
            The entity in the scene that will be a parent of kinematic tree.
        parent_link_name : str
            The name of the link in the parent entity to be linked. Default to the latest link the parent kinematic
            tree.
        """
        if self._is_attached:
            gs.raise_exception("Entity already attached.")

        if not isinstance(parent_entity, RigidEntity):
            gs.raise_exception("Parent entity must derive from 'RigidEntity'.")

        if parent_entity is self:
            gs.raise_exception("Cannot attach entity to itself.")

        if parent_entity.idx > self.idx:
            gs.raise_exception("Parent entity must be instantiated before child entity.")

        # Check if base link was fixed but no longer is
        base_link = self.links[0]
        parent_link = parent_entity.get_link(parent_link_name)
        if base_link.is_fixed and not parent_link.is_fixed:
            if not self._batch_fixed_verts:
                gs.raise_exception(
                    "Attaching fixed-based entity to parent link requires setting Morph option 'batch_fixed_verts=True'."
                )

        # Remove all root joints if necessary.
        # The requires shifting joint and dof indices of all subsequent entities.
        # Note that we do not remove world link if any, but rather remove all base joints. This is to avoid altering
        # the parent entity by moving all fixed geometries to the new parent link.
        if not base_link.is_fixed:
            n_base_joints = base_link.n_joints
            n_base_dofs = base_link.n_dofs
            n_base_qs = base_link.n_qs

            base_link._n_joints = 0
            self._joints[0].clear()
            for entity in self._solver.entities[(self.idx + 1) :]:
                entity._joint_start -= n_base_joints
                entity._dof_start -= n_base_dofs
                entity._q_start -= n_base_qs
            for joint in self._solver.joints[self.joint_start :]:
                joint._dof_start -= n_base_dofs
                joint._q_start -= n_base_qs
            for link in self._solver.links[(self.link_start + 1) :]:
                link._joint_start -= n_base_joints

        # Overwrite parent link
        base_link._parent_idx = parent_link.idx

        for link in self.links:
            # Break as soon as the root idx is -1, because the following links correspond to a different kinematic tree
            if link.root_idx == -1:
                break

            # Override root idx for child links
            assert link.root_idx == base_link.idx
            link._root_idx = parent_link.root_idx

            # Update fixed link flag
            link._is_fixed &= parent_link.is_fixed

            # Must invalidate invweight for all child links and joints
            link._invweight = None

        self._is_attached = True

    # ------------------------------------------------------------------------------------
    # ---------------------------------- control & io ------------------------------------
    # ------------------------------------------------------------------------------------
    def process_input(self, in_backward=False):
        if in_backward:
            # use negative index because buffer length might not be full
            index = self._sim.cur_step_local - self._sim._steps_local
            self._tgt = self._tgt_buffer[index].copy()
        else:
            self._tgt_buffer.append(self._tgt.copy())

        update_tgt_while_set = self._update_tgt_while_set
        # Apply targets in the order of insertion
        for key in self._tgt.keys():
            data_kwargs = self._tgt[key]

            # We do not need zero velocity here because if it was true, [set_dofs_velocity] from zero_velocity would
            # be in [tgt]
            if "zero_velocity" in data_kwargs:
                data_kwargs["zero_velocity"] = False
            # Do not update [tgt], as input information is finalized at this point
            self._update_tgt_while_set = False

            match key:
                case "set_pos":
                    self.set_pos(**data_kwargs)
                case "set_quat":
                    self.set_quat(**data_kwargs)
                case "set_dofs_velocity":
                    self.set_dofs_velocity(**data_kwargs)
                case _:
                    gs.raise_exception(f"Invalid target key: {key} not in {self._tgt_keys}")

        self._tgt = dict()
        self._update_tgt_while_set = update_tgt_while_set

    def process_input_grad(self):
        index = self._sim.cur_step_local - self._sim._steps_local
        for key in reversed(self._tgt_buffer[index].keys()):
            data_kwargs = self._tgt_buffer[index][key]

            match key:
                # We need to unpack the data_kwargs because [_backward_from_ti] only supports positional arguments
                case "set_pos":
                    pos = data_kwargs.pop("pos")
                    if pos.requires_grad:
                        pos._backward_from_ti(self.set_pos_grad, data_kwargs["envs_idx"], data_kwargs["relative"])

                case "set_quat":
                    quat = data_kwargs.pop("quat")
                    if quat.requires_grad:
                        quat._backward_from_ti(self.set_quat_grad, data_kwargs["envs_idx"], data_kwargs["relative"])

                case "set_dofs_velocity":
                    velocity = data_kwargs.pop("velocity")
                    # [velocity] could be None when we want to zero the velocity (see set_dofs_velocity of RigidSolver)
                    if velocity is not None and velocity.requires_grad:
                        velocity._backward_from_ti(
                            self.set_dofs_velocity_grad,
                            data_kwargs["dofs_idx_local"],
                            data_kwargs["envs_idx"],
                        )
                case _:
                    gs.raise_exception(f"Invalid target key: {key} not in {self._tgt_keys}")

    def save_ckpt(self, ckpt_name):
        if ckpt_name not in self._ckpt:
            self._ckpt[ckpt_name] = {}
        self._ckpt[ckpt_name]["_tgt_buffer"] = self._tgt_buffer.copy()
        self._tgt_buffer.clear()

    def load_ckpt(self, ckpt_name):
        self._tgt_buffer = self._ckpt[ckpt_name]["_tgt_buffer"].copy()

    def reset_grad(self):
        self._tgt_buffer.clear()

    # Accessor methods (get_*, set_*, control_*) are provided by RigidEntityAccessorMixin

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def is_built(self):
        """
        Whether this rigid entity is built.
        """
        return self._is_built

    @property
    def is_attached(self):
        """
        Whether this rigid entity has already been attached to another one.
        """
        return self._is_attached

    @property
    def visualize_contact(self):
        """Whether to visualize contact force."""
        return self._visualize_contact

    @property
    def init_qpos(self):
        """The initial qpos of the entity."""
        if self.joints:
            return np.concatenate([joint.init_qpos for joint in self.joints])
        return np.array([])

    @property
    def n_qs(self):
        """The number of `q` (generalized coordinates) of the entity."""
        if self._is_built:
            return self._n_qs
        return sum(joint.n_qs for joint in self.joints)

    @property
    def n_links(self):
        """The number of `RigidLink` in the entity."""
        return len(self._links)

    @property
    def main_morph(self):
        """The main morph of the entity (first morph for heterogeneous entities)."""
        return self._morph

    @property
    def morphs(self):
        """All morphs of the entity (main morph + heterogeneous variants if any)."""
        return (self._morph, *self._morph_heterogeneous)

    @property
    def n_joints(self):
        """The number of `RigidJoint` in the entity."""
        return sum(map(len, self._joints))

    @property
    def n_dofs(self):
        """The number of degrees of freedom (DOFs) of the entity."""
        if self._is_built:
            return self._n_dofs
        return sum(joint.n_dofs for joint in self.joints)

    @property
    def n_geoms(self):
        """The number of `RigidGeom` in the entity."""
        if self._is_built:
            return self._n_geoms
        return sum(link.n_geoms for link in self._links)

    @property
    def n_cells(self):
        """The number of sdf cells in the entity."""
        return sum(link.n_cells for link in self._links)

    @property
    def n_verts(self):
        """The number of vertices (from collision geom `RigidGeom`) in the entity."""
        return sum(link.n_verts for link in self._links)

    @property
    def n_faces(self):
        """The number of faces (from collision geom `RigidGeom`) in the entity."""
        return sum(link.n_faces for link in self._links)

    @property
    def n_edges(self):
        """The number of edges (from collision geom `RigidGeom`) in the entity."""
        return sum(link.n_edges for link in self._links)

    @property
    def n_vgeoms(self):
        """The number of vgeoms (visual geoms - `RigidVisGeom`) in the entity."""
        return sum(link.n_vgeoms for link in self._links)

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
    def gravity_compensation(self):
        """Apply a force to compensate gravity. A value of 1 will make a zero-gravity behavior. Default to 0"""
        return self.material.gravity_compensation

    @property
    def link_start(self):
        """The index of the entity's first RigidLink in the scene."""
        return self._link_start

    @property
    def link_end(self):
        """The index of the entity's last RigidLink in the scene *plus one*."""
        return self._link_start + self.n_links

    @property
    def joint_start(self):
        """The index of the entity's first RigidJoint in the scene."""
        return self._joint_start

    @property
    def joint_end(self):
        """The index of the entity's last RigidJoint in the scene *plus one*."""
        return self._joint_start + self.n_joints

    @property
    def dof_start(self):
        """The index of the entity's first degree of freedom (DOF) in the scene."""
        return self._dof_start

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
    def geoms(self) -> list[RigidGeom]:
        """The list of collision geoms (`RigidGeom`) in the entity."""
        if self.is_built:
            return self._geoms
        return gs.List(geom for link in self._links for geom in link.geoms)

    @property
    def vgeoms(self):
        """The list of visual geoms (`RigidVisGeom`) in the entity."""
        if self.is_built:
            return self._vgeoms
        return gs.List(vgeom for link in self._links for vgeom in link.vgeoms)

    @property
    def links(self) -> list[RigidLink]:
        """The list of links (`RigidLink`) in the entity."""
        return self._links

    @property
    def joints(self):
        """The list of joints (`RigidJoint`) in the entity."""
        return tuple(chain.from_iterable(self._joints))

    @property
    def joints_by_links(self):
        """The list of joints (`RigidJoint`) in the entity grouped by parent links."""
        return self._joints

    @property
    def base_link(self):
        """The base link of the entity"""
        return self._links[0]

    @property
    def base_joint(self):
        """The base joint of the entity"""
        return self._joints[0][0]

    @property
    def n_equalities(self):
        """The number of equality constraints in the entity."""
        return len(self._equalities)

    @property
    def equality_start(self):
        """The index of the entity's first RigidEquality in the scene."""
        return self._equality_start

    @property
    def equality_end(self):
        """The index of the entity's last RigidEquality in the scene *plus one*."""
        return self._equality_start + self.n_equalities

    @property
    def equalities(self):
        """The list of equality constraints (`RigidEquality`) in the entity."""
        return self._equalities

    @property
    def is_free(self) -> bool:
        raise DeprecationError("This property has been removed.")

    @property
    def is_local_collision_mask(self):
        """Whether the contype and conaffinity bitmasks of this entity only applies to self-collision."""
        return self._is_local_collision_mask


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_get_free_verts(
    tensor: ti.types.ndarray(),
    free_verts_idx_local: ti.types.ndarray(),
    verts_state_start: ti.i32,
    free_verts_state: array_class.VertsState,
):
    n_verts = free_verts_idx_local.shape[0]
    _B = tensor.shape[0]
    for i_v_, i, i_b in ti.ndrange(n_verts, 3, _B):
        i_v = i_v_ + verts_state_start
        tensor[i_b, free_verts_idx_local[i_v_], i] = free_verts_state.pos[i_v, i_b][i]


@ti.kernel(fastcache=gs.use_fastcache)
def _kernel_get_fixed_verts(
    tensor: ti.types.ndarray(),
    fixed_verts_idx_local: ti.types.ndarray(),
    verts_state_start: ti.i32,
    fixed_verts_state: array_class.VertsState,
):
    n_verts = fixed_verts_idx_local.shape[0]
    _B = tensor.shape[0]
    for i_v_, i, i_b in ti.ndrange(n_verts, 3, _B):
        i_v = i_v_ + verts_state_start
        tensor[i_b, fixed_verts_idx_local[i_v_], i] = fixed_verts_state.pos[i_v][i]


# FIXME: RigidEntity is not compatible with fast cache
