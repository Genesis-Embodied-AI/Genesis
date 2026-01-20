import inspect
from copy import copy
from itertools import chain
from typing import TYPE_CHECKING, Literal, Any
from functools import wraps

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
from genesis.utils.misc import DeprecationError, broadcast_tensor, ti_to_torch
from genesis.engine.states.entities import RigidEntityState

from ..base_entity import Entity
from .rigid_equality import RigidEquality
from .rigid_geom import RigidGeom
from .rigid_joint import RigidJoint
from .rigid_link import RigidLink
from .impl import (
    RigidEntityLoaderMixin,
    RigidEntityKinematicsMixin,
    compute_inertial_from_geom_infos,
    kernel_rigid_entity_inverse_kinematics,
)

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver


# Wrapper to track the arguments of a function and save them in the target buffer
def tracked(fun):
    sig = inspect.signature(fun)

    @wraps(fun)
    def wrapper(self, *args, **kwargs):
        if self._update_tgt_while_set:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            args_dict = dict(tuple(bound.arguments.items())[1:])
            self._update_tgt(fun.__name__, args_dict)
        return fun(self, *args, **kwargs)

    return wrapper


@ti.data_oriented
class RigidEntity(RigidEntityLoaderMixin, RigidEntityKinematicsMixin, Entity):
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

    @gs.assert_built
    def get_state(self):
        state = RigidEntityState(self, self._sim.cur_step_global)

        solver_state = self._solver.get_state()
        pos = solver_state.links_pos[:, self.base_link_idx]
        quat = solver_state.links_quat[:, self.base_link_idx]

        state._pos = pos
        state._quat = quat

        return state

    def _get_global_idx(self, idx_local, idx_local_max, idx_global_start=0, *, unsafe=False):
        # Handling default argument and special cases
        if idx_local is None:
            idx_global = range(idx_global_start, idx_local_max + idx_global_start)
        elif isinstance(idx_local, (slice, range)):
            idx_global = range(
                (idx_local.start or 0) + idx_global_start,
                (idx_local.stop if idx_local.stop is not None else idx_local_max) + idx_global_start,
                idx_local.step or 1,
            )
        elif isinstance(idx_local, (int, np.integer)):
            idx_global = (idx_local + idx_global_start,)
        elif isinstance(idx_local, (list, tuple)):
            try:
                idx_global = [i + idx_global_start for i in idx_local]
            except TypeError:
                gs.raise_exception("Expecting a sequence of integers for `idx_local`.")
        else:
            # Increment may be slow when dealing with heterogenuous data, so it must be avoided if possible
            if idx_global_start > 0:
                idx_global = idx_local + idx_global_start
            else:
                idx_global = idx_local

        # Early return if unsafe
        if unsafe:
            return idx_global

        # Perform a bunch of sanity checks
        if isinstance(idx_global, torch.Tensor) and idx_global.dtype == torch.bool:
            if idx_global.shape != (idx_local_max - idx_global_start,):
                gs.raise_exception("Boolean masks must be 1D tensors of fixed size.")
            idx_global = idx_global_start + idx_global.nonzero()[:, 0]
        else:
            idx_global = torch.as_tensor(idx_global, dtype=gs.tc_int, device=gs.device).contiguous()
            ndim = idx_global.ndim
            if ndim == 0:
                idx_global = idx_global[None]
            elif ndim > 1:
                gs.raise_exception("Expecting a 1D tensor for local index.")

            # FIXME: This check is too expensive
            # if (idx_global < 0).any() or (idx_global >= idx_global_start + idx_local_max).any():
            #     gs.raise_exception("`idx_local` exceeds valid range.")

        return idx_global

    def get_joint(self, name=None, uid=None):
        """
        Get a RigidJoint object by name or uid.

        Parameters
        ----------
        name : str, optional
            The name of the joint. Defaults to None.
        uid : str, optional
            The uid of the joint. This can be a substring of the joint's uid. Defaults to None.

        Returns
        -------
        joint : RigidJoint
            The joint object.
        """

        if name is not None:
            for joint in self.joints:
                if joint.name == name:
                    return joint
            gs.raise_exception(f"Joint not found for name: {name}.")

        elif uid is not None:
            for joint in self.joints:
                if uid in str(joint.uid):
                    return joint
            gs.raise_exception(f"Joint not found for uid: {uid}.")

        else:
            gs.raise_exception("Neither `name` nor `uid` is provided.")

    def get_link(self, name=None, uid=None):
        """
        Get a RigidLink object by name or uid.

        Parameters
        ----------
        name : str, optional
            The name of the link. Defaults to None.
        uid : str, optional
            The uid of the link. This can be a substring of the link's uid. Defaults to None.

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

        elif uid is not None:
            for link in self._links:
                if uid in str(link.uid):
                    return link
            gs.raise_exception(f"Link not found for uid: {uid}.")

        else:
            gs.raise_exception("Neither `name` nor `uid` is provided.")

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
        return self._solver.get_links_pos(self.base_link_idx, envs_idx)[..., 0, :]

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
        return self._solver.get_links_quat(self.base_link_idx, envs_idx)[..., 0, :]

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
        return self._solver.get_links_vel(self.base_link_idx, envs_idx)[..., 0, :]

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
        return self._solver.get_links_ang(self.base_link_idx, envs_idx)[..., 0, :]

    @gs.assert_built
    def get_links_pos(
        self,
        links_idx_local=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com", "root_com"] = "link_origin",
        unsafe=False,
    ):
        """
        Returns the position of a given reference point for all the entity's links.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        ref: "link_origin" | "link_com" | "root_com"
            The reference point being used to express the position of each link.
            * "root_com": center of mass of the sub-entities to which the link belongs. As a reminder, a single
              kinematic tree (aka. 'RigidEntity') may compromise multiple "physical" entities, i.e. a kinematic tree
              that may have at most one free joint, at its root.

        Returns
        -------
        pos : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The position of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_pos(links_idx, envs_idx, ref=ref)

    @gs.assert_built
    def get_links_quat(self, links_idx_local=None, envs_idx=None):
        """
        Returns quaternion of all the entity's links.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        quat : torch.Tensor, shape (n_links, 4) or (n_envs, n_links, 4)
            The quaternion of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_quat(links_idx, envs_idx)

    @gs.assert_built
    def get_AABB(self, envs_idx=None, *, allow_fast_approx: bool = False):
        """
        Get the axis-aligned bounding box (AABB) of the entity in world frame by aggregating all the collision
        geometries associated with this entity.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        allow_fast_approx : bool
            Whether to allow fast approximation for efficiency if supported, i.e. 'LegacyCoupler' is enabled. In this
            case, each collision geometry is approximated by their pre-computed AABB in geometry-local frame, which is
            more efficiency but inaccurate.

        Returns
        -------
        aabb : torch.Tensor, shape (2, 3) or (n_envs, 2, 3)
            The AABB of the entity, where `[:, 0] = min_corner (x_min, y_min, z_min)` and
            `[:, 1] = max_corner (x_max, y_max, z_max)`.
        """
        from genesis.engine.couplers import LegacyCoupler

        if self.n_geoms == 0:
            gs.raise_exception("Entity has no collision geometries.")

        # Already computed internally by the solver. Let's access it directly for efficiency.
        if allow_fast_approx and isinstance(self.sim.coupler, LegacyCoupler):
            return self._solver.get_AABB(entities_idx=[self._idx_in_solver], envs_idx=envs_idx)[..., 0, :]

        # For heterogeneous entities, compute AABB per-environment respecting active_envs_idx.
        # FIXME: Remove this branch after implementing 'get_verts'.
        if self._enable_heterogeneous and self._solver.n_envs > 0:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            n_envs = len(envs_idx)
            aabb_min = torch.full((n_envs, 3), float("inf"), dtype=gs.tc_float, device=gs.device)
            aabb_max = torch.full((n_envs, 3), float("-inf"), dtype=gs.tc_float, device=gs.device)
            for geom in self.geoms:
                geom_aabb = geom.get_AABB()
                active_mask = geom.active_envs_mask[envs_idx] if geom.active_envs_mask is not None else ()
                aabb_min[active_mask] = torch.minimum(aabb_min[active_mask], geom_aabb[envs_idx[active_mask], 0])
                aabb_max[active_mask] = torch.maximum(aabb_max[active_mask], geom_aabb[envs_idx[active_mask], 1])
            return torch.stack((aabb_min, aabb_max), dim=-2)

        # Compute the AABB on-the-fly based on the positions of all the vertices
        verts = self.get_verts()[envs_idx if envs_idx is not None else ()]
        return torch.stack((verts.min(dim=-2).values, verts.max(dim=-2).values), dim=-2)

    @gs.assert_built
    def get_vAABB(self, envs_idx=None):
        """
        Get the axis-aligned bounding box (AABB) of the entity in world frame by aggregating all the visual
        geometries associated with this entity.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        aabb : torch.Tensor, shape (2, 3) or (n_envs, 2, 3)
            The AABB of the entity, where `[:, 0] = min_corner (x_min, y_min, z_min)` and
            `[:, 1] = max_corner (x_max, y_max, z_max)`.
        """
        if self.n_vgeoms == 0:
            gs.raise_exception("Entity has no visual geometries.")

        # For heterogeneous entities, compute AABB per-environment respecting active_envs_idx
        if self._enable_heterogeneous:
            envs_idx = self._scene._sanitize_envs_idx(envs_idx)
            n_envs = len(envs_idx)
            aabb_min = torch.full((n_envs, 3), float("inf"), dtype=gs.tc_float, device=gs.device)
            aabb_max = torch.full((n_envs, 3), float("-inf"), dtype=gs.tc_float, device=gs.device)
            for vgeom in self.vgeoms:
                vgeom_aabb = vgeom.get_vAABB(envs_idx)
                active_mask = vgeom.active_envs_mask[envs_idx] if vgeom.active_envs_mask is not None else ()
                aabb_min[active_mask] = torch.minimum(aabb_min[active_mask], vgeom_aabb[active_mask, 0])
                aabb_max[active_mask] = torch.maximum(aabb_max[active_mask], vgeom_aabb[active_mask, 1])
            return torch.stack((aabb_min, aabb_max), dim=-2)

        aabbs = torch.stack([vgeom.get_vAABB(envs_idx) for vgeom in self._vgeoms], dim=-3)
        return torch.stack((aabbs[..., 0, :].min(dim=-2).values, aabbs[..., 1, :].max(dim=-2).values), dim=-2)

    def get_aabb(self):
        raise DeprecationError("This method has been removed. Please use 'get_AABB()' instead.")

    @gs.assert_built
    def get_links_vel(
        self,
        links_idx_local=None,
        envs_idx=None,
        *,
        ref: Literal["link_origin", "link_com"] = "link_origin",
        unsafe=False,
    ):
        """
        Returns linear velocity of all the entity's links expressed at a given reference position in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        ref: "link_origin" | "link_com"
            The reference point being used to expressed the velocity of each link.

        Returns
        -------
        vel : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear velocity of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_vel(links_idx, envs_idx, ref=ref)

    @gs.assert_built
    def get_links_ang(self, links_idx_local=None, envs_idx=None):
        """
        Returns angular velocity of all the entity's links in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        ang : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The angular velocity of all the entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_ang(links_idx, envs_idx)

    @gs.assert_built
    def get_links_acc(self, links_idx_local=None, envs_idx=None):
        """
        Returns true linear acceleration (aka. "classical acceleration") of the specified entity's links expressed at
        their respective origin in world coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        acc : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear classical acceleration of the specified entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_acc(links_idx, envs_idx)

    @gs.assert_built
    def get_links_acc_ang(self, links_idx_local=None, envs_idx=None):
        """
        Returns angular acceleration of the specified entity's links expressed at their respective origin in world
        coordinates.

        Parameters
        ----------
        links_idx_local : None | array_like
            The indices of the links. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        acc : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The linear classical acceleration of the specified entity's links.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_acc_ang(links_idx, envs_idx)

    @gs.assert_built
    def get_links_inertial_mass(self, links_idx_local=None, envs_idx=None):
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_inertial_mass(links_idx, envs_idx)

    @gs.assert_built
    def get_links_invweight(self, links_idx_local=None, envs_idx=None):
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        return self._solver.get_links_invweight(links_idx, envs_idx)

    @gs.assert_built
    @tracked
    def set_pos(self, pos, envs_idx=None, *, zero_velocity=True, relative=False):
        """
        Set position of the entity's base link.

        Parameters
        ----------
        pos : array_like
            The position to set.
        relative : bool, optional
            Whether the position to set is absolute or relative to the initial (not current!) position. Defaults to
            False.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        # Throw exception in entity no longer has a "true" base link becaused it has attached
        if self._is_attached:
            gs.raise_exception("Impossible to set position of an entity that has been attached.")
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_base_links_pos(pos, self.base_link_idx, envs_idx, relative=relative)

    @gs.assert_built
    def set_pos_grad(self, envs_idx, relative, pos_grad):
        self._solver.set_base_links_pos_grad(self.base_link_idx, envs_idx, relative, pos_grad.data)

    @gs.assert_built
    @tracked
    def set_quat(self, quat, envs_idx=None, *, zero_velocity=True, relative=False):
        """
        Set quaternion of the entity's base link.

        Parameters
        ----------
        quat : array_like
            The quaternion to set.
        relative : bool, optional
            Whether the quaternion to set is absolute or relative to the initial (not current!) quaternion. Defaults to
            False.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a
            sudden change in entity pose.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        if self._is_attached:
            gs.raise_exception("Impossible to set position of an entity that has been attached.")
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_base_links_quat(quat, self.base_link_idx, envs_idx, relative=relative)

    @gs.assert_built
    def set_quat_grad(self, envs_idx, relative, quat_grad):
        self._solver.set_base_links_quat_grad(self.base_link_idx, envs_idx, relative, quat_grad.data)

    @gs.assert_built
    def get_verts(self):
        """
        Get the all vertices of the entity based on collision geometries.

        Returns
        -------
        verts : torch.Tensor, shape (n_envs, n_verts, 3)
            The vertices of the entity.
        """
        if self._enable_heterogeneous:
            gs.raise_exception("This method is not supported for heterogeneous entities.")

        self._solver.update_verts_for_geoms(slice(self.geom_start, self.geom_end))

        n_fixed_verts, n_free_vertices = self._n_fixed_verts, self._n_free_verts
        tensor = torch.empty((self._solver._B, n_fixed_verts + n_free_vertices, 3), dtype=gs.tc_float, device=gs.device)

        if n_fixed_verts > 0:
            verts_idx = slice(self._fixed_verts_state_start, self._fixed_verts_state_start + n_fixed_verts)
            fixed_verts_state = ti_to_torch(self._solver.fixed_verts_state.pos, verts_idx)
            tensor[:, self._fixed_verts_idx_local] = fixed_verts_state
        if n_free_vertices > 0:
            verts_idx = slice(self._free_verts_state_start, self._free_verts_state_start + n_free_vertices)
            free_verts_state = ti_to_torch(self._solver.free_verts_state.pos, None, verts_idx, transpose=True)
            tensor[:, self._free_verts_idx_local] = free_verts_state

        if self._solver.n_envs == 0:
            tensor = tensor[0]
        return tensor

    @gs.assert_built
    def set_qpos(self, qpos, qs_idx_local=None, envs_idx=None, *, zero_velocity=True, skip_forward=False):
        """
        Set the entity's qpos.

        Parameters
        ----------
        qpos : array_like
            The qpos to set.
        qs_idx_local : None | array_like, optional
            The indices of the qpos to set. If None, all qpos will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a sudden change in entity pose.
        """
        qs_idx = self._get_global_idx(qs_idx_local, self.n_qs, self._q_start, unsafe=True)
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_qpos(qpos, qs_idx, envs_idx, skip_forward=skip_forward)

    @gs.assert_built
    def set_dofs_kp(self, kp, dofs_idx_local=None, envs_idx=None):
        """
        Set the entity's dofs' positional gains for the PD controller.

        Parameters
        ----------
        kp : array_like
            The positional gains to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_kp(kp, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_kv(self, kv, dofs_idx_local=None, envs_idx=None):
        """
        Set the entity's dofs' velocity gains for the PD controller.

        Parameters
        ----------
        kv : array_like
            The velocity gains to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_kv(kv, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_force_range(self, lower, upper, dofs_idx_local=None, envs_idx=None):
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
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_force_range(lower, upper, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_stiffness(self, stiffness, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_stiffness(stiffness, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_invweight(self, invweight, dofs_idx_local=None, envs_idx=None):
        raise DeprecationError(
            "This method has been removed because dof invweights are supposed to be a by-product of link properties "
            "(mass, pose, and inertia matrix), joint placements, and dof armatures. Please consider using the "
            "considering setters instead."
        )

    @gs.assert_built
    def set_dofs_armature(self, armature, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_armature(armature, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_damping(self, damping, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_damping(damping, dofs_idx, envs_idx)

    @gs.assert_built
    def set_dofs_frictionloss(self, frictionloss, dofs_idx_local=None, envs_idx=None):
        """
        Set the entity's dofs' friction loss.
        Parameters
        ----------
        frictionloss : array_like
            The friction loss values to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_frictionloss(frictionloss, dofs_idx, envs_idx)

    @gs.assert_built
    @tracked
    def set_dofs_velocity(self, velocity=None, dofs_idx_local=None, envs_idx=None, *, skip_forward=False):
        """
        Set the entity's dofs' velocity.

        Parameters
        ----------
        velocity : array_like | None
            The velocity to set. Zero if not specified.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_velocity(velocity, dofs_idx, envs_idx, skip_forward=skip_forward)

    @gs.assert_built
    def set_dofs_velocity_grad(self, dofs_idx_local, envs_idx, velocity_grad):
        dofs_idx = self._get_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.set_dofs_velocity_grad(dofs_idx, envs_idx, velocity_grad.data)

    @gs.assert_built
    def set_dofs_position(self, position, dofs_idx_local=None, envs_idx=None, *, zero_velocity=True):
        """
        Set the entity's dofs' position.

        Parameters
        ----------
        position : array_like
            The position to set.
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to set. If None, all dofs will be set. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        zero_velocity : bool, optional
            Whether to zero the velocity of all the entity's dofs. Defaults to True. This is a safety measure after a sudden change in entity pose.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        if zero_velocity:
            self.zero_all_dofs_velocity(envs_idx=envs_idx, skip_forward=True)
        self._solver.set_dofs_position(position, dofs_idx, envs_idx)

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
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_force(force, dofs_idx, envs_idx)

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
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_velocity(velocity, dofs_idx, envs_idx)

    @gs.assert_built
    def control_dofs_position(self, position, dofs_idx_local=None, envs_idx=None):
        """
        Set the position controller's target position for the entity's dofs. The controller is a proportional term
        plus a velocity damping term (virtual friction).

        Parameters
        ----------
        position : array_like
            The target position to set.
        dofs_idx_local : array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_position(position, dofs_idx, envs_idx)

    @gs.assert_built
    def control_dofs_position_velocity(self, position, velocity, dofs_idx_local=None, envs_idx=None):
        """
        Set a PD controller's target position and velocity for the entity's dofs. This is used for position control.

        Parameters
        ----------
        position : array_like
            The target position to set.
        velocity : array_like
            The target velocity
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to control. If None, all dofs will be controlled. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        self._solver.control_dofs_position_velocity(position, velocity, dofs_idx, envs_idx)

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
        qs_idx = self._get_global_idx(qs_idx_local, self.n_qs, self._q_start, unsafe=True)
        return self._solver.get_qpos(qs_idx, envs_idx)

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
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_control_force(dofs_idx, envs_idx)

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
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_force(dofs_idx, envs_idx)

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
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_velocity(dofs_idx, envs_idx)

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
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_position(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_kp(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the positional gain (kp) for the entity's dofs used by the PD controller.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        kp : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The positional gain (kp) for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_kp(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_kv(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the velocity gain (kv) for the entity's dofs used by the PD controller.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        kv : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The velocity gain (kv) for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_kv(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_force_range(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the force range (min and max limits) for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The lower limit of the force range for the entity's dofs.
        upper_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The upper limit of the force range for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_force_range(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_limit(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the positional limits (min and max) for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        lower_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The lower limit of the positional limits for the entity's dofs.
        upper_limit : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The upper limit of the positional limits for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_limit(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_stiffness(self, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_stiffness(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_invweight(self, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_invweight(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_armature(self, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_armature(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_damping(self, dofs_idx_local=None, envs_idx=None):
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_damping(dofs_idx, envs_idx)

    @gs.assert_built
    def get_dofs_frictionloss(self, dofs_idx_local=None, envs_idx=None):
        """
        Get the friction loss for the entity's dofs.

        Parameters
        ----------
        dofs_idx_local : None | array_like, optional
            The indices of the dofs to get. If None, all dofs will be returned. Note that here this uses the local `q_idx`, not the scene-level one. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.

        Returns
        -------
        frictionloss : torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
            The friction loss for the entity's dofs.
        """
        dofs_idx = self._get_global_idx(dofs_idx_local, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_dofs_frictionloss(dofs_idx, envs_idx)

    @gs.assert_built
    def get_mass_mat(self, envs_idx=None, decompose=False):
        dofs_idx = self._get_global_idx(None, self.n_dofs, self._dof_start, unsafe=True)
        return self._solver.get_mass_mat(dofs_idx, envs_idx, decompose)

    @gs.assert_built
    def zero_all_dofs_velocity(self, envs_idx=None, *, skip_forward=False):
        """
        Zero the velocity of all the entity's dofs.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        self.set_dofs_velocity(None, slice(0, self._n_dofs), envs_idx, skip_forward=skip_forward)

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
    def get_contacts(self, with_entity=None, exclude_self_contact=False):
        """
        Returns contact information computed during the most recent `scene.step()`.
        If `with_entity` is provided, only returns contact information involving the caller and the specified entity.
        Otherwise, returns all contact information involving the caller entity.
        When `with_entity` is `self`, it will return the self-collision only.

        The returned dict contains the following keys (a contact pair consists of two geoms: A and B):

        - 'geom_a'     : The global geom index of geom A in the contact pair.
                        (actual geom object can be obtained by scene.rigid_solver.geoms[geom_a])
        - 'geom_b'     : The global geom index of geom B in the contact pair.
                        (actual geom object can be obtained by scene.rigid_solver.geoms[geom_b])
        - 'link_a'     : The global link index of link A (that contains geom A) in the contact pair.
                        (actual link object can be obtained by scene.rigid_solver.links[link_a])
        - 'link_b'     : The global link index of link B (that contains geom B) in the contact pair.
                        (actual link object can be obtained by scene.rigid_solver.links[link_b])
        - 'position'   : The contact position in world frame.
        - 'force_a'    : The contact force applied to geom A.
        - 'force_b'    : The contact force applied to geom B.
        - 'valid_mask' : A boolean mask indicating whether the contact information is valid.
                        (Only when scene is parallelized)

        The shape of each entry is (n_envs, n_contacts, ...) for scene with parallel envs
                               and (n_contacts, ...) for non-parallelized scene.

        Parameters
        ----------
        with_entity : RigidEntity, optional
            The entity to check contact with. Defaults to None.
        exclude_self_contact: bool
            Exclude the self collision from the returning contacts. Defaults to False.

        Returns
        -------
        contact_info : dict
            The contact information.
        """
        contact_data = self._solver.collider.get_contacts(as_tensor=True, to_torch=True)

        logical_operation = torch.logical_xor if exclude_self_contact else torch.logical_or
        if with_entity is not None and self.idx == with_entity.idx:
            if exclude_self_contact:
                gs.raise_exception("`with_entity` is self but `exclude_self_contact` is True.")
            logical_operation = torch.logical_and

        valid_mask = logical_operation(
            torch.logical_and(
                contact_data["geom_a"] >= self.geom_start,
                contact_data["geom_a"] < self.geom_end,
            ),
            torch.logical_and(
                contact_data["geom_b"] >= self.geom_start,
                contact_data["geom_b"] < self.geom_end,
            ),
        )
        if with_entity is not None and self.idx != with_entity.idx:
            valid_mask = torch.logical_and(
                valid_mask,
                torch.logical_or(
                    torch.logical_and(
                        contact_data["geom_a"] >= with_entity.geom_start,
                        contact_data["geom_a"] < with_entity.geom_end,
                    ),
                    torch.logical_and(
                        contact_data["geom_b"] >= with_entity.geom_start,
                        contact_data["geom_b"] < with_entity.geom_end,
                    ),
                ),
            )

        if self._solver.n_envs == 0:
            contact_data = {key: value[valid_mask] for key, value in contact_data.items()}
        else:
            contact_data["valid_mask"] = valid_mask

        contact_data["force_a"] = -contact_data["force"]
        contact_data["force_b"] = +contact_data["force"]
        del contact_data["force"]

        return contact_data

    def get_links_net_contact_force(self, envs_idx=None):
        """
        Returns net force applied on each links due to direct external contacts.

        Returns
        -------
        entity_links_force : torch.Tensor, shape (n_links, 3) or (n_envs, n_links, 3)
            The net force applied on each links due to direct external contacts.
        """
        links_idx = slice(self.link_start, self.link_end)
        tensor = ti_to_torch(self._solver.links_state.contact_force, envs_idx, links_idx, transpose=True, copy=True)
        return tensor[0] if self._solver.n_envs == 0 else tensor

    def set_friction_ratio(self, friction_ratio, links_idx_local=None, envs_idx=None):
        """
        Set the friction ratio of the geoms of the specified links.

        Parameters
        ----------
        friction_ratio : torch.Tensor, shape (n_envs, n_links)
            The friction ratio
        links_idx_local : array_like
            The indices of the links to set friction ratio.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx_local = self._get_global_idx(links_idx_local, self.n_links, 0, unsafe=True)

        links_n_geoms = torch.tensor(
            [self._links[i_l].n_geoms for i_l in links_idx_local], dtype=gs.tc_int, device=gs.device
        )
        links_friction_ratio = torch.as_tensor(friction_ratio, dtype=gs.tc_float, device=gs.device)
        geoms_friction_ratio = torch.repeat_interleave(links_friction_ratio, links_n_geoms, dim=-1)
        geoms_idx = [
            i_g for i_l in links_idx_local for i_g in range(self._links[i_l].geom_start, self._links[i_l].geom_end)
        ]

        self._solver.set_geoms_friction_ratio(geoms_friction_ratio, geoms_idx, envs_idx)

    def set_friction(self, friction):
        """
        Set the friction coefficient of all the links (and in turn, geometries) of the rigid entity.

        Note
        ----
        The friction coefficient associated with a pair of geometries in contact is defined as the maximum between
        their respective values, so one must be careful the set the friction coefficient properly for both of them.

        Warning
        -------
        The friction coefficient must be in range [1e-2, 5.0] for simulation stability.

        Parameters
        ----------
        friction : float
            The friction coefficient to set.
        """

        if friction < 1e-2 or friction > 5.0:
            gs.raise_exception("`friction` must be in the range [1e-2, 5.0] for simulation stability.")

        for link in self._links:
            link.set_friction(friction)

    def set_mass_shift(self, mass_shift, links_idx_local=None, envs_idx=None):
        """
        Set the mass shift of specified links.

        Parameters
        ----------
        mass : torch.Tensor, shape (n_envs, n_links)
            The mass shift
        links_idx_local : array_like
            The indices of the links to set mass shift.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_mass_shift(mass_shift, links_idx, envs_idx)

    def set_COM_shift(self, com_shift, links_idx_local=None, envs_idx=None):
        """
        Set the center of mass (COM) shift of specified links.

        Parameters
        ----------
        com : torch.Tensor, shape (n_envs, n_links, 3)
            The COM shift
        links_idx_local : array_like
            The indices of the links to set COM shift.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        """
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_COM_shift(com_shift, links_idx, envs_idx)

    @gs.assert_built
    def set_links_inertial_mass(self, inertial_mass, links_idx_local=None, envs_idx=None):
        links_idx = self._get_global_idx(links_idx_local, self.n_links, self._link_start, unsafe=True)
        self._solver.set_links_inertial_mass(inertial_mass, links_idx, envs_idx)

    @gs.assert_built
    def set_links_invweight(self, invweight, links_idx_local=None, envs_idx=None):
        raise DeprecationError(
            "This method has been removed because links invweights are supposed to be a by-product of link properties "
            "(mass, pose, and inertia matrix), joint placements, and dof armatures. Please consider using the "
            "considering setters instead."
        )

    @gs.assert_built
    def set_mass(self, mass):
        """
        Set the mass of the entity.

        Parameters
        ----------
        mass : float
            The mass to set.
        """
        ratio = float(mass) / self.get_mass()
        for link in self.links:
            link.set_mass(link.get_mass() * ratio)

    @gs.assert_built
    def get_mass(self):
        """
        Get the total mass of the entity in kg.

        For heterogeneous entities, returns an array of masses for each environment.
        For non-heterogeneous entities, returns a scalar mass.

        Returns
        -------
        mass : float | np.ndarray
            The total mass of the entity in kg. For heterogeneous entities, returns
            an array of shape (n_envs,) with per-environment masses.
        """
        if self._enable_heterogeneous:
            # Use solver's batched links_info for accurate per-environment masses
            all_links_mass = self._solver.links_info.inertial_mass.to_numpy()
            links_idx = np.arange(self.link_start, self.link_end)
            # Shape: (n_links, n_envs) -> sum over links axis
            return all_links_mass[links_idx].sum(axis=0)
        else:
            # Original behavior: sum link masses to scalar
            mass = 0.0
            for link in self.links:
                mass += link.get_mass()
            return mass

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
