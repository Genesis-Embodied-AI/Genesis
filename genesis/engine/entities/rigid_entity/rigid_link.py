from typing import TYPE_CHECKING

import gstaichi as ti
import numpy as np
import torch
from numpy.typing import ArrayLike

import genesis as gs
import trimesh
from genesis.repr_base import RBC
from genesis.utils import geom as gu
from genesis.utils.misc import DeprecationError

from .rigid_geom import RigidGeom, RigidVisGeom, _kernel_get_free_verts, _kernel_get_fixed_verts

if TYPE_CHECKING:
    from .rigid_entity import RigidEntity
    from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver
    from genesis.ext.pyrender.interaction.vec3 import Pose


@ti.data_oriented
class RigidLink(RBC):
    """
    RigidLink class. One RigidEntity consists of multiple RigidLinks, each of which is a rigid body and could consist of multiple RigidGeoms (`link.geoms`, for collision) and RigidVisGeoms (`link.vgeoms` for visualization).
    """

    def __init__(
        self,
        entity: "RigidEntity",
        name: str,
        idx: int,
        joint_start: int,
        n_joints: int,
        geom_start: int,
        cell_start: int,
        vert_start: int,
        face_start: int,
        edge_start: int,
        free_verts_state_start: int,
        fixed_verts_state_start: int,
        vgeom_start: int,
        vvert_start: int,
        vface_start: int,
        pos: ArrayLike,
        quat: ArrayLike,
        inertial_pos: ArrayLike | None,
        inertial_quat: ArrayLike | None,
        inertial_i: ArrayLike | None,  # may be None, eg. for plane; NDArray is 3x3 matrix
        inertial_mass: float | None,  # may be None, eg. for plane
        parent_idx: int,
        root_idx: int | None,
        invweight: float | None,
        visualize_contact: bool,
    ):
        self._name: str = name
        self._entity: "RigidEntity" = entity
        self._solver: "RigidSolver" = entity.solver
        self._entity_idx_in_solver = entity._idx_in_solver

        self._uid = gs.UID()
        self._idx: int = idx
        self._parent_idx: int = parent_idx  # -1 if no parent
        self._child_idxs: list[int] = list()

        # 'is_fixed' attribute specifies whether the link is free to move.
        # In practice, this attributes determines whether the geometry vertices associated with the entity are stored
        # per batch-element and updated at every simulation step, or computed once at build time and shared among the
        # entire batch. This affects correct processing of collision detection and sensor raycasting as a side-effect.
        is_fixed = True
        link = self
        while True:
            is_fixed &= all(joint.type is gs.JOINT_TYPE.FIXED for joint in link.joints)
            if link.parent_idx == -1:
                break
            link = self.entity.links[link.parent_idx - self.entity.link_start]
        if root_idx is None:
            root_idx = link.idx
        self._root_idx: int = root_idx
        self._is_fixed: bool = is_fixed

        self._joint_start: int = joint_start
        self._n_joints: int = n_joints

        self._geom_start: int = geom_start
        self._cell_start: int = cell_start
        self._vert_start: int = vert_start
        self._face_start: int = face_start
        self._edge_start: int = edge_start
        self._verts_state_start: int = fixed_verts_state_start if is_fixed else free_verts_state_start
        self._vgeom_start: int = vgeom_start
        self._vvert_start: int = vvert_start
        self._vface_start: int = vface_start

        # Link position & rotation at creation time:
        self._pos: ArrayLike = pos
        self._quat: ArrayLike = quat
        # Link's center-of-mass position & principal axes frame rotation at creation time:
        if inertial_pos is not None:
            inertial_pos = np.asarray(inertial_pos, dtype=gs.np_float)
        self._inertial_pos: ArrayLike | None = inertial_pos
        if inertial_quat is not None:
            inertial_quat = np.asarray(inertial_quat, dtype=gs.np_float)
        self._inertial_quat: ArrayLike | None = inertial_quat
        self._inertial_mass: float | None = inertial_mass
        self._inertial_i: ArrayLike | None = inertial_i
        self._invweight: float | None = invweight

        self._visualize_contact = visualize_contact

        self._geoms: list[RigidGeom] = gs.List()
        self._vgeoms: list[RigidVisGeom] = gs.List()

    def _build(self):
        for geom in self._geoms:
            geom._build()

        for vgeom in self._vgeoms:
            vgeom._build()

        self._init_mesh = self._compose_init_mesh()

        # inertial_mass and inertia_i
        if self._inertial_mass is None:
            if len(self._geoms) == 0 and len(self._vgeoms) == 0:
                self._inertial_mass = 0.0
            else:
                if self._init_mesh.is_watertight:
                    self._inertial_mass = self._init_mesh.volume * self.entity.material.rho
                else:  # TODO: handle non-watertight mesh
                    self._inertial_mass = 1.0

        # Postpone computation of inverse weight if not specified
        if self._invweight is None:
            self._invweight = np.full((2,), fill_value=-1.0, dtype=gs.np_float)

        # inertial_pos
        if self._inertial_pos is None:
            if self._init_mesh is None:
                self._inertial_pos = gu.zero_pos()
            else:
                self._inertial_pos = np.array(self._init_mesh.center_mass, dtype=gs.np_float)

        # inertial_i
        if self._inertial_i is None:
            # FIXME: Why coef 0.4 ???
            if self._init_mesh is None:  # use sphere inertia with radius 0.1
                self._inertial_i = 0.4 * self._inertial_mass * 0.1**2 * np.eye(3)

            else:
                # attempt to fix non-watertight mesh by convexifying
                inertia_mesh = self._init_mesh.copy()
                if not inertia_mesh.is_watertight:
                    inertia_mesh = trimesh.convex.convex_hull(inertia_mesh)

                if inertia_mesh.is_watertight and self._init_mesh.mass > 0:
                    # TODO: check if this is correct. This is correct if the inertia frame is w.r.t to link frame
                    T_inertia = gu.trans_quat_to_T(self._inertial_pos, self._inertial_quat)
                    self._inertial_i = (
                        self._init_mesh.moment_inertia_frame(T_inertia) / self._init_mesh.mass * self._inertial_mass
                    )

                else:  # approximate with a sphere
                    self._inertial_i = (
                        0.4
                        * self._inertial_mass
                        * (max(self._init_mesh.bounds[1] - self._init_mesh.bounds[0]) / 2.0) ** 2
                        * np.eye(3)
                    )

        self._inertial_i = np.asarray(self._inertial_i, dtype=gs.np_float)

        # override invweight if fixed
        if self._is_fixed:
            self._invweight = np.zeros((2,), dtype=gs.np_float)

        import genesis.engine.solvers.rigid.rigid_solver_decomp as rigid_solver_decomp

        self.rsd = rigid_solver_decomp

    def _compose_init_mesh(self):
        if len(self._geoms) == 0 and len(self._vgeoms) == 0:
            return None
        else:
            init_verts = []
            init_faces = []
            vert_offset = 0
            if len(self._geoms) > 0:
                for geom in self._geoms:
                    init_verts.append(gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat))
                    init_faces.append(geom.init_faces + vert_offset)
                    vert_offset += geom.init_verts.shape[0]
            elif len(self._vgeoms) > 0:  # use vgeom if there's no geom
                for vgeom in self._vgeoms:
                    init_verts.append(gu.transform_by_trans_quat(vgeom.init_vverts, vgeom.init_pos, vgeom.init_quat))
                    init_faces.append(vgeom.init_vfaces + vert_offset)
                    vert_offset += vgeom.init_vverts.shape[0]
            init_verts = np.concatenate(init_verts)
            init_faces = np.concatenate(init_faces)
            return trimesh.Trimesh(init_verts, init_faces)

    def _add_geom(
        self,
        mesh,
        init_pos,
        init_quat,
        type,
        friction,
        sol_params,
        center_init=None,
        needs_coup=False,
        contype=1,
        conaffinity=1,
        data=None,
    ):
        geom = RigidGeom(
            link=self,
            idx=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            verts_state_start=self.n_verts + self._verts_state_start,
            mesh=mesh,
            init_pos=init_pos,
            init_quat=init_quat,
            type=type,
            friction=friction,
            sol_params=sol_params,
            center_init=center_init,
            needs_coup=needs_coup,
            contype=contype,
            conaffinity=conaffinity,
            data=data,
        )
        self._geoms.append(geom)

    def _add_vgeom(self, vmesh, init_pos, init_quat):
        vgeom = RigidVisGeom(
            link=self,
            idx=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            vmesh=vmesh,
            init_pos=init_pos,
            init_quat=init_quat,
        )
        self._vgeoms.append(vgeom)

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_pos(self, envs_idx=None):
        """
        Get the position of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the position. If None, get the position of all environments. Default is None.
        """
        return self._solver.get_links_pos([self._idx], envs_idx).squeeze(-2)

    @gs.assert_built
    def get_quat(self, envs_idx=None):
        """
        Get the quaternion of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the quaternion. If None, get the quaternion of all environments. Default is None.
        """
        return self._solver.get_links_quat([self._idx], envs_idx).squeeze(-2)

    @gs.assert_built
    def get_vel(self, envs_idx=None) -> torch.Tensor:
        """
        Get the linear velocity of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the linear velocity. If None, get the linear velocity of all environments. Default is None.
        """
        return self._solver.get_links_vel([self._idx], envs_idx).squeeze(-2)

    @gs.assert_built
    def get_ang(self, envs_idx=None) -> torch.Tensor:
        """
        Get the angular velocity of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the angular velocity. If None, get the angular velocity of all environments. Default is None.
        """
        return self._solver.get_links_ang([self._idx], envs_idx).squeeze(-2)

    @gs.assert_built
    def get_verts(self):
        """
        Get the vertices of the link's collision body (concatenation of all `link.geoms`) in the world frame.
        """
        self._solver.update_verts_for_geoms(range(self.geom_start, self.geom_end))

        if self.is_fixed:
            tensor = torch.empty((self.n_verts, 3), dtype=gs.tc_float, device=gs.device)
            _kernel_get_fixed_verts(tensor, self._verts_state_start, self.n_verts, self._solver.fixed_verts_state)
        else:
            tensor = torch.empty(
                self._solver._batch_shape((self.n_verts, 3), True), dtype=gs.tc_float, device=gs.device
            )
            _kernel_get_free_verts(tensor, self._verts_state_start, self.n_verts, self._solver.free_verts_state)
            if self._solver.n_envs == 0:
                tensor = tensor.squeeze(0)
        return tensor

    @gs.assert_built
    def get_vverts(self):
        """
        Get the vertices of the link's visualization body (concatenation of all `link.vgeoms`) in the world frame.
        """
        tensor = torch.empty(self._solver._batch_shape((self.n_vverts, 3), True), dtype=gs.tc_float, device=gs.device)
        self._kernel_get_vverts(tensor)
        if self._solver.n_envs == 0:
            tensor = tensor.squeeze(0)
        return tensor

    @ti.kernel
    def _kernel_get_vverts(self, tensor: ti.types.ndarray()):
        for i_vg_, i_b in ti.ndrange(self.n_vgeoms, self._solver._B):
            i_vg = i_vg_ + self._vgeom_start
            for i_v in range(self._solver.vgeoms_info.vvert_start[i_vg], self._solver.vgeoms_info.vvert_end[i_vg]):
                vvert_pos = gu.ti_transform_by_trans_quat(
                    self._solver.vverts_info.init_pos[i_v],
                    self._solver.vgeoms_state.pos[i_vg, i_b],
                    self._solver.vgeoms_state.quat[i_vg, i_b],
                )
                for j in range(3):
                    tensor[i_b, i_v - self._vvert_start, j] = vvert_pos[j]

    @gs.assert_built
    def get_AABB(self):
        """
        Get the axis-aligned bounding box (AABB) of the link's collision body in the world frame by aggregating all
        the collision geometries associated with this link (`link.geoms`).
        """
        verts = self.get_verts()
        return torch.stack((verts.min(axis=-2).values, verts.max(axis=-2).values), axis=-2)

    @gs.assert_built
    def set_mass(self, mass):
        """
        Set the mass of the link.
        """
        if self.is_fixed:
            gs.logger.warning(f"Updating the mass of a link that is fixed wrt world has no effect, skipping.")
            return

        if mass < gs.EPS:
            gs.raise_exception(f"Attempt to set mass of link '{self.name}' to {mass}. Mass must be strictly positive.")

        ratio = float(mass) / self._inertial_mass
        self._inertial_mass *= ratio
        if self._invweight is not None:
            self._invweight /= ratio
        self._inertial_i *= ratio

        self.rsd.kernel_adjust_link_inertia(
            link_idx=self.idx,
            ratio=ratio,
            links_info=self._solver.links_info,
            static_rigid_sim_config=self._solver._static_rigid_sim_config,
            static_rigid_sim_cache_key=self._solver._static_rigid_sim_cache_key,
        )

    @gs.assert_built
    def get_mass(self):
        """
        Get the mass of the link.
        """
        return self._inertial_mass

    def set_friction(self, friction):
        """
        Set the friction of all the link's geoms.
        """
        for geom in self._geoms:
            geom.set_friction(friction)

    # ------------------------------------------------------------------------------------
    # ----------------------------------- properties -------------------------------------
    # ------------------------------------------------------------------------------------

    @property
    def uid(self):
        """
        The unique ID of the link.
        """
        return self._uid

    @property
    def name(self) -> str:
        """
        The name of the link.
        """
        return self._name

    @property
    def entity(self) -> "RigidEntity":
        """
        The entity that the link belongs to.
        """
        return self._entity

    @property
    def solver(self) -> "RigidSolver":
        """
        The solver that the link belongs to.
        """
        return self._solver

    @property
    def visualize_contact(self) -> bool:
        """
        Whether to visualize the contact of the link.
        """
        return self._visualize_contact

    @property
    def joints(self) -> list["Joint"]:
        """
        The sequence of joints that connects the link to its parent link.
        """
        return self.entity.joints_by_links[self.idx_local]

    @property
    def n_joints(self):
        """
        Number of the joints that connects the link to its parent link.
        """
        return self._n_joints

    @property
    def joint_start(self):
        """
        The start index of the link's joints in the RigidSolver.
        """
        return self._joint_start

    @property
    def joint_end(self):
        """
        The end index of the link's joints in the RigidSolver.
        """
        return self._joint_start + self.n_joints

    @property
    def n_dofs(self):
        """The number of degrees of freedom (DOFs) of the entity."""
        return sum(joint.n_dofs for joint in self.joints)

    @property
    def dof_start(self):
        """The index of the link's first degree of freedom (DOF) in the scene."""
        if not self.joints:
            return -1
        return self.joints[0].dof_start

    @property
    def dof_end(self):
        """The index of the link's last degree of freedom (DOF) in the scene *plus one*."""
        if not self.joints:
            return -1
        return self.joints[-1].dof_end

    @property
    def n_qs(self):
        """Returns the number of `q` variables of the link."""
        return sum(joint.n_qs for joint in self.joints)

    @property
    def q_start(self):
        """Returns the starting index of the `q` variables of the link in the rigid solver."""
        if not self.joints:
            return -1
        return self.joints[0].q_start

    @property
    def q_end(self):
        """Returns the last index of the `q` variables of the link in the rigid solver *plus one*."""
        if not self.joints:
            return -1
        return self.joints[-1].q_end

    @property
    def idx(self):
        """
        The global index of the link in the RigidSolver.
        """
        return self._idx

    @property
    def parent_idx(self):
        """
        The global index of the link's parent link in the RigidSolver. If the link is the root link, return -1.
        """
        return self._parent_idx

    @property
    def root_idx(self):
        """
        The global index of the link's root link in the RigidSolver.
        """
        return self._root_idx

    @property
    def child_idxs(self):
        """
        The global indices of the link's child links in the RigidSolver.
        """
        return self._child_idxs

    @property
    def idx_local(self):
        """
        The local index of the link in the entity.
        """
        return self._idx - self._entity.link_start

    @property
    def parent_idx_local(self):
        """
        The local index of the link's parent link in the entity. If the link is the root link, return -1.
        """
        # TODO: check for parent links outside of the current entity (caused by scene.link_entities())
        if self._parent_idx >= 0:
            return self._parent_idx - self._entity.link_start
        return self._parent_idx

    @property
    def child_idxs_local(self):
        """
        The local indices of the link's child links in the entity.
        """
        # TODO: check for child links outside of the current entity (caused by scene.link_entities())
        return [idx - self._entity.link_start if idx >= 0 else idx for idx in self._child_idxs]

    @property
    def is_leaf(self):
        """
        Whether the link is a leaf link (i.e., has no child links).
        """
        return len(self._child_idxs) == 0

    @property
    def is_fixed(self):
        """
        Whether the link is fixed wrt the world.
        """
        return self._is_fixed

    @property
    def invweight(self):
        """
        The invweight of the link.
        """
        if self._invweight is None:
            self._invweight = self._solver.get_links_invweight([self._idx]).cpu().numpy()[..., 0, :]
        return self._invweight

    @property
    def pos(self) -> ArrayLike:
        """
        The initial position of the link. For real-time position, use `link.get_pos()`.
        """
        return self._pos

    @property
    def quat(self) -> ArrayLike:
        """
        The initial quaternion of the link. For real-time quaternion, use `link.get_quat()`.
        """
        return self._quat

    @property
    def inertial_pos(self) -> ArrayLike | None:
        """
        The initial position of the link's inertial frame.
        """
        return self._inertial_pos

    @property
    def inertial_quat(self) -> ArrayLike | None:
        """
        The initial quaternion of the link's inertial frame.
        """
        return self._inertial_quat

    @property
    def inertial_mass(self) -> float | None:
        """
        The initial mass of the link.
        """
        return self._inertial_mass

    @property
    def inertial_i(self) -> ArrayLike | None:
        """
        The inerial matrix of the link.
        """
        return self._inertial_i

    @property
    def geoms(self) -> list[RigidGeom]:
        """
        The list of the link's collision geometries (`RigidGeom`).
        """
        return self._geoms

    @property
    def vgeoms(self) -> list[RigidVisGeom]:
        """
        The list of the link's visualization geometries (`RigidVisGeom`).
        """
        return self._vgeoms

    @property
    def n_geoms(self) -> int:
        """
        Number of the link's collision geometries.
        """
        return len(self._geoms)

    @property
    def geom_start(self) -> int:
        """
        The start index of the link's collision geometries in the RigidSolver.
        """
        return self._geom_start

    @property
    def geom_end(self) -> int:
        """
        The end index of the link's collision geometries in the RigidSolver.
        """
        return self._geom_start + self.n_geoms

    @property
    def n_vgeoms(self) -> int:
        """
        Number of the link's visualization geometries (`vgeom`).
        """
        return len(self._vgeoms)

    @property
    def vgeom_start(self) -> int:
        """
        The start index of the link's vgeom in the RigidSolver.
        """
        return self._vgeom_start

    @property
    def vgeom_end(self) -> int:
        """
        The end index of the link's vgeom in the RigidSolver.
        """
        return self._vgeom_start + self.n_vgeoms

    @property
    def n_cells(self):
        """
        Number of sdf cells of all the link's geoms.
        """
        return sum([geom.n_cells for geom in self._geoms])

    @property
    def n_verts(self) -> int:
        """
        Number of vertices of all the link's geoms.
        """
        return sum([geom.n_verts for geom in self._geoms])

    @property
    def n_vverts(self) -> int:
        """
        Number of vertices of all the link's vgeoms.
        """
        return sum([vgeom.n_vverts for vgeom in self._vgeoms])

    @property
    def n_faces(self) -> int:
        """
        Number of faces of all the link's geoms.
        """
        return sum([geom.n_faces for geom in self._geoms])

    @property
    def n_vfaces(self) -> int:
        """
        Number of faces of all the link's vgeoms.
        """
        return sum([vgeom.n_vfaces for vgeom in self._vgeoms])

    @property
    def n_edges(self) -> int:
        """
        Number of edges of all the link's geoms.
        """
        return sum([geom.n_edges for geom in self._geoms])

    @property
    def is_built(self) -> bool:
        """
        Whether the entity the link belongs to is built.
        """
        return self.entity.is_built

    @property
    def is_free(self):
        raise DeprecationError("This property has been removed.")

    @property
    def pose(self) -> "Pose":
        """Return the current pose of the link (note, this is not necessarily the same as the principal axes frame)."""
        return Pose.from_link(self)

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        return f"{(self._repr_type())}: {self._uid}, name: '{self._name}', idx: {self._idx}"
