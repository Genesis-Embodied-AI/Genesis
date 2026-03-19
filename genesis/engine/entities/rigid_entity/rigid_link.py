from itertools import starmap
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
import trimesh

import genesis as gs
from genesis.repr_base import RBC
from genesis.typing import LaxPositiveFArrayType, Matrix3x3Type, UnitVec4FType, Vec3FType
from genesis.utils import geom as gu
from genesis.utils.misc import DeprecationError, qd_to_torch, tensor_to_array

from .rigid_geom import RigidGeom, RigidVisGeom

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

    from .rigid_entity import KinematicEntity, RigidEntity
    from .rigid_joint import RigidJoint


# If mass is too small, we do not care much about spatial inertia discrepancy
MASS_EPS = 0.005
AABB_EPS = 0.002
INERTIA_RATIO_MAX = 100.0


def get_local_inertial_from_geom(geom: RigidGeom | RigidVisGeom, rho: float) -> tuple[float, Vec3FType, Matrix3x3Type]:
    """
    Extract the local inertial properties (mass, center of mass, inertia tensor) of a given rigid geometry.
    """
    geom_type = gs.GEOM_TYPE.MESH if isinstance(geom, RigidVisGeom) else geom.type

    geom_com_local = np.zeros(3)
    if geom_type == gs.GEOM_TYPE.PLANE:
        geom_mass = 0.0
        geom_inertia_local = np.zeros(3, dtype=gs.np_float)
    elif geom_type == gs.GEOM_TYPE.SPHERE:
        radius = geom.data[0]
        geom_mass = (4.0 / 3.0) * np.pi * radius**3 * rho
        I = (2.0 / 5.0) * geom_mass * radius**2
        geom_inertia_local = np.diag([I, I, I])
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        hx, hy, hz = geom.data[:3]
        geom_mass = (4.0 / 3.0) * np.pi * hx * hy * hz * rho
        geom_inertia_local = (geom_mass / 5.0) * np.diag([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2])
    elif geom_type == gs.GEOM_TYPE.CYLINDER:
        radius, height = geom.data[:2]
        geom_mass = np.pi * radius**2 * height * rho
        I_r = (geom_mass / 12.0) * (3.0 * radius**2 + height**2)
        I_z = 0.5 * geom_mass * radius**2
        geom_inertia_local = np.diag([I_r, I_r, I_z])
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        radius, height = geom.data[:2]
        m_cyl = np.pi * radius**2 * height * rho
        m_sph = (4.0 / 3.0) * np.pi * radius**3 * rho
        geom_mass = m_cyl + m_sph
        I_r = (m_cyl * radius**2 / 12.0 * (3.0 + height**2 / radius**2)) + (
            m_sph * radius**2 / 4.0 * (83.0 / 80.0 + (height / radius + 3.0 / 4.0) ** 2)
        )
        I_h = 0.5 * m_cyl * radius**2 + (2.0 / 5.0) * m_sph * radius**2
        geom_inertia_local = np.diag([I_r, I_r, I_h])
    elif geom_type == gs.GEOM_TYPE.BOX:
        hx, hy, hz = geom.data[:3]
        geom_mass = (hx * hy * hz) * rho
        geom_inertia_local = (geom_mass / 12.0) * np.diag([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2])
    else:
        # MESH type
        if isinstance(geom, RigidVisGeom):
            inertia_mesh = trimesh.Trimesh(geom.init_vverts, geom.init_vfaces, process=False)
        else:
            inertia_mesh = trimesh.Trimesh(geom.init_verts, geom.init_faces, process=False)

        if not inertia_mesh.is_watertight:
            inertia_mesh = inertia_mesh.convex_hull

        # FIXME: without this check, some geom will have negative volume even after the above convex
        # hull operation, e.g. 'tests/test_examples.py::test_example[rigid/terrain_from_mesh.py-None]'
        if inertia_mesh.volume < 0.0:
            inertia_mesh.invert()

        inertia_mesh.density = rho
        geom_mass = inertia_mesh.mass
        geom_com_local = inertia_mesh.center_mass
        geom_inertia_local = inertia_mesh.moment_inertia

    return geom_mass, geom_com_local, geom_inertia_local


def compose_inertial_properties(
    geoms_inertial_info: Sequence[tuple[float, Vec3FType, Matrix3x3Type, Vec3FType, UnitVec4FType]],
) -> tuple[float, Vec3FType, Matrix3x3Type]:
    """
    Compose mass, center of mass, and inertia tensor from multiple geometries.
    """
    global_mass = 0.0
    if geoms_inertial_info:
        geoms_mass, geoms_com_local, geoms_I_local, geoms_pos, geoms_quat = zip(*geoms_inertial_info)
        geoms_mass = np.asarray(geoms_mass)
        global_mass = geoms_mass.sum()

    if global_mass == 0.0:
        return 0.0, np.zeros(3, dtype=np.float64), np.zeros((3, 3), dtype=np.float64)

    # Compute world COMs of each geom
    geoms_com_world = np.stack(
        tuple(starmap(gu.transform_by_trans_quat, zip(geoms_com_local, geoms_pos, geoms_quat))), axis=0
    )

    # Compute total COM
    global_com = (geoms_mass[:, None] * geoms_com_world).sum(axis=0) / global_mass

    # Accumulate inertia about global COM
    global_inertia = np.zeros((3, 3), dtype=np.float64)

    # Transform local inertias directly to global COM frame using parallel axis theorem
    for geom_mass, geom_I_local, geom_quat, geom_com_world in zip(
        geoms_mass, geoms_I_local, geoms_quat, geoms_com_world
    ):
        T_offset = gu.trans_quat_to_T(global_com - geom_com_world, geom_quat)
        geom_I_world = gu.transform_inertia_by_T(geom_I_local, T_offset, geom_mass)
        global_inertia += geom_I_world

    return global_mass, global_com, global_inertia


def compute_inertial_from_geoms(
    geoms: Sequence[RigidGeom | RigidVisGeom], rho: float
) -> tuple[float, Vec3FType, Matrix3x3Type]:
    """
    Compose inertial properties (mass, center of mass, inertia tensor) from multiple rigid geometries.

    Handles all primitive collision geometry types analytically (SPHERE, ELLIPSOID, CYLINDER, CAPSULE, BOX) and falls
    back to trimesh for MESH type.

    Parameters
    ----------
    geoms : list[RigidGeom] or list[RigidVisGeom]
        List of geometry objects to compute inertial from.
    rho : float
        Material density (kg/m^3).

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        (total_mass, center_of_mass, inertia_tensor)
    """
    # Extract inertia information
    geoms_inertial_info = tuple(
        (*get_local_inertial_from_geom(geom, rho), geom._init_pos, geom._init_quat) for geom in geoms
    )

    # Compose all inertia of all geometries in parent link frame
    return compose_inertial_properties(geoms_inertial_info)


class KinematicLink(RBC):
    """
    Kinematic class. One KinematicEntity consists of multiple KinematicLinks, each of which is a rigid body and could
    consist of multiple RigidVisGeoms (`link.vgeoms` for visualization).
    """

    def __init__(
        self,
        entity: "KinematicEntity",
        name: str,
        idx: int,
        joint_start: int,
        n_joints: int,
        vgeom_start: int,
        vvert_start: int,
        vface_start: int,
        pos: "np.typing.ArrayLike",
        quat: "np.typing.ArrayLike",
        parent_idx: int,
        root_idx: int | None,
    ):
        self._name: str = name
        self._entity: "KinematicEntity" = entity
        self._solver: "RigidSolver" = entity.solver
        self._entity_idx_in_solver = entity._idx_in_solver

        self._uid = gs.UID()
        self._idx: int = idx
        self._parent_idx: int = parent_idx  # -1 if no parent

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

        self._vgeom_start: int = vgeom_start
        self._vvert_start: int = vvert_start
        self._vface_start: int = vface_start

        # Link position & rotation at creation time:
        self._pos: "np.typing.ArrayLike" = pos
        self._quat: "np.typing.ArrayLike" = quat

        self._vgeoms: list[RigidVisGeom] = gs.List()

        # Heterogeneous variant tracking (None = not heterogeneous)
        self._variant_vgeom_ranges: list[tuple[int, int]] | None = None

    def _init_variant_tracking(self):
        """Start tracking heterogeneous variants. Records first variant from current state."""
        self._variant_vgeom_ranges = [(self._vgeom_start, self._vgeom_start + self.n_vgeoms)]

    def _record_variant_vgeom_range(self, n_new_vgeoms):
        """Record a new variant's vgeom range."""
        prev_end = self._variant_vgeom_ranges[-1][1]
        self._variant_vgeom_ranges.append((prev_end, prev_end + n_new_vgeoms))

    def _build(self):
        for vgeom in self._vgeoms:
            vgeom._build()

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
        return self._solver.get_links_pos(self._idx, envs_idx)[..., 0, :]

    @gs.assert_built
    def get_quat(self, envs_idx=None):
        """
        Get the quaternion of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the quaternion. If None, get the quaternion of all environments. Default is None.
        """
        return self._solver.get_links_quat(self._idx, envs_idx)[..., 0, :]

    @gs.assert_built
    def get_vel(self, envs_idx=None) -> torch.Tensor:
        """
        Get the linear velocity of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the linear velocity. If None, get the linear velocity of all environments. Default is None.
        """
        return self._solver.get_links_vel(self._idx, envs_idx)[..., 0, :]

    @gs.assert_built
    def get_ang(self, envs_idx=None) -> torch.Tensor:
        """
        Get the angular velocity of the link in the world frame.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the angular velocity. If None, get the angular velocity of all environments. Default is None.
        """
        return self._solver.get_links_ang(self._idx, envs_idx)[..., 0, :]

    @gs.assert_built
    def get_vAABB(self, envs_idx=None):
        """
        Get the axis-aligned bounding box (AABB) of the link's visual body in the world frame by aggregating all
        the visual geometries associated with this link (`link.vgeoms`).
        """
        if self.n_vgeoms == 0:
            gs.raise_exception("Link has no visual geometries.")

        # For heterogeneous entities, compute AABB per-environment respecting active_envs_idx
        if self.entity._enable_heterogeneous:
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
    def entity(self) -> "KinematicEntity":
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
    def joints(self) -> list["RigidJoint"]:
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
    def idx_local(self):
        """
        The local index of the link in the entity.
        """
        return self._idx - self._entity.link_start

    @property
    def is_fixed(self):
        """
        Whether the link is fixed wrt the world.
        """
        return self._is_fixed

    @property
    def invweight(self):
        """Inverse weight of the link. Always zero for KinematicLink (infinite mass)."""
        return np.zeros(2, dtype=gs.np_float)

    @property
    def pos(self) -> "np.typing.ArrayLike":
        """
        The initial position of the link. For real-time position, use `link.get_pos()`.
        """
        return self._pos

    @property
    def quat(self) -> "np.typing.ArrayLike":
        """
        The initial quaternion of the link. For real-time quaternion, use `link.get_quat()`.
        """
        return self._quat

    @property
    def inertial_pos(self):
        """Initial position of the link's inertial frame. Zero for KinematicLink."""
        return np.zeros(3, dtype=gs.np_float)

    @property
    def inertial_quat(self):
        """Initial quaternion of the link's inertial frame. Identity for KinematicLink."""
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=gs.np_float)

    @property
    def inertial_mass(self):
        """Mass of the link. Always 0.0 for KinematicLink."""
        return 0.0

    @property
    def inertial_i(self):
        """Inertia matrix of the link. Zero for KinematicLink."""
        return np.zeros((3, 3), dtype=gs.np_float)

    @property
    def vgeoms(self) -> list[RigidVisGeom]:
        """
        The list of the link's visualization geometries (`RigidVisGeom`).
        """
        return self._vgeoms

    @property
    def geom_start(self) -> int:
        """Start index of collision geoms. Always 0 for KinematicLink."""
        return 0

    @property
    def geom_end(self) -> int:
        """End index of collision geoms. Always 0 for KinematicLink."""
        return 0

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
    def n_verts(self) -> int:
        """Number of collision vertices. Always 0 for KinematicLink."""
        return 0

    @property
    def n_vverts(self) -> int:
        """
        Number of vertices of all the link's vgeoms.
        """
        return sum([vgeom.n_vverts for vgeom in self._vgeoms])

    @property
    def n_vfaces(self) -> int:
        """
        Number of faces of all the link's vgeoms.
        """
        return sum([vgeom.n_vfaces for vgeom in self._vgeoms])

    @property
    def is_built(self) -> bool:
        """
        Whether the entity the link belongs to is built.
        """
        return self.entity.is_built

    # ------------------------------------------------------------------------------------
    # -------------------------------------- repr ----------------------------------------
    # ------------------------------------------------------------------------------------

    def _repr_brief(self):
        return f"{(self.__repr_name__())}: {self._uid}, name: '{self._name}', idx: {self._idx}"


class RigidLink(KinematicLink):
    """
    RigidLink class. One RigidEntity consists of multiple RigidLinks, each of which is a rigid body and could consist of
    multiple RigidGeoms (`link.geoms`, for collision) and RigidVisGeoms (`link.vgeoms` for visualization).
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
        pos: "np.typing.ArrayLike",
        quat: "np.typing.ArrayLike",
        inertial_pos: "np.typing.ArrayLike | None",
        inertial_quat: "np.typing.ArrayLike | None",
        inertial_i: "np.typing.ArrayLike | None",  # may be None, eg. for plane; NDArray is 3x3 matrix
        inertial_mass: float | None,  # may be None, eg. for plane
        parent_idx: int,
        root_idx: int | None,
        invweight: float | None,
        visualize_contact: bool,
    ):
        super().__init__(
            entity,
            name,
            idx,
            joint_start,
            n_joints,
            vgeom_start,
            vvert_start,
            vface_start,
            pos,
            quat,
            parent_idx,
            root_idx,
        )

        if self._is_fixed and not entity._batch_fixed_verts:
            verts_state_start = fixed_verts_state_start
        else:
            verts_state_start = free_verts_state_start

        self._geom_start: int = geom_start
        self._cell_start: int = cell_start
        self._vert_start: int = vert_start
        self._face_start: int = face_start
        self._edge_start: int = edge_start
        self._verts_state_start: int = verts_state_start

        # Link's center-of-mass position & principal axes frame rotation at creation time:
        if inertial_pos is not None:
            inertial_pos = np.asarray(inertial_pos, dtype=gs.np_float)
        self._inertial_pos: "np.typing.ArrayLike | None" = inertial_pos
        if inertial_quat is None:
            inertial_quat = (1.0, 0.0, 0.0, 0.0)
        self._inertial_quat: "np.typing.ArrayLike" = np.asarray(inertial_quat, dtype=gs.np_float)
        if inertial_mass is not None:
            inertial_mass = float(inertial_mass)
        self._inertial_mass: float | None = inertial_mass
        if inertial_i is not None:
            inertial_i = np.asarray(inertial_i, dtype=gs.np_float)
        self._inertial_i: "np.typing.ArrayLike | None" = inertial_i
        self._invweight: float | None = invweight

        self._visualize_contact = visualize_contact

        self._geoms: list[RigidGeom] = gs.List()

        # Heterogeneous variant tracking (None = not heterogeneous)
        self._variant_geom_ranges: list[tuple[int, int]] | None = None

    def _init_variant_tracking(self):
        """Start tracking heterogeneous variants. Records first variant from current state."""
        super()._init_variant_tracking()
        self._variant_geom_ranges = [(self._geom_start, self._geom_start + self.n_geoms)]
        self._variant_scene_inertial: list[tuple] | None = None

    def _record_variant_geom_range(self, n_new_geoms):
        """Record a new variant's geom range."""
        prev_geom_end = self._variant_geom_ranges[-1][1]
        self._variant_geom_ranges.append((prev_geom_end, prev_geom_end + n_new_geoms))

    def _build(self):
        super()._build()

        for geom in self._geoms:
            geom._build()

        # Estimate the spatial inertia of the link. It will be used as a guess if not specified in morph, or as baseline
        # to proof-check the provided values.
        hint_mass = 0.0
        hint_com = np.zeros(3, dtype=gs.np_float)
        hint_inertia = np.zeros((3, 3), dtype=gs.np_float)
        aabb_min = np.full((3,), float("inf"), dtype=gs.np_float)
        aabb_max = np.full((3,), float("-inf"), dtype=gs.np_float)
        if not self._is_fixed:
            # Determine which geom list to use: geoms first, then vgeoms, then fallback
            if self._geoms:
                is_visual = False
                geom_list = self._geoms
            else:
                is_visual = True
                geom_list = self._vgeoms

            # Get material density
            rho = self.entity.material.rho

            # For heterogeneous links, only use the first variant's geoms for the hint.
            if self._variant_geom_ranges is not None:
                start, end = self._variant_geom_ranges[0]
                if not is_visual:
                    geom_list = [g for g in geom_list if start <= g.idx < end]
                else:
                    vs, ve = self._variant_vgeom_ranges[0]
                    geom_list = [vg for vg in geom_list if vs <= vg.idx < ve]

            hint_mass, hint_com, hint_inertia = compute_inertial_from_geoms(geom_list, rho)

            # Compute the bounding box of the links using both visual and collision geometries to be conservative
            for geoms, is_visual in zip((self._geoms, self._vgeoms), (False, True)):
                for geom in geoms:
                    verts = geom.init_vverts if is_visual else geom.init_verts
                    verts = gu.transform_by_trans_quat(verts, geom._init_pos, geom._init_quat)
                    aabb_min = np.minimum(aabb_min, verts.min(axis=0))
                    aabb_max = np.maximum(aabb_max, verts.max(axis=0))

        # Make sure that provided spatial inertia is consistent with the estimate from the geometries if not fixed
        if (self._inertial_mass or hint_mass) > MASS_EPS and hint_mass > gs.EPS:
            if self._inertial_pos is not None:
                tol = (aabb_max - aabb_min) * AABB_EPS + AABB_EPS
                if not ((aabb_min - tol < self._inertial_pos) & (self._inertial_pos < aabb_max + tol)).all():
                    com_str: list[str] = []
                    aabb_str: list[str] = []
                    for name, pos, axis_min, axis_max in zip(("x", "y", "z"), self._inertial_pos, aabb_min, aabb_max):
                        com_str.append(f"{name}={pos:0.3f}")
                        aabb_str.append(f"{name}=({axis_min:0.3f}, {axis_max:0.3f})")
                    gs.logger.warning(
                        f"Link '{self._name}' has dubious center of mass [{', '.join(com_str)}] compared to the "
                        f"bounding box from geometry [{', '.join(aabb_str)}]."
                    )

            if self._inertial_mass is not None:
                if not (hint_mass / INERTIA_RATIO_MAX <= self._inertial_mass <= INERTIA_RATIO_MAX * hint_mass):
                    gs.logger.warning(
                        f"Link '{self._name}' has dubious mass {self._inertial_mass:0.3f} compared to the estimate "
                        f"from geometry {hint_mass:0.3f} given material density {rho:0.0f}."
                    )
                hint_inertia *= self._inertial_mass / hint_mass
                hint_mass = self._inertial_mass

            if self._inertial_i is not None:
                inertia_diag = np.diag(self._inertial_i)
                hint_inertia_diag = np.diag(hint_inertia)
                if not (
                    (hint_inertia_diag / INERTIA_RATIO_MAX <= inertia_diag)
                    & (inertia_diag <= INERTIA_RATIO_MAX * hint_inertia_diag)
                ).all():
                    inertias_str = []
                    for data in (inertia_diag, hint_inertia_diag):
                        inertia_str = ",".join(f"{name}={val:0.3e}" for name, val in zip(("ixx", "iyy", "izz"), data))
                        inertias_str.append(inertia_str)
                    gs.logger.warning(
                        f"Link '{self._name}' has dubious inertia [" + inertias_str[0] + "] compared to the estimate "
                        "from geometry [" + inertias_str[1] + f"] given material density {rho:0.3f}."
                    )

        if self._inertial_mass is None or self._inertial_pos is None or self._inertial_i is None:
            if not self._is_fixed:
                if not self._geoms and not self._vgeoms:
                    if any(joint.type is not gs.JOINT_TYPE.FIXED for joint in self.joints):
                        gs.logger.info(
                            f"Mass not specified and no geoms found for link '{self.name}'. Setting to 'gs.EPS'."
                        )
                elif not self._geoms:
                    gs.logger.info(
                        f"Mass is not specified and collision geoms can not be found for link '{self.name}'. "
                        f"Using visual geoms to compute inertial properties."
                    )
            if self._inertial_mass is None:
                self._inertial_mass = hint_mass
            if self._inertial_pos is None or self._inertial_i is None:
                if self._inertial_pos is not None and self._inertial_i is None:
                    gs.logger.warning(
                        f"Ignoring center of mass of link '{self.name}' because inertia matrix is not specified."
                    )
                elif self._inertial_pos is None and self._inertial_i is not None:
                    gs.logger.warning(
                        f"Ignoring inertia matrix of link '{self.name}' because center of mass is not specified."
                    )
                self._inertial_pos = hint_com
                self._inertial_i = hint_inertia
            self._inertial_quat = gu.identity_quat()

        # FIXME: Setting zero mass even for fixed links breaks physics for some reason...
        # For non-fixed links, it must be non-zero in case for coupling with deformable body solvers.
        self._inertial_mass = max(self._inertial_mass, gs.EPS)

        # Postpone computation of inverse weight if not specified
        if self._invweight is None:
            self._invweight = np.full((2,), fill_value=-1.0, dtype=gs.np_float)

        # override invweight if fixed
        if self._is_fixed:
            self._invweight = np.zeros((2,), dtype=gs.np_float)

        # Compute per-variant inertial for heterogeneous links
        if self._variant_geom_ranges is not None:
            rho = self.entity.material.rho
            self._variant_inertial = []
            for v in range(len(self._variant_geom_ranges)):
                if v == 0:
                    # Primary variant: use the link's own parsed/computed inertial
                    self._variant_inertial.append(
                        (
                            self._inertial_mass,
                            np.asarray(self._inertial_pos, dtype=gs.np_float),
                            (
                                np.asarray(self._inertial_quat, dtype=gs.np_float)
                                if self._inertial_quat is not None
                                else gu.identity_quat()
                            ),
                            np.asarray(self._inertial_i, dtype=gs.np_float),
                        )
                    )
                    continue

                # For URDF/MJCF variants, use parsed inertial from the variant file
                # (index v-1 because variant 0 is the primary)
                if self._variant_scene_inertial is not None:
                    morph_v, v_mass, v_pos, v_quat, v_i = self._variant_scene_inertial[v - 1]
                    if (
                        not (morph_v.recompute_inertia and not self._is_fixed)
                        and v_mass is not None
                        and v_pos is not None
                        and v_i is not None
                    ):
                        self._variant_inertial.append(
                            (
                                v_mass,
                                np.asarray(v_pos, dtype=gs.np_float),
                                np.asarray(v_quat, dtype=gs.np_float) if v_quat is not None else gu.identity_quat(),
                                np.asarray(v_i, dtype=gs.np_float),
                            )
                        )
                        continue

                # Compute from geometry (Primitive/Mesh variants, or recompute_inertia)
                gs_v, ge_v = self._variant_geom_ranges[v]
                vs_v, ve_v = self._variant_vgeom_ranges[v]
                variant_geoms = [g for g in self._geoms if gs_v <= g.idx < ge_v]
                if variant_geoms:
                    mass, com, inertia = compute_inertial_from_geoms(variant_geoms, rho)
                else:
                    variant_vgeoms = [vg for vg in self._vgeoms if vs_v <= vg.idx < ve_v]
                    if variant_vgeoms:
                        mass, com, inertia = compute_inertial_from_geoms(variant_vgeoms, rho)
                    else:
                        mass = gs.EPS
                        com = np.zeros(3, dtype=gs.np_float)
                        inertia = np.zeros((3, 3), dtype=gs.np_float)
                self._variant_inertial.append((mass, com, gu.identity_quat(), inertia))

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

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_verts(self):
        """
        Get the vertices of the link's collision body (concatenation of all `link.geoms`) in the world frame.
        """
        if self.entity._enable_heterogeneous:
            gs.raise_exception("This method is not supported for heterogeneous entity.")

        geoms_idx = slice(self.geom_start, self.geom_end)
        self._solver.update_verts_for_geoms(geoms_idx)

        verts_idx = slice(self._verts_state_start, self._verts_state_start + self.n_verts)
        if self.is_fixed and not self._entity._batch_fixed_verts:
            tensor = qd_to_torch(self._solver.fixed_verts_state.pos, verts_idx, copy=True)
        else:
            tensor = qd_to_torch(self._solver.free_verts_state.pos, None, verts_idx, transpose=True, copy=True)
            if self._solver.n_envs == 0:
                tensor = tensor[0]
        return tensor

    @gs.assert_built
    def get_AABB(self):
        """
        Get the axis-aligned bounding box (AABB) of the link's collision body in the world frame by aggregating all
        the collision geometries associated with this link (`link.geoms`).
        """
        if self.n_geoms == 0:
            gs.raise_exception("Link has no collision geometries.")

        # For heterogeneous entities, compute AABB per-environment respecting active_envs_idx.
        # FIXME: Remove this branch after implementing 'get_verts'.
        if self.entity._enable_heterogeneous and self._solver.n_envs > 0:
            aabb_min = torch.full((self._solver.n_envs, 3), float("inf"), dtype=gs.tc_float, device=gs.device)
            aabb_max = torch.full((self._solver.n_envs, 3), float("-inf"), dtype=gs.tc_float, device=gs.device)
            for geom in self.geoms:
                geom_aabb = geom.get_AABB()
                active_mask = geom.active_envs_mask if geom.active_envs_mask is not None else ()
                aabb_min[active_mask] = torch.minimum(aabb_min[active_mask], geom_aabb[active_mask, 0])
                aabb_max[active_mask] = torch.maximum(aabb_max[active_mask], geom_aabb[active_mask, 1])
            return torch.stack((aabb_min, aabb_max), dim=-2)

        verts = self.get_verts()
        return torch.stack((verts.min(dim=-2).values, verts.max(dim=-2).values), dim=-2)

    @gs.assert_built
    def set_mass(self, mass: LaxPositiveFArrayType):
        """
        Set the mass of the link.

        Parameters
        ----------
        mass : float | array_like, shape (n_envs,)
            The mass to set.
        """
        if self.is_fixed:
            gs.logger.warning("Updating the mass of a link that is fixed wrt world has no effect, skipping.")
            return

        mass = tensor_to_array(mass)
        if np.any(mass < gs.EPS):
            gs.raise_exception(f"Attempt to set mass of link '{self.name}' to {mass}. Mass must be strictly positive.")
        if mass.ndim > 0 and not self._solver._options.batch_links_info:
            gs.raise_exception(
                f"Impossible to set per-env mass of link '{self.name}'. Please specify "
                "'RigidOptions.batch_links_info=True'."
            )

        ratio = mass / self._inertial_mass
        self._solver.set_links_inertia(ratio, [self.idx])
        self._inertial_mass = mass
        self._inertial_i = self._inertial_i * ratio[..., None, None]
        self._invweight = self._invweight / ratio[..., None]

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
    def visualize_contact(self) -> bool:
        """
        Whether to visualize the contact of the link.
        """
        return self._visualize_contact

    @property
    def invweight(self):
        """
        The invweight of the link.
        """
        if self._invweight is None:
            self._invweight = tensor_to_array(self._solver.get_links_invweight(self._idx))[..., 0, :]
        return self._invweight

    @property
    def inertial_pos(self) -> "np.typing.ArrayLike | None":
        """
        The initial position of the link's inertial frame.
        """
        return self._inertial_pos

    @property
    def inertial_quat(self) -> "np.typing.ArrayLike | None":
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
    def inertial_i(self) -> "np.typing.ArrayLike | None":
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
    def n_faces(self) -> int:
        """
        Number of faces of all the link's geoms.
        """
        return sum([geom.n_faces for geom in self._geoms])

    @property
    def n_edges(self) -> int:
        """
        Number of edges of all the link's geoms.
        """
        return sum([geom.n_edges for geom in self._geoms])

    @property
    def is_free(self):
        raise DeprecationError("This property has been removed.")
