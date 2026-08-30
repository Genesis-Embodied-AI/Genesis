from itertools import starmap
from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np
import torch

import genesis as gs
from genesis.constants import link_ref_frame
from genesis.engine.mesh import InertialProperties
from genesis.repr_base import RBC
from genesis.typing import LaxPositiveFArrayType, Matrix3x3Type, UnitVec4FType, Vec3FType
from genesis.utils import geom as gu
from genesis.utils.description import GeomDescription, LinkDescription
from genesis.utils.misc import DeprecationError, qd_to_torch

from .rigid_geom import RigidGeom, RigidVisGeom

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid import RigidSolver
    from genesis.options.morphs import Morph

    from .rigid_entity import KinematicEntity, RigidEntity
    from .rigid_joint import RigidJoint


RHO_OBJECT = 600.0
RHO_ROBOT = 1500.0
RHO_MUJOCO = 1000.0

# If mass is too small, we do not care much about spatial inertia discrepancy
MASS_EPS = 0.005
AABB_EPS = 0.002
INERTIA_RATIO_MAX = 100.0


def get_local_inertial_from_geom(
    geom_type: gs.GEOM_TYPE, data: np.ndarray | None, mesh: "gs.Mesh | None", rho: float = 1.0
) -> InertialProperties:
    """Local inertial properties (mass, center of mass, inertia tensor) of one described geometry.

    Primitive types use the analytic formula on `data`; MESH defers to the mesh's cached unit-density mass properties
    (`Mesh.inertial`) scaled by `rho`. This is the load-time computation available to kinematic and rigid
    entities alike (it operates on the description, not on a geom object), so the anchor it feeds matches the finalized
    geom-derived inertia exactly.
    """
    geom_com_local = np.zeros(3)
    if geom_type == gs.GEOM_TYPE.PLANE:
        geom_mass = 0.0
        geom_inertia_local = np.zeros(3, dtype=gs.np_float)
    elif geom_type == gs.GEOM_TYPE.SPHERE:
        radius = data[0]
        geom_mass = (4.0 / 3.0) * np.pi * radius**3 * rho
        I = (2.0 / 5.0) * geom_mass * radius**2
        geom_inertia_local = np.diag([I, I, I])
    elif geom_type == gs.GEOM_TYPE.ELLIPSOID:
        hx, hy, hz = data[:3]
        geom_mass = (4.0 / 3.0) * np.pi * hx * hy * hz * rho
        geom_inertia_local = (geom_mass / 5.0) * np.diag([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2])
    elif geom_type == gs.GEOM_TYPE.CYLINDER:
        radius, height = data[:2]
        geom_mass = np.pi * radius**2 * height * rho
        I_r = (geom_mass / 12.0) * (3.0 * radius**2 + height**2)
        I_z = 0.5 * geom_mass * radius**2
        geom_inertia_local = np.diag([I_r, I_r, I_z])
    elif geom_type == gs.GEOM_TYPE.CAPSULE:
        radius, height = data[:2]
        m_cyl = np.pi * radius**2 * height * rho
        m_sph = (4.0 / 3.0) * np.pi * radius**3 * rho
        geom_mass = m_cyl + m_sph
        I_r = (m_cyl * radius**2 / 12.0 * (3.0 + height**2 / radius**2)) + (
            m_sph * radius**2 / 4.0 * (83.0 / 80.0 + (height / radius + 3.0 / 4.0) ** 2)
        )
        I_h = 0.5 * m_cyl * radius**2 + (2.0 / 5.0) * m_sph * radius**2
        geom_inertia_local = np.diag([I_r, I_r, I_h])
    elif geom_type == gs.GEOM_TYPE.BOX:
        hx, hy, hz = data[:3]
        geom_mass = (hx * hy * hz) * rho
        geom_inertia_local = (geom_mass / 12.0) * np.diag([hy**2 + hz**2, hx**2 + hz**2, hx**2 + hy**2])
    else:
        # MESH type: reuse the mesh's cached unit-density mass properties; mass and inertia scale linearly with density.
        geom_mass = mesh.inertial.mass * rho
        geom_com_local = mesh.inertial.com
        geom_inertia_local = mesh.inertial.i * rho

    return InertialProperties(geom_mass, geom_com_local, geom_inertia_local)


class GeomInertialInfo(NamedTuple):
    """A geom's intrinsic 'inertial' (in the geom's own frame) placed at pose 'pos'/'quat' in the parent link, as
    consumed by 'compose_inertial_properties'."""

    inertial: InertialProperties
    pos: Vec3FType
    quat: UnitVec4FType


def compose_inertial_properties(geoms_inertial_info: Sequence[GeomInertialInfo]) -> InertialProperties:
    """
    Compose mass, center of mass, and inertia tensor from multiple geometries.
    """
    global_mass = 0.0
    if geoms_inertial_info:
        geoms_inertial, geoms_pos, geoms_quat = zip(*geoms_inertial_info)
        geoms_mass = np.asarray([inertial.mass for inertial in geoms_inertial])
        global_mass = geoms_mass.sum()

    if global_mass == 0.0:
        return InertialProperties(0.0, np.zeros(3, dtype=np.float64), np.zeros((3, 3), dtype=np.float64))

    geoms_com_local = [inertial.com for inertial in geoms_inertial]
    geoms_I_local = [inertial.i for inertial in geoms_inertial]

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

    return InertialProperties(global_mass, global_com, global_inertia)


def compose_inertial_from_geoms(g_descs: Sequence[GeomDescription], rho: float) -> InertialProperties:
    """
    Compose inertial properties (mass, center of mass, inertia tensor) from parsed geom infos.

    Handles all primitive collision geometry types analytically (SPHERE, ELLIPSOID, CYLINDER, CAPSULE, BOX) and defers
    to the mesh's cached unit-density mass properties for MESH type. Visual-only infos are treated as their visual
    mesh.

    Parameters
    ----------
    g_descs : list[GeomDescription]
        Described geoms to compute inertial from.
    rho : float
        Material density (kg/m^3), used for every geom info without its own authored density.
    """
    geoms_inertial_info = tuple(
        GeomInertialInfo(
            get_local_inertial_from_geom(
                gs.GEOM_TYPE.MESH if g_desc.vmesh is not None else g_desc.type,
                g_desc.data,
                g_desc.vmesh if g_desc.vmesh is not None else g_desc.mesh,
                rho if g_desc.density is None else g_desc.density,
            ),
            np.asarray(g_desc.pos, dtype=gs.np_float),
            np.asarray(g_desc.quat, dtype=gs.np_float),
        )
        for g_desc in g_descs
    )
    return compose_inertial_properties(geoms_inertial_info)


class LinkInertial(NamedTuple):
    """A link's (or variant's) finalized inertial in the solver's representation: mass, center of mass 'com', and the
    inertia tensor 'inertia' expressed in the principal frame 'quat' (identity when derived from geometry). Distinct
    from 'InertialProperties', whose 'i' is a single tensor; the solver stores the principal tensor and its orientation
    separately."""

    mass: float
    com: Vec3FType
    quat: UnitVec4FType
    inertia: Matrix3x3Type


class LinkVariant(NamedTuple):
    """One heterogeneous variant of a link: the morph it was loaded from, and what that asset described it with."""

    morph: "Morph"
    desc: LinkDescription


class LinkInertialInfo(NamedTuple):
    """A link's (or variant's) load-time inertial data.

    Computed while the parsed geom infos (and their authored per-geom densities) are still available, and consumed
    by the post-load passes. 'props' feeds the align anchor. 'is_mass_explicit' feeds the all-or-none source check
    in '_align_free_roots': True when the mass is explicit in the asset (an explicit mass, or an authored density on
    every geom), False for a pure geometry estimate (the true mass is a uniform material-density rescale of it), and
    None when the link mixes geoms with and without an authored density (neither explicit nor uniformly rescalable).
    'hint' is the material-density-resolved geometry estimate consumed by 'RigidLink._build' (None for kinematic
    entities, which have no dynamics)."""

    props: LinkInertial
    is_mass_explicit: bool | None
    hint: InertialProperties | None


def finalize_inertial(
    explicit_mass, explicit_com, explicit_quat, explicit_inertia, hint_mass, hint_com, hint_inertia, clamp_min_mass=True
) -> LinkInertial:
    """Resolve a link's local inertial from its parsed explicit values and a geometry-derived estimate (hint).

    Explicit values are used when given; otherwise the geometry estimate is used, and an explicit mass rescales a
    geometry-derived inertia. An omitted center of mass defaults to the link-frame origin when an explicit inertia
    matrix is provided. The hint comes from the load-time inertial info ('compose_inertial_from_geoms' over the
    parsed geom infos, feeding both 'RigidLink._build' and the align anchor) - the single resolution path keeps the
    rigid dynamics inertia and the align anchor in lockstep.

    With ``clamp_min_mass`` the resolved mass is floored at ``gs.EPS`` so a geometry-less moving link stays
    non-singular in the dynamics; the align stash passes ``False`` so a genuinely massless link keeps its ``0.0``
    mass and is excluded from the fixed-subtree composite (it must not inflate the composite by ``gs.EPS``).
    """
    mass, com, quat, inertia = explicit_mass, explicit_com, explicit_quat, explicit_inertia
    if (mass or hint_mass) > MASS_EPS and hint_mass > gs.EPS and mass is not None:
        hint_inertia = hint_inertia * (mass / hint_mass)
        hint_mass = mass
    if mass is None:
        mass = hint_mass
    if inertia is None:
        com, inertia, quat = hint_com, hint_inertia, gu.identity_quat()
    elif com is None:
        # Falling back to the geometry estimate here would discard an otherwise complete authored inertia.
        com = gu.zero_pos()
    if quat is None:
        quat = gu.identity_quat()
    return LinkInertial(
        # For non-fixed links, the mass must be non-zero in case for coupling with deformable body solvers.
        max(mass, gs.EPS) if clamp_min_mass else mass,
        np.asarray(com, dtype=gs.np_float),
        np.asarray(quat, dtype=gs.np_float),
        np.asarray(inertia, dtype=gs.np_float),
    )


class KinematicLink(RBC):
    """
    Kinematic class. One KinematicEntity consists of multiple KinematicLinks, each of which is a rigid body and could
    consist of multiple RigidVisGeoms (`link.vgeoms` for visualization).
    """

    def __init__(
        self,
        entity: "KinematicEntity",
        idx: int,
        parent_idx: int,
        root_idx: int | None,
        joint_start: int,
        n_joints: int,
        vgeom_start: int,
        vvert_start: int,
        vface_start: int,
        desc: LinkDescription,
    ):
        self.desc: LinkDescription = desc
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
        self._root_idx: int = link.idx if root_idx is None else root_idx
        self._is_fixed: bool = is_fixed

        self._joint_start: int = joint_start
        self._n_joints: int = n_joints

        self._vgeom_start: int = vgeom_start
        self._vvert_start: int = vvert_start
        self._vface_start: int = vface_start

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

    def _add_vgeom(self, desc: GeomDescription):
        vgeom = RigidVisGeom(
            link=self,
            idx=self.n_vgeoms + self._vgeom_start,
            vvert_start=self.n_vverts + self._vvert_start,
            vface_start=self.n_vfaces + self._vface_start,
            desc=desc,
        )
        self._vgeoms.append(vgeom)

    # ------------------------------------------------------------------------------------
    # -------------------------------- real-time state -----------------------------------
    # ------------------------------------------------------------------------------------

    @gs.assert_built
    def get_pos(self, envs_idx=None, *, relative=True):
        """
        Get the position of the link.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the position. If None, get the position of all environments. Default is None.
        relative : bool, optional
            Whether to report the position in the user frame, with the entity's morph pose offset and inertial
            alignment stripped, rather than the world frame used by the solver. Defaults to True.
        """
        return self._solver.get_links_pos(self._idx, envs_idx, relative=relative)[..., 0, :]

    @gs.assert_built
    def get_quat(self, envs_idx=None, *, relative=True):
        """
        Get the quaternion of the link.

        Parameters
        ----------
        envs_idx : int or array of int, optional
            The indices of the environments to get the quaternion. If None, get the quaternion of all environments. Default is None.
        relative : bool, optional
            Whether to report the orientation in the user frame, with the entity's morph pose offset and inertial
            alignment stripped, rather than the world frame used by the solver. Defaults to True.
        """
        return self._solver.get_links_quat(self._idx, envs_idx, relative=relative)[..., 0, :]

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
        return self.desc.name

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
    def aligned(self) -> bool:
        """
        Whether the link opts into center-of-mass / principal-axis reframing (the 'align' option, set for a free body
        that opts in or is a primitive). The reframing - and the resulting exactly-diagonal joint-space mass block it
        enables - is applied only when the body is a single rigid body (a free root with no DOF-bearing descendant);
        callers relying on the diagonal mass must check that condition too, as the solver does.
        """
        return self.desc.is_aligned

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
        return f"{(self.__repr_name__())}: {self._uid}, name: '{self.desc.name}', idx: {self._idx}"


class RigidLink(KinematicLink):
    """
    RigidLink class. One RigidEntity consists of multiple RigidLinks, each of which is a rigid body and could consist of
    multiple RigidGeoms (`link.geoms`, for collision) and RigidVisGeoms (`link.vgeoms` for visualization).
    """

    def __init__(
        self,
        entity: "RigidEntity",
        idx: int,
        parent_idx: int,
        root_idx: int | None,
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
        visualize_contact: bool,
        desc: LinkDescription,
    ):
        super().__init__(
            entity, idx, parent_idx, root_idx, joint_start, n_joints, vgeom_start, vvert_start, vface_start, desc
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

        # Normalize the inertial values in the description the link reads them from: arrays for the offsets and the
        # inertia, a float for the mass, and the identity quaternion for an orientation the asset leaves unset
        if desc.inertial_pos is not None:
            desc.inertial_pos = np.asarray(desc.inertial_pos, dtype=gs.np_float)
        desc.inertial_quat = np.asarray(
            (1.0, 0.0, 0.0, 0.0) if desc.inertial_quat is None else desc.inertial_quat, dtype=gs.np_float
        )
        if desc.mass is not None:
            desc.mass = float(desc.mass)
        if desc.inertia is not None:
            desc.inertia = np.asarray(desc.inertia, dtype=gs.np_float)

        self._visualize_contact = visualize_contact

        self._geoms: list[RigidGeom] = gs.List()

        # Heterogeneous collision-geom variant tracking (None = not heterogeneous)
        self._variant_geom_ranges: list[tuple[int, int]] | None = None

    def _init_variant_tracking(self):
        """Start tracking heterogeneous variants. Records first variant from current state."""
        super()._init_variant_tracking()
        self._variant_geom_ranges = [(self._geom_start, self._geom_start + self.n_geoms)]
        self._variant_sources: list[LinkVariant] | None = None

    def _record_variant_geom_range(self, n_new_geoms):
        """Record a new variant's geom range."""
        prev_geom_end = self._variant_geom_ranges[-1][1]
        self._variant_geom_ranges.append((prev_geom_end, prev_geom_end + n_new_geoms))

    def _build(self):
        super()._build()

        for geom in self._geoms:
            geom._build()

        # Estimate the spatial inertia of the link. It will be used as a guess if not specified in morph, or as baseline
        # to proof-check the provided values. The estimate was resolved at load from the described geoms (primary
        # variant), the only time their authored per-geom densities are available.
        hint_mass = 0.0
        hint_com = np.zeros(3, dtype=gs.np_float)
        hint_inertia = np.zeros((3, 3), dtype=gs.np_float)
        aabb_min = np.full((3,), float("inf"), dtype=gs.np_float)
        aabb_max = np.full((3,), float("-inf"), dtype=gs.np_float)
        if not self._is_fixed:
            hint_mass, hint_com, hint_inertia = self.entity._links_inertial_info[self.idx - self.entity._link_start][
                0
            ].hint

            # Compute the bounding box of the links using both visual and collision geometries to be conservative
            for geoms, is_visual in zip((self._geoms, self._vgeoms), (False, True)):
                for geom in geoms:
                    verts = geom.init_vverts if is_visual else geom.init_verts
                    verts = gu.transform_by_trans_quat(verts, geom.init_pos, geom.init_quat)
                    aabb_min = np.minimum(aabb_min, verts.min(axis=0))
                    aabb_max = np.maximum(aabb_max, verts.max(axis=0))

        # The consistency-check block below rescales its working copy of the geometry estimate; keep the raw hint for
        # the shared inertial resolution.
        hint_mass_raw, hint_inertia_raw = hint_mass, np.array(hint_inertia)

        # Make sure that provided spatial inertia is consistent with the estimate from the geometries if not fixed
        if (self.desc.mass or hint_mass) > MASS_EPS and hint_mass > gs.EPS:
            # An omitted center of mass is resolved to the link frame origin whenever the inertia tensor is authored
            # (see 'finalize_inertial'), so it must undergo the same consistency check as an authored one.
            inertial_pos = self.desc.inertial_pos
            if inertial_pos is None and self.desc.inertia is not None:
                inertial_pos = gu.zero_pos()
            if inertial_pos is not None:
                tol = (aabb_max - aabb_min) * AABB_EPS + AABB_EPS
                if not ((aabb_min - tol < inertial_pos) & (inertial_pos < aabb_max + tol)).all():
                    com_str: list[str] = []
                    aabb_str: list[str] = []
                    for name, pos, axis_min, axis_max in zip(("x", "y", "z"), inertial_pos, aabb_min, aabb_max):
                        com_str.append(f"{name}={pos:0.3f}")
                        aabb_str.append(f"{name}=({axis_min:0.3f}, {axis_max:0.3f})")
                    gs.logger.warning(
                        f"Link '{self.desc.name}' has dubious center of mass [{', '.join(com_str)}] compared to the "
                        f"bounding box from geometry [{', '.join(aabb_str)}]."
                    )

            if self.desc.mass is not None:
                if not (hint_mass / INERTIA_RATIO_MAX <= self.desc.mass <= INERTIA_RATIO_MAX * hint_mass):
                    gs.logger.warning(
                        f"Link '{self.desc.name}' has dubious mass {self.desc.mass:0.3f} compared to the estimate "
                        f"from geometry {hint_mass:0.3f}."
                    )
                hint_inertia *= self.desc.mass / hint_mass
                hint_mass = self.desc.mass

            if self.desc.inertia is not None:
                inertia_diag = np.diag(self.desc.inertia)
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
                        f"Link '{self.desc.name}' has dubious inertia ["
                        + inertias_str[0]
                        + "] compared to the estimate "
                        "from geometry [" + inertias_str[1] + "]."
                    )

        if self.desc.mass is None or self.desc.inertia is None:
            if not self._is_fixed and self._vgeoms and not self._geoms:
                gs.logger.info(
                    f"Mass is not specified and collision geoms can not be found for link '{self.name}'. "
                    f"Using visual geoms to compute inertial properties."
                )
            if self.desc.inertial_pos is not None and self.desc.inertia is None:
                gs.logger.warning(
                    f"Ignoring center of mass of link '{self.name}' because inertia matrix is not specified."
                )
            self.desc.invweight = None

        # Resolve the final inertial from the explicit values and the geometry estimate, sharing finalize_inertial with
        # the load-time inertial info so the dynamics inertia and the align anchor stay in lockstep.
        inertial = finalize_inertial(
            self.desc.mass,
            self.desc.inertial_pos,
            self.desc.inertial_quat,
            self.desc.inertia,
            hint_mass_raw,
            hint_com,
            hint_inertia_raw,
        )
        self.desc.mass, self.desc.inertial_pos = inertial.mass, inertial.com
        self.desc.inertial_quat, self.desc.inertia = inertial.quat, inertial.inertia

        # Postpone computation of inverse weight if not specified
        if self.desc.invweight is None:
            self.desc.invweight = np.full((2,), fill_value=-1.0, dtype=gs.np_float)

        # override invweight if fixed
        if self._is_fixed:
            self.desc.invweight = np.zeros((2,), dtype=gs.np_float)

        # Compute per-variant inertial for heterogeneous links
        if self._variant_geom_ranges is not None:
            self._variant_inertial = []
            for v in range(len(self._variant_geom_ranges)):
                if v == 0:
                    # Primary variant: use the link's own parsed/computed inertial
                    self._variant_inertial.append(
                        LinkInertial(
                            self.desc.mass,
                            np.asarray(self.desc.inertial_pos, dtype=gs.np_float),
                            (
                                np.asarray(self.desc.inertial_quat, dtype=gs.np_float)
                                if self.desc.inertial_quat is not None
                                else gu.identity_quat()
                            ),
                            np.asarray(self.desc.inertia, dtype=gs.np_float),
                        )
                    )
                    continue

                # Resolve the inertial parsed from the variant file (index v-1 because variant 0 is the primary)
                # against this variant's own geometry estimate. Going through the shared resolution path is what makes
                # a variant behave like the same asset loaded on its own, whatever it leaves unspecified. A
                # Primitive/Mesh variant has no parsed inertial, and 'recompute_inertia' discards it on purpose, so
                # both fall back to the estimate alone.
                v_mass, v_pos, v_quat, v_i = None, None, None, None
                if self._variant_sources is not None:
                    variant = self._variant_sources[v - 1]
                    if not (variant.morph.recompute_inertia and not self._is_fixed):
                        v_mass = variant.desc.mass
                        v_pos = variant.desc.inertial_pos
                        v_quat = variant.desc.inertial_quat
                        v_i = variant.desc.inertia
                hint = self.entity._links_inertial_info[self.idx - self.entity._link_start][v].hint
                self._variant_inertial.append(finalize_inertial(v_mass, v_pos, v_quat, v_i, *hint))

    def _add_geom(self, desc: GeomDescription, center_init=None, needs_coup=False):
        geom = RigidGeom(
            link=self,
            idx=self.n_geoms + self._geom_start,
            cell_start=self.n_cells + self._cell_start,
            vert_start=self.n_verts + self._vert_start,
            face_start=self.n_faces + self._face_start,
            edge_start=self.n_edges + self._edge_start,
            verts_state_start=self.n_verts + self._verts_state_start,
            needs_coup=needs_coup,
            desc=desc,
            center_init=center_init,
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
            tensor = qd_to_torch(self._solver.dyn_state.fixed_verts.pos, verts_idx, copy=True)
        else:
            tensor = qd_to_torch(self._solver.dyn_state.free_verts.pos, None, verts_idx, transpose=True, copy=True)
            if self._solver.n_envs == 0:
                tensor = tensor[0]
        return tensor

    @gs.assert_built
    def get_AABB(self):
        """
        Get the vertex-based axis-aligned bounding box (AABB) of the link's collision body in the world frame by
        aggregating all the collision geometries associated with this link (`link.geoms`).
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
    def apply_external_wrench(
        self,
        force=None,
        torque=None,
        envs_idx=None,
        *,
        pos=None,
        ref: link_ref_frame = link_ref_frame.link_origin,
        local: bool = False,
    ):
        """
        Apply an external wrench over one simulation step on the link.

        Parameters
        ----------
        force : None | array_like, optional
            The linear force to apply. None for a pure torque. Defaults to None.
        torque : None | array_like, optional
            The torque to apply, on top of the moment induced by the linear force. Defaults to None.
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments will be considered. Defaults to None.
        pos : None | array_like, optional
            Where the linear force is applied, which sets the moment arm of the induced torque. With `local=True`, an
            offset from the origin of the `ref` frame, so the point follows the link as it moves; otherwise a world
            position. None applies the force at the origin of the `ref` frame. Defaults to None.
        ref: gs.link_ref_frame, optional
            The reference frame: the origin of the link ('link_origin'), or its center of mass ('link_COM'). It fixes
            where the linear force acts when `pos` is None, and the axes of the input coordinates when `local=True`.
            Defaults to 'link_origin'.
        local: bool, optional
            Whether `force`, `torque` and `pos` are expressed in the coordinates of the `ref` frame rather than the
            world frame. Defaults to False.
        """
        self._solver.apply_links_external_wrench(force, torque, self._idx, envs_idx, pos=pos, ref=ref, local=local)

    def apply_external_force(
        self, force, envs_idx=None, *, pos=None, ref: link_ref_frame = link_ref_frame.link_origin, local: bool = False
    ):
        """
        Apply an external linear force over one simulation step on the link.

        Refer to `RigidLink.apply_external_wrench` for details.
        """
        self.apply_external_wrench(force, envs_idx=envs_idx, pos=pos, ref=ref, local=local)

    def apply_external_torque(
        self, torque, envs_idx=None, *, ref: link_ref_frame = link_ref_frame.link_origin, local: bool = False
    ):
        """
        Apply an external torque over one simulation step on the link.

        Refer to `RigidLink.apply_external_wrench` for details.
        """
        self.apply_external_wrench(torque=torque, envs_idx=envs_idx, ref=ref, local=local)

    @gs.assert_built
    def set_mass(self, mass: LaxPositiveFArrayType):
        """
        Set the mass of the link, in kg.

        The inertia of the link is left unchanged, so use `set_inertia` to change it too. `RigidEntity.set_mass`
        scales the inertia of every link of an entity instead, by the same factor as its mass, which is how a body of
        the same shape made heavier behaves.

        Parameters
        ----------
        mass : float | array_like, shape (n_envs,)
            The mass to set. Per-environment values require `RigidOptions.batch_links_info=True`.
        """
        if self.is_fixed:
            gs.logger.warning("Updating the mass of a link that is fixed wrt world has no effect, skipping.")
            return

        if np.ndim(mass) > 0 and not self._solver.is_links_info_batched:
            gs.raise_exception(
                f"Impossible to set per-env mass of link '{self.name}'. Please specify "
                "'RigidOptions.batch_links_info=True'."
            )

        self._solver.set_links_mass(mass, self._idx)

    @gs.assert_built
    def set_COM(self, com):
        """
        Set the center of mass (COM) of the link, as an offset in the link's local frame.

        Parameters
        ----------
        com : array_like, shape (3,) or (n_envs, 3)
            The center of mass to set. Per-environment values require `RigidOptions.batch_links_info=True`.
        """
        self._solver.set_links_COM(com, self._idx)

    @gs.assert_built
    def set_inertia(self, inertia):
        """
        Set the inertia matrix of the link, expressed in its inertial frame.

        Parameters
        ----------
        inertia : array_like, shape (3, 3) or (n_envs, 3, 3)
            The inertia matrix to set, which must be symmetric positive definite. Per-environment values require
            `RigidOptions.batch_links_info=True`.
        """
        self._solver.set_links_inertia(inertia, self._idx)

    @gs.assert_built
    def get_invweight(self, envs_idx=None):
        """
        Get the constraint inverse weights of the link: one for forces, one for torques.

        See `RigidEntity.get_links_invweight` for what an inverse weight is and where its value comes from.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        invweight : torch.Tensor, shape (2,) or (n_envs, 2)
            The translational and rotational inverse weight of the link, per environment for a batched scene whose link
            info is batched.
        """
        return self._solver.get_links_invweight(self._idx, envs_idx)[..., 0, :]

    @gs.assert_built
    def get_mass(self, envs_idx=None):
        """
        Get the mass of the link, in kg.

        Parameters
        ----------
        envs_idx : None | array_like, optional
            The indices of the environments. If None, all environments are returned. Defaults to None.

        Returns
        -------
        mass : torch.Tensor, shape (n_envs,) or scalar
            The mass of the link, per environment for a batched scene whose link info is batched.
        """
        return self._solver.get_links_mass(self._idx, envs_idx)[..., 0]

    def set_friction(self, friction):
        """
        Set the friction of all the link's geoms.
        """
        for geom in self._geoms:
            geom.set_friction(friction)

    def set_friction_torsional(self, friction_torsional):
        """
        Set the torsional friction of all the link's geoms (see 'gs.materials.Rigid').
        """
        for geom in self._geoms:
            geom.set_friction_torsional(friction_torsional)

    def set_friction_rolling(self, friction_rolling):
        """
        Set the rolling friction of all the link's geoms (see 'gs.materials.Rigid').
        """
        for geom in self._geoms:
            geom.set_friction_rolling(friction_rolling)

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
