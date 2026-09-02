"""Describe a rigid entity: everything its simulation uses. That description suffices to create one.

Resolving what a parse states fills these in, and an entity builds itself from them alone: every field holds
the value it is built with, so a consumer reads a field and applies no fallback of its own. The exception is the
inverse weight of a link and of a joint: only the solver can state it, and its refresh completes the sentinel '-1.0'
at build. A link the world carries is the one case stated here, as zeros.

A built link, joint, geom or equality constraint keeps its description as 'desc'. The matching 'get_*' accessor of
that object returns the value the simulation currently uses.
"""

from dataclasses import dataclass, field

import numpy as np

import genesis as gs
from genesis.constants import EQUALITY_TYPE, GEOM_TYPE, JOINT_TYPE
from genesis.engine.materials.base import Material
from genesis.options.morphs import Morph
from genesis.options.surfaces import Surface
from genesis.utils import geom as gu

from ..base_entity import EntityDescription


@dataclass(kw_only=True)
class BaseRigidGeomDescription:
    """Pose of one geom in its link frame. Collision and visual geom descriptions both hold these fields."""

    pos: np.ndarray = field(default_factory=gu.zero_pos)
    quat: np.ndarray = field(default_factory=gu.identity_quat)


@dataclass(kw_only=True)
class RigidVisGeomDescription(BaseRigidGeomDescription):
    """Describe one geometry a link is drawn with: where it sits on the link, and the mesh drawn there."""

    vmesh: "gs.Mesh"


@dataclass(kw_only=True)
class RigidGeomDescription(BaseRigidGeomDescription):
    """Describe one geometry a link collides with: where it sits on the link, its shape, and how it is simulated.

    Every field is resolved: the geometry has been split and convexified as the morph asked, and the coefficients are
    the ones the geom is built with, the material's where it states them and the asset's otherwise. What an asset
    states before any of that is resolved travels as parsed info, which is where the post-processing options live.

    'density' is the density of this geometry alone, which the link inertial is resolved against. It is None where
    neither the asset nor the material states one, leaving the geom to take the density of its entity.
    """

    type: GEOM_TYPE
    data: np.ndarray | None
    mesh: "gs.Mesh"
    contype: int
    conaffinity: int
    density: float | None
    friction: float
    friction_torsional: float
    friction_rolling: float
    sol_params: np.ndarray


@dataclass
class RigidJointDescription:
    """Describe one joint of a link, with its per-degree-of-freedom quantities.

    The type gives the number of generalized coordinates and of degrees of freedom the joint holds, and construction
    sizes every per-degree-of-freedom quantity the asset leaves unset from that number, the motion axes of a free
    joint and of a fixed one included.
    """

    name: str
    type: JOINT_TYPE
    n_qs: int
    n_dofs: int
    pos: np.ndarray = field(default_factory=gu.zero_pos)
    quat: np.ndarray = field(default_factory=gu.identity_quat)
    init_qpos: np.ndarray | None = None
    sol_params: np.ndarray | None = None
    dofs_motion_ang: np.ndarray | None = None
    dofs_motion_vel: np.ndarray | None = None
    dofs_limit: np.ndarray | None = None
    dofs_invweight: np.ndarray | None = None
    dofs_frictionloss: np.ndarray | None = None
    dofs_stiffness: np.ndarray | None = None
    dofs_damping: np.ndarray | None = None
    dofs_armature: np.ndarray | None = None
    dofs_act_gain: np.ndarray | None = None
    dofs_act_bias: np.ndarray | None = None
    dofs_force_range: np.ndarray | None = None

    def __post_init__(self):
        if self.init_qpos is None:
            self.init_qpos = np.zeros(self.n_dofs)
        if self.sol_params is None:
            self.sol_params = gu.default_solver_params()
        # Resolve the motion axes of a free joint and of a fixed one, the two types whose axes follow from the
        # number of degrees of freedom. A parser fills in the axes of every other type, here or afterwards
        if self.n_dofs in (0, 6):
            if self.dofs_motion_ang is None:
                self.dofs_motion_ang = np.eye(6, 3, -3) if self.n_dofs == 6 else np.zeros((0, 3))
            if self.dofs_motion_vel is None:
                self.dofs_motion_vel = np.eye(6, 3) if self.n_dofs == 6 else np.zeros((0, 3))
        if self.dofs_limit is None:
            self.dofs_limit = np.tile([[-np.inf, np.inf]], [self.n_dofs, 1])
        if self.dofs_force_range is None:
            self.dofs_force_range = np.tile([[-np.inf, np.inf]], [self.n_dofs, 1])
        if self.dofs_act_bias is None:
            self.dofs_act_bias = np.zeros((self.n_dofs, 3))
        if self.dofs_invweight is None:
            self.dofs_invweight = np.zeros(self.n_dofs)
        if self.dofs_frictionloss is None:
            self.dofs_frictionloss = np.zeros(self.n_dofs)
        if self.dofs_stiffness is None:
            self.dofs_stiffness = np.zeros(self.n_dofs)
        if self.dofs_damping is None:
            self.dofs_damping = np.zeros(self.n_dofs)
        if self.dofs_armature is None:
            self.dofs_armature = np.zeros(self.n_dofs)
        if self.dofs_act_gain is None:
            self.dofs_act_gain = np.zeros(self.n_dofs)


@dataclass(kw_only=True)
class KinematicLinkDescription:
    """Describe one link of an entity: the frame it stands at, the joints that move it, and the geoms it is drawn as.

    'is_aligned' says that the build moved the link frame onto the anchor its geoms share.
    """

    name: str
    parent_idx: int
    root_idx: int | None = None
    pos: np.ndarray = field(default_factory=gu.zero_pos)
    quat: np.ndarray = field(default_factory=gu.identity_quat)
    # Offset applied to the link frame: the morph pose offset composed with the root anchoring transform, identity
    # on child links. The relative pose getters strip it, so they report the pose the user gave.
    offset_pos: np.ndarray = field(default_factory=gu.zero_pos)
    offset_quat: np.ndarray = field(default_factory=gu.identity_quat)
    is_robot: bool = False
    is_aligned: bool = False
    joints: list["RigidJointDescription"] = field(default_factory=list)
    vgeoms: list[RigidVisGeomDescription] = field(default_factory=list)


@dataclass(kw_only=True)
class RigidLinkDescription(KinematicLinkDescription):
    """Describe one simulated link: the frame and geoms of a kinematic link, plus its dynamics.

    Every inertial quantity holds the simulated value: the authored one where the asset states it, the geometry
    estimate otherwise. The inverse weight holds the asset value, zeros for a link the world carries, or the sentinel
    '-1.0', which the solver's refresh completes at build.
    """

    inertial_pos: np.ndarray
    inertial_quat: np.ndarray
    inertia: np.ndarray
    mass: float
    invweight: np.ndarray
    geoms: list[RigidGeomDescription] = field(default_factory=list)


@dataclass
class RigidEqualityDescription:
    """Describe one constraint tying two links or two joints of an entity together, named as the asset names them."""

    name: str
    type: EQUALITY_TYPE
    objs_name: tuple[str, str]
    data: np.ndarray
    sol_params: np.ndarray


@dataclass(kw_only=True)
class KinematicVariantLinkDescription:
    """What one variant of a heterogeneous entity gives one link to be drawn as."""

    vgeoms: list[RigidVisGeomDescription] = field(default_factory=list)


@dataclass(kw_only=True)
class RigidVariantLinkDescription(KinematicVariantLinkDescription):
    """What one variant gives one simulated link: the geoms it adds, and the inertial resolved for them.

    A variant stands for the same asset loaded on its own, so its inertial resolves from its own file and its own
    geometry.
    """

    mass: float
    inertial_pos: np.ndarray
    inertial_quat: np.ndarray
    inertia: np.ndarray
    geoms: list[RigidGeomDescription] = field(default_factory=list)


@dataclass
class KinematicVariantDescription:
    """Describe one variant of a heterogeneous entity: where it stands, and what it gives each link.

    A heterogeneous entity holds one set of links and dispatches a variant per environment, so a variant is described
    beside those links rather than as an entity of its own.
    """

    init_qpos: np.ndarray
    offset_pos: np.ndarray
    offset_quat: np.ndarray
    links: list[KinematicVariantLinkDescription] = field(default_factory=list)


@dataclass
class KinematicAttachmentDescription:
    """Where an entity hangs: the entity its base link was attached to, and the link of it that carries it.

    Both are named as the scene names them, since an index shifts as entities are added. The mount pose lives in the
    base link's own pose, so 'attach' without one restores it.
    """

    entity_name: str
    link_name: str


@dataclass
class KinematicEntityDescription(EntityDescription):
    """Describe one entity as its build left it: every link it holds, and every constraint tying them together.

    Each description here stands as the build resolved it rather than as the asset authored it: collision meshes
    split and convexified, link frames aligned on the anchor their geoms share, and the friction the material states
    written into the geoms it overrides. A geom carries its mesh, so an entity is created from this alone, with no
    asset to
    read and no geometry to process again.

    The morphs, material and surface it was given stand here as well, since a link states none of them: the material
    carries the physics the geoms are simulated under, the surface how they are drawn, and a morph what a kind of
    entity needs beyond its links - the propeller links of a drone, the grid of a terrain. 'morphs' lists every
    morph the entity dispatches, the primary first, and a homogeneous entity lists exactly one.
    """

    morphs: list[Morph] = field(default_factory=list)
    material: Material | None = None
    surface: Surface | None = None
    visualize_contact: bool = False
    name: str | None = None
    links: list[KinematicLinkDescription] = field(default_factory=list)
    variants: list[KinematicVariantDescription] = field(default_factory=list)
    attachment: KinematicAttachmentDescription | None = None


@dataclass
class RigidEntityDescription(KinematicEntityDescription):
    """Describe one rigid entity, which is a kinematic entity whose links are simulated.

    A constraint tying two links or two joints together is described here rather than beside a kinematic entity,
    since the rigid solver is what enforces one.
    """

    links: list[RigidLinkDescription] = field(default_factory=list)
    equalities: list[RigidEqualityDescription] = field(default_factory=list)


@dataclass
class TerrainEntityDescription(RigidEntityDescription):
    """Describe one terrain, which is a rigid entity beside the height field its surface is made of.

    The field holds the elevation of every grid cell in meters, vertical scale applied, and the scale gives the
    horizontal size of a cell and the vertical factor. The collider reads them at build and the height query reads
    them at runtime. They stand here because no link describes them.
    """

    terrain_hf: np.ndarray | None = None
    terrain_scale: np.ndarray | None = None


@dataclass
class DroneEntityDescription(RigidEntityDescription):
    """Describe one drone, which is a rigid entity beside the two coefficients its asset states for the propellers.

    The thrust and torque coefficients are read from the asset and appear nowhere else, so a drone created from this
    carries the same flight behaviour without the file it was authored from.
    """

    kf: float | None = None
    km: float | None = None
