"""
We define all types of morphologies here: shape primitives, meshes, URDF, MJCF, and soft robot description files.

These are independent of backend solver type and are shared by different solvers, e.g. a mesh can be either loaded as a
rigid object / MPM object / FEM object.
"""

import os
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.misc as mu

from .misc import CoacdOptions
from .options import Options


URDF_FORMAT = ".urdf"
MJCF_FORMAT = ".xml"
MESH_FORMATS = (".obj", ".ply", ".stl")
GLTF_FORMATS = (".glb", ".gltf")
USD_FORMATS = (".usd", ".usda", ".usdc", ".usdz")


class TetGenMixin(Options):
    """
    A mixin to introduce TetGen-related options into morph classes that support tetrahedralization using TetGen.
    """

    # FEM specific
    order: int = 1

    # Volumetric mesh entity
    mindihedral: int = 10
    minratio: float = 1.1
    nobisect: bool = True
    quality: bool = True
    maxvolume: float = -1.0
    verbose: int = 0

    force_retet: bool = False


@gs.assert_initialized
class Morph(Options):
    """
    This is the base class for all genesis morphs.
    A morph in genesis is a hybrid concept, encapsulating both the geometry and pose information of an entity.
    This includes shape primitives, meshes, URDF, MJCF, Terrain, and soft robot description files.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The initial position of the entity in meters at creation time. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The initial euler angle of the entity in degrees at creation time. This follows scipy's extrinsic x-y-z
        rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The initial quaternion (w-x-y-z convention) of the entity at creation time.
        If specified, `euler` will be ignored. Defaults to None.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False.
        **This is only used for RigidEntity.**
    is_free : bool, optional
        Whether the entity is free to move. Defaults to True. **This is only used for RigidEntity.**
        This determines whether the entity's geoms have their vertices put into StructFreeVertsState or
        StructFixedVertsState, and effectively whether they're stored per batch-element, or stored once and shared
        for the entire batch. That affects correct processing of collision detection.
    """

    # Note: pos, euler, quat store only initial varlues at creation time, and are unaffected by sim
    pos: tuple = (0.0, 0.0, 0.0)
    euler: Optional[tuple] = None
    quat: Optional[tuple] = None
    visualization: bool = True
    collision: bool = True
    requires_jac_and_IK: bool = False
    is_free: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        if self.pos is not None:
            if not isinstance(self.pos, tuple) or len(self.pos) != 3:
                gs.raise_exception("`pos` should be a 3-tuple.")

        if self.euler is not None:
            if not isinstance(self.euler, tuple) or len(self.euler) != 3:
                gs.raise_exception("`euler` should be a 3-tuple.")

        if self.quat is not None:
            if not isinstance(self.quat, tuple) or len(self.quat) != 4:
                gs.raise_exception("`quat` should be a 4-tuple.")

        if (self.quat is not None) and (self.euler is not None):
            gs.raise_exception("`euler` and `quat` cannot be jointly specified.")

        if self.euler is not None:
            self.quat = tuple(gs.utils.geom.xyz_to_quat(np.array(self.euler), rpy=True, degrees=True))
        elif self.quat is None:
            self.quat = (1.0, 0.0, 0.0, 0.0)

        if not self.visualization and not self.collision:
            gs.raise_exception("`visualization` and `collision` cannot both be False.")

    def _repr_type(self):
        return f"<gs.morphs.{self.__class__.__name__}>"


############################ Nowhere ############################
class Nowhere(Morph):
    """
    Reserved for emitter. Internal use only.
    """

    n_particles: int = 0

    def __init__(self, **data):
        super().__init__(**data)

        if self.n_particles <= 0:
            gs.raise_exception("`n_particles` should be greater than 0.")


############################ Shape Primitives ############################


class Primitive(Morph):
    """
    This is the base class for all shape-primitive morphs.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics.
        Defaults to False. **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False. **This is only used for RigidEntity.**
    contype : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the contype of one geom and the
        conaffinity of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    conaffinity : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the conaffinity of one geom and
        the contype of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    """

    # Rigid specific
    fixed: bool = False
    contype: int = 0xFFFF
    conaffinity: int = 0xFFFF


class Box(Primitive, TetGenMixin):
    """
    Morph defined by a box shape.

    Note
    ----
    Either [`pos` and `size`] or [`lower` and `upper`] should be specified. The latter has a higher priority.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    lower : tuple, shape (3,), optional
        The lower corner of the box in meters. Defaults to None.
    upper : tuple, shape (3,), optional
        The upper corner of the box in meters. Defaults to None.
    size : tuple, shape (3,), optional
        The size of the box in meters. Defaults to None.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False.
        **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False. **This is only used for RigidEntity.**
    contype : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the contype of one geom and the
        conaffinity of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    conaffinity : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the conaffinity of one geom and
        the contype of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    order : int, optional
        The order of the FEM mesh. Defaults to 1. **This is only used for FEMEntity.**
    mindihedral : int, optional
        The minimum dihedral angle in degrees during tetraheralization. Defaults to 10.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    minratio : float, optional
        The minimum tetrahedron quality ratio during tetraheralization. Defaults to 1.1.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    nobisect : bool, optional
        Whether to disable bisection during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    quality : bool, optional
        Whether to improve quality during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    maxvolume : float, optional
        The maximum tetrahedron volume. Defaults to -1.0 (no limit).
        **This is only used for Volumetric Entity that requires tetraheralization.**
    verbose : int, optional
        The verbosity level during tetraheralization. Defaults to 0.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    force_retet : bool, optional
        Whether to force re-tetraheralization. Defaults to False.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    """

    lower: Optional[tuple] = None
    upper: Optional[tuple] = None
    size: Optional[tuple] = None

    def __init__(self, **data):
        super().__init__(**data)

        if self.lower is None or self.upper is None:
            if self.pos is None or self.size is None:
                gs.raise_exception("Either [`pos` and `size`] or [`lower` and `upper`] should be specified.")

            self.lower = tuple((np.array(self.pos) - 0.5 * np.array(self.size)).tolist())
            self.upper = tuple((np.array(self.pos) + 0.5 * np.array(self.size)).tolist())

        else:
            self.pos = tuple(((np.array(self.lower) + np.array(self.upper)) / 2).tolist())
            self.size = tuple((np.array(self.upper) - np.array(self.lower)).tolist())

            if not (np.array(self.upper) >= np.array(self.lower)).all():
                gs.raise_exception("Invalid lower and upper corner.")


class Cylinder(Primitive, TetGenMixin):
    """
    Morph defined by a cylinder shape.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    height : float, optional
        The height of the cylinder in meters. Defaults to 1.0.
    radius : float, optional
        The radius of the cylinder in meters. Defaults to 0.5.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False.
        **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False. **This is only used for RigidEntity.**
    contype : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the contype of one geom and the
        conaffinity of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    conaffinity : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the conaffinity of one geom and
        the contype of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    order : int, optional
        The order of the FEM mesh. Defaults to 1. **This is only used for FEMEntity.**
    mindihedral : int, optional
        The minimum dihedral angle in degrees during tetraheralization. Defaults to 10.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    minratio : float, optional
        The minimum tetrahedron quality ratio during tetraheralization. Defaults to 1.1.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    nobisect : bool, optional
        Whether to disable bisection during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    quality : bool, optional
        Whether to improve quality during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    maxvolume : float, optional
        The maximum tetrahedron volume. Defaults to -1.0 (no limit).
        **This is only used for Volumetric Entity that requires tetraheralization.**
    verbose : int, optional
        The verbosity level during tetraheralization. Defaults to 0.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    force_retet : bool, optional
        Whether to force re-tetraheralization. Defaults to False.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    """

    height: float = 1.0
    radius: float = 0.5


class Sphere(Primitive, TetGenMixin):
    """
    Morph defined by a sphere shape.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    radius : float, optional
        The radius of the sphere in meters. Defaults to 0.5.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False.
        **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False. **This is only used for RigidEntity.**
    contype : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the contype of one geom and the
        conaffinity of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    conaffinity : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the conaffinity of one geom and
        the contype of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    order : int, optional
        The order of the FEM mesh. Defaults to 1. **This is only used for FEMEntity.**
    mindihedral : int, optional
        The minimum dihedral angle in degrees during tetraheralization. Defaults to 10.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    minratio : float, optional
        The minimum tetrahedron quality ratio during tetraheralization. Defaults to 1.1.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    nobisect : bool, optional
        Whether to disable bisection during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    quality : bool, optional
        Whether to improve quality during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    maxvolume : float, optional
        The maximum tetrahedron volume. Defaults to -1.0 (no limit).
        **This is only used for Volumetric Entity that requires tetraheralization.**
    verbose : int, optional
        The verbosity level during tetraheralization. Defaults to 0.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    force_retet : bool, optional
        Whether to force re-tetraheralization. Defaults to False.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    """

    radius: float = 0.5


class Plane(Primitive):
    """
    Morph defined by a plane shape.

    Note
    ----
    Plane is a primitive with infinite size. Note that the `pos` is the center of the plane,
    but essetially only defines a point where the plane passes through.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The center position of the plane in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    normal : tuple, shape (3,), optional
        The normal normal of the plane in its local frame. Defaults to (0, 0, 1).
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False. **This is only used for RigidEntity.**
    contype : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the contype of one geom and the
        conaffinity of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    conaffinity : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the conaffinity of one geom and
        the contype of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    """

    fixed: bool = True
    normal: tuple = (0, 0, 1)

    def __init__(self, **data):
        super().__init__(**data)

        if not isinstance(self.normal, tuple) or len(self.normal) != 3:
            gs.raise_exception("`normal` should be a 3-tuple.")

        if not self.fixed:
            gs.raise_exception("`fixed` must be True for `Plane`.")

        if self.requires_jac_and_IK:
            gs.raise_exception("`requires_jac_and_IK` must be False for `Plane`.")

        self.normal = tuple(np.array(self.normal) / np.linalg.norm(self.normal))


############################ Mesh ############################


class FileMorph(Morph):
    """
    Morph loaded from a file.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly.
        If a 3-tuple, it scales along each axis. Defaults to 1.0.
        Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    decimate : bool, optional
        Whether to decimate (simplify) the mesh. Default to True. **This is only used for RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters the original
        geometry if necessary. 8 does what needs to be done at all costs. Defaults to 5.
        **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh with be decomposed into multiple convex components if a single one is not
        sufficient to met the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_nonconvex : bool, optional
        This parameter is deprecated. Please use 'convexify' and 'decompose_(robot|object)_error_threshold' instead.
    decompose_object_error_threshold : bool, optional:
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threashold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : bool, optional:
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threashold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_options : CoacdOptions, optional
        Options for configuring coacd convex decomposition. Needs to be a `gs.options.CoacdOptions` object.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False.
        **This is only used for RigidEntity.**
    """

    file: Any = ""
    scale: Union[float, tuple] = 1.0
    decimate: bool = True
    decimate_face_num: int = 500
    decimate_aggressiveness: int = 5
    convexify: Optional[bool] = None
    decompose_nonconvex: Optional[bool] = None
    decompose_object_error_threshold: float = 0.15
    decompose_robot_error_threshold: float = float("inf")
    coacd_options: Optional[CoacdOptions] = None
    recompute_inertia: bool = False

    def __init__(self, **data):
        super().__init__(**data)

        if self.decompose_nonconvex is not None:
            if self.decompose_nonconvex:
                # Convex decomposition is automatically disabled if convexify itself is already disabled.
                self.convexify = True
                self.decompose_object_error_threshold = 0.0
                self.decompose_robot_error_threshold = 0.0
            else:
                self.decompose_object_error_threshold = float("inf")
                self.decompose_robot_error_threshold = float("inf")
            gs.logger.warning(
                "FileMorph option 'decompose_nonconvex' is deprecated and will be removed in future release. Please use "
                "'convexify' and 'decompose_(robot|object)_error_threshold' instead."
            )

        # Make sure that this threshold is positive to avoid decomposition of convex and primitive shapes
        self.decompose_object_error_threshold = max(self.decompose_object_error_threshold, gs.EPS)
        self.decompose_robot_error_threshold = max(self.decompose_robot_error_threshold, gs.EPS)

        if self.coacd_options is None:
            self.coacd_options = CoacdOptions()

        if isinstance(self.file, str):
            file = os.path.abspath(self.file)

            if not os.path.exists(file):
                file = os.path.join(gs.utils.get_assets_dir(), self.file)

            if not os.path.exists(file):
                gs.raise_exception(f"File not found in either current directory or assets directory: '{self.file}'.")

            self.file = file

        if isinstance(self, Mesh):
            if isinstance(self.scale, tuple) and len(self.scale) != 3:
                gs.raise_exception("`scale` should be a float or a 3-tuple.")
        else:
            if not isinstance(self.scale, float):
                gs.raise_exception("`scale` should be a float.")

    def _repr_type(self):
        return f"<gs.morphs.{self.__class__.__name__}(file='{self.file}')>"

    def is_format(self, format):
        return self.file.lower().endswith(format)


class Mesh(FileMorph, TetGenMixin):
    """
    Morph loaded from a mesh file.

    Note
    ----
    In order to speed up simulation, the loaded mesh will first be decimated (simplified) to a target number of faces,
    followed by convexification (for collision mesh only).
    Such process can be disabled by setting `decimate` and `convexify` to False.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly.
        If a 3-tuple, it scales along each axis. Defaults to 1.0.
        Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    decimate : bool, optional
        Whether to decimate (simplify) the mesh. Defaults to True. **This is only used for RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters the original
        geometry if necessary. 8 does what needs to be done at all costs. Defaults to 5.
        **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh with be decomposed into multiple convex components if a single one is not
        sufficient to met the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_nonconvex : bool, optional
        This parameter is deprecated. Please use 'convexify' and 'decompose_(robot|object)_error_threshold' instead.
    decompose_object_error_threshold : bool, optional:
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threashold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : bool, optional:
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threashold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_options : CoacdOptions, optional
        Options for configuring coacd convex decomposition. Needs to be a `gs.options.CoacdOptions` object.
    merge_submeshes_for_collision : bool, optional
        Whether to merge submeshes for collision. Defaults to True. **This is only used for RigidEntity.**
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
        **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False.
        **This is only used for RigidEntity.**
    parse_glb_with_trimesh : bool, optional
        Whether to use trimesh to load glb files. Defaults to False, in which case pygltflib will be used.
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False. **This is only used for RigidEntity.**
    contype : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the contype of one geom and the
        conaffinity of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    conaffinity : int, optional
        The 32-bit integer bitmasks used for contact filtering of contact pairs. When the conaffinity of one geom and
        the contype of the other geom share a common bit set to 1, two geoms can collide. Defaults to 0xFFFF.
    group_by_material : bool, optional
        Whether to group submeshes by their visual material type defined in the asset file. Defaults to True.
        **This is only used for RigidEntity.**
    order : int, optional
        The order of the FEM mesh. Defaults to 1. **This is only used for FEMEntity.**
    mindihedral : int, optional
        The minimum dihedral angle in degrees during tetraheralization. Defaults to 10.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    minratio : float, optional
        The minimum tetrahedron quality ratio during tetraheralization. Defaults to 1.1.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    nobisect : bool, optional
        Whether to disable bisection during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    quality : bool, optional
        Whether to improve quality during tetraheralization. Defaults to True.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    maxvolume : float, optional
        The maximum tetrahedron volume. Defaults to -1.0 (no limit).
        **This is only used for Volumetric Entity that requires tetraheralization.**
    verbose : int, optional
        The verbosity level during tetraheralization. Defaults to 0.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    force_retet : bool, optional
        Whether to force re-tetraheralization. Defaults to False.
        **This is only used for Volumetric Entity that requires tetraheralization.**
    """

    parse_glb_with_trimesh: bool = False

    # Rigid specific
    fixed: bool = False
    contype: int = 0xFFFF
    conaffinity: int = 0xFFFF
    group_by_material: bool = True
    merge_submeshes_for_collision: bool = True


class MeshSet(Mesh):
    files: List[Any] = []
    poss: List[tuple] = []
    eulers: List[tuple] = []


############################ Rigid & Articulated ############################


class MJCF(FileMorph):
    """
    Morph loaded from a MJCF file. This morph only supports `RigidEntity`

    Note
    ----
    MJCF file always contains a 'world' body. Although this body is added to the kinematic tree, it is used to define
    the initial pose of the root link. If `pos`, `euler`, or `quat` is specified, it will override the root pose that
    was originally specified in the MJCF file.

    Note
    ----
    Genesis currently processes MJCF as if it describing a single entity instead of an actual scene. This means that
    there is a single gigantic kinematic chain comprising multiple physical kinematic chains connected together using
    fee joints. The definition of kinematic chain has been stretched a bit to allow us. In particular, there must be
    multiple root links instead of a single one. One other related limitation is global / world options defined in MJCF
    but must be set at the scene-level in Genesis are completely ignored at the moment, e.g. the simulation timestep,
    integrator or constraint solver. Building an actual scene hierarchy with multiple independent entities may be
    supported in the future.

    Note
    ----
    Collision filters defined in MJCF are considered "local", i.e. they only apply to collision pairs for which both
    geometries along to that specific entity. This means that there is no way to filter out collision pairs between
    primitive and MJCF entity at the moment.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly.
        If a 3-tuple, it scales along each axis. Defaults to 1.0.
        Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity's baselink in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity's baselink in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity's baselink. If specified, `euler` will be ignored.
        Defaults to None.
    decimate : bool, optional
        Whether to decimate (simplify) the mesh. Defaults to True. **This is only used for RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters the original
        geometry if necessary. 8 does what needs to be done at all costs. Defaults to 5.
        **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh with be decomposed into multiple convex components if a single one is not
        sufficient to met the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_nonconvex : bool, optional
        This parameter is deprecated. Please use 'convexify' and 'decompose_(robot|object)_error_threshold' instead.
    decompose_object_error_threshold : bool, optional:
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threashold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : bool, optional:
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threashold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_options : CoacdOptions, optional
        Options for configuring coacd convex decomposition. Needs to be a `gs.options.CoacdOptions` object.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False.
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to True.
    default_armature : float, optional
        Default rotor inertia of the actuators. In practice it is applied to all joints regardless of whether they are
        actuated. None to disable. Default to 0.1.
    """

    pos: Optional[tuple] = None
    euler: Optional[tuple] = None
    quat: Optional[tuple] = None
    requires_jac_and_IK: bool = True
    default_armature: Optional[float] = 0.1

    def __init__(self, **data):
        super().__init__(**data)
        if not self.is_format(MJCF_FORMAT):
            gs.raise_exception(f"Expected `{MJCF_FORMAT}` extension for MJCF file: {self.file}")

        # What you want to do with scaling is kinda "zoom" the world from the perspective of the entity, i.e. scale the
        # geometric properties of an entity wrt its root pose. In the general case, ie for a 3D vector scale, (x, y, z)
        # dimensions are scaled independently along (x, y, z) world axes respectively. With this definition, it is an
        # intrinsic uniquely-defined geometric property of the entity, and as such, it does not depend on its current
        # configuration (aka. position vector).
        # For rigid non-articulated objects, this is all good and dimension-wise scaling makes sense, but it is no
        # longer the case for poly-articulated robot. This is due to the fact that the position of each geometry in
        # world frame depends on their parent link poses, which themselves depends on the current configuration of the
        # entity. This is problematic as it means that the effect of scaling would depends on the initial configuration
        # of the robot rather then being a intrinsic uniquely-defined geometric property. There is no another way to
        # avoid this inconsistency than limiting scaling to a scalar factor. In this case, scaling between anisotropic
        # and does not depends on the orientation of each geometry anymore, and therefore is independent of the
        # configuration of the entity, which is precisely the property that we want to enforce.
        if isinstance(self.scale, np.ndarray):
            if self.scale.std() > gs.EPS:
                gs.raise_exception("Anisotropic scaling is not supported by MJCF morph.")
            self.scale = self.scale.mean()


class URDF(FileMorph):
    """
    Morph loaded from a URDF file. This morph only supports `RigidEntity`.
    If you need to create a `Drone` entity, use `gs.morphs.Drone` instead.

    Note
    ----
    As part of performance optimization, links connected via a fixed joint are merged if `merge_fixed_links` is True.
    This is turned on by default, and can help improve simulation speed without affecting any dynamics and rendering
    behaviors. However, in cases where certain links are still needed as independent links, such as virtual
    end-effector links created for being used as IK targets, these links will not be merged if their names are added
    to `links_to_keep`. You can also completely turn off link merging by setting `merge_fixed_links` to False,
    but it's recommended to use `merge_fixed_links=True` in combination with `links_to_keep` for better performance.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly.
        If a 3-tuple, it scales along each axis. Defaults to 1.0.
        Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention.
        Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    decimate : bool, optional
        Whether to decimate (simplify) the mesh. Defaults to True. **This is only used for RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters the original
        geometry if necessary. 8 does what needs to be done at all costs. Defaults to 5.
        **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh with be decomposed into multiple convex components if a single one is not
        sufficient to met the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_nonconvex : bool, optional
        This parameter is deprecated. Please use 'convexify' and 'decompose_(robot|object)_error_threshold' instead.
    decompose_object_error_threshold : bool, optional:
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threashold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : bool, optional:
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threashold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_options : CoacdOptions, optional
        Options for configuring coacd convex decomposition. Needs to be a `gs.options.CoacdOptions` object.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False.
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to True.
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False.
    prioritize_urdf_material : bool, optional
        Sometimes a geom in a urdf file will be assigned a color, and the geom asset file also contains its own visual
        material. This parameter controls whether to prioritize the URDF-defined material over the asset's own material.
        Defaults to False.
    merge_fixed_links : bool, optional
        Whether to merge links connected via a fixed joint. Defaults to True.
    links_to_keep : list of str, optional
        A list of link names that should not be skipped during link merging. Defaults to [].
    default_armature : float, optional
        Default rotor inertia of the actuators. In practice it is applied to all joints regardless of whether they are
        actuated. None to disable. Default to 0.1.
    """

    fixed: bool = False
    prioritize_urdf_material: bool = False
    requires_jac_and_IK: bool = True
    merge_fixed_links: bool = True
    links_to_keep: List[str] = []
    default_armature: Optional[float] = 0.1

    def __init__(self, **data):
        super().__init__(**data)
        if isinstance(self.file, str) and not self.is_format(URDF_FORMAT):
            gs.raise_exception(f"Expected `{URDF_FORMAT}` extension for URDF file: {self.file}")

        # Anisotropic scaling is ill-defined for poly-articulated robots. See related MJCF about this for details.
        if isinstance(self.scale, np.ndarray) and self.scale.std() > gs.EPS:
            if self.scale.std() > gs.EPS:
                gs.raise_exception("Anisotropic scaling is not supported by MJCF morph.")
            self.scale = self.scale.mean()


class Drone(FileMorph):
    """
    Morph loaded from a URDF file for creating a `DroneEntity`.

    Note
    ----
    Visual geom in the propeller links will be used for spinning animation.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly. If a 3-tuple, it scales along
        each axis. Defaults to 1.0. Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention. Defaults to
        (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    decimate : bool, optional
        Whether to decimate (simplify) the mesh. Defaults to True. **This is only used for RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    decimate_aggressiveness : int
        How hard the decimation process will try to match the target number of faces, as a integer ranging from 0 to 8.
        0 is losseless. 2 preserves all features of the original geometry. 5 may significantly alters the original
        geometry if necessary. 8 does what needs to be done at all costs. Defaults to 5.
        **This is only used for RigidEntity.**
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted
        to a set of convex hulls. The mesh with be decomposed into multiple convex components if a single one is not
        sufficient to met the desired accuracy (see 'decompose_(robot|object)_error_threshold' documentation). The
        module 'coacd' is used for this decomposition process. If not given, it defaults to `True` for `RigidEntity`
        and `False` for other deformable entities.
    decompose_nonconvex : bool, optional
        This parameter is deprecated. Please use 'convexify' and 'decompose_(robot|object)_error_threshold' instead.
    decompose_object_error_threshold : bool, optional:
        For basic rigid objects (mug, table...), skip convex decomposition if the relative difference between the
        volume of original mesh and its convex hull is lower than this threashold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to 0.15 (15%).
    decompose_robot_error_threshold : bool, optional:
        For poly-articulated robots, skip convex decomposition if the relative difference between the volume of
        original mesh and its convex hull is lower than this threashold.
        0.0 to enforce decomposition, float("inf") to disable it completely. Defaults to float("inf").
    coacd_options : CoacdOptions, optional
        Options for configuring coacd convex decomposition. Needs to be a `gs.options.CoacdOptions` object.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
    collision : bool, optional
        **NB**: Drone doesn't support collision checking for now.
    prioritize_urdf_material : bool, optional
        Sometimes a geom in a urdf file will be assigned a color, and the geom asset file also contains its own visual
        material. This parameter controls whether to prioritize the URDF-defined material over the asset's own material.
        Defaults to False.
    model : str, optional
        The model of the drone. Defaults to 'CF2X'. Supported models are 'CF2X', 'CF2P', and 'RACE'.
    COM_link_name : str, optional
        This option is deprecated. The true Center of Mass (CoM) will be used instead of requesting the user to manually
        specify the name of the link that represents the center of mass.
    propellers_link_names : sequence of str, optional
        This option is deprecated and will be removed in the future. Please use 'propellers_link_name' instead.
    propellers_link_name : sequence of str, optional
        The names of the links that represent the propellers. Defaults to
        ('prop0_link', 'prop1_link', 'prop2_link', 'prop3_link').
    propellers_spin : sequence of int, optional
        The spin direction of the propellers. 1: CCW, -1: CW. Defaults to (-1, 1, -1, 1).
    merge_fixed_links : bool, optional
        Whether to merge links connected via a fixed joint. Defaults to True.
    links_to_keep : list of str, optional
        A list of link names that should not be skipped during link merging. Defaults to ().
    default_armature : float, optional
        Default rotor inertia of the actuators. In practice it is applied to all joints regardless of whether they are
        actuated. None to disable. Default to 0.1.
    default_base_ang_damping_scale : float, optional
        Default angular damping applied on the floating base that will be rescaled by the total mass.
        None to disable. Default to 1e-5.
    """

    model: str = "CF2X"
    COM_link_name: Optional[str] = None
    prioritize_urdf_material: bool = False
    propellers_link_names: Optional[Sequence[str]] = None
    propellers_link_name: Sequence[str] = ("prop0_link", "prop1_link", "prop2_link", "prop3_link")
    propellers_spin: Sequence[int] = (-1, 1, -1, 1)  # 1: CCW, -1: CW
    merge_fixed_links: bool = True
    links_to_keep: Sequence[str] = ()
    default_armature: Optional[float] = 0.1
    default_base_ang_damping_scale: Optional[float] = 1e-5

    def __init__(self, **data):
        super().__init__(**data)

        if self.COM_link_name is not None:
            gs.logger.warning("Drone option 'COM_link_name' is deprecated and will be ignored.")

        if self.propellers_link_names is not None:
            gs.logger.warning(
                "Drone option 'propellers_link_names' is deprecated and will be remove in future release. Please use "
                "'propellers_link_name' instead."
            )
            self.propellers_link_name = self.propellers_link_names

        # Make sure that Propellers links are preserved
        self.links_to_keep = tuple(set([*self.links_to_keep, *self.propellers_link_name]))

        if isinstance(self.file, str) and not self.is_format(URDF_FORMAT):
            gs.raise_exception(f"Drone only supports `{URDF_FORMAT}` extension: {self.file}")

        if self.model not in ("CF2X", "CF2P", "RACE"):
            gs.raise_exception(f"Unsupported `model`: {self.model}.")


class Terrain(Morph):
    """

    Morph for creating a rigid terrain. This can be instantiated from two choices:
    1) a grid of subterrains generated using the given configurations
    2) a terrain generated using the given height field.

    If randomize is True, subterrain type that involves randomness will have random parameters.
    Otherwise, they will use fixed random seed 0.

    Users can easily configure the subterrain types by specifying the `subterrain_types` parameter.
    If using a single string, it will be repeated for all subterrains. If it's a 2D list, it should have the same shape
    as `n_subterrains`. The supported subterrain types are:

    - 'flat_terrain': flat terrain
    - 'random_uniform_terrain': random uniform terrain
    - 'sloped_terrain': sloped terrain
    - 'pyramid_sloped_terrain': pyramid sloped terrain
    - 'discrete_obstacles_terrain': discrete obstacles terrain
    - 'wave_terrain': wave terrain
    - 'stairs_terrain': stairs terrain
    - 'pyramid_stairs_terrain': pyramid stairs terrain
    - 'stepping_stones_terrain': stepping stones terrain

    Note
    ----
    Rigid terrain will also be represented as SDF for collision checking, but its resolution is auto-computed and
    ignores the value specified in `gs.materials.Rigid()`.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly.
        If a 3-tuple, it scales along each axis. Defaults to 1.0.
        Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision
        purposes. Defaults to True. `visualization` and `collision` cannot both be False.
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True.
        `visualization` and `collision` cannot both be False.
    randomize : bool, optional
        Whether to randomize the subterrains that involve randomness. Defaults to False.
    n_subterrains : tuple of int, optional
        The number of subterrains in x and y directions. Defaults to (3, 3).
    subterrain_size : tuple of float, optional
        The size of each subterrain in meters. Defaults to (12.0, 12.0).
    horizontal_scale : float, optional
        The size of each cell in the subterrain in meters. Defaults to 0.25.
    vertical_scale : float, optional
        The height of each step in the subterrain in meters. Defaults to 0.005.
    uv_scale : float, optional
        The scale of the UV mapping for the terrain. Defaults to 1.0.
    subterrain_types : str or 2D list of str, optional
        The types of subterrains to generate. If a string, it will be repeated for all subterrains.
        If a 2D list, it should have the same shape as `n_subterrains`.
    height_field : array-like, optional
        The height field to generate the terrain. If specified, all other configurations will be ignored.
        Defaults to None.
    name : str, optional
        The name of the terrain to save
    from_stored : str, optional
        The path of the stored terrain to load
    subterrain_parameters : dictionary, optional
        Lets users pick their own subterrain parameters.
    """

    is_free: bool = False
    randomize: bool = False  # whether to randomize the terrain
    n_subterrains: Tuple[int, int] = (3, 3)  # number of subterrains in x and y directions
    subterrain_size: Tuple[float, float] = (12.0, 12.0)  # meter
    horizontal_scale: float = 0.25  # meter size of each cell in the subterrain
    vertical_scale: float = 0.005  # meter height of each step in the subterrain
    uv_scale: float = 1.0
    subterrain_types: Any = [
        ["flat_terrain", "random_uniform_terrain", "stepping_stones_terrain"],
        ["pyramid_sloped_terrain", "discrete_obstacles_terrain", "wave_terrain"],
        ["random_uniform_terrain", "pyramid_stairs_terrain", "sloped_terrain"],
    ]
    height_field: Any = None
    name: str = "default"  # name to store and reuse the terrain
    from_stored: Any = None
    subterrain_parameters: dict[str, dict] | None = None

    def __init__(self, **data):
        custom_params = data.get("subterrain_parameters") or {}
        terrain_types = set(self.default_params) | set(custom_params)
        overwritten_params = {}

        for terrain_type in terrain_types:
            default_value = self.default_params.get(terrain_type, {})
            custom_value = custom_params.get(terrain_type, {})
            overwritten_params[terrain_type] = default_value | custom_value

        data["subterrain_parameters"] = overwritten_params
        super().__init__(**data)

        self._subterrain_parameters = overwritten_params

        supported_subterrain_types = [
            "flat_terrain",
            "fractal_terrain",
            "random_uniform_terrain",
            "sloped_terrain",
            "pyramid_sloped_terrain",
            "discrete_obstacles_terrain",
            "wave_terrain",
            "stairs_terrain",
            "pyramid_stairs_terrain",
            "stepping_stones_terrain",
        ]

        if self.height_field is not None:
            try:
                if np.array(self.height_field).ndim != 2:
                    gs.raise_exception("`height_field` should be a 2D array.")
            except:
                gs.raise_exception("`height_field` should be array-like to be converted to np.ndarray.")

            return

        if isinstance(self.subterrain_types, str):
            subterrain_types = []
            for i in range(self.n_subterrains[0]):
                row = []
                for j in range(self.n_subterrains[1]):
                    row.append(self.subterrain_types)
                subterrain_types.append(row)
            self.subterrain_types = subterrain_types
        else:
            if np.array(self.subterrain_types).shape != (self.n_subterrains[0], self.n_subterrains[1]):
                gs.raise_exception(
                    "`subterrain_types` should be either a string or a 2D list of strings with the same shape as `n_subterrains`."
                )

        for row in self.subterrain_types:
            for subterrain_type in row:
                if subterrain_type not in supported_subterrain_types:
                    gs.raise_exception(
                        f"Unsupported subterrain type: {subterrain_type}, should be one of {supported_subterrain_types}"
                    )

        if not mu.is_approx_multiple(self.subterrain_size[0], self.horizontal_scale) or not mu.is_approx_multiple(
            self.subterrain_size[1], self.horizontal_scale
        ):
            gs.raise_exception("`subterrain_size` should be divisible by `horizontal_scale`.")

    @property
    def default_params(self):
        return {
            "flat_terrain": {},
            "fractal_terrain": {
                "levels": 8,
                "scale": 5.0,
            },
            "random_uniform_terrain": {
                "min_height": -0.1,
                "max_height": 0.1,
                "step": 0.1,
                "downsampled_scale": 0.5,
            },
            "sloped_terrain": {
                "slope": -0.5,
            },
            "pyramid_sloped_terrain": {
                "slope": -0.1,
            },
            "discrete_obstacles_terrain": {
                "max_height": 0.05,
                "min_size": 1.0,
                "max_size": 5.0,
                "num_rects": 20,
            },
            "wave_terrain": {
                "num_waves": 2.0,
                "amplitude": 0.1,
            },
            "stairs_terrain": {
                "step_width": 0.75,
                "step_height": -0.1,
            },
            "pyramid_stairs_terrain": {
                "step_width": 0.75,
                "step_height": -0.1,
            },
            "stepping_stones_terrain": {
                "stone_size": 1.0,
                "stone_distance": 0.25,
                "max_height": 0.2,
                "platform_size": 0.0,
            },
        }

    @property
    def subterrain_params(self):
        return self._subterrain_parameters
