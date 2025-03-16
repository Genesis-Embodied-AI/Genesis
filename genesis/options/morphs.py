import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.misc as mu

from .misc import CoacdOptions
from .options import Options

"""
We define all types of morphologies here: shape primitives, meshes, URDF, MJCF, and soft robot description files.
These are independent of backend solver type and are shared by different solvers. E.g. a mesh can be either loaded as a rigid object / MPM object / FEM object.
"""


@gs.assert_initialized
class Morph(Options):
    """
    This is the base class for all genesis morphs.
    A morph in genesis is a hybrid concept, encapsulating both the geometry and pose information of an entity. This includes shape primitives, meshes, URDF, MJCF, Terrain, and soft robot description files.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision purposes. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False. **This is only used for RigidEntity.**
    is_free : bool, optional
        Whether the entity is free to move. Defaults to True. **This is only used for RigidEntity.**
    """

    pos: tuple = (0.0, 0.0, 0.0)
    euler: Optional[tuple] = (0.0, 0.0, 0.0)
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

        if self.euler is not None:
            if self.quat is None:
                self.quat = tuple(gs.utils.geom.xyz_to_quat(np.array(self.euler)))

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
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision purposes. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False. **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False. **This is only used for RigidEntity.**
    """

    # Rigid specific
    fixed: bool = False


class Box(Primitive):
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
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    lower : tuple, shape (3,), optional
        The lower corner of the box in meters. Defaults to None.
    upper : tuple, shape (3,), optional
        The upper corner of the box in meters. Defaults to None.
    size : tuple, shape (3,), optional
        The size of the box in meters. Defaults to None.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision purposes. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False. **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False. **This is only used for RigidEntity.**
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


class Cylinder(Primitive):
    """
    Morph defined by a cylinder shape.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    height : float, optional
        The height of the cylinder in meters. Defaults to 1.0.
    radius : float, optional
        The radius of the cylinder in meters. Defaults to 0.5.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision purposes. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False. **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False. **This is only used for RigidEntity.**
    """

    height: float = 1.0
    radius: float = 0.5


class Sphere(Primitive):
    """
    Morph defined by a sphere shape.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    radius : float, optional
        The radius of the sphere in meters. Defaults to 0.5.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision purposes. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False. **This is only used for RigidEntity.**
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False. **This is only used for RigidEntity.**
    """

    radius: float = 0.5


class Plane(Primitive):
    """
    Morph defined by a plane shape.

    Note
    ----
    Plane is a primitive with infinite size. Note that the `pos` is the center of the plane, but essetially only defines a point where the plane passes through.

    Parameters
    ----------
    pos : tuple, shape (3,), optional
        The center position of the plane in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    normal : tuple, shape (3,), optional
        The normal normal of the plane in its local frame. Defaults to (0, 0, 1).
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision purposes. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
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
        The scaling factor for the size of the entity. If a float, it scales uniformly. If a 3-tuple, it scales along each axis. Defaults to 1.0. Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will each be converted to a convex hull. If not given, it defaults to `True` for `RigidEntity` and `False` for other deformable entities.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision purposes. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False. **This is only used for RigidEntity.**
    """

    file: Any = ""
    scale: Union[float, tuple] = 1.0
    convexify: Optional[bool] = None
    recompute_inertia: bool = False

    def __init__(self, **data):
        super().__init__(**data)

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


class Mesh(FileMorph):
    """
    Morph loaded from a mesh file.

    Note
    ----
    In order to speed up simulation, the loaded mesh will first be decimated (simplified) to a target number of faces, followed by convexification (for collision mesh only). Such process can be disabled by setting `decimate` and `convexify` to False.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly. If a 3-tuple, it scales along each axis. Defaults to 1.0. Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will be converted to a convex hull. If not given, it defaults to `True` for `RigidEntity` and `False` for other deformable entities.
    decompose_nonconvex : bool, optional
        Whether to decompose meshes into convex components, if input mesh is nonconvex and `convexify=False`. We use coacd for this decomposition process. If not given, it defaults to `True` for `RigidEntity` and `False` for other deformable entities.
    coacd_options : CoacdOptions, optional
        Options for configuring coacd convex decomposition. Needs to be a `gs.options.CoacdOptions` object.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision purposes. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True. `visualization` and `collision` cannot both be False. **This is only used for RigidEntity.**
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to False. **This is only used for RigidEntity.**
    parse_glb_with_trimesh : bool, optional
        Whether to use trimesh to load glb files. Defaults to False, in which case pygltflib will be used.
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False. **This is only used for RigidEntity.**
    group_by_material : bool, optional
        Whether to group submeshes by their visual material type defined in the asset file. Defaults to True. **This is only used for RigidEntity.**
    merge_submeshes_for_collision : bool, optional
        Whether to merge submeshes for collision. Defaults to True. **This is only used for RigidEntity.**
    decimate : bool, optional
        Whether to decimate (simplify) the mesh. Defaults to True. **This is only used for RigidEntity.**
    decimate_face_num : int, optional
        The number of faces to decimate to. Defaults to 500. **This is only used for RigidEntity.**
    order : int, optional
        The order of the FEM mesh. Defaults to 1. **This is only used for FEMEntity.**
    mindihedral : int, optional
        The minimum dihedral angle in degrees during tetraheralization. Defaults to 10. **This is only used for Volumetric Entity that requires tetraheralization.**
    minratio : float, optional
        The minimum tetrahedron quality ratio during tetraheralization. Defaults to 1.1. **This is only used for Volumetric Entity that requires tetraheralization.**
    nobisect : bool, optional
        Whether to disable bisection during tetraheralization. Defaults to True. **This is only used for Volumetric Entity that requires tetraheralization.**
    quality : bool, optional
        Whether to improve quality during tetraheralization. Defaults to True. **This is only used for Volumetric Entity that requires tetraheralization.**
    verbose : int, optional
        The verbosity level during tetraheralization. Defaults to 0. **This is only used for Volumetric Entity that requires tetraheralization.**
    force_retet : bool, optional
        Whether to force re-tetraheralization. Defaults to False. **This is only used for Volumetric Entity that requires tetraheralization.**

    """

    parse_glb_with_trimesh: bool = False

    # Rigid specific
    fixed: bool = False
    group_by_material: bool = True
    merge_submeshes_for_collision: bool = True
    decimate: bool = True
    decimate_face_num: int = 500
    decompose_nonconvex: Optional[bool] = None
    coacd_options: Optional[CoacdOptions] = None

    # FEM specific
    order: int = 1

    # Volumetric mesh entity
    mindihedral: int = 10
    minratio: float = 1.1
    nobisect: bool = True
    quality: bool = True
    verbose: int = 0

    force_retet: bool = False

    def __init__(self, **data):
        super().__init__(**data)
        if self.decimate and self.decimate_face_num < 100:
            gs.raise_exception(
                "`decimate_face_num` should be greater than 100 to ensure sufficient geometry details are preserved."
            )

        if self.coacd_options is None:
            self.coacd_options = CoacdOptions()


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
    MJCF file always contains a worldbody, which we will skip during loading. The robots/objects in MJCF come with their own baselink pose. If `pos`, `euler`, or `quat` is specified, it will override the baselink pose in the MJCF file.

    The current version of Genesis asumes there's only one child of the worldbody. However, it's possible that a MJCF file contains a scene, not just a single robot, in which case the worldbody will have multiple kinematic trees. We will support such cases in the future.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly. If a 3-tuple, it scales along each axis. Defaults to 1.0. Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity's baselink in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity's baselink in degrees. This follows scipy's extrinsic x-y-z rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity's baselink. If specified, `euler` will be ignored. Defaults to None.
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will be converted to a convex hull. If not given, it defaults to `True` for `RigidEntity` and `False` for other deformable entities.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision purposes. Defaults to True. `visualization` and `collision` cannot both be False.
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True. `visualization` and `collision` cannot both be False.
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to True.
    """

    pos: Optional[tuple] = None
    euler: Optional[tuple] = None
    quat: Optional[tuple] = None
    requires_jac_and_IK: bool = True

    def __init__(self, **data):
        super().__init__(**data)
        if not self.file.endswith(".xml"):
            gs.raise_exception(f"Expected `.xml` extension for MJCF file: {self.file}")


class URDF(FileMorph):
    """
    Morph loaded from a URDF file. This morph only supports `RigidEntity`. If you need to create a `Drone` entity, use `gs.morphs.Drone` instead.

    Note
    ----
    As part of performance optimization, links connected via a fixed joint are merged if `merge_fixed_links` is True. This is turned on by default, and can help improve simulation speed without affecting any dynamics and rendering behaviors.
    However, in cases where certain links are still needed as independent links, such as virtual end-effector links created for being used as IK targets, these links will not be merged if their names are added to `links_to_keep`.
    You can also completely turn off link merging by setting `merge_fixed_links` to False, but it's recommended to use `merge_fixed_links=True` in combination with `links_to_keep` for better performance.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly. If a 3-tuple, it scales along each axis. Defaults to 1.0. Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will be converted to a convex hull. If not given, it defaults to `True` for `RigidEntity` and `False` for other deformable entities.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision purposes. Defaults to True. `visualization` and `collision` cannot both be False.
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True. `visualization` and `collision` cannot both be False.
    requires_jac_and_IK : bool, optional
        Whether this morph, if created as `RigidEntity`, requires jacobian and inverse kinematics. Defaults to True.
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False.
    prioritize_urdf_material : bool, optional
        Sometimes a geom in a urdf file will be assigned a color, and the geom asset file also contains its own visual material. This parameter controls whether to prioritize the URDF-defined material over the asset's own material. Defaults to False.
    merge_fixed_links : bool, optional
        Whether to merge links connected via a fixed joint. Defaults to True.
    links_to_keep : list of str, optional
        A list of link names that should not be skipped during link merging. Defaults to [].
    """

    fixed: bool = False
    prioritize_urdf_material: bool = False
    requires_jac_and_IK: bool = True
    merge_fixed_links: bool = True
    links_to_keep: List[str] = []

    def __init__(self, **data):
        super().__init__(**data)
        if isinstance(self.file, str) and not self.file.endswith(".urdf"):
            gs.raise_exception(f"Expected `.urdf` extension for URDF file: {self.file}")


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
        The scaling factor for the size of the entity. If a float, it scales uniformly. If a 3-tuple, it scales along each axis. Defaults to 1.0. Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    euler : tuple, shape (3,), optional
        The euler angle of the entity in degrees. This follows scipy's extrinsic x-y-z rotation convention. Defaults to (0.0, 0.0, 0.0).
    quat : tuple, shape (4,), optional
        The quaternion (w-x-y-z convention) of the entity. If specified, `euler` will be ignored. Defaults to None.
    convexify : bool, optional
        Whether to convexify the entity. When convexify is True, all the meshes in the entity will be converted to a convex hull. If not given, it defaults to `True` for `RigidEntity` and `False` for other deformable entities.
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision purposes. Defaults to True. `visualization` and `collision` cannot both be False.
    collision : bool, optional
        **NB**: Drone doesn't support collision checking for now.
    fixed : bool, optional
        Whether the baselink of the entity should be fixed. Defaults to False.
    prioritize_urdf_material : bool, optional
        Sometimes a geom in a urdf file will be assigned a color, and the geom asset file also contains its own visual material. This parameter controls whether to prioritize the URDF-defined material over the asset's own material. Defaults to False.
    model : str, optional
        The model of the drone. Defaults to 'CF2X'. Supported models are 'CF2X', 'CF2P', and 'RACE'.
    COM_link_name : str, optional
        The name of the link that represents the center of mass. Defaults to 'center_of_mass_link'.
    propellers_link_names : list of str, optional
        The names of the links that represent the propellers. Defaults to ['prop0_link', 'prop1_link', 'prop2_link', 'prop3_link'].
    propellers_spin : list of int, optional
        The spin direction of the propellers. 1: CCW, -1: CW. Defaults to [-1, 1, -1, 1].
    """

    model: str = "CF2X"
    fixed: bool = False
    prioritize_urdf_material: bool = False
    COM_link_name: str = "center_of_mass_link"
    propellers_link_names: List[str] = ["prop0_link", "prop1_link", "prop2_link", "prop3_link"]
    propellers_spin: List[int] = [-1, 1, -1, 1]  # 1: CCW, -1: CW

    def __init__(self, **data):
        super().__init__(**data)
        if isinstance(self.file, str) and not self.file.endswith(".urdf"):
            gs.raise_exception(f"Drone only supports `.urdf` extension: {self.file}")

        if self.model not in ["CF2X", "CF2P", "RACE"]:
            gs.raise_exception(f"Unsupported `model`: {self.model}.")


class Terrain(Morph):
    """

    Morph for creating a rigid terrain. This can be instantiated from two choices: 1) a grid of subterrains generated using the given configurations, 2) a terrain generated using the given height field.

    If randomize is True, subterrain type that involves randomness will have random parameters. Otherwise, they will use fixed random seed 0.

    Users can easily configure the subterrain types by specifying the `subterrain_types` parameter. If using a single string, it will be repeated for all subterrains. If it's a 2D list, it should have the same shape as `n_subterrains`. The supported subterrain types are:

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
    Rigid terrain will also be represented as SDF for collision checking, but its resolution is auto-computed and ignores the value specified in `gs.materials.Rigid()`.

    Parameters
    ----------
    file : str
        The path to the file.
    scale : float or tuple, optional
        The scaling factor for the size of the entity. If a float, it scales uniformly. If a 3-tuple, it scales along each axis. Defaults to 1.0. Note that 3-tuple scaling is only supported for `gs.morphs.Mesh`.
    pos : tuple, shape (3,), optional
        The position of the entity in meters. Defaults to (0.0, 0.0, 0.0).
    visualization : bool, optional
        Whether the entity needs to be visualized. Set it to False if you need a invisible object only for collision purposes. Defaults to True. `visualization` and `collision` cannot both be False.
    collision : bool, optional
        Whether the entity needs to be considered for collision checking. Defaults to True. `visualization` and `collision` cannot both be False.
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
    subterrain_types : str or 2D list of str, optional
        The types of subterrains to generate. If a string, it will be repeated for all subterrains. If a 2D list, it should have the same shape as `n_subterrains`.
    height_field : array-like, optional
        The height field to generate the terrain. If specified, all other configurations will be ignored. Defaults to None.
    name : str, optional
        The name of the terrain to save
    from_stored : str, optional
        The path of the stored terrain to load
    """

    is_free: bool = False
    randomize: bool = False  # whether to randomize the terrain
    n_subterrains: Tuple[int, int] = (3, 3)  # number of subterrains in x and y directions
    subterrain_size: Tuple[float, float] = (12.0, 12.0)  # meter
    horizontal_scale: float = 0.25  # meter size of each cell in the subterrain
    vertical_scale: float = 0.005  # meter height of each step in the subterrain
    subterrain_types: Any = [
        ["flat_terrain", "random_uniform_terrain", "stepping_stones_terrain"],
        ["pyramid_sloped_terrain", "discrete_obstacles_terrain", "wave_terrain"],
        ["random_uniform_terrain", "pyramid_stairs_terrain", "sloped_terrain"],
    ]
    height_field: Any = None
    name: str = "default"  # name to store and reuse the terrain
    from_stored: Any = None

    def __init__(self, **data):
        super().__init__(**data)

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
