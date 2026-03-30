import enum

# dynamic loading
ACTIVE = 1
INACTIVE = 0


class IntEnum(enum.IntEnum):
    def __repr__(self):
        return f"<gs.{self.__class__.__name__}.{self.name}: {self.value}>"

    def __format__(self, format_spec):
        return f"<{self.name}: {self.value}>"


# geom type in rigid solver
class GEOM_TYPE(IntEnum):
    # Beware PLANE must be the first geometry type as this is assumed by MPR collision detection.
    PLANE = 0
    SPHERE = 1
    ELLIPSOID = 2
    CYLINDER = 3
    CAPSULE = 4
    BOX = 5
    MESH = 6
    TERRAIN = 7


# joint type in rigid solver, ranked by number of dofs
class JOINT_TYPE(IntEnum):
    FIXED = 0
    REVOLUTE = 1
    PRISMATIC = 2
    SPHERICAL = 3
    FREE = 4


class EQUALITY_TYPE(IntEnum):
    CONNECT = 0
    WELD = 1
    JOINT = 2


class CTRL_MODE(IntEnum):
    FORCE = 0
    VELOCITY = 1
    POSITION = 2


######### User accessible constants do not capitalize #########
# rigid solver intergrator
class integrator(IntEnum):
    Euler = 0
    implicitfast = 1
    approximate_implicitfast = 2


# rigid solver constraint solver
class constraint_solver(IntEnum):
    CG = 0
    Newton = 1


# rigid solver broadphase traversal strategy
class broadphase_traversal(IntEnum):
    """
    Strategy for broad-phase collision detection in the rigid solver.

    Broad-phase quickly eliminates geometry pairs that cannot collide before
    the more expensive narrow-phase runs.

    Attributes
    ----------
    SAP : int
        Sweep-and-prune. Sorts axis-aligned bounding boxes along one axis and
        checks sequential overlaps. Required when hibernation or heterogeneous
        entities are enabled.
    ALL_VS_ALL : int
        Iterates over pre-filtered valid geometry pairs in parallel.
        Faster on GPU but uses more memory and does not support hibernation.

    Notes
    -----
    When ``RigidOptions.broadphase_traversal`` is ``None`` (the default), the
    solver selects automatically: ``SAP`` on CPU, ``ALL_VS_ALL`` on GPU.
    """

    SAP = 0
    ALL_VS_ALL = 1


# backend
class backend(IntEnum):
    cpu = 0
    gpu = 1
    cuda = 2
    amdgpu = 3
    metal = 4

    def __format__(self, format_spec):
        return f"gs.{self.name}"


# image types for visualization
class IMAGE_TYPE(IntEnum):
    RGB = 0
    DEPTH = 1
    SEGMENTATION = 2
    NORMAL = 3

    def __format__(self, format_spec):
        return self.name


# parallelize
class PARA_LEVEL(IntEnum):
    NEVER = 0  # when using cpu
    PARTIAL = 1  # when using gpu for non-batched scene
    ALL = 2  # when using gpu for batched scene
