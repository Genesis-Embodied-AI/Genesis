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

    At init time, geometry pairs that can never collide are filtered out
    (same-link, fixed-vs-fixed, contype/conaffinity mismatch, etc.), producing
    a list of *valid pairs*.  The number of valid pairs can be up to
    O(n_geoms^2) but is typically much smaller after filtering.  The two
    strategies differ in how they search these valid pairs each step:

    Attributes
    ----------
    SAP : int
        Sweep-and-prune.  Sorts geometry AABBs along one axis
        (O(n_geoms log n_geoms)) then only checks pairs that overlap on that
        axis.  The sort and sweep are single-threaded, which utilizes GPU
        cores poorly. However the cost per step is only O(n_geoms log n_geoms + k)
        where k is the number of axis-overlapping pairs — typically much less than
        the full set of valid pairs.
    ALL_VS_ALL : int
        Checks every valid pair every step (AABB overlap test), dispatching
        them in parallel across GPU threads.  Cost per step is O(n_valid_pairs)
        which is efficient on GPU when the pair count is moderate, but becomes
        expensive in scenes with many geometries since the valid pair count
        grows quadratically. Does not support hibernation or heterogeneous
        entities at this time.

    Notes
    -----
    ``RigidOptions.broadphase_traversal`` defaults to ``None``, which lets the
    solver choose automatically:

    - **CPU backend** → ``SAP`` (sequential sweep is efficient on CPU).
    - **GPU backend** → ``ALL_VS_ALL`` (parallel pair checking is faster).
    - **GPU with hibernation or heterogeneous entities** → ``SAP``
      (``ALL_VS_ALL`` is not compatible with these features).
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
