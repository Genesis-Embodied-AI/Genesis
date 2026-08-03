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
    """
    Collision shape of a rigid geom, as reported by `RigidGeom.type`.

    Attributes
    ----------
    PLANE : int
        Half-space bounded by an infinite plane.
    SPHERE : int
        Sphere.
    ELLIPSOID : int
        Ellipsoid, one radius per axis.
    CYLINDER : int
        Cylinder with flat caps.
    CAPSULE : int
        Cylinder with hemispherical caps.
    BOX : int
        Rectangular cuboid.
    MESH : int
        Triangle mesh, convex or decomposed into convex parts.
    TERRAIN : int
        Heightfield on a regular grid.
    """

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
    """
    Kinematic type of a rigid joint, as reported by `RigidJoint.type`, ranked by degree-of-freedom (dof) count.

    Attributes
    ----------
    FIXED : int
        Rigid attachment, 0 dofs.
    REVOLUTE : int
        Hinge, 1 rotational dof.
    PRISMATIC : int
        Slider, 1 translational dof.
    SPHERICAL : int
        Ball joint, 3 rotational dofs.
    FREE : int
        Floating base, 6 dofs.
    """

    FIXED = 0
    REVOLUTE = 1
    PRISMATIC = 2
    SPHERICAL = 3
    FREE = 4


class EQUALITY_TYPE(IntEnum):
    """
    Kind of equality constraint tying two objects together, as reported by `RigidEquality.type`.

    Attributes
    ----------
    CONNECT : int
        Pins two points to the same world position, removing 3 translational dofs.
    WELD : int
        Holds two frames at a fixed relative pose, removing all 6 dofs.
    JOINT : int
        Couples two scalar joints so one follows the other through a degree-4 polynomial in the driving joint's
        position, removing 1 dof.
    """

    CONNECT = 0
    WELD = 1
    JOINT = 2


class CTRL_MODE(IntEnum):
    """
    Control mode of a degree of freedom (dof), set by the control method last called on it.

    Attributes
    ----------
    POSITION : int
        Follows a position target.
    VELOCITY : int
        Follows a velocity target.
    FORCE : int
        Applies a force directly.
    """

    POSITION = 0
    VELOCITY = 1
    FORCE = 2


######### User accessible constants do not capitalize #########
# rigid solver intergrator
class integrator(IntEnum):
    """
    Time integration scheme of the rigid solver. Positions always advance with the velocity the step just produced,
    and every scheme folds joint damping into the effective mass so that a stiff damping force cannot overshoot. They
    differ in what else is folded in, and in when.

    `Euler` and `implicitfast` apply the correction in a second factorization of the mass matrix, performed for the
    entities that need it: those carrying damping, and under `implicitfast` those also carrying actuator bias.
    Setting `enable_mujoco_compatibility` skips that selection and refactors unconditionally.

    Attributes
    ----------
    Euler : int
        Damping only. Standalone free bodies take the plain position update.
    implicitfast : int
        Adds the velocity-actuator bias of every degree of freedom (dof) under position or velocity control, and
        advances standalone free bodies by the implicit midpoint rule.
    approximate_implicitfast : int
        Folds the same two corrections into the mass matrix as it is built, so one factorization serves the step. The
        correction then also reaches the accelerations produced by constraints and external forces, which it does not
        model, in exchange for never factorizing twice.
    """

    Euler = 0
    implicitfast = 1
    approximate_implicitfast = 2


# rigid solver constraint solver
class constraint_solver(IntEnum):
    """
    Numerical method that solves the constraint system of the rigid solver.

    Attributes
    ----------
    CG : int
        Preconditioned conjugate gradient. Each iteration costs a matrix-vector product rather than a factorization,
        and the solve needs more of them to converge.
    Newton : int
        Newton steps on the explicit Hessian, each a Cholesky factorization. Converges in a handful of iterations,
        every one of them paying for that factorization.
    """

    CG = 0
    Newton = 1


# rigid solver contact friction cone
class friction_cone(IntEnum):
    """
    Contact friction cone model, trading numerical robustness for physical accuracy. Prefer `pyramidal` for
    robustness; choose `elliptic` when isotropic friction or firm static friction matters - e.g. objects that must
    stay put at rest instead of slowly creeping.

    Attributes
    ----------
    pyramidal : int
        Approximates the friction cone by a pyramid: robust and easy to solve, at the price of anisotropic friction,
        whose effective limit depends on the sliding direction.
    elliptic : int
        The exact cone: friction is isotropic and bounded by its true Euclidean limit sqrt(f_t1^2 + f_t2^2) <= mu * f_n
        in every direction, and with a high `impratio` it holds resting stacks without the slow tangential creep of
        regularized friction, in return for being harder to solve and more sensitive numerically.
    """

    pyramidal = 0
    elliptic = 1


# rigid solver contact resolution
class contact_resolution(IntEnum):
    """
    How a contact's normal force and friction force are resolved against each other. Prefer `signorini` whenever
    sliding contact matters; choose `convex` for parity with engines built on that formulation, or if a stiff scene
    converges better under it.

    Attributes
    ----------
    convex : int
        Poses the whole contact as a single smooth convex cost and lets the solver trade the normal residual against
        the tangential one. Because the friction limit mu * f_n bounds the pair jointly, a contact sliding fast enough
        that its friction rows demand more force than the cone allows can be answered by raising f_n instead: a body
        launched horizontally then lifts off a flat floor, by more the faster it slides. In exchange the whole problem
        stays one convex program, which converges predictably on stiff articulated chains and high mass ratios.
    signorini : int
        Bounds friction against the normal force the contact has actually developed, so that force is set by the
        contact's own normal state rather than by tangential demand, and sliding can never inflate it - a sliding body
        decelerates at mu * g and stays down at any speed. Contacts are resolved by successive approximation, costing
        extra solver iterations and giving up the single-convex-program guarantee.

    Notes
    -----
    `signorini` requires the elliptic friction cone, whose rows separate into a normal row and a friction disc - the
    pyramidal cone mixes the normal direction into every row and admits no such split - and the Newton constraint
    solver, the only one that reaches the fixed point of the resulting successive approximation. It implements the
    Coulomb complementarity problem eq. (C.22) of Alexis Duburcq, "Learning and Optimization of the Locomotion with an
    Exoskeleton for Paraplegic People", PhD thesis, Universite Paris Sciences et Lettres, 2022 (HAL tel-04166955),
    Appendix C, whose Signorini condition is what forbids the normal force from absorbing tangential demand.
    """

    convex = 0
    signorini = 1


# rigid solver broadphase traversal strategy
class broadphase_traversal(IntEnum):
    """
    Search strategy of the broad phase of collision detection in the rigid solver, which discards the geom pairs that
    cannot collide before the more expensive narrow phase runs.

    The pairs that can never collide are filtered out once at build time, such as those sharing a link, made of two
    fixed geoms, or mismatched on contype / conaffinity. That leaves the valid pairs, up to O(n_geoms^2) of them and
    typically far fewer, which the strategies below search differently at every step.

    Attributes
    ----------
    SAP : int
        Sweep-and-prune. Sorts the geom axis-aligned bounding boxes (AABBs) along one axis in O(n_geoms log n_geoms),
        then checks only the pairs overlapping on that axis, for a cost per step of O(n_geoms log n_geoms + k) in the
        number k of such pairs, typically far below the full set of valid pairs. The sort and the sweep are
        single-threaded, which uses GPU cores poorly.
    ALL_VS_ALL : int
        Tests the AABBs of every valid pair at every step, dispatched in parallel across GPU threads, for a cost per
        step of O(n_valid_pairs). Efficient on GPU while the pair count stays moderate, and expensive in scenes with
        many geoms, where that count grows quadratically. Requires a scene free of hibernation and of heterogeneous
        entities.

    Notes
    -----
    `RigidOptions.broadphase_traversal` defaults to ``None``, which selects `ALL_VS_ALL` on a GPU backend, and
    `SAP` on the CPU backend, where the sequential sweep is efficient, and whenever hibernation or heterogeneous
    entities rule `ALL_VS_ALL` out.
    """

    SAP = 0
    ALL_VS_ALL = 1


# backend
class backend(IntEnum):
    """
    Compute backend the simulation runs on, selected with ``gs.init(backend=...)``. `gs.backend` holds the resolved
    value once initialization returns.

    Attributes
    ----------
    cpu : int
        The host processor.
    gpu : int
        The first of `cuda`, `amdgpu` and `metal` available on the machine, falling back to `cpu` with a
        warning when none is.
    cuda : int
        NVIDIA GPU.
    amdgpu : int
        AMD GPU through ROCm.
    metal : int
        Apple GPU.
    """

    cpu = 0
    gpu = 1
    cuda = 2
    amdgpu = 3
    metal = 4

    def __format__(self, format_spec):
        return f"gs.{self.name}"


# image types for visualization
class IMAGE_TYPE(IntEnum):
    """
    Image channel a camera renders, in the order `Camera.render` returns them.

    Attributes
    ----------
    RGB : int
        Color, one uint8 per channel.
    DEPTH : int
        Distance from the camera along the view direction, in meters.
    SEGMENTATION : int
        Integer index per pixel, at the level set by `VisOptions.segmentation_level`.
    NORMAL : int
        Surface normal at the visible point.
    """

    RGB = 0
    DEPTH = 1
    SEGMENTATION = 2
    NORMAL = 3

    def __format__(self, format_spec):
        return self.name


# parallelize
class PARA_LEVEL(IntEnum):
    """
    Extent to which the solvers parallelize their loops, selected from the backend and the environment count, and
    overridable through the ``GS_PARA_LEVEL`` environment variable.

    Attributes
    ----------
    NEVER : int
        Runs every loop sequentially, as the `cpu` backend does, where parallelism only pays off from as many
        environments as there are threads.
    PARTIAL : int
        Parallelizes the loops whose size fills the device, as a GPU backend does for a scene holding one environment.
    ALL : int
        Parallelizes every loop, as a GPU backend does for a batched scene.
    """

    NEVER = 0
    PARTIAL = 1
    ALL = 2
