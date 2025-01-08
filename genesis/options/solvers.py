from typing import Optional

import numpy as np

import genesis as gs

from .options import Options

############################ Top level: simulator and coupler ############################
"""
Simulator options specifies the global settings for the simulator and the coupler options specifies whether the coupling between pairs of solvers is enabled.
"""


class SimOptions(Options):
    """
    Options configuring the top-level simulator.

    Note
    ----
    1. `SimOptions` specifies the global settings for the simulator. Some parameters exist both in `SimOptions` and `SolverOptions`. In this case, if such parameters are given in `SolverOptions`, it will override the one specified in `SimOptions` for this specific solver. For example, if `dt` is only given in `SimOptions`, it will be shared by all the solvers, but it's also possible to let a solver run at a different temporal speed by setting its own `dt` to be a different value.

    2. In differentiable mode, `substeps_local` must be divisible by `substeps`, as external command is input per `step`, but `substep`. If `requires_grad` is False, we can use arbitrary `substeps_local`.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. Defaults to 1e-2.
    substeps : int, optional
        Number of substeps per simulation step. Defaults to 1.
    substeps_local : int, optional
        Number of substeps stored in GPU memory. Defaults to None. This is used for differentiable mode.
    gravity : tuple, optional
        Gravity force in N/kg. Defaults to (0.0, 0.0, -9.81).
    floor_height : float, optional
        Height of the floor in meters. Defaults to 0.0.
    requires_grad : bool, optional
        Whether to enable differentiable mode. Defaults to False.
    """

    dt: float = 1e-2
    substeps: int = 1
    substeps_local: Optional[int] = None  # number of substeps stored in GPU memory
    gravity: tuple = (0.0, 0.0, -9.81)
    floor_height: float = 0.0
    requires_grad: bool = False

    # not set by user
    _steps_local: Optional[int] = None

    def __init__(self, **data):
        super().__init__(**data)
        if self.substeps_local is None:
            if self.requires_grad:
                self.substeps_local = self.substeps
            else:
                # use 1 to save gpu memory
                self.substeps_local = 1

        if self.requires_grad:
            if self.substeps_local % self.substeps != 0:
                gs.raise_exception("`substeps_local` must be divisible by `substeps` when `requires_grad` is True.")
            else:
                self._steps_local = int(self.substeps_local / self.substeps)
        else:
            self._steps_local = None


class CouplerOptions(Options):
    """
    Options configuring the inter-solver coupling.

    Parameters
    ----------
    rigid_mpm : bool, optional
        Whether to enable coupling between rigid and MPM solvers. Defaults to True.
    rigid_sph : bool, optional
        Whether to enable coupling between rigid and SPH solvers. Defaults to True.
    rigid_pbd : bool, optional
        Whether to enable coupling between rigid and PBD solvers. Defaults to True.
    rigid_fem : bool, optional
        Whether to enable coupling between rigid and FEM solvers. Defaults to True.
    mpm_sph : bool, optional
        Whether to enable coupling between MPM and SPH solvers. Defaults to True.
    mpm_pbd : bool, optional
        Whether to enable coupling between MPM and PBD solvers. Defaults to True.
    fem_mpm : bool, optional
        Whether to enable coupling between FEM and MPM solvers. Defaults to True.
    fem_sph : bool, optional
        Whether to enable coupling between FEM and SPH solvers. Defaults to True.
    """

    rigid_mpm: bool = True
    rigid_sph: bool = True
    rigid_pbd: bool = True
    rigid_fem: bool = True
    mpm_sph: bool = True
    mpm_pbd: bool = True
    fem_mpm: bool = True
    fem_sph: bool = True


############################ Solvers inside simulator ############################
"""
Parameters in these solver-specific options will override SimOptions if available.
"""


class ToolOptions(Options):
    """
    Options configuring the ToolSolver.

    Note
    ----
    ToolEntity is a simplified form of RigidEntity. It supports one way tool->other coupling, but has *no* internal dynamics and can only be created from a single mesh. This is a temporal workaround for differentiable rigid-soft interaction. This solver will be removed once differentiable mode is supported by the RigidSolver.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. Defaults to 1e-2.
    floor_height : float, optional
        Height of the floor in meters. Defaults to 0.0.
    """

    dt: Optional[float] = None
    floor_height: float = None


class RigidOptions(Options):
    """
    Options configuring the RigidSolver.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    gravity : tuple, optional
        Gravity force in N/kg. If none, it will inherit from `SimOptions`. Defaults to None.
    enable_collision : bool, optional
        Whether to enable collision detection. Defaults to True.
    enable_joint_limit : bool, optional
        Whether to enable joint limit. Defaults to True.
    enable_self_collision : bool, optional
        Whether to enable self collision within each entity. Defaults to False.
    max_collision_pairs : int, optional
        Maximum number of collision pairs. Defaults to 100.
    integrator : gs.integrator, optional
        Integrator type. Current supported integrators are 'gs.integrator.Euler', 'gs.integrator.implicitfast' and 'gs.integrator.approximate_implicitfast'. Defaults to 'approximate_implicitfast'.
    IK_max_targets : int, optional
        Maximum number of IK targets. Increasing this doesn't affect IK solving speed, but will increase memory usage. Defaults to 6.
    constraint_solver : gs.constraint_solver, optional
        Constraint solver type. Current supported constraint solvers are 'gs.constraint_solver.CG' (conjugate gradient) and 'gs.constraint_solver.Newton' (Newton's method). Defaults to 'CG'.
    iterations : int, optional
        Number of iterations for the constraint solver. Defaults to 100.
    tolerance : float, optional
        Tolerance for the constraint solver. Defaults to 1e-5.
    ls_iterations : int, optional
        Number of line search iterations for the constraint solver. Defaults to 50.
    ls_tolerance : float, optional
        Tolerance for the line search. Defaults to 1e-2.
    sparse_solve : bool, optional
        Whether to exploit sparsity in the constraint system. Defaults to False.
    contact_resolve_time : float, optional
        Time to resolve a contact. The smaller the value, the more stiff the constraint. Defaults to 0.02. (called timeconst in https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters)
    use_contact_island : bool, optional
        Whether to use contact island to speed up contact resolving. Defaults to False.
    use_hibernation : bool, optional
        Whether to enable hibernation. Defaults to False.
    hibernation_thresh_vel : float, optional
        Velocity threshold for hibernation. Defaults to 1e-3.
    hibernation_thresh_acc : float, optional
        Acceleration threshold for hibernation. Defaults to 1e-2.

    Warning
    -------
    Hibernation hasn't been robustly tested and will be fully supported soon.
    """

    dt: Optional[float] = None
    gravity: Optional[tuple] = None
    enable_collision: bool = True
    enable_joint_limit: bool = True
    enable_self_collision: bool = False
    max_collision_pairs: int = 100
    integrator: gs.integrator = gs.integrator.approximate_implicitfast
    IK_max_targets: int = 6

    # batching info
    batch_links_info: Optional[bool] = False
    batch_dofs_info: Optional[bool] = False

    # constraint solver
    constraint_solver: gs.constraint_solver = gs.constraint_solver.CG
    iterations: int = 100
    tolerance: float = 1e-5
    ls_iterations: int = 50
    ls_tolerance: float = 1e-2
    sparse_solve: bool = False
    contact_resolve_time: Optional[float] = None
    use_contact_island: bool = False
    box_box_detection: bool = (
        False  # collision detection branch for box-box pair, slower but more stable. (Follows mujoco's implementation: https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_collision_box.c)
    )

    # hibernation threshold
    use_hibernation: bool = False
    hibernation_thresh_vel: float = 1e-3
    hibernation_thresh_acc: float = 1e-2

    def __init__(self, **data):
        super().__init__(**data)


class AvatarOptions(Options):
    """
    Options configuring the AvatarSolver. AvatarEntity is similar to RigidEntity, but without internal physics.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    enable_collision : float, optional
        Whether to enable collision detection. Defaults to False.
    enable_self_collision : float, optional
        Whether to enable self collision within each entity. Defaults to False.
    max_collision_pairs : int, optional
        Maximum number of collision pairs. Defaults to 100.
    IK_max_targets : int, optional
        Maximum number of IK targets. Increasing this doesn't affect IK solving speed, but will increase memory usage. Defaults to 6.
    """

    dt: Optional[float] = None
    enable_collision: bool = False
    enable_self_collision: bool = False
    max_collision_pairs: int = 100
    IK_max_targets: int = 6  # Increasing this doesn't affect IK solving speed, but will increase memory usage


class MPMOptions(Options):
    """
    Options configuring the MPMSolver.

    Note
    ----
    MPM is a hybrid lagrangian-eulerian method for simulating soft materials. In the eulerian phase, it uses a grid representation. The `upper_bound` and `lower_bound` specify the simulation domain, but a safety padding will be added to the actual grid boundary. Therefore, the actual boundary could be slightly tighter than the specified one. Note that the size of the domain affects the performance of the simulation, hence you should set it as tight as possible.

    `use_sparse_grid` and `leaf_block_size` are advanced parameters for sparse computation. Don't touch them unless you know what you are doing.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    gravity : tuple, optional
        Gravity force in N/kg. If none, it will inherit from `SimOptions`. Defaults to None.
    particle_size : float, optional
        Particle diameter in meters. If not given, we will compute `particle_size` based on `grid_density`, where `particle_size` will be linearly proportional to the grid cell size. A reference value is `particle_size = 0.01` for `grid_density = 64`. Defaults to None.
    grid_density : float, optional
        Number of grid cells per meter. Defaults to 64.
    enable_CPIC : bool, optional
        Whether to enable CPIC (Compatible Particle-in-Cell) to support coupling with thin objects. Defaults to False.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-1.0, -1.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (1.0, 1.0, 1.0).
    use_sparse_grid : bool, optional
        Whether to use sparse grid. Defaults to False. Don't touch unless you know what you are doing.
    leaf_block_size : int, optional
        Size of the leaf block for sparse mode. Defaults to 8.
    """

    dt: Optional[float] = None
    gravity: Optional[tuple] = None
    particle_size: Optional[float] = None  # in meters. Will be computed automatically if it's None.
    grid_density: float = 64
    enable_CPIC: bool = False

    # These will later be converted to discrete grid bound. The actual grid boundary could be slightly tighter.
    lower_bound: tuple = (-1.0, -1.0, 0.0)
    upper_bound: tuple = (1.0, 1.0, 1.0)

    # Sparse computation parameter. Don't touch unless you know what you are doing.
    use_sparse_grid: bool = False
    leaf_block_size: int = (
        8  # NOTE: taichi_elements uses 4, which in our case will hang and crash. Probably due to some memory access issue.
    )

    def __init__(self, **data):
        super().__init__(**data)
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")

        if self.particle_size is None:
            self.particle_size = 0.01 * 64.0 / self.grid_density


class SPHOptions(Options):
    """
    Options configuring the SPHSolver.

    Note
    ----
    If spatial hashing parameters are not given, we will compute them automatically this way: For `hash_grid_cell_size`, we will set it to be the `support_radius`, which is essentially 2 * `particle_size`. For `hash_grid_res`, if a small bound is given, it's used for the hash grid; otherwise, we use a default value of a 150^3 cube. Any grid bigger than that will results in too many cells hence not ideal.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    gravity : tuple, optional
        Gravity force in N/kg. If none, it will inherit from `SimOptions`. Defaults to None.
    particle_size : float, optional
        Particle diameter in meters. Defaults to 0.02.
    pressure_solver : str, optional
        Pressure solver type. Current supported pressure solvers are 'WCSPH' and 'DFSPH'. Defaults to 'WCSPH'.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-100.0, -100.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (100.0, 100.0, 100.0).
    hash_grid_res : tuple, optional
        Size of the spatially-repetitive spatial hashing grid in meters. If none, it will be computed automatically. Defaults to None.
    hash_grid_cell_size : float, optional
        Size of the lattic cell of the spatial hashing grid in meters. This should be at least 2 * `particle_size`. If none, it will be computed automatically. Defaults to None.
    max_divergence_error : float, optional
        Maximum divergence error for DFSPH. Defaults to 0.1.
    max_density_error_percent : float, optional
        Maximum density error *percent* for DFSPH, so 0.1 means 0.1%. Defaults to 0.05.
    max_divergence_solver_iterations : int, optional
        Maximum number of iterations for the divergence solver. Defaults to 100.
    max_density_solver_iterations : int, optional
        Maximum number of iterations for the density solver. Defaults to 100.
    """

    dt: Optional[float] = None
    gravity: Optional[tuple] = None
    particle_size: float = 0.02
    pressure_solver: str = "WCSPH"  # 'WCSPH' or 'DFSPH'

    lower_bound: tuple = (-100.0, -100.0, 0.0)
    upper_bound: tuple = (100.0, 100.0, 100.0)

    # spatial hashing
    hash_grid_res: Optional[tuple] = None  # size of the spatially-repetitive hash grid in meters
    hash_grid_cell_size: Optional[float] = None  # size of the cubic cell in meters

    # DFSPH parameters
    max_divergence_error: float = 0.1
    max_density_error_percent: float = 0.05  # This is percent
    max_divergence_solver_iterations: int = 100
    max_density_solver_iterations: int = 100

    def __init__(self, **data):
        super().__init__(**data)
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")

        self._support_radius = 2 * self.particle_size

        if self.hash_grid_cell_size is None:
            self.hash_grid_cell_size = self._support_radius
        else:
            if self.hash_grid_cell_size < self._support_radius:
                gs.raise_exception("`hash_grid_cell_size` should not be smaller than 2 * `particle_size`.")

        if self.hash_grid_res is None:
            max_hash_grid_res = np.ceil(
                (np.array(self.upper_bound) - np.array(self.lower_bound)) / self.hash_grid_cell_size
            ).astype(int)
            default_hash_grid_res = np.array([150, 150, 150])
            self._hash_grid_res = np.minimum(max_hash_grid_res, default_hash_grid_res)
        else:
            self._hash_grid_res = np.ceil(np.array(self.hash_grid_res) / self.hash_grid_cell_size).astype(int)

        # check pressure solver
        pressure_solver_available = ["WCSPH", "DFSPH"]
        if self.pressure_solver not in pressure_solver_available:
            gs.raise_exception(
                f"Pressure solver {self.pressure_solver} not implemented. Please select among {pressure_solver_available}."
            )


class PBDOptions(Options):
    """
    Options configuring the PBDSolver.

    Note
    ----
    If spatial hashing parameters are not given, we will compute them automatically this way: For `hash_grid_cell_size`, we will set it to be 1.25 * `particle_size`. For `hash_grid_res`, if a small bound is given, it's used for the hash grid; otherwise, we use a default value of a 150^3 cube. Any grid bigger than that will results in too many cells hence not ideal.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    gravity : tuple, optional
        Gravity force in N/kg. If none, it will inherit from `SimOptions`. Defaults to None.
    max_stretch_solver_iterations : int, optional
        Maximum number of iterations for the solving stretch constraints. Defaults to 4.
    max_bending_solver_iterations : int, optional
        Maximum number of iterations for the solving bending constraints. Defaults to 1.
    max_volume_solver_iterations : int, optional
        Maximum number of iterations for the solving volume constraints. Defaults to 1.
    max_density_solver_iterations : int, optional
        Maximum number of iterations for the solving density constraints. Defaults to 1.
    max_viscosity_solver_iterations : int, optional
        Maximum number of iterations for the solving viscosity constraints. Defaults to 1.
    particle_size : float, optional
        Particle diameter in meters. Defaults to 1e-2.
    hash_grid_res : tuple, optional
        Size of the spatially-repetitive spatial hashing grid in meters. If none, it will be computed automatically. Defaults to None.
    hash_grid_cell_size : float, optional
        Size of the lattic cell of the spatial hashing grid in meters. This should be at least 1.25 * `particle_size`. If none, it will be computed automatically. Defaults to None.
    lower_bound : tuple, shape (3,), optional
        Lower bound of the simulation domain. Defaults to (-100.0, -100.0, 0.0).
    upper_bound : tuple, shape (3,), optional
        Upper bound of the simulation domain. Defaults to (100.0, 100.0, 100.0).
    """

    dt: Optional[float] = None
    gravity: Optional[tuple] = None

    # constraints solving iterations
    max_stretch_solver_iterations: int = 4
    max_bending_solver_iterations: int = 1
    max_volume_solver_iterations: int = 1
    max_density_solver_iterations: int = 1
    max_viscosity_solver_iterations: int = 1

    # self collision
    particle_size: Optional[float] = 1e-2

    # spatial hashing
    hash_grid_res: Optional[tuple] = None  # size of the spatially-repetitive hash grid in meters
    hash_grid_cell_size: Optional[float] = None  # size of the cubic cell in meters

    lower_bound: tuple = (-100.0, -100.0, 0.0)
    upper_bound: tuple = (100.0, 100.0, 100.0)

    def __init__(self, **data):
        super().__init__(**data)
        if not np.all(np.array(self.upper_bound) > np.array(self.lower_bound)):
            gs.raise_exception("Invalid pair of upper_bound and lower_bound.")

        # NOTE: 1.25 is a safety factor, as inside one single substep, multiple substages can change the position of the particles but we only do spatial hashing once.
        # Therefore, the grid cell needs to be a bit bigger so that neighbours are not missed.
        if self.hash_grid_cell_size is None:
            self.hash_grid_cell_size = 1.25 * self.particle_size
        else:
            if self.hash_grid_cell_size < 1.25 * self.particle_size:
                gs.raise_exception("`hash_grid_cell_size` should not be smaller than 1.25 * `particle_size`.")

        if self.hash_grid_res is None:
            # compute _hash_grid_res smartly
            # if a small bound is given, it's used for the hash grid
            # Otherwise, we use a default value of a 150^3 cube. Any grid bigger than that will results in too many cells hence not ideal.
            max_hash_grid_res = np.ceil(
                (np.array(self.upper_bound) - np.array(self.lower_bound)) / self.hash_grid_cell_size
            ).astype(int)
            default_hash_grid_res = np.array([150, 150, 150])
            self._hash_grid_res = np.minimum(max_hash_grid_res, default_hash_grid_res)
        else:
            self._hash_grid_res = np.ceil(np.array(self.hash_grid_res) / self.hash_grid_cell_size).astype(int)


class FEMOptions(Options):
    """
    Options configuring the FEMSolver.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    gravity : tuple, optional
        Gravity force in N/kg. If none, it will inherit from `SimOptions`. Defaults to None.
    damping : float, optional
        Damping factor. Defaults to 45.0.
    floor_height : float, optional
        Height of the floor in meters. If none, it will inherit from `SimOptions`. Defaults to None.
    """

    dt: Optional[float] = None
    gravity: Optional[tuple] = None
    damping: Optional[float] = 0.0
    floor_height: float = None


class SFOptions(Options):
    """
    Options configuring the SFSolver.

    Parameters
    ----------
    dt : float, optional
        Time duration for each simulation step in seconds. If none, it will inherit from `SimOptions`. Defaults to None.
    """

    dt: Optional[float] = None
    res: Optional[int] = 128
    solver_iters: Optional[int] = 500
    decay: Optional[float] = 0.99

    T_low: Optional[float] = 1.0
    T_high: Optional[float] = 0.0

    inlet_pos: Optional[tuple[int, int, int]] = (0.6, 0.0, 0.1)
    inlet_vel: Optional[tuple[int, int, int]] = (0, 0, 1)
    inlet_quat: Optional[tuple[int, int, int, int]] = (1, 0, 0, 0)
    inlet_s: Optional[float] = 400.0
