from typing import Literal, TYPE_CHECKING

import numpy as np
import torch
import numpy.typing as npt
import taichi as ti

import genesis as gs
from genesis.engine.entities.base_entity import Entity
from genesis.options.solvers import RigidOptions
import genesis.utils.geom as gu
from genesis.utils.misc import ti_field_to_torch, DeprecationError, ALLOCATE_TENSOR_WARNING
from genesis.engine.entities import AvatarEntity, DroneEntity, RigidEntity
from genesis.engine.states.solvers import RigidSolverState
from genesis.styles import colors, formats

from ..base_solver import Solver
from .collider_decomp import Collider
from .constraint_solver_decomp import ConstraintSolver
from .constraint_solver_decomp_island import ConstraintSolverIsland
from .sdf_decomp import SDF

if TYPE_CHECKING:
    from genesis.engine.scene import Scene
    from genesis.engine.simulator import Simulator


@ti.kernel
def _kernel_init_dof_fields(
    dofs_motion_ang: ti.types.ndarray(),
    dofs_motion_vel: ti.types.ndarray(),
    dofs_limit: ti.types.ndarray(),
    dofs_invweight: ti.types.ndarray(),
    dofs_stiffness: ti.types.ndarray(),
    dofs_damping: ti.types.ndarray(),
    dofs_armature: ti.types.ndarray(),
    dofs_kp: ti.types.ndarray(),
    dofs_kv: ti.types.ndarray(),
    dofs_force_range: ti.types.ndarray(),
    dofs_info: ti.template(),
    dofs_state: ti.template(),
    awake_dofs: ti.template(),
    n_awake_dofs: ti.template(),
    static_args: ti.template(),
):
    for I in ti.grouped(dofs_info):
        i = I[0]  # batching (if any) will be the second dim

        for j in ti.static(range(3)):
            dofs_info[I].motion_ang[j] = dofs_motion_ang[i, j]
            dofs_info[I].motion_vel[j] = dofs_motion_vel[i, j]

        for j in ti.static(range(2)):
            dofs_info[I].limit[j] = dofs_limit[i, j]
            dofs_info[I].force_range[j] = dofs_force_range[i, j]

        dofs_info[I].armature = dofs_armature[i]
        dofs_info[I].invweight = dofs_invweight[i]
        dofs_info[I].stiffness = dofs_stiffness[i]
        dofs_info[I].damping = dofs_damping[i]
        dofs_info[I].kp = dofs_kp[i]
        dofs_info[I].kv = dofs_kv[i]

    ti.loop_config(serialize=ti.static(static_args.para_level) < gs.PARA_LEVEL.PARTIAL)
    for i, b in ti.ndrange(dofs_state.shape[0], dofs_state.shape[1]):
        dofs_state[i, b].ctrl_mode = gs.CTRL_MODE.FORCE

    if ti.static(static_args.use_hibernation):
        ti.loop_config(serialize=static_args.para_level < gs.PARA_LEVEL.PARTIAL)
        for i, b in ti.ndrange(dofs_info.shape[0], dofs_info.shape[1]):
            dofs_state[i, b].hibernated = False
            awake_dofs[i, b] = i

        ti.loop_config(serialize=static_args.para_level < gs.PARA_LEVEL.PARTIAL)
        for b in range(dofs_info.shape[1]):
            n_awake_dofs[b] = dofs_info.shape[0]
