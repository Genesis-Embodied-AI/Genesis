"""Canonical top-level Rigid substep functions."""

import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.abd.diff import func_copy_next_to_curr
from genesis.engine.solvers.rigid.abd.forward_dynamics import (
    func_forward_dynamics,
    func_implicit_damping,
    func_integrate,
    func_update_acc,
)
from genesis.engine.solvers.rigid.abd.forward_kinematics import (
    func_forward_velocity,
    func_update_cartesian_space,
)
from genesis.engine.solvers.rigid.abd.misc import func_wakeup_coupled_links


@qd.func(requires_top_level=True)
def func_substep_wakeup_coupled_links(
    rigid_data: qd.template(),
    island_data: qd.template(),
    rigid_config: qd.template(),
):
    """Wake hibernated links targeted by coupling forces before dynamics consumes them."""
    if qd.static(rigid_config.use_hibernation):
        func_wakeup_coupled_links(
            rigid_data.dyn_state,
            island_data.island,
            rigid_data.dyn_info,
            rigid_data.rigid_info,
            rigid_config,
        )


@qd.func(requires_top_level=True)
def func_step_1(
    rigid_data: qd.template(),
    island_data: qd.template(),
    rigid_config: qd.template(),
    is_forward_pos_updated: qd.template(),
    is_forward_vel_updated: qd.template(),
    is_backward: qd.template(),
):
    if qd.static(not is_forward_pos_updated):
        func_update_cartesian_space(
            rigid_data.dyn_state,
            rigid_data.dyn_info,
            rigid_data.rigid_info,
            rigid_config,
            force_update_all_geoms=False,
            is_backward=is_backward,
        )
    if qd.static(not is_forward_vel_updated):
        func_forward_velocity(
            rigid_data.dyn_state,
            rigid_data.dyn_info,
            rigid_data.rigid_info,
            rigid_config,
            is_backward,
        )
    func_forward_dynamics(
        rigid_data.dyn_state,
        island_data.island,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        rigid_config,
        is_backward,
    )


@qd.func(requires_top_level=True)
def func_step_2(
    rigid_data: qd.template(),
    rigid_config: qd.template(),
    is_backward: qd.template(),
):
    func_update_acc(
        rigid_data.dyn_state,
        rigid_data.dyn_info,
        rigid_data.rigid_info,
        rigid_config,
        update_cacc=True,
        is_backward=is_backward,
    )
    if qd.static(rigid_config.integrator != gs.integrator.approximate_implicitfast):
        func_implicit_damping(rigid_data.dyn_state, rigid_data.dyn_info, rigid_data.rigid_info, rigid_config)
    func_integrate(rigid_data.dyn_state, rigid_data.dyn_info, rigid_data.rigid_info, rigid_config, is_backward)
    if qd.static(not is_backward):
        func_copy_next_to_curr(rigid_data.dyn_state, rigid_data.rigid_info, rigid_config, rigid_data.errno)
        if qd.static(not rigid_config.enable_mujoco_compatibility):
            func_update_cartesian_space(
                rigid_data.dyn_state,
                rigid_data.dyn_info,
                rigid_data.rigid_info,
                rigid_config,
                force_update_all_geoms=False,
                is_backward=is_backward,
            )
            func_forward_velocity(
                rigid_data.dyn_state,
                rigid_data.dyn_info,
                rigid_data.rigid_info,
                rigid_config,
                is_backward,
            )
