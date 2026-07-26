import dataclasses
from typing import NamedTuple

import numpy as np
import quadrants as qd
import torch

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.mesh as mu
from genesis.utils import array_class
from genesis.utils.misc import get_gpu_core_count, qd_to_torch, tensor_to_array

from . import cost as cost_mod
from .cost import (
    func_boundary_exclusions,
    func_resolve_goal,
    func_validate,
    kernel_boundary_exclusions,
    kernel_validate,
)
from .graph import (
    _RRT_GOAL_BIAS,
    _RRT_N_SHORTCUT,
    _RRT_STALL_ITERS,
    _RRT_STEER_STEP,
    func_rrt_connect,
)
from .retime import retime_trajectory
from .sphere_proxy import build_geom_sphere_proxy
from .trajopt import (
    _LS_LADDER,
    _MPPI_ANNEAL,
    build_noise_basis,
    func_optimize,
    func_seed_trajectories,
    func_seed_trajectories_from_rrt,
    kernel_lbfgs,
)
from .world import kernel_snapshot_world

# Sphere-fill budgets (per geom) and default derivative limits when the asset carries none.
_N_MAX_SPHERES_PER_GEOM = 8
_SPHERE_PAD = 0.005
_DEFAULT_VEL_LIMIT = np.pi
_DEFAULT_ACC_LIMIT = 13.0
# Contact tolerance and antipodal-squeeze threshold of the auto-grasp detection.
_GRASP_CONTACT_TOL = 0.01
_GRASP_NORMAL_COS = -0.5
# RRT-Connect fallback extents and iteration budget.
# Lanes cooperating on one candidate's trajectory cost in the tiled arm, capped at a subgroup and rounded down to
# a power of two by the resolution below (the cross-lane sum is a butterfly over 2**k lanes).
_N_COST_LANES_MAX = 32
_N_RRT_TREES = 4
_N_RRT_NODES = 2048
_N_RRT_ITERS = 600
# Fresh restart pools drawn within one ladder attempt for envs whose pool yielded no acceptable Cartesian goal
# branch (see _resolve_pose_goal).
_N_GOAL_RESOLVE_POOLS = 4
# Reach-weighted deviation radii of the certified knot straightening ladder (see plan): search budgets only -
# the straightened polyline re-certifies through the standard validator, which is what bounds its real
# clearance. Coarse first; envs whose straightened path fails certification (a chord cutting a tight corridor)
# retry finer, so tight regions straighten as much as their clearance allows instead of not at all.
_STRAIGHTEN_RADII = (0.05, 0.015, 0.005)


def _straighten_knots(qpos_knots, dofs_reach, radius):
    """Snap near-collinear knot runs onto their chords within the given reach-weighted radius.

    Douglas-Peucker on the reach-weighted deviation selects the knots that carry the path's shape (see
    _STRAIGHTEN_RADII), and every pruned knot moves to its projection on the surviving chord, ordered along it so
    the polyline never backtracks. Endpoints, and the duplicated boundary knots, are always kept.
    """
    B, W, n_dp = qpos_knots.shape
    reach = tensor_to_array(dofs_reach)
    knots_all = tensor_to_array(qpos_knots)
    for i_b in range(B):
        knots = knots_all[i_b]
        spans = [(0, W - 1)]
        while spans:
            i_lo, i_hi = spans.pop()
            if i_hi - i_lo < 2:
                continue
            chord = knots[i_hi] - knots[i_lo]
            weights_sq = reach * reach
            denom = (chord * chord * weights_sq).sum()
            js = np.arange(i_lo + 1, i_hi)
            rel = knots[js] - knots[i_lo]
            alpha = (rel * chord * weights_sq).sum(axis=1) / max(denom, 1e-12)
            alpha = np.maximum.accumulate(np.clip(alpha, 0.0, 1.0))
            snap = knots[i_lo] + alpha[:, None] * chord
            dev = (np.abs(knots[js] - snap) * reach).sum(axis=1)
            j_worst = dev.argmax()
            if dev[j_worst] <= radius:
                knots[js] = snap
            else:
                spans.append((i_lo, i_lo + 1 + j_worst))
                spans.append((i_lo + 1 + j_worst, i_hi))
    return torch.as_tensor(knots_all, dtype=gs.tc_float, device=gs.device)


@dataclasses.dataclass(frozen=True)
class PlannerPath:
    """Time-parametrized planned trajectory.

    Waypoints are spaced ``dt`` seconds apart per env; ``is_valid`` marks the envs whose path was certified
    collision-free at the requested safety margin - paths of failed envs hold the start configuration at rest and
    must not be executed.
    """

    qpos: torch.Tensor
    dofs_vel: torch.Tensor
    dofs_acc: torch.Tensor
    dt: torch.Tensor
    is_valid: torch.Tensor


class Attachment(NamedTuple):
    """One entity carried rigidly during the plan: the attach link and the per-env grasp transform."""

    entity: object
    link: object
    pos_offset: torch.Tensor
    quat_offset: torch.Tensor


class _Budgets(NamedTuple):
    """Iteration budgets of the trajectory optimizer, resolved per arm (see _resolve_budgets)."""

    mppi_n_iters: int
    mppi_n_particles: int
    lbfgs_n_iters: int
    ls_n_trials: int


class _EntityContext(NamedTuple):
    """Per-entity planner buffers, allocated once at the entity's first plan and reused forever.

    gjk_state is the planner-owned scratch of the collider's Gilbert-Johnson-Keerthi (GJK) distance queries, one
    column per eval column (see func_gjk_clearance in cost.py). The velocity and acceleration limits are host
    tensors because only the retiming reads them, on the host.
    """

    planner_config: object
    planner_info: object
    planner_state: object
    gjk_state: object
    dofs_vel_limit: torch.Tensor
    dofs_acc_limit: torch.Tensor
    spheres_link_idx: np.ndarray
    errno: object


@qd.kernel
def kernel_set_plan_scalars(
    goal_link_idx: int,
    seed_key: int,
    has_pose_goal: int,
    w_obs: float,
    w_self: float,
    w_lim: float,
    w_acc: float,
    w_jerk: float,
    w_posture: float,
    w_pose_pos: float,
    w_pose_rot: float,
    eps_act: float,
    eps_self: float,
    d_safe: float,
    planner_info: array_class.PlannerEntityInfo,
    errno: qd.Tensor,
):
    """Kernel fallback of _set_plan_scalars (see there)."""
    planner_info.cost.boundary.goal_link_idx[None] = goal_link_idx
    planner_info.cost.boundary.has_pose_goal[None] = has_pose_goal
    planner_info.mppi.seed_key[None] = seed_key
    planner_info.cost.w_obs[None] = w_obs
    planner_info.cost.w_self[None] = w_self
    planner_info.cost.w_lim[None] = w_lim
    planner_info.cost.w_acc[None] = w_acc
    planner_info.cost.w_jerk[None] = w_jerk
    planner_info.cost.w_posture[None] = w_posture
    planner_info.cost.w_pose_pos[None] = w_pose_pos
    planner_info.cost.w_pose_rot[None] = w_pose_rot
    planner_info.cost.eps_act[None] = eps_act
    planner_info.cost.eps_self[None] = eps_self
    planner_info.cost.d_safe[None] = d_safe
    for i_dp, i_b in qd.ndrange(planner_info.fk.dofs.is_locked.shape[0], planner_info.fk.dofs.is_locked.shape[1]):
        planner_info.fk.dofs.is_locked[i_dp, i_b] = False
    for i_s, i_b in qd.ndrange(planner_info.fk.attach.is_active.shape[0], planner_info.fk.attach.is_active.shape[1]):
        planner_info.fk.attach.is_active[i_s, i_b] = False
    for i_b in range(errno.shape[0]):
        errno[i_b] = 0


@qd.kernel
def kernel_set_opt_phase(
    seed_key: int,
    w_obs: float,
    w_self: float,
    w_acc: float,
    w_jerk: float,
    w_pose_pos: float,
    w_pose_rot: float,
    eps_act: float,
    planner_info: array_class.PlannerEntityInfo,
):
    """Kernel fallback of _set_opt_phase (see there)."""
    planner_info.mppi.seed_key[None] = seed_key
    planner_info.cost.w_obs[None] = w_obs
    planner_info.cost.w_self[None] = w_self
    planner_info.cost.w_acc[None] = w_acc
    planner_info.cost.w_jerk[None] = w_jerk
    planner_info.cost.w_pose_pos[None] = w_pose_pos
    planner_info.cost.w_pose_rot[None] = w_pose_rot
    planner_info.cost.eps_act[None] = eps_act


@qd.kernel
def kernel_set_clearance(
    d_safe: float,
    planner_info: array_class.PlannerEntityInfo,
):
    """Kernel fallback of _set_clearance (see there)."""
    planner_info.cost.d_safe[None] = d_safe


def _set_plan_scalars(
    planner_info,
    errno,
    goal_link_idx,
    seed_key,
    has_pose_goal,
    w_obs,
    w_self,
    w_lim,
    w_acc,
    w_jerk,
    w_posture,
    w_pose_pos,
    w_pose_rot,
    eps_act,
    eps_self,
    d_safe,
):
    """One packed per-plan update of every runtime scalar, with the lock-mask, attach-mask and errno clears.

    The scalars are the goal designation, noise key, cost weights and clearance. The update goes in place through
    zero-copy views when available, a single kernel launch otherwise.
    """
    if gs.use_zerocopy:
        for field, value in (
            (planner_info.cost.boundary.goal_link_idx, goal_link_idx),
            (planner_info.cost.boundary.has_pose_goal, has_pose_goal),
            (planner_info.mppi.seed_key, seed_key),
            (planner_info.cost.w_obs, w_obs),
            (planner_info.cost.w_self, w_self),
            (planner_info.cost.w_lim, w_lim),
            (planner_info.cost.w_acc, w_acc),
            (planner_info.cost.w_jerk, w_jerk),
            (planner_info.cost.w_posture, w_posture),
            (planner_info.cost.w_pose_pos, w_pose_pos),
            (planner_info.cost.w_pose_rot, w_pose_rot),
            (planner_info.cost.eps_act, eps_act),
            (planner_info.cost.eps_self, eps_self),
            (planner_info.cost.d_safe, d_safe),
            (planner_info.fk.dofs.is_locked, False),
            (planner_info.fk.attach.is_active, False),
            (errno, 0),
        ):
            field_t = qd_to_torch(field, copy=False)
            field_t[...] = value
    else:
        kernel_set_plan_scalars(
            goal_link_idx,
            seed_key,
            int(has_pose_goal),
            w_obs,
            w_self,
            w_lim,
            w_acc,
            w_jerk,
            w_posture,
            w_pose_pos,
            w_pose_rot,
            eps_act,
            eps_self,
            d_safe,
            planner_info,
            errno,
        )


def _set_opt_phase(planner_info, seed_key, w_obs, w_self, w_acc, w_jerk, w_pose_pos, w_pose_rot, eps_act):
    """One packed update of the optimization-phase scalars (write mechanism of _set_plan_scalars)."""
    if gs.use_zerocopy:
        for field, value in (
            (planner_info.mppi.seed_key, seed_key),
            (planner_info.cost.w_obs, w_obs),
            (planner_info.cost.w_self, w_self),
            (planner_info.cost.w_acc, w_acc),
            (planner_info.cost.w_jerk, w_jerk),
            (planner_info.cost.w_pose_pos, w_pose_pos),
            (planner_info.cost.w_pose_rot, w_pose_rot),
            (planner_info.cost.eps_act, eps_act),
        ):
            field_t = qd_to_torch(field, copy=False)
            field_t[...] = value
    else:
        kernel_set_opt_phase(seed_key, w_obs, w_self, w_acc, w_jerk, w_pose_pos, w_pose_rot, eps_act, planner_info)


def _set_clearance(planner_info, d_safe):
    """Update the required clearance headroom (write mechanism of _set_plan_scalars)."""
    if gs.use_zerocopy:
        d_safe_t = qd_to_torch(planner_info.cost.d_safe, copy=False)
        d_safe_t[...] = d_safe
        # The next kernel reads the clearance through this same buffer, so the write has to have landed first.
        if gs.backend == gs.metal:
            torch.mps.synchronize()
    else:
        kernel_set_clearance(d_safe, planner_info)


@qd.func
def func_fold_and_check_exit(
    graph_counter: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_config: qd.template(),
    ignore_collision: qd.template(),
):
    """Fold the validator verdicts into per-env solved flags and advance the ladder counter.

    An env becomes solved once any of its seeds certifies, and its candidates are then deactivated so later
    passes leave them untouched. graph_counter is decremented every pass and zeroed once every planned env is
    solved, ending the device-side loop as soon as the whole batch is done. When collision checking is disabled
    the collision and goal-tolerance verdicts are ignored, so a straight-line seed within the joint limits is
    accepted directly.
    """
    n_seeds = qd.static(planner_config.n_seeds)
    # Ignore the collision and goal-tolerance bits when collision checking is disabled (a straight-line seed is
    # the answer); -1 keeps every bit otherwise, so the check reduces to flags == 0.
    keep_mask = qd.static(~(planner_config.flag_collision | planner_config.flag_goal_tol) if ignore_collision else -1)
    qd.loop_config(name="planner_fold_decrement")
    for _ in range(1):
        graph_counter[()] = graph_counter[()] - 1
        planner_state.pass_index[None] = planner_state.pass_index[None] - 1
        planner_state.early_exit_flag[()] = 0

    qd.loop_config(serialize=qd.static(planner_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        if not planner_state.is_env_solved[i_b]:
            is_solved = False
            for i_s in range(n_seeds):
                if (planner_state.cert.valid_flags[i_b_ * n_seeds + i_s] & keep_mask) == 0:
                    is_solved = True
            if is_solved:
                planner_state.is_env_solved[i_b] = True
                for i_s in range(n_seeds):
                    planner_state.cert.is_active[i_b_ * n_seeds + i_s] = False
            else:
                qd.atomic_max(planner_state.early_exit_flag[()], 1)

    qd.loop_config(name="planner_fold_exit")
    for _ in range(1):
        if planner_state.early_exit_flag[()] == 0:
            graph_counter[()] = 0


@qd.kernel
def kernel_init_ladder(n_attempts: int, envs_idx: qd.types.ndarray(), planner_state: array_class.PlannerState):
    """Reset the ladder bookkeeping of every planned env at the start of a plan.

    Split from kernel_plan so a ladder driven one pass per launch does not reset itself between passes, and so the
    graph kernel holds nothing outside its loop - work placed there replays per task on the backends that fall back
    to a host-driven loop. pass_index counts the attempt budget down, the order the restart and seed draws key on.
    """
    for i_b_ in range(envs_idx.shape[0]):
        planner_state.is_env_solved[envs_idx[i_b_]] = False
        planner_state.is_env_seeded[envs_idx[i_b_]] = False
    qd.loop_config(name="planner_init_pass_index")
    for _ in range(1):
        planner_state.pass_index[None] = n_attempts


@qd.kernel(graph=True, fastcache=True)
def kernel_resolve_goal(
    envs_idx: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
    ignore_collision_static: qd.template(),
    errno: qd.Tensor,
):
    """Resolve every planned env's Cartesian goal into a joint configuration, for one ladder pass.

    Judge each hold-at-goal restart with the start-side allowances only: a previously resolved branch's allowances
    would excuse a fresh hold's contacts and smuggle a penetrating branch past the gate, so the goal side is
    stripped before the probe and folded back in once the goals are resolved (the ladder validator then excuses the
    resolved goal's own contacts).

    This is the pass's first half, split from kernel_plan because goal resolution is by far its largest body and
    compiling the two together costs far more than compiling them apart. graph_counter is read, never advanced: it
    the resolution keys its restart draws on planner_state.pass_index, which the ladder in kernel_plan owns.
    """
    # Judge each hold-at-goal restart with the start-side allowances only: a previously resolved branch's
    # allowances would excuse a fresh hold's contacts and smuggle a penetrating branch past the gate, so the
    # goal side is stripped before the probe and folded back in once the goals are resolved (the ladder
    # validator below then excuses the resolved goal's own contacts). See the host ladder for the rationale.
    func_boundary_exclusions(
        envs_idx,
        planner_state,
        planner_world,
        dyn_state,
        collider_state,
        gjk_state,
        planner_info,
        dyn_info,
        rigid_info,
        collider_info,
        sdf_info,
        rigid_config,
        collider_static_config,
        planner_config,
        include_goal=False,
        errno=errno,
    )
    func_resolve_goal(
        envs_idx,
        planner_state,
        planner_world,
        dyn_state,
        collider_state,
        gjk_state,
        planner_info,
        dyn_info,
        rigid_info,
        collider_info,
        sdf_info,
        rigid_config,
        collider_static_config,
        planner_config,
        ignore_collision_static,
    )
    func_boundary_exclusions(
        envs_idx,
        planner_state,
        planner_world,
        dyn_state,
        collider_state,
        gjk_state,
        planner_info,
        dyn_info,
        rigid_info,
        collider_info,
        sdf_info,
        rigid_config,
        collider_static_config,
        planner_config,
        include_goal=True,
        errno=errno,
    )


@qd.kernel(graph=True, fastcache=True)
def kernel_rrt_escalate(
    envs_idx: qd.types.ndarray(),
    trees_is_active: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
):
    """Escalate one attempt pass of the still-unsolved envs to RRT-Connect, seeding from the certified polylines.

    Tree growth is the ladder's heaviest phase and shares nothing with the rest of a pass but planner state, so it
    compiles apart from it: the whole collision model is inlined into whichever kernel reaches it, and compilation
    grows faster than the kernel does.
    """
    n_rrt_trees = qd.static(planner_config.n_rrt_trees)
    # Escalate the still-unsolved envs already seeded on an earlier pass to RRT-Connect: activate their tree
    # pairs, search at a smaller headroom than the optimizer (a goal snugger than the full headroom would wall
    # off its own tree), then transplant the certified polylines into the candidate columns. Fresh envs (not
    # yet seeded) instead take straight-line seeds; every unsolved env is then marked seeded so its next pass
    # escalates. On the first pass no env is seeded, so RRT runs on no tree and only the straight-line seed fires.
    qd.loop_config(name="planner_plan_rrt_active")
    for i_t in range(envs_idx.shape[0] * n_rrt_trees):
        i_b = envs_idx[i_t // n_rrt_trees]
        trees_is_active[i_t] = 0
        if not planner_state.is_env_solved[i_b] and planner_state.is_env_seeded[i_b]:
            trees_is_active[i_t] = 1
    d_safe_opt = planner_info.cost.d_safe[None]
    planner_info.cost.d_safe[None] = d_safe_opt - 0.01
    func_rrt_connect(
        envs_idx,
        trees_is_active,
        planner_state,
        planner_world,
        dyn_state,
        collider_state,
        gjk_state,
        planner_info,
        dyn_info,
        rigid_info,
        collider_info,
        sdf_info,
        rigid_config,
        collider_static_config,
        planner_config,
    )
    planner_info.cost.d_safe[None] = d_safe_opt
    func_seed_trajectories_from_rrt(envs_idx, planner_state, planner_config)


@qd.kernel(fastcache=True)
def kernel_fold(
    graph_counter: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_config: qd.template(),
    ignore_collision_static: qd.template(),
):
    """Fold one attempt pass's verdicts and advance the attempt budget (see func_fold_and_check_exit)."""
    func_fold_and_check_exit(graph_counter, envs_idx, planner_state, planner_config, ignore_collision_static)


@qd.kernel(graph=True, fastcache=True)
def kernel_plan(
    envs_idx: qd.types.ndarray(),
    trees_is_active: qd.types.ndarray(),
    planner_state: array_class.PlannerState,
    planner_world: array_class.PlannerWorldState,
    dyn_state: array_class.DynState,
    collider_state: array_class.ColliderState,
    gjk_state: array_class.GJKState,
    planner_info: array_class.PlannerEntityInfo,
    dyn_info: array_class.DynInfo,
    rigid_info: array_class.RigidInfo,
    collider_info: array_class.ColliderInfo,
    sdf_info: array_class.SDFInfo,
    rigid_config: qd.template(),
    collider_static_config: qd.template(),
    planner_config: qd.template(),
    ignore_collision_static: qd.template(),
    errno: qd.Tensor,
):
    """Seed and refine one attempt pass of every planned env.

    An env seeds from straight lines while it is unseeded and from its escalated trees once it is not, and is then
    marked seeded so its next pass escalates. The world snapshot and boundary exclusions this reads must already
    be in planner state, and the caller certifies and folds the pass (see kernel_validate and kernel_fold).
    """
    n_rrt_trees = qd.static(planner_config.n_rrt_trees)
    func_seed_trajectories(envs_idx, planner_state, planner_info, planner_config)
    qd.loop_config(name="planner_plan_mark_seeded")
    for i_b_ in range(envs_idx.shape[0]):
        if not planner_state.is_env_solved[envs_idx[i_b_]]:
            planner_state.is_env_seeded[envs_idx[i_b_]] = True
    if qd.static(not ignore_collision_static):
        func_optimize(
            0,
            envs_idx,
            planner_state,
            planner_world,
            dyn_state,
            collider_state,
            gjk_state,
            planner_info,
            dyn_info,
            rigid_info,
            collider_info,
            sdf_info,
            rigid_config,
            collider_static_config,
            planner_config,
        )


class Planner:
    """Optimization-based motion planner of the rigid solver (see plan_path for the user entrypoint).

    Planning runs entirely on planner-owned scratch buffers: the live solver state is only ever read (world
    snapshot, start configuration, grasp transforms), so a plan can never alter the kinematic state of the scene.
    """

    def __init__(self, solver):
        self._solver = solver
        self._entity_contexts = {}
        self._geom_proxies = {}
        self._plan_counter = 0
        # The planner queries obstacle meshes through their signed distance field (SDF) grids, which the collider
        # only loads when its own narrowphase needs them.
        solver.collider._sdf.activate()
        self._planner_world = array_class.get_planner_world_state(solver.n_geoms, max(solver.n_envs, 1))

    # ------------------------------------------------------------------------------------
    # -------------------------------- context assembly ----------------------------------
    # ------------------------------------------------------------------------------------

    def _get_geom_proxy(self, geom):
        proxy = self._geom_proxies.get(geom.idx)
        if proxy is None:
            proxy = build_geom_sphere_proxy(geom, _N_MAX_SPHERES_PER_GEOM, _SPHERE_PAD)
            self._geom_proxies[geom.idx] = proxy
        return proxy

    def _resolve_budgets(self, B):
        """Arm and iteration budgets: a pure function of backend, options, and env count (deterministic)."""
        options = self._solver._options
        arm = options.planner_arm
        if arm is None:
            arm = gs.planner_arm.SERIAL if gs.backend == gs.cpu else gs.planner_arm.BATCHED
        n_seeds = options.planner_n_seeds
        if n_seeds is None:
            if arm == gs.planner_arm.SERIAL:
                n_seeds = 8 if B <= 1 else max(4, 8 // B)
            else:
                # Candidates buy basin diversity, which a handful already provides; they do not buy occupancy any
                # more, now that a candidate's own work is spread over a subgroup of lanes. Seeding past that costs
                # every phase - the goal resolution draws restarts per candidate, and the optimizer refines each of
                # them - for diversity the escalation already covers.
                n_seeds = 8
        if arm == gs.planner_arm.SERIAL:
            budgets = _Budgets(mppi_n_iters=6, mppi_n_particles=4, lbfgs_n_iters=48, ls_n_trials=4)
        else:
            budgets = _Budgets(mppi_n_iters=12, mppi_n_particles=8, lbfgs_n_iters=48, ls_n_trials=4)
        return arm, n_seeds, budgets

    def _get_entity_context(self, entity):
        context = self._entity_contexts.get(entity.idx)
        if context is not None:
            return context

        solver = self._solver
        B = max(solver.n_envs, 1)
        for joint in entity.joints:
            if joint.type in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.SPHERICAL):
                gs.raise_exception(f"Planning does not support {joint.type} joints.")

        # Robot collision proxy, sorted by link, each sphere mapped to the collision geom it proxies. The geoms
        # themselves are collected alongside (global index, link-frame offset, bounding sphere of their mesh) so
        # certification paths can pose them at hypothetical FK poses for the exact GJK re-checks.
        spheres_link_idx, spheres_geom_idx, spheres_pos_local, spheres_radius = [], [], [], []
        rgeoms_idx, rgeoms_offset_pos, rgeoms_offset_quat = [], [], []
        rgeoms_bound_center, rgeoms_bound_radius, rgeoms_links_start = [], [], [0]
        links_verts_extent = np.zeros(entity.n_links, dtype=gs.np_float)
        for i_l, link in enumerate(entity.links):
            for geom in link.geoms:
                proxy = self._get_geom_proxy(geom)
                spheres_link_idx.extend([link.idx - entity.link_start] * len(proxy.radius))
                spheres_geom_idx.extend([len(rgeoms_idx)] * len(proxy.radius))
                spheres_pos_local.append(proxy.pos)
                spheres_radius.append(proxy.radius)
                verts = gu.transform_by_trans_quat(geom.init_verts, geom.init_pos, geom.init_quat)
                bound_center = 0.5 * (verts.min(axis=0) + verts.max(axis=0))
                rgeoms_idx.append(geom.idx)
                rgeoms_offset_pos.append(geom.init_pos)
                rgeoms_offset_quat.append(geom.init_quat)
                rgeoms_bound_center.append(bound_center)
                rgeoms_bound_radius.append(np.linalg.norm(verts - bound_center, axis=-1).max())
                links_verts_extent[i_l] = max(links_verts_extent[i_l], np.linalg.norm(verts, axis=-1).max())
            rgeoms_links_start.append(len(rgeoms_idx))
        spheres_link_idx = np.array(spheres_link_idx, dtype=gs.np_int)
        spheres_geom_idx = np.array(spheres_geom_idx, dtype=gs.np_int)
        spheres_pos_local = np.concatenate(spheres_pos_local, dtype=gs.np_float)
        spheres_radius = np.concatenate(spheres_radius, dtype=gs.np_float)
        spheres_links_start = np.zeros(entity.n_links + 1, dtype=gs.np_int)
        np.cumsum(np.bincount(spheres_link_idx, minlength=entity.n_links), out=spheres_links_start[1:])
        links_bound_center = np.zeros((entity.n_links, 3), dtype=gs.np_float)
        links_bound_radius = np.zeros(entity.n_links, dtype=gs.np_float)
        for i_l in range(entity.n_links):
            pos_l = spheres_pos_local[spheres_links_start[i_l] : spheres_links_start[i_l + 1]]
            radius_l = spheres_radius[spheres_links_start[i_l] : spheres_links_start[i_l + 1]]
            if len(radius_l):
                center = 0.5 * ((pos_l - radius_l[:, None]).min(axis=0) + (pos_l + radius_l[:, None]).max(axis=0))
                links_bound_center[i_l] = center
                links_bound_radius[i_l] = (np.linalg.norm(pos_l - center, axis=-1) + radius_l).max()

        # Self-collision sphere pairs from the collider's build-time geom-pair filter, collapsed to link pairs.
        pair_mask = tensor_to_array(qd_to_torch(solver.collider._collider_info.collision_pair_idx) != -1)
        link_pairs = set()
        for link_a in entity.links:
            for link_b in entity.links:
                if link_b.idx <= link_a.idx:
                    continue
                if any(
                    pair_mask[geom_a.idx, geom_b.idx] or pair_mask[geom_b.idx, geom_a.idx]
                    for geom_a in link_a.geoms
                    for geom_b in link_b.geoms
                ):
                    link_pairs.add((link_a.idx - entity.link_start, link_b.idx - entity.link_start))
        self_pairs = [
            (i_sa, i_sb)
            for i_la, i_lb in sorted(link_pairs)
            for i_sa in np.flatnonzero(spheres_link_idx == i_la)
            for i_sb in np.flatnonzero(spheres_link_idx == i_lb)
        ]
        # Prune pairs the coarse proxy cannot discriminate: pairs resting inside the self-collision activation
        # band at the context-creation configuration would pay a permanent, unresolvable cost and poison the
        # certificate (the collider applies the same neutral-config filtering to its geom pairs); the per-plan
        # start exclusions absorb the rare residual cases.
        if self_pairs:
            envs_kw = dict(envs_idx=[0]) if solver.n_envs > 0 else {}
            links_pos = tensor_to_array(solver.get_links_pos([link.idx for link in entity.links], **envs_kw))
            links_quat = tensor_to_array(solver.get_links_quat([link.idx for link in entity.links], **envs_kw))
            if links_pos.ndim == 3:
                links_pos, links_quat = links_pos[0], links_quat[0]
            spheres_now = links_pos[spheres_link_idx] + gu.transform_by_quat(
                spheres_pos_local, links_quat[spheres_link_idx]
            )
            pairs_arr = np.array(self_pairs, dtype=gs.np_int)
            sd_now = (
                np.linalg.norm(spheres_now[pairs_arr[:, 0]] - spheres_now[pairs_arr[:, 1]], axis=-1)
                - spheres_radius[pairs_arr[:, 0]]
                - spheres_radius[pairs_arr[:, 1]]
            )
            self_pairs = [pair for pair, sd in zip(self_pairs, sd_now) if sd > 0.05]

        # Sphere pairs stay grouped by link pair (built from sorted link pairs, pruning preserves order):
        # link_pairs_idx / link_pairs_start range them per link pair, one exact rescue per failing link pair.
        self_link_pairs, self_link_pairs_counts = [], []
        for i_sa, i_sb in self_pairs:
            pair_links = (spheres_link_idx[i_sa], spheres_link_idx[i_sb])
            if not self_link_pairs or self_link_pairs[-1] != pair_links:
                self_link_pairs.append(pair_links)
                self_link_pairs_counts.append(0)
            self_link_pairs_counts[-1] += 1
        self_link_pairs_start = np.zeros(len(self_link_pairs) + 1, dtype=gs.np_int)
        np.cumsum(self_link_pairs_counts, out=self_link_pairs_start[1:])

        # Covering surface samples of the collision meshes, partitioned per proxy sphere for the certification
        # rescue of proxy-failing pairs (see _EXACT_RESCUE_WINDOW in cost.py), at the decision (fine) and
        # screening (coarse) covering radii. Each sample goes to the geom sphere containing it most deeply, so
        # wherever the proxy covers the surface, a deep point sits in the sample set of a sphere whose own
        # proxy check fails. Surface regions outside every proxy sphere are blind spots of the proxy check
        # itself (the fill proxy is not a strict cover); the rescue inherits them rather than paying their
        # containment gap, keeping its demand at the covering radius.
        spheres_verts_fine = [np.zeros((0, 3), dtype=gs.np_float)] * len(spheres_radius)
        spheres_verts_coarse = [np.zeros((0, 3), dtype=gs.np_float)] * len(spheres_radius)
        i_sphere_base = 0
        for link in entity.links:
            for geom in link.geoms:
                proxy = self._get_geom_proxy(geom)
                centers = gu.transform_by_trans_quat(proxy.pos, geom.init_pos, geom.init_quat)
                for r_cov, level_verts in (
                    (cost_mod._EXACT_SAMPLE_COV, spheres_verts_fine),
                    (cost_mod._EXACT_SAMPLE_COV_COARSE, spheres_verts_coarse),
                ):
                    samples = mu.surface_sample_covering(geom.init_verts, geom.init_faces, r_cov)
                    samples = gu.transform_by_trans_quat(samples, geom.init_pos, geom.init_quat)
                    depth = proxy.radius[None, :] - np.linalg.norm(samples[:, None] - centers[None, :], axis=-1)
                    assigned = depth.argmax(axis=-1)
                    for i_s in range(len(proxy.radius)):
                        level_verts[i_sphere_base + i_s] = samples[assigned == i_s]
                i_sphere_base += len(proxy.radius)
        spheres_vert_start = np.zeros(len(spheres_radius) + 1, dtype=gs.np_int)
        np.cumsum([len(samples) for samples in spheres_verts_fine], out=spheres_vert_start[1:])
        verts_pos_local = np.concatenate(spheres_verts_fine, dtype=gs.np_float)
        spheres_coarse_start = np.zeros(len(spheres_radius) + 1, dtype=gs.np_int)
        np.cumsum([len(samples) for samples in spheres_verts_coarse], out=spheres_coarse_start[1:])
        coarse_pos_local = np.concatenate(spheres_verts_coarse, dtype=gs.np_float)

        # Attached-sphere capacity: every proxy sphere of every single-link free entity in the scene.
        n_attach_max = sum(
            len(self._get_geom_proxy(geom).radius)
            for other in solver.entities
            if other is not entity and len(other.links) == 1 and not other.links[0].is_fixed
            for geom in other.links[0].geoms
        )

        dof_reach_np = self._compute_dof_reach(
            entity, spheres_link_idx, spheres_pos_local, spheres_radius, links_verts_extent
        )

        arm, n_seeds, budgets = self._resolve_budgets(B)
        # A GPU lane owns one candidate and the lanes of a warp advance through the knots together, so the knot-major
        # working set (see get_planner_state) serves a warp's reads from one cache line. A CPU thread instead walks one
        # whole candidate, which the candidate-major order keeps contiguous, by an order of magnitude either way.
        is_knot_major = gs.backend != gs.cpu
        # One thread per candidate leaves a GPU idle whenever the candidate columns alone do not fill it, since a
        # candidate's trajectory cost is a serial walk over its knots. Below that occupancy the walk is spread over a
        # subgroup of lanes, each taking a slice of the knots, and their partial costs are summed across the lanes.
        # Above it the columns already saturate the device and the extra evaluation scratch each lane needs would be
        # pure cost, so the one-thread-per-candidate arm stays.
        n_cost_lanes = 1
        if gs.backend != gs.cpu and B * n_seeds <= get_gpu_core_count():
            n_lanes_max = min(_N_COST_LANES_MAX, solver._options.planner_n_knots)
            n_cost_lanes = 1 << (n_lanes_max.bit_length() - 1)
        planner_config = array_class.PlannerStaticConfig(
            para_level=solver._para_level,
            is_batched_arm=arm == gs.planner_arm.BATCHED,
            entity_idx=entity._idx_in_solver,
            link_offset=entity.link_start,
            joint_offset=entity.joint_start,
            q_offset=entity.q_start,
            n_dp=entity.n_qs,
            n_links=entity.n_links,
            n_joints=entity.n_joints,
            n_spheres=len(spheres_radius),
            n_attach_max=n_attach_max,
            n_knots=solver._options.planner_n_knots,
            n_seeds=n_seeds,
            n_cost_lanes=n_cost_lanes,
            n_eval_per_candidate=n_cost_lanes,
            is_knot_major=is_knot_major,
            n_rrt_trees=_N_RRT_TREES,
            n_rrt_nodes=_N_RRT_NODES,
            n_noise_knots=array_class.PLANNER_N_NOISE_KNOTS,
            n_mppi_particles_max=array_class.PLANNER_MPPI_P_MAX,
            n_lbfgs_hist=array_class.PLANNER_LBFGS_M,
            n_ls_trials_max=array_class.PLANNER_LS_TRIALS_MAX,
            n_upsample=cost_mod._VALIDATE_UPSAMPLE,
            n_refine=cost_mod._VALIDATE_REFINE,
            flag_collision=cost_mod.COLLISION,
            flag_joint_limit=cost_mod.JOINT_LIMIT,
            flag_goal_tol=cost_mod.GOAL_TOL,
            flag_goal_in_collision=cost_mod.GOAL_IN_COLLISION,
            goal_ik_iters=cost_mod._GOAL_IK_ITERS,
            goal_ik_batches=cost_mod._GOAL_IK_BATCHES,
            goal_ik_damping=cost_mod._GOAL_IK_DAMPING,
            goal_ik_pos_tol=cost_mod._GOAL_IK_POS_TOL,
            goal_ik_rot_tol=cost_mod._GOAL_IK_ROT_TOL,
            goal_ik_max_step=cost_mod._GOAL_IK_MAX_STEP,
        )
        planner_info = array_class.get_planner_entity_info(
            planner_config,
            len(self_pairs),
            len(self_link_pairs),
            len(verts_pos_local),
            len(coarse_pos_local),
            len(rgeoms_idx),
            B,
        )
        planner_state = array_class.get_planner_state(planner_config, B)
        # Planner-owned scratch of the collider's GJK distance queries, one column per eval column (the RRT
        # tree columns alias the low eval columns). EPA never runs in the planner, so the contact-only layout
        # suffices - see func_gjk_clearance in cost.py.
        gjk_state = array_class.get_gjk_state_contact_only(B * n_seeds * planner_config.n_eval_per_candidate)
        errno = array_class.V(dtype=gs.qd_int, shape=(B,))

        spheres_link_idx_t = qd_to_torch(planner_info.fk.spheres.link_idx, copy=False)
        spheres_link_idx_t[:] = torch.as_tensor(spheres_link_idx, device=gs.device)
        spheres_geom_idx_t = qd_to_torch(planner_info.fk.spheres.geom_idx, copy=False)
        spheres_geom_idx_t[:] = torch.as_tensor(spheres_geom_idx, device=gs.device)
        spheres_pos_local_t = qd_to_torch(planner_info.fk.spheres.pos_local, copy=False)
        spheres_pos_local_t[:] = torch.as_tensor(spheres_pos_local, device=gs.device)
        spheres_radius_t = qd_to_torch(planner_info.fk.spheres.radius, copy=False)
        spheres_radius_t[:] = torch.as_tensor(spheres_radius, device=gs.device)
        spheres_links_start_t = qd_to_torch(planner_info.fk.spheres.links_start, copy=False)
        spheres_links_start_t[:] = torch.as_tensor(spheres_links_start, device=gs.device)
        links_bound_center_t = qd_to_torch(planner_info.fk.spheres.links_bound_center_local, copy=False)
        links_bound_center_t[:] = torch.as_tensor(links_bound_center, device=gs.device)
        links_bound_radius_t = qd_to_torch(planner_info.fk.spheres.links_bound_radius, copy=False)
        links_bound_radius_t[:] = torch.as_tensor(links_bound_radius, device=gs.device)
        rgeoms_idx_t = qd_to_torch(planner_info.fk.geoms.geoms_idx, copy=False)
        rgeoms_idx_t[:] = torch.as_tensor(np.array(rgeoms_idx, dtype=gs.np_int), device=gs.device)
        rgeoms_links_start_t = qd_to_torch(planner_info.fk.geoms.links_start, copy=False)
        rgeoms_links_start_t[:] = torch.as_tensor(np.array(rgeoms_links_start, dtype=gs.np_int), device=gs.device)
        rgeoms_offset_pos_t = qd_to_torch(planner_info.fk.geoms.offset_pos, copy=False)
        rgeoms_offset_pos_t[:] = torch.as_tensor(np.array(rgeoms_offset_pos, dtype=gs.np_float), device=gs.device)
        rgeoms_offset_quat_t = qd_to_torch(planner_info.fk.geoms.offset_quat, copy=False)
        rgeoms_offset_quat_t[:] = torch.as_tensor(np.array(rgeoms_offset_quat, dtype=gs.np_float), device=gs.device)
        rgeoms_bound_center_t = qd_to_torch(planner_info.fk.geoms.bound_center_local, copy=False)
        rgeoms_bound_center_t[:] = torch.as_tensor(np.array(rgeoms_bound_center, dtype=gs.np_float), device=gs.device)
        rgeoms_bound_radius_t = qd_to_torch(planner_info.fk.geoms.bound_radius, copy=False)
        rgeoms_bound_radius_t[:] = torch.as_tensor(np.array(rgeoms_bound_radius, dtype=gs.np_float), device=gs.device)
        if len(verts_pos_local) > 0:
            verts_pos_local_t = qd_to_torch(planner_info.fk.verts.pos_local, copy=False)
            verts_pos_local_t[:] = torch.as_tensor(verts_pos_local, device=gs.device)
        spheres_vert_start_t = qd_to_torch(planner_info.fk.verts.spheres_start, copy=False)
        spheres_vert_start_t[:] = torch.as_tensor(spheres_vert_start, device=gs.device)
        if len(coarse_pos_local) > 0:
            coarse_pos_local_t = qd_to_torch(planner_info.fk.verts.coarse_pos_local, copy=False)
            coarse_pos_local_t[:] = torch.as_tensor(coarse_pos_local, device=gs.device)
        spheres_coarse_start_t = qd_to_torch(planner_info.fk.verts.spheres_coarse_start, copy=False)
        spheres_coarse_start_t[:] = torch.as_tensor(spheres_coarse_start, device=gs.device)
        if self_pairs:
            self_pairs_idx_t = qd_to_torch(planner_info.fk.self_pairs.spheres_idx, copy=False)
            self_pairs_idx_t[:] = torch.as_tensor(np.array(self_pairs, dtype=gs.np_int), device=gs.device)
            self_pairs_reach_t = qd_to_torch(planner_info.fk.self_pairs.reach, copy=False)
            self_pairs_reach_t[:] = torch.as_tensor(
                self._compute_self_pairs_reach(entity, spheres_link_idx, self_pairs, dof_reach_np), device=gs.device
            )
            link_pairs_idx_t = qd_to_torch(planner_info.fk.self_pairs.link_pairs_idx, copy=False)
            link_pairs_idx_t[:] = torch.as_tensor(np.array(self_link_pairs, dtype=gs.np_int), device=gs.device)
            link_pairs_start_t = qd_to_torch(planner_info.fk.self_pairs.link_pairs_start, copy=False)
            link_pairs_start_t[:] = torch.as_tensor(self_link_pairs_start, device=gs.device)

        # Joint-space box limits and derivative limits (model velocity limits, defaults when the asset has none).
        q_limit_lower, q_limit_upper = entity.q_limit
        q_limit_lower_t = qd_to_torch(planner_info.fk.dofs.q_limit_lower, copy=False)
        q_limit_lower_t[:] = torch.as_tensor(q_limit_lower, dtype=gs.tc_float, device=gs.device)
        q_limit_upper_t = qd_to_torch(planner_info.fk.dofs.q_limit_upper, copy=False)
        q_limit_upper_t[:] = torch.as_tensor(q_limit_upper, dtype=gs.tc_float, device=gs.device)
        vel_limit = entity.get_dofs_vel_limit()
        if vel_limit.ndim > 1:
            vel_limit = vel_limit[0]
        dofs_vel_limit = torch.where(vel_limit.isfinite(), vel_limit, torch.full_like(vel_limit, _DEFAULT_VEL_LIMIT))
        dofs_acc_limit = torch.full_like(dofs_vel_limit, _DEFAULT_ACC_LIMIT)
        dof_reach_t = qd_to_torch(planner_info.fk.dofs.reach, copy=False)
        dof_reach_t[:] = torch.as_tensor(dof_reach_np, device=gs.device)

        # MPPI exploration scale per DOF, capped for huge ranges.
        sigma = np.minimum(0.15 * (q_limit_upper - q_limit_lower), 0.5)
        mppi_sigma_t = qd_to_torch(planner_info.mppi.sigma, copy=False)
        mppi_sigma_t[:] = torch.as_tensor(sigma, dtype=gs.tc_float, device=gs.device)
        # Runtime scalars filled once per entity from the module constants (the funcs read the fields, so their
        # host source stays a single named constant): the iteration budgets, MPPI annealing, RRT-Connect budgets
        # and steering, and the certification thresholds (see cost.py / graph.py for each).
        for field, value in (
            (planner_info.mppi.n_iters, budgets.mppi_n_iters),
            (planner_info.mppi.n_particles, budgets.mppi_n_particles),
            (planner_info.lbfgs.n_iters, budgets.lbfgs_n_iters),
            (planner_info.lbfgs.n_ls_trials, budgets.ls_n_trials),
            (planner_info.mppi.anneal, _MPPI_ANNEAL),
            (planner_info.rrt.n_iters, _N_RRT_ITERS),
            (planner_info.rrt.n_shortcut, _RRT_N_SHORTCUT),
            (planner_info.rrt.n_stall_iters, _RRT_STALL_ITERS),
            (planner_info.rrt.edge_density, solver._options.planner_edge_check_density),
            (planner_info.rrt.steer_step, _RRT_STEER_STEP),
            (planner_info.rrt.goal_bias, _RRT_GOAL_BIAS),
            (planner_info.cert.excl_depth_max, cost_mod._EXCL_DEPTH_MAX),
            (planner_info.cert.excl_contact_band, cost_mod._EXCL_CONTACT_BAND),
            (planner_info.cert.excl_anchor_slack_cert, cost_mod._EXCL_ANCHOR_SLACK_CERT),
            (planner_info.cert.excl_anchor_slack_opt, cost_mod._EXCL_ANCHOR_SLACK_OPT),
            (planner_info.cert.exact_rescue_window, cost_mod._EXACT_RESCUE_WINDOW),
            (planner_info.cert.exact_sample_cov, cost_mod._EXACT_SAMPLE_COV),
            (planner_info.cert.exact_sample_cov_coarse, cost_mod._EXACT_SAMPLE_COV_COARSE),
            (planner_info.cert.goal_real_pen_max, cost_mod._GOAL_REAL_PEN_MAX),
        ):
            field_t = qd_to_torch(field, copy=False)
            field_t.fill_(value)
        ls_ladder_t = qd_to_torch(planner_info.lbfgs.ls_ladder, copy=False)
        ls_ladder_t[:] = torch.as_tensor(_LS_LADDER, dtype=gs.tc_float, device=gs.device)

        context = _EntityContext(
            planner_config=planner_config,
            planner_info=planner_info,
            planner_state=planner_state,
            gjk_state=gjk_state,
            dofs_vel_limit=dofs_vel_limit,
            dofs_acc_limit=dofs_acc_limit,
            spheres_link_idx=spheres_link_idx,
            errno=errno,
        )
        self._entity_contexts[entity.idx] = context
        return context

    def _compute_dof_reach(self, entity, spheres_link_idx, spheres_pos_local, spheres_radius, links_verts_extent):
        """Config-independent per-DOF workspace reach bound.

        The bound is the chain-length sum of the link offsets distal of the joint, plus the largest material extent
        on that subtree, plus the prismatic ranges. It bounds how far any checked material point can move per radian
        (or meter) of that DOF - the Lipschitz constant of the forward-kinematics map used by every swept-collision
        cover. The extent covers both the proxy spheres
        and the collision-mesh vertices: mesh points can lie outside every proxy sphere (the fill proxy is not
        a strict cover), and the exact re-checks certify those points, so the sweep must bound their motion too.
        """
        children = {i_l: [] for i_l in range(entity.n_links)}
        for link in entity.links:
            if link.parent_idx >= entity.link_start:
                children[link.parent_idx - entity.link_start].append(link.idx - entity.link_start)
        links = list(entity.links)
        dof_reach = np.zeros(entity.n_qs, dtype=gs.np_float)
        for joint in entity.joints:
            if joint.n_dofs != 1:
                continue
            reach = 0.0
            subtree = []
            stack = [joint.link.idx - entity.link_start]
            while stack:
                i_l = stack.pop()
                subtree.append(i_l)
                reach += float(np.linalg.norm(links[i_l].pos))
                stack.extend(children[i_l])
            for sub_joint in entity.joints:
                if sub_joint.type == gs.JOINT_TYPE.PRISMATIC and sub_joint.link.idx - entity.link_start in subtree:
                    limit = sub_joint.dofs_limit[0]
                    if np.isfinite(limit).all():
                        reach += float(limit[1] - limit[0])
            extent = links_verts_extent[subtree].max()
            mask = np.isin(spheres_link_idx, subtree)
            if mask.any():
                extent = max(extent, (np.linalg.norm(spheres_pos_local[mask], axis=-1) + spheres_radius[mask]).max())
            dof_reach[joint.q_start - entity.q_start] = reach + extent
        return dof_reach

    def _compute_self_pairs_reach(self, entity, spheres_link_idx, self_pairs, dof_reach):
        """Per-pair relative reach bound: the summed reach of the DOFs on the path between the pair's links.

        Only those DOFs change the pair's relative pose, so this bounds their mutual approach per radian of L-inf
        joint motion far tighter than the absolute reach of the whole chain.
        """
        links = list(entity.links)
        dofs_of_link = {}
        root_chains = {}
        for i_l in range(entity.n_links):
            chain = []
            i_cur = i_l
            while i_cur >= 0:
                chain.append(i_cur)
                parent = links[i_cur].parent_idx
                i_cur = parent - entity.link_start if parent >= entity.link_start else -1
            root_chains[i_l] = chain
            dofs_of_link[i_l] = [joint.q_start - entity.q_start for joint in links[i_l].joints if joint.n_dofs == 1]
        pairs_reach = np.zeros(len(self_pairs), dtype=gs.np_float)
        for i_p, (i_sa, i_sb) in enumerate(self_pairs):
            chain_a, chain_b = root_chains[int(spheres_link_idx[i_sa])], root_chains[int(spheres_link_idx[i_sb])]
            path_links = set(chain_a) ^ set(chain_b)
            pairs_reach[i_p] = sum(dof_reach[i_dp] for i_l in path_links for i_dp in dofs_of_link[i_l])
        return pairs_reach

    # ------------------------------------------------------------------------------------
    # --------------------------------- grasp handling -----------------------------------
    # ------------------------------------------------------------------------------------

    def _capture_attachment(self, entity, held, attach_link, envs_idx):
        """Grasp transform of a held entity in the attach-link frame, from the live poses (read-only)."""
        solver = self._solver
        envs_kw = dict(envs_idx=envs_idx) if solver.n_envs > 0 else {}
        link_pos = solver.get_links_pos(attach_link.idx, **envs_kw)[..., 0, :]
        link_quat = solver.get_links_quat(attach_link.idx, **envs_kw)[..., 0, :]
        obj_pos = solver.get_links_pos(held.links[0].idx, **envs_kw)[..., 0, :]
        obj_quat = solver.get_links_quat(held.links[0].idx, **envs_kw)[..., 0, :]
        pos_offset, quat_offset = gu.inv_transform_pos_quat_by_trans_quat(obj_pos, obj_quat, link_pos, link_quat)
        if pos_offset.ndim == 1:
            pos_offset, quat_offset = pos_offset[None], quat_offset[None]
        return Attachment(entity=held, link=attach_link, pos_offset=pos_offset, quat_offset=quat_offset)

    def _detect_held_attachments(self, entity, context, envs_idx, excluded_entities):
        """Auto-grasp characterization from the live state and the planner's own sphere proxies.

        An entity is held iff proxy spheres of at least two distinct robot links touch it with an antipodal squeeze
        (some pair of contact directions opposing), which distinguishes a grasped object from one merely resting
        against a link. The attach frame is the lowest common ancestor of the squeezing links, and every DOF between
        that
        ancestor and the squeezing links is locked for the plan (the grasp transform is only rigid while the
        squeeze geometry holds). Objects carried by a single link (suction, tray) need the explicit arguments.
        """
        solver = self._solver
        envs_kw = dict(envs_idx=envs_idx) if solver.n_envs > 0 else {}
        links_idx = [link.idx for link in entity.links]
        links_pos = tensor_to_array(solver.get_links_pos(links_idx, **envs_kw))
        links_quat = tensor_to_array(solver.get_links_quat(links_idx, **envs_kw))
        if links_pos.ndim == 2:
            links_pos, links_quat = links_pos[None], links_quat[None]
        spheres_link_idx = context.spheres_link_idx
        spheres_pos_local = tensor_to_array(qd_to_torch(context.planner_info.fk.spheres.pos_local))
        spheres_radius = tensor_to_array(qd_to_torch(context.planner_info.fk.spheres.radius))
        robot_spheres = links_pos[:, spheres_link_idx] + gu.transform_by_quat(
            np.tile(spheres_pos_local, (links_pos.shape[0], 1, 1)), links_quat[:, spheres_link_idx]
        )

        attachments = []
        dof_locked = qd_to_torch(context.planner_info.fk.dofs.is_locked, copy=False)
        for held in solver.entities:
            if held is entity or held in excluded_entities:
                continue
            if len(held.links) != 1 or held.links[0].is_fixed:
                continue
            held_pos = tensor_to_array(solver.get_links_pos(held.links[0].idx, **envs_kw))[..., 0, :]
            held_quat = tensor_to_array(solver.get_links_quat(held.links[0].idx, **envs_kw))[..., 0, :]
            if held_pos.ndim == 1:
                held_pos, held_quat = held_pos[None], held_quat[None]
            held_local, held_radius = [], []
            for geom in held.links[0].geoms:
                proxy = self._get_geom_proxy(geom)
                held_local.append(proxy.pos)
                held_radius.append(proxy.radius)
            held_local = np.concatenate(held_local)
            held_radius = np.concatenate(held_radius)
            held_spheres = held_pos[:, None] + gu.transform_by_quat(
                np.tile(held_local, (held_pos.shape[0], 1, 1)), np.repeat(held_quat[:, None], len(held_local), axis=1)
            )

            # Per env: closest sphere pair per robot link, then the antipodal-squeeze test across links.
            dist = (
                np.linalg.norm(robot_spheres[:, :, None] - held_spheres[:, None, :], axis=-1)
                - spheres_radius[None, :, None]
                - held_radius[None, None, :]
            )
            is_held_any = False
            for i_b_ in range(dist.shape[0]):
                contact_links, contact_dirs = [], []
                for i_l in np.unique(spheres_link_idx):
                    sel = spheres_link_idx == i_l
                    d_link = dist[i_b_][sel]
                    if d_link.min() < _GRASP_CONTACT_TOL:
                        i_s, i_h = np.unravel_index(np.argmin(d_link), d_link.shape)
                        direction = held_spheres[i_b_, i_h] - robot_spheres[i_b_][sel][i_s]
                        norm = np.linalg.norm(direction)
                        if norm > gs.EPS:
                            contact_links.append(i_l)
                            contact_dirs.append(direction / norm)
                if len(contact_links) >= 2:
                    dirs = np.array(contact_dirs)
                    if (dirs @ dirs.T).min() < _GRASP_NORMAL_COS:
                        is_held_any = True
                        # Lowest common ancestor of the squeezing links, then lock the connecting chains.
                        ancestors = []
                        for i_l in contact_links:
                            chain = []
                            i_cur = i_l
                            while i_cur >= 0:
                                chain.append(i_cur)
                                parent = list(entity.links)[i_cur].parent_idx
                                i_cur = parent - entity.link_start if parent >= entity.link_start else -1
                            ancestors.append(chain)
                        lca = next(i_l for i_l in ancestors[0] if all(i_l in chain for chain in ancestors))
                        for chain in ancestors:
                            for i_l in chain[: chain.index(lca)]:
                                for joint in list(entity.links)[i_l].joints:
                                    if joint.n_dofs == 1:
                                        dof_locked[joint.q_start - entity.q_start] = True
                        attach_link = list(entity.links)[lca]
            if is_held_any:
                attachments.append(self._capture_attachment(entity, held, attach_link, envs_idx))
        return attachments

    # ------------------------------------------------------------------------------------
    # ------------------------------------- plan -----------------------------------------
    # ------------------------------------------------------------------------------------

    def plan(
        self,
        entity,
        qpos_goal,
        qpos_start,
        num_waypoints,
        max_retry,
        safety_margin,
        ignore_collision,
        envs_idx,
        goal_link,
        goal_pos,
        goal_quat,
        explicit_attachments,
        attach_held_entities,
        seed,
    ):
        solver = self._solver
        context = self._get_entity_context(entity)
        planner_config, planner_info, planner_state = (
            context.planner_config,
            context.planner_info,
            context.planner_state,
        )
        planner_world = self._planner_world
        W, S, n_dp = planner_config.n_knots, planner_config.n_seeds, planner_config.n_dp
        B = max(solver.n_envs, 1)
        if solver.n_envs > 0:
            envs_idx_np = tensor_to_array(solver._scene._sanitize_envs_idx(envs_idx), dtype=gs.np_int)
        else:
            envs_idx_np = np.zeros(1, dtype=gs.np_int)
        B_plan = len(envs_idx_np)

        self._plan_counter += 1
        seed_key = seed if seed is not None else (gs.SEED if gs.SEED is not None else 0) + self._plan_counter

        # Start / goal configurations (entity-local q coordinates).
        qpos_start_t = (
            entity.get_qpos(envs_idx=envs_idx)
            if qpos_start is None
            else torch.as_tensor(qpos_start, dtype=gs.tc_float, device=gs.device)
        )
        if qpos_start_t.ndim == 1:
            qpos_start_t = qpos_start_t[None].expand(B_plan, n_dp)
        has_pose_goal = goal_pos is not None or goal_quat is not None
        if goal_pos is not None:
            goal_pos = torch.as_tensor(goal_pos, dtype=gs.tc_float, device=gs.device)
            if goal_pos.ndim == 1 and solver.n_envs > 0:
                goal_pos = goal_pos[None].expand(B_plan, 3)
        if goal_quat is not None:
            goal_quat = torch.as_tensor(goal_quat, dtype=gs.tc_float, device=gs.device)
            if goal_quat.ndim == 1 and solver.n_envs > 0:
                goal_quat = goal_quat[None].expand(B_plan, 4)
        if has_pose_goal:
            # Resolved below by the certified multi-restart inverse kinematics.
            qpos_goal_t = None
        else:
            qpos_goal_t = torch.as_tensor(qpos_goal, dtype=gs.tc_float, device=gs.device)
            if qpos_goal_t.ndim == 1:
                qpos_goal_t = qpos_goal_t[None].expand(B_plan, n_dp)

        boundary_qpos_start_t = qd_to_torch(planner_info.cost.boundary.qpos_start, copy=False).T
        boundary_qpos_start_t[envs_idx_np] = qpos_start_t
        boundary_qpos_goal_t = qd_to_torch(planner_info.cost.boundary.qpos_goal, copy=False).T
        if qpos_goal_t is not None:
            boundary_qpos_goal_t[envs_idx_np] = qpos_goal_t
        else:
            # Until inverse kinematics resolves the Cartesian goal, the start stands in as the goal so every
            # goal-side consumer (boundary exclusions, joint-limit allowances) reads a well-defined
            # configuration; envs whose goal never resolves keep it, pinning their goal exclusions to the start
            # contact set.
            boundary_qpos_goal_t[envs_idx_np] = qpos_start_t
        if has_pose_goal:
            if goal_pos is not None:
                goal_pos_t = qd_to_torch(planner_info.cost.boundary.goal_pos, copy=False)
                goal_pos_t[envs_idx_np] = goal_pos
            if goal_quat is not None:
                goal_quat_t = qd_to_torch(planner_info.cost.boundary.goal_quat, copy=False)
                goal_quat_t[envs_idx_np] = goal_quat

        # Per-plan runtime scalars: goal designation, noise key, coarse cost weights and margins (the polish
        # phase tightens them below), plus the lock-mask, attach-mask, and errno clears.
        _set_plan_scalars(
            planner_info,
            context.errno,
            goal_link_idx=goal_link.idx if has_pose_goal else 0,
            seed_key=int(seed_key) & 0x7FFFFFFF,
            has_pose_goal=has_pose_goal,
            w_obs=100.0,
            w_self=100.0,
            w_lim=500.0,
            w_acc=2.0,
            w_jerk=1.0,
            w_posture=1e-3,
            w_pose_pos=2e3,
            w_pose_rot=5e2,
            eps_act=0.05,
            eps_self=0.02,
            # The optimizer plans with a headroom over the requested clearance: the certificate additionally
            # covers the between-samples sweep, so paths must arrive with that much slack to certify.
            d_safe=float(safety_margin) + 0.02,
        )
        noise_basis_t = qd_to_torch(planner_info.mppi.noise_basis, copy=False)
        noise_basis_t[:] = torch.as_tensor(build_noise_basis(W), device=gs.device)

        # Attachments: explicit first, then auto-detected held entities.
        attachments = [
            self._capture_attachment(entity, held, attach_link, envs_idx) for held, attach_link in explicit_attachments
        ]
        if attach_held_entities:
            excluded = [attachment.entity for attachment in attachments]
            attachments += self._detect_held_attachments(entity, context, envs_idx, excluded)

        attach_active = qd_to_torch(planner_info.fk.attach.is_active, copy=False)
        attach_link_idx = qd_to_torch(planner_info.fk.attach.link_idx, copy=False)
        attach_pos = qd_to_torch(planner_info.fk.attach.pos_local, copy=False)
        attach_radius = qd_to_torch(planner_info.fk.attach.radius, copy=False)
        attached_geoms = []
        i_slot = 0
        for attachment in attachments:
            for geom in attachment.entity.links[0].geoms:
                proxy = self._get_geom_proxy(geom)
                attached_geoms.append(geom.idx)
                for i_s in range(len(proxy.radius)):
                    sphere_obj = torch.as_tensor(proxy.pos[i_s], dtype=gs.tc_float, device=gs.device)
                    pos_env = attachment.pos_offset + gu.transform_by_quat(
                        sphere_obj.expand(B_plan, 3), attachment.quat_offset
                    )
                    attach_link_idx[i_slot] = attachment.link.idx - entity.link_start
                    attach_pos[i_slot, envs_idx_np] = pos_env
                    attach_radius[i_slot] = float(proxy.radius[i_s])
                    attach_active[i_slot, envs_idx_np] = True
                    i_slot += 1

        # World snapshot: every geom outside the planned entity and its attachments is an obstacle.
        obstacle_geoms_idx = np.array(
            [
                geom.idx
                for other in solver.entities
                if other is not entity
                for link in other.links
                for geom in link.geoms
                if geom.idx not in attached_geoms
            ],
            dtype=gs.np_int,
        )
        kernel_snapshot_world(
            envs_idx_np, obstacle_geoms_idx, planner_world, solver.dyn_state, solver.rigid_info, planner_config
        )
        # Grid signed distance fields answer metrically only within their padded box; analytic primitives always.
        # Convex geoms (analytic primitives except the unbounded plane, and convex meshes) additionally answer
        # the exact GJK re-checks (see geoms_is_convex in array_class.py).
        max_band = qd_to_torch(planner_world.geoms_max_band, copy=False)
        is_convex = qd_to_torch(planner_world.geoms_is_convex, copy=False)
        analytic_types = (
            gs.GEOM_TYPE.SPHERE,
            gs.GEOM_TYPE.PLANE,
            gs.GEOM_TYPE.BOX,
            gs.GEOM_TYPE.CAPSULE,
            gs.GEOM_TYPE.CYLINDER,
        )
        for i_gw, i_g in enumerate(obstacle_geoms_idx):
            geom = solver.geoms[i_g]
            if geom.type in analytic_types:
                max_band[i_gw] = np.inf
            else:
                verts = geom.init_verts
                max_band[i_gw] = 0.1 * float((verts.max(axis=0) - verts.min(axis=0)).max())
            is_convex[i_gw] = (geom.type in analytic_types and geom.type != gs.GEOM_TYPE.PLANE) or (
                geom.type == gs.GEOM_TYPE.MESH and geom.is_convex
            )

        sdf_info = solver.collider._sdf._sdf_info
        kernel_args = (
            envs_idx_np,
            planner_state,
            planner_world,
            solver.dyn_state,
            solver.collider._collider_state,
            context.gjk_state,
            planner_info,
            solver.dyn_info,
            solver.rigid_info,
            solver.collider._collider_info,
            sdf_info,
            solver.rigid_config,
            solver.collider._collider_static_config,
        )
        if not has_pose_goal:
            # Boundary (start and goal) contact exclusions: pairs violating the margin at either boundary keep
            # their worst boundary clearance for the whole plan, which is what makes grasp and place goals
            # plannable.
            kernel_boundary_exclusions(*kernel_args, planner_config, include_goal=True, errno=context.errno)
        errno_t = qd_to_torch(context.errno)
        if bool((errno_t != 0).any()):
            gs.raise_exception(
                "Too many boundary-configuration contacts for the planner exclusion lists; the start or goal "
                "configuration is deeply entangled with the world."
            )

        # Optimize, escalating on failed envs: fresh seeds first, RRT-Connect seeds from the second attempt on.
        is_active_t = qd_to_torch(planner_state.cert.is_active, copy=False).reshape(B, S)
        flags_t = qd_to_torch(planner_state.cert.valid_flags, copy=False).reshape(B, S)
        seed_key_cur = int(seed_key) & 0x7FFFFFFF
        env_solved = torch.zeros(B, dtype=torch.bool, device=gs.device)
        env_pending_mask = torch.zeros(B, dtype=torch.bool, device=gs.device)
        env_pending_mask[envs_idx_np] = True
        env_goal_resolved = torch.zeros(B, dtype=torch.bool, device=gs.device)

        # Joint and Cartesian goals run their whole attempt ladder - Cartesian goals additionally resolving the
        # goal by in-kernel multi-restart inverse kinematics - in one graph-captured launch, held attachments
        # included (the attach spheres are placed in state before the launch and the in-kernel forward kinematics,
        # collision cost, and validator already carry them). The host ladder below then only works the envs it
        # leaves unsolved. Collision-free requests stay on the host ladder, which owns the straight-line shortcut
        # the graph kernel does not cover.
        planner_state.graph_counter.from_numpy(np.array(2 + max_retry, dtype=np.int32))
        # Per-pass RRT tree-activity scratch (one entry per planned env's tree pair); the kernel rewrites it each
        # escalation pass. The graph-kernel ndarray arguments must be device-resident: the sm90+ device-side graph
        # loop rejects host ndarrays outright. The pre-sm90 host fallback would tolerate host arrays, but the fast
        # path is the one that matters, so both the env index and the tree-activity scratch live on device.
        envs_idx_dev = torch.as_tensor(envs_idx_np, dtype=gs.tc_int, device=gs.device)
        trees_is_active = torch.zeros(len(envs_idx_np) * planner_config.n_rrt_trees, dtype=gs.tc_int, device=gs.device)
        # The ladder is driven from here, one pass per iteration, ending as soon as every planned env certifies.
        # Its phases are compiled apart - goal resolution, tree escalation, and the pass itself - because each of
        # them inlines the whole collision model, and compilation grows faster than a kernel does.
        kernel_init_ladder(2 + max_retry, envs_idx_dev, planner_state)
        plan_args = (
            envs_idx_dev,
            trees_is_active,
            planner_state,
            planner_world,
            solver.dyn_state,
            solver.collider._collider_state,
            context.gjk_state,
            planner_info,
            solver.dyn_info,
            solver.rigid_info,
            solver.collider._collider_info,
            sdf_info,
            solver.rigid_config,
            solver.collider._collider_static_config,
            planner_config,
            ignore_collision,
            context.errno,
        )
        rrt_args = (
            envs_idx_dev,
            trees_is_active,
            planner_state,
            planner_world,
            solver.dyn_state,
            solver.collider._collider_state,
            context.gjk_state,
            planner_info,
            solver.dyn_info,
            solver.rigid_info,
            solver.collider._collider_info,
            sdf_info,
            solver.rigid_config,
            solver.collider._collider_static_config,
            planner_config,
        )
        for _ in range(2 + max_retry):
            if has_pose_goal:
                kernel_resolve_goal(
                    envs_idx_dev,
                    planner_state,
                    planner_world,
                    solver.dyn_state,
                    solver.collider._collider_state,
                    context.gjk_state,
                    planner_info,
                    solver.dyn_info,
                    solver.rigid_info,
                    solver.collider._collider_info,
                    sdf_info,
                    solver.rigid_config,
                    solver.collider._collider_static_config,
                    planner_config,
                    ignore_collision,
                    context.errno,
                )
            if not ignore_collision:
                kernel_rrt_escalate(*rrt_args)
            planner_state.graph_counter.from_numpy(np.array(1, dtype=np.int32))
            kernel_plan(*plan_args)
            # Certify at the user clearance, not the optimizer headroom: the optimizer plans with an extra
            # headroom so refined paths arrive with slack, but an env is solved once its trajectory clears the
            # requested margin (the validator adds the swept Lipschitz allowance on top, so this still covers the
            # continuous path). A tight fallback corridor is feasible at the margin yet not at the headroom, so
            # folding at the headroom would leave every cluttered env unsolved.
            _set_clearance(planner_info, float(safety_margin))
            # The straightening pass below certifies through this same kernel; passing the env indices the same
            # way keeps both on one compiled instantiation.
            kernel_validate(
                envs_idx_np,
                planner_state,
                planner_world,
                solver.dyn_state,
                solver.collider._collider_state,
                context.gjk_state,
                planner_info,
                solver.dyn_info,
                solver.rigid_info,
                solver.collider._collider_info,
                sdf_info,
                solver.rigid_config,
                solver.collider._collider_static_config,
                planner_config,
                True,
                1,
            )
            _set_clearance(planner_info, float(safety_margin) + 0.02)
            kernel_fold(planner_state.graph_counter, envs_idx_dev, planner_state, planner_config, ignore_collision)
            if bool((qd_to_torch(planner_state.is_env_solved) != 0)[envs_idx_np].all()):
                break
        errno_t = qd_to_torch(context.errno)
        if bool((errno_t != 0).any()):
            gs.raise_exception(
                "Too many boundary-configuration contacts for the planner exclusion lists; the start or goal "
                "configuration is deeply entangled with the world."
            )
        env_solved |= (qd_to_torch(planner_state.is_env_solved) != 0) & env_pending_mask
        # A graph-solved env certified a plan to its resolved goal in-kernel, so its goal is resolved by
        # construction; without this the host-only goal-resolved bookkeeping would stay clear for those envs
        # and the post-ladder unreachable-goal stamp below would flag them GOAL_IN_COLLISION, corrupting the
        # very verdicts the graph kernel just certified.
        env_goal_resolved |= env_solved

        # Envs whose goal never yielded an acceptable branch carry the dedicated code on every candidate: the
        # validator rewrites the flags each attempt, so the code is stamped once the flags are final, and it
        # holds whatever stale content their never-seeded candidates carry away from certification.
        if has_pose_goal:
            env_goal_invalid = env_pending_mask & ~env_goal_resolved
            if bool(env_goal_invalid.any()):
                gs.logger.info(
                    f"Cartesian goal unreachable or in collision in {int(env_goal_invalid.sum())} environment(s)."
                )
                flags_t[env_goal_invalid] |= cost_mod.GOAL_IN_COLLISION

        # Per-env best certified seed (lowest cost, lowest index tie-break), start-hold for failed envs.
        is_seed_valid = self._seed_validity(flags_t, S, ignore_collision)
        costs = qd_to_torch(planner_state.cost.cost).reshape(B, S).nan_to_num(nan=1e30, posinf=1e30)
        costs_gated = torch.where(is_seed_valid, costs, torch.full_like(costs, torch.inf))
        best_seed = costs_gated.argmin(dim=-1)
        is_env_valid = is_seed_valid.gather(-1, best_seed[:, None])[:, 0]

        knots = qd_to_torch(planner_state.cost.qpos).permute(1, 2, 0).reshape(B, S, W, n_dp)
        knots_best = knots.gather(1, best_seed[:, None, None, None].expand(B, 1, W, n_dp))[:, 0]
        if not ignore_collision and bool(is_env_valid.any()):
            # Certified smoothing polish: each valid env's winning candidate alone takes one more refinement
            # round with the smoothness weights boosted (collision weights untouched), then re-certifies. An
            # env keeps the smoothed trajectory only where the fresh certificate holds, so the polish can never
            # trade a certified plan away for smoothness.
            is_active_polish = torch.zeros(B, S, dtype=torch.bool, device=gs.device)
            is_active_polish[torch.arange(B, device=gs.device), best_seed] = is_env_valid
            is_active_t[...] = is_active_polish
            _set_opt_phase(
                planner_info,
                seed_key=seed_key_cur,
                w_obs=100.0,
                w_self=100.0,
                w_acc=30.0,
                w_jerk=15.0,
                w_pose_pos=2e3,
                w_pose_rot=5e2,
                eps_act=0.05,
            )
            kernel_lbfgs(*kernel_args, planner_config)
            _set_opt_phase(
                planner_info,
                seed_key=seed_key_cur,
                w_obs=100.0,
                w_self=100.0,
                w_acc=2.0,
                w_jerk=1.0,
                w_pose_pos=2e3,
                w_pose_rot=5e2,
                eps_act=0.05,
            )
            _set_clearance(planner_info, float(safety_margin))
            kernel_validate(*kernel_args, planner_config, check_start=True, is_swept=1)
            is_env_smooth = self._seed_validity(flags_t, S, ignore_collision).gather(-1, best_seed[:, None])[:, 0]
            knots = qd_to_torch(planner_state.cost.qpos).permute(1, 2, 0).reshape(B, S, W, n_dp)
            knots_smooth = knots.gather(1, best_seed[:, None, None, None].expand(B, 1, W, n_dp))[:, 0]
            knots_best = torch.where((is_env_smooth & is_env_valid)[:, None, None], knots_smooth, knots_best)
            # Certified straightening ladder: near-collinear knot runs snap onto their chords, so the retimed
            # profile cruises between the few essential corners instead of slowing at every residual knot
            # wiggle. The snapped polyline is a knot trajectory like any other, so the same certificate gates
            # it per env; envs whose coarse straightening fails retry at the finer radii (see _STRAIGHTEN_RADII).
            traj_t = qd_to_torch(planner_state.cost.qpos, copy=False).permute(1, 2, 0).reshape(B, S, W, n_dp)
            env_rough = is_env_valid.clone()
            for radius in _STRAIGHTEN_RADII:
                if not bool(env_rough.any()):
                    break
                knots_straight = _straighten_knots(knots_best, qd_to_torch(planner_info.fk.dofs.reach), radius)
                traj_t[env_rough, best_seed[env_rough]] = knots_straight[env_rough]
                kernel_validate(*kernel_args, planner_config, check_start=True, is_swept=1)
                is_env_straight = (
                    self._seed_validity(flags_t, S, ignore_collision).gather(-1, best_seed[:, None])[:, 0] & env_rough
                )
                knots_best = torch.where(is_env_straight[:, None, None], knots_straight, knots_best)
                env_rough &= ~is_env_straight
            _set_clearance(planner_info, float(safety_margin) + 0.02)
        start_hold = qd_to_torch(planner_info.cost.boundary.qpos_start).T[:, None, :].expand(B, W, n_dp)
        knots_best = torch.where(is_env_valid[:, None, None], knots_best, start_hold)

        knots_plan = knots_best[envs_idx_np]
        qpos_out, vel_out, acc_out, dt_out = retime_trajectory(
            knots_plan,
            context.dofs_vel_limit,
            context.dofs_acc_limit,
            qd_to_torch(planner_info.fk.dofs.reach),
            num_waypoints,
            scene_dt=solver._scene.dt,
        )

        is_valid_out = is_env_valid[envs_idx_np]
        n_failed = int((~is_valid_out).sum())
        if n_failed > 0:
            gs.logger.info(f"Motion planning failed in {n_failed} environments.")

        if solver.n_envs == 0:
            return PlannerPath(
                qpos=qpos_out[0], dofs_vel=vel_out[0], dofs_acc=acc_out[0], dt=dt_out[0], is_valid=is_valid_out[0]
            )
        return PlannerPath(qpos=qpos_out, dofs_vel=vel_out, dofs_acc=acc_out, dt=dt_out, is_valid=is_valid_out)

    def _seed_validity(self, flags_t, S, ignore_collision):
        flags = flags_t.reshape(-1, S)
        if ignore_collision:
            return (flags & ~(cost_mod.COLLISION | cost_mod.GOAL_TOL)) == 0
        return flags == 0
