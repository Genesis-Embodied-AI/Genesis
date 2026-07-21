import dataclasses
from typing import NamedTuple

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class
from genesis.utils.misc import qd_to_torch, tensor_to_array

from . import cost as cost_mod
from .graph import kernel_planner_rrt_connect
from .retime import resample_trajectory, retime_trajectory
from .sphere_proxy import build_geom_sphere_proxy
from .trajopt import build_noise_basis, kernel_planner_lbfgs, kernel_planner_mppi
from .world import kernel_planner_snapshot_world

# Sphere-fill budgets (per geom) and default derivative limits when the asset carries none.
_N_MAX_SPHERES_PER_GEOM = 8
_SPHERE_PAD = 0.005
_DEFAULT_VEL_LIMIT = np.pi
_DEFAULT_ACC_LIMIT = 2.0 * np.pi
_DEFAULT_JERK_LIMIT = 50.0 * np.pi
# Contact tolerance and antipodal-squeeze threshold of the auto-grasp detection.
_GRASP_CONTACT_TOL = 0.01
_GRASP_NORMAL_COS = -0.5
# RRT-Connect fallback extents and iteration budget.
_N_RRT_TREES = 4
_N_RRT_NODES = 2048
_N_RRT_ITERS = 2000


@dataclasses.dataclass(frozen=True)
class PlannerPath:
    """
    Time-parametrized planned trajectory.

    Waypoints are spaced ``dt`` seconds apart per env; ``is_valid`` marks the envs whose path was certified
    collision-free at the requested safety margin - paths of failed envs hold the start configuration at rest and
    must not be executed.
    """

    qpos: torch.Tensor
    dofs_vel: torch.Tensor
    dofs_acc: torch.Tensor
    dt: torch.Tensor
    is_valid: torch.Tensor


class PlannerAttachment(NamedTuple):
    """One entity carried rigidly during the plan: the attach link and the per-env grasp transform."""

    entity: object
    link: object
    pos_offset: torch.Tensor
    quat_offset: torch.Tensor


class _EntityContext(NamedTuple):
    """Per-entity planner buffers, allocated once at the entity's first plan and reused forever."""

    planner_config: object
    plan_info: object
    plan_state: object
    spheres_link_idx: np.ndarray
    errno: object


class Planner:
    """
    Optimization-based motion planner of the rigid solver (see plan_path for the user entrypoint).

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
        self._plan_world = array_class.get_planner_world_state(solver.n_geoms, max(solver.n_envs, 1))

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
                n_seeds = min(max(4096 // B, 12), 64)
        if arm == gs.planner_arm.SERIAL:
            budgets = dict(mppi_n_iters=6, mppi_n_particles=4, lbfgs_n_iters=48, ls_n_trials=8)
        else:
            budgets = dict(mppi_n_iters=12, mppi_n_particles=8, lbfgs_n_iters=48, ls_n_trials=4)
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

        # Robot collision proxy, sorted by link.
        spheres_link_idx, spheres_pos_local, spheres_radius = [], [], []
        for link in entity.links:
            for geom in link.geoms:
                proxy = self._get_geom_proxy(geom)
                spheres_link_idx.extend([link.idx - entity.link_start] * len(proxy.radius))
                spheres_pos_local.append(proxy.pos)
                spheres_radius.append(proxy.radius)
        spheres_link_idx = np.array(spheres_link_idx, dtype=gs.np_int)
        spheres_pos_local = np.concatenate(spheres_pos_local, dtype=gs.np_float)
        spheres_radius = np.concatenate(spheres_radius, dtype=gs.np_float)

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

        # Attached-sphere capacity: every proxy sphere of every single-link free entity in the scene.
        n_attach_max = sum(
            len(self._get_geom_proxy(geom).radius)
            for other in solver.entities
            if other is not entity and len(other.links) == 1 and not other.links[0].is_fixed
            for geom in other.links[0].geoms
        )

        dof_reach_np = self._compute_dof_reach(entity, spheres_link_idx, spheres_pos_local, spheres_radius)

        arm, n_seeds, budgets = self._resolve_budgets(B)
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
            n_eval_per_candidate=1,
            n_rrt_trees=_N_RRT_TREES,
            n_rrt_nodes=_N_RRT_NODES,
        )
        plan_info = array_class.get_planner_entity_info(planner_config, len(self_pairs), B)
        plan_state = array_class.get_planner_state(planner_config, B)
        errno = array_class.V(dtype=gs.qd_int, shape=(B,))

        qd_to_torch(plan_info.spheres_link_idx, copy=False)[:] = torch.as_tensor(spheres_link_idx, device=gs.device)
        qd_to_torch(plan_info.spheres_pos_local, copy=False)[:] = torch.as_tensor(spheres_pos_local, device=gs.device)
        qd_to_torch(plan_info.spheres_radius, copy=False)[:] = torch.as_tensor(spheres_radius, device=gs.device)
        if self_pairs:
            qd_to_torch(plan_info.self_pairs, copy=False)[:] = torch.as_tensor(
                np.array(self_pairs, dtype=gs.np_int), device=gs.device
            )
            qd_to_torch(plan_info.self_pairs_reach, copy=False)[:] = torch.as_tensor(
                self._compute_self_pairs_reach(entity, spheres_link_idx, self_pairs, dof_reach_np), device=gs.device
            )

        # Joint-space box limits and derivative limits (model velocity limits, defaults when the asset has none).
        q_limit_lower, q_limit_upper = entity.q_limit
        qd_to_torch(plan_info.q_limit_lower, copy=False)[:] = torch.as_tensor(
            q_limit_lower, dtype=gs.tc_float, device=gs.device
        )
        qd_to_torch(plan_info.q_limit_upper, copy=False)[:] = torch.as_tensor(
            q_limit_upper, dtype=gs.tc_float, device=gs.device
        )
        vel_limit = entity.get_dofs_vel_limit()
        if vel_limit.ndim > 1:
            vel_limit = vel_limit[0]
        qd_to_torch(plan_info.vel_limit, copy=False)[:] = torch.where(
            vel_limit.isfinite(), vel_limit, torch.full_like(vel_limit, _DEFAULT_VEL_LIMIT)
        )
        qd_to_torch(plan_info.acc_limit, copy=False).fill_(_DEFAULT_ACC_LIMIT)
        qd_to_torch(plan_info.jerk_limit, copy=False).fill_(_DEFAULT_JERK_LIMIT)
        qd_to_torch(plan_info.dof_reach, copy=False)[:] = torch.as_tensor(dof_reach_np, device=gs.device)

        # MPPI exploration scale per DOF, capped for huge ranges.
        sigma = np.minimum(0.15 * (np.asarray(q_limit_upper) - np.asarray(q_limit_lower)), 0.5)
        qd_to_torch(plan_info.mppi_sigma, copy=False)[:] = torch.as_tensor(sigma, dtype=gs.tc_float, device=gs.device)
        for name, value in budgets.items():
            qd_to_torch(getattr(plan_info, name), copy=False).fill_(value)

        context = _EntityContext(
            planner_config=planner_config,
            plan_info=plan_info,
            plan_state=plan_state,
            spheres_link_idx=spheres_link_idx,
            errno=errno,
        )
        self._entity_contexts[entity.idx] = context
        return context

    def _compute_dof_reach(self, entity, spheres_link_idx, spheres_pos_local, spheres_radius):
        """
        Config-independent per-DOF workspace reach bound: chain-length sum of the link offsets distal of the
        joint, plus the largest sphere offset + radius on that subtree, plus the prismatic ranges. Bounds how far
        any proxy point can move per radian (or meter) of that DOF - the Lipschitz constant of the forward-
        kinematics map used by every swept-collision cover.
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
            mask = np.isin(spheres_link_idx, subtree)
            if mask.any():
                reach += float((np.linalg.norm(spheres_pos_local[mask], axis=-1) + spheres_radius[mask]).max())
            dof_reach[joint.q_start - entity.q_start] = reach
        return dof_reach

    def _compute_self_pairs_reach(self, entity, spheres_link_idx, self_pairs, dof_reach):
        """
        Per-pair relative reach bound: the sum of the reach of the DOFs on the kinematic path between the pair's
        links - only those DOFs change the pair's relative pose, so this bounds their mutual approach per radian
        of L-inf joint motion far tighter than the absolute reach of the whole chain.
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
        return PlannerAttachment(entity=held, link=attach_link, pos_offset=pos_offset, quat_offset=quat_offset)

    def _detect_held_attachments(self, entity, context, envs_idx, excluded_entities):
        """
        Auto-grasp characterization from the live state and the planner's own sphere proxies: an entity is held
        iff proxy spheres of at least two distinct robot links touch it with an antipodal squeeze (some pair of
        contact directions opposing), which distinguishes a grasped object from one merely resting against a
        link. The attach frame is the lowest common ancestor of the squeezing links, and every DOF between that
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
        spheres_pos_local = tensor_to_array(qd_to_torch(context.plan_info.spheres_pos_local))
        spheres_radius = tensor_to_array(qd_to_torch(context.plan_info.spheres_radius))
        robot_spheres = links_pos[:, spheres_link_idx] + gu.transform_by_quat(
            np.tile(spheres_pos_local, (links_pos.shape[0], 1, 1)), links_quat[:, spheres_link_idx]
        )

        attachments = []
        dof_locked = qd_to_torch(context.plan_info.dof_is_locked, copy=False)
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
        planner_config, plan_info, plan_state = context.planner_config, context.plan_info, context.plan_state
        plan_world = self._plan_world
        W, S, n_dp = planner_config.n_knots, planner_config.n_seeds, planner_config.n_dp
        B = max(solver.n_envs, 1)
        if solver.n_envs > 0:
            envs_idx_np = tensor_to_array(solver._scene._sanitize_envs_idx(envs_idx), dtype=gs.np_int)
        else:
            envs_idx_np = np.zeros(1, dtype=gs.np_int)
        B_plan = len(envs_idx_np)

        self._plan_counter += 1
        seed_key = seed if seed is not None else (gs.SEED if gs.SEED is not None else 0) + self._plan_counter
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed_key))
        qd_to_torch(plan_info.seed_key, copy=False).fill_(int(seed_key) & 0x7FFFFFFF)

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
        if goal_quat is not None:
            goal_quat = torch.as_tensor(goal_quat, dtype=gs.tc_float, device=gs.device)
        if has_pose_goal:
            ik_kwargs = {}
            if goal_pos is not None:
                ik_kwargs["pos"] = goal_pos
            if goal_quat is not None:
                ik_kwargs["quat"] = goal_quat
            if solver.n_envs > 0:
                ik_kwargs["envs_idx"] = envs_idx
            qpos_goal_t = entity.inverse_kinematics(goal_link, **ik_kwargs)
        else:
            qpos_goal_t = torch.as_tensor(qpos_goal, dtype=gs.tc_float, device=gs.device)
        if qpos_goal_t.ndim == 1:
            qpos_goal_t = qpos_goal_t[None].expand(B_plan, n_dp)

        qd_to_torch(plan_info.qpos_start, copy=False).T[envs_idx_np] = qpos_start_t
        qd_to_torch(plan_info.qpos_goal, copy=False).T[envs_idx_np] = qpos_goal_t
        qd_to_torch(plan_info.has_pose_goal, copy=False).fill_(has_pose_goal)
        if has_pose_goal:
            qd_to_torch(plan_info.goal_link_idx, copy=False).fill_(goal_link.idx)
            if goal_pos is not None:
                qd_to_torch(plan_info.goal_pos, copy=False)[envs_idx_np] = goal_pos
            if goal_quat is not None:
                qd_to_torch(plan_info.goal_quat, copy=False)[envs_idx_np] = goal_quat

        # Cost weights and margins (coarse phase; the polish phase tightens them below).
        for name, value in (
            ("w_obs", 100.0),
            ("w_self", 100.0),
            ("w_lim", 500.0),
            ("w_acc", 2.0),
            ("w_jerk", 1.0),
            ("w_posture", 1e-3),
            ("w_pose_pos", 2e3),
            ("w_pose_rot", 5e2),
            ("eps_act", 0.05),
            ("eps_self", 0.02),
            ("d_safe", float(safety_margin)),
        ):
            qd_to_torch(getattr(plan_info, name), copy=False).fill_(value)
        qd_to_torch(plan_info.noise_basis, copy=False)[:] = torch.as_tensor(build_noise_basis(W), device=gs.device)

        # Attachments: explicit first, then auto-detected held entities.
        qd_to_torch(plan_info.dof_is_locked, copy=False).fill_(False)
        attachments = [
            self._capture_attachment(entity, held, attach_link, envs_idx) for held, attach_link in explicit_attachments
        ]
        if attach_held_entities:
            excluded = [attachment.entity for attachment in attachments]
            attachments += self._detect_held_attachments(entity, context, envs_idx, excluded)

        attach_active = qd_to_torch(plan_info.attach_spheres_is_active, copy=False)
        attach_active.fill_(False)
        attach_link_idx = qd_to_torch(plan_info.attach_spheres_link_idx, copy=False)
        attach_pos = qd_to_torch(plan_info.attach_spheres_pos_local, copy=False)
        attach_radius = qd_to_torch(plan_info.attach_spheres_radius, copy=False)
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
        kernel_planner_snapshot_world(
            envs_idx_np, obstacle_geoms_idx, plan_world, solver.dyn_state, solver.rigid_info, planner_config
        )
        # Grid signed distance fields answer metrically only within their padded box; analytic primitives always.
        max_band = qd_to_torch(plan_world.geoms_max_band, copy=False)
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
                verts = np.asarray(geom.init_verts)
                max_band[i_gw] = 0.1 * float((verts.max(axis=0) - verts.min(axis=0)).max())

        sdf_info = solver.collider._sdf._sdf_info
        errno_t = qd_to_torch(context.errno, copy=False)
        errno_t.fill_(0)
        kernel_args = (
            envs_idx_np,
            plan_state,
            plan_info,
            plan_world,
            solver.dyn_state,
            solver.dyn_info,
            solver.rigid_info,
            sdf_info,
            solver.rigid_config,
        )
        cost_mod.kernel_planner_start_exclusions(
            envs_idx_np,
            plan_state,
            plan_info,
            plan_world,
            solver.dyn_state,
            solver.dyn_info,
            solver.rigid_info,
            sdf_info,
            solver.rigid_config,
            planner_config,
            context.errno,
        )
        if bool((errno_t != 0).any()):
            gs.raise_exception(
                "Too many start-configuration contacts for the planner exclusion lists; the start pose is deeply "
                "entangled with the world."
            )

        # Optimize, escalating on failed envs: fresh seeds first, RRT-Connect seeds from the second attempt on.
        is_active_t = qd_to_torch(plan_state.is_active, copy=False)
        flags_t = qd_to_torch(plan_state.valid_flags, copy=False)
        env_solved = torch.zeros(B, dtype=torch.bool, device=gs.device)
        env_pending_mask = torch.zeros(B, dtype=torch.bool, device=gs.device)
        env_pending_mask[envs_idx_np] = True

        for i_attempt in range(2 + max_retry):
            env_pending = env_pending_mask & ~env_solved
            if not bool(env_pending.any()):
                break
            if i_attempt == 0:
                self._seed_trajectories(context, gen, env_pending)
            elif i_attempt == 1 and not ignore_collision:
                self._seed_from_rrt(context, envs_idx_np, env_pending, kernel_args)
            else:
                qd_to_torch(plan_info.seed_key, copy=False).fill_((int(seed_key) + i_attempt) & 0x7FFFFFFF)
                self._seed_trajectories(context, gen, env_pending)

            if not ignore_collision:
                kernel_planner_mppi(*kernel_args, planner_config)
                kernel_planner_lbfgs(*kernel_args, solver.collider._collider_static_config, planner_config)
                # Polish: tighter activation band, harder collision weights, and a pinning pose weight sharpen
                # the margin and land the free end knot inside the goal tolerance.
                qd_to_torch(plan_info.eps_act, copy=False).fill_(0.02)
                qd_to_torch(plan_info.w_obs, copy=False).fill_(400.0)
                qd_to_torch(plan_info.w_self, copy=False).fill_(400.0)
                qd_to_torch(plan_info.w_pose_pos, copy=False).fill_(4e4)
                qd_to_torch(plan_info.w_pose_rot, copy=False).fill_(1e4)
                kernel_planner_lbfgs(*kernel_args, solver.collider._collider_static_config, planner_config)
                qd_to_torch(plan_info.eps_act, copy=False).fill_(0.05)
                qd_to_torch(plan_info.w_obs, copy=False).fill_(100.0)
                qd_to_torch(plan_info.w_self, copy=False).fill_(100.0)
                qd_to_torch(plan_info.w_pose_pos, copy=False).fill_(2e3)
                qd_to_torch(plan_info.w_pose_rot, copy=False).fill_(5e2)

            cost_mod.kernel_planner_validate(*kernel_args, planner_config)
            is_seed_valid = self._seed_validity(flags_t, S, ignore_collision)
            env_solved |= is_seed_valid.any(dim=-1) & env_pending_mask
            # Freeze the candidates of solved envs so later attempts cannot disturb them.
            is_active_t.reshape(B, S)[env_solved] = False
            if ignore_collision:
                env_solved |= env_pending_mask
                break

        # Per-env best certified seed (lowest cost, lowest index tie-break), start-hold for failed envs.
        is_seed_valid = self._seed_validity(flags_t, S, ignore_collision)
        costs = qd_to_torch(plan_state.cost).reshape(B, S)
        costs_gated = torch.where(is_seed_valid, costs, torch.full_like(costs, torch.inf))
        best_seed = costs_gated.argmin(dim=-1)
        is_env_valid = is_seed_valid.gather(-1, best_seed[:, None])[:, 0]

        knots = qd_to_torch(plan_state.qpos_traj).T.reshape(B, S, W, n_dp)
        knots_best = knots.gather(1, best_seed[:, None, None, None].expand(B, 1, W, n_dp))[:, 0]
        start_hold = qd_to_torch(plan_info.qpos_start).T[:, None, :].expand(B, W, n_dp)
        knots_best = torch.where(is_env_valid[:, None, None], knots_best, start_hold)

        knots_plan = knots_best[envs_idx_np]
        dt_knot = retime_trajectory(
            knots_plan,
            qd_to_torch(plan_info.vel_limit),
            qd_to_torch(plan_info.acc_limit),
            qd_to_torch(plan_info.jerk_limit),
        )
        qpos_out, vel_out, acc_out, dt_out = resample_trajectory(
            knots_plan, dt_knot, num_waypoints, scene_dt=solver._scene.dt
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

    def _seed_trajectories(self, context, gen, env_pending):
        """Straight-line seed plus smooth-noise variants for every pending env (torch RNG, deterministic)."""
        planner_config, plan_info, plan_state = context.planner_config, context.plan_info, context.plan_state
        W, S, n_dp = planner_config.n_knots, planner_config.n_seeds, planner_config.n_dp
        start = qd_to_torch(plan_info.qpos_start).T[:, None, :]
        goal = qd_to_torch(plan_info.qpos_goal).T[:, None, :]
        alpha = torch.linspace(0.0, 1.0, W, dtype=gs.tc_float, device=gs.device)[None, :, None]
        line = start * (1 - alpha) + goal * alpha
        traj = line[:, None].repeat(1, S, 1, 1)
        basis = torch.as_tensor(build_noise_basis(W), device=gs.device)
        noise = torch.randn((traj.shape[0], S - 1, basis.shape[1], n_dp), generator=gen, dtype=gs.tc_float).to(
            gs.device
        )
        traj[:, 1:] += 0.5 * torch.einsum("wk,bskd->bswd", basis, noise)
        locked = qd_to_torch(plan_info.dof_is_locked).T
        traj = torch.where(locked[:, None, None, :], start[:, None], traj)
        traj = traj.clamp(qd_to_torch(plan_info.q_limit_lower), qd_to_torch(plan_info.q_limit_upper))
        traj[:, :, :2] = start[:, None]
        traj[:, :, -2:] = goal[:, None]

        qd_to_torch(plan_state.qpos_traj, copy=False).T.reshape(-1, S, W, n_dp)[env_pending] = traj[env_pending]
        qd_to_torch(plan_state.is_active, copy=False).reshape(-1, S)[env_pending] = True

    def _seed_from_rrt(self, context, envs_idx_np, env_pending, kernel_args):
        """RRT-Connect on the failed envs; a solved tree re-seeds every candidate of its env."""
        planner_config, plan_info, plan_state = context.planner_config, context.plan_info, context.plan_state
        W, S, n_dp = planner_config.n_knots, planner_config.n_seeds, planner_config.n_dp
        NT, NN = planner_config.n_rrt_trees, planner_config.n_rrt_nodes

        trees_is_active = np.repeat(tensor_to_array(env_pending[envs_idx_np], dtype=np.bool_), NT).astype(gs.np_int)
        kernel_planner_rrt_connect(kernel_args[0], trees_is_active, _N_RRT_ITERS, *kernel_args[1:], planner_config)

        rrt_done = qd_to_torch(plan_state.rrt_is_done)
        rrt_len = qd_to_torch(plan_state.rrt_path_len)
        rrt_path = qd_to_torch(plan_state.rrt_path).T.reshape(-1, NN, n_dp)
        traj_all = qd_to_torch(plan_state.qpos_traj, copy=False).T.reshape(-1, S, W, n_dp)
        for i_b_, i_b in enumerate(envs_idx_np):
            if not bool(env_pending[i_b]):
                continue
            for i_t in range(i_b_ * NT, (i_b_ + 1) * NT):
                if bool(rrt_done[i_t]) and int(rrt_len[i_t]) >= 2:
                    n_path = int(rrt_len[i_t])
                    path = rrt_path[i_t, :n_path]
                    # Arclength resample of the certified path to the knot count.
                    seg = (path[1:] - path[:-1]).norm(dim=-1)
                    s_cum = torch.cat([torch.zeros(1, dtype=gs.tc_float, device=gs.device), seg.cumsum(0)])
                    s_tgt = torch.linspace(0.0, float(s_cum[-1]), W, dtype=gs.tc_float, device=gs.device)
                    i_seg = (torch.searchsorted(s_cum, s_tgt, right=True) - 1).clamp(min=0, max=n_path - 2)
                    seg_len = (s_cum[i_seg + 1] - s_cum[i_seg]).clamp(min=1e-9)
                    u = ((s_tgt - s_cum[i_seg]) / seg_len)[:, None]
                    traj_all[i_b, :] = (path[i_seg] * (1 - u) + path[i_seg + 1] * u)[None]
                    break
        qd_to_torch(plan_state.is_active, copy=False).reshape(-1, S)[tensor_to_array(env_pending, dtype=np.bool_)] = (
            True
        )
