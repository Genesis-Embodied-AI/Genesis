from abc import ABC, abstractmethod
import time
from typing import TYPE_CHECKING

import gstaichi as ti
import torch
import torch.nn.functional as F

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils import array_class

if TYPE_CHECKING:
    from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver


class PathPlanner(ABC):
    def __init__(self, entity):
        self._entity = entity
        self._solver: "RigidSolver" = entity._solver

        self.PENETRATION_EPS = 1e-5 if gs.ti_float == ti.f32 else 0.0

        for joint in entity.joints:
            if joint.type == gs.JOINT_TYPE.FREE:
                gs.raise_exception("planning for the gs.JOINT_TYPE.FREE is not supported (yet)")
            elif joint.type == gs.JOINT_TYPE.SPHERICAL:
                gs.raise_exception("planning for the gs.JOINT_TYPE.SPHERICAL is not supported (yet)")

    @abstractmethod
    def plan(
        self,
        qpos_goal,
        qpos_start=None,
    ): ...

    def get_link_pose(self, robot_g_link_idx, obj_g_link_idx, envs_idx):
        """
        Get the relative pose of a given robot link wrt some object link.

        Parameters
        ----------
        robot_g_link_idx: int
            Global link idx of the link of the robot.
        obj_g_link_idx: int
            Global link idx of the base link of the object.
        """
        if self._solver.n_envs > 0:
            robot_trans = self._solver.get_links_pos(links_idx=robot_g_link_idx, envs_idx=envs_idx)
            robot_quat = self._solver.get_links_quat(links_idx=robot_g_link_idx, envs_idx=envs_idx)
            obj_trans = self._solver.get_links_pos(links_idx=obj_g_link_idx, envs_idx=envs_idx)
            obj_quat = self._solver.get_links_quat(links_idx=obj_g_link_idx, envs_idx=envs_idx)
        else:
            robot_trans = self._solver.get_links_pos(links_idx=robot_g_link_idx)
            robot_quat = self._solver.get_links_quat(links_idx=robot_g_link_idx)
            obj_trans = self._solver.get_links_pos(links_idx=obj_g_link_idx)
            obj_quat = self._solver.get_links_quat(links_idx=obj_g_link_idx)

        trans = gu.inv_transform_by_trans_quat(obj_trans, robot_trans, robot_quat)
        quat = gu.transform_quat_by_quat(obj_quat, gu.inv_quat(robot_quat))

        return trans, quat

    def update_object(self, ee_link_idx, obj_link_idx, _pos, _quat, envs_idx):
        if self._solver.n_envs > 0:
            robot_trans = self._solver.get_links_pos(ee_link_idx, envs_idx=envs_idx)
            robot_quat = self._solver.get_links_quat(ee_link_idx, envs_idx=envs_idx)
        else:
            robot_trans = self._solver.get_links_pos(ee_link_idx)
            robot_quat = self._solver.get_links_quat(ee_link_idx)

        trans, quat = gu.transform_pos_quat_by_trans_quat(_pos, _quat, robot_trans, robot_quat)

        if self._solver.n_envs > 0:
            self._solver.set_base_links_pos(trans, obj_link_idx, envs_idx=envs_idx)
            self._solver.set_base_links_quat(quat, obj_link_idx, envs_idx=envs_idx)
        else:
            self._solver.set_base_links_pos(trans, obj_link_idx)
            self._solver.set_base_links_quat(quat, obj_link_idx)

    # ------------------------------------------------------------------------------------
    # ------------------------------ util funcs ------------------------------------------
    # ------------------------------------------------------------------------------------

    def _sanitize_qposs(self, qpos_goal, qpos_start, envs_idx):
        qpos_cur = self._entity.get_qpos(envs_idx=envs_idx)

        qpos_goal, _, _ = self._solver._sanitize_1D_io_variables(
            qpos_goal, None, self._entity.n_qs, envs_idx, idx_name="qpos_idx", skip_allocation=True
        )
        if qpos_start is None:
            qpos_start = qpos_cur
        qpos_start, _, envs_idx = self._solver._sanitize_1D_io_variables(
            qpos_start, None, self._entity.n_qs, envs_idx, idx_name="qpos_idx", skip_allocation=True
        )
        if self._solver.n_envs == 0:
            qpos_goal = qpos_goal.unsqueeze(0)
            qpos_start = qpos_start.unsqueeze(0)

        return qpos_cur, qpos_goal, qpos_start, envs_idx

    def get_exclude_geom_pairs(self, qposs, envs_idx):
        """
        Parameters
        ----------
        qposs : list of torch.Tensor
            List of qpos tensors to ignore the collision check.
        envs_idx : torch.Tensor
            Environment indices.

        Returns
        -------
        unique_pairs : torch.Tensor
            Unique pairs of geom indices to ignore the collision check.
        """
        collision_pairs = []
        for qpos in qposs:
            if self._solver.n_envs > 0:
                self._entity.set_qpos(qpos, envs_idx=envs_idx, zero_velocity=False)
            else:
                self._entity.set_qpos(qpos[0], zero_velocity=False)
            self._solver._kernel_detect_collision()
            scene_contact_info = self._entity.get_contacts()
            geom_a = scene_contact_info["geom_a"]
            geom_b = scene_contact_info["geom_b"]
            if self._solver.n_envs > 0:
                valid_mask = scene_contact_info["valid_mask"]
                geom_a = geom_a[valid_mask]
                geom_b = geom_b[valid_mask]
            collision_pairs.append(torch.stack((geom_a, geom_b), dim=1))
        collision_pairs = torch.cat(collision_pairs, dim=0)  # N, 2

        if gs.backend != gs.metal:
            unique_pairs = torch.unique(collision_pairs, dim=0)
        else:
            # Apple Metal GPU backend does not support `torch.unique([...], dim=[...])`
            unique_pairs = torch.unique(collision_pairs.cpu(), dim=0).to(device=gs.device)
        return unique_pairs

    @ti.kernel
    def interpolate_path(
        self,
        path: ti.types.ndarray(),  # [N, B, Dof]
        sample_ind: ti.types.ndarray(),  # [B, 2]
        mask: ti.types.ndarray(),  # [B]
        tensor: ti.types.ndarray(),  # [N, B, Dof]
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(path.shape[1]):
            for i_q in range(self._entity.n_qs):
                for i_s in range(path.shape[0]):
                    tensor[i_s, i_b, i_q] = path[i_s, i_b, i_q]
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(path.shape[1]):
            if mask[i_b]:
                num_samples = sample_ind[i_b, 1] - sample_ind[i_b, 0]
                for i_q in range(self._entity.n_qs):
                    start = path[sample_ind[i_b, 0], i_b, i_q]
                    end = path[sample_ind[i_b, 1], i_b, i_q]
                    step = (end - start) / num_samples
                    for i_s in range(num_samples):
                        tensor[sample_ind[i_b, 0] + i_s, i_b, i_q] = start + step * i_s

    def check_collision(
        self,
        path,
        ignore_geom_pairs,
        envs_idx,
        *,
        is_plan_with_obj=False,
        obj_geom_start=-1,
        obj_geom_end=-1,
        ee_link_idx=0,
        obj_link_idx=0,
        _pos=None,
        _quat=None,
    ):
        out = torch.zeros((path.shape[1],), dtype=gs.tc_bool, device=gs.device)
        for qpos in path:
            if self._solver.n_envs > 0:
                self._entity.set_qpos(qpos, envs_idx=envs_idx, zero_velocity=False)
            else:
                self._entity.set_qpos(qpos[0], zero_velocity=False)

            if is_plan_with_obj:
                self.update_object(ee_link_idx, obj_link_idx, _pos, _quat, envs_idx)
            self._solver._kernel_detect_collision()
            self._kernel_check_collision(
                ignore_geom_pairs,
                envs_idx,
                is_plan_with_obj=is_plan_with_obj,
                obj_geom_start=obj_geom_start,
                obj_geom_end=obj_geom_end,
                out=out,
                collider_state=self._solver.collider._collider_state,
            )
        return out

    @ti.kernel
    def _kernel_check_collision(
        self,
        ignore_geom_pairs: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        is_plan_with_obj: ti.i32,
        obj_geom_start: ti.i32,
        obj_geom_end: ti.i32,
        out: ti.types.ndarray(),
        collider_state: array_class.ColliderState,
    ):
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]

            collision_detected = self._func_check_collision(
                collider_state,
                ignore_geom_pairs,
                i_b,
                is_plan_with_obj=is_plan_with_obj,
                obj_geom_start=obj_geom_start,
                obj_geom_end=obj_geom_end,
            )
            out[i_b] = out[i_b] or ti.cast(collision_detected, gs.ti_bool)

    @ti.func
    def _func_check_collision(
        self,
        collider_state: array_class.ColliderState,
        ignore_geom_pairs: ti.types.ndarray(),
        i_b: ti.i32,
        is_plan_with_obj: ti.i32 = False,
        obj_geom_start: ti.i32 = -1,
        obj_geom_end: ti.i32 = -1,
    ) -> ti.i32:
        is_collision_detected = ti.cast(False, gs.ti_int)
        for i_c in range(collider_state.n_contacts[i_b]):
            if not is_collision_detected:
                i_ga = collider_state.contact_data.geom_a[i_c, i_b]
                i_gb = collider_state.contact_data.geom_b[i_c, i_b]

                is_ignored = False
                if collider_state.contact_data.penetration[i_c, i_b] < self.PENETRATION_EPS:
                    is_ignored = True
                for i_p in range(ignore_geom_pairs.shape[0]):
                    if not is_ignored:
                        if (ignore_geom_pairs[i_p, 0] == i_ga and ignore_geom_pairs[i_p, 1] == i_gb) or (
                            ignore_geom_pairs[i_p, 0] == i_gb and ignore_geom_pairs[i_p, 1] == i_ga
                        ):
                            is_ignored = True
                if not is_ignored:
                    if (self._entity.geom_start <= i_ga < self._entity.geom_end) or (
                        self._entity.geom_start <= i_gb < self._entity.geom_end
                    ):
                        is_collision_detected = True
                    if is_plan_with_obj:
                        if (obj_geom_start <= i_ga < obj_geom_end) or (obj_geom_start <= i_gb < obj_geom_end):
                            is_collision_detected = True
        return is_collision_detected

    def shortcut_path(
        self,
        path_mask,
        path,
        iterations=50,
        ignore_geom_pairs=None,
        envs_idx=None,
        is_plan_with_obj=False,
        obj_geom_start=-1,
        obj_geom_end=-1,
        ee_link_idx=0,
        obj_link_idx=0,
        _pos=None,
        _quat=None,
    ):
        """
        path_mask: torch.Tensor
            valid waypoint mask [N,B] for the obtained path
        path: torch.Tensor
            the [N,B,Dof] tensor containing batched waypoints
        iterations: int
            the number of refine iterations
        """
        for i in range(iterations):
            ind = torch.multinomial(path_mask.T, 2).sort().values.to(gs.tc_int)  # B, 2
            ind_mask = (ind[:, 1] - ind[:, 0]) > 1
            result_path = torch.empty_like(path)
            self.interpolate_path(path.contiguous(), ind, ind_mask, result_path)
            collision_mask = self.check_collision(
                result_path,
                ignore_geom_pairs,
                envs_idx,
                is_plan_with_obj=is_plan_with_obj,
                obj_geom_start=obj_geom_start,
                obj_geom_end=obj_geom_end,
                ee_link_idx=ee_link_idx,
                obj_link_idx=obj_link_idx,
                _pos=_pos,
                _quat=_quat,
            )  # B
            path[:, ~collision_mask] = result_path[:, ~collision_mask]
        return path


@ti.data_oriented
class RRT(PathPlanner):
    def _init_rrt_fields(self, goal_bias=0.05, max_nodes=2000, pos_tol=5e-3, max_step_size=0.1):
        self._is_rrt_init = getattr(self, "_is_rrt_init", False)
        if not self._is_rrt_init:
            self._rrt_goal_bias = goal_bias
            self._rrt_max_nodes = max_nodes
            self._rrt_pos_tol = pos_tol
            self._rrt_max_step_size = max_step_size
            self._rrt_start_configuration = ti.field(
                dtype=gs.ti_float, shape=self._solver._batch_shape(self._entity.n_qs)
            )
            self._rrt_goal_configuration = ti.field(
                dtype=gs.ti_float, shape=self._solver._batch_shape(self._entity.n_qs)
            )
            self.struct_rrt_node_info = ti.types.struct(
                configuration=ti.types.vector(self._entity.n_qs, gs.ti_float),
                parent_idx=gs.ti_int,
            )
            self._rrt_node_info = self.struct_rrt_node_info.field(shape=self._solver._batch_shape(self._rrt_max_nodes))
            self._rrt_tree_size = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape())
            self._rrt_is_active = ti.field(dtype=gs.ti_bool, shape=self._solver._batch_shape())
            self._rrt_goal_reached_node_idx = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape())
            self._is_rrt_init = True

    def _reset_rrt_fields(self):
        self._rrt_start_configuration.fill(0.0)
        self._rrt_goal_configuration.fill(0.0)
        self._rrt_node_info.parent_idx.fill(-1)
        self._rrt_node_info.configuration.fill(0.0)
        self._rrt_tree_size.fill(0)
        self._rrt_is_active.fill(False)
        self._rrt_goal_reached_node_idx.fill(-1)

    @ti.kernel
    def _kernel_rrt_init(
        self, qpos_start: ti.types.ndarray(), qpos_goal: ti.types.ndarray(), envs_idx: ti.types.ndarray()
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for i_q in range(self._entity.n_qs):
                # save original qpos
                self._rrt_start_configuration[i_q, i_b] = qpos_start[i_b_, i_q]
                self._rrt_goal_configuration[i_q, i_b] = qpos_goal[i_b_, i_q]
                self._rrt_node_info[0, i_b].configuration[i_q] = qpos_start[i_b_, i_q]
            self._rrt_node_info[0, i_b].parent_idx = 0
            self._rrt_tree_size[i_b] = 1
            self._rrt_is_active[i_b] = True

    @ti.kernel
    def _kernel_rrt_step1(
        self,
        q_limit_lower: ti.types.ndarray(),
        q_limit_upper: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        joints_state: array_class.JointsState,
        joints_info: array_class.JointsInfo,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        dofs_state: array_class.DofsState,
        dofs_info: array_class.DofsInfo,
        entities_info: array_class.EntitiesInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        """
        Step 1 includes:
        - generate random sample
        - find nearest neighbor
        - steer from nearest neighbor to random sample
        - add new node
        - set the steer result (to prepare for collision checking)
        """
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]

            if self._rrt_is_active[i_b]:
                random_sample = ti.Vector(
                    [
                        q_limit_lower[i_q] + ti.random(dtype=gs.ti_float) * (q_limit_upper[i_q] - q_limit_lower[i_q])
                        for i_q in range(self._entity.n_qs)
                    ]
                )
                if ti.random() < self._rrt_goal_bias:
                    random_sample = ti.Vector(
                        [self._rrt_goal_configuration[i_q, i_b] for i_q in range(self._entity.n_qs)]
                    )

                # find nearest neighbor
                nearest_neighbor_idx = -1
                nearest_neighbor_dist = gs.ti_float(1e30)
                for i_n in range(self._rrt_tree_size[i_b]):
                    dist = (self._rrt_node_info.configuration[i_n, i_b] - random_sample).norm_sqr()
                    if dist < nearest_neighbor_dist:
                        nearest_neighbor_dist = dist
                        nearest_neighbor_idx = i_n

                # steer from nearest neighbor to random sample
                nearest_config = self._rrt_node_info.configuration[nearest_neighbor_idx, i_b]
                direction = random_sample - nearest_config
                steer_result = ti.Vector.zero(gs.ti_float, self._entity.n_qs)
                for i_q in range(self._entity.n_qs):
                    # If the step size exceeds max_step_size, clip it
                    if ti.abs(direction[i_q]) > self._rrt_max_step_size:
                        direction[i_q] = (-1.0 if direction[i_q] < 0.0 else 1.0) * self._rrt_max_step_size
                    steer_result[i_q] = nearest_config[i_q] + direction[i_q]

                if self._rrt_tree_size[i_b] < self._rrt_max_nodes - 1:
                    # add new node
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].configuration = steer_result
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].parent_idx = nearest_neighbor_idx
                    self._rrt_tree_size[i_b] += 1

                    # set the steer result and collision check for i_b
                    for i_q in range(self._entity.n_qs):
                        self._solver.qpos[i_q + self._entity._q_start, i_b] = steer_result[i_q]
                    gs.engine.solvers.rigid.rigid_solver_decomp.func_forward_kinematics_entity(
                        self._entity._idx_in_solver,
                        i_b,
                        links_state,
                        links_info,
                        joints_state,
                        joints_info,
                        dofs_state,
                        dofs_info,
                        entities_info,
                        rigid_global_info,
                        self._solver._static_rigid_sim_config,
                        False,
                    )
                    gs.engine.solvers.rigid.rigid_solver_decomp.func_update_geoms(
                        i_b,
                        entities_info,
                        geoms_info,
                        geoms_state,
                        links_state,
                        rigid_global_info,
                        self._solver._static_rigid_sim_config,
                        force_update_fixed_geoms=False,
                        is_backward=False,
                    )

    @ti.kernel
    def _kernel_rrt_step2(
        self,
        ignore_geom_pairs: ti.types.ndarray(),
        ignore_collision: ti.i32,
        envs_idx: ti.types.ndarray(),
        is_plan_with_obj: ti.i32,
        obj_geom_start: ti.i32,
        obj_geom_end: ti.i32,
        collider_state: array_class.ColliderState,
    ):
        """
        Step 2 includes:
        - check collision
        - if collision is detected, remove the new node
        - if collision is not detected, check if the new node is within goal configuration
        """
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]

            if self._rrt_is_active[i_b]:
                is_collision_detected = ti.cast(False, gs.ti_int)
                if not ignore_collision:
                    is_collision_detected = self._func_check_collision(
                        collider_state, ignore_geom_pairs, i_b, is_plan_with_obj, obj_geom_start, obj_geom_end
                    )
                if is_collision_detected:
                    self._rrt_tree_size[i_b] -= 1
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].configuration = 0.0
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].parent_idx = -1
                else:
                    # check the obtained steer result is within goal configuration only if no collision
                    is_goal = True
                    for i_q in range(self._entity.n_qs):
                        if (
                            ti.abs(
                                self._solver.qpos[i_q + self._entity._q_start, i_b]
                                - self._rrt_goal_configuration[i_q, i_b]
                            )
                            > self._rrt_pos_tol
                        ):
                            is_goal = False
                            break
                    if is_goal:
                        self._rrt_goal_reached_node_idx[i_b] = self._rrt_tree_size[i_b] - 1
                        self._rrt_is_active[i_b] = False

    def plan(
        self,
        qpos_goal,
        qpos_start=None,
        resolution=0.05,
        timeout=None,
        max_nodes=2000,
        smooth_path=True,
        num_waypoints=100,
        ignore_collision=False,
        ee_link_idx=None,
        obj_entity=None,
        envs_idx=None,
    ):
        assert self._solver.n_envs > 0 or envs_idx is None

        qpos_cur, qpos_goal, qpos_start, envs_idx = self._sanitize_qposs(qpos_goal, qpos_start, envs_idx)
        envs_idx_local = torch.arange(len(envs_idx), device=gs.device)
        ignore_geom_pairs = self.get_exclude_geom_pairs((qpos_goal, qpos_start), envs_idx)

        is_plan_with_obj = False
        _pos, _quat = None, None
        obj_geom_start, obj_geom_end = -1, -1
        if ee_link_idx is not None and obj_entity is not None:
            is_plan_with_obj = True
            obj_geom_start = obj_entity.geom_start
            obj_geom_end = obj_entity.geom_end
            obj_link_idx = obj_entity._links[0].idx
            _pos, _quat = self.get_link_pose(ee_link_idx, obj_link_idx, envs_idx)

        self._init_rrt_fields(max_nodes=max_nodes, max_step_size=resolution)
        self._reset_rrt_fields()
        self._kernel_rrt_init(qpos_start, qpos_goal, envs_idx)

        gs.logger.debug("Start RRT planning...")
        time_start = time.time()
        for i_n in range(self._rrt_max_nodes):
            if self._rrt_is_active.to_torch().any():
                self._kernel_rrt_step1(
                    q_limit_lower=self._entity.q_limit[0],
                    q_limit_upper=self._entity.q_limit[1],
                    envs_idx=envs_idx,
                    links_state=self._solver.links_state,
                    links_info=self._solver.links_info,
                    joints_state=self._solver.joints_state,
                    joints_info=self._solver.joints_info,
                    geoms_state=self._solver.geoms_state,
                    geoms_info=self._solver.geoms_info,
                    dofs_state=self._solver.dofs_state,
                    dofs_info=self._solver.dofs_info,
                    entities_info=self._solver.entities_info,
                    rigid_global_info=self._solver._rigid_global_info,
                )
                if is_plan_with_obj:
                    self.update_object(ee_link_idx, obj_link_idx, _pos, _quat, envs_idx)
                self._solver._kernel_detect_collision()
                self._kernel_rrt_step2(
                    collider_state=self._solver.collider._collider_state,
                    ignore_geom_pairs=ignore_geom_pairs,
                    ignore_collision=ignore_collision,
                    envs_idx=envs_idx,
                    is_plan_with_obj=is_plan_with_obj,
                    obj_geom_start=obj_geom_start,
                    obj_geom_end=obj_geom_end,
                )
            else:
                break
            if timeout is not None:
                if time.time() - time_start > timeout:
                    gs.logger.info(f"RRT planning timeout.")
                    break

        gs.logger.debug(f"RRT planning time: {time.time() - time_start}")

        is_invalid = self._rrt_is_active.to_torch(device=gs.device).bool()
        ts = self._rrt_tree_size.to_torch(device=gs.device)
        g_n = self._rrt_goal_reached_node_idx.to_torch(device=gs.device)  # B

        node_info = self._rrt_node_info.to_torch(device=gs.device)
        parents_idx = node_info["parent_idx"]
        configurations = node_info["configuration"]

        res = [g_n]
        for _ in range(ts.max()):
            g_n = parents_idx[g_n, envs_idx_local]
            res.append(g_n)
            if (g_n == 0).all():
                break
        res_idx = torch.stack(res[::-1], dim=0)
        sol = configurations[res_idx, envs_idx_local]  # N, B, DoF

        if is_invalid.all():
            if self._solver.n_envs > 0:
                self._entity.set_qpos(qpos_cur, envs_idx=envs_idx, zero_velocity=False)
            else:
                self._entity.set_qpos(qpos_cur, zero_velocity=False)
            sol = torch.zeros((num_waypoints, len(envs_idx), sol.shape[-1]), dtype=gs.tc_float, device=gs.device)
            return sol, is_invalid

        mask = rrt_valid_mask(res_idx)
        if self._solver.n_envs > 1:
            sol = align_waypoints_length(sol, mask, mask.sum(dim=0).max())
        if smooth_path:
            if is_plan_with_obj:
                sol = self.shortcut_path(
                    torch.ones_like(sol[..., 0]),
                    sol,
                    iterations=10,
                    ignore_geom_pairs=ignore_geom_pairs,
                    envs_idx=envs_idx,
                    is_plan_with_obj=is_plan_with_obj,
                    obj_geom_start=obj_geom_start,
                    obj_geom_end=obj_geom_end,
                    ee_link_idx=ee_link_idx,
                    obj_link_idx=obj_link_idx,
                    _pos=_pos,
                    _quat=_quat,
                )
            else:
                sol = self.shortcut_path(
                    torch.ones_like(sol[..., 0]),
                    sol,
                    iterations=10,
                    ignore_geom_pairs=ignore_geom_pairs,
                    envs_idx=envs_idx,
                )
        sol = align_waypoints_length(sol, torch.ones_like(sol[..., 0], dtype=torch.bool), num_waypoints)

        if not ignore_collision:
            if is_plan_with_obj:
                is_invalid |= self.check_collision(
                    sol,
                    ignore_geom_pairs,
                    envs_idx,
                    is_plan_with_obj=is_plan_with_obj,
                    obj_geom_start=obj_geom_start,
                    obj_geom_end=obj_geom_end,
                    ee_link_idx=ee_link_idx,
                    obj_link_idx=obj_link_idx,
                    _pos=_pos,
                    _quat=_quat,
                ).bool()
            else:
                is_invalid |= self.check_collision(sol, ignore_geom_pairs, envs_idx).bool()

        if self._solver.n_envs > 0:
            self._entity.set_qpos(qpos_cur, envs_idx=envs_idx, zero_velocity=False)
        else:
            self._entity.set_qpos(qpos_cur, zero_velocity=False)

        if is_plan_with_obj:
            self.update_object(ee_link_idx, obj_link_idx, _pos, _quat, envs_idx)

        if is_invalid.any():
            gs.logger.info(f"RRT planning failed in {int(is_invalid.sum())} environments")
        return sol, is_invalid


@ti.data_oriented
class RRTConnect(PathPlanner):
    def _init_rrt_connect_fields(self, goal_bias=0.1, max_nodes=4000, max_step_size=0.05):
        self._is_rrt_connect_init = getattr(self, "_is_rrt_connect_init", False)
        if not self._is_rrt_connect_init:
            self._rrt_goal_bias = goal_bias
            self._rrt_max_nodes = max_nodes
            self._rrt_max_step_size = max_step_size
            self._rrt_start_configuration = ti.field(
                dtype=gs.ti_float, shape=self._solver._batch_shape(self._entity.n_qs)
            )
            self._rrt_goal_configuration = ti.field(
                dtype=gs.ti_float, shape=self._solver._batch_shape(self._entity.n_qs)
            )
            self.struct_rrt_node_info = ti.types.struct(
                configuration=ti.types.vector(self._entity.n_qs, gs.ti_float),
                parent_idx=gs.ti_int,
                child_idx=gs.ti_int,
            )
            self._rrt_node_info = self.struct_rrt_node_info.field(shape=self._solver._batch_shape(self._rrt_max_nodes))
            self._rrt_tree_size = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape())
            self._rrt_is_active = ti.field(dtype=gs.ti_bool, shape=self._solver._batch_shape())
            self._rrt_goal_reached_node_idx = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape())
            self._is_rrt_connect_init = True

    def _reset_rrt_connect_fields(self):
        self._rrt_start_configuration.fill(0.0)
        self._rrt_goal_configuration.fill(0.0)
        self._rrt_node_info.parent_idx.fill(-1)
        self._rrt_node_info.child_idx.fill(-1)
        self._rrt_node_info.configuration.fill(0.0)
        self._rrt_tree_size.fill(0)
        self._rrt_is_active.fill(False)
        self._rrt_goal_reached_node_idx.fill(-1)

    @ti.kernel
    def _kernel_rrt_connect_init(
        self, qpos_start: ti.types.ndarray(), qpos_goal: ti.types.ndarray(), envs_idx: ti.types.ndarray()
    ):
        # NOTE: run IK before this
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for i_q in range(self._entity.n_qs):
                # save original qpos
                self._rrt_start_configuration[i_q, i_b] = qpos_start[i_b_, i_q]
                self._rrt_goal_configuration[i_q, i_b] = qpos_goal[i_b_, i_q]
                self._rrt_node_info[0, i_b].configuration[i_q] = qpos_start[i_b_, i_q]
                self._rrt_node_info[1, i_b].configuration[i_q] = qpos_goal[i_b_, i_q]
            self._rrt_node_info[0, i_b].parent_idx = 0
            self._rrt_node_info[1, i_b].child_idx = 1
            self._rrt_tree_size[i_b] = 2
            self._rrt_is_active[i_b] = True

    @ti.kernel
    def _kernel_rrt_connect_step1(
        self,
        qpos: array_class.V_ANNOTATION,
        forward_pass: ti.i32,
        q_limit_lower: ti.types.ndarray(),
        q_limit_upper: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
        links_state: array_class.LinksState,
        links_info: array_class.LinksInfo,
        joints_state: array_class.JointsState,
        joints_info: array_class.JointsInfo,
        geoms_state: array_class.GeomsState,
        geoms_info: array_class.GeomsInfo,
        dofs_state: array_class.DofsState,
        dofs_info: array_class.DofsInfo,
        entities_info: array_class.EntitiesInfo,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        """
        Step 1 includes:
        - generate random sample
        - find nearest neighbor
        - steer from nearest neighbor to random sample
        - add new node
        - set the steer result (to prepare for collision checking)
        """
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]

            if self._rrt_is_active[i_b]:
                random_sample = ti.Vector(
                    [
                        q_limit_lower[i_q] + ti.random(dtype=gs.ti_float) * (q_limit_upper[i_q] - q_limit_lower[i_q])
                        for i_q in range(self._entity.n_qs)
                    ]
                )
                if ti.random() < self._rrt_goal_bias:
                    if forward_pass:
                        random_sample = ti.Vector(
                            [self._rrt_goal_configuration[i_q, i_b] for i_q in range(self._entity.n_qs)]
                        )
                    else:
                        random_sample = ti.Vector(
                            [self._rrt_start_configuration[i_q, i_b] for i_q in range(self._entity.n_qs)]
                        )

                # find nearest neighbor
                nearest_neighbor_idx = -1
                nearest_neighbor_dist = gs.ti_float(1e30)
                for i_n in range(self._rrt_tree_size[i_b]):
                    if forward_pass:
                        # NOTE: in forward pass, we only consider the previous forward pass nodes (which has parent_idx != -1)
                        if self._rrt_node_info[i_n, i_b].parent_idx == -1:
                            continue
                    else:
                        # NOTE: in backward pass, we only consider the previous backward pass nodes (which has child_idx != -1)
                        if self._rrt_node_info[i_n, i_b].child_idx == -1:
                            continue
                    dist = (self._rrt_node_info.configuration[i_n, i_b] - random_sample).norm_sqr()
                    if dist < nearest_neighbor_dist:
                        nearest_neighbor_dist = dist
                        nearest_neighbor_idx = i_n

                # steer from nearest neighbor to random sample
                nearest_config = self._rrt_node_info.configuration[nearest_neighbor_idx, i_b]
                direction = random_sample - nearest_config
                steer_result = ti.Vector.zero(gs.ti_float, self._entity.n_qs)
                for i_q in range(self._entity.n_qs):
                    # If the step size exceeds max_step_size, clip it
                    if ti.abs(direction[i_q]) > self._rrt_max_step_size:
                        direction[i_q] = (-1.0 if direction[i_q] < 0.0 else 1.0) * self._rrt_max_step_size
                    steer_result[i_q] = nearest_config[i_q] + direction[i_q]

                if self._rrt_tree_size[i_b] < self._rrt_max_nodes - 1:
                    # add new node
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].configuration = steer_result
                    if forward_pass:
                        self._rrt_node_info[self._rrt_tree_size[i_b], i_b].parent_idx = nearest_neighbor_idx
                    else:
                        self._rrt_node_info[self._rrt_tree_size[i_b], i_b].child_idx = nearest_neighbor_idx
                    self._rrt_tree_size[i_b] += 1

                    # set the steer result and collision check for i_b
                    for i_q in range(self._entity.n_qs):
                        qpos[i_q + self._entity._q_start, i_b] = steer_result[i_q]
                    gs.engine.solvers.rigid.rigid_solver_decomp.func_forward_kinematics_entity(
                        self._entity._idx_in_solver,
                        i_b,
                        links_state,
                        links_info,
                        joints_state,
                        joints_info,
                        dofs_state,
                        dofs_info,
                        entities_info,
                        rigid_global_info,
                        self._solver._static_rigid_sim_config,
                        False,
                    )
                    gs.engine.solvers.rigid.rigid_solver_decomp.func_update_geoms(
                        i_b,
                        entities_info,
                        geoms_info,
                        geoms_state,
                        links_state,
                        rigid_global_info,
                        self._solver._static_rigid_sim_config,
                        force_update_fixed_geoms=False,
                        is_backward=False,
                    )

    @ti.kernel
    def _kernel_rrt_connect_step2(
        self,
        forward_pass: ti.i32,
        ignore_geom_pairs: ti.types.ndarray(),
        ignore_collision: ti.i32,
        envs_idx: ti.types.ndarray(),
        is_plan_with_obj: ti.i32,
        obj_geom_start: ti.i32,
        obj_geom_end: ti.i32,
        collider_state: array_class.ColliderState,
        rigid_global_info: array_class.RigidGlobalInfo,
    ):
        """
        Step 2 includes:
        - check collision
        - if collision is detected, remove the new node
        - if collision is not detected, check if the new node is within goal configuration
        """
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]

            if self._rrt_is_active[i_b]:
                is_collision_detected = ti.cast(False, gs.ti_int)
                if not ignore_collision:
                    is_collision_detected = self._func_check_collision(
                        collider_state, ignore_geom_pairs, i_b, is_plan_with_obj, obj_geom_start, obj_geom_end
                    )
                if is_collision_detected:
                    self._rrt_tree_size[i_b] -= 1
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].configuration = 0.0
                    if forward_pass:
                        self._rrt_node_info[self._rrt_tree_size[i_b], i_b].parent_idx = -1
                    else:
                        self._rrt_node_info[self._rrt_tree_size[i_b], i_b].child_idx = -1
                else:
                    # check the obtained steer result is within goal configuration only if no collision
                    for i_n in range(self._rrt_tree_size[i_b]):
                        if forward_pass:
                            if self._rrt_node_info[i_n, i_b].child_idx == -1:
                                continue
                        else:
                            if self._rrt_node_info[i_n, i_b].parent_idx == -1:
                                continue
                        is_connected = True
                        for i_q in range(self._entity.n_qs):
                            if (
                                ti.abs(
                                    rigid_global_info.qpos[i_q + self._entity._q_start, i_b]
                                    - self._rrt_node_info.configuration[i_n, i_b][i_q]
                                )
                                > self._rrt_max_step_size
                            ):
                                is_connected = False
                                break
                        if is_connected:
                            self._rrt_goal_reached_node_idx[i_b] = self._rrt_tree_size[i_b] - 1
                            if forward_pass:
                                self._rrt_node_info[self._rrt_tree_size[i_b] - 1, i_b].child_idx = i_n
                            else:
                                self._rrt_node_info[self._rrt_tree_size[i_b] - 1, i_b].parent_idx = i_n
                            self._rrt_is_active[i_b] = False
                            break

    def plan(
        self,
        qpos_goal,
        qpos_start=None,
        resolution=0.05,
        timeout=None,
        max_nodes=4000,
        smooth_path=True,
        num_waypoints=300,
        ignore_collision=False,
        ee_link_idx=None,
        obj_entity=None,
        envs_idx=None,
    ):
        assert self._solver.n_envs > 0 or envs_idx is None

        qpos_cur, qpos_goal, qpos_start, envs_idx = self._sanitize_qposs(qpos_goal, qpos_start, envs_idx)
        envs_idx_local = torch.arange(len(envs_idx), device=gs.device)
        ignore_geom_pairs = self.get_exclude_geom_pairs([qpos_goal, qpos_start], envs_idx)

        is_plan_with_obj = False
        _pos, _quat = None, None
        obj_geom_start, obj_geom_end = -1, -1
        if ee_link_idx is not None and obj_entity is not None:
            is_plan_with_obj = True
            obj_geom_start = obj_entity.geom_start
            obj_geom_end = obj_entity.geom_end
            obj_link_idx = obj_entity._links[0].idx
            _pos, _quat = self.get_link_pose(ee_link_idx, obj_link_idx, envs_idx)

        self._init_rrt_connect_fields(max_nodes=max_nodes, max_step_size=resolution)
        self._reset_rrt_connect_fields()
        self._kernel_rrt_connect_init(qpos_start, qpos_goal, envs_idx)

        gs.logger.debug("Start RRTConnect planning...")
        time_start = time.time()
        forward_pass = True
        for _ in range(self._rrt_max_nodes):
            self._kernel_rrt_connect_step1(
                qpos=self._solver.qpos,
                forward_pass=forward_pass,
                q_limit_lower=self._entity.q_limit[0],
                q_limit_upper=self._entity.q_limit[1],
                envs_idx=envs_idx,
                links_state=self._solver.links_state,
                links_info=self._solver.links_info,
                joints_state=self._solver.joints_state,
                joints_info=self._solver.joints_info,
                geoms_state=self._solver.geoms_state,
                geoms_info=self._solver.geoms_info,
                dofs_state=self._solver.dofs_state,
                dofs_info=self._solver.dofs_info,
                entities_info=self._solver.entities_info,
                rigid_global_info=self._solver._rigid_global_info,
            )
            if is_plan_with_obj:
                self.update_object(ee_link_idx, obj_link_idx, _pos, _quat, envs_idx)
            self._solver._kernel_detect_collision()
            self._kernel_rrt_connect_step2(
                forward_pass=forward_pass,
                ignore_geom_pairs=ignore_geom_pairs,
                ignore_collision=ignore_collision,
                envs_idx=envs_idx,
                is_plan_with_obj=is_plan_with_obj,
                obj_geom_start=obj_geom_start,
                obj_geom_end=obj_geom_end,
                collider_state=self._solver.collider._collider_state,
                rigid_global_info=self._solver._rigid_global_info,
            )
            forward_pass = not forward_pass

            if not self._rrt_is_active.to_torch().any():
                break
            if timeout is not None:
                if time.time() - time_start > timeout:
                    gs.logger.info(f"RRTConnect planning timeout.")
                    break
        else:
            gs.logger.info(f"RRTConnect planning exceeded maximum number of nodes ({self._rrt_max_nodes}).")

        gs.logger.debug(f"RRTConnect planning time: {time.time() - time_start}")
        is_invalid = self._rrt_is_active.to_torch(device=gs.device).bool()
        ts = self._rrt_tree_size.to_torch(device=gs.device)
        g_n = self._rrt_goal_reached_node_idx.to_torch(device=gs.device)  # B

        node_info = self._rrt_node_info.to_torch(device=gs.device)
        parents_idx = node_info["parent_idx"]
        children_idx = node_info["child_idx"]
        configurations = node_info["configuration"]

        res = [g_n]
        for _ in range(ts.max() // 2):
            g_n = parents_idx[g_n, envs_idx_local]
            res.append(g_n)
            if torch.all(g_n == 0):
                break
        res_idx = torch.stack(res[::-1], dim=0)

        c_n = self._rrt_goal_reached_node_idx.to_torch(device=gs.device)  # B
        res = []
        for _ in range(ts.max() // 2):
            c_n = children_idx[c_n, envs_idx_local]
            res.append(c_n)
            if torch.all(c_n == 1):
                break
        res_idx = torch.cat([res_idx, torch.stack(res, dim=0)], dim=0)
        sol = configurations[res_idx, envs_idx_local]  # N, B, DoF

        if is_invalid.all():
            if self._solver.n_envs > 0:
                self._entity.set_qpos(qpos_cur, envs_idx=envs_idx, zero_velocity=False)
            else:
                self._entity.set_qpos(qpos_cur, zero_velocity=False)
            return torch.zeros(num_waypoints, len(envs_idx), sol.shape[-1], device=gs.device), is_invalid

        mask = rrt_connect_valid_mask(res_idx)
        if self._solver.n_envs > 1:
            sol = align_waypoints_length(sol, mask, mask.sum(dim=0).max())
        if smooth_path:
            if is_plan_with_obj:
                sol = self.shortcut_path(
                    torch.ones_like(sol[..., 0]),
                    sol,
                    iterations=10,
                    ignore_geom_pairs=ignore_geom_pairs,
                    envs_idx=envs_idx,
                    is_plan_with_obj=is_plan_with_obj,
                    obj_geom_start=obj_geom_start,
                    obj_geom_end=obj_geom_end,
                    ee_link_idx=ee_link_idx,
                    obj_link_idx=obj_link_idx,
                    _pos=_pos,
                    _quat=_quat,
                )
            else:
                sol = self.shortcut_path(
                    torch.ones_like(sol[..., 0]),
                    sol,
                    iterations=10,
                    ignore_geom_pairs=ignore_geom_pairs,
                    envs_idx=envs_idx,
                )
        sol = align_waypoints_length(sol, torch.ones_like(sol[..., 0], dtype=torch.bool), num_waypoints)

        if not ignore_collision:
            if is_plan_with_obj:
                is_invalid |= self.check_collision(
                    sol,
                    ignore_geom_pairs,
                    envs_idx,
                    is_plan_with_obj=is_plan_with_obj,
                    obj_geom_start=obj_geom_start,
                    obj_geom_end=obj_geom_end,
                    ee_link_idx=ee_link_idx,
                    obj_link_idx=obj_link_idx,
                    _pos=_pos,
                    _quat=_quat,
                ).bool()
            else:
                is_invalid |= self.check_collision(sol, ignore_geom_pairs, envs_idx).bool()

        if self._solver.n_envs > 0:
            self._entity.set_qpos(qpos_cur, envs_idx=envs_idx, zero_velocity=False)
        else:
            self._entity.set_qpos(qpos_cur, zero_velocity=False)

        if is_plan_with_obj:
            self.update_object(ee_link_idx, obj_link_idx, _pos, _quat, envs_idx)

        if is_invalid.any():
            gs.logger.info(f"RRTConnect planning failed in {int(is_invalid.sum())} environments")

        return sol, is_invalid


# ------------------------------------------------------------------------------------
# ------------------------------------ utils -----------------------------------------
# ------------------------------------------------------------------------------------


def align_waypoints_length(path: torch.Tensor, mask: torch.Tensor, num_points: int) -> torch.Tensor:
    """
    Aligns each waypoints length to the given num_points.

    Parameters
    ----------
    path: torch.Tensor
        path tensor in [N, B, Dof]
    mask: torch.Tensor
        the masking of path, indicating active waypoints [N, B]
    num_points: int
        the number of the desired waypoints

    Returns
    -------
        A new 2D PyTorch tensor [num_points, B, Dof]
    """
    t_path = path.permute(1, 2, 0)  # [B, Dof, N]
    res = torch.zeros(
        (num_points, t_path.shape[0], t_path.shape[1]), dtype=gs.tc_float, device=gs.device
    )  # [num_points, B, Dof]
    for i_b in range(t_path.shape[0]):
        if not mask[:, i_b].any():
            continue
        interpolated_path = torch.nn.functional.interpolate(
            t_path[i_b : i_b + 1, :, mask[:, i_b]], size=num_points, mode="linear", align_corners=True
        ).squeeze(0)
        res[:, i_b] = interpolated_path.T
    return res


def rrt_valid_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns valid mask of the RRTConnect result node indicies

    Parameters
    ----------
    tensor: torch.Tensor
        path tensor in [N, B]
    """
    mask = (tensor > 0.0).to(gs.tc_float)  # N, B
    mask_float = mask.T.unsqueeze(1)  # B 1, N
    kernel = torch.ones((1, 1, 3), device=tensor.device, dtype=gs.tc_float)
    dilated_mask_float = F.conv1d(mask_float, kernel.to(mask_float.dtype), padding="same")
    dilated_mask = (dilated_mask_float > 0.0).squeeze(1).T
    return dilated_mask


def rrt_connect_valid_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns valid mask of the RRTConnect result node indicies

    Parameters
    ----------
    tensor: torch.Tensor
        path tensor in [N, B]
    """
    mask = (tensor > 0.0).to(gs.tc_float)  # N, B
    mask_float = mask.T.unsqueeze(1)  # B 1, N
    kernel = torch.ones(1, 1, 3, device=tensor.device, dtype=gs.tc_float)
    dilated_mask_float = F.conv1d(mask_float, kernel.to(mask_float.dtype), padding="same")
    dilated_mask = (dilated_mask_float > 0).squeeze(1).T
    return dilated_mask
