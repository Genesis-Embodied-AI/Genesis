from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
import taichi as ti
import torch
import torch.nn.functional as F
import math
import time
import genesis as gs
from genesis.utils.misc import tensor_to_array
try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
    IS_OMPL_AVAILABLE = True

    __all__ = [
        "RRTConnect",
        "RRT",
        "RRTConnect_OMPL",
        "RRT_OMPL",
        "PRM_OMPL",
        "EST_OMPL",
        "FMT_OMPL",
        "BITstar_OMPL",
        "ABITstar_OMPL"
    ]

except ImportError:
    IS_OMPL_AVAILABLE = False
    
    __all__ = [
        "RRTConnect",
        "RRT",
    ]



@ti.data_oriented
class PathPlanner(ABC):
    """
    Base class for path planners
    """
    def __init__(self, entity):
        self._entity = entity
        self._solver = entity._solver

    @property
    def default_q_limit(self):
        return self._entity.q_limit

    @property
    def n_geoms(self):
        """Number of geometries."""
        return self._entity.n_geoms

    @property
    def geom_start(self):
        """Starting geometry."""
        return self._entity.geom_start

    @property
    def geom_end(self):
        """Ending geometry."""
        return self._entity.geom_end

    @property
    def n_dofs(self):
        """Number of degrees of freedom."""
        return self._entity.n_dofs

    @property
    def n_qs(self):
        """Number of generalized coordinates."""
        return self._entity.n_qs

    @property
    def _q_start(self):
        """Starting generalized coordinates."""
        return self._entity._q_start
    
    def validate_input_qpos_batch(self, qpos_goal, qpos_start, envs_idx, *, check_joint_limits=True):
        if qpos_start is None:
            qpos_start = self._entity.get_qpos()
        else:
            if qpos_start.ndim == 1:
                qpos_start = qpos_start.unsqueeze(0)
            assert qpos_start.ndim == 2
            if qpos_start.shape[0] == 1:
                qpos_start = qpos_start.repeat(len(envs_idx), 1)

        if qpos_goal.ndim == 1:
            qpos_goal = qpos_goal.unsqueeze(0)
        assert qpos_goal.ndim == 2
        if qpos_goal.shape[0] == 1:
            qpos_goal = qpos_goal.repeat(len(envs_idx), 1)
        else:
            assert qpos_goal.shape[0] == len(envs_idx), f"Batch size mismatch: {qpos_goal.shape=} {len(envs_idx)=}"

        if qpos_start.shape[1] != self.n_qs or qpos_goal.shape[1] != self.n_qs:
            gs.raise_exception("Invalid shape for `qpos_start` or `qpos_goal`.")
        
        if check_joint_limits:
            # NOTE: process joint limit
            if (qpos_start.cpu().numpy() < self.default_q_limit[0]).any() or (qpos_start.cpu().numpy() > self.default_q_limit[1]).any():
                gs.logger.warning("`qpos_start` exceeds joint limit.")

            if (qpos_goal.cpu().numpy() < self.default_q_limit[0]).any() or (qpos_goal.cpu().numpy() > self.default_q_limit[1]).any():
                gs.logger.warning("`qpos_goal` exceeds joint limit.")
        return qpos_goal, qpos_start
        
    @abstractmethod
    def plan(
        self,
        qpos_goal,
        qpos_start=None,
    ):...

    # ------------------------------------------------------------------------------------
    # -------------------------------path planing utils-----------------------------------
    # ------------------------------------------------------------------------------------
    @ti.kernel
    def interpolate_path(
        self,
        tensor: ti.types.ndarray(),     # [B, N, Dof]
        path: ti.types.ndarray(),       # [B, N, Dof]
        sample_ind: ti.types.ndarray(), # [B, 2]
        mask: ti.types.ndarray(),       # [B]
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(path.shape[0]):
            if not mask[i_b]:
                continue
            num_samples = sample_ind[i_b, 1] - sample_ind[i_b, 0]
            for i_q in range(self.n_qs):
                start = path[i_b, sample_ind[i_b, 0], i_q]
                end = path[i_b, sample_ind[i_b, 1], i_q] 
                step = (end - start) / num_samples
                for i_s in range(num_samples):
                    tensor[i_b, sample_ind[i_b, 0] + i_s, i_q] = start + step * i_s

    def check_collision(self, path, ignore_geom_pairs=None, envs_idx=None):
        path = path.transpose(1,0) # N, B, Dof
        res = torch.zeros(path.shape[1], dtype=gs.tc_int, device=gs.device)
        for qpos in path:
            self._entity.set_qpos(qpos, envs_idx=envs_idx)
            self._solver._kernel_detect_collision()
            self._kernel_check_collision(res, ignore_geom_pairs, envs_idx)
        return res
    
    @ti.func
    def _func_check_collision(
        self,
        ignore_geom_pairs: ti.types.ndarray(),
        i_b: ti.int32,
    ):
        collision_detected = False
        for i_c in range(self._solver.collider.n_contacts[i_b]):
            i_ga = self._solver.collider.contact_data[i_c, i_b].geom_a
            i_gb = self._solver.collider.contact_data[i_c, i_b].geom_b

            ignore = False
            for i_p in range(ignore_geom_pairs.shape[0]):
                if (ignore_geom_pairs[i_p, 0] == i_ga and ignore_geom_pairs[i_p, 1] == i_gb) or (
                    ignore_geom_pairs[i_p, 0] == i_gb and ignore_geom_pairs[i_p, 1] == i_ga):
                    ignore = True
                    break
            if ignore:
                continue

            # TODO: handle self-collision (except the case for closed gripper)
            if (self.geom_start <= i_ga < self.geom_end) ^ (self.geom_start <= i_gb < self.geom_end):
                collision_detected = True
                break
        return collision_detected
    
    @ti.kernel
    def _kernel_check_collision(
        self,
        tensor: ti.types.ndarray(),
        ignore_geom_pairs: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in envs_idx:
            collision_detected = self._func_check_collision(ignore_geom_pairs, i_b)
            tensor[i_b] = tensor[i_b] or collision_detected
            
    def shortcut_path(self, path_mask, path, iterations=50, ignore_geom_pairs=None, envs_idx=None):
        """
        path_mask: torch.Tensor
            node mask [N,B] for the obtained path
        path: torch.Tensor
            the [N,B,Dof] tensor
        iterations: int
            the number of refine iterations
        """
        path_mask = path_mask.t()  # B, N
        path = path.transpose(1,0) # B, N, Dof
        for i in range(iterations):
            ind = torch.multinomial(path_mask, 2).sort()[0] # B, 2
            ind_mask = (ind[:,1] - ind[:,0]) > 1
            tmp_path = path.clone().contiguous()
            self.interpolate_path(tmp_path, path.contiguous(), ind.contiguous(), ind_mask)
            collision_mask = self.check_collision(tmp_path, ignore_geom_pairs, envs_idx) # B
            path[~collision_mask] = tmp_path[~collision_mask]
            if i == 10:
                break
        return path.transpose(1,0)


@ti.data_oriented
class RRT(PathPlanner):
    """
    Batched Rapidly exploring random tree (RRT) path planner

    Parameters
    ----------
    entity : RigidEntity
        the entity to use the path planning
    goal_bias: float
        the probability of sampling the goal
    max_nodes: int
        the number of maximum nodes for sampling
    pos_tol: float
        the error tolerance of the found path
    max_step_size: float
        the maximum step size of qpos in randians (or in meters)
    """

    def _init_rrt_fields(self, goal_bias=0.05, max_nodes=2000, pos_tol=5e-3, max_step_size=0.1):
        self._rrt_goal_bias = goal_bias
        self._rrt_max_nodes = max_nodes
        self._rrt_pos_tol = pos_tol # .28 degree
        self._rrt_max_step_size = max_step_size # NOTE: in radian (about 5.7 degree)
        self._rrt_start_configuration = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.n_qs))
        self._rrt_goal_configuration = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.n_qs))
        self.struct_rrt_node_info = ti.types.struct(
            configuration=ti.types.vector(self.n_qs, gs.ti_float),
            parent_idx=gs.ti_int,
        )
        self._rrt_node_info = self.struct_rrt_node_info.field(shape=self._solver._batch_shape(self._rrt_max_nodes))
        self._rrt_tree_size = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape())
        self._rrt_is_active = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape())
        self._rrt_goal_reached_node_idx = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape())
        
    def _reset_rrt_fields(self):
        self._rrt_start_configuration.fill(0.0)
        self._rrt_goal_configuration.fill(0.0)
        self._rrt_node_info.parent_idx.fill(-1)
        self._rrt_node_info.configuration.fill(0.0)
        self._rrt_tree_size.fill(0)
        self._rrt_is_active.fill(0)
        self._rrt_goal_reached_node_idx.fill(-1)
    
    @ti.kernel
    def _kernel_rrt_init(
        self,
        qpos_start: ti.types.ndarray(),
        qpos_goal: ti.types.ndarray(),
        envs_idx: ti.types.ndarray()
    ):
        # NOTE: run IK before this
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for i_q in ti.static(range(self.n_qs)):
                # save original qpos
                self._rrt_start_configuration[i_q, i_b] = qpos_start[i_b_, i_q]
                self._rrt_goal_configuration[i_q, i_b] = qpos_goal[i_b_, i_q]
                self._rrt_node_info[0, i_b].configuration[i_q] = qpos_start[i_b_, i_q]
            self._rrt_node_info[0, i_b].parent_idx = 0
            self._rrt_tree_size[i_b] = 1
            self._rrt_is_active[i_b] = 1

    @ti.kernel
    def _kernel_rrt_cleanup(
        self,
        qpos_current: ti.types.ndarray(),
        envs_idx: ti.types.ndarray()
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for i_q in ti.static(range(self.n_qs)):
                self._solver.qpos[i_q + self._q_start, i_b] = qpos_current[i_b_, i_q]
            self._solver._func_forward_kinematics_entity(self._entity._idx_in_solver, i_b)
    
    @ti.kernel
    def _kernel_rrt_step1(
        self,
        q_limit_lower: ti.types.ndarray(),
        q_limit_upper: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in envs_idx:
            if not self._rrt_is_active[i_b]:
                continue
            
            random_sample = ti.Vector([
                q_limit_lower[i_q] + ti.random(dtype=gs.ti_float) * (q_limit_upper[i_q] - q_limit_lower[i_q])
                for i_q in ti.static(range(self.n_qs))
            ])
            if ti.random() < self._rrt_goal_bias:
                random_sample = ti.Vector([
                    self._rrt_goal_configuration[i_q, i_b]
                    for i_q in ti.static(range(self.n_qs))
                ])

            # find nearest neighbor
            nearest_neighbor_idx = -1
            nearest_neighbor_dist = gs.ti_float(1e30) 
            ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
            for i_n in range(self._rrt_tree_size[i_b]):
                dist = (self._rrt_node_info.configuration[i_n, i_b] - random_sample).norm()
                if dist < nearest_neighbor_dist:
                    nearest_neighbor_dist = dist
                    nearest_neighbor_idx = i_n

            # steer from nearest neighbor to random sample
            nearest_config = self._rrt_node_info.configuration[nearest_neighbor_idx, i_b]
            direction = random_sample - nearest_config
            steer_result = ti.Vector.zero(gs.ti_float, self.n_qs)
            for i_q in ti.static(range(self.n_qs)):
                # If the step size exceeds max_step_size, clip it
                if abs(direction[i_q]) > self._rrt_max_step_size:
                    direction[i_q] = ti.math.sign(direction[i_q]) * self._rrt_max_step_size
                steer_result[i_q] = nearest_config[i_q] + direction[i_q]

            if self._rrt_tree_size[i_b] < self._rrt_max_nodes - 1:
                # add new node
                self._rrt_node_info[self._rrt_tree_size[i_b], i_b].configuration = steer_result
                self._rrt_node_info[self._rrt_tree_size[i_b], i_b].parent_idx = nearest_neighbor_idx
                self._rrt_tree_size[i_b] += 1

                # set the steer result and collision check for i_b
                for i_q in ti.static(range(self.n_qs)):
                    self._solver.qpos[i_q + self._q_start, i_b] = steer_result[i_q]
                self._solver._func_forward_kinematics_entity(self._entity._idx_in_solver, i_b)
                self._solver._func_update_geoms(i_b)
    
    @ti.func
    def _func_check_collision(
        self,
        ignore_geom_pairs: ti.types.ndarray(),
        i_b: ti.int32,
    ):
        collision_detected = False
        for i_c in range(self._solver.collider.n_contacts[i_b]):
            i_ga = self._solver.collider.contact_data[i_c, i_b].geom_a
            i_gb = self._solver.collider.contact_data[i_c, i_b].geom_b

            ignore = False
            for i_p in range(ignore_geom_pairs.shape[0]):
                if (ignore_geom_pairs[i_p, 0] == i_ga and ignore_geom_pairs[i_p, 1] == i_gb) or (
                    ignore_geom_pairs[i_p, 0] == i_gb and ignore_geom_pairs[i_p, 1] == i_ga):
                    ignore = True
                    break
            if ignore:
                continue

            # TODO: handle self-collision (except the case for closed gripper)
            if (self.geom_start <= i_ga < self.geom_end) ^ (self.geom_start <= i_gb < self.geom_end):
                collision_detected = True
                break
        return collision_detected

    @ti.kernel
    def _kernel_rrt_step2(
        self,
        ignore_geom_pairs: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in envs_idx:
            if not self._rrt_is_active[i_b]:
                continue

            collision_detected = self._func_check_collision(ignore_geom_pairs, i_b)
            if collision_detected:
                self._rrt_tree_size[i_b] -= 1
                self._rrt_node_info[self._rrt_tree_size[i_b], i_b].configuration = 0.0
                self._rrt_node_info[self._rrt_tree_size[i_b], i_b].parent_idx = -1
            else:
                # check the obtained steer result is within goal configuration only if no collision
                flag = True
                for i_q in range(self.n_qs):
                    if abs(self._solver.qpos[i_q + self._q_start, i_b] - self._rrt_goal_configuration[i_q, i_b]) > self._rrt_pos_tol:
                        flag = False
                        break
                if flag:
                    self._rrt_goal_reached_node_idx[i_b] = self._rrt_tree_size[i_b] - 1
                    self._rrt_is_active[i_b] = 0
        
    def plan(
        self,
        qpos_goal,
        qpos_start=None,
        *,
        resolution=0.01,
        smooth_path=True,
        num_waypoints=100,
        envs_idx=None,
    ):
        """
        Plan a path from `qpos_start` to `qpos_goal`.
        qpos_goal : array_like
            The goal state. [B, Nq] or [1, Nq]
        qpos_start : None | array_like, optional
            The start state. If None, the current state of the rigid entity will be used. Defaults to None. [B, Nq] or [1, Nq]
        """
        import time
        if self._solver.n_envs > 0:
            envs_idx = self._solver._sanitize_envs_idx(envs_idx)
        else:
            envs_idx = torch.zeros(1, dtype=gs.tc_int, device=gs.device)

        qpos_cur = self._entity.get_qpos()
        qpos_goal, qpos_start = self.validate_input_qpos_batch(qpos_goal, qpos_start, envs_idx)
        
        self._init_rrt_fields()
        self._reset_rrt_fields()
        self._kernel_rrt_init(qpos_start.contiguous(), qpos_goal.contiguous(), envs_idx)

        self._entity.set_qpos(qpos_start)
        self._solver._kernel_detect_collision()
        scene_contact_info = self._solver.collider.contact_data.to_torch(gs.device)
        n_contacts = self._solver.collider.n_contacts.to_torch(gs.device)

        valid_mask = torch.logical_or(
            torch.logical_and(
                scene_contact_info["geom_a"] >= self.geom_start,
                scene_contact_info["geom_a"] < self.geom_end,
            ),
            torch.logical_and(
                scene_contact_info["geom_b"] >= self.geom_start,
                scene_contact_info["geom_b"] < self.geom_end,
            ),
        )
        contact_indices = torch.arange(valid_mask.shape[0], device=valid_mask.device).unsqueeze(1)
        valid_mask = torch.logical_and(valid_mask, contact_indices < n_contacts)

        max_env_collisions = int(torch.max(n_contacts).item())
        valid_mask = valid_mask[:max_env_collisions]
        geom_a = scene_contact_info["geom_a"][:max_env_collisions][valid_mask] # N 
        geom_b = scene_contact_info["geom_b"][:max_env_collisions][valid_mask] # N
        assert len(geom_a) == len(geom_b)

        # NOTE: we will reduce the contacts in batch dim assuming internal geom collisions are the same for a robot
        stacked_tensors = torch.stack((geom_a, geom_b), dim=1)
        unique_pairs = torch.unique(stacked_tensors, dim=0) # N', 2

        gs.logger.info("start rrt planning...")
        start = time.time()
        for i_n in range(self._rrt_max_nodes):
            if self._rrt_is_active.to_torch().any():
                self._kernel_rrt_step1(
                    q_limit_lower=self._entity.q_limit[0],
                    q_limit_upper=self._entity.q_limit[1],
                    envs_idx=envs_idx,
                )
                self._solver._kernel_detect_collision()
                self._kernel_rrt_step2(
                    ignore_geom_pairs=unique_pairs,
                    envs_idx=envs_idx,
                )
            else:
                break
        
        if self._rrt_is_active.to_torch().any():
            gs.logger.warning(f"rrt planning failed in {self._rrt_is_active.to_torch().sum()} envs")
        
        gs.logger.info(f"rrt planning time: {time.time() - start}")
        start = time.time()
        ts = self._rrt_tree_size.to_torch(device=gs.device)[envs_idx]
        g_n = self._rrt_goal_reached_node_idx.to_torch(device=gs.device)[envs_idx] # B

        node_info = self._rrt_node_info.to_torch(device=gs.device)
        parents_idx = node_info["parent_idx"][:, envs_idx]
        configurations = node_info["configuration"][:, envs_idx]

        res = [g_n]
        for _ in range(ts.max()):
            g_n = parents_idx[g_n, torch.arange(len(envs_idx))]
            res.append(g_n)
            if torch.all(g_n == 0):
                break
        res_idx = torch.stack(list(reversed(res)), dim=0)
        sol = configurations[res_idx, torch.arange(len(envs_idx))] # N, B, DoF

        sol = align_weypoints_length(sol, rrt_valid_mask(res_idx), len(res_idx))
        sol = self.shortcut_path(
            torch.ones_like(sol[...,0]), sol, iterations=10, ignore_geom_pairs=unique_pairs, envs_idx=envs_idx
        )

        # interpolate to make num_waypoints
        sol = align_weypoints_length(sol, torch.ones_like(sol[...,0]).bool(), num_waypoints)
        gs.logger.info(f"path post-processing time: {time.time() - start}")

        self._kernel_rrt_cleanup(qpos_cur.contiguous(), envs_idx)
        return sol


@ti.data_oriented
class RRTConnect(PathPlanner):
    def _init_rrt_connect_fields(self, goal_bias=0.05, max_nodes=2000, pos_tol=5e-3, max_step_size=0.1):
        self._rrt_goal_bias = goal_bias
        self._rrt_max_nodes = max_nodes
        self._rrt_pos_tol = pos_tol # .28 degree
        self._rrt_max_step_size = max_step_size # NOTE: in radian (about 5.7 degree)
        self._rrt_start_configuration = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.n_qs))
        self._rrt_goal_configuration = ti.field(dtype=gs.ti_float, shape=self._solver._batch_shape(self.n_qs))
        self.struct_rrt_node_info = ti.types.struct(
            configuration=ti.types.vector(self.n_qs, gs.ti_float),
            parent_idx=gs.ti_int,
            child_idx=gs.ti_int,
        )
        self._rrt_node_info = self.struct_rrt_node_info.field(shape=self._solver._batch_shape(self._rrt_max_nodes))
        self._rrt_tree_size = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape())
        self._rrt_is_active = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape())
        self._rrt_goal_reached_node_idx = ti.field(dtype=gs.ti_int, shape=self._solver._batch_shape())
        
    def _reset_rrt_connect_fields(self):
        self._rrt_start_configuration.fill(0.0)
        self._rrt_goal_configuration.fill(0.0)
        self._rrt_node_info.parent_idx.fill(-1)
        self._rrt_node_info.child_idx.fill(-1)
        self._rrt_node_info.configuration.fill(0.0)
        self._rrt_tree_size.fill(0)
        self._rrt_is_active.fill(0)
        self._rrt_goal_reached_node_idx.fill(-1)
    
    @ti.kernel
    def _kernel_rrt_connect_init(
        self,
        qpos_start: ti.types.ndarray(),
        qpos_goal: ti.types.ndarray(),
        envs_idx: ti.types.ndarray()
    ):
        # NOTE: run IK before this
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for i_q in ti.static(range(self.n_qs)):
                # save original qpos
                self._rrt_start_configuration[i_q, i_b] = qpos_start[i_b_, i_q]
                self._rrt_goal_configuration[i_q, i_b] = qpos_goal[i_b_, i_q]
                self._rrt_node_info[0, i_b].configuration[i_q] = qpos_start[i_b_, i_q]
                self._rrt_node_info[1, i_b].configuration[i_q] = qpos_goal[i_b_, i_q]
            self._rrt_node_info[0, i_b].parent_idx = 0
            self._rrt_node_info[1, i_b].child_idx = 1
            self._rrt_tree_size[i_b] = 2
            self._rrt_is_active[i_b] = 1

    @ti.kernel
    def _kernel_rrt_connect_cleanup(
        self,
        qpos_current: ti.types.ndarray(),
        envs_idx: ti.types.ndarray()
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b_ in range(envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for i_q in ti.static(range(self.n_qs)):
                self._solver.qpos[i_q + self._q_start, i_b] = qpos_current[i_b_, i_q]
            self._solver._func_forward_kinematics_entity(self._entity._idx_in_solver, i_b)
    
    @ti.kernel
    def _kernel_rrt_connect_step1(
        self,
        forward_pass: ti.i32,
        q_limit_lower: ti.types.ndarray(),
        q_limit_upper: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in envs_idx:
            if not self._rrt_is_active[i_b]:
                continue
            
            random_sample = ti.Vector([
                q_limit_lower[i_q] + ti.random(dtype=gs.ti_float) * (q_limit_upper[i_q] - q_limit_lower[i_q])
                for i_q in ti.static(range(self.n_qs))
            ])
            if ti.random() < self._rrt_goal_bias:
                if forward_pass:
                    random_sample = ti.Vector([
                        self._rrt_goal_configuration[i_q, i_b]
                        for i_q in ti.static(range(self.n_qs))
                    ])
                else:
                    random_sample = ti.Vector([
                        self._rrt_start_configuration[i_q, i_b]
                        for i_q in ti.static(range(self.n_qs))
                    ])

            # find nearest neighbor
            nearest_neighbor_idx = -1
            nearest_neighbor_dist = gs.ti_float(1e30) 
            ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
            for i_n in range(self._rrt_tree_size[i_b]):
                if forward_pass:
                    # NOTE: in forward pass, we only consider the previous forward pass nodes (which has parent_idx != -1)
                    if self._rrt_node_info[i_n, i_b].parent_idx == -1:
                        continue
                else:
                    # NOTE: in backward pass, we only consider the previous backward pass nodes (which has child_idx != -1)
                    if self._rrt_node_info[i_n, i_b].child_idx == -1:
                        continue
                dist = (self._rrt_node_info.configuration[i_n, i_b] - random_sample).norm()
                if dist < nearest_neighbor_dist:
                    nearest_neighbor_dist = dist
                    nearest_neighbor_idx = i_n
            
            # steer from nearest neighbor to random sample
            nearest_config = self._rrt_node_info.configuration[nearest_neighbor_idx, i_b]
            direction = random_sample - nearest_config
            steer_result = ti.Vector.zero(gs.ti_float, self.n_qs)
            for i_q in ti.static(range(self.n_qs)):
                # If the step size exceeds max_step_size, clip it
                if abs(direction[i_q]) > self._rrt_max_step_size:
                    direction[i_q] = ti.math.sign(direction[i_q]) * self._rrt_max_step_size
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
                for i_q in ti.static(range(self.n_qs)):
                    self._solver.qpos[i_q + self._q_start, i_b] = steer_result[i_q]
                self._solver._func_forward_kinematics_entity(self._entity._idx_in_solver, i_b)
                self._solver._func_update_geoms(i_b)

    @ti.kernel
    def _kernel_rrt_connect_step2(
        self,
        forward_pass: ti.i32,
        ignore_geom_pairs: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in envs_idx:
            if not self._rrt_is_active[i_b]:
                continue

            collision_detected = self._func_check_collision(ignore_geom_pairs, i_b)
            if collision_detected:
                self._rrt_tree_size[i_b] -= 1
                self._rrt_node_info[self._rrt_tree_size[i_b], i_b].configuration = 0.0
                if forward_pass:
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].parent_idx = -1
                else:
                    self._rrt_node_info[self._rrt_tree_size[i_b], i_b].child_idx = -1
            else:
                # check the obtained steer result is within goal configuration only if no collision
                ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
                for i_n in range(self._rrt_tree_size[i_b]):
                    if forward_pass:
                        # NOTE: in forward pass, we only consider the previous backward pass nodes (which has child_idx != -1)
                        if self._rrt_node_info[i_n, i_b].child_idx == -1:
                            continue
                    else:
                        # NOTE: in backward pass, we only consider the previous forward pass nodes (which has parent_idx != -1)
                        if self._rrt_node_info[i_n, i_b].parent_idx == -1:
                            continue
                    flag = True
                    for i_q in range(self.n_qs):
                        if abs(self._solver.qpos[i_q + self._q_start, i_b] - self._rrt_node_info.configuration[i_n, i_b][i_q]) > self._rrt_max_step_size:
                            flag = False
                            break
                    if flag:
                        self._rrt_goal_reached_node_idx[i_b] = self._rrt_tree_size[i_b] - 1
                        if forward_pass:
                            self._rrt_node_info[self._rrt_goal_reached_node_idx[i_b], i_b].child_idx = i_n
                        else:
                            self._rrt_node_info[self._rrt_goal_reached_node_idx[i_b], i_b].parent_idx = i_n
                        self._rrt_is_active[i_b] = 0
                        break
        
    def plan(
        self,
        qpos_goal,
        qpos_start=None,
        resolution=0.01,
        timeout=5.0,
        max_retry=1,
        smooth_path=True,
        num_waypoints=100,
        ignore_collision=False,
        envs_idx=None
    ):
        """
        Plan a path from `qpos_start` to `qpos_goal`.
        qpos_goal : array_like
            The goal state. [B, Nq] or [1, Nq]
        qpos_start : None | array_like, optional
            The start state. If None, the current state of the rigid entity will be used. Defaults to None. [B, Nq] or [1, Nq]
        """
        import time
        if self._solver.n_envs > 0:
            envs_idx = self._solver._sanitize_envs_idx(envs_idx)
        else:
            envs_idx = torch.zeros(1, dtype=gs.tc_int, device=gs.device)

        qpos_cur = self._entity.get_qpos()
        qpos_goal, qpos_start = self.validate_input_qpos_batch(qpos_goal, qpos_start, envs_idx)
        
        self._init_rrt_connect_fields()
        self._reset_rrt_connect_fields()
        self._kernel_rrt_connect_init(qpos_start.contiguous(), qpos_goal.contiguous(), envs_idx)

        self._entity.set_qpos(qpos_start)
        self._solver._kernel_detect_collision()
        scene_contact_info = self._solver.collider.contact_data.to_torch(gs.device)
        n_contacts = self._solver.collider.n_contacts.to_torch(gs.device)

        valid_mask = torch.logical_or(
            torch.logical_and(
                scene_contact_info["geom_a"] >= self.geom_start,
                scene_contact_info["geom_a"] < self.geom_end,
            ),
            torch.logical_and(
                scene_contact_info["geom_b"] >= self.geom_start,
                scene_contact_info["geom_b"] < self.geom_end,
            ),
        )
        contact_indices = torch.arange(valid_mask.shape[0], device=valid_mask.device).unsqueeze(1)
        valid_mask = torch.logical_and(valid_mask, contact_indices < n_contacts)

        max_env_collisions = int(torch.max(n_contacts).item())
        valid_mask = valid_mask[:max_env_collisions]
        geom_a = scene_contact_info["geom_a"][:max_env_collisions][valid_mask] # N 
        geom_b = scene_contact_info["geom_b"][:max_env_collisions][valid_mask] # N
        assert len(geom_a) == len(geom_b)

        # NOTE: we will reduce the contacts in batch dim assuming internal geom collisions are the same for a robot
        stacked_tensors = torch.stack((geom_a, geom_b), dim=1)
        unique_pairs = torch.unique(stacked_tensors, dim=0) # N', 2

        gs.logger.info("start rrt connect planning...")
        start = time.time()
        forward_pass = True
        for i_n in range(self._rrt_max_nodes):
            if self._rrt_is_active.to_torch().any():
                self._kernel_rrt_connect_step1(
                    forward_pass=forward_pass,
                    q_limit_lower=self._entity.q_limit[0],
                    q_limit_upper=self._entity.q_limit[1],
                    envs_idx=envs_idx,
                )
                self._solver._kernel_detect_collision()
                self._kernel_rrt_connect_step2(
                    forward_pass=forward_pass,
                    ignore_geom_pairs=unique_pairs,
                    envs_idx=envs_idx,
                )
                forward_pass = not forward_pass
            else:
                break
            
        if self._rrt_is_active.to_torch().any():
            gs.logger.warning(f"rrt connect planning failed in {self._rrt_is_active.to_torch().sum()} envs")

        gs.logger.info(f"rrt connect planning time: {time.time() - start}")
        start = time.time()

        ts = self._rrt_tree_size.to_torch(device=gs.device)[envs_idx]
        g_n = self._rrt_goal_reached_node_idx.to_torch(device=gs.device)[envs_idx] # B

        node_info = self._rrt_node_info.to_torch(device=gs.device)
        parents_idx = node_info["parent_idx"][:, envs_idx]
        children_idx = node_info["child_idx"][:, envs_idx]
        configurations = node_info["configuration"][:, envs_idx]
        res = [g_n]
        for _ in range(ts.max() // 2):
            g_n = parents_idx[g_n, torch.arange(len(envs_idx))]
            res.append(g_n)
            if torch.all(g_n == 0):
                break
        res_idx = torch.stack(list(reversed(res)), dim=0)

        c_n = self._rrt_goal_reached_node_idx.to_torch(device=gs.device)[envs_idx] # B
        res = []
        for _ in range(ts.max() // 2):
            c_n = children_idx[c_n, torch.arange(len(envs_idx))]
            res.append(c_n)
            if torch.all(c_n == 1):
                break
        res_idx = torch.cat([res_idx, torch.stack(res, dim=0)], dim=0)

        sol = configurations[res_idx, torch.arange(len(envs_idx))] # N, B, DoF
        mask = rrt_connect_valid_mask(res_idx)
        sol = align_weypoints_length(sol, mask, mask.sum(dim=0).max())

        sol = self.shortcut_path(
            torch.ones_like(sol[...,0]), sol, iterations=10, ignore_geom_pairs=unique_pairs, envs_idx=envs_idx
        )

        # interpolate to make num_waypoints
        sol = align_weypoints_length(sol, torch.ones_like(sol[...,0]).bool(), num_waypoints)
        gs.logger.info(f"path post-processing time: {time.time() - start}")

        self._kernel_rrt_connect_cleanup(qpos_cur.contiguous(), envs_idx)
        return sol


# ------------------------------------------------------------------------------------
# -------------------------------- OMPL Planners -------------------------------------
# ------------------------------------------------------------------------------------

class OMPL(PathPlanner):
    """
    OMPL path planner interfeace

    Parameters
    ----------
    entity : RigidEntity
        the entity to use the path planning
    planner : str, optional
        The name of the motion planning algorithm to use. Defaults to 'RRTConnect'.
        Supported planners: 'PRM', 'RRT', 'RRTConnect', 'RRTstar', 'EST', 'FMT', 'BITstar', 'ABITstar'.
    """
    def __init__(
        self,
        entity,
        planner: Literal["RRTConnect", "RRT", "PRM", "EST", "FMT", "BITstar", "ABITstar"] = "RRTConnect",
    ):
        super().__init__(entity)
        if not IS_OMPL_AVAILABLE:
            if gs.platform == "Windows":
                gs.raise_exception_from("No pre-compiled binaries of OMPL are distributed on Windows OS.")
            else:
                gs.raise_exception("OMPL is not installed. Please install OMPL to use this planner.")
        self.planner = planner
    
    def plan(
        self,
        qpos_goal,
        qpos_start=None,
        *,
        resolution=0.01,
        timeout=5.0,
        max_retry=1,
        smooth_path=True,
        num_waypoints=100,
        ignore_joint_limit:bool=False,
        ignore_collision:bool=False
    ) -> torch.Tensor:
        """
        Plan a path from `qpos_start` to `qpos_goal`.

        Parameters
        ----------
        qpos_goal : array_like
            The goal state.
        qpos_start : None | array_like, optional
            The start state. If None, the current state of the rigid entity will be used. Defaults to None.
        resolution : float, optiona
            Joint-space resolution in pourcentage. It corresponds to the maximum distance between states to be checked
            for validity along a path segment. Default to 1%.
        timeout : float, optional
            The maximum time (in seconds) allowed for the motion planning algorithm to find a solution. Defaults to 5.0.
        max_retry : float, optional
            Maximum number of retry in case of timeout or convergence failure. Default to 1.
        smooth_path : bool, optional
            Whether to smooth the path after finding a solution. Defaults to True.
        num_waypoints : int, optional
            The number of waypoints to interpolate the path. If None, no interpolation will be performed. Defaults to 100.
        
        Returns
        -------
        waypoints : torch.Tensor
            A list of waypoints representing the planned path. Each waypoint is an array storing the entity's qpos of a single time step.
        """
        assert timeout > 0.0 and math.isfinite(timeout)
        assert max_retry > 0

        if self._solver.n_envs > 0:
            gs.raise_exception("Motion planning is not supported for batched envs with OMPL. Use `RRT` or `RRTConnect` planner instead.")

        if self.n_qs != self.n_dofs:
            gs.raise_exception("Motion planning is not yet supported for rigid entities with free joints.")

        if qpos_start is None:
            qpos_start = self._entity.get_qpos()
        qpos_start = tensor_to_array(qpos_start)
        qpos_goal = tensor_to_array(qpos_goal)

        if qpos_start.shape != (self.n_qs,) or qpos_goal.shape != (self.n_qs,):
            gs.raise_exception("Invalid shape for `qpos_start` or `qpos_goal`.")

        ######### process joint limit ##########
        if ignore_joint_limit:
            gs.logger.warning("This option is deprecated and is no longer doing anything.")
        q_limit_lower, q_limit_upper = self._entity.q_limit[0], self._entity.q_limit[1]

        if (qpos_start < q_limit_lower).any() or (qpos_start > q_limit_upper).any():
            gs.logger.warning(
                "`qpos_start` exceeds joint limit. Relaxing joint limit to contain `qpos_start` for planning."
            )
            q_limit_lower = np.minimum(q_limit_lower, qpos_start)
            q_limit_upper = np.maximum(q_limit_upper, qpos_start)

        if (qpos_goal < q_limit_lower).any() or (qpos_goal > q_limit_upper).any():
            gs.logger.warning(
                "`qpos_goal` exceeds joint limit. Relaxing joint limit to contain `qpos_goal` for planning."
            )
            q_limit_lower = np.minimum(q_limit_lower, qpos_goal)
            q_limit_upper = np.maximum(q_limit_upper, qpos_goal)

        ######### setup OMPL ##########
        ou.setLogLevel(ou.LOG_ERROR)
        space = ob.RealVectorStateSpace(self.n_qs)
        bounds = ob.RealVectorBounds(self.n_qs)

        for i_q in range(self.n_qs):
            bounds.setLow(i_q, q_limit_lower[i_q])
            bounds.setHigh(i_q, q_limit_upper[i_q])
        space.setBounds(bounds)
        ss = og.SimpleSetup(space)

        geoms_idx = tuple(range(self.geom_start, self.geom_start + self.n_geoms))
        mask_collision_pairs = set(
            (i_ga, i_gb) for i_ga, i_gb in self._entity.detect_collision() if i_ga in geoms_idx or i_gb in geoms_idx
        )
        if not ignore_collision and mask_collision_pairs:
            gs.logger.info("Ignoring collision pairs already active for starting pos.")

        def is_ompl_state_valid(state):
            if ignore_collision:
                return True
            qpos = torch.tensor([state[i] for i in range(self.n_qs)], dtype=gs.tc_float, device=gs.device)
            self._entity.set_qpos(qpos)
            collision_pairs = set(map(tuple, self._entity.detect_collision()))
            return not (collision_pairs - mask_collision_pairs)

        ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_ompl_state_valid))

        si = ss.getSpaceInformation()
        si.setStateValidityCheckingResolution(resolution)

        def allocOBValidStateSampler(si):
            vss = ob.UniformValidStateSampler(si)
            vss.setNrAttempts(100)
            return vss

        si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(allocOBValidStateSampler))

        try:
            planner_cls = getattr(og, self.planner)
            if not issubclass(planner_cls, ob.Planner):
                raise ValueError
            planner = planner_cls(si)
        except (AttributeError, ValueError) as e:
            gs.raise_exception_from(f"'{planner}' is not a valid planner. See OMPL documentation for details.", e)
        ss.setPlanner(planner)

        state_start = ob.State(space)
        state_goal = ob.State(space)
        for i_q in range(self.n_qs):
            state_start[i_q] = float(qpos_start[i_q])
            state_goal[i_q] = float(qpos_goal[i_q])
        ss.setStartAndGoalStates(state_start, state_goal)

        ######### solve ##########
        waypoints = []
        for i in range(max_retry):
            # Try solve the motion planning problem
            if ss.getPlanner():
                ss.getPlanner().clear()
            status = ss.solve(timeout)
            status_type = status.getStatus()

            # Check if there was some unrecoverable failure
            if status_type in (
                ob.PlannerStatus.StatusType.UNKNOWN,
                ob.PlannerStatus.StatusType.CRASH,
                ob.PlannerStatus.StatusType.ABORT,
            ):
                gs.raise_exception("Unknown error.")
            if status_type in (
                ob.PlannerStatus.StatusType.INVALID_START,
                ob.PlannerStatus.StatusType.INVALID_GOAL,
                ob.PlannerStatus.StatusType.UNRECOGNIZED_GOAL_TYPE,
                ob.PlannerStatus.StatusType.INFEASIBLE,
            ):
                gs.logger.warning("Path planning infeasible. Returning empty path.")
                break

            # Extract solution if any
            if status:
                # ss.simplifySolution()
                path = ss.getSolutionPath()

                # Simplify path
                if smooth_path:
                    ps = og.PathSimplifier(si)
                    try:
                        # ps.simplifyMax(path)
                        ps.partialShortcutPath(path)
                        ps.ropeShortcutPath(path)
                    except:
                        ps.shortcutPath(path)
                    ps.smoothBSpline(path)

                # Interpolate path
                if num_waypoints is not None:
                    path.interpolate(num_waypoints)

                # Extract waypoints
                waypoints = [
                    torch.as_tensor([state[i] for i in range(self.n_qs)], dtype=gs.tc_float, device=gs.device)
                    for state in path.getStates()
                ]

            # Return once an exact solution was found or maximum number of iterations was reached
            if status_type in (ob.PlannerStatus.StatusType.TIMEOUT, ob.PlannerStatus.StatusType.APPROXIMATE_SOLUTION):
                if i + 1 < max_retry:
                    gs.logger.warning("Path planning did not converge. Trying again...")
                    continue
                else:
                    if waypoints:
                        gs.logger.warning("Path planning did not converge. Returning approximation path.")
                    else:
                        gs.logger.warning("Path planning did not converge. Returning empty path.")
                    break
            gs.logger.info("Path solution found successfully.")
            break

        ########## restore original state #########
        self._entity.set_qpos(qpos_start, zero_velocity=False)

        return waypoints

    

class RRTConnect_OMPL(OMPL):
    """OMPL RRT-Connect planner."""
    def __init__(self, entity, **kwargs):
        super().__init__(entity, planner="RRTConnect", **kwargs)


class RRT_OMPL(OMPL):
    """OMPL RRT planner."""
    def __init__(self, entity, **kwargs):
        super().__init__(entity, planner="RRT", **kwargs)


class PRM_OMPL(OMPL):
    """OMPL PRM (Probabilistic Roadmap) planner."""
    def __init__(self, entity, **kwargs):
        super().__init__(entity, planner="PRM", **kwargs)


class EST_OMPL(OMPL):
    """OMPL EST (Expansive-Space Tree) planner."""
    def __init__(self, entity, **kwargs):
        super().__init__(entity, planner="EST", **kwargs)


class FMT_OMPL(OMPL):
    """OMPL FMT* (Fast Marching Tree) planner."""
    def __init__(self, entity, **kwargs):
        super().__init__(entity, planner="FMT", **kwargs)


class BITstar_OMPL(OMPL):
    """OMPL BIT* (Batch Informed Trees) planner."""
    def __init__(self, entity, **kwargs):
        super().__init__(entity, planner="BITstar", **kwargs)


class ABITstar_OMPL(OMPL):
    """OMPL ABIT* (Asymptotically-Optimal Batch Informed Trees) planner."""
    def __init__(self, entity, **kwargs):
        super().__init__(entity, planner="ABITstar", **kwargs)


# ------------------------------------------------------------------------------------
# ------------------------------------- utils ----------------------------------------
# ------------------------------------------------------------------------------------

def align_weypoints_length(
    path: torch.Tensor, # [N, B, Dof]
    mask: torch.Tensor, # [N, B, ]
    num_points: int
) -> torch.Tensor:
    """
    Aligns each waypoints length to the given num_points.

    Parameters
    ----------
    path: torch.Tensor
        path tensor in [N, B, Dof]
    mask: torch.Tensor
        the masking of path, indicating active waypoints
    num_points: int
        the number of the desired waypoints
    
    Returns
    -------
        A new 2D PyTorch tensor [num_points, B, Dof]
    """
    res = torch.zeros(path.shape[1], num_points, path.shape[-1], device=gs.device)
    for i_b in range(path.shape[1]):
        res[i_b] = torch.nn.functional.interpolate(
            path[mask[:,i_b], i_b].t().unsqueeze(0), size=num_points, mode="linear", align_corners=True
        )[0].t()
    return res.transpose(1,0)


def rrt_valid_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns valid mask of the rrt connect result node indicies

    Parameters
    ----------
    tensor: torch.Tensor
        path tensor in [N, B]
    """
    mask = tensor > 0
    mask_float = mask.float().T.unsqueeze(1)
    kernel = torch.ones(1, 1, 3, device=tensor.device)
    dilated_mask_float = F.conv1d(mask_float, kernel, padding='same')
    dilated_mask = (dilated_mask_float > 0).squeeze(1).T
    return dilated_mask


def rrt_connect_valid_mask(tensor: torch.Tensor) -> torch.Tensor:
    """
    Returns valid mask of the rrt connect result node indicies

    Parameters
    ----------
    tensor: torch.Tensor
        path tensor in [N, B]
    """
    mask = tensor > 1
    mask_float = mask.float().T.unsqueeze(1)
    kernel = torch.ones(1, 1, 3, device=tensor.device)
    dilated_mask_float = F.conv1d(mask_float, kernel, padding='same')
    dilated_mask = (dilated_mask_float > 0).squeeze(1).T
    return dilated_mask
