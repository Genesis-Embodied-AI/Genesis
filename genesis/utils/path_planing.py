from abc import ABC, abstractmethod
from typing import Literal
import numpy as np
import taichi as ti
import torch
import math
import time
import genesis as gs
from genesis.utils.misc import tensor_to_array
try:
    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou
    IS_OMPL_AVAILABLE = True
except ImportError:
    IS_OMPL_AVAILABLE = False
    

@ti.data_oriented
class PathPlanner(ABC):
    """
    Base class for path planners
    """
    def __init__(self, entity):
        self._entity = entity
        self._solver = entity._solver

        # properties
        self.n_geoms = entity.n_geoms
        self.geom_start = entity.geom_start
        self.geom_end = entity.geom_end
        self.n_dofs = entity.n_dofs
        self.n_qs = entity.n_qs
        self._q_start = entity._q_start
        self.q_limit = entity.q_limit

    def validate_input_qpos(self, qpos_goal, qpos_start):
        if qpos_start is None:
            qpos_start = self._entity.get_qpos()
        qpos_start = tensor_to_array(qpos_start)
        qpos_goal = tensor_to_array(qpos_goal)

        if qpos_start.shape != (self.n_qs,) or qpos_goal.shape != (self.n_qs,):
            gs.raise_exception("Invalid shape for `qpos_start` or `qpos_goal`.")

        # NOTE: process joint limit
        if (qpos_start < self.q_limit[0]).any() or (qpos_start > self.q_limit[1]).any():
            gs.raise_exception_from("`qpos_start` exceeds joint limit.")

        if (qpos_goal < self.q_limit[0]).any() or (qpos_goal > self.q_limit[1]).any():
            gs.raise_exception_from("`qpos_goal` exceeds joint limit.")
        return qpos_goal, qpos_start
    
    def validate_input_qpos_batch(self, qpos_goal, qpos_start, envs_idx):
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

        if qpos_start.shape[1] != self.n_qs or qpos_goal.shape[1] != self.n_qs:
            gs.raise_exception("Invalid shape for `qpos_start` or `qpos_goal`.")
        
        # NOTE: process joint limit
        if (qpos_start < self.q_limit[0]).any() or (qpos_start > self.q_limit[1]).any():
            gs.raise_exception_from(
                "`qpos_start` exceeds joint limit. Relaxing joint limit to contain `qpos_start` for planning."
            )

        if (qpos_goal < self.q_limit[0]).any() or (qpos_goal > self.q_limit[1]).any():
            gs.raise_exception_from(
                "`qpos_goal` exceeds joint limit. Relaxing joint limit to contain `qpos_goal` for planning."
            )
        return qpos_goal, qpos_start
        
    @abstractmethod
    def plan(
        self,
        qpos_goal,
        qpos_start=None,
    ):...


    @ti.kernel
    def interpolate_path(
        self,
        tensor: ti.types.ndarray(),     # [B, N, Dof]
        path: ti.types.ndarray(),       # [B, N, Dof]
        sample_ind: ti.types.ndarray(), # [B, 2]
        mask: ti.types.ndarray(),       # [B]
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in range(path.shape[1]):
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
        save_qpos = self._entity.get_qpos()
        path = path.transpose(1,0) # N, B, Dof
        res = torch.zeros(path.shape[1], dtype=gs.tc_int, device=gs.device)
        for qpos in path:
            self._entity.set_qpos(qpos)
            self._solver._kernel_detect_collision()
            self._kernel_check_collision(res, ignore_geom_pairs, envs_idx)
        self._entity.set_qpos(save_qpos)
        return res
        
    @ti.kernel
    def _kernel_check_collision(
        self,
        tensor: ti.types.ndarray(),
        ignore_geom_pairs: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        ti.loop_config(serialize=self._solver._para_level < gs.PARA_LEVEL.ALL)
        for i_b in envs_idx:
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
                    # collision detected
                    collision_detected = True
                    break
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
        for _ in range(iterations):
            ind = torch.multinomial(path_mask, 2).sort()[0] # B, 2
            ind_mask = (ind[:,1] - ind[:,0]) > 1
            tmp_path = path.clone()
            self.interpolate_path(tmp_path, path, ind, ind_mask)
            collision_mask = self.check_collision(tmp_path, ignore_geom_pairs, envs_idx) # B
            path[~collision_mask] = tmp_path[~collision_mask]
        return path.transpose(1,0)

    def smooth_path(self, path):
        pass


    def simplify_path(self, path):
        """
        Paramters
        ---------
        path: torch.Tensor
            path in [N,B,Ndof]
        """


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


@ti.data_oriented
class RRTConnect(PathPlanner):
    pass


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
    resolution : float, optiona
        Joint-space resolution in pourcentage. It corresponds to the maximum distance between states to be checked
        for validity along a path segment. Default to 1%.
    """
    def __init__(
        self,
        entity,
        planner: Literal["RRTConnect", "RRT", "PRM", "EST", "FMT", "BITstar", "ABITstar"] = "RRTConnect",
        resolution=0.01,
    ):
        super().__init__(entity)
        if not IS_OMPL_AVAILABLE:
            if gs.platform == "Windows":
                gs.raise_exception_from("No pre-compiled binaries of OMPL are distributed on Windows OS.")
            else:
                gs.raise_exception("OMPL is not installed. Please install OMPL to use this planner.")
        
        ou.setLogLevel(ou.LOG_ERROR)
        self.space = ob.RealVectorStateSpace(self.n_qs)
        bounds = ob.RealVectorBounds(self.n_qs)

        for i_q in range(self.n_qs):
            bounds.setLow(i_q, self.q_limit[0][i_q])
            bounds.setHigh(i_q, self.q_limit[1][i_q])
        self.space.setBounds(bounds)
        self.ss = og.SimpleSetup(self.space)

        self.si = self.ss.getSpaceInformation()
        self.si.setStateValidityCheckingResolution(resolution)

        def allocOBValidStateSampler(si):
            vss = ob.UniformValidStateSampler(si)
            vss.setNrAttempts(100)
            return vss

        self.si.setValidStateSamplerAllocator(ob.ValidStateSamplerAllocator(allocOBValidStateSampler))

        try:
            planner_cls = getattr(og, planner)
            if not issubclass(planner_cls, ob.Planner):
                raise ValueError
            planner = planner_cls(self.si)
        except (AttributeError, ValueError) as e:
            gs.raise_exception_from(f"'{planner}' is not a valid planner. See OMPL documentation for details.", e)
        self.ss.setPlanner(planner)
    
    def plan(
        self,
        qpos_goal,
        qpos_start=None,
        *,
        timeout=5.0,
        max_retry=1,
        smooth_path=True,
        num_waypoints=100,
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
        if not IS_OMPL_AVAILABLE:
            if gs.platform == "Windows":
                gs.raise_exception_from("No pre-compiled binaries of OMPL are not distributed on Windows OS.")
            else:
                gs.raise_exception_from("OMPL not found.")

        assert timeout > 0.0 and math.isfinite(timeout)
        assert max_retry > 0

        if self._solver.n_envs > 0:
            gs.raise_exception("Motion planning is not supported for batched envs with OMPL. Use `RRT` or `RRTConnect` planner instead.")

        if self.n_qs != self.n_dofs:
            gs.raise_exception("Motion planning is not yet supported for rigid entities with free joints.")

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

        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_ompl_state_valid))

        qpos_goal, qpos_start = self.validate_input_qpos(qpos_goal, qpos_start)

        # setup OMPL
        state_start = ob.State(self.space)
        state_goal = ob.State(self.space)
        for i_q in range(self.n_qs):
            state_start[i_q] = float(qpos_start[i_q])
            state_goal[i_q] = float(qpos_goal[i_q])
        self.ss.setStartAndGoalStates(state_start, state_goal)

        # solve
        waypoints = []
        for i in range(max_retry):
            # Try solve the motion planning problem
            if self.ss.getPlanner():
                self.ss.getPlanner().clear()
            status = self.ss.solve(timeout)
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
                path = self.ss.getSolutionPath()

                import time

                start = time.time()
                # Simplify path
                if smooth_path:
                    ps = og.PathSimplifier(self.si)
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

                print("smooting:", time.time() - start)

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
    def __init__(self, entity, resolution: float = 0.01, **kwargs):
        super().__init__(entity, planner="RRTConnect", resolution=resolution, **kwargs)


class RRT_OMPL(OMPL):
    """OMPL RRT planner."""
    def __init__(self, entity, resolution: float = 0.01, **kwargs):
        super().__init__(entity, planner="RRT", resolution=resolution, **kwargs)


class PRM_OMPL(OMPL):
    """OMPL PRM (Probabilistic Roadmap) planner."""
    def __init__(self, entity, resolution: float = 0.01, **kwargs):
        super().__init__(entity, planner="PRM", resolution=resolution, **kwargs)


class EST_OMPL(OMPL):
    """OMPL EST (Expansive-Space Tree) planner."""
    def __init__(self, entity, resolution: float = 0.01, **kwargs):
        super().__init__(entity, planner="EST", resolution=resolution, **kwargs)


class FMT_OMPL(OMPL):
    """OMPL FMT* (Fast Marching Tree) planner."""
    def __init__(self, entity, resolution: float = 0.01, **kwargs):
        super().__init__(entity, planner="FMT", resolution=resolution, **kwargs)


class BITstar_OMPL(OMPL):
    """OMPL BIT* (Batch Informed Trees) planner."""
    def __init__(self, entity, resolution: float = 0.01, **kwargs):
        super().__init__(entity, planner="BITstar", resolution=resolution, **kwargs)


class ABITstar_OMPL(OMPL):
    """OMPL ABIT* (Asymptotically-Optimal Batch Informed Trees) planner."""
    def __init__(self, entity, resolution: float = 0.01, **kwargs):
        super().__init__(entity, planner="ABITstar", resolution=resolution, **kwargs)


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
        A new 2D PyTorch tensor
    """
    res = torch.zeros(path.shape[1], num_points, path.shape[-1], device=gs.device)
    for i_b in range(path.shape[1]):
        res[i_b] = torch.nn.functional.interpolate(
            path[mask[:,i_b], i_b].t().unsqueeze(0), size=num_points, mode="linear", align_corners=True
        )[0].t()
    return res.transpose(1,0)


def move_padding_to_tail(tensor: torch.Tensor) -> torch.Tensor:
    """
    Moves leading zero-padding to the tail for each column in a 2D tensor.
    The tail is padded with the last value of the original column.

    Parameters
    ----------
    tensor: torch.Tensor
        A 2D PyTorch tensor of shape [N, B] with integer types.

    Returns 
    -------
        A new 2D PyTorch tensor with the padding transformed.
    """
    n_dim, b_dim = tensor.shape
    non_zero_mask = (tensor != 0)

    # Find the index of the first non-zero element in each column.
    # .argmax() finds the first 'True' (or 1) along dimension 0.
    first_nonzero_indices = non_zero_mask.int().argmax(dim=0)

    # Handle the edge case of all-zero columns. For these columns, argmax
    # returns 0, which is incorrect for our logic. We find these columns
    # and set their first non-zero index to N, effectively treating them
    # as having no content and full padding.
    is_all_zero_column = ~non_zero_mask.any(dim=0)
    first_nonzero_indices[is_all_zero_column] = n_dim

    content_len = n_dim - first_nonzero_indices # [B]

    arange_n = torch.arange(n_dim, device=tensor.device).unsqueeze(1)
    is_padding_part = arange_n >= content_len # [N, B]
    # The index for all padding values is the last row (N-1).
    last_row_index = torch.full((1, b_dim), n_dim - 1, device=tensor.device)
    # Create the indices for the content part of each column.
    # This effectively "rolls" the data up by `first_nonzero_indices`.
    content_indices = arange_n + first_nonzero_indices # [N, B]

    # Use the `is_padding_part` mask to choose the final indices.
    # If it's a content part, use the rolled `content_indices`.
    # If it's a padding part, use the `last_row_index`.
    final_indices = torch.where(is_padding_part, last_row_index, content_indices)
    result_tensor = torch.gather(tensor, 0, final_indices)

    return result_tensor