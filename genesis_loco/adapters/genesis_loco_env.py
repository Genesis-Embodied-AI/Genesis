"""
Adaptation of LocoMujoco Environment for Genesis 

Uses Genesis physics backend instead of Mujoco
"""

import warnings
import torch
import math
from typing import Dict, Optional, Union, Any, List, Tuple
import numpy as np

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from loco_mujoco.trajectory import TrajectoryHandler
from loco_mujoco.trajectory import Trajectory, TrajState, TrajectoryTransitions


class GenesisLocoBaseEnv:
    """
    Base class for LocoMujoco adapted environments using Genesis physics

    This class provides the core functionality for running LocoMujoco imitation learning
    environments with Genesis as the physics backend instead of MuJoCo.
    """

    def __init__(self,
                 num_envs: int,
                 robot_file: str,
                 dt: float = 0.02,
                 episode_length_s: float = 20.0,
                 obs_cfg: Dict = None,
                 reward_cfg: Dict = None,
                 th_params: Dict = None,
                 traj_params: Dict = None,
                 show_viewer: bool = False,
                 **kwargs):
        """
        Constructor.

        Args:
            num_envs (int): Number of parallel environments
            robot_file (str): Path to robot URDF/MJCF file
            dt (float): Physics timestep
            episode_length_s (float): Episode length in seconds
            obs_cfg (Dict): Observation configuration
            reward_cfg (Dict): Reward configuration
            th_params (Dict): Dictionary of parameters for the trajectory handler
            traj_params (Dict): Dictionary of parameters to load trajectories
            show_viewer (bool): Whether to show the viewer
            **kwargs: Additional arguments
        """
        
        self.num_envs = num_envs
        self.robot_file = robot_file
        self.dt = dt
        self.episode_length_s = episode_length_s
        self.max_episode_length = math.ceil(episode_length_s / dt)
        self.device = gs.device
        
        # Store configurations
        self.obs_cfg = obs_cfg or {}
        self.reward_cfg = reward_cfg or {}
        self.th_params = th_params or {}
        
        # Initialize Genesis scene
        self._init_genesis_scene(show_viewer)
        
        # Load robot
        self._load_robot()
        
        # Build scene
        self.scene.build(n_envs=num_envs)
        
        # Initialize buffers and state
        self._init_buffers()
        
        # Initialize trajectory handler
        self.th = None
        if traj_params:
            self.load_trajectory(**traj_params)
            
        # Initialize reward functions
        self._init_reward_functions()
    
    def _init_genesis_scene(self, show_viewer: bool = False):
        """Initialize Genesis scene with appropriate settings"""
        self.scene = gs.Scene(
            show_FPS=False,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3.0, 0.0, 2.0),
                camera_lookat=(0.0, 0.0, 1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )
        
        # Add ground plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
    
    def _load_robot(self):
        """Load robot model into Genesis scene"""
        # Determine robot file type and load accordingly
        if self.robot_file.endswith('.xml'):
            self.robot = self.scene.add_entity(gs.morphs.MJCF(file=self.robot_file))
        elif self.robot_file.endswith('.urdf'):
            self.robot = self.scene.add_entity(gs.morphs.URDF(file=self.robot_file))
        else:
            raise ValueError(f"Unsupported robot file format: {self.robot_file}")

        # Store robot properties for later use
        self.num_dofs = self.robot.n_dofs
        # Get joint names from the robot's joint list
        self.dof_names = [joint.name for joint in self.robot.joints]
    
    def _init_buffers(self):
        """Initialize state buffers for environments"""
        # Episode management
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Robot state buffers
        self.dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float32)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float32)
        self.root_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.root_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self.root_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.root_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        
        # Observation and reward buffers
        self.obs_buf = None  # Will be initialized based on observation space
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        
        # Trajectory tracking buffers
        self.traj_idx = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.traj_time = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        
        # Extras for logging
        self.extras = {"observations": {}}
    
    def _init_reward_functions(self):
        """Initialize reward functions based on configuration"""
        self.reward_functions = {}
        self.episode_sums = {}
        
        for reward_name, reward_scale in self.reward_cfg.items():
            if hasattr(self, f"_reward_{reward_name}"):
                self.reward_functions[reward_name] = getattr(self, f"_reward_{reward_name}")
                self.episode_sums[reward_name] = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
            else:
                warnings.warn(f"Reward function '_reward_{reward_name}' not found, skipping.")
    
    def load_trajectory(self, 
                        traj: Trajectory = None,
                        traj_path: str = None,
                        warn: bool = True) -> None:
        """
        Loads trajectories. If there were trajectories loaded already, this function overrides the latter.

        Args:
            traj (Trajectory): Datastructure containing all trajectory files. If traj_path is specified, this
                should be None.
            traj_path (string): path with the trajectory for the model to follow. Should be a numpy zipped file (.npz)
                with a 'traj_data' array and possibly a 'split_points' array inside. The 'traj_data'
                should be in the shape (joints x observations). If traj_files is specified, this should be None.
            warn (bool): If True, a warning will be raised.
        """

        if self.th is not None and warn:
            warnings.warn("New trajectories loaded, which overrides the old ones.", RuntimeWarning)

        # Create a mock model object with necessary attributes for TrajectoryHandler
        mock_model = type('MockModel', (), {
            'nq': self.num_dofs,
            'nv': self.num_dofs,
            'dt': self.dt,
            'joint_names': self.dof_names
        })()
        
        th_params = self.th_params.copy()
        self.th = TrajectoryHandler(model=mock_model, warn=warn, traj_path=traj_path,
                                    traj=traj, control_dt=self.dt, **th_params)
        
        # Initialize trajectory tracking state
        self._reset_trajectory_state()
    
    def _reset_trajectory_state(self, env_ids: Optional[torch.Tensor] = None):
        """Reset trajectory tracking state for given environments"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        if self.th is not None:
            # Reset trajectory indices and time
            self.traj_idx[env_ids] = 0
            self.traj_time[env_ids] = 0.0
            
            # Sample random starting positions if desired
            if hasattr(self.th, 'sample_init_state'):
                for env_id in env_ids:
                    init_state = self.th.sample_init_state()
                    self._apply_trajectory_state(init_state, env_id.unsqueeze(0))
    
    def _apply_trajectory_state(self, traj_state: TrajState, env_ids: torch.Tensor):
        """Apply trajectory state to Genesis physics simulation"""
        if traj_state is None:
            return
            
        # Apply joint positions and velocities
        if hasattr(traj_state, 'qpos') and traj_state.qpos is not None:
            qpos_tensor = torch.tensor(traj_state.qpos, device=self.device, dtype=torch.float32)
            if len(qpos_tensor.shape) == 1:
                qpos_tensor = qpos_tensor.unsqueeze(0).repeat(len(env_ids), 1)
            self.robot.set_dofs_position(qpos_tensor, envs_idx=env_ids)
            
        if hasattr(traj_state, 'qvel') and traj_state.qvel is not None:
            qvel_tensor = torch.tensor(traj_state.qvel, device=self.device, dtype=torch.float32)
            if len(qvel_tensor.shape) == 1:
                qvel_tensor = qvel_tensor.unsqueeze(0).repeat(len(env_ids), 1)
            self.robot.set_dofs_velocity(qvel_tensor, envs_idx=env_ids)
            
        # Apply root pose if available
        if hasattr(traj_state, 'root_pos') and traj_state.root_pos is not None:
            root_pos_tensor = torch.tensor(traj_state.root_pos, device=self.device, dtype=torch.float32)
            if len(root_pos_tensor.shape) == 1:
                root_pos_tensor = root_pos_tensor.unsqueeze(0).repeat(len(env_ids), 1)
            self.robot.set_pos(root_pos_tensor, envs_idx=env_ids)
            
        if hasattr(traj_state, 'root_quat') and traj_state.root_quat is not None:
            root_quat_tensor = torch.tensor(traj_state.root_quat, device=self.device, dtype=torch.float32)
            if len(root_quat_tensor.shape) == 1:
                root_quat_tensor = root_quat_tensor.unsqueeze(0).repeat(len(env_ids), 1)
            self.robot.set_quat(root_quat_tensor, envs_idx=env_ids)
    
    def _update_robot_state(self):
        """Update robot state buffers from Genesis simulation"""
        # Update joint states
        self.dof_pos[:] = self.robot.get_dofs_position()
        self.dof_vel[:] = self.robot.get_dofs_velocity()
        
        # Update root states
        self.root_pos[:] = self.robot.get_pos()
        self.root_quat[:] = self.robot.get_quat()
        self.root_lin_vel[:] = self.robot.get_vel()
        self.root_ang_vel[:] = self.robot.get_ang()
    
    def _get_trajectory_target_state(self) -> Optional[TrajState]:
        """Get target state from trajectory at current time"""
        if self.th is None:
            return None
            
        # Get current trajectory state
        current_traj_state = self.th.get_current_traj_data()
        return current_traj_state
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step the environment forward one timestep
        
        Args:
            actions: Action tensor [num_envs, action_dim]
            
        Returns:
            observations, rewards, dones, info
        """
        # Apply actions (can be overridden by subclasses)
        self._apply_actions(actions)
        
        # Step Genesis physics
        self.scene.step()
        
        # Update robot state from simulation
        self._update_robot_state()
        
        # Update episode length
        self.episode_length_buf += 1
        
        # Update trajectory time if using trajectories
        if self.th is not None:
            self.traj_time += self.dt
            # TODO: Update trajectory handler state
        
        # Check for episode termination
        self.reset_buf = self._check_termination()
        
        # Reset environments that are done
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).reshape((-1,))
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        # Compute rewards
        self._compute_rewards()
        
        # Get observations
        obs = self._get_observations()
        
        return obs, self.rew_buf, self.reset_buf, self.extras
    
    def _apply_actions(self, actions: torch.Tensor):
        """Apply actions to the robot (to be implemented by subclasses)"""
        
        # Default implementation: apply actions as joint position targets
        if actions.shape[-1] == self.num_dofs:
            self.robot.control_dofs_position(actions)
        else:
            warnings.warn(f"Action dimension {actions.shape[-1]} does not match DOF count {self.num_dofs}")
    
    def _check_termination(self) -> torch.Tensor:
        """Check if episodes should terminate"""
        # Default termination: episode length exceeded
        done = self.episode_length_buf >= self.max_episode_length
        
        # Add trajectory-based termination if available
        if self.th is not None:
            # TODO: Add trajectory completion termination
            pass
            
        return done
    
    def _compute_rewards(self):
        """Compute rewards for current step"""
        self.rew_buf[:] = 0.0
        
        for reward_name, reward_func in self.reward_functions.items():
            if reward_name in self.reward_cfg:
                reward_scale = self.reward_cfg[reward_name]
                reward_value = reward_func()
                
                self.rew_buf += reward_scale * reward_value
                self.episode_sums[reward_name] += reward_scale * reward_value
    
    def _get_observations(self) -> torch.Tensor:
        """Get observations (to be implemented by subclasses)"""
        # Default implementation: return joint states
        obs_components = [
            self.dof_pos,
            self.dof_vel,
            self.root_pos,
            self.root_quat,
        ]
        
        obs = torch.cat(obs_components, dim=-1)
        
        if self.obs_buf is None:
            self.obs_buf = torch.zeros_like(obs)
        
        self.obs_buf[:] = obs
        self.extras["observations"]["policy"] = obs
        
        return obs
    
    def reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments"""
        if len(env_ids) == 0:
            return
            
        # Reset episode length
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        
        # Reset trajectory tracking
        self._reset_trajectory_state(env_ids)
        
        # Reset robot to default pose (can be overridden by subclasses)
        self._reset_robot_pose(env_ids)
        
        # Log episode rewards
        self._log_episode_rewards(env_ids)
    
    def _reset_robot_pose(self, env_ids: torch.Tensor):
        """Reset robot to default pose"""
        # Default implementation: set to zero pose
        default_dof_pos = torch.zeros((len(env_ids), self.num_dofs), device=self.device, dtype=torch.float32)
        default_root_pos = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float32).repeat(len(env_ids), 1)
        default_root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=torch.float32).repeat(len(env_ids), 1)
        
        self.robot.set_dofs_position(default_dof_pos, envs_idx=env_ids, zero_velocity=True)
        self.robot.set_pos(default_root_pos, envs_idx=env_ids, zero_velocity=True)
        self.robot.set_quat(default_root_quat, envs_idx=env_ids, zero_velocity=True)
    
    def _log_episode_rewards(self, env_ids: torch.Tensor):
        """Log episode reward summaries"""
        if not hasattr(self, 'extras'):
            self.extras = {}
        if 'episode' not in self.extras:
            self.extras['episode'] = {}
            
        for reward_name, reward_sum in self.episode_sums.items():
            mean_reward = torch.mean(reward_sum[env_ids]).item()
            self.extras['episode'][f'rew_{reward_name}'] = mean_reward / self.episode_length_s
            reward_sum[env_ids] = 0.0
    
    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Reset all environments"""
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
        obs = self._get_observations()
        return obs, self.extras
    
    def get_observations(self) -> Tuple[torch.Tensor, Dict]:
        """Get current observations"""
        obs = self._get_observations()
        return obs, self.extras
    
    def get_privileged_observations(self) -> Optional[torch.Tensor]:
        """Get privileged observations (default: None)"""
        return None
    
    # Default reward functions (can be overridden by subclasses)
    def _reward_trajectory_tracking(self) -> torch.Tensor:
        """Reward for tracking trajectory"""
        if self.th is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
            
        target_state = self._get_trajectory_target_state()
        if target_state is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        
        # Compute position tracking error
        if hasattr(target_state, 'qpos') and target_state.qpos is not None:
            target_qpos = torch.tensor(target_state.qpos, device=self.device, dtype=torch.float32)
            if len(target_qpos.shape) == 1:
                target_qpos = target_qpos.unsqueeze(0).repeat(self.num_envs, 1)
            
            pos_error = torch.norm(self.dof_pos - target_qpos, dim=-1)
            return torch.exp(-pos_error)
        
        return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
    
    def _reward_velocity_tracking(self) -> torch.Tensor:
        """Reward for tracking trajectory velocities"""
        if self.th is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
            
        target_state = self._get_trajectory_target_state()
        if target_state is None or not hasattr(target_state, 'qvel') or target_state.qvel is None:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        
        target_qvel = torch.tensor(target_state.qvel, device=self.device, dtype=torch.float32)
        if len(target_qvel.shape) == 1:
            target_qvel = target_qvel.unsqueeze(0).repeat(self.num_envs, 1)
        
        vel_error = torch.norm(self.dof_vel - target_qvel, dim=-1)
        return torch.exp(-vel_error)
    
    def _reward_action_smoothness(self, actions: torch.Tensor) -> torch.Tensor:
        """Reward for smooth actions"""
        if not hasattr(self, 'prev_actions'):
            self.prev_actions = torch.zeros_like(actions)
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        
        action_diff = torch.norm(actions - self.prev_actions, dim=-1)
        self.prev_actions = actions.clone()
        
        return torch.exp(-action_diff)

