"""
Skeleton Humanoid Environment for Genesis

Implementation of LocoMujoco's SkeletonTorque environment adapted for Genesis physics.
Based on: /home/ez/Documents/loco-mujoco/loco_mujoco/environments/humanoids/skeletons.py
"""

import torch
import numpy as np
from typing import Dict, Tuple, Any, List
import warnings

import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapters.genesis_loco_env import GenesisLocoBaseEnv


class SkeletonHumanoidEnv(GenesisLocoBaseEnv):
    """
    Skeleton Humanoid Environment using torque control
    
    Matches LocoMujoco's SkeletonTorque environment structure:
    - 27 actions (after removing foot joints with box feet)
    - 59 observations (root + joint states)
    - Torque control with direct force application
    """
    
    def __init__(self, 
                 num_envs: int = 1024,
                 episode_length_s: float = 10.0,
                 dt: float = 0.02,
                 use_box_feet: bool = True,
                 disable_arms: bool = False,
                 show_viewer: bool = False,
                 **kwargs):
        
        self.use_box_feet = use_box_feet
        self.disable_arms = disable_arms
        
        # Initialize base environment
        super().__init__(
            num_envs=num_envs,
            robot_file="/home/ez/Documents/Genesis/genesis_loco/skeleton/genesis_skeleton_torque.xml",
            dt=dt,
            episode_length_s=episode_length_s,
            obs_cfg=self._get_obs_config(),
            reward_cfg=self._get_reward_config(),
            show_viewer=show_viewer,
            **kwargs
        )
        
        # Initialize skeleton-specific components
        self._init_skeleton_components()
    
    def _get_obs_config(self) -> Dict[str, Any]:
        """Observation configuration matching LocoMujoco skeleton"""
        return {
            "include_root_pos_z": True,
            "include_root_quat": True,
            "include_root_vel": True,
            "include_joint_pos": True,
            "include_joint_vel": True,
        }
    
    def _get_reward_config(self) -> Dict[str, Any]:
        """Reward configuration for skeleton locomotion"""
        return {
            "trajectory_tracking": 1.0,
            "upright_orientation": 0.2,
            "energy_efficiency": -0.01,
            "root_height": 0.1,
        }
    
    def _init_skeleton_components(self):
        """Initialize skeleton-specific parameters and buffers"""
        # Get action specification matching LocoMujoco
        self._setup_action_spec()
        
        # Setup torque control
        self._setup_torque_control()
        
        # Initialize buffers
        self._init_skeleton_buffers()
    
    def _setup_action_spec(self):
        """Setup action specification with direct LocoMujoco action name to joint index mapping"""
        # print(f"Available DOF names: {self.dof_names}")
        
        # Direct mapping from LocoMujoco action names to Genesis joint indices
        action_to_joint_mapping = {
            # Lumbar spine
            "mot_lumbar_ext": "lumbar_extension",
            "mot_lumbar_bend": "lumbar_bending", 
            "mot_lumbar_rot": "lumbar_rotation",
            
            # Right leg
            "mot_hip_flexion_r": "hip_flexion_r",
            "mot_hip_adduction_r": "hip_adduction_r",
            "mot_hip_rotation_r": "hip_rotation_r",
            "mot_knee_angle_r": "knee_angle_r", 
            "mot_ankle_angle_r": "ankle_angle_r",
            "mot_subtalar_angle_r": "subtalar_angle_r",
            "mot_mtp_angle_r": "mtp_angle_r",
            
            # Left leg  
            "mot_hip_flexion_l": "hip_flexion_l",
            "mot_hip_adduction_l": "hip_adduction_l",
            "mot_hip_rotation_l": "hip_rotation_l",
            "mot_knee_angle_l": "knee_angle_l",
            "mot_ankle_angle_l": "ankle_angle_l", 
            "mot_subtalar_angle_l": "subtalar_angle_l",
            "mot_mtp_angle_l": "mtp_angle_l",
            
            # Right arm
            "mot_shoulder_flex_r": "arm_flex_r",
            "mot_shoulder_add_r": "arm_add_r",
            "mot_shoulder_rot_r": "arm_rot_r",
            "mot_elbow_flex_r": "elbow_flex_r",
            "mot_pro_sup_r": "pro_sup_r",
            "mot_wrist_flex_r": "wrist_flex_r",
            "mot_wrist_dev_r": "wrist_dev_r",
            
            # Left arm
            "mot_shoulder_flex_l": "arm_flex_l",
            "mot_shoulder_add_l": "arm_add_l", 
            "mot_shoulder_rot_l": "arm_rot_l",
            "mot_elbow_flex_l": "elbow_flex_l",
            "mot_pro_sup_l": "pro_sup_l",
            "mot_wrist_flex_l": "wrist_flex_l",
            "mot_wrist_dev_l": "wrist_dev_l",
        }
        
        # Build action spec and direct joint index mapping
        self.action_spec = []
        self.action_to_joint_idx = {}
        
        for action_name, joint_name in action_to_joint_mapping.items():
            if joint_name in self.dof_names:
                # Apply filtering rules
                if self.use_box_feet and action_name in ["mot_subtalar_angle_l", "mot_mtp_angle_l", 
                                                        "mot_subtalar_angle_r", "mot_mtp_angle_r"]:
                    continue
                self.action_to_joint_idx[action_name] = self.robot.get_joint(joint_name).dof_idx_local
                local_dof_idx = self.robot.get_joint(joint_name).dof_idx_local
                local_joint_idx = self.robot.get_joint(joint_name).idx_local
                # print(f"Action name: {action_name}, Joint name: {joint_name}, Joint_index: {local_joint_idx}, Dof_idx_local: {local_dof_idx}")
                self.action_spec.append(action_name)
            
        self.num_skeleton_actions = len(self.action_spec)
        
        # print(f"Skeleton actions: {self.num_skeleton_actions}")
        # print(f"Action spec: {self.action_spec}")
        # print(f"Action to joint mapping: {self.action_to_joint_idx}")
    
    def _setup_torque_control(self):
        """Setup pure torque control using control_dofs_force"""
        print("Setting up pure torque control with control_dofs_force...")
        
        # Explicitly set PD gains to zero to ensure no interference
        # kp_values = torch.zeros(self.num_dofs, device=self.device)
        # kv_values = torch.zeros(self.num_dofs, device=self.device)
        # self.robot.set_dofs_kp(kp_values)
        # self.robot.set_dofs_kv(kv_values)
        # print(f"Set all PD gains to zero: kp={kp_values[0]}, kv={kv_values[0]}")
        
        # Set torque limits
        torque_limit = 1000.0   # revise torque limit to be more reasonable
        limits = torch.ones(self.num_dofs, device=self.device) * torque_limit
        self.robot.set_dofs_force_range(-limits, limits)
        
        # print(f"Torque control setup complete. Using control_dofs_force with limits: ±{torque_limit}")
        # print("Note: Explicitly zeroed PD gains and using control_dofs_force")
    
    def setup_pd_control(self):
        """Setup PD gains for position control"""
        print("Setting up PD control gains for control_dofs_position...")
        
        # Set reasonable PD gains for humanoid control
        kp_values = torch.ones(self.num_dofs, device=self.device) * 500.0  # Position gain
        kv_values = torch.ones(self.num_dofs, device=self.device) * 50.0   # Velocity gain
        
        # Lower gains for spine joints for stability
        for joint_name in ["lumbar_extension", "lumbar_bending", "lumbar_rotation"]:
            if joint_name in self.dof_names:
                idx = self.dof_names.index(joint_name)
                kp_values[idx] = 200.0
                kv_values[idx] = 20.0
        
        self.robot.set_dofs_kp(kp_values)
        self.robot.set_dofs_kv(kv_values)
        print(f"Set PD gains: kp=500.0, kv=50.0 (spine: kp=200.0, kv=20.0)")
    
    def _init_skeleton_buffers(self):
        """Initialize skeleton-specific state buffers"""
        # Previous actions for observations and smoothness
        self.prev_actions = torch.zeros((self.num_envs, self.num_skeleton_actions), 
                                       device=self.device)
        
        # Energy consumption tracking
        self.energy_consumption = torch.zeros((self.num_envs,), device=self.device)
        
        # Target velocity for locomotion
        self.target_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_velocity[:, 0] = 1.0  # Default forward velocity
    
    def _apply_actions(self, actions: torch.Tensor):
        """Apply torque actions directly to joints using direct mapping"""
        
        # Create torque tensor
        torques = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        
        # Direct mapping using pre-computed indices
        for i, action_name in enumerate(self.action_spec):
            if i < actions.shape[-1]:  # Ensure we don't exceed action dimensions
                joint_idx = self.action_to_joint_idx[action_name]
                torques[:, joint_idx] = actions[:, i]
        
        # Apply torques using control_dofs_force
        self.robot.control_dofs_force(torques)
        
        # Track energy consumption
        power = torch.sum(torch.abs(torques * self.dof_vel), dim=1)
        self.energy_consumption += power * self.dt
    
    
    def _get_observations(self) -> torch.Tensor:
        """Get observations matching LocoMujoco skeleton structure"""
        obs_components = []
        
        # Root pose (no x,y position, only z + quaternion) - 5D
        obs_components.append(self.root_pos[:, 2:3])  # z position only
        obs_components.append(self.root_quat)         # quaternion (4D)
        
        # Joint positions - simplified to match available DOFs
        num_joint_obs = min(22, self.num_dofs)  # LocoMujoco has 22 joint positions
        joint_pos = self.dof_pos[:, :num_joint_obs]
        obs_components.append(joint_pos)
        
        # Root velocity (full 6D) 
        obs_components.append(self.root_lin_vel)      # 3D
        obs_components.append(self.root_ang_vel)      # 3D
        
        # Joint velocities
        joint_vel = self.dof_vel[:, :num_joint_obs]
        obs_components.append(joint_vel)
        
        obs = torch.cat(obs_components, dim=-1)
        
        # Update buffers
        if self.obs_buf is None:
            self.obs_buf = torch.zeros_like(obs)
        self.obs_buf[:] = obs
        
        self.extras["observations"]["policy"] = obs
        return obs
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Step environment with action history tracking"""
        # Store for next observation
        self._current_actions = actions.clone()
        
        # Call parent step
        obs, rewards, dones, info = super().step(actions)
        
        # Update action history
        self.prev_actions[:] = actions[:, :self.num_skeleton_actions]
        
        return obs, rewards, dones, info
    
    def _check_termination(self) -> torch.Tensor:
        """Check skeleton-specific termination conditions"""
        done = super()._check_termination()
        
        # Height limits (LocoMujoco: 0.8-1.1m)
        height_violation = (self.root_pos[:, 2] < 0.8) | (self.root_pos[:, 2] > 1.1)
        done = done | height_violation
        
        # Orientation limits (45 degree roll/pitch)
        root_euler = quat_to_xyz(self.root_quat)  # Returns roll, pitch, yaw in radians
        extreme_tilt = (torch.abs(root_euler[:, 0]) > torch.pi/4) | (torch.abs(root_euler[:, 1]) > torch.pi/4)
        done = done | extreme_tilt
        
        return done
    
    def _reset_robot_pose(self, env_ids: torch.Tensor):
        """Reset to upright standing pose"""
        num_reset = len(env_ids)
        
        # Zero joint positions
        default_pose = torch.zeros((num_reset, self.num_dofs), device=self.device)
        
        # Standing position
        default_root_pos = torch.tensor([0.0, 0.0, 0.975], device=self.device).repeat(num_reset, 1)
        default_root_quat = torch.tensor([0.7071067811865475, 0.7071067811865475, 0.0, 0.0], device=self.device).repeat(num_reset, 1)
        
        # Apply reset
        self.robot.set_dofs_position(default_pose, envs_idx=env_ids, zero_velocity=True)
        self.robot.set_pos(default_root_pos, envs_idx=env_ids, zero_velocity=True)
        self.robot.set_quat(default_root_quat, envs_idx=env_ids, zero_velocity=True)
        
        # Reset buffers
        self.energy_consumption[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.target_velocity[env_ids] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
    
    # Reward functions
    def _reward_upright_orientation(self) -> torch.Tensor:
        """Reward staying upright"""
        root_euler = quat_to_xyz(self.root_quat)  # Returns roll, pitch, yaw in radians
        orientation_error = torch.abs(root_euler[:, 0]) + torch.abs(root_euler[:, 1])
        return torch.exp(-orientation_error * 2.0)
    
    def _reward_energy_efficiency(self) -> torch.Tensor:
        """Penalize energy consumption"""
        return torch.sum(torch.abs(self.prev_actions), dim=1)
    
    def _reward_root_height(self) -> torch.Tensor:
        """Reward proper standing height"""
        height_error = torch.abs(self.root_pos[:, 2] - 1.0)
        return torch.exp(-height_error * 5.0)
    
    @property
    def num_observations(self) -> int:
        """Calculate observation space size"""
        # Root: 5 (z + quat) + Joints: 22 + Root vel: 6 + Joint vel: 22 = 55
        return 5 + min(22, self.num_dofs) + 6 + min(22, self.num_dofs)
    
    @property 
    def num_actions(self) -> int:
        """Action space size"""
        return self.num_skeleton_actions