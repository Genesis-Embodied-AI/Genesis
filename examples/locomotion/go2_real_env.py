import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

try:
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.utils.thread import RecurrentThread
    UNITREE_SDK_AVAILABLE = True
except ImportError:
    print("Warning: Unitree SDK not available. Install unitree_sdk2py for real robot control.")
    UNITREE_SDK_AVAILABLE = False


class Go2RealEnv:
    def __init__(self, 
                 env_cfg: Dict,
                 obs_cfg: Dict,
                 device: str = "cpu"):
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.device = device
        
        # Joint configuration
        self.joint_names = env_cfg["joint_names"]
        self.default_joint_angles = env_cfg["default_joint_angles"]
        self.num_actions = env_cfg["num_actions"]
        self.action_scale = env_cfg["action_scale"]
        self.kp = env_cfg["kp"]
        self.kd = env_cfg["kd"]
        
        # Observation configuration
        self.num_obs = obs_cfg["num_obs"]
        self.obs_scales = obs_cfg["obs_scales"]
        
        # Initialize joint mappings
        self._setup_joint_mappings()
        
        # Initialize state variables
        self.obs_buf = torch.zeros(1, self.num_obs, device=self.device)
        self.actions = torch.zeros(1, self.num_actions, device=self.device)
        self.last_actions = torch.zeros(1, self.num_actions, device=self.device)
        
        # Robot state
        self.base_ang_vel = torch.zeros(1, 3, device=self.device)
        self.projected_gravity = torch.zeros(1, 3, device=self.device)
        self.dof_pos = torch.zeros(1, 12, device=self.device)
        self.dof_vel = torch.zeros(1, 12, device=self.device)
        self.default_dof_pos = torch.zeros(1, 12, device=self.device)
        
        # Episode tracking
        self.episode_length_buf = torch.zeros(1, device=self.device)
        self.max_episode_length = int(env_cfg["episode_length_s"] * 500)  # Assuming 500Hz
        self.reset_buf = torch.zeros(1, dtype=torch.bool, device=self.device)
        
        # Initialize robot communication if SDK is available
        if UNITREE_SDK_AVAILABLE:
            self._init_robot_communication()
        
        # Set default joint positions
        self._set_default_joint_positions()
        
    def _setup_joint_mappings(self):
        """Setup mapping between joint names and indices"""
        self.joint_name_to_index = {name: i for i, name in enumerate(self.joint_names)}
        
        # Create mapping from default joint angles to ordered positions
        self.default_positions = []
        for joint_name in self.joint_names:
            self.default_positions.append(self.default_joint_angles[joint_name])
        
    def _set_default_joint_positions(self):
        """Set default joint positions"""
        self.default_dof_pos = torch.tensor([self.default_positions], 
                                          device=self.device, dtype=torch.float32)
        
    def _init_robot_communication(self):
        """Initialize communication with the real robot"""
        if not UNITREE_SDK_AVAILABLE:
            raise RuntimeError("Unitree SDK not available")
            
        # Initialize publishers and subscribers
        self.state_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.cmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        
        # Initialize low command
        self.cmd = LowCmd_()
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        
        # Initialize motor commands
        for i in range(12):
            self.cmd.motor_cmd[i].mode = 0x01  # Position control mode
            self.cmd.motor_cmd[i].q = self.default_positions[i]
            self.cmd.motor_cmd[i].kp = self.kp
            self.cmd.motor_cmd[i].kd = self.kd
            self.cmd.motor_cmd[i].tau = 0.0
        
        # Start communication thread
        self.communication_thread = RecurrentThread(
            interval=0.002,  # 500Hz
            target=self._communication_loop,
            daemon=True
        )
        self.communication_thread.start()
        
        # Wait for initial state
        time.sleep(0.1)
        
    def _communication_loop(self):
        """Main communication loop with the robot"""
        if not UNITREE_SDK_AVAILABLE:
            return
            
        # Send command
        self.cmd.crc = CRC.Crc32(self.cmd.head, len(self.cmd.head) + len(self.cmd.level_flag) + 
                               len(self.cmd.gpio) + len(self.cmd.motor_cmd))
        self.cmd_publisher.Write(self.cmd)
        
    def _get_robot_state(self):
        """Get current robot state"""
        if not UNITREE_SDK_AVAILABLE:
            return None
            
        return self.state_subscriber.Read()
        
    def _update_observations_from_robot(self):
        """Update observations from real robot state"""
        if not UNITREE_SDK_AVAILABLE:
            # Use dummy values for testing without real robot
            self._update_dummy_observations()
            return
            
        state = self._get_robot_state()
        if state is None:
            return
            
        # Update base angular velocity
        self.base_ang_vel[0, 0] = state.imu.gyroscope[0]
        self.base_ang_vel[0, 1] = state.imu.gyroscope[1] 
        self.base_ang_vel[0, 2] = state.imu.gyroscope[2]
        
        # Update projected gravity (from IMU accelerometer)
        acc = np.array([state.imu.accelerometer[0], 
                       state.imu.accelerometer[1], 
                       state.imu.accelerometer[2]])
        acc_norm = np.linalg.norm(acc)
        if acc_norm > 0:
            self.projected_gravity[0] = torch.tensor(acc / acc_norm, device=self.device)
        
        # Update joint positions and velocities
        for i in range(12):
            self.dof_pos[0, i] = state.motor_state[i].q
            self.dof_vel[0, i] = state.motor_state[i].dq
            
    def _update_dummy_observations(self):
        """Update observations with dummy values when robot is not available"""
        # Simulate some basic robot state
        self.base_ang_vel[0] = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        self.projected_gravity[0] = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        self.dof_pos[0] = self.default_dof_pos[0]
        self.dof_vel[0] = torch.zeros(12, device=self.device)
        
    def get_observations(self):
        """Get observations for the policy (same structure as simulation)"""
        # Update robot state
        self._update_observations_from_robot()
        
        # Calculate phase based on episode progress
        phase = torch.pi * self.episode_length_buf[:, None] / self.max_episode_length
        
        # Construct observation vector (same as simulation)
        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
            self.projected_gravity,  # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],  # 12
            self.actions,  # 12
            self.last_actions,  # 12
            torch.sin(phase),  # 1
            torch.cos(phase),  # 1
            torch.sin(phase / 2),  # 1
            torch.cos(phase / 2),  # 1
            torch.sin(phase / 4),  # 1
            torch.cos(phase / 4),  # 1
        ], axis=-1)
        
        return self.obs_buf
        
    def step(self, actions):
        """Execute actions on the real robot"""
        # Store previous actions
        self.last_actions = self.actions.clone()
        self.actions = actions.clone()
        
        # Scale actions
        scaled_actions = actions * self.action_scale
        
        # Send commands to robot
        self._send_actions_to_robot(scaled_actions)
        
        # Update episode length
        self.episode_length_buf += 1
        
        # Check for episode termination
        done = self.episode_length_buf >= self.max_episode_length
        self.reset_buf = done
        
        # Get new observations
        obs = self.get_observations()
        
        # Return dummy reward and info
        reward = torch.zeros(1, device=self.device)
        info = {}
        
        return obs, reward, self.reset_buf, info
        
    def _send_actions_to_robot(self, actions):
        """Send joint position commands to the robot"""
        if not UNITREE_SDK_AVAILABLE:
            return
            
        # Convert actions to joint positions (actions are relative to default positions)
        target_positions = self.default_positions + actions[0].cpu().numpy()
        
        # Update motor commands
        for i in range(12):
            self.cmd.motor_cmd[i].q = float(target_positions[i])
            self.cmd.motor_cmd[i].kp = self.kp
            self.cmd.motor_cmd[i].kd = self.kd
            self.cmd.motor_cmd[i].tau = 0.0
            
    def reset(self):
        """Reset the environment"""
        # Reset episode tracking
        self.episode_length_buf.zero_()
        self.reset_buf.zero_()
        
        # Reset actions
        self.actions.zero_()
        self.last_actions.zero_()
        
        # Move robot to default position
        if UNITREE_SDK_AVAILABLE:
            for i in range(12):
                self.cmd.motor_cmd[i].q = self.default_positions[i]
                self.cmd.motor_cmd[i].kp = self.kp
                self.cmd.motor_cmd[i].kd = self.kd
                self.cmd.motor_cmd[i].tau = 0.0
        
        # Wait for robot to reach default position
        time.sleep(1.0)
        
        # Get initial observations
        obs = self.get_observations()
        info = {}
        
        return obs, info
        
    def close(self):
        """Close the environment and stop communication"""
        if hasattr(self, 'communication_thread'):
            self.communication_thread.stop()