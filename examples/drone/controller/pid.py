

import torch
import genesis as gs

import math
from genesis.utils.geom import quat_to_R
from flight.odom import ve2vb

class PIDcontroller:
    def __init__(
            self, 
            num_envs, 
            rc_command, 
            odom, 
            config, 
            controller = "angle",
            use_rc = False, 
            device = torch.device("cuda")):

        self.rc_command = rc_command
        self.device = device
        self.num_envs = num_envs
        self.odom = odom
        self.use_rc = use_rc
        self.thrust_compensate = config.get("thrust_compensate", 0.5)  
        self.controller = controller
            
        if self.controller == "position":
            self.controller = self.position_controller
        elif self.controller == "rate":
            self.controller = self.rate_controller
        elif self.controller == "angle":
            self.controller = self.angle_controller
        else:
            raise ValueError(f"Invalid controller type: {self.controller}. Choose from 'angle', 'rate', or 'position'.")
        
        # Shape: (n, 3)
        ang_cfg = config.get("ang", {})
        rat_cfg = config.get("rat", {})
        pos_cfg = config.get("pos", {})
        def get3(cfg, k1, k2, k3):
            return torch.tensor([
                cfg.get(k1, 0.0),
                cfg.get(k2, 0.0),
                cfg.get(k3, 0.0)
            ], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        # Angular controller (angle controller PID)
        self.kp_a = get3(ang_cfg, "kp_r", "kp_p", "kp_y")
        self.ki_a = get3(ang_cfg, "ki_r", "ki_p", "ki_y")
        self.kd_a = get3(ang_cfg, "kd_r", "kd_p", "kd_y")
        self.kf_a = get3(ang_cfg, "kf_r", "kf_p", "kf_y")
        self.P_term_a = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.I_term_a = torch.zeros_like(self.P_term_a)
        self.D_term_a = torch.zeros_like(self.P_term_a)
        self.F_term_a = torch.zeros_like(self.P_term_a)

        # Angular Rate controller (angular rate PID)
        self.kp_r = get3(rat_cfg, "kp_r", "kp_p", "kp_y")
        self.ki_r = get3(rat_cfg, "ki_r", "ki_p", "ki_y")
        self.kd_r = get3(rat_cfg, "kd_r", "kd_p", "kd_y")
        self.kf_r = get3(rat_cfg, "kf_r", "kf_p", "kf_y")
        self.P_term_r = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.I_term_r = torch.zeros_like(self.P_term_r)
        self.D_term_r = torch.zeros_like(self.P_term_r)
        self.F_term_r = torch.zeros_like(self.P_term_r)

        # Position controller (xyz and throttle)
        self.kp_p = get3(pos_cfg, "kp_x", "kp_y", "kp_t")
        self.ki_p = get3(pos_cfg, "ki_x", "ki_y", "ki_t")
        self.kd_p = get3(pos_cfg, "kd_x", "kd_y", "kd_t")
        self.kf_p = get3(pos_cfg, "kf_x", "kf_y", "kf_t")
        self.P_term_p = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.I_term_p = torch.zeros_like(self.P_term_p)
        self.D_term_p = torch.zeros_like(self.P_term_p)
        self.F_term_p = torch.zeros_like(self.P_term_p)                
        
        self.pid_freq = config.get("pid_exec_freq", 60)     # no use
        self.base_rpm = config.get("base_rpm", 14468.429183500699)
        self.dT = 1 / self.pid_freq                         # no use
        self.tpa_factor = 1
        self.tpa_rate = 0
        self.throttle_command = torch.zeros((self.num_envs, ), device=self.device, dtype=gs.tc_float)

        self.last_body_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.angle_rate_error = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.angle_error = torch.zeros_like(self.angle_rate_error)

        self.body_set_point = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.pid_output = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.cur_setpoint_error = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_setpoint_error = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.drone = None

        self.cnt = 0

    def set_drone(self, drone):
        self.drone = drone


    def mixer(self, action=None) -> torch.Tensor:

        throttle_rc = torch.clamp((self.rc_command[3] + self.throttle_command) * 3, 0.0, 3.0) * self.base_rpm
        if action is None:
            throttle = throttle_rc
        else:
            throttle_action = torch.clamp(action[:, -1] * 3 + self.thrust_compensate, min=0.0, max=3.0) * self.base_rpm
            throttle = throttle_rc + throttle_action

        # self.pid_output[:] = torch.clip(self.pid_output[:], -3.0, 3.0)
        motor_outputs = torch.stack([
           throttle - self.pid_output[:, 0] - self.pid_output[:, 1] - self.pid_output[:, 2],  # M1
           throttle - self.pid_output[:, 0] + self.pid_output[:, 1] + self.pid_output[:, 2],  # M2
           throttle + self.pid_output[:, 0] + self.pid_output[:, 1] - self.pid_output[:, 2],  # M3
           throttle + self.pid_output[:, 0] - self.pid_output[:, 1] + self.pid_output[:, 2],  # M4
        ], dim = 1)

        return torch.clamp(motor_outputs, min=1, max=self.base_rpm * 3.5)  # size: tensor(num_envs, 4)

    def pid_update_TpaFactor(self):
        if (self.rc_command[3] > 0.35):       # 0.35 is the tpa_breakpoint, the same as Betaflight, 
            if (self.rc_command[3] < 1.0): 
                self.tpa_rate *= (self.rc_command[3] - 0.35) / (1.0 - 0.35);            
            else:
                self.tpa_rate = 0.0
            self.tpa_factor = 1.0 - self.tpa_rate

    def step(self, action=None):
        # controller test
        # self.cnt += 1
        # if self.cnt % 40 == 0:     
        #     quat = random_quaternion(self.num_envs)
        #     self.drone.set_quat(quat)
        #     self.cnt = 0
        self.odom.odom_update()
        if self.use_rc is True:
            self.pid_update_TpaFactor() 
            if(self.rc_command[5] == 0):        # not arm
                self.drone.set_propellels_rpm(torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float))
                return
            if self.rc_command[4] == 0:         # angle mode
                self.angle_controller(action)
            elif self.rc_command[4] == 1:       # angle rate mode
                self.rate_controller(action)
            else:                               # undifined
                print("undifined mode, do nothing!!")
                return
        else:
            self.controller(action)
            
        self.drone.set_propellels_rpm(self.mixer(action))

    def rate_controller(self, action=None): 
        """
        Anglular rate controller, sequence is (roll, pitch, yaw), use previous-D-term PID controller
        :param: 
            action: torch.Size([num_envs, 4]), like [[roll, pitch, yaw, thrust]] if num_envs = 1
        """
        if action is None:
            self.body_set_point[:] = self.rc_command[:3] * 15   # max 15 rad/s
        else:
            self.body_set_point[:] = action[:, :3] * 1

        self.last_setpoint_error[:] = self.cur_setpoint_error
        self.cur_setpoint_error[:] = self.body_set_point - self.odom.body_ang_vel
        self.P_term_r[:] = (self.cur_setpoint_error * self.kp_r) * self.tpa_factor
        self.I_term_r[:] = torch.clamp(self.I_term_r + self.cur_setpoint_error * self.ki_r, -0.5, 0.5)
        self.D_term_r[:] = (self.last_body_ang_vel - self.odom.body_ang_vel) * self.kd_r * self.tpa_factor    

        self.pid_output[:] = (self.P_term_r + self.I_term_r + self.D_term_r)
        self.last_body_ang_vel[:] = self.odom.body_ang_vel

    def angle_controller(self, action=None):  
        """
        Angle controller, sequence is (roll, pitch, yaw), use previous-D-term PID controller
        :param: 
            action: torch.Size([num_envs, 4]), like [[roll, pitch, yaw, thrust]] if num_envs = 1
        """
        yaw_mask = torch.tensor([1,1,0], device=self.device)
        if action is None:  # in RC mode
            self.body_set_point[:] = -self.odom.body_euler * yaw_mask + self.rc_command[:3]  
        else:               # in RL mode
            self.body_set_point[:] = -self.odom.body_euler * yaw_mask + action[:, :3]  # action is in rad, like [[roll, pitch, yaw, thrust]] if num_envs = 1

        self.last_setpoint_error[:] = self.cur_setpoint_error
        self.cur_setpoint_error[:] = (self.body_set_point * 15 - self.odom.body_ang_vel)
        self.P_term_a[:] = (self.cur_setpoint_error[:] * self.kp_a) * self.tpa_factor
        self.I_term_a[:] = torch.clamp(self.I_term_a + self.cur_setpoint_error[:] * self.ki_a, -0.5, 0.5)
        self.D_term_a[:] = torch.clamp((self.last_body_ang_vel - self.odom.body_ang_vel) * self.kd_a * self.tpa_factor, -0.5, 0.5)    
        
        self.pid_output[:] = (self.P_term_a + self.I_term_a + self.D_term_a)
        self.last_body_ang_vel[:] = self.odom.body_ang_vel

    def position_controller(self, action, head_free=False):
        """
        Position controller, sequence is (x, y, z), use previous-D-term PID controller
        Note that the action is in world frame, (x, y, z) -> (pitch, -roll, throttle)
        :param: 
            action: torch.Size([num_envs, 4]), like [[x, y, z, 0]] if num_envs = 1, 0 used to adapt thrust
            head_free: Keep yaw follow the target, namely keep the drone facing the target direction.
        """
        cur_pos_error = (action[:, :3] - self.odom.world_pos)
        self.cur_setpoint_error[:] = cur_pos_error*5 - self.odom.world_linear_vel
        self.cur_setpoint_error[:, 1] *= -1                       # since roll increase, y decrease
        self.cur_setpoint_error = self.cur_setpoint_error[:, [1, 0, 2]]     # change to (roll, pitch, throttle)

        if head_free:
            # cur_pos_error = ve2vb(cur_pos_error, self.odom.body_euler[:, 2])
            print("head_free not implemented yet")

        self.P_term_p[:] = (self.cur_setpoint_error[:] * self.kp_p)
        self.I_term_p[:] = torch.clamp(self.I_term_p + self.cur_setpoint_error[:] * self.ki_p, -0.5, 0.5)
        self.D_term_p[:] = torch.clamp((self.odom.last_world_linear_vel - self.odom.world_linear_vel) * self.kd_p, -0.5, 0.5)  

        sum = self.P_term_p + self.I_term_p + self.D_term_p 
        self.throttle_command = torch.clamp(sum[:,-1], min=0.0, max=0.5)

        # sum[:,-1] = torch.atan2(action[:, 1], action[:, 0])
        sum[:, -1] = 0
        self.angle_controller(torch.clamp(sum, -0.5, 0.5))

    def reset(self, env_idx=None):
        if env_idx is None:
            reset_range = torch.arange(self.num_envs, device=self.device)
        else:
            reset_range = env_idx
        # Reset the PID terms (P, I, D, F)
        self.P_term_a.index_fill_(0, reset_range, 0.0)
        self.I_term_a.index_fill_(0, reset_range, 0.0)
        self.D_term_a.index_fill_(0, reset_range, 0.0)
        self.F_term_a.index_fill_(0, reset_range, 0.0)

        self.P_term_r.index_fill_(0, reset_range, 0.0)
        self.I_term_r.index_fill_(0, reset_range, 0.0)
        self.D_term_r.index_fill_(0, reset_range, 0.0)
        self.F_term_r.index_fill_(0, reset_range, 0.0)

        self.P_term_p.index_fill_(0, reset_range, 0.0)
        self.I_term_p.index_fill_(0, reset_range, 0.0)
        self.D_term_p.index_fill_(0, reset_range, 0.0)
        self.F_term_p.index_fill_(0, reset_range, 0.0)

        # Reset the angle, position, and velocity errors
        self.angle_rate_error.index_fill_(0, reset_range, 0.0)
        self.angle_error.index_fill_(0, reset_range, 0.0)

        # Reset the body set points and pid output
        self.body_set_point.index_fill_(0, reset_range, 0.0)
        self.pid_output.index_fill_(0, reset_range, 0.0)
        
        # Reset the last angular velocity
        self.last_body_ang_vel.index_fill_(0, reset_range, 0.0)
        self.last_setpoint_error.index_fill_(0, reset_range, 0.0)
        self.cur_setpoint_error.index_fill_(0, reset_range, 0.0)
        # Reset the TPA factor and rate
        self.tpa_factor = 1
        self.tpa_rate = 0
        # Reset the RC command values if necessary
        if self.use_rc:
            self.rc_command = 0



def random_quaternion(num_envs=1, device="cuda"):
    max_rad = math.radians(180)
    roll  = (torch.rand(num_envs, 1, device=device) * 2 - 1) * max_rad
    pitch = (torch.rand(num_envs, 1, device=device) * 2 - 1) * max_rad
    yaw   = (torch.rand(num_envs, 1, device=device) * 2 - 1) * max_rad / 10

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    quat = torch.cat([w, x, y, z], dim=1)
    quat = quat / quat.norm(dim=1, keepdim=True)
    return quat

