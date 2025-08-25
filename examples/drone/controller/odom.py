import time
import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

# NOTE :IMU is not an odomtry, only has anglear_velocity and linear_accerleration
class Odom:
    def __init__(self, num_envs, device = torch.device("cuda")):
        self.device = device
        self.drone = None
        self.num_envs = num_envs
        self.has_nan = torch.zeros((self.num_envs,), device=self.device, dtype=bool)

        # body data
        self.body_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.body_linear_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.body_linear_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)        
        self.body_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float)
        self.body_quat = identity_quat.unsqueeze(0).repeat(self.num_envs, 1)
        self.body_quat_inv = identity_quat.unsqueeze(0).repeat(self.num_envs, 1)

        self.last_body_linear_vel = torch.zeros_like(self.body_linear_vel)    # used to cal acc

        # global data
        self.world_euler = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.world_linear_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.world_linear_acc = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.world_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.world_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

        self.last_world_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_world_linear_vel = torch.zeros_like(self.body_linear_vel)
        self.last_time = time.perf_counter()

    def set_drone(self, drone):
        self.drone = drone

    def cal_cur_quat(self):
        self.body_quat[:] = self.drone.get_quat()
        self.has_nan[:] = torch.isnan(self.body_quat).any(dim=1)
        if (torch.any(self.has_nan)):
            print("get_quat NaN env_idx:", torch.nonzero(self.has_nan).squeeze())
            self.body_quat_inv[self.has_nan] = inv_quat(self.body_quat[self.has_nan])
            self.reset(self.has_nan.nonzero(as_tuple=False).flatten())
        else:
            self.body_quat_inv = inv_quat(self.body_quat)

    def gyro_update(self):  
        self.world_ang_vel[:] = self.drone.get_ang()     # (roll, pitch, yaw)
        self.body_ang_vel[:] = self.world_to_body_vector(self.world_ang_vel)

    def vel_update(self):
        self.last_world_linear_vel[:] = self.world_linear_vel
        self.last_body_linear_vel[:] = self.body_linear_vel
        self.world_linear_vel[:] = self.drone.get_vel()
        self.body_linear_vel[:] = self.world_to_body_vector(self.world_linear_vel)

    def acc_update(self, dT):
        self.body_linear_acc[:] = (self.body_linear_vel - self.last_body_linear_vel) / dT
        self.world_linear_acc[:] = (self.world_linear_vel - self.last_world_linear_vel) / dT

    def att_update(self):
        self.cal_cur_quat()  
        self.body_euler[:] = quat_to_xyz(self.body_quat, rpy=True)

    def pos_update(self):
        self.last_world_pos[:] = self.world_pos
        self.world_pos[:] = self.drone.get_pos()

    def odom_update(self):
        self.att_update()
        self.gyro_update()
        self.vel_update()
        self.acc_update(time.perf_counter() - self.last_time)
        self.pos_update()
        self.last_time = time.perf_counter()
        
    def reset(self, rand_quat, envs_idx):
        # use reset when the status mutation occurs
        if envs_idx is None:
            reset_range = torch.arange(self.num_envs, device=self.device)
        else:
            reset_range = envs_idx
        # Reset body data to zero
        self.body_euler.index_fill_(0, reset_range, 0.0)
        self.body_linear_vel.index_fill_(0, reset_range, 0.0)
        self.body_linear_acc.index_fill_(0, reset_range, 0.0)
        self.body_ang_vel.index_fill_(0, reset_range, 0.0)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device, dtype=gs.tc_float)
        self.body_quat[reset_range] = rand_quat
        self.body_quat_inv[reset_range] = inv_quat(rand_quat)
        self.body_euler[:] = quat_to_xyz(self.body_quat, rpy=True)
        
        self.last_body_linear_vel.index_fill_(0, reset_range, 0.0)

        # Reset global data to zero
        self.world_euler.index_fill_(0, reset_range, 0.0)
        self.world_linear_vel.index_fill_(0, reset_range, 0.0)
        self.world_linear_acc.index_fill_(0, reset_range, 0.0)
        self.world_pos.index_fill_(0, reset_range, 0.0)
        self.world_ang_vel.index_fill_(0, reset_range, 0.0)

        self.last_world_pos.index_fill_(0, reset_range, 0.0)
        self.last_world_linear_vel.index_fill_(0, reset_range, 0.0)

        # Reset the time
        self.last_time = time.perf_counter()

        # self.odom_update()
        self.last_world_pos[reset_range] = self.world_pos[reset_range]
        self.last_world_linear_vel[reset_range] = self.world_linear_vel[reset_range]
        self.last_body_linear_vel[reset_range] = self.body_linear_vel[reset_range]


    # utils
    def world_to_body_vector(self, input_tensor):
        """
        Convert body frame vector tensor to world frame.
        :param:
            input_tensor: vectors like vel, acc ...(N, 3) or quat(N, 3), where N is the number of environments.
        """
        if input_tensor.shape[-1] == 3:
            return transform_by_quat(input_tensor, self.body_quat_inv)
        elif input_tensor.shape[-1] == 4:
            return transform_quat_by_quat(input_tensor, self.body_quat_inv)
        else:
            raise ValueError("Input tensor must have shape (N, 3) or (N, 4).")
        
    def body_to_world_vector(self, input_tensor):
        """
        Convert world frame vector tensor to body frame.
        :param:
            input_tensor: vectors like vel, acc ...(N, 3) or quat(N, 4), where N is the number of environments.
        """
        if input_tensor.shape[-1] == 3:
            return transform_by_quat(input_tensor, self.body_quat)
        elif input_tensor.shape[-1] == 4:
            return transform_quat_by_quat(input_tensor, self.body_quat)
        else:
            raise ValueError("Input tensor must have shape (N, 3) or (N, 4).")


        
def ve2vb(input_vec: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    assert input_vec.ndim == 2 and input_vec.shape[1] == 3, "input_vec must be (N, 3)"
    assert yaw.ndim == 1 or (yaw.ndim == 2 and yaw.shape[1] == 1), "yaw must be (N,) or (N,1)"

    yaw = -yaw.view(-1)
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    R = torch.zeros((input_vec.shape[0], 3, 3), dtype=input_vec.dtype, device=input_vec.device)
    R[:, 0, 0] = cos_yaw
    R[:, 0, 1] = sin_yaw
    R[:, 1, 0] = -sin_yaw
    R[:, 1, 1] = cos_yaw
    R[:, 2, 2] = 1.0

    input_vec = input_vec.unsqueeze(-1)
    output = torch.bmm(R, input_vec)

    return output.squeeze(-1) 
   
