import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import yaml
import torch
from genesis_env import Genesis_env
import genesis as gs


def gs_rand_float(lower, upper, device="cuda"):
    shape = lower.shape  # scalar
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def main():
    # logging_level="warning"
    gs.init(logging_level="warning")
    num_envs = 1
    command_buf = torch.zeros((num_envs, 4), device="cuda", dtype=gs.tc_float)
    def update_commands(cur_pos, envs_idx=None):
        if envs_idx is None:
            idx = torch.arange(num_envs, "cuda")
        else:
            idx = envs_idx
        command_buf[idx, 0] = gs_rand_float(cur_pos[idx, 0]-0.3, cur_pos[idx, 0]+0.3)
        command_buf[idx, 1] = gs_rand_float(cur_pos[idx, 1]-0.3, cur_pos[idx, 1]+0.3)
        command_buf[idx, 2] = gs_rand_float(torch.clamp(cur_pos[idx, 2]-0.2, min=0.3, max=2.0), cur_pos[idx, 2]+0.2)

    
    def at_target(cur_pos):
        cur_pos_error = cur_pos - command_buf[:, :3]
        at_target = ((torch.norm(cur_pos_error, dim=1) < 0.1).nonzero(as_tuple=False).flatten())
        return at_target

    with open("examples/drone/controller/config/pos_ctrl_eval/genesis_env.yaml", "r") as file:
        env_config = yaml.load(file, Loader=yaml.FullLoader)
    with open("examples/drone/controller/config/pos_ctrl_eval/flight.yaml", "r") as file:
        flight_config = yaml.load(file, Loader=yaml.FullLoader)


    genesis_env = Genesis_env(
        env_config = env_config, 
        flight_config = flight_config,
    )

    while True:
        cur_pos = genesis_env.drone.odom.world_pos
        update_commands(cur_pos, at_target(cur_pos))
        genesis_env.target.set_pos(command_buf[:, :3], zero_velocity=True, envs_idx=list(range(num_envs)))
        genesis_env.step(command_buf)

if __name__ == "__main__" :
    main()


    