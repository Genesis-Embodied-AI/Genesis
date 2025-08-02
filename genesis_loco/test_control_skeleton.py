import genesis as gs
import torch
import numpy as np
from environments.skeleton_humanoid import SkeletonHumanoidEnv


def test_genesis_dof_control():
    """
    Test control of humanoid joints using Genesis built in DOF force control for robotic models
    """

    gs.init()

    env = SkeletonHumanoidEnv(num_envs=1
                              episode_length_s=20.0,
                              dt=0.02,
                              use_box_feet=True,
                              disable_arms=False,
                              show_viewer=True)

    

    return 
            









if name == "__main__":
    print("Starting Genesis motor control testing for Skeleton Humanoid Environment...\n") 

