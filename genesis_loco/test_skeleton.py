"""
Test script for SkeletonHumanoidEnv

Simple test to verify the environment can be created and stepped.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv


def test_skeleton_environment():
    """Test basic skeleton environment functionality"""
    print("Testing SkeletonHumanoidEnv...")
    
    # Initialize Genesis
    gs.init()
    
    # Create environment
    try:
        env = SkeletonHumanoidEnv(
            num_envs=1,
            episode_length_s=5.0,
            dt=0.02,
            use_box_feet=True,
            disable_arms=False,
            show_viewer=True
        )
        print(f"✓ Environment created successfully")
        print(f"  - Number of environments: {env.num_envs}")
        print(f"  - Action space: {env.num_actions}")
        print(f"  - Observation space: {env.num_observations}")
        print(f"  - DOFs: {env.num_dofs}")
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False
    
    # Test reset
    try:
        obs, info = env.reset()
        print(f"✓ Reset successful")
        print(f"  - Observation shape: {obs.shape}")
        
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        return False
    
    cam = env.scene.add_camera(
        res    = (1280, 960),
        pos=(3.0, 0.0, 2.0),
        lookat=(0.0, 0.0, 1.0),
        fov=40,
        GUI    = True
    )

    # Test stepping
    try:
        for step in range(100):

            cam.start_recording()

            # Random actions
            actions = torch.randn(env.num_envs, env.num_actions) * 100
            
            obs, rewards, dones, info = env.step(actions)
            
            if step == 0:
                print(f"✓ Step successful")
                print(f"  - Observation shape: {obs.shape}")
                print(f"  - Reward shape: {rewards.shape}")
                print(f"  - Done shape: {dones.shape}")
            
            # Check for any obvious issues
            if torch.any(torch.isnan(obs)):
                print(f"✗ NaN detected in observations at step {step}")
                return False
                
            if torch.any(torch.isnan(rewards)):
                print(f"✗ NaN detected in rewards at step {step}")
                return False

            cam.render()
        
        print(f"✓ Stepping test completed (10 steps)")

        cam.stop_recording(save_to_filename='test_skeleton.mp4', fps=60)
        
    except Exception as e:
        print(f"✗ Stepping failed: {e}")
        return False
    
    print("✓ All tests passed!")
    return True

    env.scene.reset_grad()


if __name__ == "__main__":
    success = test_skeleton_environment()
    if not success:
        exit(1)