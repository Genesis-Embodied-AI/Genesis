#!/usr/bin/env python3
"""
Test if gravity is working and skeleton can fall
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv


def test_gravity():
    """Test basic gravity and physics"""
    print("Testing gravity and physics...")
    
    # Initialize Genesis
    gs.init()
    
    # Create environment
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=5.0,
        dt=0.02,
        use_box_feet=True,
        disable_arms=False,
        show_viewer=True
    )
    
    print(f"Environment created - DOFs: {env.num_dofs}, Actions: {env.num_actions}")
    
    # Reset to get initial state
    obs, info = env.reset()
    print(f"Initial root position: {env.root_pos[0]}")
    
    # Step with ZERO actions (should fall due to gravity)
    print("Stepping with zero actions (gravity test)...")
    for step in range(200):
        zero_actions = torch.zeros(env.num_envs, env.num_actions)
        obs, rewards, dones, info = env.step(zero_actions)
        
        if step % 50 == 0:
            pos = env.root_pos[0]
            vel = env.root_lin_vel[0]
            print(f"Step {step}: pos={pos}, vel={vel}")
            
        # Check if falling
        if env.root_pos[0, 2] < 0.5:  # Below 0.5m height
            print(f"✓ Skeleton fell at step {step}!")
            break
    else:
        print("✗ Skeleton did not fall - gravity issue?")
        
    # Test with small torques
    print("\nTesting with small torques...")
    for step in range(100):
        small_torques = torch.ones(env.num_envs, env.num_actions) * 0.5
        obs, rewards, dones, info = env.step(small_torques)
        
        if step % 25 == 0:
            pos = env.root_pos[0]
            print(f"Step {step}: pos={pos}")

if __name__ == "__main__":
    test_gravity()