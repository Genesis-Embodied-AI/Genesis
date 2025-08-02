#!/usr/bin/env python3
"""
Simple test of single joint to verify correct mapping
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import genesis as gs
import numpy as np
import time
from environments.skeleton_humanoid import SkeletonHumanoidEnv


def test_joint_range():
    """Test joint through its full range of motion"""
    print("=== Joint Range of Motion Test ===\n")
    
    # Initialize Genesis
    gs.init()
    
    # Create environment
    try:
        env = SkeletonHumanoidEnv(
            num_envs=1,
            episode_length_s=20.0,
            dt=0.02,
            use_box_feet=True,
            disable_arms=False,
            show_viewer=True
        )
        print(f"✓ Environment created successfully")
        print(f"  - DOFs: {env.num_dofs}")
        print(f"  - Actions: {env.num_actions}")
        print(f"  - Action spec: {env.action_spec}")
        print()
        
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return False
    
    # Reset environment and let it stabilize
    obs, info = env.reset()
    print("Environment reset. Stabilizing skeleton...")
    
    # Let skeleton stabilize for a moment with small holding torques
    stabilize_actions = torch.zeros(env.num_envs, env.num_actions)
    for step in range(50):  # 1 second stabilization
        obs, rewards, dones, info = env.step(stabilize_actions)
    
    print("Skeleton stabilized. Starting joint tests...\n")
    
    # Test joint action

    action_idx = 0
    action_name = "mot_lumbar_ext"

    target_joint_idx = env.action_to_joint_idx[action_name]
    joint_name = env.robot.joints[target_joint_idx].name

    print(f"Testing Action {action_name}")

    print(f"  Joint: {joint_name} (index: {target_joint_idx})")

    actions = torch.zeros(env.num_envs, env.num_actions)
    actions[0, action_idx] = 500

    for step in range(100):
        obs, rewards, dones, info = env.step(actions)
        current_pos = env.dof_pos[0, target_joint_idx].item()

    if step % 15 == 0:  # Print every 0.3 seconds
        print(f"      Step {step:2d}: Joint pos = {current_pos:+.3f} rad ({np.degrees(current_pos):+.1f}°)")


    
    print("=== Joint Range Test Complete ===")
    
    # Summary
    print(f"\nSummary:")
    print(f"- Total DOFs: {env.num_dofs}")
    print(f"- Actuated joints: {env.num_actions}")
    print(f"- Tested joints: {len(env.action_spec)}")
    
    # List joint ranges
    # print(f"\nJoint Limits Summary:")
    # for action_name in env.action_spec:
    #     joint_idx = env.action_to_joint_idx[action_name]
    #     joint_name = env.dof_names[joint_idx]
    #     xml_limit = xml_limits.get(joint_name, (-3.14, 3.14))
    #     print(f"  {joint_name:20s}: {xml_limit[0]:+.3f} to {xml_limit[1]:+.3f} rad "
    #           f"({np.degrees(xml_limit[0]):+.1f}° to {np.degrees(xml_limit[1]):+.1f}°)")
    
    return True


if __name__ == "__main__":
    print("Starting comprehensive joint testing...\n")
    
    # Test joint ranges
    success = test_joint_range()
    
    print("\nTesting complete!")