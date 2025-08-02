#!/usr/bin/env python3
"""
Debug script to isolate joint mapping issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import genesis as gs
import numpy as np
from environments.skeleton_humanoid import SkeletonHumanoidEnv

def debug_specific_joints():
    """Test specific problematic joint mappings"""
    print("=== Joint Mapping Debug Test ===\n")
    
    gs.init()
    
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=10.0,
        dt=0.02,
        use_box_feet=True,
        show_viewer=True
    )
    
    # Reset and stabilize
    obs, info = env.reset()
    for _ in range(50):
        env.step(torch.zeros(1, env.num_actions))
    
    # Test specific joint mapping
    test_actions = [
        "mot_shoulder_flex_r",
        "mot_ankle_angle_r", 
        "mot_shoulder_add_r",
        "mot_ankle_angle_l"
    ]
    
    for action_name in test_actions:
        if action_name not in env.action_to_joint_idx:
            continue
            
        print(f"\n=== Testing {action_name} ===")
        
        action_idx = env.action_spec.index(action_name)
        joint_idx = env.action_to_joint_idx[action_name]
        joint_name = env.dof_names[joint_idx]
        
        print(f"Action index: {action_idx}")
        print(f"Maps to joint: {joint_name} (DOF index: {joint_idx})")
        
        # Record initial positions of ALL joints
        initial_positions = env.dof_pos[0].clone()
        
        # Apply strong torque to this action only
        actions = torch.zeros(1, env.num_actions)
        actions[0, action_idx] = 50.0  # Strong positive torque
        
        print(f"Applying torque +50.0 to action {action_idx} ({action_name})")
        
        # Apply for multiple steps
        for step in range(30):
            env.step(actions)
        
        # Check which joints actually moved
        final_positions = env.dof_pos[0]
        position_changes = torch.abs(final_positions - initial_positions)
        
        print(f"Joint position changes (threshold > 0.01 rad):")
        for i, change in enumerate(position_changes):
            if change > 0.01:  # Only show significant changes
                joint_name_actual = env.dof_names[i]
                print(f"  DOF {i:2d} ({joint_name_actual:20s}): {change:.3f} rad ({np.degrees(change.item()):.1f}°)")
        
        # Check if the expected joint moved the most
        max_change_idx = torch.argmax(position_changes).item()
        max_change_joint = env.dof_names[max_change_idx]
        
        if max_change_idx == joint_idx:
            print(f"✅ CORRECT: {joint_name} moved the most ({position_changes[joint_idx]:.3f} rad)")
        else:
            print(f"❌ INCORRECT: Expected {joint_name} to move most, but {max_change_joint} moved most ({position_changes[max_change_idx]:.3f} rad)")
        
        # Reset for next test
        env.reset()
        for _ in range(50):
            env.step(torch.zeros(1, env.num_actions))

if __name__ == "__main__":
    debug_specific_joints()