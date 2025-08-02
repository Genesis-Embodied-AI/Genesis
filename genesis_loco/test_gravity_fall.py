#!/usr/bin/env python3
"""
Test script to verify skeleton falls under gravity with pure torque control
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import genesis as gs
from environments.skeleton_humanoid import SkeletonHumanoidEnv

def test_gravity_fall():
    """Test that skeleton falls under gravity with zero torques"""
    print("=== Testing Gravity Fall with Pure Torque Control ===\n")
    
    gs.init()
    
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=5.0,
        dt=0.02,
        use_box_feet=True,
        show_viewer=True
    )
    
    print("Environment created. Testing gravity fall...")
    
    # Reset environment
    obs, info = env.reset()
    initial_height = env.root_pos[0, 2].item()
    initial_quat = env.root_quat[0].clone()
    
    print(f"Initial height: {initial_height:.3f}m")
    print(f"Initial orientation: {initial_quat}")
    
    # Apply ZERO torques for several seconds - skeleton should fall
    zero_actions = torch.zeros(env.num_envs, env.num_actions)
    
    for step in range(100):  # 2 seconds
        obs, rewards, dones, info = env.step(zero_actions)
        
        current_height = env.root_pos[0, 2].item()
        current_quat = env.root_quat[0]
        
        if step % 25 == 0:  # Print every 0.5 seconds
            height_change = current_height - initial_height
            print(f"Step {step:3d}: Height = {current_height:.3f}m, Change = {height_change:+.3f}m")
            print(f"         Quat = [{current_quat[0]:.3f}, {current_quat[1]:.3f}, {current_quat[2]:.3f}, {current_quat[3]:.3f}]")
    
    final_height = env.root_pos[0, 2].item()
    height_change = final_height - initial_height
    
    print(f"\n=== Results ===")
    print(f"Initial height: {initial_height:.3f}m")
    print(f"Final height: {final_height:.3f}m")
    print(f"Total height change: {height_change:+.3f}m")
    
    if height_change < -0.1:
        print("✅ SUCCESS: Skeleton fell under gravity (torque control working)")
        return True
    else:
        print("❌ FAILURE: Skeleton didn't fall (torque control not working)")
        return False

if __name__ == "__main__":
    success = test_gravity_fall()
    if success:
        print("\n🎯 Pure torque control is working correctly!")
    else:
        print("\n⚠️  Pure torque control needs debugging.")