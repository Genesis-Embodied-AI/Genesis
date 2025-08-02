#!/usr/bin/env python3
"""
Joint Range of Motion Test Script for Skeleton Model

Tests each action joint sequentially through its full range of motion
and displays joint names and limits.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import genesis as gs
import numpy as np
import time
from environments.skeleton_humanoid import SkeletonHumanoidEnv


def get_joint_limits_from_xml():
    """Extract joint limits from XML file"""
    xml_file = "/home/ez/Documents/Genesis/genesis_loco/skeleton/skeleton_restructured_panda_format.xml"
    joint_limits = {}
    
    try:
        with open(xml_file, 'r') as f:
            content = f.read()
            
        # Parse joint limits from XML
        import re
        joint_pattern = r'<joint name="([^"]+)"[^>]*range="([^"]+)"[^>]*>'
        matches = re.findall(joint_pattern, content)
        
        for joint_name, range_str in matches:
            limits = [float(x) for x in range_str.split()]
            if len(limits) == 2:
                joint_limits[joint_name] = (limits[0], limits[1])
    except Exception as e:
        print(f"Warning: Could not parse XML joint limits: {e}")
    
    return joint_limits


def test_joint_ranges():
    """Test each joint through its full range of motion"""
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
    
    # Get joint limits from XML
    xml_limits = get_joint_limits_from_xml()
    
    # Reset environment and let it stabilize
    obs, info = env.reset()
    print("Environment reset. Stabilizing skeleton...")
    
    # Let skeleton stabilize for a moment with small holding torques
    stabilize_actions = torch.zeros(env.num_envs, env.num_actions)
    for step in range(50):  # 1 second stabilization
        obs, rewards, dones, info = env.step(stabilize_actions)
    
    print("Skeleton stabilized. Starting joint tests...\n")

    
    
    # Test each action joint
    for action_idx, action_name in enumerate(env.action_spec):
        print(f"Testing Action {action_idx}: {action_name}")
        
        # Get corresponding joint name and index using direct mapping (Option C)
        # This uses the pre-computed action-to-joint-index mapping for O(1) lookup
        joint_idx = env.action_to_joint_idx[action_name]
        joint_name = env.robot.joints[joint_idx].name
        
        print(f"  Joint: {joint_name} (index: {joint_idx})")
        
        # Get joint limits
        xml_limit = xml_limits.get(joint_name, (-3.14, 3.14))  # Default to ±π
        print(f"  XML Limits: {xml_limit[0]:.3f} to {xml_limit[1]:.3f} radians")
        print(f"  XML Limits: {np.degrees(xml_limit[0]):.1f}° to {np.degrees(xml_limit[1]):.1f}°")
        
        # Test sequence: neutral -> max -> min -> neutral
        # Use large torques like test_skeleton.py since action clamping is disabled
        test_positions = [0.0, 100.0, -100.0, 0.0]  # Use strong torques to overcome gravity
        position_names = ["Neutral", "Positive Limit", "Negative Limit", "Return to Neutral"]
        
        for pos_idx, target_action in enumerate(test_positions):

            print(f"    {position_names[pos_idx]} (action: {target_action:+.1f})")
            
            # Create action vector (all zeros except current joint)
            actions = torch.zeros(env.num_envs, env.num_actions)
            actions[0, action_idx] = target_action
            
            # Apply action for multiple steps to reach position
            for step in range(50):  # 1.0 seconds at 50Hz - more time to move
                obs, rewards, dones, info = env.step(actions)
                
                # Get current joint position
                current_pos = env.dof_pos[0, joint_idx].item()
                
                if step % 15 == 0:  # Print every 0.3 seconds
                    print(f"      Step {step:2d}: Joint pos = {current_pos:+.3f} rad ({np.degrees(current_pos):+.1f}°)")
                    
                # Check if environment reset due to falling
                if torch.any(dones):
                    print(f"      Warning: Environment reset at step {step}")
                    break
            
            time.sleep(0.2)  # Brief pause between positions
        
        print()  # Blank line between joints
        
        # Brief pause between different joints
        time.sleep(0.5)
    
    print("=== Joint Range Test Complete ===")
    
    # Summary
    print(f"\nSummary:")
    print(f"- Total DOFs: {env.num_dofs}")
    print(f"- Actuated joints: {env.num_actions}")
    print(f"- Tested joints: {len(env.action_spec)}")
    
    # List joint ranges
    print(f"\nJoint Limits Summary:")
    for action_name in env.action_spec:
        joint_idx = env.action_to_joint_idx[action_name]
        joint_name = env.dof_names[joint_idx]
        xml_limit = xml_limits.get(joint_name, (-3.14, 3.14))
        print(f"  {joint_name:20s}: {xml_limit[0]:+.3f} to {xml_limit[1]:+.3f} rad "
              f"({np.degrees(xml_limit[0]):+.1f}° to {np.degrees(xml_limit[1]):+.1f}°)")
    
    return True


if __name__ == "__main__":
    print("Starting comprehensive joint testing...\n")
    
    # Test joint ranges
    success = test_joint_ranges()
    
    print("\nTesting complete!")