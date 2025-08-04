import numpy as np
import torch
import genesis as gs
import sys
import os

# Add the path to access the skeleton environment
sys.path.append('/home/ez/Documents/Genesis/genesis_loco/environments')
from skeleton_humanoid import SkeletonHumanoidEnv

def create_walking_poses():
    """Create keyframe poses for walking motion"""
    
    # Joint names in order as they appear in the action spec
    joint_names = [
        "lumbar_extension", "lumbar_bending", "lumbar_rotation",  # Spine (3)
        "hip_flexion_r", "hip_adduction_r", "hip_rotation_r",    # Right hip (3)  
        "knee_angle_r", "ankle_angle_r",                         # Right leg (2)
        "hip_flexion_l", "hip_adduction_l", "hip_rotation_l",    # Left hip (3)
        "knee_angle_l", "ankle_angle_l",                         # Left leg (2)
        "arm_flex_r", "arm_add_r", "arm_rot_r",                  # Right arm (3)
        "elbow_flex_r", "pro_sup_r", "wrist_flex_r", "wrist_dev_r", # Right arm (4)
        "arm_flex_l", "arm_add_l", "arm_rot_l",                  # Left arm (3) 
        "elbow_flex_l", "pro_sup_l", "wrist_flex_l", "wrist_dev_l"  # Left arm (4)
    ]
    
    # Standing pose (neutral position)
    standing_pose = np.zeros(len(joint_names))
    
    # Walking pose 1: Left leg forward, right leg back
    walk_pose1 = np.array([
        0.0, 0.0, 0.0,           # Spine - keep neutral
        -0.5, 0.0, 0.0,          # Right hip - extend back
        0.1, 0.0,                # Right knee slightly bent, ankle neutral
        0.5, 0.0, 0.0,           # Left hip - flex forward  
        0.8, 0.0,                # Left knee more bent, ankle neutral
        0.3, 0.0, 0.0,           # Right arm - swing forward
        0.0, 0.0, 0.0, 0.0,      # Right arm details
        -0.3, 0.0, 0.0,          # Left arm - swing back
        0.0, 0.0, 0.0, 0.0       # Left arm details
    ])
    
    # Walking pose 2: Right leg forward, left leg back  
    walk_pose2 = np.array([
        0.0, 0.0, 0.0,           # Spine - keep neutral
        0.5, 0.0, 0.0,           # Right hip - flex forward
        0.8, 0.0,                # Right knee more bent, ankle neutral
        -0.5, 0.0, 0.0,          # Left hip - extend back
        0.1, 0.0,                # Left knee slightly bent, ankle neutral  
        -0.3, 0.0, 0.0,          # Right arm - swing back
        0.0, 0.0, 0.0, 0.0,      # Right arm details
        0.3, 0.0, 0.0,           # Left arm - swing forward
        0.0, 0.0, 0.0, 0.0       # Left arm details
    ])
    
    # Mid-step poses for smoother transitions
    walk_mid1 = (standing_pose + walk_pose1) / 2
    walk_mid2 = (walk_pose1 + walk_pose2) / 2
    walk_mid3 = (walk_pose2 + standing_pose) / 2
    
    return [standing_pose, walk_mid1, walk_pose1, walk_mid2, walk_pose2, walk_mid3]

def main():
    # Initialize Genesis
    gs.init(backend=gs.gpu)
    
    # Create the skeleton humanoid environment
    env = SkeletonHumanoidEnv(
        num_envs=1,
        episode_length_s=20.0,
        dt=0.02,
        show_viewer=True
    )
    
    # Setup PD control for position control
    env.setup_pd_control()
    
    # Get walking poses
    walking_poses = create_walking_poses()
    print(f"Created {len(walking_poses)} walking poses")
    print(f"Number of DOFs in environment: {env.num_dofs}")
    print(f"Number of actions: {env.num_actions}")
    
    # Convert poses to torch tensors
    pose_tensors = []
    for pose in walking_poses:
        # Pad or truncate to match the number of controllable DOFs
        if len(pose) < env.num_dofs:
            padded_pose = np.zeros(env.num_dofs)
            padded_pose[:len(pose)] = pose
            pose = padded_pose
        elif len(pose) > env.num_dofs:
            pose = pose[:env.num_dofs]
        
        pose_tensors.append(torch.tensor(pose, device=env.device, dtype=torch.float32).unsqueeze(0))
    
    print("Starting walking sequence...")
    
    # Walking control loop
    current_pose_idx = 0
    steps_per_pose = 50  # Hold each pose for 50 simulation steps
    step_counter = 0
    total_steps = 0
    max_steps = 2000
    
    # Reset environment first
    obs, _ = env.reset()
    
    while total_steps < max_steps:
        # Get current target pose
        target_pose = pose_tensors[current_pose_idx]
        
        # Apply position control using the mapped joint indices
        joint_positions = torch.zeros((1, env.num_dofs), device=env.device)
        
        # Map the target pose to actual joint positions using the action mapping
        for i, action_name in enumerate(env.action_spec):
            if i < target_pose.shape[1]:
                joint_idx = env.action_to_joint_idx[action_name]
                joint_positions[0, joint_idx] = target_pose[0, i]
        
        # Use position control
        env.robot.control_dofs_position(joint_positions.squeeze(0))
        
        # Step simulation
        env.scene.step()
        
        step_counter += 1
        total_steps += 1
        
        # Move to next pose
        if step_counter >= steps_per_pose:
            current_pose_idx = (current_pose_idx + 1) % len(pose_tensors)
            step_counter = 0
            print(f"Switching to pose {current_pose_idx}, total steps: {total_steps}")
    
    print("Walking sequence completed!")

if __name__ == "__main__":
    main()