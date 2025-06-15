import argparse
import time
import torch
from go2_real_env import Go2RealEnv


def get_cfgs():
    """Same configuration as simulation"""
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD gains (tuned for real robot - may need adjustment)
        "kp": 70.0,
        "kd": 3.0,
        # termination
        "termination_if_roll_greater_than": 1000,  # degree
        "termination_if_pitch_greater_than": 1000,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.35],
        "base_init_quat": [0.0, 0.0, 0.0, 1.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.5,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 60,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "reward_scales": {},
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0, 0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser(description="Execute backflip on real Unitree Go2 robot")
    parser.add_argument("-e", "--exp_name", type=str, default="single", 
                       choices=["single", "double"], help="Experiment name")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on")
    parser.add_argument("--safety_timeout", type=float, default=10.0, 
                       help="Safety timeout in seconds")
    parser.add_argument("--dry_run", action="store_true", 
                       help="Run without real robot (for testing)")
    args = parser.parse_args()

    print(f"Starting backflip execution: {args.exp_name}")
    print("=" * 50)
    
    if not args.dry_run:
        print("WARNING: This will execute backflip on real robot!")
        print("Make sure:")
        print("1. Robot is in safe environment with enough space")
        print("2. Emergency stop is accessible")
        print("3. Robot is properly calibrated and charged")
        print("4. You are ready to catch the robot if needed")
        print()
        
        response = input("Continue? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return

    # Get configurations
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    # Set episode length based on experiment
    if args.exp_name == "single":
        env_cfg["episode_length_s"] = 2
        print("Single backflip mode: 2 seconds")
    elif args.exp_name == "double":
        env_cfg["episode_length_s"] = 3
        print("Double backflip mode: 3 seconds")
    else:
        raise ValueError(f"Unknown experiment name: {args.exp_name}")

    # Create real robot environment
    try:
        env = Go2RealEnv(
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            device=args.device
        )
        print("Robot environment initialized successfully")
    except Exception as e:
        print(f"Failed to initialize robot environment: {e}")
        if not args.dry_run:
            return
        else:
            print("Continuing in dry run mode...")

    # Load trained policy
    try:
        policy_path = f"./backflip/{args.exp_name}.pt"
        policy = torch.jit.load(policy_path, map_location=args.device)
        print(f"Policy loaded from: {policy_path}")
    except Exception as e:
        print(f"Failed to load policy: {e}")
        return

    # Initialize robot to standing position
    print("Resetting robot to initial position...")
    obs, _ = env.reset()
    print("Robot ready for backflip execution")
    
    # Wait for user confirmation
    if not args.dry_run:
        input("Press Enter to start backflip execution...")
    
    # Execute backflip
    print("Starting backflip execution!")
    start_time = time.time()
    step_count = 0
    
    try:
        with torch.no_grad():
            while True:
                # Get action from policy
                actions = policy(obs)
                
                # Execute action
                obs, rewards, dones, infos = env.step(actions)
                step_count += 1
                
                # Check for completion
                if dones[0] or (time.time() - start_time) > args.safety_timeout:
                    break
                    
                # Small delay to match real-time execution
                time.sleep(0.002)  # 500Hz control loop
                
                # Print progress
                if step_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Step {step_count}, Elapsed: {elapsed:.2f}s")
    
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
    except Exception as e:
        print(f"\nError during execution: {e}")
    finally:
        # Safe shutdown
        print("Returning robot to safe position...")
        try:
            env.reset()  # Return to default position
            time.sleep(2.0)  # Allow time to settle
            env.close()
        except:
            pass
        
        total_time = time.time() - start_time
        print(f"Execution completed in {total_time:.2f} seconds ({step_count} steps)")
        print("Robot control terminated safely")


if __name__ == "__main__":
    main()