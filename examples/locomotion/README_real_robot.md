# Real Robot Deployment for Go2 Backflip

This directory contains code for deploying the trained backflip policy on a real Unitree Go2 robot.

## Files

- `go2_real_env.py`: Real robot environment that interfaces with Unitree Go2
- `go2_real_backflip.py`: Main script to execute backflip on real robot  
- `requirements_real.txt`: Python dependencies for real robot deployment

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_real.txt
```

### 2. Install Unitree SDK

Install the official Unitree SDK for Go2:

```bash
pip install unitree_sdk2py
```

Or build from source:
```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

### 3. Robot Network Setup

1. Connect to the robot's WiFi network
2. Ensure your computer can communicate with the robot
3. Default robot IP is typically `192.168.123.161`

## Usage

### Safety First!

⚠️ **IMPORTANT SAFETY WARNINGS** ⚠️

- Ensure the robot is in a safe, open environment (minimum 3x3 meters)
- Have emergency stop ready at all times  
- Never run this on a table or elevated surface
- Ensure the robot is fully charged and calibrated
- Have someone ready to catch/support the robot if needed
- Start with `--dry_run` mode for testing

### Basic Usage

```bash
# Test without real robot (dry run)
python go2_real_backflip.py -e single --dry_run

# Execute single backflip on real robot
python go2_real_backflip.py -e single

# Execute double backflip on real robot  
python go2_real_backflip.py -e double
```

### Advanced Options

```bash
python go2_real_backflip.py \
    -e single \
    --device cuda \
    --safety_timeout 15.0
```

## Parameters

- `-e, --exp_name`: Experiment type (`single` or `double`)
- `--device`: PyTorch device (`cpu` or `cuda`)
- `--safety_timeout`: Maximum execution time in seconds (default: 10.0)
- `--dry_run`: Test mode without real robot

## Troubleshooting

### Robot Connection Issues

1. Check network connectivity: `ping 192.168.123.161`
2. Verify SDK installation: `python -c "import unitree_sdk2py"`
3. Check robot is in the correct mode (should be in low-level control mode)

### Policy Issues

1. Ensure policy files exist in `./backflip/` directory
2. Check policy file format (should be `.pt` TorchScript files)
3. Verify policy was trained with compatible observation space

### Robot Behavior Issues

1. **Robot doesn't move**: Check if robot is in correct control mode
2. **Unstable movements**: May need to tune PD gains (`kp`, `kd`) for real hardware
3. **Robot falls**: Environment might need safety modifications or different initial pose

## Sim2Real Considerations

The real robot deployment includes several adaptations from simulation:

1. **Observation Processing**: Real IMU and joint sensor data
2. **Action Scaling**: May need adjustment for real actuators  
3. **PD Control Gains**: Tuned for real robot dynamics
4. **Safety Features**: Timeouts and emergency stops
5. **Communication**: Real-time control loop at 500Hz

## Tuning for Your Robot

You may need to adjust these parameters in `go2_real_env.py`:

- `kp`, `kd`: PD control gains
- `action_scale`: Action scaling factor
- `obs_scales`: Observation normalization scales
- Joint limits and safety bounds

## Emergency Procedures

If something goes wrong:

1. **Immediate**: Press emergency stop button on robot
2. **Software**: Press Ctrl+C to interrupt the script
3. **Physical**: Be ready to support/catch the robot
4. **Power**: Turn off robot power if necessary

The script includes automatic safety features:
- Timeout-based termination
- Graceful shutdown on interruption  
- Return to safe position after execution