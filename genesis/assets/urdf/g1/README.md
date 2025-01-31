# Unitree G1 Description (URDF & MJCF)

## Overview

This package includes a universal humanoid robot description (URDF & MJCF) for the [Unitree G1](https://www.unitree.com/g1/), developed by [Unitree Robotics](https://www.unitree.com/).

MJCF/URDF for the G1 robot:

| MJCF/URDF file name           | `mode_machine` | Hip roll reduction ratio | Update status | dof#leg | dof#waist | dof#arm | dof#hand |
| ----------------------------- | :------------: | :----------------------: | ------------- | :-----: | :-------: | :-----: | :------: |
| `g1_23dof`                    |       1        |           14.5           | Beta          |   6*2   |     1     |   5*2   |    0     |
| `g1_29dof`                    |       2        |           14.5           | Beta          |   6*2   |     3     |   7*2   |    0     |
| `g1_29dof_with_hand`          |       2        |           14.5           | Beta          |   6*2   |     3     |   7*2   |   7*2    |
| `g1_29dof_lock_waist`         |       3        |           14.5           | Beta          |   6*2   |     1     |   7*2   |    0     |
| `g1_23dof_rev_1_0`            |       4        |           22.5           | Up-to-date    |   6*2   |     1     |   5*2   |    0     |
| `g1_29dof_rev_1_0`            |       5        |           22.5           | Up-to-date    |   6*2   |     3     |   7*2   |    0     |
| `g1_29dof_with_hand_rev_1_0`  |       5        |           22.5           | Up-to-date    |   6*2   |     3     |   7*2   |   7*2    |
| `g1_29dof_lock_waist_rev_1_0` |       6        |           22.5           | Up-to-date    |   6*2   |     1     |   7*2   |    0     |
| `g1_dual_arm`                 |       9        |           null           | Up-to-date    |    0    |     0     |   7*2   |    0     |

## Visulization with [MuJoCo](https://github.com/google-deepmind/mujoco)

1. Open MuJoCo Viewer

   ```bash
   pip install mujoco
   python -m mujoco.viewer
   ```

2. Drag and drop the MJCF/URDF model file (`g1_XXX.xml`/`g1_XXX.urdf`) to the MuJoCo Viewer.
