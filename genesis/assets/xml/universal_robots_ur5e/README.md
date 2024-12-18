# Universal Robots UR5e Description (MJCF)

Requires MuJoCo 2.3.3 or later.

## Overview

This package contains a simplified robot description (MJCF) of the
[UR5e](https://www.universal-robots.com/products/ur5-robot/) developed by
[Universal Robots](https://www.universal-robots.com/). It is derived from the
[publicly available URDF
description](https://github.com/ros-industrial/universal_robot/tree/kinetic-devel/ur_e_description).

<p float="left">
  <img src="ur5e.png" width="400">
</p>

### URDF → MJCF derivation steps

1. Converted the DAE [mesh
   files](https://github.com/ros-industrial/universal_robot/tree/kinetic-devel/ur_e_description/meshes/ur5e/visual)
   to OBJ format using [Blender](https://www.blender.org/).
2. Processed `.obj` files with  [`obj2mjcf`](https://github.com/kevinzakka/obj2mjcf).
3. Added `<mujoco> <compiler discardvisual="false"/> </mujoco>` to the URDF's
   `<robot>` clause in order to preserve visual geometries.
4. Loaded the URDF into MuJoCo and saved a corresponding MJCF.
5. Added a tracking light to the base.
6. Manually edited the MJCF to extract common properties into the `<default>` section.
7. Added position-controlled actuators. Max joint torque values were taken from
   [here](https://www.universal-robots.com/articles/ur/robot-care-maintenance/max-joint-torques/).
8. Added home joint configuration as a `keyframe`.
9. Manually designed collision geometries.
10. Added `scene.xml` which includes the robot, with a textured ground plane, skybox and haze.

## License

This model is released under a [BSD-3-Clause License](LICENSE).
