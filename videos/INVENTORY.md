# Catalogue media inventory

Total entries in README catalogue: **88**

All media (rendered videos and placeholder thumbnails alike) are normalized to **480×270** so the README grid renders uniformly.

## Summary

| Status | Count |
|---|---|
| Rendered locally | 48 |
| Placeholder — can't render | 32 |
| External (nyx-for-genesis) | 7 |
| Existing capture in repo | 1 |

## Detail

| # | Section | Entry | Example | Media | Status | Notes |
|---|---|---|---|---|---|---|
| 1 | Physics | Rigid: franka_cube | `examples/rigid/franka_cube.py` | `videos/rigid_franka_cube.webp` | Rendered locally | video_script/rigid_franka_cube.py |
| 2 | Physics | Rigid: grasp_bottle | `examples/rigid/grasp_bottle.py` | `videos/rigid_grasp_bottle.webp` | Rendered locally | video_script/rigid_grasp_bottle.py |
| 3 | Physics | Rigid: collision tower | `examples/collision/tower.py` | `videos/collision_tower.webp` | Rendered locally | video_script/collision_tower.py |
| 4 | Physics | Rigid: collision pyramid | `examples/collision/pyramid.py` | `videos/collision_pyramid.png` | Existing capture in repo |  |
| 5 | Physics | Rigid: contype | `examples/collision/contype.py` | `videos/collision_contype.webp` | Rendered locally | video_script/collision_contype.py |
| 6 | Physics | FEM: elastic_dragon | `examples/elastic_dragon.py` | `videos/elastic_dragon.webp` | Rendered locally | video_script/elastic_dragon.py |
| 7 | Physics | FEM: hard & soft constraint | `examples/fem_hard_and_soft_constraint.py` | `videos/fem_hard_and_soft_constraint.webp` | Rendered locally | video_script/fem_hard_and_soft_constraint.py |
| 8 | Physics | FEM: SAP fixed constraint | `examples/sap_coupling/fem_fixed_constraint.py` | `videos/sap_fem_fixed_constraint.webp` | Rendered locally | video_script/sap_fem_fixed_constraint.py |
| 9 | Physics | MPM: tutorial | `examples/tutorials/mpm.py` | `videos/tutorials_mpm.webp` | Rendered locally | video_script/tutorials_mpm.py |
| 10 | Physics | MPM: sand wheel | `examples/coupling/sand_wheel.py` | `videos/coupling_sand_wheel.webp` | Rendered locally | video_script/coupling_sand_wheel.py |
| 11 | Physics | MPM: differentiable push | `examples/differentiable_push.py` | `candidate_readme_img/diff_push.webp` | Placeholder — can't render | optimization loop, no inherent visualization |
| 12 | Physics | SPH: sph_rigid | `examples/coupling/sph_rigid.py` | `videos/coupling_sph_rigid.webp` | Rendered locally | video_script/coupling_sph_rigid.py |
| 13 | Physics | SPH: sph_mpm | `examples/coupling/sph_mpm.py` | `videos/coupling_sph_mpm.webp` | Rendered locally | video_script/coupling_sph_mpm.py |
| 14 | Physics | SPH: liquid | `examples/tutorials/sph_liquid.py` | `videos/tutorials_sph_liquid.webp` | Rendered locally | video_script/tutorials_sph_liquid.py |
| 15 | Physics | PBD: liquid | `examples/pbd_liquid.py` | `videos/pbd_liquid.webp` | Rendered locally | video_script/pbd_liquid.py |
| 16 | Physics | PBD: cloth | `examples/tutorials/pbd_cloth.py` | `videos/tutorials_pbd_cloth.webp` | Rendered locally | video_script/tutorials_pbd_cloth.py |
| 17 | Physics | Stable Fluid: smoke | `examples/smoke.py` | `candidate_readme_img/catalogue/smoke.webp` | Placeholder — can't render | density-grid PNG export, not a camera |
| 18 | Physics | IPC: objects_falling | `examples/IPC_Solver/ipc_objects_falling.py` | `candidate_readme_img/catalogue/ipc_objects_falling.webp` | Placeholder — can't render | needs pyuipc (not installed) |
| 19 | Physics | IPC: robot_grasp_cube | `examples/IPC_Solver/ipc_robot_grasp_cube.py` | `candidate_readme_img/catalogue/ipc_robot_grasp_cube.webp` | Placeholder — can't render | needs pyuipc (not installed) |
| 20 | Physics | IPC: robot_cloth_teleop | `examples/IPC_Solver/ipc_robot_cloth_teleop.py` | `candidate_readme_img/ipc_cloth.webp` | Placeholder — can't render | needs pyuipc (not installed) |
| 21 | Physics | Coupler: cloth_on_rigid | `examples/coupling/cloth_on_rigid.py` | `videos/coupling_cloth_on_rigid.webp` | Rendered locally | video_script/coupling_cloth_on_rigid.py |
| 22 | Physics | Coupler: rigid_mpm_attachment | `examples/coupling/rigid_mpm_attachment.py` | `videos/coupling_rigid_mpm_attachment.webp` | Rendered locally | video_script/coupling_rigid_mpm_attachment.py |
| 23 | Physics | Coupler: cut_dragon | `examples/coupling/cut_dragon.py` | `videos/coupling_cut_dragon.webp` | Rendered locally | video_script/coupling_cut_dragon.py |
| 24 | Physics | Coupler: water_wheel | `examples/coupling/water_wheel.py` | `videos/coupling_water_wheel.webp` | Rendered locally | video_script/coupling_water_wheel.py |
| 25 | Physics | Coupler: flush_cubes | `examples/coupling/flush_cubes.py` | `videos/coupling_flush_cubes.webp` | Rendered locally | video_script/coupling_flush_cubes.py |
| 26 | Physics | Coupler: grasp_soft_cube | `examples/coupling/grasp_soft_cube.py` | `videos/coupling_grasp_soft_cube.webp` | Rendered locally | video_script/coupling_grasp_soft_cube.py |
| 27 | Physics | Coupler: FEM cube + arm | `examples/coupling/fem_cube_linked_with_arm.py` | `videos/coupling_fem_cube_linked_with_arm.webp` | Rendered locally | video_script/coupling_fem_cube_linked_with_arm.py |
| 28 | Physics | Coupler: cloth_attached_to_rigid | `examples/coupling/cloth_attached_to_rigid.py` | `videos/coupling_cloth_attached_to_rigid.webp` | Rendered locally | video_script/coupling_cloth_attached_to_rigid.py |
| 29 | Physics | SAP: franka_grasp_fem_sphere | `examples/sap_coupling/franka_grasp_fem_sphere.py` | `videos/sap_franka_grasp_fem_sphere.webp` | Rendered locally | video_script/sap_franka_grasp_fem_sphere.py |
| 30 | Physics | SAP: franka_grasp_rigid_cube | `examples/sap_coupling/franka_grasp_rigid_cube.py` | `videos/sap_franka_grasp_rigid_cube.webp` | Rendered locally | video_script/sap_franka_grasp_rigid_cube.py |
| 31 | Physics | SAP: fem_sphere_and_cube | `examples/sap_coupling/fem_sphere_and_cube.py` | `candidate_readme_img/catalogue/sap_fem_sphere_and_cube.webp` | Placeholder — can't render | Needs huggingface_hub download of cube8.obj |
| 32 | Rendering | Camera demo (Nyx / Luisa / Pyrender) | `examples/rendering/demo.py` | `videos/rendering_demo.webp` | Rendered locally | video_script/rendering_demo.py |
| 33 | Rendering | Follow entity | `examples/rendering/follow_entity.py` | `videos/rendering_follow_entity.webp` | Rendered locally | video_script/rendering_follow_entity.py |
| 34 | Rendering | Animated camera | `examples/rendering/moving_camera.py` | `videos/rendering_moving_camera.webp` | Rendered locally | video_script/rendering_moving_camera.py |
| 35 | Rendering | Async / off-thread render | `examples/render_async.py` | `candidate_readme_img/catalogue/render_async.webp` | Placeholder — can't render | async/off-thread render — not naturally captured |
| 36 | Rendering | Render throughput | `examples/rendering/speed_test.py` | `candidate_readme_img/catalogue/rendering_speed_test.webp` | Placeholder — can't render | Throughput benchmark, no visual loop |
| 37 | Rendering | Nyx: hello | `nyx-for-genesis/blob/main/examples/01_hello_nyx.py` | `videos/nyx_01_hello_nyx.png` | External (nyx-for-genesis) |  |
| 38 | Rendering | Nyx: attached camera | `nyx-for-genesis/blob/main/examples/02_attached_camera.py` | `videos/nyx_02_attached_camera.webp` | External (nyx-for-genesis) |  |
| 39 | Rendering | Nyx: PBR materials | `nyx-for-genesis/blob/main/examples/03_materials.py` | `videos/nyx_03_materials.png` | External (nyx-for-genesis) |  |
| 40 | Rendering | Nyx: light types | `nyx-for-genesis/blob/main/examples/04_light_types.py` | `videos/nyx_04_light_types.png` | External (nyx-for-genesis) |  |
| 41 | Rendering | Nyx: 3D Gaussian splat | `nyx-for-genesis/blob/main/examples/05_gaussian_splat.py` | `videos/nyx_05_gaussian_splat.png` | External (nyx-for-genesis) |  |
| 42 | Rendering | Nyx: object picking | `nyx-for-genesis/blob/main/examples/06_object_picking.py` | `videos/nyx_06_object_picking.png` | External (nyx-for-genesis) |  |
| 43 | Rendering | Nyx: multi-cam multi-env | `nyx-for-genesis/blob/main/examples/07_multi_camera_multi_env.py` | `videos/nyx_07_multi_camera_multi_env.png` | External (nyx-for-genesis) |  |
| 44 | Compiler | Cross-backend bench (anymal_c) | `examples/speed_benchmark/anymal_c.py` | `candidate_readme_img/catalogue/speed_benchmark_anymal_c.webp` | Placeholder — can't render | non-visual / multi-process benchmark |
| 45 | Compiler | Cross-backend bench (franka) | `examples/speed_benchmark/franka.py` | `candidate_readme_img/catalogue/speed_benchmark_franka.webp` | Placeholder — can't render | non-visual / multi-process benchmark |
| 46 | Compiler | Parallel simulation | `examples/tutorials/parallel_simulation.py` | `videos/tutorials_parallel_simulation.webp` | Rendered locally | video_script/tutorials_parallel_simulation.py |
| 47 | Compiler | Multi-GPU training (DDP) | `examples/ddp_multi_gpu.py` | `candidate_readme_img/catalogue/ddp_multi_gpu.webp` | Placeholder — can't render | needs trained RL policy |
| 48 | Simulation Interface | Controlling a robot | `examples/tutorials/control_your_robot.py` | `videos/tutorials_control_your_robot.webp` | Rendered locally | video_script/tutorials_control_your_robot.py |
| 49 | Simulation Interface | Visualization | `examples/tutorials/visualization.py` | `videos/tutorials_visualization.webp` | Rendered locally | video_script/tutorials_visualization.py |
| 50 | Simulation Interface | Entity name | `examples/tutorials/entity_name.py` | `videos/tutorials_entity_name.webp` | Rendered locally | video_script/tutorials_entity_name.py |
| 51 | Simulation Interface | Heterogeneous envs | `examples/rigid/heterogeneous_simulation.py` | `videos/rigid_heterogeneous_simulation.webp` | Rendered locally | video_script/rigid_heterogeneous_simulation.py |
| 52 | Simulation Interface | Domain randomization | `examples/rigid/domain_randomization.py` | `videos/rigid_domain_randomization.webp` | Rendered locally | video_script/rigid_domain_randomization.py |
| 53 | Simulation Interface | Select rendered envs | `examples/tutorials/selecting_rendered_envs.py` | `videos/tutorials_selecting_rendered_envs.webp` | Rendered locally | video_script/tutorials_selecting_rendered_envs.py |
| 54 | Simulation Interface | Sensor: RGB camera | `examples/sensors/camera_as_sensor.py` | `candidate_readme_img/catalogue/sensors_camera_as_sensor.webp` | Placeholder — can't render | Multi-backend camera, dumps per-step PNGs |
| 55 | Simulation Interface | Sensor: depth camera | `examples/sensors/depth_camera_custom_vverts.py` | `videos/sensors_depth_camera_custom_vverts.webp` | Rendered locally | video_script/sensors_depth_camera_custom_vverts.py |
| 56 | Simulation Interface | Sensor: IMU | `examples/sensors/imu_franka.py` | `videos/sensors_imu_franka.webp` | Rendered locally | video_script/sensors_imu_franka.py |
| 57 | Simulation Interface | Sensor: lidar | `examples/sensors/lidar_teleop.py` | `candidate_readme_img/catalogue/sensors_lidar_teleop.webp` | Placeholder — can't render | Interactive keyboard teleop |
| 58 | Simulation Interface | Sensor: tactile | `examples/sensors/tactile_franka.py` | `candidate_readme_img/catalogue/sensors_tactile_elastomer_franka.webp` | Placeholder — can't render | Interactive keyboard teleop |
| 59 | Simulation Interface | Sensor: tactile sandbox | `examples/sensors/tactile_sandbox.py` | `candidate_readme_img/catalogue/sensors_tactile_sandbox.webp` | Placeholder — can't render | Interactive keyboard teleop |
| 60 | Simulation Interface | Sensor: contact force | `examples/sensors/contact_force_go2.py` | `videos/sensors_contact_force_go2.webp` | Rendered locally | video_script/sensors_contact_force_go2.py |
| 61 | Simulation Interface | Sensor: surface distance | `examples/sensors/surface_distance_shadowhand.py` | `candidate_readme_img/catalogue/sensors_proximity_shadowhand.webp` | Placeholder — can't render | Interactive keyboard teleop |
| 62 | Simulation Interface | Sensor: temperature grid | `examples/sensors/temperature_grid.py` | `candidate_readme_img/catalogue/sensors_temperature_grid.webp` | Placeholder — can't render | Interactive keyboard teleop |
| 63 | Simulation Interface | GUI: debug drawing | `examples/tutorials/draw_debug.py` | `candidate_readme_img/catalogue/tutorials_draw_debug.webp` | Placeholder — can't render | interactive (keyboard/mouse) |
| 64 | Simulation Interface | GUI: interactive debugging | `examples/tutorials/interactive_debugging.py` | `candidate_readme_img/catalogue/tutorials_interactive_debugging.webp` | Placeholder — can't render | interactive (keyboard/mouse) |
| 65 | Simulation Interface | GUI: mesh point picker | `examples/viewer_plugin/mesh_point_selector.py` | `candidate_readme_img/catalogue/viewer_mesh_point_selector.webp` | Placeholder — can't render | interactive (keyboard/mouse) |
| 66 | Simulation Interface | GUI: mouse interaction | `examples/viewer_plugin/mouse_interaction.py` | `candidate_readme_img/catalogue/viewer_mouse_interaction.webp` | Placeholder — can't render | interactive (keyboard/mouse) |
| 67 | Simulation Interface | GUI: ImGui joint control | `examples/gui/imgui_joint_control.py` | `candidate_readme_img/catalogue/gui_imgui_joint_control.webp` | Placeholder — can't render | interactive (keyboard/mouse) |
| 68 | Simulation Interface | GUI: keyboard teleop | `examples/keyboard_teleop.py` | `candidate_readme_img/catalogue/keyboard_teleop.webp` | Placeholder — can't render | interactive (keyboard/mouse) |
| 69 | Simulation Interface | GUI: interactive drone | `examples/drone/interactive_drone.py` | `candidate_readme_img/drone.webp` | Placeholder — can't render | interactive (keyboard/mouse) |
| 70 | Simulation Interface | Diff-IK controller | `examples/rigid/diffik_controller.py` | `videos/rigid_diffik_controller.webp` | Rendered locally | video_script/rigid_diffik_controller.py |
| 71 | Simulation Interface | Closed-loop control | `examples/rigid/closed_loop.py` | `videos/rigid_closed_loop.webp` | Rendered locally | video_script/rigid_closed_loop.py |
| 72 | Simulation Interface | Control franka | `examples/rigid/control_franka.py` | `videos/rigid_control_franka.webp` | Rendered locally | video_script/rigid_control_franka.py |
| 73 | Simulation Interface | Position control comparison | `examples/tutorials/position_control_comparison.py` | `videos/tutorials_position_control_comparison.webp` | Rendered locally | video_script/tutorials_position_control_comparison.py |
| 74 | Simulation Interface | IK + motion planning | `examples/tutorials/IK_motion_planning_grasp.py` | `videos/tutorials_IK_motion_planning_grasp.webp` | Rendered locally | video_script/tutorials_IK_motion_planning_grasp.py |
| 75 | Simulation Interface | Batched IK | `examples/tutorials/batched_IK.py` | `videos/tutorials_batched_IK.webp` | Rendered locally | video_script/tutorials_batched_IK.py |
| 76 | Simulation Interface | Advanced IK multilink | `examples/tutorials/advanced_IK_multilink.py` | `videos/tutorials_advanced_IK_multilink.webp` | Rendered locally | video_script/tutorials_advanced_IK_multilink.py |
| 77 | Simulation Interface | Scene save / load | `examples/hibernation.py` | `candidate_readme_img/catalogue/hibernation.webp` | Placeholder — can't render | Quadrants internal error on contact-island flag |
| 78 | Simulation Interface | USD I/O | `examples/usd/` | `candidate_readme_img/catalogue/usd_io.webp` | Placeholder — can't render | file I/O, no animation |
| 79 | Simulation Interface | Locomotion: Go2 backflip | `examples/locomotion/go2_backflip.py` | `candidate_readme_img/catalogue/locomotion_go2_backflip.webp` | Placeholder — can't render | Deploys a trained RL policy (no checkpoint locally) |
| 80 | Simulation Interface | Locomotion: Go2 train (RL) | `examples/locomotion/go2_train.py` | `candidate_readme_img/catalogue/locomotion_go2_train.webp` | Placeholder — can't render | needs trained RL policy |
| 81 | Simulation Interface | Manipulation: Franka grasp (RL) | `examples/manipulation/grasp_train.py` | `candidate_readme_img/catalogue/manipulation_grasp_train.webp` | Placeholder — can't render | needs trained RL policy |
| 82 | Simulation Interface | Manipulation: behavior cloning | `examples/manipulation/behavior_cloning.py` | `candidate_readme_img/catalogue/manipulation_behavior_cloning.webp` | Placeholder — can't render | needs trained RL policy |
| 83 | Simulation Interface | Drone: fly | `examples/drone/fly.py` | `videos/drone_fly.webp` | Rendered locally | video_script/drone_fly.py |
| 84 | Simulation Interface | Drone: fly route | `examples/drone/fly_route.py` | `candidate_readme_img/catalogue/drone_fly_route.webp` | Placeholder — can't render | Custom DronePIDController, complex import |
| 85 | Simulation Interface | Drone: hover train (RL) | `examples/drone/hover_train.py` | `candidate_readme_img/catalogue/drone_hover_train.webp` | Placeholder — can't render | needs trained RL policy |
| 86 | Simulation Interface | Advanced: muscle | `examples/tutorials/advanced_muscle.py` | `videos/tutorials_advanced_muscle.webp` | Rendered locally | video_script/tutorials_advanced_muscle.py |
| 87 | Simulation Interface | Advanced: worm | `examples/tutorials/advanced_worm.py` | `videos/tutorials_advanced_worm.webp` | Rendered locally | video_script/tutorials_advanced_worm.py |
| 88 | Simulation Interface | Advanced: hybrid robot | `examples/tutorials/advanced_hybrid_robot.py` | `videos/tutorials_advanced_hybrid_robot.webp` | Rendered locally | video_script/tutorials_advanced_hybrid_robot.py |
