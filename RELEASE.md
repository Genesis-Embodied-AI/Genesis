# Genesis Release Note

## 0.3.1

This small release addresses the most pressing regressions that has been pointed out by the community since 0.3.0. Support of coupling between Rigid Body and FEM has been improved and should be more reliable, though it is still considered experimental for now. Apart from that, no behavior changes are to be expected.

### New Features

* Support 2-channel (LA) textures in Rasterizer. (@LeonLiu4) (#1519)
* Add 'get_weld_constraints' API. (@LeonLiu4) (#1370)
* Add USD Materials Baking. (@ACMLCZH) (#1300)
* Add dedicated sensor manager. (@Milotrince) (#1518)
* Enhance SAP coupler for coupling between Rigid body and Fem object. (@Libero0809) (#1458)
* Add IMU sensor. (@Milotrince) (#1551)
* Add Fem fixed constraint for implicit solver. (@Libero0809) (#1562)
* Add Joint Equality Constraints for the SAP Coupler. (@Libero0809) (#1565)

### Bug Fixes

* Fix point-cloud rendering from Camera depth map. (@@ceasor-mao, @duburcqa) (#1512, #1515)
* Fix various rendering bugs. (@duburcqa) (#1537)
* Fix Z-up orientation and vertex color. (@ceasor-mao) (#1540)
* Fix hibernation. (@gasnica) (#1542)
* Fix backend fallback mechanism causing deadlock in Rasterizer. (@duburcqa) (#1546)
* Fix video recording dialog when running viewer in thread. (@duburcqa) (#1547)
* Fix joint friction loss. (@duburcqa) (#1555)
* Fix taichi debug mode errors and warnings. (@Libero0809) (#1560)

### Miscellaneous

* Migrate to native Python API for 'splashsurf'. (@duburcqa) (#1531)
* Refactor weld constraint API. (@duburcqa) (#1536)
* Allow following entity or mounting camera building scene. (@duburcqa) (#1548)
* Faster Genesis import. (@duburcqa) (#1549)
* Avoid rendering cameras at reset. (@Kashu7100) (#1552)
* Enable taichi debug mode in tests if possible. (@duburcqa) (#1541)
* Re-enable markers by default on RGB offscreen cameras. (@duburcqa) (#1570)

## 0.3.0

This release focuses primarily on stability, covering everything from MJCF/URDF parsing to rendering and physics, backend by a new CI infrastructure running more than 200 unit tests on all supported platforms. The most requested Mujoco features that were previously missing have been implemented. Native support of batching has been extended to all solvers except Stable Fluid, motion planning, and rendering via [gs-madrona](https://github.com/Genesis-Embodied-AI/gs-madrona). Finally, support of soft body dynamics has been enhanced, with the introduction of constraints and coupling between soft and rigid body dynamics.

### New Features

* Add link-wise mask for poss and quats in multilink Inverse Kinematics. (@ziyanx02) (#499)
* Update HoverEnv, update hyperparams, and visualization while training. (@KafuuChikai) (#533)
* Add Mac OS and Windows OS support to the viewer. (@kohya-ss, @duburcqa) (#610, #782)
* Add method to compute classical links acceleration. (@zswang666, @duburcqa) (#451, #1228)
* Support mounted cameras on rigid links. (@wangyian-me, @abhijitmajumdar) (#618, #1323)
* Cameras and main viewer can now track an entity. (@jgleyze) (#611)
* Support fixed entity. (@ziyanx02, @duburcqa) (#673, #1187)
* Support separated rendering for each environment. (@ACMLCZH) (#545, #723)
* Support environment masking to Inverse Kinematics. (@Kashu7100) (#732)
* Support equality constraint and closed-loop robots. (@YilingQiao) (#636)
* Add support on headless rendering on Windows OS. (@duburcqa) (#798)
* Expose public API for Forward Kinematics. (@Kashu7100) (#802)
* Add setter for robot mass. (@Kashu7100, @YilingQiao) (#828, #605)
* Support advanced collision pair filtering. (@duburcqa, @Kashu7100) (#816, #1438, #1499)
* Enable visualizing path. (@bxtbold) (#815)
* Add environment masking to more rigid body methods. (@Kashu7100) (#832)
* Add support of Ellipsoid geometry. (@duburcqa) (#864)
* Add 'show_link_frame' option to the viewer for better debugging. (@Kashu7100) (#871)
* Add environment masking to 'get_links_net_contact_force'. (@Kashu7100) (#880)
* Add support of ball joint type and compound joints. (@YilingQiao, @duburcqa) (#853, #1078, #1080)
* Luisa Render upgrade to support Apple Silicon. (@ACMLCZH) (#886)
* Support equality joint constraint. (@YilingQiao) (#919)
* Support mimic joint in URDF. (@YilingQiao) (#928)
* Support weld constraint. (@YilingQiao) (#948)
* Add method to render pointcloud on cameras. (@wangyian-me) (#897)
* Expose different convex decomposition error thresholds for robots and objects. (@duburcqa) (#1058)
* Force interactive viewer camera Z-axis up. (@duburcqa) (#1060)
* Add helper to convert terrain mesh to height field. (@YilingQiao) (#1033)
* Add maxvolume support for TetGen-based tetrahedralization. (@kosuke1701) (#1088)
* Support parallel simulation for deformable materials. (@wangyian-me) (#1005)
* Add public API to access rigid body mass matrix. (@Kashu7100) (#1132)
* Add public API getter/setter for constraint solver parameters. (@duburcqa) (#1173)
* Add option to exclude self-collision. (@Kashu7100) (#1229)
* Add profiling options. (@hughperkins) (#1247)
* Add implicit FEM with newton and conjugate gradient methods. (@Libero0809) (#1215)
* Implement BVH using Linear BVH. (@Libero0809) (#1241)
* Add USD parsing. (@ACMLCZH) (#1051)
* Expose optional subterrain parameters. (@LeonLiu4) (#1289)
* Add GJK-EPA algorithm for rigid body collision detection. (@SonSang) (#1213, #1357)
* Add linear_corotated elastic material for FEM. (@Libero0809) (#1304)
* Add fast vs high-performance taichi compilation mode. (@hughperkins) (#1330)
* Support environment-wise gravity. (@LeonLiu4, @Milotrince) (#1324, #1498)
* Expose method to compute Jacobian at a specific point. (@LeonLiu4) (#1353)
* Support segmentation map for deformable materials. (@ACMLCZH) (#1363)
* Support parallel path planning. (@Kashu7100) (#1316)
* Support dragging to physical object interactively in viewer. (@gasnica) (#1346, #1378, #1411, #1443)
* Enhance SAP Coupler to support self collision between FEM objects. (@Libero0809) (#1375)
* Support vertex constraints for FEM objects. (@Milotrince) (#1310)
* Add sensor abstraction. (@Milotrince) (#1381)
* Integrate Madrona batch renderer. (@yuhongyi) (#1416)
* Support joint friction. (@YilingQiao) (#1479)

### Bug Fixes

* Improve URDF and MJCF loading. (@zhenjia-xu, @bxtbold, @YilingQiao, @zswang666, @duburcqa) (#517, #675, #735, #744, #765, #777, #792, #872, #913, #936, #940, #988, #1147, #1154, #1159, #1169, #1218, #1235, #1262, #1287, #1501)
* Rework backend & device selection logic. (@lgleim) (#568)
* More robust cross-platform rendering support (viewer and cameras). (@alesof, @eratc, @duburcqa, @YilingQiao, @Kashu7100) (#404, #644, #774, #779, #783, #784, #787, #796, #799, #800, #807, #809, #810, #813, #814, #915, #983, #1069, #1070, #1071, #1073, #1074, #1420, #1421, #1426)
* More robust cachie mechanism for simulation pre-processing. (@duburcqa) (#801)
* Improve debug and high-precision mode. (@duburcqa) (#863)
* Fix constraint solver termination condition. (@duburcqa) (#867)
* Reset collision detection state when setting qpos. (@duburcqa) (#868)
* Use collision geometry as visual for bodies not having any. (@duburcqa) (#870)
* Only add world link/joint if it has at least one geom. (@duburcqa) (#884)
* Avoid segfault because of exceeding number of collisions. (@duburcqa) (#898)
* Fix box-box collision detection. (@duburcqa) (#910)
* Fix fixed joint and body handling. (@duburcqa) (#916, #952)
* Fix link velocity computation. (@duburcqa) (#941)
* Avoid useless convex decomposition. (@duburcqa) (#957)
* Improve numerical stability of MPR collision detection algorithm. (@duburcqa) (#966, #977, #1336)
* Improve multi-point contact stability. (@duburcqa) (#967, #1012, #1117, #1297)
* Fix genesis destroy. (@duburcqa) (#1007)
* Fix issue when adding multiple FEM entities. (@kosuke1701) (#1014)
* Using Vulkan backend for taichi even if no Intel XPU device is available. (@duburcqa) (#1025)
* Try to repair partially "broken" meshes if possible. (@duburcqa) (#1023, #1075, #1077)
* Fix scaling of poly-articulated robots. (@duburcqa) (#1039, #1108)
* Fix mujoco vs genesis discrepancies. (@duburcqa) (#1097)
* More robust robot loading and default options. (@duburcqa) (#1098)
* Improve external force handling. (@duburcqa) (#1292)
* Fix terrain collision detection. (@duburcqa) (#1338)
* Prevent unrealistic angular velocity of Drone entities causing numerical instability. (@duburcqa) (#1405)
* Do not consider markers as physical objects for rendering. (@duburcqa) (#14948)

### Miscellaneous

* Update Drone Entity and Training Performance Enhancements. (@KafuuChikai) (#598)
* Added PID controller util, quadcopter PID controller, flight examples. (@jebbrysacz) (#501)
* Add support of Numpy 2.0. (@johnnynunez, @duburcqa) (#711, #791) 
* Setup unit test infrastructure. (@duburcqa) (#876)
* Various minor improvements. (@duburcqa) (#889)
* Get rid of internally maintained trimesh dependency. (@duburcqa) (#918)
* Refactor rigid body data accessors. (@duburcqa) (#924)
* Speed-up convert taichi field to torch. (@duburcqa) (#935)
* Cleanup Genesis init / exit. (@duburcqa) (#989)
* Use torch cuda with vulkan taichi backend if available. (@duburcqa) (#1043)
* Fix naming convention inconsistencies. (@duburcqa) (#1053)
* Add API documentation. (@zswang666) (#1105)
* Use 2*dt as default timeconst for urdf and mesh. (@YilingQiao) (#1115)
* Improve runtime and compile time performance. (@YilingQiao, @duburcqa) (#1164, #1268, #1277)
* More robust decimation processing. Enable decimation by default. (@duburcqa) (#1186)
* Add compile- and run-time performance monitoring. (@duburcqa) (#1209)
* More efficient unit test distribution across workers. (@duburcqa) (#1275)
* Enable GJK collision detection algorithm by default. (@duburcqa) (#1439)
* Enable box-box collision detection by default. (@duburcqa) (#1442)

## 0.2.1

### Bug Fixes
* Fix various visualization and rendering bugs. (@RobRoyce, @VincentCCandela, @Likhithsai2580)
* Resolve some platform-dependent issues. (@abhaybd, @NekoAsakura)
* Fix the issue with loading box textures when parsing MJCF files.
* Correct asset path handling.
* Fix repr output in IPython. (@JohnnyDing)
* Resolve bugs in locomotion examples. (@yang-zj1026)
* Fix several issues with MPR and contact islands during collision detection.

### New Features
* Add a smoke simulator driven by Stable Fluid, along with a demo. (@PingchuanMa)
* Introduce APIs for applying external forces and torques.
* Introduce APIs for setting friction ratios.
* Add a domain randomization example.
* Improve kernel cache loading speed by 20~30%. (@erizmr)
* Provide an interactive drone control and visualization script. (@PieterBecking)
* Introduce an RL environment and examples for the drone environment. (@KafuuChikai)
* Add an option to enable or disable the batch dimension for (links/DOFs) information. Users can choose the trade-off between performance and flexibility.
* Add Docker files. (@Kashu7100)
* Implement the MuJoCo box-box collision detection algorithm for more stable grasping.
* Include the backflip training script and checkpoints. (@ziyanx02)
* Enable support for entity merging.
* Add support for custom inverse kinematics (IK) chains.

### Miscellaneous
* Improve documentation and fix typos. (@sangminkim-99, @sjtuyinjie, @CharlesCNorton, @eltociear, @00make, @marcbone, @00make, @pierridotite, @takeshi8989, @NicholasZiglio, @AmbarishGK)
* Add CONTRIBUTING.md and update CI configurations.
* Introduce multi-language support. (@TitanSage02, @GengYiran, @DocyNoah)

We would also like to acknowledge the ongoing PRs. Some of them have not yet been merged because we have not had enough time to fully test them:
* Unitree G1 walking. (@0nhc)
* Blood, vessels, and heart simulation. (@lhemerly)
* Cross-platform and rendering compatibility. (@VioletBenin, @DearVa, @JonnyDing)
* Docker support. (@skurtyyskirts, @yuxiang-gao, @serg-yalosovetsky)
