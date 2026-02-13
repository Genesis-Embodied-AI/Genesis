# Genesis Release Note

## 0.3.14

This release mainly focuses on usability, by extending support of USD and introducing a new external plugin mechanism for the interactive viewer. Besides, the performance of the simulation has been significantly improved for collision-heavy scenes (up to 30%).

### New Features

* Introduce interactive viewer plugins. (@Milotrince) (#2004, #2357)
* Add naming logics to entities. (@YilingQiao) (#2303)
* Support rendering textures for USD scenes. (@ACMLCZH) (#2286)
* Add invalid spatial inertia diagnosis. (@duburcqa) (#2297, #2321, #2367)

### Bug Fixes

* Fix wrong default sampler for SPH solver causing numerical stability issues. (@erizmr) (#2280)
* Fix render destroy. (@nimrod-gileadi, @duburcqa) (#2282, #2358)
* Fix IPC Coupler. (@duburcqa) (#2299)
* Fix batched numpy 'euler_to_R' geom util. (@Kashu7100) (#2306)
* Fix glTF mesh loading. (@duburcqa) (#2296, #2311, #2316, #2329)
* Fix 'Mesh.convert_to_zup' by applying scaling out-of-place. (@duburcqa)
* Fix rigid body entity hibernation mechanism. (@YilingQiao) (#2294)
* Fix DFSPH solver. (@erizmr) (#2302)
* Fix parsing material of primitive geometries in MJCF files. (@ACMLCZH) (#2328)
* Fix 'draw_debug_frames' after PR#1869. (@duburcqa) (#2330)
* Fix compatibility with 'trimesh<4.6.0'. (@duburcqa) (#2334)
* Fix linesearch edge-case fallback. (@erizmr) (#2339)
* Fix mujoco-compatible GJK multi-contact for box primitive. (@hughperkins) (#2341)
* FIX FEM entity rendering with Raytracer backend. (@duburcqa) (#2356)
* Fix combining Rasterizer-based camera sensors and interactive viewer. (@YilingQiao) (#2351)

### Miscellaneous

* Track mesh UVs in FEM/PBD solvers for rendering of deformable entities. (@alelievr) (#2323)
* Support scrolling menu in 'gs view'. (@Kashu7100) (#2335)
* Add memory to CI performance monitoring report. (@hughperkins) (#2281, #2291, #2293, #2295, #2298, #2300, #2312, #2315, #2320)
* Improve support of Linux ARM. (@duburcqa) (#2317)
* Clearer error message of 'RigidEntity.(get_joint|get_link)'. (@Kashu7100) (#2313)
* Add Markdown Files to Facilitate AI Tools. (@YilingQiao) (#2305)
* Support --record in the RL stage in Manipulation example (@SnakeOnex) (#2344)
* More robust OpenGL context initialisation for Rasterizer. (@duburcqa) (#2354)
* Add benchmark for Unitree G1. (@hughperkins) (#2310)
* Recomputing inertia for primitive geometries using analytical formula. (@duburcqa) (#2337)
* Speed up linesearch via batched alpha evals and reduced global memory access. (@erizmr) (#2350)
* Disable shadow and plane reflection when using software rendering. (@duburcqa) (#2365)
* Workaround for 'pyglet' bug. (@duburcqa) (#2385)

## 0.3.13

This small release adds user-friendly diagnosis of invalid Rigid physics properties and improves support of GLTF meshes.

### Breaking changes

* Apply 'FileMorph.file_meshes_are_zup' to all meshes including GLTF. (@duburcqa) (#2275)
* Do not officially support importing GLTF morph Mesh as Z-UP. (@duburcqa) (#2279)

### New Features

* Add Magnetometer measurement to IMU Sensor. (@sunkmechie) (#2265)
* Check validity of links spatial inertia and support forcing computation from geoms. (@duburcqa) (#2273, #2276)

### Bug Fixes

* Improve support of attached RigidEntity. (@duburcqa) (#2256, #2259)
* Fix attaching RayTracer camera sensor. (@duburcqa) (#2266)
* Fix empty data when adding more than 3 Raycast Sensors. (@JackLowry) (#2268)
* Fix Raycast Sensor for batched environments. (@d-corsi) (#2269)
* More robust filtering of self-collision in neutral configuration.  (@duburcqa) (#2278)

### Miscellaneous

* Improve performance and add torch support to 'utils.geom.slerp'. (@Kashu7100) (#2260)

## 0.3.12

This PR focuses on performance improvements (x4 faster for complex scenes with 64 < n_dofs < 96 and n_envs=4096 compared to 0.3.10). Besides, initial support of heterogenous object and USD stage import for rigid body simulation has been introduced.

### New Features

* Add method to compute axis-aligned bounding boxes of visual geometries. (@duburcqa) (#2185)
* Add partial support of batched camera sensor with Rasterizer. (@Narsil) (#2207, #2212)
* Add support for attaching MPM particles to rigid links. (@YilingQiao) (#2205)
* Add support of USD import for Rigid Body. (@alanray-tech) (#2067)
* Add support of batched textures to BatchRenderer. (@ACMLCZH) (#2077)
* Add support of fisheye camera mode to BatchRenderer. (@ACMLCZH) (#2138)
* Add batched simulation of heterogeneous objects. (@YilingQiao) (#2202)
* Filter out self-collision pairs active in neutral configuration. (@duburcqa) (#2251)

### Bug Fixes

* Fix zero-copy for fields on Apple Metal. (@duburcqa) (#2188, #2223)
* Fix compatibility with 'numpy<2.0'. (@duburcqa) (#2197)
* Fix invalid default particle sampler on Linux ARM. (@duburcqa) (#2211)
* Fix 'RigidGeom.get_(pos|quat)' invalid shape. (@duburcqa) (#2218)
* Fix various sensor bugs and add zero-copy to contact force sensors. (@duburcqa) (#2232, #2235)
* Clear dynamic weld at scene reset. (@YilingQiao) (#2233)
* Fix viewer not closed at scene destroy if running in background thread. (@duburcqa) (#2236)
* More robust handling of corrupted cache. (@duburcqa) (#2241)

### Miscellaneous

* More intuitive visualisation of camera frustum in interactive viewer. (@duburcqa) (#2180)
* Remove broken and unmaintained Avatar Solver. (@duburcqa) (#2181)
* Speedup non-tiled hessian cholesky factor and solve. (@duburcqa) (#2182, #2183)
* Force public getters to return by-value to avoid mistake. (@duburcqa) (#2184)
* Improve CI infrastructure. (@hughperkins, @duburcqa) (#1981, #2166, #2190, #2194, #2195, #2242, #2245, #2250)
* Add Ruff format. (@Narsil) (#2213, #2214, #2215)
* Store texture path for primitive morphs as metadata. (@Rush2k) (#2227)
* Improve import logics of y-up vs z-up file meshes. (@AnisB) (#2237)
* Simplify boolean contact sensors update logics. (@duburcqa) (#2238)
* Rigid methods set_qpos,set_dofs_position now clear error code. (@duburcqa) (#2253)

## 0.3.11

The main focus of this release is to improve scaling of the simulation wrt the complexity of the scene, and better leverage GPUn compute for small to moderate batch sizes (0<=n_envs<=8192). As usual, a bunch of minor bugs have been fixed.

### New Features

* Support specifying offset transform for camera sensor. (@YilingQiao) (#2126)
* Enable zero-copy for fields on Metal if supported. (@duburcqa) (#2174)

### Bug Fixes

* Avoid discontinuities in smooth animations caused by singularities. (@Rush2k) (#2116)
* Fix forward update logics. (@duburcqa) (#2122)
* Fix kernel caching mechanism hindering performance. (@duburcqa) (#2123)
* Fix support of old torch for 'set_dofs_velocity' when velocity=None. (@YilingQiao) (#2160)
* Force rendering systematically when updating camera sensor. (@YilingQiao) (#2162)
* Fix incorrect lighting when offscreen cameras based on rasterizer. (@duburcqa) (#2163)
* Fix rasterizer race conditions when running in background thread. (@duburcqa) (#2169)
* Fix broken exception handling when loading obj files with unsupported face type. (@Kashu7100) (#2170)
* Fix 'pysplashsurf' memory leak causing OOM error. (@duburcqa) (#2173, #2176)
* Diagnose out-of-bound SDF gradient index. (@duburcqa) (#2177)

### Miscellaneous

* Stop assessing warmstart vs smooth acc at constraint solver init. (@duburcqa) (#2117)
* Speedup collision detection broad phase on GPU. (@duburcqa) (#2128)
* More comprehensive benchmarks. (@duburcqa) (#2137)
* Accelerate constraint solver first pass using shared memory. (@duburcqa) (#2136, #2140)
* Further optimize cholesky solve using warp reduction and memory padding. (@duburcqa) (#2145, #2146)
* Improve runtime speed by optimize memory layout of constraint solver. (@duburcqa) (#2147)
* Fast mass matrix factorisation on GPU using shared memory. (@duburcqa) (#2154)
* Optimize rigid body dynamics to scale better wrt dofs and entities. (@duburcqa) (#2161)
* Fix spurious deprecated property warnings during introspection. (@duburcqa) (#2168)
* Various solver refactoring to support GsTaichi Main. (@hughperkins, @duburcqa) (#2131, #2135, #2143, #2151)
* Improve single-threaded cpu-based simulation runtime speed by upgrading gstaichi. (@hughperkins) (#2129, #2153)

## 0.3.10

Small release mainly fixing bugs.

### Bug Fixes

* Fix parsing for special material properties in glTF meshes (@duburcqa) (#2110)

### Miscellaneous

* More robust detection of invalid simulation state. (@duburcqa) (#2112)

## 0.3.9

Small release mainly polishing features that were introduced in previous release.

### New Features

* [CHANGING] Replace SDF fallback by GJK. (@duburcqa) (#2081)
* [CHANGING] Improve inertial estimation if undefined. (@YilingQiao) (#2100)
* Add support of boolean masking as index. (@duburcqa) (#2087)
* Fix and improve merging of rigid entities. (@duburcqa) (#2098)

### Bug Fixes

* Fix increased memory usage due to differentiable simulation. (@duburcqa) (#2074)
* Fix 'envs_idx' in motion planning. (@duburcqa) (#2093)
* Fix 'DroneEntity.set_propellels_rpm'. (@duburcqa) (#2095)
* Fix extended broadcasting. (@duburcqa) (#2096)
* Fix 'RigidEntity.set_dofs_velocity'. (@robin271828) (#2102)
* Fix joint stiffness not taking into account neutral position. (@YilingQiao) (#2105)
* Fix explicit URDF material color being ignored. (@duburcqa) (#2107)

### Miscellaneous

* Speed up torch-based geom utils via 'torch.jit.script'. (@duburcqa) (#2075)
* Improve scalability wrt number of contacts. (@duburcqa) (#2085, #2103)
* Make Go2 RL env GPU-sync free. (@duburcqa) (#2092)

## 0.3.8

The performance of data accessors have been dramatically improved by leveraging zero-copy memory sharing between GsTaichi and Torch. Beyond that, the robustness of the default contact algorithm has been improved, and differentiable forward dynamics for Rigid Body simulation is not partially available. Last, but not least, GsTaichi dynamic array mode is finally enabled back by default!

### New Features

* [CHANGING] More robust MPR+SDF collision detection algorithm. (@duburcqa) (#1983, #1985)
* [CHANGING] Disable box-box by default. (@duburcqa) (#1982)
* Enable back GsTaichi dynamic array mode by default except for MacOS. (@duburcqa) (#1977)
* Add error code to rigid solver. (@duburcqa) (#1979)
* Add option to force batching of fixed vertices. (@duburcqa) (#1998)
* Leverage GsTaichi zero-copy in data accessors. (@duburcqa) (#2011, #2019, #2021, #2023, #2025, #2030, #2037, #2048, #2054)
* Add an option to disable keyboard shortcuts (@YilingQiao) (#2026)
* Add support of 'capsule' primitive in URDF file. (@duburcqa) (#2045)
* Add full support of tensor broadcasting in getters. (@duburcqa) (#2051)
* Add rasterizer, batch renderer, and raytracer as sensor (@YilingQiao) (#2010)
* Differentiable forward dynamics for rigid body sim. (@SonSang) (#1808, #2063, #2068)

### Bug Fixes

* Fix sensor IMU accelerometer signal. (@Milotrince) (#1962)
* Fix 'RigidJoint.(get_anchor_pos | get_anchor_axis)' getters. (@alexis779) (#2012)
* Prevent nan to propagate in position and raise exception. (@duburcqa) (#2033)
* Fix camera following entity for 'fix_orientation=True'. (@duburcqa) (#2038)
* Fix support of Hybrid entity with non-fixed base link. (@duburcqa) (#2040)
* Raise exception if trying to load PointCloud as Mesh. (@duburcqa) (#2042)
* Fix boolean mask inversion for PyTorch 2.x (@yoneken) (#2056)
* Fix URDF color overwrite. (@duburcqa) (#2065)

### Miscellaneous

* Reduce memory footprint. (@duburcqa) (#2000, #2031)
* Only enable GJK by default if gradient computation is required. (@duburcqa) (#1984)
* Bump GsTaichi Support Nvidia GPU Blackwell. (@johnnynunez) (#2002)
* Add dependency version upper-bound 'tetgen< 0.7.0'. (@YilingQiao) (#2029)
* Bump up min version requirement for Torch after introducing zero-copy. (@duburcqa) (#2034)
* Add 'parse_glb_with_zup' option to all file-based Morph. (@ACMLCZH) (#1938)
* Enable more example scripts in CI. (@duburcqa) (#2057)
* Fix fast cache and zero-copy bugs. (@hughperkins) (#2050)

## 0.3.7

The performance of GsTaichi dynamic array mode has been greatly improved. Now it should be on par with fixed-size array mode (aka performance mode) for very large batch sizes, and up to 30% slower for non-batched simulations. This mode is still considered experimental and must be enabled manually by setting the env var 'GS_ENABLE_NDARRAY=1'. Just try it if you are tired of endlessly waiting for the simulation to compile!

### New Features

* Implement position-velocity controller. (@matthieuvigne) (#1948)

### Bug Fixes

* Fix missing option `diffuse_texture` to `Glass` surface. (@Kashu7100) (#1934)
* Fix interactive viewer. (@YilingQiao) (#1931)
* Fix external coupling forces from other solvers not affecting rigid bodies. (@SonSang) (#1941)
* Fix silent process killing issue in MPM simulation by raising an exception. (@SonSang) (#1949)
* Fix 'discrete_obstacles_terrain' being completely flat. (@jgillick) (#1972)

### Miscellaneous

* Added warning message about stable timestep for SPH solver. (@SonSang) (#1925)
* Reduce memory usage due to diff constraint solver. (@YilingQiao) (#1930)
* Faster non-batched simulation. (@duburcqa) (#1935)
* Fix or silent dev warnings. (@duburcqa) (#1944)
* Add caching to Rigid Link state getters to improve performance. (@duburcqa) (#1940, #1955)
* Add support of Linux ARM. (@duburcqa) (#1961)
* Add 'GS_PARA_LEVEL' env var to force kernel parallelization level. (@duburcqa) (#1968)

## 0.3.6

A new experimental interface with the Incremental Potential Contact coupling solver [libuipc](https://github.com/spiriMirror/libuipc) has been introduced, mainly targeting cloth simulation.

### New Features

* Add Rigid ‘get_dofs_frictionloss’ public API method. (@ax-anoop) (#1904)
* Add IPC (Incremental Potential Contact) coupling system. (@Roushelfy) (#1859)

### Bug Fixes

* Fix missing dependencies in Docker (@schlagercollin) (#1896)
* Fix sensor recorder Matplotlib plotter on Linux and Windows. (@duburcqa) (#1894)
* Fix particle emitter failure if batch size = num emit particles. (@duburcqa) (#1901)
* Fix MJCF handling of visual groups. (@duburcqa) (#1902)

### Miscellaneous

* Disable GsTaichi dynamic arrays by default (again!) (@duburcqa) (#1915)
* Reset control during reset. (@YilingQiao) (#1920)

## 0.3.5

Minor release mainly aiming at polishing existing features and addressing some major performance regression that was introduced end of august. GsTaichi dynamic array type and fast cache are now enabled by default on Linux and Windows (opt-in on MacOS via env variable 'GS_ENABLE_NDARRAY'), which should help avoiding recompilation in most cases.

### New Features

* Make convex decomposition cache scale-invariant. (@SonSang) (#1810)
* Add parsing of joint friction and damping for URDF files. (@duburcqa) (#1833)
* Differentiable constraint solver (@SonSang) (#1733)
* Add support of GsTaichi dynamic array type and fast cache mode. (@YilingQiao, @duburcqa) (#1868, #1873, #1875, #1880)

### Bug Fixes

* Fix hybrid entity with rigid. (@duburcqa) (#1819)
* Reduce general memory usage. (@hughperkins) (#1828)
* Improve reliability and performance of plotters. (@duburcqa) (#1836)
* Fix loading MJCF file with includes. (@YilingQiao, @duburcqa) (#1838, #1840)
* Fix DepthCamera sensor. (@Milotrince) (#1842)
* Fix and refactor named terrain. (@duburcqa) (#1845)
* Raise exception in case of invalid set entity pos/quat. (@duburcqa) (#1847, #1869)
* Fix gltf Loading for URDF. (@ACMLCZH) (#1857)
* Fix friction loss causing nan on Apple GPU. (@duburcqa) (#1860)
* Fixed minor bug in differentiable contact detection. (@SonSang) (#1864)
* Fix race condition during concurrent viewer refresh and draw_debug_*. (@duburcqa) (#1883, #1884)
* Enable GPU and fix dt in some example scripts. (@schlagercollin) (#1890)

### Miscellaneous

* Reduce memory usage of Raycaster sensor. (@Milotrince) (#1850)
* Add 'get_particles_pos' helper method to SPHEntity. (@YilingQiao) (#1871)
* Enabling gstaichi fast cache by default. (@hughperkins) (#1885)
* Refactor Rigid simulation data management. (@duburcqa) (#1888)
* Do not silent errors during build. (@duburcqa) (#1889)

## 0.3.4

This minor release mainly introduces first-class sensor support (IMU, Contact Sensor, LiDAR, Depth camera and more), incl. recording and plotting facilities. The rigid-rigid hydroelastic contact model has also been added. As usual, a fair share of bugs have been fixed, with unit test coverage gradually improving.

### Behavior Changing

* Support rendering deformable body for batched env. (@YilingQiao) (#1697)
* More sensible defaults for camera far, near. (@duburcqa) (#1678)
* Fix invweight and meaninertia not always considering scale and dofs armature. (@duburcqa) (#1696)

### New Features

* Refactor 'FrameImageExporter' to improve performance and support normal & segmentation. (@duburcqa) (#1671)
* Add support of normal & segmentation for Madrona Batch Rendering. (@ACMLCZH) (#1563)
* Add 'noslip' optional post-processing step to suppress slip/drift. (@YilingQiao) (#1669)
* Add first-class data recorders and plotters. (@Milotrince) (#1646, #1718)
* Add rigid-rigid hydroelastic contact model. (@Libero0809) (#1572)
* Add option to display sensor information in the interactive viewer. (@Milotrince) (#1770)
* Add support of differentiable contact detection (Work In Progress). (@SonSang) (#1701)
* Add Raycaster sensor (Lidar and DepthCamera). (@Milotrince, @duburcqa, @jgillick) (#1726, #1772, #1809, #1815)
* Add full support gstaichi ndarray to Rigid Body solver. (@YilingQiao, @SonSang, @duburcqa) (#1674, #1682, #1683, #1690, #1693, #1695)
* Add full support gstaichi fast caching mechanism. (@hughperkins, @YilingQiao) (#1709, #1720, #1730, #1812)

### Bug Fixes

* Fix data races in getting contact and equality constraints. (@YilingQiao) (#1676)
* Fix MPM muscle activation. (@YilingQiao) (#1692)
* Fix non-flat terrain support. (@Kashu7100, @YilingQiao, @duburcqa) (#1691, #1777, #1779)
* Disable mesh processing when loading URDF for consistency. (@duburcqa) (#1708)
* Fix segfault at exit when running viewer in background thread with offscreen cameras. (@duburcqa) (#1703)
* Fix all the example scripts. (@YilingQiao, @duburcqa) (#1743, #1773, #1724, #1785, #1787, #1801, #1804)
* Fix logics for duplicating collision geometries as visual in MJCF. (@hokindeng, @duburcqa) (#1732, #1750)
* Randomize uniform terrain along both axes. (@jgillick) (#1747)
* Fix contact sensors always returning zeros. (@Milotrince) (#1761)
* Fix LBVH stuck in infinite loop for small number of AABBs. (@duburcqa) (#1766)
* Fix broken interactive viewer backend fallback mechanism. (@duburcqa) (#1797)
* Fix camera follow entity. (@duburcqa) (#1805)
* Fix compound joints for 'set_dofs_position'. (@duburcqa) (#1678)
* Fix some mesh-related issues (@ACMLCZH) (#1800)

### Miscellaneous

* Fix support of 'pyglet<2.0'. (@Kashu7100) (#1670)
* Add vision-based manipulation example (@yun-long) (#1493)
* Reduce max_collision_pairs to save memory (@YilingQiao) (#1672)
* Remove 'gs clean' utility. (@duburcqa) (#1723)
* Rename 'get_aabb' in 'get_AABB' and add deprecation warning. (@duburcqa) (#1778)
* Improve interactive viewer performance. (@duburcqa) (#1784)
* Raise exception in cause of particle sampling failure. (@duburcqa) (#1792)
* Remove 'is_free' that was confusing and partially redundant with 'is_fixed'. (@duburcqa) (#1795)

## 0.3.3

This minor release fixes a few non-blocking rendering issues for the Rasterizer backend.

### Bug Fixes

* Fix shadow map not properly rendered for objects far away from floor plane. (@duburcqa) (#1664)
* Fix genesis import failure if tkinter is failing at init on MacOS. (@duburcqa) (#1666)
* Fix default visualization mode for emitter surface. (@duburcqa) (#1665)

### Miscellaneous

* Expose parameters for ground plane tiling. (@yuhongyi) (#1657)
* Add support of 'ti.ndarray' to 'ti_field_to_torch' and rename in 'ti_to_torch'. (@duburcqa) (#1661)

## 0.3.2

This minor release fixes a few additional regressions and initiates migration to our own open-source fork of Taichi, [GsTaichi](https://github.com/Genesis-Embodied-AI/gstaichi) (contributions are welcome!).

### Behavior Changing

* Disable decimation if deemed unnecessary and unreliable. Reduce default aggressiveness. (@duburcqa) (#1644)
* Fix primitive and mesh's COM. (@YilingQiao) (#1638)

### New Features

* Add 'set_dofs_frictionloss' method for dynamic joint friction control. (@LeonLiu4) (#1614)
* Add initial experimental support of gstaichi fast cache feature. (@hughperkins) (#1631)
* Add 'ref' optional argument to 'get_links_pos'. (@YilingQiao) (#1638)

### Bug Fixes

* Fix Inverse Kinematics algorithm. (@Kashu7100, @duburcqa) (#1582, #1586)
* Fix save frame as png image in interactive viewer. (@Kashu7100) (#1606)
* Filter out collision pairs involved in weld equality constraints. (@duburcqa) (#1621)
* Fix viewer backend fallback when running in main thread. (@duburcqa) (#1630)
* Fix 'quat_to_xyz' singularity edge-case. (@duburcqa) (#1628)
* Fix CUDA runtime being initialized by 'get_device'. (@duburcqa) (#1634)

### Miscellaneous

* Re-enable world-frame in the 'gs view' standalone viewer. (@Kashu7100) (#1584)
* Migrate from 'taichi' to 'gstaichi'. (@duburcqa, @hughperkins) (#1550, #1618, #1645)
* Update documentation so the doc will be compiled based on the latest main. (@YilingQiao) (#1616)

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
