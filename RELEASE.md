# Genesis Release Note

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
