![Genesis](imgs/big_text.png)

![Teaser](imgs/teaser.png)

[![PyPI - Version](https://img.shields.io/pypi/v/genesis-world)](https://pypi.org/project/genesis-world/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/genesis-world)](https://pypi.org/project/genesis-world/)
[![GitHub Issues](https://img.shields.io/github/issues/Genesis-Embodied-AI/Genesis)](https://github.com/Genesis-Embodied-AI/Genesis/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/Genesis-Embodied-AI/Genesis)](https://github.com/Genesis-Embodied-AI/Genesis/discussions)

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README_CN.md)

# What is Genesis?

Genesis is a physics platform designed for general purpose *Robotics/Embodied AI/Physical AI* applications. It is simultaneously multiple things:

1. A **universal physics engine** re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
2. A **lightweight**, **ultra-fast**, **pythonic**, and **user-friendly** robotics simulation platform.
3. A powerful and fast **photo-realistic rendering system**.
4. A **generative data engine** that transforms user-prompted natural language description into various modalities of data.

Powered by a universal physics engine re-designed and re-built from the ground up, Genesis integrates various physics solvers and their coupling into a unified framework. This core physics engine is further enhanced by a generative agent framework that operates at an upper level, aiming towards fully **automated data generation** for robotics and beyond.

Currently, we are open-sourcing the **underlying physics engine and the simulation platform**. Our generative framework is a modular system that incorporates many different generative modules, each handling a certain range of data modalities, routed by a high level agent. Some of the modules integrated existing papers and some are still under submission. Access to our generative feature will be gradually rolled out in the near future. If you are interested, feel free to explore more the [paper list](#papers-behind-genesis) below.

Genesis is built and will continuously evolve with the following ***long-term missions***:

1. **Lowering the barrier** to using physics simulations and making robotics research accessible to everyone. (See our [commitment](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/mission.html))
2. **Unifying a wide spectrum of state-of-the-art physics solvers** into a single framework, allowing re-creating the whole physical world in a virtual realm with the highest possible physical, visual and sensory fidelity, using the most advanced simulation techniques.
3. **Minimizing human effort** in collecting and generating data for robotics and other domains, letting the data flywheel spin on its own.

Project Page: <https://genesis-embodied-ai.github.io/>

## Key Features

- **Speed**: Genesis delivers an unprecedented simulation speed -- over 43 million FPS when simulating a Franka robotic arm with a single RTX 4090 (430,000 times faster than real-time).
- **Cross-platform**: Genesis runs natively across different systems (Linux, MacOS, Windows), and across different compute backend (CPU, Nvidia GPU, AMD GPU, Apple Metal).
- **Unification of various physics solvers**: Genesis develops a unified simulation framework that integrates various physics solvers: Rigid body, MPM, SPH, FEM, PBD, Stable Fluid.
- **Support a wide range of material models**:  Genesis supports simulation (and the coupling) of rigid and articulated bodies, various types of liquids, gaseous phenomenon, deformable objects, thin-shell objects and granular materials.
- **Support for a wide range of robots**: Robot arm, legged robot, drone, *soft robot*, etc., and extensive support for loading different file types: `MJCF (.xml)`, `URDF`, `.obj`, `.glb`, `.ply`, `.stl`, etc.
- **Photorealistic and high-performance ray-tracer**: Genesis supports native ray-tracing based rendering.
- **Differentiability**: Genesis is designed to be fully compatible with differentiable simulation. Currently, our MPM solver and Tool Solver are differentiable, and differentiability for other solvers will be added soon (starting with rigid-body simulation).
- **Physics-based Tactile Sensor**: Genesis involves a physics-based and differentiable [tactile sensor simulation module](https://github.com/Genesis-Embodied-AI/DiffTactile). This will be integrated to the public version soon (expected in version 0.3.0).
- **User-friendliness**: Genesis is designed in a way to make using simulation as simple as possible. From installation to API design, if there's anything you found counter-intuitive or difficult to use, please [let us know](https://github.com/Genesis-Embodied-AI/Genesis/issues).

## Getting Started

### Quick Installation

Genesis is available via PyPI:

```bash
pip install genesis-world  # Requires Python >=3.9;
```

You also need to install **PyTorch** following the [official instructions](https://pytorch.org/get-started/locally/).

If you would like to try out the latest version, we suggest you to git clone from the repo and do `pip install -e .` instead of via PyPI.

### Documentation

Please refer to our [documentation site (English)](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html) / [(Chinese)](https://genesis-world.readthedocs.io/zh-cn/latest/user_guide/index.html) for detailed installation steps, tutorials and API references. 

## Contributing to Genesis

The goal of the Genesis project is to build a fully transparent, user-friendly ecosystem where contributors from both robotics and computer graphics can **come together to collaboratively create a high-efficiency, realistic (both physically and visually) virtual world for robotics research and beyond**.

We sincerely welcome *any forms of contributions* from the community to make the world a better place for robots. From **pull requests** for new features, **bug reports**, to even tiny **suggestions** that will make Genesis API more intuitive, all are wholeheartedly appreciated!

## Support

- Please use Github [Issues](https://github.com/Genesis-Embodied-AI/Genesis/issues) for bug reports and feature requests.

- Please use GitHub [Discussions](https://github.com/Genesis-Embodied-AI/Genesis/discussions) for discussing ideas, and asking questions.

## License and Acknowledgment

The Genesis source code is licensed under Apache 2.0.
The development of Genesis won't be possible without these amazing open-source projects:

- [Taichi](https://github.com/taichi-dev/taichi): for providing a high-performance cross-platform compute backend. Kudos to all the members providing technical support from taichi!
- [FluidLab](https://github.com/zhouxian/FluidLab) for providing a reference MPM solver implementation
- [SPH_Taichi](https://github.com/erizmr/SPH_Taichi) for providing a reference SPH solver implementation
- [Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) and [PBF3D](https://github.com/WASD4959/PBF3D) for providing a reference PBD solver implementation
- [MuJoCo](https://github.com/google-deepmind/mujoco) and [Brax](https://github.com/google/brax) for providing reference for rigid body dynamics
- [libccd](https://github.com/danfis/libccd) for providing reference for collision detection
- [PyRender](https://github.com/mmatl/pyrender) for rasterization-based renderer
- [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) and [LuisaRender](https://github.com/LuisaGroup/LuisaRender) for its ray-tracing DSL
- [trimesh](https://github.com/mikedh/trimesh), [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) and [CoACD](https://github.com/SarahWeiii/CoACD) for geometry processing

## Papers behind Genesis

Genesis is a large scale effort that integrates state-of-the-art technologies of various existing and on-going research work into a single system. Here we include a non-exhaustive list of all the papers that contributed to the Genesis project in one way or another:

- Xian, Zhou, et al. "Fluidlab: A differentiable environment for benchmarking complex fluid manipulation." arXiv preprint arXiv:2303.02346 (2023).
- Xu, Zhenjia, et al. "Roboninja: Learning an adaptive cutting policy for multi-material objects." arXiv preprint arXiv:2302.11553 (2023).
- Wang, Yufei, et al. "Robogen: Towards unleashing infinite data for automated robot learning via generative simulation." arXiv preprint arXiv:2311.01455 (2023).
- Wang, Tsun-Hsuan, et al. "Softzoo: A soft robot co-design benchmark for locomotion in diverse environments." arXiv preprint arXiv:2303.09555 (2023).
- Wang, Tsun-Hsuan Johnson, et al. "Diffusebot: Breeding soft robots with physics-augmented generative diffusion models." Advances in Neural Information Processing Systems 36 (2023): 44398-44423.
- Katara, Pushkal, Zhou Xian, and Katerina Fragkiadaki. "Gen2sim: Scaling up robot learning in simulation with generative models." 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024.
- Si, Zilin, et al. "DiffTactile: A Physics-based Differentiable Tactile Simulator for Contact-rich Robotic Manipulation." arXiv preprint arXiv:2403.08716 (2024).
- Wang, Yian, et al. "Thin-Shell Object Manipulations With Differentiable Physics Simulations." arXiv preprint arXiv:2404.00451 (2024).
- Lin, Chunru, et al. "UBSoft: A Simulation Platform for Robotic Skill Learning in Unbounded Soft Environments." arXiv preprint arXiv:2411.12711 (2024).
- Zhou, Wenyang, et al. "EMDM: Efficient motion diffusion model for fast and high-quality motion generation." European Conference on Computer Vision. Springer, Cham, 2025.
- Qiao, Yi-Ling, Junbang Liang, Vladlen Koltun, and Ming C. Lin. "Scalable differentiable physics for learning and control." International Conference on Machine Learning. PMLR, 2020.
- Qiao, Yi-Ling, Junbang Liang, Vladlen Koltun, and Ming C. Lin. "Efficient differentiable simulation of articulated bodies." In International Conference on Machine Learning, PMLR, 2021.
- Qiao, Yi-Ling, Junbang Liang, Vladlen Koltun, and Ming Lin. "Differentiable simulation of soft multi-body systems." Advances in Neural Information Processing Systems 34 (2021).
- Wan, Weilin, et al. "Tlcontrol: Trajectory and language control for human motion synthesis." arXiv preprint arXiv:2311.17135 (2023).
- Wang, Yian, et al. "Architect: Generating Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting." arXiv preprint arXiv:2411.09823 (2024).
- Zheng, Shaokun, et al. "LuisaRender: A high-performance rendering framework with layered and unified interfaces on stream architectures." ACM Transactions on Graphics (TOG) 41.6 (2022): 1-19.
- Fan, Yingruo, et al. "Faceformer: Speech-driven 3d facial animation with transformers." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
- Wu, Sichun, Kazi Injamamul Haque, and Zerrin Yumak. "ProbTalk3D: Non-Deterministic Emotion Controllable Speech-Driven 3D Facial Animation Synthesis Using VQ-VAE." Proceedings of the 17th ACM SIGGRAPH Conference on Motion, Interaction, and Games. 2024.
- Dou, Zhiyang, et al. "C· ase: Learning conditional adversarial skill embeddings for physics-based characters." SIGGRAPH Asia 2023 Conference Papers. 2023.

... and many more on-going work.

## Citation

If you used Genesis in your research, we would appreciate it if you could cite it. We are still working on a technical report, and before it's public, you could consider citing:

```bibtex
@software{Genesis,
  author = {Genesis Authors},
  title = {Genesis: A Universal and Generative Physics Engine for Robotics and Beyond},
  month = {December},
  year = {2024},
  url = {https://github.com/Genesis-Embodied-AI/Genesis}
}
```
