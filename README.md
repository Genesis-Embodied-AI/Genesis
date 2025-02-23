![Genesis](imgs/big_text.png)

![Teaser](imgs/teaser.png)

[![PyPI - Version](https://img.shields.io/pypi/v/genesis-world)](https://pypi.org/project/genesis-world/)
[![PyPI Downloads](https://static.pepy.tech/badge/genesis-world)](https://pepy.tech/projects/genesis-world)
[![GitHub Issues](https://img.shields.io/github/issues/Genesis-Embodied-AI/Genesis)](https://github.com/Genesis-Embodied-AI/Genesis/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/Genesis-Embodied-AI/Genesis)](https://github.com/Genesis-Embodied-AI/Genesis/discussions)
[![Discord](https://img.shields.io/discord/1322086972302430269?logo=discord)](https://discord.gg/nukCuhB47p)
<a href="https://drive.google.com/uc?export=view&id=1ZS9nnbQ-t1IwkzJlENBYqYIIOOZhXuBZ"><img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white" height="20" style="display:inline"></a>

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README.md)
[![README en Français](https://img.shields.io/badge/Francais-d9d9d9)](./README_FR.md)
[![한국어 README](https://img.shields.io/badge/한국어-d9d9d9)](./README_KR.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README_CN.md)
[![日本語版 README](https://img.shields.io/badge/日本語-d9d9d9)](./README_JA.md)

# Genesis

## 🔥 News
- [2025-01-09] We released a [detailed performance benchmarking and comparison report](https://github.com/zhouxian/genesis-speed-benchmark) on Genesis, together with all the test scripts.
- [2025-01-08] Released v0.2.1 🎊 🎉
- [2025-01-08] Created [Discord](https://discord.gg/nukCuhB47p) and [Wechat](https://drive.google.com/uc?export=view&id=1ZS9nnbQ-t1IwkzJlENBYqYIIOOZhXuBZ) group.
- [2024-12-25] Added a [docker](#docker) including support for the ray-tracing renderer
- [2024-12-24] Added guidelines for [contributing to Genesis](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/.github/CONTRIBUTING.md)

## Table of Contents

1. [What is Genesis?](#what-is-genesis)
2. [Key Features](#key-features)
3. [Quick Installation](#quick-installation)
4. [Docker](#docker)
5. [Documentation](#documentation)
6. [Contributing to Genesis](#contributing-to-genesis)
7. [Support](#support)
8. [License and Acknowledgments](#license-and-acknowledgments)
9. [Associated Papers](#associated-papers)
10. [Citation](#citation)

## What is Genesis?

Genesis is a physics platform designed for general-purpose *Robotics/Embodied AI/Physical AI* applications. It is simultaneously multiple things:

1. A **universal physics engine** re-built from the ground up, capable of simulating a wide range of materials and physical phenomena.
2. A **lightweight**, **ultra-fast**, **pythonic**, and **user-friendly** robotics simulation platform.
3. A powerful and fast **photo-realistic rendering system**.
4. A **generative data engine** that transforms user-prompted natural language description into various modalities of data.

Powered by a universal physics engine re-designed and re-built from the ground up, Genesis integrates various physics solvers and their coupling into a unified framework. This core physics engine is further enhanced by a generative agent framework that operates at an upper level, aiming towards fully automated data generation for robotics and beyond.

**Note**: Currently, we are open-sourcing the _underlying physics engine_ and the _simulation platform_. Our _generative framework_ is a modular system that incorporates many different generative modules, each handling a certain range of data modalities, routed by a high level agent. Some of the modules integrated existing papers and some are still under submission. Access to our generative feature will be gradually rolled out in the near future. If you are interested, feel free to explore more in the [paper list](#associated-papers) below.

Genesis aims to:

- **Lower the barrier** to using physics simulations, making robotics research accessible to everyone. See our [mission statement](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/mission.html).
- **Unify diverse physics solvers** into a single framework to recreate the physical world with the highest fidelity.
- **Automate data generation**, reducing human effort and letting the data flywheel spin on its own.

Project Page: <https://genesis-embodied-ai.github.io/>

## Key Features

- **Speed**: Over 43 million FPS when simulating a Franka robotic arm with a single RTX 4090 (430,000 times faster than real-time).
- **Cross-platform**: Runs on Linux, macOS, Windows, and supports multiple compute backends (CPU, Nvidia/AMD GPUs, Apple Metal).
- **Integration of diverse physics solvers**: Rigid body, MPM, SPH, FEM, PBD, Stable Fluid.
- **Wide range of material models**: Simulation and coupling of rigid bodies, liquids, gases, deformable objects, thin-shell objects, and granular materials.
- **Compatibility with various robots**: Robotic arms, legged robots, drones, *soft robots*, and support for loading `MJCF (.xml)`, `URDF`, `.obj`, `.glb`, `.ply`, `.stl`, and more.
- **Photo-realistic rendering**: Native ray-tracing-based rendering.
- **Differentiability**: Genesis is designed to be fully differentiable. Currently, our MPM solver and Tool Solver support differentiability, with other solvers planned for future versions (starting with rigid & articulated body solver).
- **Physics-based tactile simulation**: Differentiable [tactile sensor simulation](https://github.com/Genesis-Embodied-AI/DiffTactile) coming soon (expected in version 0.3.0).
- **User-friendliness**: Designed for simplicity, with intuitive installation and APIs.

## Quick Installation

Install **PyTorch** first following the [official instructions](https://pytorch.org/get-started/locally/).

Then, install Genesis via PyPI:
```bash
pip install genesis-world  # Requires Python >=3.9;
```

For the latest version, clone the repository and install locally:

```bash
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e .
```

## Docker

If you want to use Genesis from Docker, you can first build the Docker image as:

```bash
docker build -t genesis -f docker/Dockerfile docker
```

Then you can run the examples inside the docker image (mounted to `/workspace/examples`):

```bash
xhost +local:root # Allow the container to access the display

docker run --gpus all --rm -it \
-e DISPLAY=$DISPLAY \
-v /dev/dri:/dev/dri \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $PWD:/workspace \
genesis
```

## Documentation

Comprehensive documentation is available in [English](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html), [Chinese](https://genesis-world.readthedocs.io/zh-cn/latest/user_guide/index.html), and [Japanese](https://genesis-world.readthedocs.io/ja/latest/user_guide/index.html). This includes detailed installation steps, tutorials, and API references.

## Contributing to Genesis

The Genesis project is an open and collaborative effort. We welcome all forms of contributions from the community, including:

- **Pull requests** for new features or bug fixes.
- **Bug reports** through GitHub Issues.
- **Suggestions** to improve Genesis's usability.

Refer to our [contribution guide](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/.github/CONTRIBUTING.md) for more details.

## Support

- Report bugs or request features via GitHub [Issues](https://github.com/Genesis-Embodied-AI/Genesis/issues).
- Join discussions or ask questions on GitHub [Discussions](https://github.com/Genesis-Embodied-AI/Genesis/discussions).

## License and Acknowledgments

The Genesis source code is licensed under Apache 2.0.

Genesis's development has been made possible thanks to these open-source projects:

- [Taichi](https://github.com/taichi-dev/taichi): High-performance cross-platform compute backend. Kudos to the Taichi team for their technical support!
- [FluidLab](https://github.com/zhouxian/FluidLab): Reference MPM solver implementation.
- [SPH_Taichi](https://github.com/erizmr/SPH_Taichi): Reference SPH solver implementation.
- [Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) and [PBF3D](https://github.com/WASD4959/PBF3D): Reference PBD solver implementations.
- [MuJoCo](https://github.com/google-deepmind/mujoco): Reference for rigid body dynamics.
- [libccd](https://github.com/danfis/libccd): Reference for collision detection.
- [PyRender](https://github.com/mmatl/pyrender): Rasterization-based renderer.
- [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) and [LuisaRender](https://github.com/LuisaGroup/LuisaRender): Ray-tracing DSL.

## Associated Papers

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

If you use Genesis in your research, please consider citing:

```bibtex
@software{Genesis,
  author = {Genesis Authors},
  title = {Genesis: A Universal and Generative Physics Engine for Robotics and Beyond},
  month = {December},
  year = {2024},
  url = {https://github.com/Genesis-Embodied-AI/Genesis}
}
