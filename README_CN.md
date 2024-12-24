![Genesis](imgs/big_text.png)

![Teaser](imgs/teaser.png)

[![PyPI - Version](https://img.shields.io/pypi/v/genesis-world)](https://pypi.org/project/genesis-world/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/genesis-world)](https://pypi.org/project/genesis-world/)
[![GitHub Issues](https://img.shields.io/github/issues/Genesis-Embodied-AI/Genesis)](https://github.com/Genesis-Embodied-AI/Genesis/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/Genesis-Embodied-AI/Genesis)](https://github.com/Genesis-Embodied-AI/Genesis/discussions)

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README_CN.md)

# 概述

Genesis 是一个为通用 *机器人/嵌入式 AI/物理 AI* 应用设计的物理平台。它同时具备多种功能：

1. 一个从头开始重建的 **通用物理引擎**，能够模拟各种材料和物理现象。
2. 一个 **轻量级**、**超快**、**Python 风格** 和 **用户友好** 的机器人模拟平台。
3. 一个强大且快速的 **照片级真实感渲染系统**。
4. 一个 **生成数据引擎**，将用户提示的自然语言描述转换为各种数据模式。

Genesis 由一个重新设计和重建的通用物理引擎驱动，集成了各种物理求解器及其耦合到一个统一的框架中。这个核心物理引擎通过一个生成代理框架在上层进行增强，旨在实现机器人及其他领域的完全 **自动化数据生成**。

目前，我们正在开源 **底层物理引擎和模拟平台**。我们的生成框架是一个模块化系统，包含许多不同的生成模块，每个模块处理一定范围的数据模式，由一个高级代理路由。一些模块集成了现有的论文，一些仍在提交中。我们的生成功能将逐步开放。如果您感兴趣，请参阅下面的 [论文列表](#papers-behind-genesis)。

Genesis 的构建和持续发展基于以下 ***长期使命***：

1. **降低使用物理模拟的门槛**，使机器人研究对所有人都可访问。（参见我们的 [承诺](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/mission.html)）
2. **将各种最先进的物理求解器统一到一个框架中**，使用最先进的模拟技术在虚拟领域中重新创建整个物理世界，达到最高的物理、视觉和感官保真度。
3. **最小化人类在收集和生成机器人及其他领域数据上的努力**，让数据飞轮自行旋转。

项目页面：<https://genesis-embodied-ai.github.io/>

## 主要特点

- **速度**：Genesis 提供了前所未有的模拟速度——在单个 RTX 4090 上模拟 Franka 机器人手臂时超过 4300 万 FPS（比实时快 430,000 倍）。
- **跨平台**：Genesis 原生运行在不同系统（Linux、MacOS、Windows）和不同计算后端（CPU、Nvidia GPU、AMD GPU、Apple Metal）上。
- **各种物理求解器的统一**：Genesis 开发了一个统一的模拟框架，集成了各种物理求解器：刚体、MPM、SPH、FEM、PBD、稳定流体。
- **支持广泛的材料模型**：Genesis 支持刚体和关节体、各种液体、气体现象、可变形物体、薄壳物体和颗粒材料的模拟（及其耦合）。
- **支持广泛的机器人**：机器人手臂、腿式机器人、无人机、*软体机器人*等，并广泛支持加载不同文件类型：`MJCF (.xml)`、`URDF`、`.obj`、`.glb`、`.ply`、`.stl` 等。
- **照片级真实感和高性能光线追踪器**：Genesis 支持基于光线追踪的原生渲染。
- **可微分性**：Genesis 设计为完全兼容可微分模拟。目前，我们的 MPM 求解器和工具求解器是可微分的，其他求解器的可微分性将很快添加（从刚体模拟开始）。
- **基于物理的触觉传感器**：Genesis 包含一个基于物理的可微分 [触觉传感器模拟模块](https://github.com/Genesis-Embodied-AI/DiffTactile)。这将很快集成到公共版本中（预计在 0.3.0 版本中）。
- **用户友好性**：Genesis 设计为尽可能简化模拟的使用。从安装到 API 设计，如果有任何您觉得不直观或难以使用的地方，请 [告诉我们](https://github.com/Genesis-Embodied-AI/Genesis/issues)。

## 入门

### 快速安装

Genesis 可通过 PyPI 获取：

```bash
pip install genesis-world  # 需要 Python >=3.9;
```

您还需要按照 [官方说明](https://pytorch.org/get-started/locally/) 安装 **PyTorch**。

### 文档

请参阅我们的 [文档网站（英文）](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html)/[（中文）](https://genesis-world.readthedocs.io/zh-cn/latest/user_guide/index.html)以获取详细的安装步骤、教程和 API 参考。

## 贡献 Genesis

Genesis 项目的目标是构建一个完全透明、用户友好的生态系统，让来自机器人和计算机图形学的贡献者 **共同创建一个高效、真实（物理和视觉上）的虚拟世界，用于机器人研究及其他领域**。

我们真诚地欢迎来自社区的 *任何形式的贡献*，以使世界对机器人更友好。从 **新功能的拉取请求**、**错误报告**，到甚至是使 Genesis API 更直观的微小 **建议**，我们都全心全意地感谢！

## 支持

- 请使用 Github [Issues](https://github.com/Genesis-Embodied-AI/Genesis/issues) 报告错误和提出功能请求。

- 请使用 GitHub [Discussions](https://github.com/Genesis-Embodied-AI/Genesis/discussions) 讨论想法和提问。

## 许可证和致谢

Genesis 源代码根据 Apache 2.0 许可证授权。
没有这些令人惊叹的开源项目，Genesis 的开发是不可能的：

- [Taichi](https://github.com/taichi-dev/taichi)：提供高性能跨平台计算后端。感谢 taichi 的所有成员提供的技术支持！
- [FluidLab](https://github.com/zhouxian/FluidLab) 提供参考 MPM 求解器实现
- [SPH_Taichi](https://github.com/erizmr/SPH_Taichi) 提供参考 SPH 求解器实现
- [Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) 和 [PBF3D](https://github.com/WASD4959/PBF3D) 提供参考 PBD 求解器实现
- [MuJoCo](https://github.com/google-deepmind/mujoco) 和 [Brax](https://github.com/google/brax) 提供刚体动力学参考
- [libccd](https://github.com/danfis/libccd) 提供碰撞检测参考
- [PyRender](https://github.com/mmatl/pyrender) 提供基于光栅化的渲染器
- [LuisaCompute](https://github.com/LuisaGroup/LuisaCompute) 和 [LuisaRender](https://github.com/LuisaGroup/LuisaRender) 提供其光线追踪 DSL
- [trimesh](https://github.com/mikedh/trimesh)、[PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) 和 [CoACD](https://github.com/SarahWeiii/CoACD) 提供几何处理

## Genesis 背后的论文

Genesis 是一个大规模的努力，将各种现有和正在进行的研究工作的最先进技术集成到一个系统中。这里我们列出了一些对 Genesis 项目有贡献的论文（非详尽列表）：

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

... 以及许多正在进行的工作。

## 引用

如果您在研究中使用了 Genesis，我们将非常感谢您引用它。我们仍在撰写技术报告，在其公开之前，您可以考虑引用：

```
@software{Genesis,
  author = {Genesis Authors},
  title = {Genesis: A Universal and Generative Physics Engine for Robotics and Beyond},
  month = {December},
  year = {2024},
  url = {https://github.com/Genesis-Embodied-AI/Genesis}
}
```
