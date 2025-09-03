![Genesis](imgs/big_text.png)

![Teaser](imgs/teaser.png)

[![PyPI - Version](https://img.shields.io/pypi/v/genesis-world)](https://pypi.org/project/genesis-world/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/genesis-world)](https://pypi.org/project/genesis-world/)
[![GitHub Issues](https://img.shields.io/github/issues/Genesis-Embodied-AI/Genesis)](https://github.com/Genesis-Embodied-AI/Genesis/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/Genesis-Embodied-AI/Genesis)](https://github.com/Genesis-Embodied-AI/Genesis/discussions)
[![Discord](https://img.shields.io/discord/1322086972302430269?logo=discord)](https://discord.gg/nukCuhB47p)
<a href="https://drive.google.com/uc?export=view&id=1ZS9nnbQ-t1IwkzJlENBYqYIIOOZhXuBZ"><img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white" height="20" style="display:inline"></a>

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README.md)
[![README en Français](https://img.shields.io/badge/Francais-d9d9d9)](./README_FR.md)
[![한국어 README](https://img.shields.io/badge/한국어-d9d9d9)](./README_KR.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README_CN.md)
[![日本語版 README](https://img.shields.io/badge/日本語-d9d9d9)](./README_JA.md)

# Genesis 通用物理引擎

## 🔥 最新消息

  - [2025-08-05] 发布 v0.3.0 🎊 🎉
  - [2025-07-02] Genesis 的开发工作现已获得 [Genesis AI](https://genesis-ai.company/) 的官方支持。
  - [2025-01-09] 我们发布了一份关于 Genesis 的[详细性能基准测试和比较报告](https://github.com/zhouxian/genesis-speed-benchmark)，并附上所有测试脚本。
  - [2025-01-08] 发布 v0.2.1 🎊 🎉
  - [2025-01-08] 创建了 [Discord](https://discord.gg/nukCuhB47p) 和 [微信](https://drive.google.com/uc?export=view&id=1ZS9nnbQ-t1IwkzJlENBYqYIIOOZhXuBZ) 社区群。
  - [2024-12-25] 添加了支持光线追踪渲染器的 [docker](https://www.google.com/search?q=%23docker)。
  - [2024-12-24] 添加了[为 Genesis 做贡献](https://github.com/Genesis-Embodied-AI/Genesis/blob/main/.github/CONTRIBUTING.md)的指南。

## 目录

1. [概述](#概述)
2. [主要特点](#主要特点)
3. [快速入门](#快速入门)
4. [参与贡献](#参与贡献)
5. [帮助支持](#帮助支持)
6. [许可证与致谢](#许可证和致谢)
7. [相关论文](#genesis-背后的论文)
8. [引用](#引用)

## 概述

Genesis 是专为 *机器人/嵌入式 AI/物理 AI* 应用设计的通用物理平台，集成了以下核心功能：

- **通用物理引擎**: 从底层重建,支持多种材料和物理现象模拟
- **机器人模拟平台**: 轻量、高速、Python友好的开发环境
- **真实感渲染**: 内置光线追踪渲染系统
- **生成数据引擎**: 自然语言驱动的多模态数据生成

我们的长期使命:

- 降低物理模拟使用门槛
- 统一各类物理求解器
- 实现数据生成自动化

项目主页: <https://genesis-embodied-ai.github.io/>

## 主要特点

- **速度**：Genesis 提供了前所未有的模拟速度——在单个 RTX 4090 上模拟 Franka 机器人手臂时超过 4300 万 FPS（比实时快 430,000 倍）。
- **跨平台**：Genesis 原生运行在不同系统（Linux、MacOS、Windows）和不同计算后端（CPU、Nvidia GPU、AMD GPU、Apple Metal）上。
- **各种物理求解器的统一**：Genesis 开发了一个统一的模拟框架，集成了各种物理求解器：刚体、MPM、SPH、FEM、PBD、稳定流体。
- **支持广泛的材料模型**：Genesis 支持刚体和关节体、各种液体、气体现象、可变形物体、薄壳物体和颗粒材料的模拟（及其耦合）。
- **支持广泛的机器人**：机器人手臂、腿式机器人、无人机、*软体机器人*等，并广泛支持加载不同文件类型：`MJCF (.xml)`、`URDF`、`.obj`、`.glb`、`.ply`、`.stl` 等。
- **照片级真实感和高性能光线追踪器**：Genesis 支持基于光线追踪的原生渲染。
- **可微分性**：Genesis 设计为完全兼容可微分模拟。目前，我们的 MPM 求解器和工具求解器是可微分的，其他求解器的可微分性将很快添加（从刚体模拟开始）。
- **用户友好性**：Genesis 设计为尽可能简化模拟的使用。从安装到 API 设计，如果有任何您觉得不直观或难以使用的地方，请 [告诉我们](https://github.com/Genesis-Embodied-AI/Genesis/issues)。


## 快速安装

首先，请根据[官方指南](https://pytorch.org/get-started/locally/)安装 **PyTorch**。

然后，通过 PyPI 安装 Genesis：

```bash
pip install genesis-world  # 要求 Python>=3.10,<3.14;
```

要安装最新的版本，请先通过 `pip install --upgrade pip` 命令确保 `pip` 是最新版本，然后运行以下命令：

```bash
pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git
```

请注意，您仍需手动更新此软件包以与主分支保持同步。

如果用户希望编辑 Genesis 的源代码，我们鼓励以可编辑模式安装。首先，请确保已卸载 `genesis-world`，然后克隆代码仓库并在本地安装：

```bash
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis
pip install -e ".[dev]"
```

## Docker

如果您希望通过 Docker 使用 Genesis，可以先像这样构建 Docker 镜像：

```bash
docker build -t genesis -f docker/Dockerfile docker
```

然后您可以在 Docker 镜像中运行示例（示例文件挂载于 `/workspace/examples`）：

```bash
xhost +local:root # 允许容器访问显示设备

docker run --gpus all --rm -it \
-e DISPLAY=$DISPLAY \
-e LOCAL_USER_ID="$(id -u)" \
-v /dev/dri:/dev/dri \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-v $(pwd):/workspace \
--name genesis genesis:latest
```

### AMD 用户

AMD 用户可以使用 `docker/Dockerfile.amdgpu` 文件来使用 Genesis，该文件可通过运行以下命令构建：

```
docker build -t genesis-amd -f docker/Dockerfile.amdgpu docker
```

然后通过运行以下命令来使用：

```xhost +local:docker \
docker run -it --network=host \
 --device=/dev/kfd \
 --device=/dev/dri \
 --group-add=video \
 --ipc=host \
 --cap-add=SYS_PTRACE \
 --security-opt seccomp=unconfined \
 --shm-size 8G \
 -v $PWD:/workspace \
 -e DISPLAY=$DISPLAY \
 genesis-amd
```

示例文件将可以在 `/workspace/examples` 目录下找到。注意：AMD 用户应使用 Vulkan 后端。这意味着您需要调用 `gs.init(vulkan)` 来初始化 Genesis。

### 文档

- [英文文档](https://genesis-world.readthedocs.io/en/latest/user_guide/index.html)
- [中文文档](https://genesis-world.readthedocs.io/zh-cn/latest/user_guide/index.html)
- [日文文档](https://genesis-world.readthedocs.io/ja/latest/user_guide/index.html)

## 参与贡献

Genesis 项目的目标是构建一个完全透明、用户友好的生态系统，让来自机器人和计算机图形学的贡献者 **共同创建一个高效、真实（物理和视觉上）的虚拟世界，用于机器人研究及其他领域**。

我们真诚地欢迎来自社区的 *任何形式的贡献*，以使世界对机器人更友好。从 **新功能的拉取请求**、**错误报告**，到甚至是使 Genesis API 更直观的微小 **建议**，我们都全心全意地感谢！

## 帮助支持

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

```bibtex
@misc{Genesis,
  author = {Genesis Authors},
  title = {Genesis: A Generative and Universal Physics Engine for Robotics and Beyond},
  month = {December},
  year = {2024},
  url = {https://github.com/Genesis-Embodied-AI/Genesis}
}
```