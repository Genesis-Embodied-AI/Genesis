<div align="center">
  <img width="500px" src="https://github.com/Genesis-Embodied-AI/Genesis/blob/main/images/logo.png"/>
  
  ### A generative and simulated physical realm for general-purpose embodied-AI learning.
</div>

---
<div align="center">
  <img src="https://github.com/Genesis-Embodied-AI/Genesis/blob/main/images/demo.png"/>
</div>

# What is Genesis?

Genesis is a **generative** and **simulated** physical world for general-purpose robot learning, providing **unified** simulation platform supporting diverse range of materials, allowing simulating vast range of robotic tasks, meanwhile being **fully differentiable**.

Genesis is also a _**next-gen**_ simulation infrastructure that natively supports **generative simulation**: a future paradigm combining generative AI and physically-grounded simulation, aiming for unlocking _**infinite and diverse data**_ for robotic agents to learn vast range of skills across diverse environments like never before.

Genesis is still under development, and will be made publicly available soon.

# What can Genesis do?
Genesis differs from prior simulation platforms with a number of distinct key features:

## :white_check_mark: :tshirt: Support for diverse range of materials and their coupling
Genesis supports physics simulation of a wide range of materials encountered in humans and (future) robots' daily life, and coupling (interaction) between these objects, including:
- :door: Rigid and articulated bodies
- :sweat_drops: Liquid: Newtonian, non-Newtonian, viscosity, surface tension, etc.
- :dash: Gaseous phenomenon: air flow, heat flow, etc.
- :dumpling: Deformable objects: elastic, plastic, elasto-plastic
- :shirt: Thin-shell objects: :worm: ropes, :jeans: cloths, :page_facing_up: papers, :black_joker: cards, etc.
- :hourglass_flowing_sand: Granular objects: sand, beans, etc.

## :white_check_mark: :robot: Support for diverse types of robots
Genesis supports simulating of vast range of robots, including 
  - 🦾 Robot arm
  - 🦿 Legged robot
  - :writing_hand: Dexterous hand
  - 🖲️ Mobile robot
  - 🚁 Drone
  - :lizard: Soft robot

Note that Genesis is the first-ever platform providing comprehensive support for **soft muscles** and **soft robot**, as well as their interaction with rigid robots. Genesis also ships with a URDF-like soft-robot configuration system.

## :white_check_mark: 🚀 Support for diverse physics backend solvers
Various physics-based solvers tailored to different materials and needs  have been developed in the past decades. Some prioritize high **simulation fidelity**, while others favor **performance**, albeit sometimes sacrificing accuracy.

Genesis, in contrast to its predecessors, natively supports a wide range of different physics solvers. Users are able to effortlessly toggle between solvers, depending on their specific requirements.

Our current supported solvers include:
  - Material Point Method (MPM)
  - Finite Element Method (FEM)
  - Position Based Dynamics (PBD)
  - Smoothed-Particle Hydrodynamics (SPH)
  - Articulated Body Algorithm (ABA)-based Rigid Body Dynamics

We also provide contact resolving via multiple methods:
  - Convex mesh
  - SDF-based non-convex contact
  - Incremental Potential Contact (IPC)


## :white_check_mark: :pinching_hand: Physically-accurate tactile sensor simulation, applicable to diverse material types
Genesis is the first platform that integrate physics-based simulation of GelSight-type tacile sensors, providing dense state-based and RGB-based tactile feedback simulation when handling diverse range of materials.


## :white_check_mark: :camera_flash: Ultra-fast ray-tracing based renderer
Genesis provides both rasterization-based and ray-tracing-based rendering pipeline. Our ray tracer is ultra fast, providing real-time photo-realistic rendering at RL-sufficient resolution. Notably, users can effortlessly switch between different rendering backend with one line of code.

## :white_check_mark: :boom: GPU-accelerated and fully differentiable
Genesis supports massive parallelization over GPUs, and supports efficient gradient checkpointing for differentiable simulation.
Genesis is internally powered by [**Taichi**](https://github.com/taichi-dev/taichi), but users are shielded from any potential debugging intricacies associated with Taichi. Genesis implements a custom Tensor system (`Genesis.Tensor`) seamlessly integrated with PyTorch. This integration guarantees that the Genesis simulation pipeline mirrors the operation of a standard PyTorch neural network in both functionality and familiarity. Provided that input variables are `Genesis.Tensor`, users are able to compute custom loss using simulation outputs, and one simple call of `loss.backward()` will trigger gradient flow all the way back through time and back to input variables, allowing effortless gradient-based policy optimization.


## :white_check_mark: :baby: Optimized ease of use
The rule of thumb we kept in mind is to make Genesis as user-friendly and intuitive as possible, ensuring a seamless experience even for those new to the realm of simulations. Genesis is fully embedded in Python. With **one single line of code**, users are able to switch between different physics backend, or switching from rasterization to ray-tracing for rendering photorealistic visuals.

## :white_check_mark: :milky_way: Most importantly, native support for [Generative Simulation](https://arxiv.org/abs/2305.10455)
Genesis, while being a powerful simulation infrastructure, natively embraces the upcoming paradigm of [Generative Simulation](https://arxiv.org/abs/2305.10455). Powered by generative AIs and physics-based simulations, generative simulation is a paradigm that automates generation of diverse environments, robots, tasks, skills, training supervisions (e.g. reward functions), thereby generating infinite and diverse training data and scaling up diverse skill learning for embodied AI agents, in a fully automated manner.

Genesis provides a set of APIs for generating diverse targets, ranging from **interactable** and **actionable** environments, to different tasks, robots, and ultimately vast skills.

Users will be able to do the following:
```Python
import genesis as gs
# generate environments
gs.generate('A robotic arm operating in a typical kitchen environment.')

# generate robot
gs.generate('A 6-DoF robotic arm with a mobile base, a camera attached to its wrist, and a tactile sensor on its gripper.')
gs.generate('A bear-like soft robot walking in the sand.')

# generate skills
gs.generate('A Franka arm tossing trashes into a trashcan')
gs.generate('A UR-5 arm bends a noodle into a U-shape')

# or, let it generate on its own
gs.generate('A random robot learning a random but meaningful skill')
# or simply let's it surprise you
gs.generate()
```


## Stay tuned :P
Genesis is a joint effort between multiple universities and industrial partners, and is still under active development. The Genesis team is diligently working towards an alpha release soon. Stay tuned for updates!
