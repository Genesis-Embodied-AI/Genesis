"""
ComFree (Complementarity-Free) constraint solver for rigid body simulation.

Implements the analytical contact resolution method from:
  ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine
  for Scalable Contact-Rich Robotics Simulation and Control
  (arXiv:2603.12185, reference impl: https://github.com/asu-iris/comfree_warp)

Instead of iterative complementarity-based solving (Newton/CG), ComFree computes
constraint forces in closed form via an impedance-style prediction-correction
update, then solves a single mass-weighted linear system for the acceleration.
"""

from .solver import ComFreeSolver
