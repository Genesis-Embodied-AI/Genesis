"""
ComFree (Complementarity-Free) constraint solver for rigid body simulation.

Implements the analytical contact resolution method from:
  ComFree-Sim: A GPU-Parallelized Analytical Contact Physics Engine
  for Scalable Contact-Rich Robotics Simulation and Control
  (Borse et al., 2026, arXiv:2603.12185)

Instead of iterative complementarity-based solving (Newton/CG),
ComFree computes constraint forces in closed form via an impedance-style
prediction-correction update in the dual cone of Coulomb friction.
"""

from .solver import ComFreeSolver
