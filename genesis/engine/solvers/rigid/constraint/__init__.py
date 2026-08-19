"""
Constraint solver submodule for rigid body simulation.

Contains constraint solving, island detection, and backward pass.
"""

from .solver import ConstraintSolver

# first declare func_solve_body:
from . import solver

# now register decomposed with func_solve_body:
from . import solver_breakdown

# AMDGPU func_solve_body variants use legacy ROCm signatures; re-enable after
# porting to dyn_state to match genesis-world main perf_dispatch prototype.
# from . import solver_amdgpu
