"""
Constraint solver submodule for rigid body simulation.

Contains constraint solving, island detection, and backward pass.
"""

from .solver import ConstraintSolver
from .solver_island import ConstraintSolverIsland

# first declare func_solve_body:
from . import solver

# now register decomposed with func_solve_body:
from . import solver_breakdown
