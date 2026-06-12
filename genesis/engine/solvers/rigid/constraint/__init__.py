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

# AMDGPU-specific wave-cooperative variants of func_solve_body. These recover
# the AMD CG-solver perf path (tiled wave-coop + wavecoop monolith) that beats
# the generic func_solve_body_monolith on gfx942 by ~1.5x at high env counts.
#
# Ported to the upstream-v1.0.0 func_solve_body signature, which inserted
# `dofs_info` as the 2nd positional arg (the pre-merge fork variants were 6-arg
# and failed perf_dispatch registration with PERFDISPATCH_ANNOTATION_SEQUENCE_MISMATCH).
# The self-contained wavecoop / tiled_wc / lifted-loop variants are active; the
# split / decomposed variants that depended on upstream device functions removed
# or re-signed by the merge are gated off inside solver_amdgpu.py.
#
# Enabled by default; opt out with GS_ENABLE_AMDGPU_SOLVER_VARIANTS=0.
import os as _os

if _os.environ.get("GS_ENABLE_AMDGPU_SOLVER_VARIANTS", "1") == "1":
    from . import solver_amdgpu  # noqa: F401
