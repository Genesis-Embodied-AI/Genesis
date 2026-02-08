"""
Collider submodule for collision detection.

Contains broad-phase, narrow-phase, contact management, and geometric algorithms.

Modules:
- collider: Main Collider class
- broadphase: AABB and sweep-and-prune algorithms
- narrowphase: SDF, convex-convex, terrain collision
- box_contact: Specialized box collision (plane-box, box-box)
- contact: Contact management utilities
- gjk: Gilbert-Johnson-Keerthi algorithm
- epa: Expanding Polytope Algorithm
- multi_contact: Multi-contact detection (polygon clipping)
- mpr: Minkowski Portal Refinement
"""

from .collider import Collider
from .broadphase import *
from .narrowphase import *
from .box_contact import *
from .contact import *
