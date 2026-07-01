"""
Constants and enums for the collider module.
"""

from enum import IntEnum

# How far the origin's projection may extrapolate beyond the portal triangle, as a fraction of the triangle
# (barycentric), before the infinite-plane penetration is deemed an unreliable extrapolation (portal INVALID
# -> refine with GJK).
# FIXME: This is a compile-time constant instead of an MPRInfo scalar field because one extra field read pushes
# '_func_narrowphase_multicontact' past Metal's limit of 31 buffer bindings per kernel. Move it back to MPRInfo
# once quadrants packs root buffers below that limit (e.g. via Metal argument buffers).
CCD_EXTRAPOLATION_TOL = 1.0


class RETURN_CODE(IntEnum):
    """
    Return codes for the general subroutines used in GJK and EPA algorithms.
    """

    SUCCESS = 0
    FAIL = 1


class GJK_RETURN_CODE(IntEnum):
    """
    Return codes for the GJK algorithm.
    """

    SEPARATED = 0
    INTERSECT = 1
    NUM_ERROR = 2


class PORTAL_STATUS(IntEnum):
    """
    Reliability of the MPR portal left in simplex_support after a contact, driving whether GJK must refine the result
    and whether the portal may be reused (perturbation reconstruction, EPA seeding).

    INVALID: unconverged (hit the iteration cap), or a degenerate sliver triangle whose origin projects outside. The
    penetration/normal are unreliable, so the contact must be refined by GJK.
    DEGENERATED: the portal is not a trustworthy exact contact face - either the origin projects outside an otherwise
    well-formed portal (the depth is only a lower-bound estimate, Theorem 4.3) or the contact came from a degenerate
    touch/segment path with no refined portal. Not reusable, but trusted enough not to force a GJK refine.
    VALID: converged and the origin projects inside the portal triangle, so the depth is exact (Theorem 4.2). Reliable
    and reusable.
    """

    INVALID = 0
    DEGENERATED = 1
    VALID = 2


class EPA_POLY_INIT_RETURN_CODE(IntEnum):
    """
    Return codes for the EPA polytope initialization.
    """

    SUCCESS = 0
    P2_NONCONVEX = 1
    P2_FALLBACK3 = 2
    P3_BAD_NORMAL = 3
    P3_INVALID_V4 = 4
    P3_INVALID_V5 = 5
    P3_MISSING_ORIGIN = 6
    P3_ORIGIN_ON_FACE = 7
    P4_MISSING_ORIGIN = 8
    P4_FALLBACK3 = 9
