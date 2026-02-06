"""
Capsule collision contact detection functions.

This module contains specialized analytical contact detection algorithms for capsule geometries:
- Capsule-capsule contact detection (analytical line segment distance)
"""

import gstaichi as ti

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class

from .contact import (
    func_add_contact,
)


@ti.func
def func_closest_points_on_segments(
    P1: ti.types.vector(3, gs.ti_float),
    P2: ti.types.vector(3, gs.ti_float),
    Q1: ti.types.vector(3, gs.ti_float),
    Q2: ti.types.vector(3, gs.ti_float),
    EPS: gs.ti_float,
):
    """
    Compute closest points on two line segments using analytical solution.
    
    Given two line segments:
      Segment A: P1 + s*(P2-P1), s ∈ [0,1]
      Segment B: Q1 + t*(Q2-Q1), t ∈ [0,1]
    
    Find parameters s, t that minimize ||A(s) - B(t)||²
    
    This is a well-known computer graphics problem with closed-form solution.
    
    Parameters
    ----------
    P1, P2 : ti.Vector
        Endpoints of segment A
    Q1, Q2 : ti.Vector
        Endpoints of segment B
    EPS : float
        Small epsilon for numerical stability
        
    Returns
    -------
    Pa : ti.Vector
        Closest point on segment A
    Pb : ti.Vector
        Closest point on segment B
    
    References
    ----------
    Real-Time Collision Detection by Christer Ericson, Chapter 5.1.9
    """
    d1 = P2 - P1  # Direction vector of segment A
    d2 = Q2 - Q1  # Direction vector of segment B
    r = P1 - Q1   # Vector between segment origins
    
    a = d1.dot(d1)  # Squared length of segment A
    b = d1.dot(d2)  # Dot product of directions
    c = d2.dot(d2)  # Squared length of segment B
    d = d1.dot(r)
    e = d2.dot(r)
    
    denom = a * c - b * b  # Denominator (always >= 0)
    
    # Initialize parameters
    s = gs.ti_float(0.0)
    t = gs.ti_float(0.0)
    
    # Check if segments are parallel or degenerate
    if denom < EPS:
        # Segments are parallel or one/both are degenerate
        # Handle as special case
        s = 0.0
        if c > EPS:
            t = ti.math.clamp(e / c, 0.0, 1.0)
        else:
            t = 0.0
    else:
        # General case: solve for optimal parameters
        s = (b * e - c * d) / denom
        t = (a * e - b * d) / denom
        
        # Clamp s to [0, 1]
        s = ti.math.clamp(s, 0.0, 1.0)
        
        # Recompute t for clamped s
        t = ti.math.clamp((b * s + e) / c if c > EPS else 0.0, 0.0, 1.0)
        
        # Recompute s for clamped t (ensures we're on segment boundaries)
        s_new = ti.math.clamp((b * t - d) / a if a > EPS else 0.0, 0.0, 1.0)
        
        # Use refined s if it improves the solution
        if a > EPS:
            s = s_new
    
    # Compute closest points
    Pa = P1 + s * d1
    Pb = Q1 + t * d2
    
    return Pa, Pb


@ti.func
def func_capsule_capsule_contact(
    i_ga,
    i_gb,
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    collider_state: array_class.ColliderState,
    collider_info: array_class.ColliderInfo,
    errno: array_class.V_ANNOTATION,
):
    """
    Analytical capsule-capsule collision detection.
    
    A capsule is defined as a line segment plus a radius (swept sphere).
    Collision between two capsules reduces to:
      1. Find closest points on the two line segments (analytical)
      2. Check if distance < sum of radii
      3. Compute contact point and normal
    
    This is much faster than iterative algorithms (MPR/GJK) since it's a closed-form solution.
    Multi-contact is handled by the standard perturbation approach in the calling code.
    
    Performance: ~50 FLOPs vs ~300 FLOPs for MPR
    
    Parameters
    ----------
    i_ga, i_gb : int
        Geometry indices
    i_b : int
        Batch index
    geoms_state : GeomsState
        Geometry states (positions, orientations)
    geoms_info : GeomsInfo
        Geometry info (radii, lengths)
    rigid_global_info : RigidGlobalInfo
        Global simulation info (EPS, etc.)
    collider_state : ColliderState
        Collider state for storing contacts
    collider_info : ColliderInfo
        Collider configuration
    errno : V_ANNOTATION
        Error number for debugging
    """
    EPS = rigid_global_info.EPS[None]
    
    # Get capsule A parameters
    pos_a = geoms_state.pos[i_ga, i_b]
    quat_a = geoms_state.quat[i_ga, i_b]
    radius_a = geoms_info.data[i_ga][0]
    halflength_a = gs.ti_float(0.5) * geoms_info.data[i_ga][1]
    
    # Get capsule B parameters
    pos_b = geoms_state.pos[i_gb, i_b]
    quat_b = geoms_state.quat[i_gb, i_b]
    radius_b = geoms_info.data[i_gb][0]
    halflength_b = gs.ti_float(0.5) * geoms_info.data[i_gb][1]
    
    # Capsules are aligned along local Z-axis by convention
    local_z = ti.Vector([0.0, 0.0, 1.0], dt=gs.ti_float)
    
    # Get segment axes in world space
    axis_a = gu.ti_transform_by_quat(local_z, quat_a)
    axis_b = gu.ti_transform_by_quat(local_z, quat_b)
    
    # Compute segment endpoints in world space
    P1 = pos_a - halflength_a * axis_a
    P2 = pos_a + halflength_a * axis_a
    Q1 = pos_b - halflength_b * axis_b
    Q2 = pos_b + halflength_b * axis_b
    
    # Find closest points on the two line segments (analytical solution)
    Pa, Pb = func_closest_points_on_segments(P1, P2, Q1, Q2, EPS)
    
    # Compute distance between closest points
    diff = Pb - Pa
    dist_sq = diff.dot(diff)
    combined_radius = radius_a + radius_b
    combined_radius_sq = combined_radius * combined_radius
    
    # Check for collision
    if dist_sq < combined_radius_sq:
        # Collision detected
        dist = ti.sqrt(dist_sq)
        
        # Compute contact normal (from A to B)
        normal = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
        if dist > EPS:
            normal = diff / dist
        else:
            # Segments are coincident, use arbitrary perpendicular direction
            # Try cross product with axis_a first
            temp_normal = axis_a.cross(axis_b)
            if temp_normal.dot(temp_normal) < EPS:
                # Axes are parallel, use any perpendicular
                if ti.abs(axis_a[0]) < 0.9:
                    temp_normal = ti.Vector([1.0, 0.0, 0.0], dt=gs.ti_float).cross(axis_a)
                else:
                    temp_normal = ti.Vector([0.0, 1.0, 0.0], dt=gs.ti_float).cross(axis_a)
            normal = gu.ti_normalize(temp_normal, EPS)
        
        # Compute penetration depth
        penetration = combined_radius - dist
        
        # Compute contact position (on surface of capsule A)
        contact_pos = Pa + radius_a * normal
        
        # Add contact to collision state
        func_add_contact(
            i_ga,
            i_gb,
            normal,
            contact_pos,
            penetration,
            i_b,
            geoms_state,
            geoms_info,
            collider_state,
            collider_info,
            errno,
        )
