"""
Parametric NEMA plug/receptacle generator for robot-insertion simulation.

Generates watertight, simulation-ready OBJ pairs (plug + socket) modeled on
NEMA WD-6 connector dimensions:
  - nema_1_15 : 2-prong US plug (two flat blades)
  - nema_5_15 : 3-prong US plug (two blades + round ground pin)

Conventions (matching the AutoMate dataset):
  - units: meters
  - plug and socket are exported in a SHARED assembled-state frame
    (spawn both at the same world pose -> fully inserted)
  - socket base sits at z = 0

Simulation-friendly features:
  - per-side clearance is a parameter (default 0.25 mm; tighten later)
  - lead-in chamfers on every socket opening (funnel for RL exploration)
  - tapered blade/pin tips (like real molded plugs)
  - everything verified: watertightness, zero intersection volume when
    assembled, collision-free straight-line insertion sweep, measured min gap

Requires: trimesh + manifold3d   (pip install trimesh manifold3d)
"""

import json
import os

import numpy as np
import trimesh
from trimesh.transformations import translation_matrix as T

MM = 1e-3  # build in mm, export in meters

# ------------------------------------------------------------------ parameters
CLEARANCE = 3.0           # mm, per side (0.1 = near-real ISO-ish, 0.5 = easy, 3.0 = robot-insertion tolerant)
BLADE_W = 6.35            # mm blade width  (0.25 in, WD-6)
BLADE_T = 1.524           # mm blade thickness (0.06 in)
BLADE_L = 16.0            # mm exposed blade length (~5/8 in)
BLADE_SPACING = 12.7      # mm center-to-center (0.5 in)
PIN_D = 4.75              # mm ground pin diameter (0.187 in)
PIN_EXTRA = 1.5           # mm ground pin longer than blades (ground-first)
PIN_OFFSET = 11.91        # mm pin center to blade centerline (0.469 in)
TIP_CHAMFER = 1.5         # mm tapered tip height on blades/pin
LEAD_IN = 1.2             # mm chamfer widening per side at socket openings
LEAD_DEPTH = 1.6          # mm chamfer depth into the face
SLOT_MARGIN = 2.0         # mm slot deeper than blade

SOCKET_BLOCK = (40.0, 40.0, 22.0)   # like the IndustReal tray blocks
BODY_2P = (28.0, 18.0, 22.0)        # molded plug body (grasp target)
BODY_3P = (30.0, 22.0, 24.0)

SEC = 64  # cylinder facets


def boo(op, meshes):
    fn = trimesh.boolean.union if op == "u" else trimesh.boolean.difference
    m = fn(meshes, engine="manifold")
    return m


def box(ext, center):
    return trimesh.creation.box(extents=ext, transform=T(center))


def frustum_rect(w0, h0, w1, h1, z0, z1, cx, cy):
    """Convex hull between rect (w0,h0)@z0 and rect (w1,h1)@z1, centered (cx,cy)."""
    p = []
    for (w, h, z) in [(w0, h0, z0), (w1, h1, z1)]:
        p += [[cx + sx * w / 2, cy + sy * h / 2, z] for sx in (-1, 1) for sy in (-1, 1)]
    return trimesh.convex.convex_hull(np.array(p))


def frustum_circ(r0, r1, z0, z1, cx, cy, n=SEC):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = [[cx + r0 * np.cos(t), cy + r0 * np.sin(t), z0] for t in a]
    p += [[cx + r1 * np.cos(t), cy + r1 * np.sin(t), z1] for t in a]
    return trimesh.convex.convex_hull(np.array(p))


def blade(cx, cy, z_top, L):
    """Flat blade pointing down from z_top, tapered tip."""
    body = box((BLADE_T, BLADE_W, L - TIP_CHAMFER),
               (cx, cy, z_top - (L - TIP_CHAMFER) / 2))
    z0 = z_top - L + TIP_CHAMFER
    tip = frustum_rect(BLADE_T - 1.0, BLADE_W - 1.0, BLADE_T, BLADE_W,
                       z_top - L, z0 + 0.01, cx, cy)
    return boo("u", [body, tip])


def pin(cx, cy, z_top, L):
    body = trimesh.creation.cylinder(radius=PIN_D / 2, height=L - TIP_CHAMFER,
                                     sections=SEC,
                                     transform=T((cx, cy, z_top - (L - TIP_CHAMFER) / 2)))
    tip = frustum_circ(PIN_D / 2 - 0.5, PIN_D / 2, z_top - L, z_top - L + TIP_CHAMFER + 0.01,
                       cx, cy)
    return boo("u", [body, tip])


def make_pair(three_prong: bool, out_dir: str, name: str):
    Xs, Ys, Zs = SOCKET_BLOCK
    face_z = Zs
    yb = 4.0 if three_prong else 0.0           # blade row y (shift up if pin present)
    yp = yb - PIN_OFFSET                        # pin y
    bx = BLADE_SPACING / 2
    c = CLEARANCE

    # ------------------------------------------------------------------ socket
    block = box((Xs, Ys, Zs), (0, 0, Zs / 2))
    cut = []
    slot_depth = BLADE_L + SLOT_MARGIN
    for sx in (-1, 1):
        cut.append(box((BLADE_T + 2 * c, BLADE_W + 2 * c, 2 * slot_depth),
                       (sx * bx, yb, face_z)))                       # through-ish slot
        cut.append(frustum_rect(BLADE_T + 2 * c, BLADE_W + 2 * c,
                                BLADE_T + 2 * c + 2 * LEAD_IN, BLADE_W + 2 * c + 2 * LEAD_IN,
                                face_z - LEAD_DEPTH, face_z + 0.2, sx * bx, yb))
    if three_prong:
        pin_depth = BLADE_L + PIN_EXTRA + SLOT_MARGIN
        cut.append(trimesh.creation.cylinder(radius=PIN_D / 2 + c, height=2 * pin_depth,
                                             sections=SEC, transform=T((0, yp, face_z))))
        cut.append(frustum_circ(PIN_D / 2 + c, PIN_D / 2 + c + LEAD_IN,
                                face_z - LEAD_DEPTH, face_z + 0.2, 0, yp))
    socket = boo("d", [block] + cut)

    # -------------------------------------------------------------------- plug
    BX, BY, BZ = BODY_3P if three_prong else BODY_2P
    body = box((BX, BY, BZ), (0, yb if three_prong else yb, face_z + BZ / 2))
    # small chamfered top so the gripper has a distinct feature
    top = frustum_rect(BX, BY, BX - 6, BY - 6, face_z + BZ, face_z + BZ + 4, 0, yb)
    parts = [body, top,
             blade(-bx, yb, face_z + 0.5, BLADE_L + 0.5),
             blade(+bx, yb, face_z + 0.5, BLADE_L + 0.5)]
    if three_prong:
        parts.append(pin(0, yp, face_z + 0.5, BLADE_L + PIN_EXTRA + 0.5))
    plug = boo("u", parts)

    # --------------------------------------------------------------- verify
    for label, m in [("plug", plug), ("socket", socket)]:
        assert m.is_watertight, f"{name} {label} not watertight"
    inter = trimesh.boolean.intersection([plug, socket], engine="manifold")
    assert inter.is_empty or inter.volume < 1e-6, \
        f"{name}: assembled-state penetration! vol={inter.volume}"

    lift = BLADE_L + (PIN_EXTRA if three_prong else 0) + 2.0
    for t in np.linspace(0, lift, 9):
        p2 = plug.copy(); p2.apply_translation((0, 0, t))
        iv = trimesh.boolean.intersection([p2, socket], engine="manifold")
        assert iv.is_empty or iv.volume < 1e-6, f"{name}: sweep collision at z=+{t:.2f}mm"

    # measured min gap: sample blade side faces vs socket surface
    pts = plug.sample(4000)
    pts = pts[pts[:, 2] < face_z - 1.0]          # only the inserted portion
    if len(pts):
        d = trimesh.proximity.ProximityQuery(socket).signed_distance(pts)
        min_gap = float(-d.max())                # sd>0 would mean inside socket solid
        assert min_gap > 0, f"{name}: sampled point inside socket solid"
    else:
        min_gap = float("nan")

    # --------------------------------------------------------------- export
    os.makedirs(f"{out_dir}/mesh/{name}", exist_ok=True)
    os.makedirs(f"{out_dir}/urdf", exist_ok=True)
    meta = {"clearance_mm": c, "min_sampled_gap_mm": round(min_gap, 4),
            "disassembly_dist_m": round(lift * MM, 5),
            "blade": {"w_mm": BLADE_W, "t_mm": BLADE_T, "l_mm": BLADE_L,
                      "spacing_mm": BLADE_SPACING},
            "frame": "shared assembled frame; socket base at z=0; units meters"}
    for label, m in [("plug", plug), ("socket", socket)]:
        m = m.copy(); m.apply_scale(MM)
        # densify for SDF-based collision (a la Factory's *_subdiv meshes);
        # uniform midpoint subdivision keeps the mesh watertight
        while len(m.faces) < 4000:
            m = m.subdivide()
        assert m.is_watertight
        m.export(f"{out_dir}/mesh/{name}/asset_{label}.obj")
        with open(f"{out_dir}/urdf/{name}_{label}.urdf", "w") as f:
            f.write(f"""<?xml version="1.0"?>
<robot name="{name}_{label}">
  <link name="{name}_{label}">
    <visual><geometry><mesh filename="../mesh/{name}/asset_{label}.obj"/></geometry></visual>
    <collision><geometry><mesh filename="../mesh/{name}/asset_{label}.obj"/></geometry></collision>
  </link>
</robot>
""")
    with open(f"{out_dir}/mesh/{name}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{name}: OK  watertight, no penetration, sweep clean | "
          f"min sampled gap {min_gap:.3f} mm (target {c}) | lift {lift:.1f} mm | "
          f"plug {len(plug.faces)} tris, socket {len(socket.faces)} tris")
    return plug, socket, lift


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    make_pair(False, out, "nema_1_15")
    make_pair(True,  out, "nema_5_15")
