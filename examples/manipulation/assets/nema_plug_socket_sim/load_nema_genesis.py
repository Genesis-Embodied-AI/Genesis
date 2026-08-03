"""
Load an AutoMate plug/socket assembly in Genesis.

Assets: AutoMate (Tang et al., RSS 2024), 100 plug/socket pairs, extracted from
the `automate` branch of isaac-sim/IsaacGymEnvs (assets/automate/).
Both OBJs of a pair are authored in a SHARED "assembled-state" frame:
spawning plug and socket at the same world pose = fully inserted.

data/disassembly_dist.json  -> per-assembly lift distance to separate plug from socket
data/plug_grasps.json       -> per-plug grasp pose samples (for a Franka gripper)

Tested layout (keep it):
  mesh/<ID>/asset_plug.obj, asset_socket.obj
  urdf/<ID>_plug.urdf, <ID>_socket.urdf     (optional; script loads OBJs directly)

NOTE: written against the open-source Genesis Python API (gs.init / gs.Scene /
gs.morphs.Mesh). Argument names occasionally shift between Genesis versions --
if a kwarg is rejected, check `help(gs.morphs.Mesh)` and the comments below.
"""

import json
import os

import genesis as gs

# ----------------------------------------------------------------------------- config
ASSET_ID = "nema_5_15"   # or "nema_1_15"
HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(HERE, "mesh", ASSET_ID)

MODE = "insert"             # "assembled" -> spawn mated (sanity-check contacts/stability)
                            # "insert"    -> plug starts lifted above the socket

SOCKET_POSE = (0.0, 0.0, 0.02)   # raise slightly off the ground plane
DT = 1.0 / 240.0                 # small dt helps tight-clearance contact
SUBSTEPS = 4

with open(os.path.join(HERE, "mesh", ASSET_ID, "meta.json")) as f:
    LIFT = json.load(f)["disassembly_dist_m"]   # meters to fully separate the pair

# ----------------------------------------------------------------------------- scene
gs.init(backend=gs.gpu)  # or gs.cpu

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=DT, substeps=SUBSTEPS),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(0.25, -0.25, 0.20),
        camera_lookat=(0.0, 0.0, 0.05),
    ),
    show_viewer=True,
)

scene.add_entity(gs.morphs.Plane())

# --- socket: MUST keep its concavity. A convex hull would seal the hole shut. ---
# Genesis handles non-convex rigid meshes either via pre-baked SDF collision or
# via convex decomposition. Two options:
#   1) convexify=False           -> raw mesh + SDF narrow phase (preferred here)
#   2) convexify=True + coacd    -> convex decomposition (if your build lacks (1))
socket = scene.add_entity(
    gs.morphs.Mesh(
        file=os.path.join(MESH_DIR, "asset_socket.obj"),
        pos=SOCKET_POSE,
        fixed=True,          # hold the socket rigidly (like a bench vise)
        convexify=False,     # <-- critical for a socket
    ),
    # material=gs.materials.Rigid(friction=0.5),   # AutoMate uses friction 0.5
)

plug_z = SOCKET_POSE[2] if MODE == "assembled" else SOCKET_POSE[2] + LIFT + 0.005
plug = scene.add_entity(
    gs.morphs.Mesh(
        file=os.path.join(MESH_DIR, "asset_plug.obj"),
        pos=(SOCKET_POSE[0], SOCKET_POSE[1], plug_z),
        fixed=False,
        convexify=False,     # plugs are often near-convex, but keep exact geometry
    ),
    # material=gs.materials.Rigid(friction=0.5),
)

# Optional: add a Franka (ships with Genesis) and drive it toward the grasp poses
# in data/plug_grasps.json (poses are in the plug frame).
# franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))

scene.build()

for i in range(2000):
    scene.step()
    if i % 120 == 0:
        pos = plug.get_pos()
        print(f"step {i:5d}  plug pos: {pos}")

# What to look for:
#  - MODE="assembled": pair should rest mated without jitter/penetration blow-up.
#    If it explodes: lower dt, raise substeps, or subdivide the meshes (denser
#    vertices -> better SDF sampling; see AutoMate/Assemble-Them-All notes).
#  - MODE="insert": the plug free-falls; without guidance most pairs won't
#    self-align (clearances are sub-mm) -- that's your control/RL problem.
#    Scripted-insertion sanity check: kinematically lower the plug along -z.
