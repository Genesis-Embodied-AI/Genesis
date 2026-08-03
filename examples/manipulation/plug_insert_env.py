"""Plug insertion environment on an AgileX PiPER arm, using the NEMA 5-15 plug/socket assets.

The socket is fixed to the table and never moves. The episode starts with the plug already held by the gripper,
directly above the socket, so the only task left is the insertion itself: lower the plug straight down into the
socket, then release and retract. This mirrors the shared assembled-state frame convention of the NEMA assets
(`assets/nema_plug_socket_sim`): spawning the plug at the same pose as the socket means it is fully inserted, so the
insertion motion is a pure vertical descent with no reorientation.

Run with `--scripted` to drive the task open-loop and validate the physics and success criterion without a policy.
"""

import argparse
import json
import os

import numpy as np

import genesis as gs
from genesis.utils.geom import transform_by_trans_quat
from genesis.utils.misc import tensor_to_array

NEMA_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "nema_plug_socket_sim")
MESH_DIR = os.path.join(NEMA_ASSET_DIR, "mesh", "nema_5_15")

# Centre of the finger pad surface, expressed in the link6 frame: the IK target that behaves like a tool center point.
TCP_IN_WRIST = np.array([0.0, 0.0, 0.105])
# Quaternion aligning the wrist approach axis with world -z, i.e. a vertical top-down grasp. With this orientation the
# jaws close along world y (confirmed by sweeping joint7 and reading link7/link8 positions), so the grasp target must
# be centred on the plug's y-extent at the grasp height, not on its mesh origin; see PLUG_GRASP_Y_OFFSET.
GRASP_DOWN_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
# The model's own home keyframe, used only to seed the inverse kinematics for the reset pose.
KEYFRAME_QPOS = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0])
# joint7 travel; joint8 mirrors it through the model's own joint equality constraint, so it is never commanded.
GRIPPER_OPEN = 0.035
# Half the plug's width along the jaws' closing axis at the grasp height is ~0.009, so commanding the joint limit of
# 0.0 drives the jaws roughly a centimetre past contact. The resulting squeeze is unstable: the plug is extruded
# sideways out of the grip under the first contact load of the insertion rather than held. Stopping just short of the
# half-width gives a firm, stable grip instead.
GRIPPER_HOLD = 0.008
TABLE_HEIGHT = 0.7
# Height, in the plug's own mesh frame, of the body region the gripper closes on. Above the flat body (z <= 0.046)
# the plug tapers to a smaller flat cap at z=0.05, staying rectangular and symmetric about the same y-offset the whole
# way up, so the grasp point is pushed almost to that top edge (1mm of margin) rather than the flat body's middle:
# this keeps the jaws as far as possible from the socket opening once the plug is fully inserted, so they never dip
# into the socket block itself.
PLUG_GRASP_HEIGHT = 0.049
# The plug's cross-section at PLUG_GRASP_HEIGHT spans y in [-0.00475, 0.01275], not centred on the mesh origin (this
# offset is constant with height: the taper is symmetric about it). Since the jaws close along world y at
# GRASP_DOWN_QUAT, the grasp target must be offset by the cross-section's y midpoint, or the jaws squeeze one side of
# the plug harder than the other and it rotates loose during the insertion motion.
PLUG_GRASP_Y_OFFSET = 0.004
# Height of the platform the socket is mounted on. The socket's own footprint (40x40mm) is narrower than the open
# jaws (70mm span), so with the socket sitting flush on the table, the open gripper hovers close enough to the bare
# table around the socket to graze it once the plug is fully seated. Raising the socket on a riser lifts that whole
# interaction point clear of the table without changing the plug/socket/grasp geometry at all.
RISER_HEIGHT = 0.04
SOCKET_POS = np.array([0.28, 0.0, RISER_HEIGHT])
# Clearance added on top of the disassembly distance so the plug starts visibly separated from the socket opening.
START_CLEARANCE = 0.005


class PiperPlugInsertEnv:
    """A NEMA 5-15 socket fixed to a table; the task is to insert the plug already held by the gripper."""

    def __init__(self, show_viewer: bool = False) -> None:
        with open(os.path.join(MESH_DIR, "meta.json")) as f:
            self.lift = json.load(f)["disassembly_dist_m"]
        self.plug_start_pos = SOCKET_POS + (0.0, 0.0, self.lift + START_CLEARANCE)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1.0 / 240.0,
                substeps=4,
            ),
            rigid_options=gs.options.RigidOptions(
                # The PiPER MJCF declares an elliptic friction cone; honouring it keeps the held plug from slipping
                # through the jaws under the tangential load of the insertion contact.
                friction_cone=gs.friction_cone.elliptic,
                impratio=10.0,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.9, -0.7, 0.6),
                camera_lookat=(0.28, 0.0, 0.03),
                camera_fov=40,
                res=(960, 640),
            ),
            show_viewer=show_viewer,
        )

        # The table top defines z = 0, matching the arm base and the socket pose.
        self.scene.add_entity(
            gs.morphs.Plane(
                pos=(0.0, 0.0, -TABLE_HEIGHT),
            ),
        )
        self.scene.add_entity(
            gs.morphs.Box(
                size=(0.9, 1.0, TABLE_HEIGHT),
                pos=(0.28, 0.0, -0.5 * TABLE_HEIGHT),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.82, 0.71, 0.55),
            ),
        )
        # Mounts the socket above the table, wide enough to stay under the open jaws' 70mm span.
        self.scene.add_entity(
            gs.morphs.Box(
                size=(0.09, 0.09, RISER_HEIGHT),
                pos=(SOCKET_POS[0], SOCKET_POS[1], 0.5 * RISER_HEIGHT),
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.6, 0.62, 0.65),
            ),
        )
        self.piper = self.scene.add_entity(
            gs.morphs.MJCF(
                file="xml/agilex_piper/piper.xml",
            ),
        )

        # Convex decomposition (coacd), not a single convex hull: a single hull would seal the socket opening shut,
        # but a decomposition into several convex pieces preserves the concavity. This is required for stability, not
        # just speed: the raw SDF narrow phase against this socket's sub-millimetre-scale slot walls is only stable
        # for a plug centred to the micron: an off-centre plug (a few mm, typical of a real grasp) throws it into a
        # divergent contact response that launches the plug off the table within about 100 steps.
        self.socket = self.scene.add_entity(
            gs.morphs.Mesh(
                file=os.path.join(MESH_DIR, "asset_socket.obj"),
                pos=SOCKET_POS,
                fixed=True,
                convexify=True,
            ),
            material=gs.materials.Rigid(friction=0.5),
        )
        self.plug = self.scene.add_entity(
            gs.morphs.Mesh(
                file=os.path.join(MESH_DIR, "asset_plug.obj"),
                pos=self.plug_start_pos,
                convexify=True,
            ),
            material=gs.materials.Rigid(friction=0.5),
        )

        self.cam = self.scene.add_camera(
            res=(960, 640),
            pos=(0.55, -0.42, 0.28),
            lookat=(0.28, 0.0, 0.02),
            fov=35,
        )

        self.scene.build()

        self.wrist_link = self.piper.get_link("link6")
        self.arm_dofs = np.arange(6)
        self.roll_limit = tuple(tensor_to_array(limit)[5] for limit in self.piper.get_dofs_limit())

        # Genesis does not implement the model's gravity compensation, so the stock MJCF gains sag by up to 0.15 rad
        # at this pose. These hold the arm to within a few milliradians instead.
        self.piper.set_dofs_kp(np.array([4000.0, 4000.0, 4000.0, 2000.0, 500.0, 500.0, 200.0]), np.arange(7))
        self.piper.set_dofs_kv(np.array([35.0, 35.0, 35.0, 35.0, 12.0, 12.0, 10.0]), np.arange(7))

        self.piper.set_qpos(np.concatenate((KEYFRAME_QPOS, (GRIPPER_HOLD, -GRIPPER_HOLD))), zero_velocity=True)
        self.scene.step()

    def grasp_tcp_for(self, plug_pos: np.ndarray) -> np.ndarray:
        """Tool-center-point target that closes the jaws centred on the plug's cross-section, not its mesh origin."""
        return plug_pos + (0.0, PLUG_GRASP_Y_OFFSET, PLUG_GRASP_HEIGHT)

    def reset(self) -> None:
        """Start with the plug already grasped: the arm is placed directly at the holding pose, gripper closed."""
        grasp_qpos = self.solve_ik(self.grasp_tcp_for(self.plug_start_pos))
        self.piper.set_qpos(np.concatenate((grasp_qpos, (GRIPPER_HOLD, -GRIPPER_HOLD))), zero_velocity=True)
        self.piper.control_dofs_position(np.concatenate((grasp_qpos, (GRIPPER_HOLD,))), np.arange(7))
        self.plug.set_pos(self.plug_start_pos)
        self.plug.set_quat(np.array([1.0, 0.0, 0.0, 0.0]))
        self.plug.zero_all_dofs_velocity()
        self.scene.step()

    def apply_action(self, arm_qpos: np.ndarray, gripper: float) -> None:
        self.piper.control_dofs_position(np.concatenate((arm_qpos, (gripper,))), np.arange(7))
        self.scene.step()

    @property
    def tcp_pos(self) -> np.ndarray:
        return transform_by_trans_quat(
            TCP_IN_WRIST, tensor_to_array(self.wrist_link.get_pos()), tensor_to_array(self.wrist_link.get_quat())
        )

    @property
    def is_success(self) -> bool:
        """True when the plug pose matches the socket's shared assembled-state pose within the generated clearance."""
        plug_pos = tensor_to_array(self.plug.get_pos())
        return bool(np.linalg.norm(plug_pos - SOCKET_POS) < 0.005)

    def solve_ik(self, tcp_pos: np.ndarray) -> np.ndarray:
        """Arm joint angles placing the gripper pad centre at `tcp_pos` in a top-down orientation.

        The wrist roll is folded by the jaw's 180 degree symmetry onto the equivalent angle nearest the current one,
        otherwise the position controller chases the difference as a violent wrist spin.
        """
        qpos = self.piper.inverse_kinematics(
            link=self.wrist_link,
            pos=tcp_pos,
            quat=GRASP_DOWN_QUAT,
            local_point=TCP_IN_WRIST,
            dofs_idx_local=self.arm_dofs,
        )
        qpos = tensor_to_array(qpos)[:6]
        rolls = qpos[5] + np.pi * np.arange(-2, 3)
        rolls = rolls[(rolls >= self.roll_limit[0]) & (rolls <= self.roll_limit[1])]
        qpos[5] = rolls[np.argmin(np.abs(rolls - tensor_to_array(self.piper.get_qpos())[5]))]
        return qpos

    def move_to(self, tcp_pos: np.ndarray, gripper: float, steps: int, arrival_frac: float = 0.7) -> None:
        """Ramp the arm to the pose reaching `tcp_pos`, then hold the target to let it settle.

        The joint target is interpolated from the measured pose rather than applied as a step, because a step input
        makes the arm swing along a path that can knock the held plug loose. `arrival_frac` trades settling time for
        dwell time at the target: the insertion leg needs it close to 1 because, at this connector's clearance, the
        plug creeps back out of a fully-seated pose the longer it dwells there under the position-controlled grip.
        """
        start = tensor_to_array(self.piper.get_qpos())[:6]
        goal = self.solve_ik(tcp_pos)
        arrival = max(1, int(arrival_frac * steps))
        for i in range(steps):
            joint_target = start + min(1.0, (i + 1) / arrival) * (goal - start)
            self.apply_action(joint_target, gripper)

    def run_scripted_insert(self) -> bool:
        """Lower the already-grasped plug straight into the socket, then release and retract."""
        insert_tcp = self.grasp_tcp_for(SOCKET_POS)
        retract_tcp = insert_tcp + (0.0, 0.0, 0.05)
        self.move_to(insert_tcp, gripper=GRIPPER_HOLD, steps=160, arrival_frac=0.95)
        self.move_to(insert_tcp, gripper=GRIPPER_OPEN, steps=40)
        self.move_to(retract_tcp, gripper=GRIPPER_OPEN, steps=60)
        return self.is_success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("--scripted", action="store_true", default=False, help="run the IK waypoint policy")
    parser.add_argument("--video", type=str, default=None, help="record the scripted run to this .mp4 path")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32")
    env = PiperPlugInsertEnv(show_viewer=args.vis)
    env.reset()
    print(f"plug pos after reset: {np.round(tensor_to_array(env.plug.get_pos()), 4)}")
    print(f"tcp pos after reset:  {np.round(env.tcp_pos, 4)}")

    if args.video is not None:
        env.cam.start_recording(save_to_filename=args.video)
    if args.scripted:
        print(f"scripted insertion success: {env.run_scripted_insert()}")
        print(f"final plug pos: {np.round(tensor_to_array(env.plug.get_pos()), 4)}")
    if args.video is not None:
        env.cam.stop_recording()
        print(f"video saved to {args.video}")


if __name__ == "__main__":
    main()
