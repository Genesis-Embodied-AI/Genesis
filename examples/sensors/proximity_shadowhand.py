"""
Interactive Proximity sensor with Shadow Hand and keyboard teleop.

Proximity sensors on the hand measure distance to a rubber duck (mesh) and a box.
Use keyboard controls to move the hand via IK; the hand tracks target positions
for the wrist and fingertips.
"""

import argparse
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind

# Teleop
KEY_DPOS = 0.015

# Proximity sensor
MAX_RANGE = 0.5

# Objects
DUCK_POS = (-0.2, 0.4, 0.6)
DUCK_QUAT = gu.euler_to_quat((90.0, 0.0, -90.0))
BOX_POS = (-0.3, 0.2, 0.4)
BOX_QUAT = (1, 0, 0, 0)


def main():
    parser = argparse.ArgumentParser(description="Interactive Proximity sensor with Shadow Hand")
    parser.add_argument("--vis", "-v", action="store_true", default=False, help="Show visualization GUI")
    parser.add_argument("--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--seconds", "-t", type=float, default=3.0, help="Seconds to simulate (headless mode)")
    args = parser.parse_args()

    gs.init(
        backend=gs.gpu if args.gpu else gs.cpu,
        precision="32",
        logging_level="info",
    )

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(-1.2, 0.6, 1.0),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=args.vis,
    )

    hand_pos_init = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/shadow_hand/shadow_hand.urdf",
            pos=hand_pos_init,
            euler=(0.0, 0.0, -90.0),
        ),
    )
    duck = scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            pos=DUCK_POS,
            scale=0.06,
            quat=DUCK_QUAT,
        ),
        surface=gs.surfaces.Default(color=(0.95, 0.75, 0.2, 1.0)),
    )
    box = scene.add_entity(
        material=gs.materials.Rigid(),
        morph=gs.morphs.Box(
            size=(0.06, 0.06, 0.08),
            pos=BOX_POS,
        ),
        surface=gs.surfaces.Default(color=(0.3, 0.7, 0.4, 1.0)),
    )

    sensors = []
    for link_name in (
        "palm",
        "index_finger_distal",
        "middle_finger_distal",
        "ring_finger_distal",
        "little_finger_distal",
        "thumb_distal",
    ):
        sensor = scene.add_sensor(
            gs.sensors.ProximityOptions(
                entity_idx=robot.idx,
                link_idx_local=robot.get_link(link_name).idx_local,
                probe_local_pos=((0.0, 0.0, 0.0),),
                track_link_idx=(duck.base_link_idx, box.base_link_idx),
                max_range=MAX_RANGE,
                draw_debug=args.vis,
            )
        )
        sensors.append(sensor)

    scene.build()

    hand_pos = hand_pos_init.copy()

    # Register keybindings
    is_running = True
    if args.vis:

        def stop():
            nonlocal is_running
            is_running = False

        def move(delta: tuple[float, float, float]):
            hand_pos[0] += delta[0]
            hand_pos[1] += delta[1]
            hand_pos[2] += delta[2]
            hand_pos[2] = max(0.05, min(0.6, hand_pos[2]))

        def reset_pose():
            hand_pos[:] = hand_pos_init
            duck.set_pos(DUCK_POS)
            duck.set_quat(DUCK_QUAT)
            box.set_pos(BOX_POS)
            box.set_quat(BOX_QUAT)

        scene.viewer.register_keybinds(
            Keybind("move_forward", Key.UP, KeyAction.HOLD, callback=move, args=((KEY_DPOS, 0, 0),)),
            Keybind("move_back", Key.DOWN, KeyAction.HOLD, callback=move, args=((-KEY_DPOS, 0, 0),)),
            Keybind("move_right", Key.RIGHT, KeyAction.HOLD, callback=move, args=((0, -KEY_DPOS, 0),)),
            Keybind("move_left", Key.LEFT, KeyAction.HOLD, callback=move, args=((0, KEY_DPOS, 0),)),
            Keybind("move_down", Key.J, KeyAction.HOLD, callback=move, args=((0, 0, -KEY_DPOS),)),
            Keybind("move_up", Key.K, KeyAction.HOLD, callback=move, args=((0, 0, KEY_DPOS),)),
            Keybind("reset", Key.U, KeyAction.PRESS, callback=reset_pose),
            Keybind("quit", Key.ESCAPE, KeyAction.PRESS, callback=stop),
        )

    print("\n=== Proximity sensor with Shadow Hand ===")
    print("Proximity sensors on hand palm and fingertips, tracking duck and box links")
    if args.vis:
        print("Keyboard: [↑/↓/←/→] move hand XY, [n/m] up/down, [u] reset, [ESC] quit")
    else:
        print(f"Running headless for {args.seconds}s ...")
    print()

    # Simulation loop
    try:
        while is_running:
            if args.vis:
                robot.set_pos(hand_pos)
            else:
                distances = []
                for sensor in sensors:
                    data = sensor.read()
                    distances.append(data.distance)
                print(f"Proximity distances: {distances}")

            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
            if not args.vis and scene.t > args.seconds:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
