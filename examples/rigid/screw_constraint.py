"""Nut screwed onto a fixed bolt by a screw constraint, seated on the bolt head, then unscrewed off the tip.

Without the viewer, this runs a scripted sequence. With it, the keyboard drives the nut:
Up      - Apply a clockwise torque, seen from the tip of the bolt
Down    - Apply a counterclockwise torque, seen from the tip of the bolt
\\       - Reset the nut onto the bolt
Esc     - Quit
"""

import argparse
import math
import os

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", help="Show visualization GUI")
    parser.add_argument("-g", "--gpu", action="store_true", help="Run on GPU instead of CPU")
    parser.add_argument("--torque", type=float, default=0.003, help="Driving torque about the bolt [N*m]")
    args = parser.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_screw_constraints=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.12, -0.2, 0.12),
            camera_lookat=(0.02, 0.0, 0.04),
            camera_fov=35,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    steel = gs.materials.Rigid(rho=7850.0)  # Density of steel

    bolt = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/bolt_nut/bolt.stl",
            pos=(0.0, 0.0, 0.05),
            euler=(0.0, 90.0, 0.0),
            fixed=True,
        ),
        material=steel,
    )
    nut = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/bolt_nut/nut.stl",
            pos=(0.013, 0.0, 0.05),
            euler=(0.0, 90.0, 0.0),
        ),
        material=steel,
        surface=gs.surfaces.Default(
            color=(0.85, 0.65, 0.25),
        ),
    )

    scene.build()

    # The bolt axis is the world x axis, so the travel of the nut is its x position counted from where it starts: its
    # base reaches the head at -13 mm (the seat) and clears the tip at +19 mm, so the constraint is deleted a little
    # further, at +21 mm. In the frame of the bolt, the axis is the z axis of its mesh.
    travel_seat, travel_release = -0.013, 0.021
    rigid = scene.sim.rigid_solver
    bolt_idx, nut_idx = bolt.base_link.idx, nut.base_link.idx
    nut_qpos = nut.get_qpos()
    nut_x0 = nut.get_pos()[0]

    def add_screw():
        rigid.add_screw_constraint(
            bolt_idx, nut_idx, axis=(0.0, 0.0, 1.0), pitch=3.0e-3, limit=(travel_seat, math.inf), frictionloss=0.002
        )

    add_screw()
    is_screwed, is_running, is_reset_requested, torque = True, True, False, 0.0

    # Keybind callbacks run on the viewer thread, so they only set the request that the main loop carries out between
    # steps. The torque is applied about the local z axis of the nut, along which a positive turn moves it to the tip.
    def drive(direction):
        nonlocal torque
        torque = direction * args.torque

    def request_reset():
        nonlocal is_reset_requested
        is_reset_requested = True

    def stop():
        nonlocal is_running
        is_running = False

    def step(i_step):
        nonlocal is_screwed, is_reset_requested
        if is_reset_requested:
            if is_screwed:
                rigid.delete_screw_constraint(bolt_idx, nut_idx)
            nut.set_qpos(nut_qpos)
            add_screw()
            is_screwed, is_reset_requested = True, False
        travel = nut.get_pos()[0] - nut_x0
        if is_screwed and travel > travel_release:
            rigid.delete_screw_constraint(bolt_idx, nut_idx)
            is_screwed = False
            gs.logger.info(f"step {i_step:4d}  released past the tip, the nut is free")
        elif i_step % 25 == 0:
            gs.logger.info(f"step {i_step:4d}  travel = {travel * 1e3:6.2f} mm  z = {nut.get_pos()[2] * 1e3:6.2f} mm")
        rigid.apply_links_external_wrench(torque=(0.0, 0.0, torque), links_idx=(nut_idx,), local=True)
        scene.step()

    if args.vis:
        scene.viewer.register_keybinds(
            Keybind("torque_clockwise", Key.UP, KeyAction.PRESS, callback=drive, args=(-1.0,)),
            Keybind("torque_clockwise_stop", Key.UP, KeyAction.RELEASE, callback=drive, args=(0.0,)),
            Keybind("torque_counterclockwise", Key.DOWN, KeyAction.PRESS, callback=drive, args=(1.0,)),
            Keybind("torque_counterclockwise_stop", Key.DOWN, KeyAction.RELEASE, callback=drive, args=(0.0,)),
            Keybind("reset", Key.BACKSLASH, KeyAction.RELEASE, callback=request_reset),
            Keybind("quit", Key.ESCAPE, KeyAction.RELEASE, callback=stop),
        )
        i_step = 0
        while is_running and scene.viewer.is_alive():
            step(i_step)
            i_step += 1
    else:
        # Screw the nut down onto the head, hold it seated for half a second, then unscrew it until it leaves the thread.
        horizon = 500 if "PYTEST_VERSION" not in os.environ else 5
        drive(-1.0)
        n_seated = 0
        for i_step in range(horizon):
            if is_screwed and nut.get_pos()[0] - nut_x0 < travel_seat + 1e-4:
                n_seated += 1
                if n_seated == 50:
                    drive(1.0)
            step(i_step)


if __name__ == "__main__":
    main()
