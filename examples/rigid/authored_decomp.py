import argparse
import os

import genesis as gs

BASE_HEIGHT = 0.020
RING_HEIGHT = 0.020
BALL_HEIGHT = 0.0215
POLE_HEIGHT = 0.145
RINGS_ORDER = (0, 1, 2, 3, 5, 4)
RING_COLORS = [
    (0.95, 0.95, 0.95, 1.0),
    (0.60, 0.80, 0.70, 1.0),
    (0.78, 0.88, 0.80, 1.0),
    (0.90, 0.55, 0.60, 1.0),
    (0.85, 0.72, 0.35, 1.0),
    (0.95, 0.95, 0.95, 1.0),
]


def build_scene(show_viewer, debug, dt, gjk):
    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=gjk,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.4, 0.0, 0.3),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    # Base/pole rests directly on the plane.
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/base_pole.urdf",
            pos=(0.0, 0.0, BASE_HEIGHT / 2),
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(
            rho=600.0,
        ),
        vis_mode="collision" if debug else "visual",
        visualize_contact=False,
    )

    # Stack the rings in the demo order. A slight overlap (-1e-4) ensures they rest in contact at init.
    height = BASE_HEIGHT
    for ring_idx in RINGS_ORDER:
        scene.add_entity(
            morph=gs.morphs.URDF(
                file=f"tower/ring_{ring_idx + 1:02d}.urdf",
                pos=(0.0, 0.0, height + (RING_HEIGHT - 1e-4) / 2),
                file_meshes_are_zup=True,
            ),
            surface=gs.surfaces.Default(color=RING_COLORS[ring_idx]),
            material=gs.materials.Rigid(rho=600.0),
            vis_mode="collision" if debug else "visual",
            visualize_contact=debug,
        )
        height += RING_HEIGHT - 1e-4

    # Ball caps the pole.
    scene.add_entity(
        morph=gs.morphs.URDF(
            file="tower/ball.urdf",
            pos=(0.0, 0.0, height + BALL_HEIGHT),
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(rho=600.0),
        vis_mode="collision" if debug else "visual",
        visualize_contact=False,
    )

    scene.build()

    if debug:
        obj_com_rel = scene.rigid_solver.dyn_info.geoms.center.to_numpy()
        obj_pos = scene.rigid_solver.dyn_state.geoms.pos.to_numpy()[..., 0, :]
        obj_quat = scene.rigid_solver.dyn_state.geoms.quat.to_numpy()[..., 0, :]
        obj_com_world = gs.utils.geom.transform_by_trans_quat(obj_com_rel, obj_pos, obj_quat)
        scene.draw_debug_spheres(poss=obj_com_world, radius=0.002, color=(1, 1, 1, 1))
        scene.visualizer.update(force=True)

    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=False)
    parser.add_argument("-n", "--steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--gjk", action="store_true", default=False)
    args = parser.parse_args()

    scene = build_scene(args.vis, args.debug, args.dt, args.gjk)
    horizon = args.steps if "PYTEST_VERSION" not in os.environ else 5
    for _ in range(horizon):
        scene.step()


if __name__ == "__main__":
    main()
