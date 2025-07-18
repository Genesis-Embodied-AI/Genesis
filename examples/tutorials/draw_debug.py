import time
import numpy as np
import genesis as gs


def main():
    gs.init(backend=gs.cpu)

    # Scene setup
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(5.0, -5.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
        max_FPS=200,
    )

    scene = gs.Scene(
        viewer_options=viewer_options,
        show_viewer=True,
    )

    # Add a plane for reference
    scene.add_entity(morph=gs.morphs.Plane())
    scene.build()

    # Create debug objects
    # Debug box
    debug_box = scene.draw_debug_box(
        bounds=[[-0.25, -0.25, 0], [0.25, 0.25, 0.5]],
        color=(1, 0, 1, 1),
        wireframe=True,
        wireframe_radius=0.005,  # Magenta
    )

    # Debug line
    debug_line = scene.draw_debug_line(
        start=(0.5, -0.25, 0.5), end=(0.5, 0.25, 0.5), radius=0.01, color=(1, 0, 0, 1)
    )  # Red

    # Debug arrow
    debug_arrow = scene.draw_debug_arrow(pos=(1, 0, 0), vec=(0, 0, 1), radius=0.02, color=(1, 0, 0, 0.5))  # Green

    # Debug sphere
    debug_sphere = scene.draw_debug_sphere(pos=(1.5, 0, 0.5), radius=0.1, color=(0, 0, 1, 0.5))  # Blue with alpha

    # Debug multiple spheres
    sphere_positions = np.array([[2, 0, 0.3], [2, 0, 0.5], [2, 0, 0.7]])
    debug_spheres = scene.draw_debug_spheres(poss=sphere_positions, radius=0.05, color=(1, 1, 0, 0.5))  # Yellow

    # Transformation matrix for frame (identity matrix with translation)
    T = np.eye(4)
    T[:3, 3] = [2.5, 0, 0.5]
    debug_frame = scene.draw_debug_frame(T=T, axis_length=0.5, origin_size=0.03, axis_radius=0.02)

    # Simulation loop
    for step in range(500):
        scene.step()
        time.sleep(0.01)

        # Clear individual objects after 200 steps
        if step == 100:
            scene.clear_debug_object(debug_box)
        elif step == 200:
            scene.clear_debug_object(debug_line)
        elif step == 300:
            scene.clear_debug_object(debug_arrow)
        # All remaining objects are removed
        elif step == 400:
            scene.clear_debug_objects()


if __name__ == "__main__":
    main()
