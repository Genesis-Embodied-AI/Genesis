import os

import numpy as np

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind


class DroneController:
    def __init__(self):
        self.thrust = 14475.8  # Base RPM for constant hover
        self.rotation_delta = 100.0  # Differential RPM for rotation
        self.thrust_delta = 10.0  # Amount to change thrust by when accelerating/decelerating
        self.cur_dir = np.array([0.0, 0.0, 0.0, 0.0])  # rotor directions

    def update_rpms(self):
        """Compute RPMs based on current direction and thrust"""
        clipped_dir = np.clip(self.cur_dir, -1.0, 1.0)
        rpms = self.thrust + clipped_dir * self.rotation_delta
        return np.clip(rpms, 0, 25000)

    def add_direction(self, direction: np.ndarray):
        """Add direction vector (on key press)"""
        self.cur_dir += direction

    def accelerate(self):
        """Increase base thrust"""
        self.thrust = min(self.thrust + self.thrust_delta, 25000)

    def decelerate(self):
        """Decrease base thrust"""
        self.thrust = max(self.thrust - self.thrust_delta, 0)


def main():
    # Initialize Genesis
    gs.init(backend=gs.cpu)

    # Create scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    # Add entities
    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0.0, 0, 0.5),  # Start a bit higher
        ),
    )

    scene.viewer.follow_entity(drone)

    # Initialize controller
    controller = DroneController()

    # Build scene
    scene.build()

    # Register keybindings
    def direction_keybinds(key: Key, name: str, direction: tuple[float, float, float, float]):
        """Helper to create press/release keybinds for a direction"""
        dir_arr = np.array(direction)
        return [
            Keybind(
                key_name=f"{name}_press",
                key=key,
                key_action=KeyAction.PRESS,
                callback=controller.add_direction,
                args=(dir_arr,),
            ),
            Keybind(
                key_name=f"{name}_release",
                key=key,
                key_action=KeyAction.RELEASE,
                callback=controller.add_direction,
                args=(-dir_arr,),
            ),
        ]

    scene.viewer.register_keybinds(
        *direction_keybinds(Key.UP, "move_forward", (1.0, 1.0, -1.0, -1.0)),
        *direction_keybinds(Key.DOWN, "move_backward", (-1.0, -1.0, 1.0, 1.0)),
        *direction_keybinds(Key.LEFT, "move_left", (-1.0, 1.0, -1.0, 1.0)),
        *direction_keybinds(Key.RIGHT, "move_right", (1.0, -1.0, 1.0, -1.0)),
        Keybind("accelerate", Key.SPACE, KeyAction.HOLD, callback=controller.accelerate),
        Keybind("decelerate", Key.LSHIFT, KeyAction.HOLD, callback=controller.decelerate),
    )

    # Print control instructions
    print("\nDrone Controls:")
    print("↑ - Move Forward (North)")
    print("↓ - Move Backward (South)")
    print("← - Move Left (West)")
    print("→ - Move Right (East)")
    print("space - Increase RPM")
    print("shift - Decrease RPM")
    print("\nPlus all default viewer controls (press 'i' to see them)\n")
    print("Initial hover RPM:", controller.thrust)

    # Run simulation
    try:
        while True:
            # Update and apply RPMs based on current direction
            rpms = controller.update_rpms()
            print("Current RPMs:", rpms)
            drone.set_propellels_rpm(rpms)

            # Step simulation
            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
