import os

import numpy as np

import genesis as gs
from genesis.ext.pyrender.interaction.keybindings import KeyAction, Keybind


class DroneController:
    def __init__(self):
        self.thrust = 14475.8  # Base hover RPM - constant hover
        self.rotation_delta = 200.0  # Differential RPM for rotation
        self.thrust_delta = 10.0  # Amount to change thrust by when accelerating/decelerating
        self.rpms = [self.thrust] * 4

    def update_thrust(self):
        # Reset RPMs to hover thrust
        self.rpms = [self.thrust] * 4
        return self.rpms

    def move_forward(self):
        """Front rotors spin faster"""
        self.rpms[0] += self.rotation_delta  # Front left
        self.rpms[1] += self.rotation_delta  # Front right
        self.rpms[2] -= self.rotation_delta  # Back left
        self.rpms[3] -= self.rotation_delta  # Back right
        self.rpms = np.clip(self.rpms, 0, 25000)

    def move_backward(self):
        """Back rotors spin faster"""
        self.rpms[0] -= self.rotation_delta  # Front left
        self.rpms[1] -= self.rotation_delta  # Front right
        self.rpms[2] += self.rotation_delta  # Back left
        self.rpms[3] += self.rotation_delta  # Back right
        self.rpms = np.clip(self.rpms, 0, 25000)

    def move_left(self):
        """Left rotors spin faster"""
        self.rpms[0] -= self.rotation_delta  # Front left
        self.rpms[2] -= self.rotation_delta  # Back left
        self.rpms[1] += self.rotation_delta  # Front right
        self.rpms[3] += self.rotation_delta  # Back right
        self.rpms = np.clip(self.rpms, 0, 25000)

    def move_right(self):
        """Right rotors spin faster"""
        print("move right")
        self.rpms[0] += self.rotation_delta  # Front left
        self.rpms[2] += self.rotation_delta  # Back left
        self.rpms[1] -= self.rotation_delta  # Front right
        self.rpms[3] -= self.rotation_delta  # Back right
        self.rpms = np.clip(self.rpms, 0, 25000)

    def accelerate(self):
        """All rotors spin faster"""
        self.thrust += self.thrust_delta
        self.rpms = [self.thrust] * 4
        self.rpms = np.clip(self.rpms, 0, 25000)

    def decelerate(self):
        """All rotors spin slower"""
        self.thrust -= self.thrust_delta
        self.rpms = [self.thrust] * 4
        self.rpms = np.clip(self.rpms, 0, 25000)


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
    from pyglet.window import key

    scene.viewer.register_keybinds(
        (
            Keybind(
                key_code=key.UP, key_action=KeyAction.HOLD, name="move_forward", callback_func=controller.move_forward
            ),
            Keybind(
                key_code=key.DOWN,
                key_action=KeyAction.HOLD,
                name="move_backward",
                callback_func=controller.move_backward,
            ),
            Keybind(key_code=key.LEFT, key_action=KeyAction.HOLD, name="move_left", callback_func=controller.move_left),
            Keybind(
                key_code=key.RIGHT, key_action=KeyAction.HOLD, name="move_right", callback_func=controller.move_right
            ),
            Keybind(
                key_code=key.SPACE, key_action=KeyAction.HOLD, name="accelerate", callback_func=controller.accelerate
            ),
            Keybind(
                key_code=key.LSHIFT, key_action=KeyAction.HOLD, name="decelerate", callback_func=controller.decelerate
            ),
        )
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
            # Update drone with current RPMs
            rpms = controller.update_thrust()
            drone.set_propellels_rpm(rpms)

            # Update physics
            scene.step()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
