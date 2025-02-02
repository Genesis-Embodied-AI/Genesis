import argparse
import numpy as np
import genesis as gs
import time
import threading
from pynput import keyboard


class DroneController:
    def __init__(self):
        self.thrust = 14468.429183500699  # Base hover RPM - constant hover
        self.rotation_delta = 200  # Differential RPM for rotation
        self.thrust_delta = 10  # Amount to change thrust by when accelerating/decelerating
        self.running = True
        self.rpms = [self.thrust] * 4
        self.pressed_keys = set()

    def on_press(self, key):
        try:
            if key == keyboard.Key.esc:
                self.running = False
                return False
            self.pressed_keys.add(key)
            print(f"Key pressed: {key}")
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.discard(key)
        except KeyError:
            pass

    def update_thrust(self):
        # Store previous RPMs for debugging
        prev_rpms = self.rpms.copy()

        # Reset RPMs to hover thrust
        self.rpms = [self.thrust] * 4

        # Acceleration (Spacebar) - All rotors spin faster
        if keyboard.Key.space in self.pressed_keys:
            self.thrust += self.thrust_delta
            self.rpms = [self.thrust] * 4
            print("Accelerating")

        # Deceleration (Left Shift) - All rotors spin slower
        if keyboard.Key.shift in self.pressed_keys:
            self.thrust -= self.thrust_delta
            self.rpms = [self.thrust] * 4
            print("Decelerating")

        # Forward (North) - Front rotors spin faster
        if keyboard.Key.up in self.pressed_keys:
            self.rpms[0] += self.rotation_delta  # Front left
            self.rpms[1] += self.rotation_delta  # Front right
            self.rpms[2] -= self.rotation_delta  # Back left
            self.rpms[3] -= self.rotation_delta  # Back right
            print("Moving Forward")

        # Backward (South) - Back rotors spin faster
        if keyboard.Key.down in self.pressed_keys:
            self.rpms[0] -= self.rotation_delta  # Front left
            self.rpms[1] -= self.rotation_delta  # Front right
            self.rpms[2] += self.rotation_delta  # Back left
            self.rpms[3] += self.rotation_delta  # Back right
            print("Moving Backward")

        # Left (West) - Left rotors spin faster
        if keyboard.Key.left in self.pressed_keys:
            self.rpms[0] -= self.rotation_delta  # Front left
            self.rpms[2] -= self.rotation_delta  # Back left
            self.rpms[1] += self.rotation_delta  # Front right
            self.rpms[3] += self.rotation_delta  # Back right
            print("Moving Left")

        # Right (East) - Right rotors spin faster
        if keyboard.Key.right in self.pressed_keys:
            self.rpms[0] += self.rotation_delta  # Front left
            self.rpms[2] += self.rotation_delta  # Back left
            self.rpms[1] -= self.rotation_delta  # Front right
            self.rpms[3] -= self.rotation_delta  # Back right
            print("Moving Right")

        self.rpms = np.clip(self.rpms, 0, 25000)

        # Debug print if any RPMs changed
        if not np.array_equal(prev_rpms, self.rpms):
            print(f"RPMs changed from {prev_rpms} to {self.rpms}")

        return self.rpms


def run_sim(scene, drone, controller):
    while controller.running:
        try:
            # Update drone with current RPMs
            rpms = controller.update_thrust()
            drone.set_propellels_rpm(rpms)

            # Update physics
            scene.step()

            time.sleep(1 / 60)  # Limit simulation rate
        except Exception as e:
            print(f"Error in simulation loop: {e}")

    if scene.viewer:
        scene.viewer.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=True, help="Enable visualization (default: True)")
    parser.add_argument("-m", "--mac", action="store_true", default=False, help="Running on MacOS (default: False)")
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(backend=gs.cpu)

    # Create scene with initial camera view
    viewer_options = gs.options.ViewerOptions(
        camera_pos=(0.0, -4.0, 2.0),  # Now behind the drone (negative Y)
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=45,
        max_FPS=60,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        viewer_options=viewer_options,
        show_viewer=args.vis,
    )

    # Add entities
    plane = scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(0.0, 0, 0.5),  # Start a bit higher
        ),
    )

    scene.viewer.follow_entity(drone)

    # Build scene
    scene.build()

    # Initialize controller
    controller = DroneController()

    # Print control instructions
    print("\nDrone Controls:")
    print("↑ - Move Forward (North)")
    print("↓ - Move Backward (South)")
    print("← - Move Left (West)")
    print("→ - Move Right (East)")
    print("ESC - Quit\n")
    print("Initial hover RPM:", controller.thrust)

    # Start keyboard listener
    listener = keyboard.Listener(on_press=controller.on_press, on_release=controller.on_release)
    listener.start()

    if args.mac:
        # Run simulation in another thread
        sim_thread = threading.Thread(target=run_sim, args=(scene, drone, controller))
        sim_thread.start()

        if args.vis:
            scene.viewer.start()

        # Wait for threads to finish
        sim_thread.join()
    else:
        # Run simulation in main thread
        run_sim(scene, drone, controller)
    listener.stop()


if __name__ == "__main__":
    main()
