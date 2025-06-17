import pytest
import genesis as gs


def test_drone_hover_same_with_and_without_substeps():
    TOL = 0.02
    base_rpm = 14468.429183500699
    scene_ref = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=0.002, substeps=1))
    drone_ref = scene_ref.add_entity(morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 1.0)))
    scene_ref.build()

    for _ in range(100):
        drone_ref.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])
        scene_ref.step()

    z_ref = float(drone_ref.get_dofs_position()[2])

    scene_test = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=0.01, substeps=5))
    drone_test = scene_test.add_entity(morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 1.0)))
    scene_test.build()

    for _ in range(100):
        drone_test.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])
        scene_test.step()

    z_test = float(drone_test.get_dofs_position()[2])

    assert abs(z_ref - z_test) < TOL
