import pytest
import genesis as gs

from .utils import assert_allclose


def test_drone_hover_same_with_and_without_substeps():
    TOL = 1e-7
    base_rpm = 15000
    scene_ref = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=0.002, substeps=1))
    drone_ref = scene_ref.add_entity(morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 1.0)))
    scene_ref.build()

    for _ in range(2500):
        drone_ref.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])
        scene_ref.step()

    x_ref = drone_ref.get_dofs_position()[0]
    y_ref = drone_ref.get_dofs_position()[1]
    z_ref = drone_ref.get_dofs_position()[2]

    scene_test = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=0.01, substeps=5))
    drone_test = scene_test.add_entity(morph=gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 1.0)))
    scene_test.build()

    for _ in range(500):
        drone_test.set_propellels_rpm([base_rpm, base_rpm, base_rpm, base_rpm])
        scene_test.step()

    x_test = drone_test.get_dofs_position()[0]
    y_test = drone_test.get_dofs_position()[1]
    z_test = drone_test.get_dofs_position()[2]

    assert_allclose(x_test, x_ref, tol=TOL)
    assert_allclose(y_test, y_ref, tol=TOL)
    assert_allclose(z_test, z_ref, tol=TOL)
