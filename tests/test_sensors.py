import genesis as gs
import torch

from .utils import assert_allclose


def test_rigid_tactile_sensors_gravity_force(show_viewer):
    """Test if the sensor will detect the correct forces being applied on a falling box."""
    GRAVITY = -10.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-2,
            substeps=1,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(morph=gs.morphs.Plane())

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(1.0, 1.0, 1.0),  # volume = 1 m^3
            pos=(0.0, 0.0, 0.1),
        ),
        material=gs.materials.Rigid(rho=1.0),  # mass = 1 kg
    )
    bool_sensor = gs.sensors.RigidContactSensor(entity=box)
    grid_force_sensor = gs.sensors.RigidContactForceGridSensor(entity=box, grid_size=(1, 1, 2))
    grid_normtan_force_sensor = gs.sensors.RigidNormalTangentialForceGridSensor(entity=box, grid_size=(1, 1, 2))

    scene.build()

    assert not bool_sensor.read(), "Sensor should not be in contact with the ground yet"
    (
        assert_allclose(grid_force_sensor.read(), 0.0, tol=1e-9),
        "Force should be zero before contact",
    )
    (
        assert_allclose(grid_normtan_force_sensor.read(), 0.0, tol=1e-9),
        "Normal-tangential force should be zero before contact",
    )

    for _ in range(500):
        scene.step()

    assert bool_sensor.read(), "Sensor should detect contact with the ground"
    grid_forces = grid_force_sensor.read()  # shape (batch_size, grid_x, grid_y, grid_z, 3)
    (
        assert_allclose(
            grid_forces[0, 0, 0, 0, :],
            torch.tensor([0.0, 0.0, -GRAVITY]),
            tol=1e-5,
        ),
        "Force should be equal to -gravity (normal) force at the bottom of the box",
    )
    (
        assert_allclose(grid_forces[0, 0, 0, 1, :], torch.tensor([0.0, 0.0, 0.0]), tol=1e-9),
        "Force should be zero at the top of the box",
    )
    grid_normtan_forces = grid_normtan_force_sensor.read()  # shape (batch_size, grid_x, grid_y, grid_z, 4)
    (
        assert_allclose(
            grid_normtan_forces[0, 0, 0, 0, :],
            torch.tensor([-GRAVITY, 0.0, 0.0, 0.0]),
            tol=1e-5,
        ),
        "Normal force should be equal to -gravity (normal) force at the bottom of the box, with no tangential force",
    )
    (
        assert_allclose(grid_normtan_forces[0, 0, 0, 1, :], torch.tensor([0.0, 0.0, 0.0, 0.0]), tol=1e-9),
        "Normal-tangential force should be zero at the top of the box",
    )
