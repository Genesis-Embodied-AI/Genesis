import pytest

import genesis as gs

from .utils import get_hf_dataset

try:
    import uipc
except ImportError:
    pytest.skip("IPC Coupler is not supported because 'uipc' module is not available.", allow_module_level=True)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_ipc_cloth(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            contact_friction_mu=0.3,
            IPC_self_contact=False,
            two_way_coupling=True,
            disable_genesis_ground_contact=True,
            enable_ipc_gui=False,
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="grid20x20.obj")
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/grid20x20.obj",
            scale=2.0,
            pos=(0.0, 0.0, 1.5),
            euler=(0, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e6,
            nu=0.499,
            rho=200,
            thickness=0.001,
            bending_stiffness=50.0,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
            double_sided=True,
        ),
    )
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0, 0, 0.3),
        ),
        material=gs.materials.Rigid(
            rho=500,
            friction=0.3,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.3, 0.2, 0.8),
        ),
    )
    soft_ball = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.08,
            pos=(0.5, 0.0, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e3,
            nu=0.3,
            rho=1000.0,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.3, 0.8),
        ),
    )

    scene.build(n_envs=n_envs)

    scene.step()
