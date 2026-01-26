import pytest
import genesis as gs
from huggingface_hub import snapshot_download

def test_uipc_setup():
    dt = 0.01
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
            contact_d_hat=0.001,
            contact_friction_mu=0.02,
            IPC_self_contact=False,
            disable_genesis_contact=True,
        ),
        show_viewer=False,
    )

    scene.add_entity(gs.morphs.Plane())

    asset_path = snapshot_download(
        repo_type="dataset",
        repo_id="Genesis-Intelligence/assets",
        revision="72b04f7125e21df1bebd54a7f7b39d1cd832331c",
        allow_patterns="grid20x20.obj",
        max_workers=1,
    )

    scene.add_entity(
        morph=gs.morphs.Mesh(file=f"{asset_path}/grid20x20.obj", scale=1.0, pos=(0.0, 0.0, 0.85)),
        material=gs.materials.FEM.Cloth(
            E=1e5,
            nu=0.499,
            rho=200,
            thickness=0.001,
            bending_stiffness=1.0,
        ),
    )

    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.25, 0.0, 0.15), size=(0.1, 0.1, 0.1)),
        material=gs.materials.Rigid(rho=500, friction=0.3),
    )

    scene.add_entity(
        morph=gs.morphs.Sphere(pos=(-0.25, 0.0, 0.12), radius=0.1),
        material=gs.materials.FEM.Elastic(E=1.0e4, nu=0.3, rho=1000.0, model="stable_neohookean"),
    )

    scene.build(n_envs=1)
    scene.step()

