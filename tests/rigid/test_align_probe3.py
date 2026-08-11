"""Geometry diagnostics for the test_align_mesh scene.

Env 0 and env 1 differ only in which variant of the heterogeneous entity they hold: env 0 the bowl, env 1 the
mango. CI's failing run matches a local run bit-for-bit on env 1 and diverges by four orders of magnitude on
env 0, so this dumps the processed collision geometry (convex-piece counts, vertex/face totals, AABBs) alongside
the settling residuals of both the measured mango and the heterogeneous entity, to show whether the two
environments are looking at the same bowl.
"""

import numpy as np
import pytest

import genesis as gs

from ..utils import get_hf_dataset


@pytest.mark.required
def test_align_probe3(show_viewer, tol):
    INIT_POS = (0.0, 0.0, 0.1)

    mango_path = get_hf_dataset(pattern="glb/mango.glb")
    bowl_path = get_hf_dataset(pattern="glb/orange_plastic_bowl.glb")

    for name, path in (("mango.glb", f"{mango_path}/glb/mango.glb"), ("bowl.glb", f"{bowl_path}/glb/orange_plastic_bowl.glb")):
        with open(path, "rb") as fd:
            data = fd.read()
        import hashlib

        print(f"GEOM asset {name} bytes={len(data)} sha256={hashlib.sha256(data).hexdigest()}", flush=True)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    mango_morph = gs.morphs.Mesh(file=f"{mango_path}/glb/mango.glb", scale=0.045, pos=INIT_POS, align=True)
    mango = scene.add_entity(
        mango_morph,
        material=gs.materials.Rigid(rho=1000.0),
        vis_mode="collision",
        visualize_contact=True,
    )
    ghost_mango = scene.add_entity(mango_morph, material=gs.materials.Kinematic())
    HET_POS = (0.5, 0.0, 0.1)
    het_obj = scene.add_entity(
        morph=(
            gs.morphs.Mesh(
                file=f"{bowl_path}/glb/orange_plastic_bowl.glb",
                scale=0.5,
                pos=HET_POS,
                offset_euler=(30.0, 0.0, 0.0),
                align=True,
            ),
            gs.morphs.Mesh(file=f"{mango_path}/glb/mango.glb", scale=0.045, pos=HET_POS, align=True),
        ),
        material=gs.materials.Rigid(rho=1000.0),
    )
    scene.build(n_envs=2)

    for label, entity in (("mango", mango), ("het_obj", het_obj)):
        print(f"GEOM {label} n_geoms={len(entity.geoms)}", flush=True)
        for i, g in enumerate(entity.geoms):
            verts = np.asarray(g.init_verts)
            faces = np.asarray(g.init_faces)
            print(
                f"GEOM {label}[{i}] verts={verts.shape} faces={faces.shape} "
                f"vmin={np.round(verts.min(axis=0), 9).tolist()!r} vmax={np.round(verts.max(axis=0), 9).tolist()!r}",
                flush=True,
            )

    qpos = (0.3, -0.2, 1.0, 0.6, 0.5, 0.3, 0.0)
    mango.set_qpos(qpos)
    ghost_mango.set_qpos(qpos)
    scene.reset()

    for i in range(600):
        scene.step()
        if (i + 1) % 200 == 0:
            m = mango.get_dofs_velocity(dofs_idx_local=(3, 4, 5)).cpu().numpy()
            h = het_obj.get_dofs_velocity().cpu().numpy()
            print(f"PROBE3 step={i + 1:4d} mango_ang={m.tolist()!r} het_absmax={np.abs(h).max(axis=-1).tolist()!r}", flush=True)

    m = mango.get_dofs_velocity(dofs_idx_local=(3, 4, 5)).cpu().numpy()
    h = het_obj.get_dofs_velocity().cpu().numpy()
    print(f"PROBE3 FINAL mango_ang={m.tolist()!r}", flush=True)
    print(f"PROBE3 FINAL het_vel={h.tolist()!r}", flush=True)
