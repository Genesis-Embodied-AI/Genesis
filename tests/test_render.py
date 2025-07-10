import numpy as np
import pytest

import genesis as gs


@pytest.mark.required
@pytest.mark.parametrize("segmentation_level", ["entity", "link"])
@pytest.mark.parametrize("particle_mode", ["visual", "particle"])
def test_segmentation(segmentation_level, particle_mode):
    """Test segmentation rendering."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        fem_options=gs.options.FEMOptions(use_implicit_solver=True),
        vis_options=gs.options.VisOptions(segmentation_level=segmentation_level),
        show_viewer=False,
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_link_arm.urdf",
            pos=(-1.0, -1.0, 0.5),
            euler=(0, 0, 90),
        ),
    )

    # We don't test "recon" for vis_mode because it is hard to install.
    sph_mode = "particle" if particle_mode == "visual" else particle_mode
    materials = [
        (gs.materials.Rigid(), "visual"),
        (gs.materials.Tool(), "visual"),
        (gs.materials.FEM.Elastic(), "visual"),
        (gs.materials.MPM.Elastic(), particle_mode),
        (gs.materials.PBD.Cloth(), particle_mode),
        (gs.materials.SPH.Liquid(), sph_mode),
        # TODO: Add avatar. Currently avatar solver is buggy.
    ]
    ducks = []
    spacing = 0.5
    for i, pack in enumerate(materials):
        col_idx = i // 3 - 1
        row_idx = i % 3 - 1
        material, vis_mode = pack
        ducks.append(
            scene.add_entity(
                material=material,
                morph=gs.morphs.Mesh(
                    file="meshes/duck.obj",
                    scale=0.1,
                    pos=(col_idx * spacing, row_idx * spacing, 0.5),
                ),
                surface=gs.surfaces.Default(
                    color=np.random.rand(3),
                    vis_mode=vis_mode,
                ),
            )
        )

    camera = scene.add_camera(
        res=(512, 512),
        pos=(2.0, 0.0, 2.0),
        lookat=(0, 0, 0.5),
        fov=40,
    )
    scene.build()

    seg_num = len(materials) + 3 if segmentation_level == "link" else len(materials) + 2
    idx_dict = camera.get_segmentation_idx_dict()
    assert len(idx_dict) == seg_num
    comp_key = 0
    for seg_key in idx_dict.values():
        if isinstance(seg_key, tuple):
            comp_key += 1
    assert comp_key == (3 if segmentation_level == "link" else 0)

    for i in range(2):
        scene.step()
        _, _, seg, _ = camera.render(rgb=False, depth=False, segmentation=True, colorize_seg=False, normal=False)
        uni_count = len(np.unique(seg))
        assert seg.min() == 0
        assert seg.max() == uni_count - 1 == seg_num - 1
