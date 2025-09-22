import pytest

import genesis as gs


pytestmark = [
    pytest.mark.field_only,
]


# Note that "session" scope must NOT be used because the material while be altered without copy when building the scene
@pytest.fixture(scope="function")
def pbd_material():
    """Fixture for common FEM material properties"""
    return gs.materials.PBD.Elastic()


@pytest.mark.required
def test_maxvolume(pbd_material, show_viewer, box_obj_path):
    """Test that imposing a maximum element volume constraint produces a finer mesh (i.e., more elements)."""
    scene = gs.Scene(
        pbd_options=gs.options.PBDOptions(
            particle_size=0.1,
        ),
        show_viewer=show_viewer,
    )

    # Mesh without any maximum-element-volume constraint
    pbd1 = scene.add_entity(
        morph=gs.morphs.Mesh(file=box_obj_path, nobisect=False, verbose=1),
        material=pbd_material,
    )

    # Mesh with maximum element volume limited to 0.001
    pbd2 = scene.add_entity(
        morph=gs.morphs.Mesh(file=box_obj_path, nobisect=False, maxvolume=0.001, verbose=1),
        material=pbd_material,
    )

    assert pbd1.n_elems < pbd2.n_elems, (
        f"Mesh with maxvolume=0.01 generated {pbd2.n_elems} elements; "
        f"expected more than {pbd1.n_elems} elements without a volume limit."
    )


@pytest.mark.required
def test_cloth_attachment(show_viewer):
    """Attach 8 cloth particles to a cube, batched (n_envs=2), verify attachment with leftward motion."""

    particle_size = 1e-2
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-2, substeps=10),
        pbd_options=gs.options.PBDOptions(particle_size=particle_size),
        show_viewer=show_viewer,
    )
    rigid_material = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)
    scene.add_entity(gs.morphs.Plane(), rigid_material)

    box = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.25, 0.25, 0.25), size=(0.25, 0.25, 0.25)),
        material=rigid_material,
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/cloth.obj",
            pos=(0.25, 0.25, 0.25 + 0.125 + particle_size),
            scale=0.5,
        ),
        material=gs.materials.PBD.Cloth(),
        surface=gs.surfaces.Default(color=(0.2, 0.4, 0.8, 1.0)),
    )

    scene.build(n_envs=2)

    particles_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    box_link_idx = box.links[0].idx
    scene.pbd_solver.set_animate_particles_by_link(particles_idx, box_link_idx, scene.rigid_solver.links_state)

    # leftward velocity for both envs on base linear DOFs [3,4,5]
    vel = np.array([[-0.0, 5.0, 0.0], [-5.0, 0.0, 0.0]], dtype=np.float32)
    box.set_dofs_velocity(vel, dofs_idx_local=np.arange(3))

    cloth_pos0 = cloth.get_particles_pos().cpu().numpy()[:, particles_idx]
    link_pos0 = scene.rigid_solver.links[box_link_idx].get_pos().cpu().numpy()

    for _ in range(50):
        scene.step()

    # Check that the attached particles followed the link displacement per env
    cloth_pos1 = cloth.get_particles_pos().cpu().numpy()[:, particles_idx]
    link_pos1 = scene.rigid_solver.links[box_link_idx].get_pos().cpu().numpy()

    cloth_disp = cloth_pos1 - cloth_pos0
    link_disp = link_pos1 - link_pos0
    # broadcast link_disp to match cloth_disp shape
    link_disp = np.tile(link_disp[:, None, :], (1, len(particles_idx), 1))

    assert_allclose(cloth_disp, link_disp, atol=1e-1)
