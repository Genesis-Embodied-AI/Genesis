import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

import genesis as gs
from genesis.utils.misc import tensor_to_array

from ..utils import assert_allclose, rgb_array_to_png_bytes


@pytest.mark.required
def test_scale_requires_enable_option(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
        rigid_options=gs.options.RigidOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 1.0),
        ),
    )
    scene.build()
    with pytest.raises(Exception):
        box.set_scale(2.0)


@pytest.mark.required
def test_scale_rejects_unsupported_entities(show_viewer):
    hinge = ET.Element("mujoco")
    body = ET.SubElement(ET.SubElement(hinge, "worldbody"), "body", pos="0 0 1")
    ET.SubElement(body, "joint", type="hinge", axis="0 1 0")
    ET.SubElement(body, "geom", type="box", size="0.1 0.1 0.1")

    scene = gs.Scene(
        show_viewer=show_viewer,
        rigid_options=gs.options.RigidOptions(
            enable_geom_scaling=True,
            batch_links_info=True,
            gravity=(0.0, 0.0, 0.0),
        ),
    )
    # A fixed mesh (batch_fixed_verts defaults False) shares its vertices across environments, and an articulated
    # entity has joint anchors that are not scaled; neither can be uniformly scaled per environment.
    fixed_mesh = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.1,
            pos=(0.0, 0.0, 1.0),
            fixed=True,
            convexify=True,
        ),
    )
    articulated = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=ET.tostring(hinge, encoding="unicode"),
        ),
    )
    scene.build()
    with pytest.raises(Exception):
        fixed_mesh.set_scale(2.0)
    with pytest.raises(Exception):
        articulated.set_scale(2.0)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_scaled_shapes_mass_extent_and_rest(n_envs, show_viewer):
    box_size = 0.2
    sphere_radius = 0.1
    scale = 1.5 if n_envs == 0 else np.array([1.0, 1.5])
    scale_1d = np.atleast_1d(scale)

    scene = gs.Scene(
        show_viewer=show_viewer,
        rigid_options=gs.options.RigidOptions(
            enable_geom_scaling=True,
            batch_links_info=True,
            gravity=(0.0, 0.0, -9.81),
            dt=0.01,
        ),
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(
            size=(box_size, box_size, box_size),
            pos=(0.0, 0.0, 0.3),
        ),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=sphere_radius,
            pos=(1.0, 0.0, 0.3),
        ),
    )
    # Convex mesh exercises the scaled support path against real vertices, which primitives do not.
    duck = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.1,
            pos=(-1.0, 0.0, 0.3),
            convexify=True,
        ),
    )
    scene.build(n_envs=n_envs)

    mass0 = box.get_mass()
    duck_extent0 = np.asarray(duck.get_AABB().cpu()).reshape(-1, 2, 3)
    for shape in (box, sphere, duck):
        shape.set_scale(scale)
    assert_allclose(box.get_scale(), scale, tol=1e-6)

    # A uniform scale s is a similarity transform: mass grows by s^3 and every extent by s (mesh included). The
    # extents are read at rest orientation before settling, isolating geometry scale from the drop.
    assert_allclose(box.get_mass() / mass0, scale**3, tol=1e-4)
    box_aabb = np.asarray(box.get_AABB().cpu()).reshape(-1, 2, 3)
    assert_allclose(box_aabb[:, 1] - box_aabb[:, 0], box_size * scale_1d[:, None], tol=1e-4)
    duck_extent1 = np.asarray(duck.get_AABB().cpu()).reshape(-1, 2, 3)
    duck_ratio = (duck_extent1[:, 1] - duck_extent1[:, 0]) / (duck_extent0[:, 1] - duck_extent0[:, 0])
    assert_allclose(duck_ratio, scale_1d[:, None], tol=1e-4)

    for _ in range(80):
        scene.step()

    box_z = np.atleast_1d(box.get_pos().cpu().numpy()[..., 2])
    sphere_z = np.atleast_1d(sphere.get_pos().cpu().numpy()[..., 2])
    assert_allclose(box_z, 0.5 * box_size * scale_1d, tol=5e-3)
    assert_allclose(sphere_z, sphere_radius * scale_1d, tol=5e-3)

    # The scaled mesh rests on the plane without tunnelling through it.
    duck_min_z = np.asarray(duck.get_AABB().cpu()).reshape(-1, 2, 3)[:, 0, 2]
    assert (duck_min_z > -0.01).all()


@pytest.mark.required
def test_isotropic_scale_rotational_inertia():
    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            enable_geom_scaling=True,
            batch_links_info=True,
            gravity=(0.0, 0.0, 0.0),
            integrator=gs.integrator.Euler,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.3, 0.4),
            pos=(0.0, 0.0, 1.0),
        ),
    )
    scene.build(n_envs=2)

    # Constant torque gives angular accel tau/I, and inertia scales by s^5, so env 1 (2x) spins at 1/2^5 of env 0.
    scale = np.array([1.0, 2.0])
    box.set_scale(scale)

    torque = torch.zeros((2, box.n_dofs), dtype=gs.tc_float, device=gs.device)
    torque[:, 5] = 0.01
    for _ in range(50):
        box.control_dofs_force(torque)
        scene.step()

    omega = box.get_dofs_velocity()[:, 5]
    assert_allclose(omega[1] / omega[0], 1.0 / 2.0**5, tol=1e-3)


@pytest.mark.required
def test_per_env_scale_render_matches_snapshot(png_snapshot, show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
        rigid_options=gs.options.RigidOptions(
            enable_geom_scaling=True,
            batch_links_info=True,
            gravity=(0.0, 0.0, -9.81),
            dt=0.01,
        ),
        vis_options=gs.options.VisOptions(
            shadow=False,
        ),
    )
    # Finite plane stays inside the camera frustum for the CI software renderer.
    scene.add_entity(
        morph=gs.morphs.Plane(
            plane_size=(6.0, 6.0),
        ),
    )
    duck = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck/duck.obj",
            scale=0.003,
            euler=(90.0, 0.0, 90.0),
            convexify=True,
        ),
    )
    camera = scene.add_camera(
        res=(240, 120),
        pos=(0.0, 4.5, 2.2),
        lookat=(0.0, 0.0, 0.7),
        fov=50.0,
        GUI=show_viewer,
    )
    scene.build(n_envs=2)

    # Symmetric placement keeps both ducks equidistant from the camera; env 1 is 2x and starts higher to avoid an
    # initial penetration. A short drop lets the image also show the physics resolving per env.
    duck.set_pos(np.array([[-1.2, 0.0, 0.25], [1.2, 0.0, 0.65]]))
    duck.set_scale(np.array([1.0, 2.0]))
    for _ in range(20):
        scene.step()

    duck_min_z = np.asarray(duck.get_AABB().cpu()).reshape(2, 2, 3)[:, 0, 2]
    assert (duck_min_z > -0.02).all(), f"scaled duck penetrated the plane: min_z per env = {duck_min_z}"

    rgb = tensor_to_array(camera.render(rgb=True)[0])
    assert rgb_array_to_png_bytes(rgb) == png_snapshot
