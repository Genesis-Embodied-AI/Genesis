import numpy as np
import pytest
import torch
import igl

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .utils import assert_allclose, get_hf_dataset


@pytest.fixture(scope="session")
def fem_material():
    """Fixture for common FEM material properties"""
    return gs.materials.FEM.Muscle(
        E=3.0e4,
        nu=0.45,
        rho=1000.0,
        model="stable_neohookean",
    )


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_multiple_fem_entities(fem_material, show_viewer):
    """Test adding multiple FEM entities to the scene"""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-4,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        fem_options=gs.options.FEMOptions(
            damping=0.0,
        ),
        show_viewer=show_viewer,
    )

    # Add first FEM entity
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.5, -0.2, 0.3),
            radius=0.1,
        ),
        material=fem_material,
    )

    # Add second FEM entity
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
        material=fem_material,
    )

    # Build the scene
    scene.build()

    # Run simulation
    for _ in range(100):
        scene.step()


@pytest.mark.required
def test_interior_tetrahedralized_vertex(fem_material, show_viewer, box_obj_path, cube_verts_and_faces):
    """
    Test tetrahedralization of a FEM entity with a small maxvolume value that introduces
    internal vertices during tetrahedralization:
      1. Verify all surface vertices lie exactly on the original quad faces of the mesh.
      2. Ensure the visualizer's mesh triangles match the FEM entity's surface triangles.
    """
    verts, faces = cube_verts_and_faces

    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    fem = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=box_obj_path,
            nobisect=False,
            minratio=1.5,
            verbose=1,
            maxvolume=0.01,
        ),
        material=fem_material,
    )

    scene.build()

    state = fem.get_state()
    vertices = state.pos[0].cpu().numpy()
    surface_indices = np.unique(fem.surface_triangles)

    # Ensure there are interior vertices; this is a prerequisite for this test
    assert surface_indices.size < vertices.shape[0]

    # Verify each surface vertex lies on the original surface mesh
    def _point_on_surface(p, verts, faces, tol=1e-6):
        """Check if point p lies on any of the quad faces (as two triangles)."""
        for face in faces:
            # Convert 1-based face indices to 0-based
            idx = [i - 1 for i in face]
            # Extract vertices
            v0, v1, v2, v3 = [np.array(verts[i]) for i in idx]
            # Decompose quad into two triangles: (v0,v1,v2) and (v0,v2,v3)
            for tri in ((v0, v1, v2), (v0, v2, v3)):
                a, b, c = tri
                # Compute normal for plane
                n = np.cross(b - a, c - a)
                norm_n = np.linalg.norm(n)
                if norm_n < tol:
                    continue
                # Check distance to plane
                distance = abs(np.dot(n / norm_n, p - a))
                if distance > tol:
                    continue
                # Barycentric coordinates
                v0v1 = b - a
                v0v2 = c - a
                v0p = p - a
                dot00 = np.dot(v0v2, v0v2)
                dot01 = np.dot(v0v2, v0v1)
                dot02 = np.dot(v0v2, v0p)
                dot11 = np.dot(v0v1, v0v1)
                dot12 = np.dot(v0v1, v0p)
                denom = dot00 * dot11 - dot01 * dot01
                if abs(denom) < tol:
                    continue
                u = (dot11 * dot02 - dot01 * dot12) / denom
                v = (dot00 * dot12 - dot01 * dot02) / denom
                if u >= -tol and v >= -tol and (u + v) <= 1 + tol:
                    return True
        return False

    for idx in surface_indices:
        p = vertices[idx]
        assert _point_on_surface(
            p, verts, faces
        ), f"Surface vertex index {idx} with coordinate {p} does not lie on any original face"

    # Verify whether surface faces in the visualizer mesh matches the surface faces of the FEM entity
    rasterizer_context = scene.visualizer.context
    static_nodes = rasterizer_context.static_nodes
    fem_node_mesh = static_nodes[fem.uid].mesh

    (fem_node_primitive,) = fem_node_mesh.primitives
    fem_node_vertices = fem_node_primitive.positions
    fem_node_faces = fem_node_primitive.indices
    if fem_node_faces is None:
        fem_node_faces = np.arange(fem_node_vertices.shape[0]).reshape(-1, 3)

    def _make_triangle_set(verts, faces, tol=4):
        """
        Return a hashable, order-independent representation of a given set of triangle faces.

        Rounds each vertex coordinate to the given tolerance, sorts vertices within each triangle,
        and returns all triangles as a sorted tuple, eliminating any dependence on vertex or face order.
        """
        tri_set = set()
        for tri in faces:
            coords = [tuple(round(float(coord), tol) for coord in verts[i]) for i in tri]
            tri_set.add(tuple(sorted(coords)))
        return tuple(sorted(tri_set))

    # Triangles of FEM entity
    entity_tris = _make_triangle_set(vertices, fem.surface_triangles)

    # Triangles of visualizer
    viz_tris = _make_triangle_set(np.asarray(fem_node_vertices), np.asarray(fem_node_faces))

    assert entity_tris == viz_tris, (
        "FEM entity surface triangles and visualizer mesh triangles do not match.\n"
        f"Differences: {set(entity_tris) ^ set(viz_tris)}"
    )


@pytest.mark.required
def test_maxvolume(fem_material, show_viewer, box_obj_path):
    """Test that imposing a maximum element volume constraint produces a finer mesh (i.e., more elements)."""
    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    # Mesh without any maximum-element-volume constraint
    fem1 = scene.add_entity(
        morph=gs.morphs.Mesh(file=box_obj_path, nobisect=False, verbose=1),
        material=fem_material,
    )

    # Mesh with maximum element volume limited to 0.01
    fem2 = scene.add_entity(
        morph=gs.morphs.Mesh(file=box_obj_path, nobisect=False, maxvolume=0.01, verbose=1),
        material=fem_material,
    )

    assert len(fem1.elems) < len(fem2.elems), (
        f"Mesh with maxvolume=0.01 generated {len(fem2.elems)} elements; "
        f"expected more than {len(fem1.elems)} elements without a volume limit."
    )


@pytest.fixture(scope="session")
def fem_material_linear():
    """Fixture for common FEM linear material properties"""
    return gs.materials.FEM.Elastic()


# @pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_sphere_box_fall_implicit_fem_coupler(fem_material_linear, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    # Add first FEM entity
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.5, -0.2, 0.3),
            radius=0.1,
        ),
        material=fem_material_linear,
    )

    # Add second FEM entity
    scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
        material=fem_material_linear,
    )

    # Build the scene
    scene.build()

    # Run simulation
    for _ in range(200):
        scene.step()

    for entity in scene.entities:
        state = entity.get_state()
        min_pos_z = state.pos[..., 2].min()
        assert_allclose(
            min_pos_z,
            0.0,  # FIXME: Compute desired penetration analytically
            atol=5e-2,
            err_msg=f"Entity {entity.uid} minimum Z position {min_pos_z} is not close to 0.0.",
        )


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_sphere_fall_implicit_fem_sap_coupler(fem_material_linear, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.5, -0.2, 0.5),
            radius=0.1,
        ),
        material=fem_material_linear,
    )

    # Build the scene
    scene.build()

    # Run simulation
    for _ in range(100):
        scene.step()

    for entity in scene.entities:
        state = entity.get_state()
        min_pos_z = state.pos[..., 2].min()
        assert_allclose(
            min_pos_z,
            -1e-3,  # FIXME: Compute desired penetration analytically
            atol=1e-4,
            err_msg=f"Entity {entity.uid} minimum Z position {min_pos_z} is not close to -1e-3.",
        )


@pytest.fixture(scope="session")
def fem_material_linear_corotated():
    """Fixture for common FEM linear material properties"""
    return gs.materials.FEM.Elastic(model="linear_corotated")


# FIXME: Compilation is crashing on Apple Metal backend
@pytest.mark.required
def test_linear_corotated_sphere_fall_implicit_fem_sap_coupler(fem_material_linear_corotated, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            substeps=2,
        ),
        # Not using default fem_options to make it faster, linear material only need one iteration without linesearch
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.5, -0.2, 0.5),
            radius=0.1,
        ),
        material=fem_material_linear_corotated,
    )

    # Build the scene
    scene.build()

    # Run simulation
    for _ in range(100):
        scene.step()

    for entity in scene.entities:
        state = entity.get_state()
        pos = tensor_to_array(state.pos.reshape(-1, 3))
        min_pos_z = np.min(pos[..., 2])
        assert_allclose(
            min_pos_z,
            -1e-3,  # FIXME: Compute desired penetration analytically
            atol=1e-4,
            err_msg=f"Entity {entity.uid} minimum Z position {min_pos_z} is not close to -1e-3.",
        )
        BV, BF = igl.bounding_box(pos)
        scale = BV[0] - BV[-1]
        assert_allclose(
            scale,
            (0.2, 0.2, 0.2 - 1e-3),  # FIXME: Compute desired scale analytically
            atol=1e-3,
            err_msg=f"Entity {entity.uid} scale {scale} is not close to 0.2.",
        )


@pytest.fixture(scope="session")
def fem_material_linear_corotated_soft():
    """Fixture for common FEM linear material properties"""
    return gs.materials.FEM.Elastic(model="linear_corotated", E=1.0e5, nu=0.4)


# @pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_fem_sphere_box_self(fem_material_linear_corotated, fem_material_linear_corotated_soft, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        show_viewer=show_viewer,
    )

    # Add first FEM entity
    scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.1),
            radius=0.1,
        ),
        material=fem_material_linear_corotated,
    )

    # Add second FEM entity
    scale = 0.1
    asset_path = get_hf_dataset(pattern="meshes/cube8.obj")
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/meshes/cube8.obj",
            scale=scale,
            pos=(0.0, 0.0, scale * 4.0),
        ),
        material=fem_material_linear_corotated,
    )

    # Build the scene
    scene.build()
    # Run simulation
    for _ in range(200):
        scene.step()

    depths = (-1e-3, -2e-5)  # FIXME: Compute desired penetration analytically
    atols = (2e-4, 4e-6)
    for i, entity in enumerate(scene.entities):
        state = entity.get_state()
        min_pos_z = state.pos[..., 2].min()
        assert_allclose(
            min_pos_z,
            depths[i],
            atol=atols[i],
            err_msg=f"Entity {entity.uid} minimum Z position {min_pos_z} is not close to {depths[i]}.",
        )


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_box_hard_vertex_constraint(show_viewer):
    """
    Test if a box with hard vertex constraints has those vertices fixed,
    and that updating and removing constraints works correctly.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=1,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=False,
            gravity=(0.0, 0.0, -9.81),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
        material=gs.materials.FEM.Elastic(),
    )
    verts_idx = [0, 3]
    initial_target_poss = box.init_positions[verts_idx]

    scene.build(n_envs=2)

    if show_viewer:
        scene.draw_debug_spheres(poss=initial_target_poss, radius=0.02, color=(1, 0, 1, 0.8))

    box.set_vertex_constraints(verts_idx=verts_idx, target_poss=initial_target_poss)

    for _ in range(100):
        scene.step()

    positions = box.get_state().pos[0][verts_idx]
    assert_allclose(
        positions,
        initial_target_poss,
        tol=gs.EPS,
        err_msg="Vertices should stay at initial target positions with hard constraints",
    )
    new_target_poss = initial_target_poss + gs.tensor(
        [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
    )
    box.update_constraint_targets(verts_idx=verts_idx, target_poss=new_target_poss)

    for _ in range(100):
        scene.step()

    positions_after_update = box.get_state().pos[0][verts_idx]
    assert_allclose(
        positions_after_update,
        new_target_poss,
        tol=gs.EPS,
        err_msg="Vertices should be at new target positions after updating constraints",
    )

    box.remove_vertex_constraints()

    for _ in range(100):
        scene.step()

    positions_after_removal = box.get_state().pos[0][verts_idx]
    with np.testing.assert_raises(AssertionError):
        assert_allclose(
            positions_after_removal,
            new_target_poss,
            tol=1e-3,
            err_msg="Vertices should have moved after removing constraints",
        )


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_box_soft_vertex_constraint(show_viewer):
    """Test if a box with strong soft vertex constraints has those vertices near."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
            substeps=1,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=False,
            gravity=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        ),
        material=gs.materials.FEM.Elastic(),
    )
    verts_idx = [0, 1]
    target_poss = box.init_positions[verts_idx]

    scene.build()

    if show_viewer:
        scene.draw_debug_spheres(poss=target_poss, radius=0.02, color=(1, 0, 1, 0.8))

    box.set_vertex_constraints(
        verts_idx=verts_idx,
        target_poss=target_poss,
        is_soft_constraint=True,
        stiffness=2.0e5,
    )
    box.set_velocity(gs.tensor([1.0, 1.0, 1.0]) * 1e-2)

    for _ in range(500):
        scene.step()

    positions = box.get_state().pos[0][verts_idx]
    assert_allclose(
        positions,
        target_poss,
        tol=5e-5,
        err_msg="Vertices should be near target positions with strong soft constraints",
    )


# @pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_fem_articulated(fem_material_linear_corotated_soft, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.2),
            radius=0.2,
        ),
        material=fem_material_linear_corotated_soft,
    )

    asset_path = get_hf_dataset(pattern="heavy_three_joint_link.xml")
    link = scene.add_entity(
        gs.morphs.MJCF(file=f"{asset_path}/heavy_three_joint_link.xml", scale=0.5, pos=(-0.5, -0.5, 0.4)),
    )

    # Build the scene
    scene.build()
    for _ in range(300):
        scene.step()

    state = sphere.get_state()
    center = state.pos.mean(axis=(0, 1))
    min_pos_z = state.pos[..., 2].min()
    assert_allclose(
        min_pos_z,
        -1.0e-3,  # FIXME: Compute desired penetration analytically
        atol=2e-4,
        err_msg=f"Sphere minimum Z position {min_pos_z} is not close to -1.0e-3.",
    )
    assert_allclose(
        center,
        np.array([0.0, 0.0, 0.2], dtype=np.float32),
        atol=0.2,
        err_msg=f"Sphere center {center} moves too far from [0.0, 0.0, 0.2].",
    )

    link_verts = link.get_verts()
    center = link_verts.mean(axis=0)
    min_pos_z = link_verts[..., 2].min()
    assert_allclose(
        min_pos_z,
        -1.0e-4,  # FIXME: Compute desired penetration analytically
        atol=5e-5,
        err_msg=f"Link minimum Z position {min_pos_z} is not close to -1.0e-4.",
    )
    assert_allclose(
        center,
        np.array([-0.5, -0.5, 0.04], dtype=np.float32),
        atol=0.2,
        err_msg=f"Link center {center} moves too far from [-0.5, -0.5, 0.04].",
    )


@pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_implicit_hard_vertex_constraint(fem_material_linear_corotated, show_viewer):
    """
    Test if a box with hard vertex constraints has those vertices fixed, and that updating and removing constraints
    works correctly.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=1,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            enable_vertex_constraints=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    asset_path = get_hf_dataset(pattern="meshes/cube8.obj")
    cube = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/meshes/cube8.obj",
            scale=0.1,
            pos=np.array([0.0, 0.0, 0.6], dtype=np.float32),
        ),
        material=fem_material_linear_corotated,
    )

    verts_idx = [0]
    initial_target_poss = cube.init_positions[verts_idx]

    scene.build()

    if show_viewer:
        sphere = scene.draw_debug_spheres(poss=initial_target_poss, radius=0.02, color=(1, 0, 1, 0.8))

    cube.set_vertex_constraints(verts_idx=verts_idx, target_poss=initial_target_poss)

    for _ in range(100):
        scene.step()

    positions = cube.get_state().pos[0][verts_idx]
    assert_allclose(
        positions,
        initial_target_poss,
        tol=gs.EPS,
        err_msg="Vertices should stay at initial target positions with hard constraints",
    )
    new_target_poss = initial_target_poss + 0.1
    cube.update_constraint_targets(verts_idx=verts_idx, target_poss=new_target_poss)
    if show_viewer:
        scene.clear_debug_object(sphere)
        sphere = scene.draw_debug_spheres(poss=new_target_poss, radius=0.02, color=(1, 0, 1, 0.8))
    for _ in range(100):
        scene.step()

    positions_after_update = cube.get_state().pos[0][verts_idx]
    assert_allclose(
        positions_after_update,
        new_target_poss,
        tol=gs.EPS,
        err_msg="Vertices should be at new target positions after updating constraints",
    )

    cube.remove_vertex_constraints()
    if show_viewer:
        scene.clear_debug_object(sphere)

    for _ in range(70):
        scene.step()

    state = cube.get_state()
    center = state.pos.mean(axis=(0, 1))
    assert_allclose(
        center,
        np.array([0.2, 0.13, 0.1], dtype=np.float32),
        atol=0.2,
        err_msg=f"Cube center {center} moves too far from [0.2, 0.13, 0.1] after removing constraints.",
    )

    velocity = state.vel.mean(axis=(0, 1))
    assert_allclose(
        velocity, 0.0, atol=4e-5, err_msg=f"Cube velocity {velocity} should be close to zero after settling."
    )

    # The contact requires some penetration to generate enough contact force to cancel out gravity
    min_pos_z = state.pos[..., 2].min()
    assert_allclose(
        min_pos_z,
        -2.0e-5,  # FIXME: Compute desired penetration analytically
        atol=5e-6,
        err_msg=f"Cube minimum Z position {min_pos_z} is not close to -2.0e-5.",
    )


# @pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_sphere_box_vertex_constraint(fem_material_linear_corotated, show_viewer):
    """
    Test if a box with hard vertex constraints has those vertices fixed, and collisiong with a sphere works correctly.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,
            substeps=1,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            enable_vertex_constraints=True,
        ),
        coupler_options=gs.options.SAPCouplerOptions(),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    asset_path = get_hf_dataset(pattern="meshes/cube8.obj")
    cube = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/meshes/cube8.obj",
            scale=0.1,
            pos=np.array([0.0, 0.0, 0.35], dtype=np.float32),
        ),
        material=fem_material_linear_corotated,
    )

    verts_idx = [0]
    initial_target_poss = cube.init_positions[verts_idx]

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.1),
            radius=0.1,
        ),
        material=fem_material_linear_corotated,
    )

    scene.build()
    if show_viewer:
        sphere_debug = scene.draw_debug_spheres(poss=initial_target_poss, radius=0.02, color=(1, 0, 1, 0.8))

    cube.set_vertex_constraints(verts_idx=verts_idx, target_poss=initial_target_poss)

    for _ in range(200):
        scene.step()

    pos = cube.get_state().pos
    fixed_pos = pos[0][verts_idx]
    assert_allclose(
        fixed_pos,
        initial_target_poss,
        tol=gs.EPS,
        err_msg="Vertices should stay at initial target positions with hard constraints",
    )

    state = sphere.get_state()
    center = state.pos.mean(axis=(0, 1))
    assert_allclose(
        center,
        np.array([0.4, 0.4, 0.1], dtype=np.float32),
        atol=0.2,
        err_msg=f"Sphere center {center} moved too much from initial position [0.4, 0.4, 0.1].",
    )

    # Using a larger tolerance here since the sphere is rolling, rolling friction is not accurately modeled.
    velocity = state.vel.mean(axis=(0, 1))
    assert_allclose(
        velocity, 0.0, atol=0.03, err_msg=f"Sphere velocity {velocity} should be close to zero after settling."
    )

    min_sphere_pos_z = state.pos[..., 2].min()
    assert_allclose(
        min_sphere_pos_z,
        -1e-3,  # FIXME: Compute desired penetration analytically
        atol=2e-4,
        err_msg=f"Sphere minimum Z position {min_sphere_pos_z} is not close to cube bottom surface.",
    )


@pytest.fixture(scope="session")
def fem_material_linear_corotated_rough():
    """Fixture for rough FEM linear material properties"""
    return gs.materials.FEM.Elastic(model="linear_corotated", friction_mu=1.0)


# @pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_franka_panda_grasp_cube(fem_material_linear_corotated_rough, show_viewer):
    """
    Test if the Franka Panda can successfully grasp the cube.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            pcg_threshold=1e-10,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            pcg_threshold=1e-10,
            sap_convergence_atol=1e-10,
            sap_convergence_rtol=1e-10,
            linesearch_ftol=1e-10,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    friction = 1.0
    force = 1.0
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=friction, friction=friction),
    )
    asset_path = get_hf_dataset(pattern="meshes/cube8.obj")
    cube = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/meshes/cube8.obj",
            scale=0.02,
            pos=np.array([0.65, 0.0, 0.02], dtype=np.float32),
        ),
        material=fem_material_linear_corotated_rough,
    )

    scene.build()
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()

    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )

    franka.control_dofs_position(qpos[:-2], motors_dof)

    # hold
    for i in range(10):
        scene.step()
    # grasp
    for i in range(30):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-force, -force]), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]),
        quat=np.array([0, 1, 0, 0]),
    )
    for i in range(100):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-force, -force]), fingers_dof)
        scene.step()
        if i == 49:
            old_pos = cube.get_state().pos.mean(axis=(0, 1))
        if i == 99:
            new_pos = cube.get_state().pos.mean(axis=(0, 1))

    assert_allclose(
        new_pos, old_pos, atol=5e-4, err_msg=f"Cube should be not moving much. Old pos: {old_pos}, new pos: {new_pos}."
    )


@pytest.fixture(scope="session")
def fem_material_linear_corotated_soft_rough():
    """Fixture for soft rough FEM linear material properties"""
    return gs.materials.FEM.Elastic(model="linear_corotated", E=1e5, nu=0.4, friction_mu=1.0)


# @pytest.mark.required
@pytest.mark.parametrize("precision", ["64"])
def test_franka_panda_grasp_soft_sphere(fem_material_linear_corotated_soft_rough, show_viewer):
    """
    Test if the Franka Panda can successfully grasp the soft sphere.
    """
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60,
            substeps=2,
        ),
        fem_options=gs.options.FEMOptions(
            use_implicit_solver=True,
            pcg_threshold=1e-10,
        ),
        coupler_options=gs.options.SAPCouplerOptions(
            pcg_threshold=1e-10,
            sap_convergence_atol=1e-10,
            sap_convergence_rtol=1e-10,
            linesearch_ftol=1e-10,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    friction = 1.0
    force = 1.0
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
        material=gs.materials.Rigid(coup_friction=friction, friction=friction),
    )
    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.02,
            pos=np.array([0.65, 0.0, 0.02], dtype=np.float32),
        ),
        material=fem_material_linear_corotated_soft_rough,
    )

    scene.build()
    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    qpos = np.array([-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04])
    franka.set_qpos(qpos)
    scene.step()

    end_effector = franka.get_link("hand")
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.135]),
        quat=np.array([0, 1, 0, 0]),
    )

    franka.control_dofs_position(qpos[:-2], motors_dof)

    # hold
    for i in range(10):
        scene.step()
    # grasp
    for i in range(30):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-force, -force]), fingers_dof)
        scene.step()

    # lift
    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.3]),
        quat=np.array([0, 1, 0, 0]),
    )
    for i in range(100):
        franka.control_dofs_position(qpos[:-2], motors_dof)
        franka.control_dofs_force(np.array([-force, -force]), fingers_dof)
        scene.step()
        if i == 49:
            old_pos = sphere.get_state().pos.mean(axis=(0, 1))
        if i == 99:
            new_pos = sphere.get_state().pos.mean(axis=(0, 1))
    pos_np = sphere.get_state().pos.cpu().numpy().reshape(-1, 3)
    BV, BF = igl.bounding_box(pos_np)
    deformation = BV[0, :] - BV[-1, :]
    assert_allclose(
        new_pos,
        old_pos,
        atol=5e-4,
        err_msg=f"Sphere should be not moving much. Old pos: {old_pos}, new pos: {new_pos}.",
    )
    assert_allclose(
        deformation,
        np.array([0.04, 0.038, 0.04], dtype=np.float32),
        atol=1e-3,
        err_msg=f"Sphere deformation should be small. Deformation: {deformation}.",
    )
