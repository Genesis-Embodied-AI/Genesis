import numpy as np
import pytest

import genesis as gs


@pytest.fixture(scope="session")
def fem_material():
    """Fixture for common FEM material properties"""
    return gs.materials.FEM.Muscle(
        E=3.0e4,
        nu=0.45,
        rho=1000.0,
        model="stable_neohooken",
    )


@pytest.mark.parametrize("backend", [gs.cpu])
def test_multiple_fem_entities(fem_material, show_viewer):
    """Test adding multiple FEM entities to the scene"""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=5e-4,
            substeps=10,
            gravity=(0.0, 0.0, 0.0),
        ),
        fem_options=gs.options.FEMOptions(
            damping=45.0,
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


@pytest.mark.parametrize("backend", [gs.cpu])
def test_interior_tetrahedralized_vertex(fem_material, show_viewer, tmp_path):
    """
    Test tetrahedralization of a FEM entity with a mesh shape that introduces
    internal vertices during tetrahedralization:
      1. Verify all surface vertices lie exactly on the original quad faces of the mesh.
      2. Ensure the visualizer's mesh triangles match the FEM entity's surface triangles.
    """

    def _write_extruded_box(center, large_length, small_length, filename):
        cx, cy, cz = center
        hL = large_length / 2.0
        hl = small_length / 2.0

        z0 = cz + hL
        z1 = z0 + small_length

        verts = [
            # Vertices of large cube
            (cx - hL, cy - hL, cz - hL),  # v1
            (cx + hL, cy - hL, cz - hL),  # v2
            (cx + hL, cy + hL, cz - hL),  # v3
            (cx - hL, cy + hL, cz - hL),  # v4
            (cx - hL, cy - hL, cz + hL),  # v5
            (cx + hL, cy - hL, cz + hL),  # v6
            (cx + hL, cy + hL, cz + hL),  # v7
            (cx - hL, cy + hL, cz + hL),  # v8
            # Vertices of a extruded small cube on +Z surface of the large cube
            (cx - hl, cy - hl, z0),  # v9
            (cx + hl, cy - hl, z0),  # v10
            (cx + hl, cy + hl, z0),  # v11
            (cx - hl, cy + hl, z0),  # v12
            # Other vertices of the extruded small cube
            (cx - hl, cy - hl, z1),  # v13
            (cx + hl, cy - hl, z1),  # v14
            (cx + hl, cy + hl, z1),  # v15
            (cx - hl, cy + hl, z1),  # v16
        ]

        faces = []

        # Large cube
        faces += [
            (1, 2, 3, 4),  # -Z
            (1, 2, 6, 5),  # -Y
            (2, 3, 7, 6),  # +X
            (3, 4, 8, 7),  # +Y
            (4, 1, 5, 8),  # -X
        ]

        # Large cube (+Z)
        faces += [
            (5, 6, 10, 9),
            (6, 7, 11, 10),
            (7, 8, 12, 11),
            (8, 5, 9, 12),
        ]

        # Small cube (+Z)
        faces += [(9, 10, 14, 13), (10, 11, 15, 14), (11, 12, 16, 15), (12, 9, 13, 16), (13, 14, 15, 16)]

        # Write obj file
        with open(filename, "w", encoding="utf-8") as f:
            for x, y, z in verts:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
            f.write("\n")
            for face in faces:
                f.write("f " + " ".join(str(idx) for idx in face) + "\n")

        return verts, faces

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        fem_options=gs.options.FEMOptions(),
        show_viewer=show_viewer,
    )

    obj_path = tmp_path / "fem.obj"
    verts, faces = _write_extruded_box(
        center=[0.0, 0.0, 0.0],
        large_length=1.0,
        small_length=0.1,
        filename=str(obj_path),
    )

    fem = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=str(obj_path),
            nobisect=False,
            minratio=1.5,
            verbose=1,
        ),
        material=fem_material,
    )

    scene.build()

    state = fem.get_state()
    vertices = state.pos.cpu().numpy()
    surface_indices = np.unique(fem.surface_triangles)

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


@pytest.mark.parametrize("backend", [gs.cpu])
def test_maxvolume(fem_material, show_viewer, box_obj_path):
    """Test that imposing a maximum element volume constraint produces a finer mesh (i.e., more elements)."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        fem_options=gs.options.FEMOptions(),
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
