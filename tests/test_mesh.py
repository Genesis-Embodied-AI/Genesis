import os
import platform
from contextlib import nullcontext

import xml.etree.ElementTree as ET
import numpy as np
import pytest
import trimesh

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.gltf as gltf_utils
import genesis.utils.mesh as mu

from .utils import assert_allclose, assert_array_equal, get_hf_dataset


def extract_mesh(gs_mesh):
    """Extract vertices, normals, uvs, and faces from a gs.Mesh object."""
    vertices = gs_mesh.trimesh.vertices
    normals = gs_mesh.trimesh.vertex_normals
    uvs = gs_mesh.trimesh.visual.uv
    faces = gs_mesh.trimesh.faces

    indices = np.lexsort(
        [
            uvs[:, 1],
            uvs[:, 0],
            normals[:, 2],
            normals[:, 1],
            normals[:, 0],
            vertices[:, 2],
            vertices[:, 1],
            vertices[:, 0],
        ]
    )

    vertices = vertices[indices]
    normals = normals[indices]
    uvs = uvs[indices]
    invdices = np.argsort(indices)
    faces = invdices[faces]
    return vertices, faces, normals, uvs


def check_gs_meshes(gs_mesh1, gs_mesh2, mesh_name, vertices_tol, normals_tol):
    """Check if two gs.Mesh objects are equal."""
    vertices1, faces1, normals1, uvs1 = extract_mesh(gs_mesh1)
    vertices2, faces2, normals2, uvs2 = extract_mesh(gs_mesh2)

    assert_allclose(vertices1, vertices2, atol=vertices_tol, err_msg=f"Vertices match failed in mesh {mesh_name}.")
    assert_array_equal(faces1, faces2, err_msg=f"Faces match failed in mesh {mesh_name}.")
    assert_allclose(normals1, normals2, atol=normals_tol, err_msg=f"Normals match failed in mesh {mesh_name}.")
    assert_allclose(uvs1, uvs2, rtol=gs.EPS, err_msg=f"UVs match failed in mesh {mesh_name}.")


def check_gs_tm_meshes(gs_mesh, tm_mesh, mesh_name, vertices_tol, normals_tol):
    """Check if a gs.Mesh object and a trimesh.Trimesh object are equal."""
    assert_allclose(
        tm_mesh.vertices,
        gs_mesh.trimesh.vertices,
        tol=vertices_tol,
        err_msg=f"Vertices match failed in mesh {mesh_name}.",
    )
    assert_array_equal(
        tm_mesh.faces,
        gs_mesh.trimesh.faces,
        err_msg=f"Faces match failed in mesh {mesh_name}.",
    )
    assert_allclose(
        tm_mesh.vertex_normals,
        gs_mesh.trimesh.vertex_normals,
        tol=normals_tol,
        err_msg=f"Normals match failed in mesh {mesh_name}.",
    )
    if not isinstance(tm_mesh.visual, trimesh.visual.color.ColorVisuals):
        assert_allclose(
            tm_mesh.visual.uv,
            gs_mesh.trimesh.visual.uv,
            rtol=gs.EPS,
            err_msg=f"UVs match failed in mesh {mesh_name}.",
        )


def check_gs_tm_textures(gs_texture, tm_color, tm_image, default_value, dim, material_name, texture_name):
    """Check if a gs.Texture object and a trimesh.Texture object are equal."""
    if isinstance(gs_texture, gs.textures.ColorTexture):
        tm_color = tm_color or (default_value,) * dim
        assert_allclose(
            tm_color,
            gs_texture.color,
            rtol=gs.EPS,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
    elif isinstance(gs_texture, gs.textures.ImageTexture):
        tm_color = tm_color or (1.0,) * dim
        assert_allclose(
            tm_color,
            gs_texture.image_color,
            rtol=gs.EPS,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
        assert_array_equal(
            tm_image,
            gs_texture.image_array,
            err_msg=f"Texture mismatch for material {material_name} in {texture_name}.",
        )


def check_gs_textures(gs_texture1, gs_texture2, default_value, material_name, texture_name):
    """Check if two gs.Texture objects are equal."""
    if gs_texture1 is None:
        gs_texture1, gs_texture2 = gs_texture2, gs_texture1
    if gs_texture1 is not None:
        gs_texture1 = gs_texture1.check_simplify()
    if gs_texture2 is not None:
        gs_texture2 = gs_texture2.check_simplify()

    if isinstance(gs_texture1, gs.textures.ColorTexture):
        gs_color2 = (default_value,) * len(gs_texture1.color) if gs_texture2 is None else gs_texture2.color
        assert_allclose(
            gs_texture1.color,
            gs_color2,
            rtol=gs.EPS,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
    elif isinstance(gs_texture1, gs.textures.ImageTexture):
        assert isinstance(gs_texture2, gs.textures.ImageTexture)
        assert_allclose(
            gs_texture1.image_color,
            gs_texture2.image_color,
            rtol=gs.EPS,
            err_msg=f"Color mismatch for material {material_name} in {texture_name}.",
        )
        assert_array_equal(
            gs_texture1.image_array,
            gs_texture2.image_array,
            err_msg=f"Texture mismatch for material {material_name} in {texture_name}.",
        )
    else:
        assert gs_texture1 is None and gs_texture2 is None, (
            f"Both textures should be None for material {material_name} in {texture_name}."
        )


def check_gs_surfaces(gs_surface1, gs_surface2, material_name):
    """Check if two gs.Surface objects are equal."""
    check_gs_textures(gs_surface1.get_texture(), gs_surface2.get_texture(), 1.0, material_name, "color")
    check_gs_textures(gs_surface1.opacity_texture, gs_surface2.opacity_texture, 1.0, material_name, "opacity")
    check_gs_textures(gs_surface1.roughness_texture, gs_surface2.roughness_texture, 1.0, material_name, "roughness")
    check_gs_textures(gs_surface1.metallic_texture, gs_surface2.metallic_texture, 0.0, material_name, "metallic")
    check_gs_textures(gs_surface1.normal_texture, gs_surface2.normal_texture, 0.0, material_name, "normal")
    check_gs_textures(gs_surface1.emissive_texture, gs_surface2.emissive_texture, 0.0, material_name, "emissive")


# ==================== Scale Tests ====================


@pytest.mark.required
@pytest.mark.parametrize("scale", [(0.5, 2.0, 8.0), (2.0, 2.0, 2.0)])
@pytest.mark.parametrize("mesh_file", ["meshes/camera/camera.glb", "meshes/axis.obj"])
def test_morph_scale(scale, mesh_file, tmp_path):
    urdf_path = tmp_path / "model.urdf"
    urdf_path.write_text(
        f"""<robot name="cannon">
              <link name="base">
                <visual>
                  <geometry><mesh filename="{mu.get_asset_path(mesh_file)}"/></geometry>
                </visual>
              </link>
            </robot>
         """
    )

    scene = gs.Scene(show_viewer=False)
    obj_orig = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=mesh_file,
            file_meshes_are_zup=False,
            pos=(0, 0, 1.0),
            scale=1.0,
            convexify=False,
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(1.0, 0.0, 0.0, 1.0),
        ),
    )
    for vgeom in obj_orig.vgeoms:
        mesh_orig = vgeom.vmesh.trimesh
        mesh_orig.apply_transform(mu.Y_UP_TRANSFORM)
        mesh_orig.apply_scale(scale)

    obj_scaled = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=mesh_file,
            file_meshes_are_zup=True,
            pos=(0, 0, 1.0),
            scale=scale,
            convexify=False,
            fixed=True,
        ),
        surface=gs.surfaces.Default(
            color=(0.0, 1.0, 0.0, 1.0),
        ),
    )
    assert obj_orig.n_vgeoms == obj_scaled.n_vgeoms

    is_isotropic = np.unique(scale).size == 1
    with nullcontext() if is_isotropic else pytest.raises(gs.GenesisException):
        robot_scaled = scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                file_meshes_are_zup=True,
                pos=(0, 0, 1.0),
                scale=scale,
                convexify=False,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.0, 0.0, 1.0, 1.0),
            ),
        )
        assert robot_scaled.n_vgeoms == obj_scaled.n_vgeoms

    for i_vg in range(obj_orig.n_vgeoms):
        mesh_orig = obj_orig.vgeoms[i_vg].vmesh.trimesh.copy()
        mesh_orig.apply_transform(gu.trans_quat_to_T(obj_orig.base_link.pos, obj_orig.base_link.quat))
        mesh_scaled = obj_scaled.vgeoms[i_vg].vmesh.trimesh.copy()
        mesh_scaled.apply_transform(gu.trans_quat_to_T(obj_scaled.base_link.pos, obj_scaled.base_link.quat))
        assert_allclose(mesh_orig.vertices, mesh_scaled.vertices, tol=gs.EPS)

        if is_isotropic:
            mesh_robot_scaled = robot_scaled.vgeoms[i_vg].vmesh.trimesh.copy()
            mesh_robot_scaled.apply_transform(
                gu.trans_quat_to_T(robot_scaled.base_link.pos, robot_scaled.base_link.quat)
            )
            assert_allclose(mesh_robot_scaled.vertices, mesh_scaled.vertices, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("mesh_file", ["glb/combined_transform.glb", "yup_zup_coverage/cannon_y_-z.stl"])
def test_urdf_scale(mesh_file, tmp_path, show_viewer):
    SCALE_FACTOR = 2.0

    asset_path = get_hf_dataset(pattern=mesh_file)

    urdf_path = tmp_path / "model.urdf"
    urdf_path.write_text(
        f"""<robot name="shoe">
              <link name="base">
                <visual>
                  <geometry><mesh filename="{os.path.join(asset_path, mesh_file)}"/></geometry>
                </visual>
              </link>
            </robot>
         """
    )

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    obj_1 = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            convexify=False,
            fixed=True,
        ),
    )
    mesh_1 = obj_1.vgeoms[0].vmesh.trimesh
    obj_2 = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            scale=SCALE_FACTOR,
            convexify=False,
            fixed=True,
        ),
    )
    mesh_2 = obj_2.vgeoms[0].vmesh.trimesh

    assert_allclose(SCALE_FACTOR * mesh_1.extents, mesh_2.extents, tol=gs.EPS)


# ==================== Y-Up Coordinate Tests ====================


@pytest.mark.required
def test_mesh_yup(show_viewer):
    scene = gs.Scene(show_viewer=show_viewer)

    asset_path = get_hf_dataset(pattern="yup_zup_coverage/*")

    glb_y = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/yup_zup_coverage/cannon_y.glb",
            convexify=False,
            fixed=True,
            file_meshes_are_zup=False,
        ),
    )
    glb_geom_y = glb_y.vgeoms[0]
    glb_z = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/yup_zup_coverage/cannon_z.glb",
            convexify=False,
            fixed=True,
            file_meshes_are_zup=True,
        ),
    )
    glb_geom_z = glb_z.vgeoms[0]
    stl_y = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/yup_zup_coverage/cannon_y_-z.stl",
            convexify=False,
            fixed=True,
            file_meshes_are_zup=False,
        ),
    )
    stl_geom_y = stl_y.vgeoms[0]
    stl_z = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/yup_zup_coverage/cannon_z_y.stl",
            convexify=False,
            fixed=True,
        ),
    )
    stl_geom_z = stl_z.vgeoms[0]
    obj_y = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/yup_zup_coverage/cannon_y_-z.obj",
            convexify=False,
            fixed=True,
            file_meshes_are_zup=False,
        ),
    )
    obj_geom_y = obj_y.vgeoms[0]
    obj_z = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/yup_zup_coverage/cannon_z_y.obj",
            convexify=False,
            fixed=True,
        ),
    )
    obj_geom_z = obj_z.vgeoms[0]

    if show_viewer:
        scene.build()

    assert not glb_geom_y.vmesh.metadata["imported_as_zup"]
    assert not glb_geom_z.vmesh.metadata["imported_as_zup"]
    assert not stl_geom_y.vmesh.metadata["imported_as_zup"]
    assert stl_geom_z.vmesh.metadata["imported_as_zup"]
    assert not obj_geom_y.vmesh.metadata["imported_as_zup"]
    assert obj_geom_z.vmesh.metadata["imported_as_zup"]

    for geom in (glb_geom_y, glb_geom_z, stl_geom_y, stl_geom_z, obj_geom_y, obj_geom_z):
        mesh = geom.vmesh.copy()
        mesh.apply_transform(gu.trans_quat_to_T(geom.link.pos, geom.link.quat))
        assert_allclose(mesh.trimesh.center_mass, (-0.012, -0.142, 0.397), tol=0.002)


@pytest.mark.required
@pytest.mark.parametrize(
    "mesh_file, file_meshes_are_zup",
    [("yup_zup_coverage/cannon_z.glb", True), ("yup_zup_coverage/cannon_y_-z.stl", False)],
)
def test_urdf_yup(mesh_file, file_meshes_are_zup, tmp_path, show_viewer):
    asset_path = get_hf_dataset(pattern=mesh_file)
    urdf_path = tmp_path / "model.urdf"
    urdf_path.write_text(
        f"""<robot name="cannon">
              <link name="base">
                <visual>
                  <geometry><mesh filename="{os.path.join(asset_path, mesh_file)}"/></geometry>
                </visual>
              </link>
            </robot>
         """
    )

    scene = gs.Scene(show_viewer=show_viewer)
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
            convexify=False,
            fixed=True,
            file_meshes_are_zup=file_meshes_are_zup,
        ),
    )
    mesh = robot.vgeoms[0].vmesh

    if show_viewer:
        scene.build()

    assert_allclose(mesh.trimesh.center_mass, (-0.012, -0.142, 0.397), tol=0.002)


# ==================== Geometry Parsing Tests ====================


@pytest.mark.required
@pytest.mark.parametrize("precision", ["32"])
@pytest.mark.parametrize("glb_file", ["glb/combined_srt.glb", "glb/combined_transform.glb"])
def test_glb_parse_geometry(glb_file, tol):
    """Test glb mesh geometry parsing."""
    asset_path = get_hf_dataset(pattern=glb_file)
    glb_file = os.path.join(asset_path, glb_file)
    gs_meshes = gltf_utils.parse_mesh_glb(
        glb_file,
        group_by_material=False,
        scale=None,
        is_mesh_zup=True,
        surface=gs.surfaces.Default(),
    )

    tm_scene = trimesh.load(glb_file, process=False)
    tm_meshes = {}
    for node_name in tm_scene.graph.nodes_geometry:
        transform, geometry_name = tm_scene.graph[node_name]
        ts_mesh = tm_scene.geometry[geometry_name].copy(include_cache=True)
        ts_mesh = ts_mesh.apply_transform(transform)
        tm_meshes[geometry_name] = ts_mesh
    assert len(tm_meshes) == len(gs_meshes)

    for gs_mesh in gs_meshes:
        mesh_name = gs_mesh.metadata["name"]
        tm_mesh = tm_meshes[mesh_name]
        check_gs_tm_meshes(gs_mesh, tm_mesh, mesh_name, tol, tol)


@pytest.mark.required
def test_urdf_mesh_processing(tmp_path, show_viewer):
    stl_file = "1707/base_link.stl"
    asset_path = get_hf_dataset(pattern=stl_file)
    stl_path = os.path.join(asset_path, stl_file)

    urdf_path = tmp_path / "model.urdf"
    urdf_path.write_text(
        f"""<robot name="shoe">
              <link name="base">
                <visual>
                  <geometry><mesh filename="{stl_path}"/></geometry>
                </visual>
              </link>
            </robot>
         """
    )

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=stl_path,
        ),
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
        ),
    )

    tmesh_obj_col = obj.geoms[0].mesh.trimesh
    tmesh_obj_vis = obj.vgeoms[0].vmesh.trimesh
    tmesh_robot_vis = robot.vgeoms[0].vmesh.trimesh

    assert len(tmesh_obj_col.vertices) != len(tmesh_obj_vis.vertices)
    assert len(tmesh_obj_vis.vertices) == len(tmesh_robot_vis.vertices)
    assert len(tmesh_obj_vis.faces) == len(tmesh_robot_vis.faces)

    tmesh = trimesh.Trimesh(vertices=tmesh_obj_vis.vertices, faces=tmesh_obj_vis.faces, process=True)
    assert len(tmesh.vertices) != len(tmesh_obj_vis.vertices)


# ==================== Material/Texture Parsing Tests ====================


@pytest.mark.required
@pytest.mark.parametrize("glb_file", ["glb/chopper.glb"])
def test_glb_parse_material(glb_file):
    """Test glb mesh geometry parsing."""
    asset_path = get_hf_dataset(pattern=glb_file)
    glb_file = os.path.join(asset_path, glb_file)
    gs_meshes = gltf_utils.parse_mesh_glb(
        glb_file,
        group_by_material=True,
        scale=None,
        is_mesh_zup=True,
        surface=gs.surfaces.Default(),
    )

    tm_scene = trimesh.load(glb_file, process=False)
    tm_materials = {}
    for geometry_name in tm_scene.geometry:
        ts_mesh = tm_scene.geometry[geometry_name]
        ts_material = ts_mesh.visual.material.copy()
        tm_materials[ts_material.name] = ts_material

    assert len(tm_materials) == len(gs_meshes)
    for gs_mesh in gs_meshes:
        material_name = gs_mesh.metadata["name"]
        tm_material = tm_materials[material_name]
        gs_material = gs_mesh.surface

        assert isinstance(tm_material, trimesh.visual.material.PBRMaterial)
        check_gs_tm_textures(
            gs_material.get_texture(),
            tm_material.baseColorFactor,
            np.array(tm_material.baseColorTexture),
            1.0,
            3,
            material_name,
            "color",
        )

        if tm_material.metallicRoughnessTexture is not None:
            tm_mr_image = np.array(tm_material.metallicRoughnessTexture)
            tm_roughness_image = tm_mr_image[:, :, 1]
            tm_metallic_image = tm_mr_image[:, :, 2]
        else:
            tm_roughness_image, tm_metallic_image = None, None
        check_gs_tm_textures(
            gs_material.roughness_texture,
            tm_material.roughnessFactor,
            tm_roughness_image,
            1.0,
            1,
            material_name,
            "roughness",
        )
        check_gs_tm_textures(
            gs_material.metallic_texture,
            tm_material.metallicFactor,
            tm_metallic_image,
            0.0,
            1,
            material_name,
            "metallic",
        )

        if tm_material.emissiveFactor is None and tm_material.emissiveFactor is None:
            assert gs_material.emissive_texture is None
        else:
            check_gs_tm_textures(
                gs_material.emissive_texture,
                tm_material.emissiveFactor,
                np.array(tm_material.emissiveTexture),
                0.0,
                3,
                material_name,
                "emissive",
            )


@pytest.fixture
def material_mjcf(tmp_path):
    """Generate an MJCF model with materials and geom-level colors."""
    mjcf = ET.Element("mujoco", model="materials")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", armature="0.0")

    # Define materials with different properties (at top level, not in default)
    asset = ET.SubElement(mjcf, "asset")
    ET.SubElement(
        asset,
        "material",
        name="red_material",
        rgba="1.0 0.0 0.0 0.6",
        specular="0.5",
        shininess="0.3",
    )

    worldbody = ET.SubElement(mjcf, "worldbody")
    floor = ET.SubElement(worldbody, "body", name="/worldbody/floor")
    ET.SubElement(floor, "geom", type="plane", pos="0. 0. 0.", size="40. 40. 40.")

    # Box with red material (material-level rgba)
    box1 = ET.SubElement(worldbody, "body", name="/worldbody/box1", pos="-0.3 0. 0.3")
    ET.SubElement(
        box1,
        "geom",
        type="box",
        size="0.2 0.2 0.2",
        pos="0. 0. 0.",
        material="red_material",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(box1, "joint", name="/worldbody/box1_joint", type="free")

    # Box with geom-level rgba (no material, tests geom-level color)
    box2 = ET.SubElement(worldbody, "body", name="/worldbody/box2", pos="0.0 0. 0.6")
    ET.SubElement(
        box2,
        "geom",
        type="box",
        size="0.2 0.2 0.2",
        pos="0. 0. 0.",
        rgba="0.0 1.0 0.0 1.0",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(box2, "joint", name="/worldbody/box2_joint", type="free")

    # Write to temporary file
    xml_tree = ET.ElementTree(mjcf)
    file_path = str(tmp_path / "material_mjcf.xml")
    xml_tree.write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path


@pytest.mark.parametrize("precision", ["32"])
def test_mjcf_parse_material(material_mjcf, tol):
    """Test that MJCF materials and geom colors are correctly parsed."""
    scene = gs.Scene()
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=material_mjcf,
            scale=1.0,
            convexify=False,
        ),
        material=gs.materials.Rigid(rho=1000.0),
    )
    scene.build()

    # Find boxes by their names
    box1_vgeom = None
    box2_vgeom = None
    for link in entity.links:
        if link.name == "/worldbody/box1":
            box1_vgeom = link.vgeoms[0]
        elif link.name == "/worldbody/box2":
            box2_vgeom = link.vgeoms[0]
    assert box1_vgeom is not None, "box1 not found"
    assert box2_vgeom is not None, "box2 not found"

    # Check red material (box1) - material-level rgba
    box1_surface = box1_vgeom.vmesh.surface
    box1_roughness = mu.glossiness_to_roughness(0.3 * 128.0)
    check_gs_textures(
        box1_surface.diffuse_texture, gs.textures.ColorTexture(color=(1.0, 0.0, 0.0)), 1.0, "box1", "color"
    )
    check_gs_textures(box1_surface.roughness_texture, None, box1_roughness, "box1", "roughness")
    check_gs_textures(box1_surface.opacity_texture, None, 0.6, "box1", "opacity")

    box2_surface = box2_vgeom.vmesh.surface
    check_gs_textures(
        box2_surface.diffuse_texture, gs.textures.ColorTexture(color=(0.0, 1.0, 0.0)), 1.0, "box2", "color"
    )
    check_gs_textures(box2_surface.opacity_texture, None, 1.0, "box2", "opacity")


@pytest.mark.required
def test_2_channels_luminance_alpha_textures(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    asset_path = get_hf_dataset(pattern="fridge/*")
    fridge = scene.add_entity(
        gs.morphs.URDF(
            file=f"{asset_path}/fridge/fridge.urdf",
            fixed=True,
        )
    )


@pytest.mark.required
def test_plane_texture_path_preservation(show_viewer):
    """Test that plane primitives preserve texture paths in metadata."""
    scene = gs.Scene(show_viewer=show_viewer, show_FPS=False)
    plane = scene.add_entity(gs.morphs.Plane())

    # The texture path should be stored in metadata
    assert plane.vgeoms[0].vmesh.metadata["texture_path"] == "textures/checker.png"


@pytest.mark.required
@pytest.mark.parametrize(
    "n_channels, float_type",
    [
        (1, np.float32),  # grayscale → H×W
        (2, np.float64),  # L+A       → H×W×2
    ],
)
def test_urdf_with_float_texture_glb(tmp_path, show_viewer, n_channels, float_type):
    vertices = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

    mesh = trimesh.Trimesh(vertices, faces, process=False)

    H = W = 16
    if n_channels == 1:
        img = np.random.rand(H, W).astype(float_type)
    else:
        img = np.random.rand(H, W, n_channels).astype(float_type)

    mesh.visual = trimesh.visual.texture.TextureVisuals(
        uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        material=trimesh.visual.material.SimpleMaterial(image=img),
    )

    glb_path = tmp_path / f"tex_{n_channels}c.glb"
    urdf_path = tmp_path / f"tex_{n_channels}c.urdf"
    trimesh.Scene([mesh]).export(glb_path)
    urdf_path.write_text(
        f"""<robot name="tex{n_channels}c">
              <link name="base">
                <visual>
                  <geometry><mesh filename="{glb_path}"/></geometry>
                </visual>
              </link>
            </robot>
         """
    )

    scene = gs.Scene(show_viewer=show_viewer, show_FPS=False)
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=urdf_path,
        ),
    )


# ==================== Surface Reconstruction Tests ====================


@pytest.mark.required
def test_splashsurf_surface_reconstruction(show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    scene.add_entity(
        material=gs.materials.SPH.Liquid(),
        morph=gs.morphs.Box(
            pos=(0.15, 0.15, 0.22),
            size=(0.25, 0.25, 0.4),
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.6, 1.0, 1.0),
            vis_mode="recon",
        ),
    )
    cam = scene.add_camera(
        pos=(1.3, 1.3, 0.8),
        lookat=(0.0, 0.0, 0.2),
        GUI=show_viewer,
    )
    scene.build()
    cam.render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)


# ==================== Mesh Processing/Caching Tests ====================


# FIXME: This test is taking too much time on some platform (~1200s)
# @pytest.mark.required
def test_convex_decompose_cache(monkeypatch):
    # Check if the convex decomposition cache is correctly tracked regardless of the scale

    # Monkeypatch the get_cvx_path function to track the cache path
    seen_paths = []
    real_get_cvx_path = mu.get_cvx_path

    def wrapped_get_cvx_path(verts, faces, opts):
        path = real_get_cvx_path(verts, faces, opts)
        seen_paths.append(path)
        return path

    monkeypatch.setattr(mu, "get_cvx_path", wrapped_get_cvx_path)

    # Monkeypatch the convex_decompose function to track the convex decomposition result
    seen_results = []
    real_convex_decompose = mu.convex_decompose

    def wrapped_convex_decompose(mesh, opts):
        result = real_convex_decompose(mesh, opts)
        seen_results.append(result)
        return result

    monkeypatch.setattr(mu, "convex_decompose", wrapped_convex_decompose)

    # First scene building to create the cache
    scene = gs.Scene(
        show_viewer=False,
    )
    first_scale = 2.0
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=first_scale,
            pos=(0, 0, 1.0),
            quat=(0, 0, 0, 1),
        ),
    )
    scene.build()

    # Second scene building, duck with different scale, translation, and rotation
    scene = gs.Scene(
        show_viewer=False,
    )
    second_scale = 4.0
    scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=second_scale,
            pos=(1.0, 0, 1.0),
            quat=(1, 0, 0, 0),
        ),
    )
    scene.build()

    assert len(seen_paths) == 2
    assert len(seen_results) == 2

    # scaled mesh should have the same cache path as the original mesh
    cached_path = seen_paths[0]
    scaled_path = seen_paths[-1]
    assert cached_path == scaled_path

    # check if the scaled parts match the scaled version of the original parts
    cached_parts = seen_results[0]
    scaled_parts = seen_results[-1]
    assert len(scaled_parts) == len(cached_parts)
    for scaled_part, cached_part in zip(scaled_parts, cached_parts):
        assert_allclose(scaled_part.vertices, cached_part.vertices * (second_scale / first_scale), rtol=1e-6)
        assert_array_equal(scaled_part.faces, cached_part.faces)
