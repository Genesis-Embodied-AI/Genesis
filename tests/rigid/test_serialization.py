import json
import xml.etree.ElementTree as ET
import zipfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.scene import SCENE_FORMAT
from genesis.utils.misc import get_assets_dir
from genesis.utils.serialization import ARRAY_MEMBER, MANIFEST_NAME

from ..utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["two_free_boxes"])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_export_and_load_rigid(
    n_envs, xml_path, mimic_hinges, urdf_with_external_mesh, xacro_robot, tmp_path, show_viewer, caplog
):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 1.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.5),
            size=(0.2, 0.2, 0.2),
        ),
        material=gs.materials.Rigid(
            friction=0.7,
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.ImageTexture(
                image_path="textures/checker.png",
            ),
        ),
    )
    payload = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.9),
            size=(0.1, 0.1, 0.1),
            fixed=True,
            batch_fixed_verts=True,
        ),
        surface=gs.surfaces.Default(
            diffuse_texture=gs.textures.BatchTexture.from_images(
                image_paths=["textures/indoor_bright.png"],
            ),
        ),
    )
    payload.attach(
        box,
        parent_link_name=box.base_link.name,
        pos=(0.0, 0.0, 0.2),
    )
    brick = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=xml_path,
        ),
    )
    # Articulated and tied by an equality, so the per-degree-of-freedom arrays and the constraints both travel
    # non-trivially.
    articulated = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=ET.tostring(mimic_hinges, encoding="unicode"),
            pos=(-0.5, 0.0, 0.5),
        ),
    )
    # Several meshes as one entity, which carries its geometry and no file at all
    pieces = [trimesh.creation.box(extents=(0.08, 0.08, 0.08)) for _ in range(2)]
    pieces[1].apply_translation((0.1, 0.0, 0.0))
    mesh_set = scene.add_entity(
        morph=gs.morphs.MeshSet(
            files=pieces,
            pos=(0.0, 0.5, 0.5),
        ),
    )
    # A link whose frame is anchored on the geometry it holds, given a pose offset as well
    anchored = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.05,
            pos=(0.0, -0.6, 0.7),
            euler=(0.0, 0.0, 30.0),
            align=True,
        ),
    )
    # A height field, coefficients only the asset states, and a variant per environment, none of it held by a link.
    # Placed away from the rest so none of them ever meets another.
    terrain = scene.add_entity(
        morph=gs.morphs.Terrain(
            horizontal_scale=1.0,
            vertical_scale=0.2,
            height_field=torch.zeros((8, 8)),
            pos=(10.0, 10.0, 0.0),
        ),
    )
    drone = scene.add_entity(
        morph=gs.morphs.Drone(
            file="urdf/drones/cf2x.urdf",
            pos=(20.0, 20.0, 1.0),
        ),
    )
    heterogeneous = scene.add_entity(
        morph=(
            gs.morphs.Box(
                pos=(30.0, 30.0, 1.0),
                size=(0.2, 0.2, 0.2),
            ),
            gs.morphs.Box(
                pos=(30.0, 30.0, 1.0),
                size=(0.3, 0.1, 0.2),
            ),
        ),
    )
    kinematic_arm = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=ET.tostring(mimic_hinges, encoding="unicode"),
            pos=(40.0, 40.0, 1.0),
        ),
        material=gs.materials.Kinematic(),
    )
    # A second entity from the same asset as 'anchored', so both share its geometry in the written file
    twin = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.05,
            pos=(0.0, -1.4, 0.7),
            euler=(0.0, 0.0, 30.0),
            align=True,
        ),
    )
    # An entity attached to another is simulated as one tree with it, so a restored scene must hold the pair
    holder = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(50.0, 50.0, 1.0),
            size=(0.2, 0.2, 0.2),
            fixed=True,
        ),
    )
    mounted = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(50.0, 50.0, 1.3),
            size=(0.1, 0.1, 0.1),
        ),
    )
    mounted.attach(
        holder,
        parent_link_name=holder.base_link.name,
        pos=(0.0, 0.0, 0.15),
    )
    # A scale stated per axis, which its declaration accepts beside a single factor. Its surface holds a texture
    # read from an HDR file. Such a texture keeps a path rather than pixels, so no file can carry it.
    hdr_path = tmp_path / "sky.hdr"
    hdr_path.touch()
    stretched = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=Path(get_assets_dir()) / "meshes" / "sphere.obj",
            scale=(0.05, 0.1, 0.15),
            pos=(60.0, 60.0, 1.0),
            enable_custom_vverts=True,
        ),
        surface=gs.surfaces.Default(
            normal_texture=gs.textures.ImageTexture(
                image_path=str(hdr_path),
                encoding="linear",
            ),
        ),
    )
    # A XACRO is expanded at creation, so this morph holds the model its file was parsed into
    expanded = scene.add_entity(
        morph=gs.morphs.URDF(
            file=xacro_robot,
            pos=(70.0, 70.0, 1.0),
        ),
    )
    # An asset may name a mesh outside its own directory, which the mesh states as an absolute path of its own
    far_mesh = scene.add_entity(
        morph=gs.morphs.URDF(
            file=urdf_with_external_mesh,
            pos=(80.0, 80.0, 1.0),
        ),
    )
    # A sensor and a recorder observe the scene rather than taking part in it, and a file holds no code to run.
    # All three are written without.
    scene.add_sensor(
        gs.sensors.IMU(
            entity_idx=box.idx,
        ),
    )
    scene.start_recording(
        data_func=box.get_pos,
        rec_options=gs.recorders.CSVFile(
            filename=str(tmp_path / "recorded.csv"),
        ),
    )
    scene.register_pre_step_callback(lambda: None)
    scene.build(n_envs=n_envs)

    # Where the driven entity stands before it is driven anywhere, which is the configuration a file carries.
    driven_rest = kinematic_arm.get_links_pos()
    kinematic_arm.set_dofs_position(0.2)
    # An anchored link reports the pose it received, not its geometry's anchor frame. A restored entity must report
    # the same.
    anchored_pos = anchored.get_pos()
    anchored_quat = anchored.get_quat()
    stretched.set_vverts(stretched.get_vverts() + 0.01)
    for _ in range(20):
        scene.step()

    exported = tmp_path / f"authored{SCENE_FORMAT}"
    caplog.clear()
    with caplog.at_level("WARNING"):
        scene.export(exported)
    # A sensor and vertices written at runtime take no part in the simulation. The file is written without them,
    # and the export names each rather than rejecting.
    left_out = " ".join(record.getMessage() for record in caplog.records)
    assert "IMUSensor" in left_out
    assert "normal_texture" in left_out
    assert "CSVFileWriter" in left_out
    assert "pre_step_callback" in left_out
    assert "visual vertices" in left_out
    # The manifest records each asset by bare filename, so the archive reads the same on any machine.
    with zipfile.ZipFile(exported) as archive:
        stored = archive.read(MANIFEST_NAME).decode()
    assert str(tmp_path) not in stored
    assert str(get_assets_dir()) not in stored
    assert '"two_free_boxes.xml"' in stored
    # An entity's creation uses the description its build resolved, and that description carries the geometry.
    # Genesis therefore reads the asset file once and never again.
    Path(xml_path).unlink()

    restored = gs.Scene.load(exported)
    assert not restored.is_built
    restored.build(n_envs=n_envs)
    assert restored.n_envs == scene.n_envs
    assert_equal(restored.options.sim.dt, scene.options.sim.dt)
    assert_equal(restored.options.rigid.dt, scene.options.rigid.dt)
    assert [entity.name for entity in restored.entities] == [entity.name for entity in scene.entities]
    assert_equal(restored.entities[1].material.friction, box.material.friction)
    # A texture is drawn from its pixels, which travel, and the name of the image is all that is kept of its path.
    assert_equal(restored.entities[1].surface.diffuse_texture.image_array, box.surface.diffuse_texture.image_array)
    assert restored.entities[1].surface.diffuse_texture.image_path == "checker.png"
    # A batch holds textures of its own, and each keeps the name of the image it was read from
    assert restored.entities[2].surface.diffuse_texture.textures[0].image_path == "indoor_bright.png"
    # A texture holding a path rather than pixels stands for nothing, in the surface and in the meshes drawn with it
    assert restored.entities[14].surface.normal_texture is None
    assert restored.entities[14].vgeoms[0].vmesh.surface.normal_texture is None
    # A morph names the asset it was created from, since the description stands for what was parsed out of it
    assert restored.entities[15].morph.file == "two_link.urdf.xacro"
    # A mesh states the asset it was read from by name, wherever that asset stood
    assert restored.entities[16].geoms[0].mesh.metadata["mesh_path"] == "sphere.obj"
    # The document that entity was created from names that mesh as well, and the name travels without its directory
    assert "external_meshes" not in restored.entities[16].morph.file
    assert 'filename="sphere.obj"' in restored.entities[16].morph.file
    # A scale per axis comes back as the three factors it was given, and the geometry it produced with them
    assert_equal(restored.entities[14].morph.scale, stretched.morph.scale)
    assert_equal(restored.entities[14].geoms[0].mesh.verts, stretched.geoms[0].mesh.verts)
    authored_by_index = dict(
        enumerate((brick, articulated, mesh_set, anchored, terrain, drone, heterogeneous), start=3)
    )
    authored_by_index.update({14: stretched, 15: expanded, 16: far_mesh})
    for index, authored in authored_by_index.items():
        assert restored.entities[index].n_links == authored.n_links
        assert restored.entities[index].n_geoms == authored.n_geoms
        assert_equal(restored.entities[index].get_mass(), authored.get_mass())

    # A kinematic entity's description holds its frames, the geoms that render it, and no dynamics. The restored
    # entity keeps the same links and degrees of freedom, with nothing simulated behind them.
    ghost = restored.entities[10]
    assert isinstance(ghost.material, gs.materials.Kinematic)
    assert (ghost.n_links, ghost.n_dofs) == (kinematic_arm.n_links, kinematic_arm.n_dofs)
    # At the configuration it was authored at rather than the one it was driven to, as every restored entity is.
    assert_equal(ghost.get_dofs_position(), 0.0)
    assert_equal(ghost.get_links_pos(), driven_rest)

    # An articulated entity travels as its degrees of freedom and the constraint tying two of them together, which
    # only the joint and equality descriptions carry.
    arm = restored.entities[4]
    assert (arm.n_dofs, arm.n_qs) == (articulated.n_dofs, articulated.n_qs)
    assert [joint.name for joint in arm.joints] == [joint.name for joint in articulated.joints]
    assert [equality.name for equality in arm.equalities] == ["joint_equality"]
    assert_equal(arm.get_dofs_limit(), articulated.get_dofs_limit())
    assert_equal(arm.get_dofs_armature(), articulated.get_dofs_armature())

    # An anchored link reports the pose it was given, not the frame its geometry was anchored on. That holds only
    # because the offset the anchoring left behind travels in the description.
    assert_equal(restored.entities[6].get_pos(), anchored_pos)
    assert_equal(restored.entities[6].get_quat(), anchored_quat)
    assert_allclose(anchored_pos[..., 2], 0.7, tol=gs.EPS)
    # A terrain is a height field, and no link describes it
    assert_equal(restored.entities[7].terrain_hf, terrain.terrain_hf)
    assert_equal(restored.entities[7].terrain_scale, terrain.terrain_scale)
    # The thrust and torque coefficients are stated by the asset and read nowhere else
    assert_equal(restored.entities[8].KF, drone.KF)
    assert_equal(restored.entities[8].KM, drone.KM)
    # A variant is dispatched per environment, the primary standing first
    assert len(restored.entities[9].desc.variants) == len(heterogeneous.desc.variants)

    # A fixed-base entity attached onto a moving one is no longer carried by the world, so its inertial comes from
    # its geoms. That mass is resolved when the entity is created, so it travels.
    assert_equal(restored.entities[2].get_mass(), payload.get_mass())

    # An attached entity hangs from the link it names, so the pair comes back as one tree standing at its mount
    mounted_again = restored.entities[13]
    assert mounted_again.is_attached
    assert mounted_again.base_link.parent_idx == restored.entities[12].base_link.idx
    assert_allclose(mounted_again.get_pos(), (50.0, 50.0, 1.15), tol=gs.EPS)

    # The file carries the authored scene and none of its simulated state. The box therefore stands where the scene
    # placed it, not where it fell, and falls again from there.
    assert_equal(restored.entities[1].get_pos()[..., 2], 0.5)
    assert (box.get_pos()[..., 2] < 0.5 - gs.EPS).all()
    for _ in range(20):
        restored.step()
    assert_equal(restored.entities[1].get_pos(), box.get_pos())

    # Entities built from one asset share the geometry post-processing gives them. A file holds one copy of it, so
    # a restored pile of one asset costs the memory of one.
    for scene_geoms, twin_geoms in (
        (anchored.geoms, twin.geoms),
        (restored.entities[6].geoms, restored.entities[11].geoms),
    ):
        for geom, twin_geom in zip(scene_geoms, twin_geoms):
            assert np.shares_memory(geom.mesh.verts, twin_geom.mesh.verts)
            assert np.shares_memory(geom.mesh.faces, twin_geom.mesh.faces)
            assert np.shares_memory(geom.mesh.get_unique_edges(), twin_geom.mesh.get_unique_edges())
            assert np.shares_memory(geom.mesh.get_vert_adjacency()[0], twin_geom.mesh.get_vert_adjacency()[0])
            assert geom.mesh.inertial is twin_geom.mesh.inertial

    # A restored scene keeps exactly the description it loaded, so a second export produces the same file
    again = tmp_path / f"again{SCENE_FORMAT}"
    restored.export(again)
    with zipfile.ZipFile(exported) as first, zipfile.ZipFile(again) as second:
        assert first.namelist() == second.namelist()
        assert json.loads(first.read(MANIFEST_NAME)) == json.loads(second.read(MANIFEST_NAME))
        assert first.read(ARRAY_MEMBER) == second.read(ARRAY_MEMBER)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_export_before_build(n_envs, tmp_path, show_viewer):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 1.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.5),
            size=(0.2, 0.2, 0.2),
        ),
    )
    # Adding an entity resolves its description, so a scene is written before its build as readily as after
    exported = tmp_path / f"authored{SCENE_FORMAT}"
    scene.export(exported)
    scene.build(n_envs=n_envs)

    # A file states no environment layout, so whoever opens one builds it with the layout they ask for
    restored = gs.Scene.load(exported)
    assert not restored.is_built
    assert [entity.name for entity in restored.entities] == [entity.name for entity in scene.entities]
    restored.build(n_envs=n_envs)
    assert restored.n_envs == scene.n_envs
    assert_equal(restored.entities[1].get_mass(), box.get_mass())
    for _ in range(20):
        restored.step()
    assert_equal(restored.entities[1].get_pos()[..., 2] > 0.0, True)

    # A build resolves the options it sizes its buffers from, so a file written after one states those
    built = tmp_path / f"built{SCENE_FORMAT}"
    scene.export(built)
    assert scene.options.rigid.max_contacts is not None
    assert gs.Scene.load(built).options.rigid.max_contacts == scene.options.rigid.max_contacts


@pytest.mark.required
def test_load_rejects_damaged_file(mimic_hinges, tmp_path, show_viewer):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.MJCF(
            file=ET.tostring(mimic_hinges, encoding="unicode"),
        ),
    )
    exported = tmp_path / f"authored{SCENE_FORMAT}"
    scene.export(exported)

    # A file records what the classes its values load against are made of. A field it holds that Genesis no longer
    # declares means its values say something else, so the file is rejected by the name of what moved.
    with zipfile.ZipFile(exported) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    original = json.loads(members[MANIFEST_NAME])
    manifest = deepcopy(original)
    manifest["schema"]["RigidLinkDescription"].append("inertial_frame")
    members[MANIFEST_NAME] = json.dumps(manifest).encode()
    with zipfile.ZipFile(exported, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(gs.GenesisException, match="inertial_frame.*RigidLinkDescription"):
        gs.Scene.load(exported)

    # An enumeration member whose value moved means every value written under it names something else.
    manifest = deepcopy(original)
    manifest["schema"]["JOINT_TYPE"]["REVOLUTE"] = 7
    members[MANIFEST_NAME] = json.dumps(manifest).encode()
    with zipfile.ZipFile(exported, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(gs.GenesisException, match="REVOLUTE=7"):
        gs.Scene.load(exported)

    # A field a class gained since is not a mismatch: an option states which of its fields were given, and a
    # description field a file leaves out stands at the default it declares.
    manifest = deepcopy(original)
    manifest["schema"]["RigidLinkDescription"].remove("invweight")
    members[MANIFEST_NAME] = json.dumps(manifest).encode()
    with zipfile.ZipFile(exported, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    gs.Scene.load(exported)

    # A file states a layout under each class name it holds. Anything else standing there is rejected before it is
    # read.
    manifest = deepcopy(original)
    manifest["schema"] = []
    members[MANIFEST_NAME] = json.dumps(manifest).encode()
    with zipfile.ZipFile(exported, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(gs.GenesisException, match="not the manifest Genesis writes"):
        gs.Scene.load(exported)

    # A value shaped like a container is rejected where a field states one number.
    manifest = deepcopy(original)
    manifest["held"]["scene"]["entities"][0]["links"][0]["mass"] = []
    members[MANIFEST_NAME] = json.dumps(manifest).encode()
    with zipfile.ZipFile(exported, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(gs.GenesisException, match="holds a list where"):
        gs.Scene.load(exported)

    # A file arrives from wherever it was written, so a value contradicting its declaration is rejected rather than
    # reaching a constructor.
    manifest = deepcopy(original)
    manifest["held"]["scene"] = 3
    members[MANIFEST_NAME] = json.dumps(manifest).encode()
    with zipfile.ZipFile(exported, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(gs.GenesisException, match="values a Genesis scene is not made of"):
        gs.Scene.load(exported)

    # A value standing where another kind is declared is rejected as well, and so is one its field does not admit.
    manifest = deepcopy(original)
    manifest["held"]["scene"]["options"]["values"]["sim"]["values"]["dt"] = -1.0
    members[MANIFEST_NAME] = json.dumps(manifest).encode()
    with zipfile.ZipFile(exported, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(gs.GenesisException, match="states a 'dt' that SimOptions rejects"):
        gs.Scene.load(exported)

    manifest = deepcopy(original)
    manifest["held"]["scene"]["options"]["values"]["sim"]["values"]["dt"] = "oops"
    members[MANIFEST_NAME] = json.dumps(manifest).encode()
    with zipfile.ZipFile(exported, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(gs.GenesisException, match="holds a str where a float was declared"):
        gs.Scene.load(exported)


@pytest.mark.required
def test_export_rejects_unsupported_physics(tmp_path, show_viewer):
    # A scene holding what alters the simulation is rejected by name, since a file leaving it out would restore
    # other physics. No description carries a particle entity either.
    unsupported = gs.Scene(
        show_viewer=show_viewer,
    )
    unsupported.add_entity(morph=gs.morphs.Box(pos=(0.5, 0.5, 0.5), size=(0.2, 0.2, 0.2)))
    unsupported.add_entity(
        morph=gs.morphs.Box(pos=(0.5, 0.5, 0.8), size=(0.1, 0.1, 0.1)), material=gs.materials.MPM.Elastic()
    )
    unsupported.add_force_field(gs.force_fields.Wind(direction=(1.0, 0.0, 0.0)))
    with pytest.raises(gs.GenesisException, match="1 MPMEntity, 1 Wind cannot be exported"):
        unsupported.export(tmp_path / f"wind{SCENE_FORMAT}")
