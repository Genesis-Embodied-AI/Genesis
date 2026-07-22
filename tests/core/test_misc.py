"""Tests for the entity naming system."""

import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.point_cloud as pc
from genesis.utils.misc import tensor_to_array

from ..utils import assert_allclose, assert_equal


@pytest.mark.required
def test_repr_does_not_crash():
    inline_mjcf = '<mujoco model="probe"><worldbody><body><geom type="box" size="1 1 1"/></body></worldbody></mujoco>'

    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane())
    scene.add_entity(morph=gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    panda = scene.add_entity(morph=gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    inline = scene.add_entity(morph=gs.morphs.MJCF(file=inline_mjcf))
    het = scene.add_entity(
        morph=(
            gs.morphs.Box(size=(0.2, 0.2, 0.2)),
            gs.morphs.Cylinder(radius=0.05, height=0.2),
        ),
    )
    scene.add_entity(
        morph=(
            gs.morphs.Box(size=(0.2, 0.2, 0.2)),
            gs.morphs.Sphere(radius=0.1),
        ),
        material=gs.materials.Kinematic(),
    )
    cam = scene.add_camera(
        res=(64, 64),
        pos=(1.0, 1.0, 1.0),
        lookat=(0.0, 0.0, 0.0),
    )
    scene.build(n_envs=2)

    # Every printable object renders without raising, across both the brief and the full colorized form.
    for obj in (scene, scene.entities, cam, scene.sim.rigid_solver):
        assert repr(obj)
    for entity in scene.entities:
        assert entity._repr_brief()
        assert repr(entity)
        for morph in entity.morphs:
            assert repr(morph)
        sub_objects = [*entity.links, *entity.joints, *entity.vgeoms]
        if isinstance(entity, gs.engine.entities.RigidEntity):
            sub_objects += list(entity.geoms)
        for sub in sub_objects:
            assert sub._repr_brief()
            assert repr(sub)

    # Sanity on the parts worth enforcing.
    # A file-based morph shows its path; an in-memory description is identified by its model name, not dumped.
    assert "panda.xml" in repr(panda.main_morph)
    assert "<inline probe>" in inline.main_morph.__repr_name__()
    assert inline_mjcf not in repr(inline.main_morph)
    # A heterogeneous entity reports its variants instead of collapsing to a single ambiguous morph.
    assert "morph variants" in het._repr_brief()


@pytest.mark.required
def test_scene_destroy_cleans_up_simulator():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane())
    scene.build()
    scene.step()

    assert scene._sim is not None

    scene.destroy()

    assert scene._sim is None
    assert scene._visualizer is None


@pytest.mark.required
def test_scene_destroy_idempotent():
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(morph=gs.morphs.Plane())
    scene.build()
    scene.step()

    scene.destroy()
    assert scene._sim is None

    scene.destroy()
    assert scene._sim is None


@pytest.mark.required
@pytest.mark.parametrize("raise_before_build", [True, False])
def test_destroy_after_aborted_camera_build(monkeypatch, raise_before_build):
    from genesis.engine.sensors.camera import RasterizerCameraSensor

    scene = gs.Scene(show_viewer=False)
    camera = scene.add_sensor(gs.sensors.RasterizerCameraOptions(res=(64, 64)))

    # Capture the shared metadata reference now; SensorManager.destroy() drops its dict entry,
    # but the dataclass instance itself stays alive through our local reference so we can
    # inspect its fields after teardown.
    shared_metadata = camera._shared_metadata

    # Inject a bug either at build entry (no metadata population) or after the original build
    # has populated renderer / context / sensors / image_cache.
    original_build = RasterizerCameraSensor.build

    def buggy_build(self):
        if not raise_before_build:
            original_build(self)
        raise RuntimeError("injected camera build failure")

    monkeypatch.setattr(RasterizerCameraSensor, "build", buggy_build)

    with pytest.raises(RuntimeError, match="injected camera build failure"):
        scene.build()

    if raise_before_build:
        assert shared_metadata.renderer is None
    else:
        assert shared_metadata.renderer is not None
        assert shared_metadata.context is not None
        assert shared_metadata.sensors is not None
        assert shared_metadata.image_cache is not None

    # Track shared_metadata.destroy() invocations via instance-level shadow. Assigning to the
    # instance __dict__ takes precedence over class-level lookup for this instance only, so
    # neither the class nor any other metadata instance is affected. The `del` reverts the
    # instance to plain class-level lookup before any finalizer can fire.
    original_destroy = shared_metadata.destroy
    destroy_call_count = [0]

    def tracked_destroy():
        destroy_call_count[0] += 1
        original_destroy()

    shared_metadata.destroy = tracked_destroy
    try:
        scene.destroy()
    finally:
        del shared_metadata.destroy

    assert destroy_call_count[0] == 1
    assert shared_metadata.renderer is None
    assert shared_metadata.context is None
    assert shared_metadata.sensors is None
    assert shared_metadata.image_cache is None


@pytest.mark.required
def test_auto_and_user_names():
    scene = gs.Scene()

    # Auto-generated name
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    assert box.name.startswith("box_")

    # Multiple identical entities should have unique names
    box2 = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    assert box2.name.startswith("box_")
    assert box.name != box2.name

    # User-specified name
    sphere = scene.add_entity(gs.morphs.Sphere(radius=0.1), name="my_sphere")
    assert sphere.name == "my_sphere"

    # Duplicate name raises error
    with pytest.raises(Exception, match="already exists"):
        scene.add_entity(gs.morphs.Cylinder(radius=0.1, height=0.2), name="my_sphere")


@pytest.mark.required
def test_get_entity_by_name():
    scene = gs.Scene()

    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)), name="test_box")
    assert scene.get_entity(name="test_box") is box

    # Non-existent name raises error
    with pytest.raises(Exception, match="not found"):
        scene.get_entity(name="nonexistent")


@pytest.mark.required
def test_get_entity_by_uid():
    scene = gs.Scene()

    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))

    # Short UID lookup (7-character prefix shown in terminal)
    assert scene.get_entity(uid=box.uid.short()) is box

    # Non-existent UID raises error
    with pytest.raises(Exception, match="not found"):
        scene.get_entity(uid=gs.UID().short())


@pytest.mark.required
def test_entity_names_property():
    scene = gs.Scene()

    # Use "B" then "A" to confirm insertion order (not sorted)
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)), name="B")
    scene.add_entity(gs.morphs.Sphere(radius=0.1), name="A")
    assert tuple(scene.entity_names) == ("B", "A")


@pytest.mark.required
def test_urdf_mjcf_names_from_file():
    scene = gs.Scene()

    # URDF: plane.urdf has <robot name="plane">
    urdf_entity = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf"))
    assert urdf_entity.name.startswith("plane_")

    # MJCF: panda.xml has <mujoco model="panda">
    mjcf_entity = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    assert mjcf_entity.name.startswith("panda_")

    # Multiple URDF entities should have unique names
    urdf_entity2 = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf"))
    assert urdf_entity2.name.startswith("plane_")
    assert urdf_entity.name != urdf_entity2.name


@pytest.mark.required
def test_morph_orientation_offset_resolution():
    quat_90z = gu.xyz_to_quat(np.array((0.0, 0.0, 90.0)), rpy=True, degrees=True)

    # An unset offset resolves to identity, whether omitted or passed as the None unset sentinel.
    assert_equal(gs.morphs.Box(size=(0.1, 0.1, 0.1)).offset_quat, (1.0, 0.0, 0.0, 0.0))
    assert_equal(gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_quat=None).offset_quat, (1.0, 0.0, 0.0, 0.0))

    # 'offset_euler' resolves into 'offset_quat'.
    assert_allclose(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_euler=(0.0, 0.0, 90.0)).offset_quat, quat_90z, tol=gs.EPS
    )

    # An explicit 'offset_quat' is kept verbatim.
    assert_equal(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_quat=(0.0, 1.0, 0.0, 0.0)).offset_quat, (0.0, 1.0, 0.0, 0.0)
    )

    # A None 'offset_quat' means unset, so a serializer can forward it unconditionally alongside 'offset_euler';
    # it resolves to 'offset_euler'.
    assert_allclose(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_euler=(0.0, 0.0, 90.0), offset_quat=None).offset_quat,
        quat_90z,
        tol=gs.EPS,
    )

    # 'offset_euler' and 'offset_quat' are mutually exclusive.
    with pytest.raises(Exception, match="'offset_euler' and 'offset_quat' cannot both be set"):
        gs.morphs.Box(size=(0.1, 0.1, 0.1), offset_euler=(0.0, 0.0, 90.0), offset_quat=(0.0, 1.0, 0.0, 0.0))


@pytest.mark.required
def test_coacd_options_pca_validation():
    gs.options.CoacdOptions(pca=False)
    with pytest.raises(gs.GenesisException, match="pca=True"):
        gs.options.CoacdOptions(pca=True)


@pytest.mark.required
def test_fps_algorithm_core():
    # Shape, dtype, determinism, anchor-on-no-seed, and invalid n_samples all in one test.
    points = np.random.default_rng(1).random((50, 3))
    out_a = pc.furthest_point_sample(points, 10, seed=42)
    out_b = pc.furthest_point_sample(points, 10, seed=42)
    assert out_a.shape == (10, 3)
    assert out_a.dtype == gs.np_float
    assert_equal(out_a, out_b)

    # With seed=None the first sample is the first input point (deterministic anchor).
    anchor = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    out_anchor = pc.furthest_point_sample(anchor, 3, seed=None)
    assert_allclose(out_anchor[0], anchor[0], tol=1e-5)

    with pytest.raises(gs.GenesisException):
        pc.furthest_point_sample(np.zeros((5, 3)), 10, seed=None)


@pytest.mark.required
def test_fps_mesh_sampling_end_to_end():
    # Shape/dtype/determinism, on-surface points, minimum separation, and box-aligned normals all in one scene.
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    points_a = pc.sample_mesh_point_cloud(mesh.vertices, mesh.faces, 16, n_candidates=400, seed=7, use_cache=False)
    points_b = pc.sample_mesh_point_cloud(mesh.vertices, mesh.faces, 16, n_candidates=400, seed=7, use_cache=False)
    assert points_a.shape == (16, 3)
    assert points_a.dtype == gs.np_float
    assert_equal(points_a, points_b)

    _, dist, _ = mesh.nearest.on_surface(points_a)
    assert dist.max() < 1e-5

    pairwise = np.linalg.norm(points_a[:, None, :] - points_a[None, :, :], axis=-1)
    np.fill_diagonal(pairwise, np.inf)
    assert pairwise.min() > 0.2

    # Box face normals are axis-aligned unit vectors. Unit-length + `max(|n|) == 1` is sufficient: by Pythagoras the
    # other two components must be ~0.
    _, normals = pc.sample_mesh_point_cloud(
        mesh.vertices, mesh.faces, 32, n_candidates=800, seed=0, use_cache=False, return_normals=True
    )
    assert_allclose(np.linalg.norm(normals, axis=1), 1.0, tol=1e-5)
    assert_allclose(np.abs(normals).max(axis=1), 1.0, tol=1e-4)


@pytest.mark.required
def test_fps_cache_round_trip():
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    # Run both `return_normals` paths: first call writes the cache; second call hits it; outputs identical.
    for return_normals in (True, False):
        kwargs = dict(
            verts=mesh.vertices,
            faces=mesh.faces,
            n_points=5,
            n_candidates=10,
            return_normals=return_normals,
            seed=7,
        )
        path = pc.get_fps_pc_path(**kwargs)
        if os.path.exists(path):
            os.remove(path)
        first = pc.sample_mesh_point_cloud(**kwargs, use_cache=True)
        assert os.path.exists(path)
        cached = pc.sample_mesh_point_cloud(**kwargs, use_cache=True)
        assert_equal(first, cached)


@pytest.mark.required
def test_gs_mesh_sample_point_cloud_wrapper():
    mesh = trimesh.creation.box((0.2, 0.4, 0.6))
    gmesh = gs.Mesh.from_trimesh(mesh)
    points, normals = gmesh.sample_point_cloud(10, n_candidates=300, seed=67, use_cache=False, return_normals=True)
    assert points.shape == (10, 3)
    assert normals.shape == (10, 3)
    _, dist, _ = mesh.nearest.on_surface(points)
    assert dist.max() < 1e-5
    assert_allclose(np.linalg.norm(normals, axis=1), 1.0, tol=1e-5)


@pytest.fixture
def two_link_fixed_urdf():
    robot = ET.Element("robot", name="two_link_fixed")
    ET.SubElement(robot, "link", name="base")
    child = ET.SubElement(robot, "link", name="child")
    collision = ET.SubElement(child, "collision")
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", size="0.1 0.1 0.1")
    joint = ET.SubElement(robot, "joint", name="weld", type="fixed")
    ET.SubElement(joint, "parent", link="base")
    ET.SubElement(joint, "child", link="child")
    ET.SubElement(joint, "origin", xyz="0 0 0.1")
    return ET.tostring(robot, encoding="unicode")


@pytest.mark.required
def test_solver_state_change_subscribers(show_viewer, two_link_fixed_urdf):
    # Imported lazily: the solver package pulls in quadrants kernels that need gs.qd_float, set only by gs.init.
    from genesis.engine.solvers.base_solver import StateChange, Subscriber

    scene = gs.Scene(show_viewer=show_viewer)
    plane = scene.add_entity(gs.morphs.Plane())
    # A fixed entity whose collision geometry lives on a fixed child link, the common URDF/MJCF layout: teleporting
    # it through its base link must reach subscribers watching the child.
    tower = scene.add_entity(
        gs.morphs.URDF(
            file=two_link_fixed_urdf,
            pos=(2.0, 0.0, 0.0),
            fixed=True,
            merge_fixed_links=False,
        )
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.5),
        ),
    )
    scene.build(n_envs=2)

    solver = scene.sim.rigid_solver

    # Eager mode: a callback fires immediately on each matching change and nothing is retained.
    eager_events = []
    eager = Subscriber(
        to=frozenset({StateChange.GEOMETRY}),
        callback=lambda change, envs_idx: eager_events.append((change, envs_idx)),
    )
    solver.subscribe(eager)
    # Lazy mode: matching changes accumulate on the Subscriber handle until cleared.
    lazy = Subscriber(to=frozenset({StateChange.GEOMETRY}))
    solver.subscribe(lazy)
    # A DYNAMICS-only subscriber must stay silent on GEOMETRY changes (filter).
    dynamics = Subscriber(to=frozenset({StateChange.DYNAMICS}))
    solver.subscribe(dynamics)

    # zero_velocity=False isolates the pure GEOMETRY change (a default set_pos also zeroes velocity; see below).
    cube.set_pos([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], zero_velocity=False)
    # Eager fired once with the right category; envs_idx forwarded verbatim (None == every env).
    assert len(eager_events) == 1
    assert eager_events[0][0] is StateChange.GEOMETRY
    assert eager_events[0][1] is None
    # Lazy accumulated the category; the DYNAMICS subscriber saw nothing; eager retains nothing.
    assert lazy.pending == frozenset({StateChange.GEOMETRY})
    assert dynamics.pending == frozenset()
    assert eager.pending == frozenset()

    # A targeted setter forwards the exact env subset to the eager callback.
    cube.set_pos([[0.0, 0.0, 3.0]], envs_idx=[1], zero_velocity=False)
    assert len(eager_events) == 2
    forwarded = eager_events[1][1]
    assert forwarded is not None
    assert int(np.atleast_1d(tensor_to_array(forwarded))[0]) == 1

    # Lazy state is idempotent across repeated changes and resets on clear().
    assert lazy.pending == frozenset({StateChange.GEOMETRY})
    lazy.clear()
    assert lazy.pending == frozenset()

    # A velocity setter is a DYNAMICS change only: it wakes the DYNAMICS subscriber, not the GEOMETRY ones (setting a
    # velocity does not move the surface).
    cube.set_dofs_velocity([0.0] * cube.n_dofs)
    assert dynamics.pending == frozenset({StateChange.DYNAMICS})
    assert lazy.pending == frozenset()
    assert len(eager_events) == 2

    # Reads never notify.
    solver.get_links_pos()
    solver.get_links_quat()
    assert len(eager_events) == 2
    assert lazy.pending == frozenset()

    # Physics integration mutates state through kernels, not a tagged method, so it never notifies.
    scene.step()
    assert len(eager_events) == 2
    assert lazy.pending == frozenset()

    # A default set_pos both moves the link and zeroes its velocity, so a subscriber listening for either category
    # receives both - the accumulated union of every change the call produced.
    both = Subscriber(to=frozenset({StateChange.GEOMETRY, StateChange.DYNAMICS}))
    solver.subscribe(both)
    cube.set_pos([[0.0, 0.0, 4.0], [0.0, 0.0, 5.0]])
    assert both.pending == frozenset({StateChange.GEOMETRY, StateChange.DYNAMICS})

    # A link-filtered subscriber only wakes on changes that can affect its links. The plane's base link is fixed, so
    # the cube's base-pose setter (an explicit, disjoint link selection) and its configuration-space setters (which
    # can never displace a link without degrees of freedom) both stay silent.
    plane_watcher = Subscriber(to=frozenset({StateChange.GEOMETRY}), links_filter=[plane.base_link_idx])
    solver.subscribe(plane_watcher)
    cube.set_pos([[0.0, 0.0, 6.0], [0.0, 0.0, 7.0]], zero_velocity=False)
    cube.set_qpos([[0.0, 0.0, 8.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 9.0, 1.0, 0.0, 0.0, 0.0]])
    assert plane_watcher.pending == frozenset()
    # A base-pose setter naming a watched link wakes it.
    plane.set_pos((0.0, 0.0, -0.1))
    assert plane_watcher.pending == frozenset({StateChange.GEOMETRY})
    plane_watcher.clear()
    # A base-pose setter with the implicit selection (every base link, the watched fixed one included) wakes it
    # too. The pose is env-agnostic: an env-specific pose on a fixed link would require batch_fixed_verts.
    solver.set_base_links_pos(solver.get_links_pos(links_idx=solver._base_links_idx)[0])
    assert plane_watcher.pending == frozenset({StateChange.GEOMETRY})
    plane_watcher.clear()
    # A base-pose setter reaches the named links' whole kinematic subtree (forward kinematics carries the new pose
    # to every descendant): teleporting the fixed tower through its base link wakes a watcher of its fixed child
    # link, where the collision geometry lives.
    child_link = next(link for link in tower.links if link.name == "child")
    child_watcher = Subscriber(to=frozenset({StateChange.GEOMETRY}), links_filter=[child_link.idx])
    solver.subscribe(child_watcher)
    cube.set_pos([[0.0, 0.0, 10.0], [0.0, 0.0, 11.0]], zero_velocity=False)
    assert child_watcher.pending == frozenset()
    tower.set_pos((2.0, 1.0, 0.0))
    assert child_watcher.pending == frozenset({StateChange.GEOMETRY})

    # A whole-state restore may move any link, so the unbounded reach always passes the filter.
    scene.reset()
    assert plane_watcher.pending == frozenset({StateChange.GEOMETRY})


@pytest.mark.parametrize("backend", [None])
def test_set_gravity_accepts_field_and_tensor():
    # The 'gravity: qd.Tensor' annotation must accept both a raw qd.field() (subclass solvers like MPM) and
    # a qd.Tensor wrapper (base_solver / rigid solver).
    os.environ["GS_ENABLE_NDARRAY"] = "0"
    try:
        gs.init(backend=gs.cpu, seed=0)

        scene = gs.Scene(
            show_viewer=False,
            rigid_options=gs.options.RigidOptions(gravity=(0.0, 0.0, -9.81)),
            mpm_options=gs.options.MPMOptions(gravity=(0.0, 0.0, -9.81)),
        )
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(gs.morphs.Box(size=(0.4, 0.4, 0.4), pos=(0.0, 0.0, 0.5)))
        scene.add_entity(gs.morphs.Sphere(pos=(0.0, 0.0, 0.5), radius=0.1), material=gs.materials.MPM.Liquid())
        scene.build()

        new_gravity = [0.0, 0.0, -5.0]

        # Rigid solver: _gravity is a qd.Tensor (from base_solver.build via V())
        rigid = scene.sim.rigid_solver
        assert isinstance(rigid._gravity, qd.Tensor), f"Expected qd.Tensor, got {type(rigid._gravity)}"
        rigid.set_gravity(new_gravity)
        np.testing.assert_allclose(rigid.get_gravity(), new_gravity, atol=1e-6)

        # MPM solver: _gravity is a raw qd.field() (subclass override)
        mpm = scene.sim.mpm_solver
        assert isinstance(mpm._gravity, qd.Field), f"Expected qd.Field, got {type(mpm._gravity)}"
        mpm.set_gravity(new_gravity)
        np.testing.assert_allclose(mpm._gravity.to_numpy().flatten(), new_gravity, atol=1e-6)

    finally:
        gs.destroy()
        os.environ.pop("GS_ENABLE_NDARRAY", None)
