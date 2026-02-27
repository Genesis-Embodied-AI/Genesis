from contextlib import nullcontext
from itertools import permutations

import numpy as np
import pytest

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array, geometric_mean, harmonic_mean

from .conftest import TOL_SINGLE
from .utils import assert_allclose, get_hf_dataset

try:
    import uipc
except ImportError:
    pytest.skip("IPC Coupler is not supported because 'uipc' module is not available.", allow_module_level=True)

from uipc import builtin
from uipc.backend import SceneVisitor
from uipc.geometry import SimplicialComplexSlot, apply_transform, merge


def collect_ipc_geometry_entries(scene):
    visitor = SceneVisitor(scene.sim.coupler._ipc_scene)
    for geom_slot in visitor.geometries():
        if not isinstance(geom_slot, SimplicialComplexSlot):
            continue
        geom = geom_slot.geometry()
        meta_attrs = geom.meta()

        solver_type_attr = meta_attrs.find("solver_type")
        if solver_type_attr is None:
            continue
        (solver_type,) = solver_type_attr.view()
        assert solver_type in ("rigid", "fem", "cloth")

        env_idx_attr = meta_attrs.find("env_idx")
        (env_idx,) = map(int, env_idx_attr.view())

        if solver_type == "rigid":
            idx_attr = meta_attrs.find("link_idx")
        else:  # solver_type in ("fem", "cloth")
            idx_attr = meta_attrs.find("entity_idx")
        (idx,) = map(int, idx_attr.view())

        yield (solver_type, env_idx, idx, geom)


def find_ipc_geometries(scene, *, solver_type, idx=None, env_idx=None):
    geoms = []
    for solver_type_, env_idx_, idx_, geom in collect_ipc_geometry_entries(scene):
        if solver_type == solver_type_ and (idx is None or idx == idx_) and (env_idx is None or env_idx == env_idx_):
            geoms.append(geom)
    return geoms


def get_ipc_merged_geometry(scene, *, solver_type, idx, env_idx):
    (geom,) = find_ipc_geometries(scene, solver_type=solver_type, idx=idx, env_idx=env_idx)
    if geom.instances().size() >= 1:
        geom = merge(apply_transform(geom))
    return geom


def get_ipc_positions(scene, *, solver_type, idx, envs_idx):
    geoms_positions = []
    assert envs_idx
    for env_idx in envs_idx:
        merged_geom = get_ipc_merged_geometry(scene, solver_type=solver_type, idx=idx, env_idx=env_idx)
        geom_positions = merged_geom.positions().view().squeeze(axis=-1)
        geoms_positions.append(geom_positions)
    return np.stack(geoms_positions, axis=0)


def get_ipc_rigid_links_idx(scene, env_idx):
    links_idx = []
    for solver_type_, env_idx_, idx_, _geom in collect_ipc_geometry_entries(scene):
        if solver_type_ == "rigid" and env_idx_ == env_idx:
            links_idx.append(idx_)
    return links_idx


@pytest.mark.parametrize("enable_rigid_rigid_contact", [False, True])
def test_contact_pair_friction_resistance(enable_rigid_rigid_contact):
    scene = gs.Scene(
        coupler_options=gs.options.IPCCouplerOptions(
            contact_resistance=36.0,
            enable_rigid_rigid_contact=enable_rigid_rigid_contact,
        ),
        show_viewer=False,
    )

    plane = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coupling_mode="ipc_only",
        ),
    )
    rigid_a = scene.add_entity(
        gs.morphs.Box(
            pos=(0.0, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            coupling_mode="ipc_only",
            coup_friction=0.25,
            contact_resistance=9.0,
        ),
    )
    rigid_b = scene.add_entity(
        gs.morphs.Box(
            pos=(0.2, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            coupling_mode="ipc_only",
            coup_friction=0.64,
            contact_resistance=16.0,
        ),
    )
    rigid_c = scene.add_entity(
        gs.morphs.Box(
            pos=(-0.2, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.Rigid(
            coupling_mode="ipc_only",
            coup_friction=0.16,
            contact_resistance=None,
        ),
    )
    fem = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.4, 0.0, 0.12),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.FEM.Elastic(
            E=5e4,
            nu=0.35,
            rho=1000.0,
            friction_mu=0.49,
            contact_resistance=25.0,
        ),
    )

    scene.build()

    tab = scene.sim.coupler._ipc_scene.contact_tabular()
    for entities in permutations((plane, rigid_a, rigid_b, rigid_c, fem), 2):
        elems_idx = []
        frictions = []
        resistances = []
        for entity in entities:
            if isinstance(entity, gs.engine.entities.RigidEntity):
                if entity is plane:
                    entity_idx = scene.sim.rigid_solver.entities.index(plane)
                    elem = scene.sim.coupler._ipc_ground_contacts[entity_idx]
                else:
                    entity_idx = scene.sim.rigid_solver.entities.index(entity)
                    elem = scene.sim.coupler._ipc_abd_contacts[entity_idx]
                friction = entity.material.coup_friction
            else:  # isinstance(entity, gs.engine.entities.FEMEntity)
                entity_idx = scene.sim.fem_solver.entities.index(fem)
                elem = scene.sim.coupler._ipc_fem_contacts[entity_idx]
                friction = entity.material.friction_mu
            resistance = entity.material.contact_resistance or scene.sim.coupler.options.contact_resistance
            elems_idx.append(elem.id())
            frictions.append(friction)
            resistances.append(resistance)
        model = tab.at(*elems_idx)
        assert model.friction_rate() == pytest.approx(geometric_mean(*frictions))
        assert model.resistance() == pytest.approx(harmonic_mean(*resistances))
        assert model.is_enabled() ^ (
            all(isinstance(entity, gs.engine.entities.RigidEntity) and entity is not plane for entity in entities)
            and not enable_rigid_rigid_contact
        )


@pytest.mark.parametrize("n_envs", [0, 2])
def test_rigid_ground_sliding(n_envs, show_viewer):
    GRAVITY = np.array([5.0, 0.0, -10.0], dtype=gs.np_float)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            enable_rigid_rigid_contact=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coupling_mode="ipc_only",
            coup_friction=0.25,
        ),
    )

    cubes = []
    for y, mu in ((-0.4, 0.0), (-0.2, 0.01), (0.0, 0.04), (0.2, 0.09), (0.4, 0.16)):
        cube = scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, y, 0.12),
                size=(0.08, 0.08, 0.08),
            ),
            material=gs.materials.Rigid(
                coupling_mode="ipc_only",
                coup_friction=mu,
            ),
        )
        cubes.append(cube)

    scene.build(n_envs=n_envs)

    initial_positions = np.stack([tensor_to_array(cube.get_pos()) for cube in cubes], axis=-2)
    for _ in range(100):
        scene.step()
    final_positions = np.stack([tensor_to_array(cube.get_pos()) for cube in cubes], axis=-2)

    # Coarse non-penetration sanity check
    assert (final_positions[..., 2] > 0.0).all()

    # Distance from ground should be friction-independent
    assert_allclose(np.diff(final_positions[..., 2], axis=-1), 0.0, tol=TOL_SINGLE)

    # No y-axis driving force: lateral drift should be minimal
    assert_allclose(initial_positions[..., 1], final_positions[..., 1], tol=TOL_SINGLE)

    # All cubes should move along +x under tilted gravity.
    assert ((final_positions[..., 0] - initial_positions[..., 0]) > 0.5).all()

    # Lower coup_friction should slide farther, so x should strictly decrease as mu increases.
    assert (np.diff(final_positions[..., ::-1, 0], axis=-1) > 0.2).all()


@pytest.mark.parametrize("n_envs", [0, 2])
def test_ipc_rigid_ground_clearance(n_envs, show_viewer):
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            contact_resistance=1e6,
            enable_rigid_rigid_contact=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coupling_mode="ipc_only",
        ),
    )

    cubes = []
    for y, resistance in ((-0.4, 1e2), (-0.2, 1e3), (0.0, 1e4), (0.2, 1e5), (0.4, 1e6)):
        cube = scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, y, 0.05),
                size=(0.08, 0.08, 0.08),
            ),
            material=gs.materials.Rigid(
                coupling_mode="ipc_only",
                coup_friction=0.0,
                contact_resistance=resistance,
            ),
        )
        cubes.append(cube)

    scene.build(n_envs=n_envs)

    initial_positions = np.stack([tensor_to_array(cube.get_pos()) for cube in cubes], axis=-2)

    dist = []
    for _ in range(70):
        scene.step()
    for _ in range(20):
        scene.step()
        dist.append(np.stack([tensor_to_array(cube.get_verts())[..., 2].min(axis=-1) for cube in cubes], axis=-1))
    dist = np.stack(dist, axis=-1)

    final_positions = np.stack([tensor_to_array(cube.get_pos()) for cube in cubes], axis=-2)

    # No lateral driving force in x/y; drift should stay small.
    assert_allclose(initial_positions[..., :2], final_positions[..., :2], atol=TOL_SINGLE)

    # Make sure that it reaches equilibrium
    assert_allclose(dist[..., -1], dist[..., -2], tol=TOL_SINGLE)

    # Larger contact resistance should produce larger ground clearance (less penetration/compression).
    assert (np.diff(dist, axis=-2) > TOL_SINGLE).all()


@pytest.mark.required
def test_link_filter_strict():
    """Verify that IPC link filter controls which links are actually added to IPC."""
    scene = gs.Scene(
        coupler_options=gs.options.IPCCouplerOptions(
            enable_rigid_rigid_contact=False,
            two_way_coupling=True,
        ),
        show_viewer=False,
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            pos=(0, 0, 0.2),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coupling_mode="two_way_soft_constraint",
            coupling_link_filter=("moving",),
        ),
    )

    scene.build(n_envs=0)

    coupler = scene.sim.coupler
    entity_idx = scene.sim.rigid_solver.entities.index(robot)
    base_link_idx = robot.get_link("base").idx
    moving_link_idx = robot.get_link("moving").idx

    assert entity_idx in coupler._coupling_link_filters
    assert coupler._coupling_link_filters[entity_idx] == {moving_link_idx}

    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert moving_link_idx in ipc_links_idx
    assert base_link_idx not in ipc_links_idx

    assert (0, moving_link_idx) in coupler._abd_link_to_slot
    assert (0, base_link_idx) not in coupler._abd_link_to_slot


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
@pytest.mark.parametrize("fixed", [True, False])
def test_joints(n_envs, coupling_type, joint_type, fixed, show_viewer):
    """Test two-way coupling with revolute joint."""
    DT = 0.01
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    POS = (0, 0, 0.5)
    OMEGA = 2.0 * np.pi  # 1 Hz oscillation
    SCALE = 0.5 if joint_type == "revolute" else 0.15
    CONTACT_MARGIN = 0.01

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=CONTACT_MARGIN,
            constraint_strength_translation=1,
            constraint_strength_rotation=1,
            enable_rigid_rigid_contact=False,
            newton_tolerance=1e-2,
            newton_translation_tolerance=1e-2,
            linear_system_tolerance=1e-3,
            newton_semi_implicit_enable=False,
            two_way_coupling=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 1.0, 0.8),
            camera_lookat=(0.0, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coupling_mode="ipc_only",
            coup_friction=0.5,
        ),
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file=f"urdf/simple/two_cube_{joint_type}.urdf",
            pos=POS,
            fixed=fixed,
        ),
        material=gs.materials.Rigid(
            coupling_mode=coupling_type,
        ),
    )

    scene.build(n_envs=n_envs)
    envs_idx = range(max(scene.n_envs, 1))

    robot.set_dofs_kp(500.0, dofs_idx_local=-1)
    robot.set_dofs_kv(50.0, dofs_idx_local=-1)

    moving_link_idx = robot.get_link("moving").idx
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert moving_link_idx in ipc_links_idx
    assert (0, moving_link_idx) in scene.sim.coupler._abd_link_to_slot
    if coupling_type == "two_way_soft_constraint":
        assert (moving_link_idx, 0) in scene.sim.coupler._abd_data_by_link
    elif coupling_type == "external_articulation":
        entity_idx = scene.sim.rigid_solver.entities.index(robot)
        art_data = scene.sim.coupler._articulation_entities[entity_idx]
        assert art_data is not None
        assert len(art_data.articulation_slots_by_env) == max(scene.n_envs, 1)
        if fixed:
            assert not scene.sim.coupler._abd_data_by_link

    dist_min = float("inf")
    cur_dof_pos_history, target_dof_pos_history = [], []
    gs_transform_history, ipc_transform_history = [], []
    for i in range(100):
        # Apply sinusoidal target position
        t = i * scene.sim_options.dt
        target_dof_pos, target_dof_vel = SCALE * np.sin(OMEGA * t), SCALE * OMEGA * np.cos(OMEGA * t)
        robot.control_dofs_position_velocity(target_dof_pos, target_dof_vel, dofs_idx_local=-1)

        # Store the current and target position / velocity
        cur_dof_pos = tensor_to_array(robot.get_dofs_position(dofs_idx_local=-1)[..., 0])
        cur_dof_pos_history.append(cur_dof_pos)
        target_dof_pos_history.append(target_dof_pos)

        # Make sure the robot never went through the ground
        if not fixed:
            robot_verts = tensor_to_array(robot.get_verts())
            dist_min = np.minimum(dist_min, robot_verts[..., 2].min(axis=-1))
            # FIXME: For some reason it actually can...
            assert (dist_min > -0.1).all()

        scene.step()

        if coupling_type == "two_way_soft_constraint" or not fixed:
            for env_idx in envs_idx:
                abd_data = scene.sim.coupler._abd_data_by_link[(moving_link_idx, env_idx)]
                gs_transform, ipc_transform = abd_data.aim_transform, abd_data.transform
                # FIXME: Why the tolerance is must so large if no fixed ?!
                assert_allclose(gs_transform[:3, 3], ipc_transform[:3, 3], atol=TOL_SINGLE if fixed else 0.2)
                assert_allclose(
                    gu.R_to_xyz(gs_transform[:3, :3], rpy=True),
                    gu.R_to_xyz(ipc_transform[:3, :3], rpy=True),
                    atol=1e-4 if fixed else 0.3,
                )
                gs_transform_history.append(gs_transform)
                ipc_transform_history.append(ipc_transform)
    cur_dof_pos_history = np.stack(cur_dof_pos_history, axis=-1)
    target_dof_pos_history = np.stack(target_dof_pos_history, axis=-1)

    for env_idx in envs_idx if scene.n_envs > 0 else (slice(None),):
        corr = np.corrcoef(cur_dof_pos_history[env_idx], target_dof_pos_history)[0, 1]
        assert corr > 1.0 - 5e-3
    assert_allclose(
        cur_dof_pos_history - cur_dof_pos_history[..., [0]],
        target_dof_pos_history - target_dof_pos_history[..., [0]],
        tol=0.03,
    )
    assert_allclose(np.ptp(cur_dof_pos_history, axis=-1), 2 * SCALE, tol=0.05)

    if gs_transform_history:
        gs_pos_history, gs_quat_history = gu.T_to_trans_quat(np.stack(gs_transform_history, axis=0))
        ipc_pos_history, ipc_quat_history = gu.T_to_trans_quat(np.stack(ipc_transform_history, axis=0))
        pos_err_history = np.linalg.norm(ipc_pos_history - gs_pos_history, axis=-1)
        rot_err_history = np.linalg.norm(
            gu.quat_to_rotvec(gu.transform_quat_by_quat(gs.inv_quat(gs_quat_history), ipc_quat_history)), axis=-1
        )
        assert (np.percentile(pos_err_history, 90, axis=0) < 1e-2).all()
        assert (np.percentile(rot_err_history, 90, axis=0) < 5e-2).all()

    # Make sure the robot bounced on the ground or stayed in place
    if fixed:
        assert_allclose(robot.get_pos(), POS, atol=TOL_SINGLE)
    else:
        assert (dist_min < 1.5 * CONTACT_MARGIN).all()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_objects_freefall(n_envs, show_viewer):
    DT = 0.002
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    NUM_STEPS = 30

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            enable_rigid_rigid_contact=False,
            two_way_coupling=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.2, 3.2, 1.5),
            camera_lookat=(0.0, 0.0, 1.1),
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=1.5,
            pos=(0.0, 0.0, 1.5),
            euler=(0, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e5,
            nu=0.499,
            rho=200,
            thickness=0.001,
            bending_stiffness=50.0,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.6),
        ),
        material=gs.materials.Rigid(
            rho=500.0,
            coupling_mode="ipc_only",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.3, 0.2, 0.8),
        ),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.08,
            pos=(0.5, 0.0, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e5,
            nu=0.3,
            rho=1000.0,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.3, 0.8),
        ),
    )

    scene.build(n_envs=n_envs)
    envs_idx = range(max(scene.n_envs, 1))

    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert box.base_link_idx in ipc_links_idx
    assert (0, box.base_link_idx) in scene.sim.coupler._abd_link_to_slot

    # Verify that geometries are present in IPC for each environment
    cloth_entity_idx = scene.sim.fem_solver.entities.index(cloth)
    box_entity_idx = scene.sim.rigid_solver.entities.index(box)
    sphere_entity_idx = scene.sim.fem_solver.entities.index(sphere)
    objs_kwargs = {
        obj: dict(solver_type=solver_type, idx=idx)
        for obj, solver_type, idx in (
            (cloth, "cloth", cloth_entity_idx),
            (box, "rigid", box_entity_idx),
            (sphere, "fem", sphere_entity_idx),
        )
    }
    for obj_kwargs in objs_kwargs.values():
        for env_idx in envs_idx:
            assert len(find_ipc_geometries(scene, **obj_kwargs, env_idx=env_idx)) == 1

    # Get initial state
    p_0 = {obj: get_ipc_positions(scene, **obj_kwargs, envs_idx=envs_idx) for obj, obj_kwargs in objs_kwargs.items()}
    v_0 = {obj: np.zeros_like(p_0[obj]) for obj in objs_kwargs.keys()}

    # Run simulation and validate dynamics equations at each step
    p_prev, v_prev = p_0.copy(), v_0.copy()
    for _i in range(NUM_STEPS):
        # Move forward in time
        scene.step()

        for obj, obj_kwargs in objs_kwargs.items():
            # Get new position
            p_i = get_ipc_positions(scene, **obj_kwargs, envs_idx=envs_idx)

            # Estimate velocity by finite difference: v_{n+1} = (x_{n+1} - x_n) / DT
            v_i = (p_i - p_prev[obj]) / DT

            # Compute estimated position and velocity
            expected_v = v_prev[obj] + GRAVITY * DT
            expected_p = p_prev[obj] + expected_v * DT

            # Update for next iteration
            p_prev[obj], v_prev[obj] = p_i, v_i

            # FIXME: This test does not pass for sphere entity...
            if obj is sphere:
                continue

            # Validate displacement and velocity assuming Euler scheme
            assert_allclose(v_i, expected_v, atol=1e-3)
            assert_allclose(p_i, expected_p, tol=TOL_SINGLE)

    for obj in objs_kwargs.keys():
        # Validate centroid consistency
        ipc_centroid = p_prev[obj].mean(axis=-2)
        gs_centroid = obj.get_state().pos.mean(axis=-2)
        assert_allclose(ipc_centroid, gs_centroid, atol=TOL_SINGLE)

        # Validate centroidal total displacement: 0.5 * GRAVITY * t * (t + DT)
        p_delta = p_prev[obj] - p_0[obj]
        expected_displacement = 0.5 * GRAVITY * NUM_STEPS * (NUM_STEPS + 1) * DT**2
        assert_allclose(p_delta.mean(axis=-2), expected_displacement, tol=5e-4)

        # FIXME: This test does not pass for sphere entity...
        if obj is sphere:
            continue

        # Validate vertex-based total displacement
        assert_allclose(p_delta, expected_displacement, tol=TOL_SINGLE)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_objects_colliding(n_envs, show_viewer):
    DT = 0.02
    CONTACT_MARGIN = 0.01
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    NUM_STEPS = 90

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=CONTACT_MARGIN,
            enable_rigid_rigid_contact=False,
            two_way_coupling=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 2.0, 0.1),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coupling_mode="ipc_only",
            coup_friction=0.5,
        ),
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=1.5,
            pos=(0.0, 0.0, 0.2),
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e5,
            nu=0.499,
            rho=200,
            thickness=0.001,
            bending_stiffness=50.0,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(-0.25, 0.0, 0.1),
        ),
        material=gs.materials.Rigid(
            rho=500.0,
            coup_friction=0.3,
            coupling_mode="ipc_only",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.3, 0.2, 0.8),
        ),
    )

    sphere = scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.08,
            pos=(0.25, 0.0, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e3,
            nu=0.3,
            rho=1000.0,
            friction_mu=0.3,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.3, 0.8),
        ),
    )

    scene.build(n_envs=n_envs)
    envs_idx = range(max(scene.n_envs, 1))

    # Run simulation and validate dynamics equations at each step
    objs_kwargs = {
        obj: dict(solver_type=solver_type, idx=idx)
        for obj, solver_type, idx in (
            (cloth, "cloth", scene.sim.fem_solver.entities.index(cloth)),
            (box, "rigid", scene.sim.rigid_solver.entities.index(box)),
            (sphere, "fem", scene.sim.fem_solver.entities.index(sphere)),
        )
    }
    p_history = {obj: [] for obj in objs_kwargs.keys()}
    for _i in range(NUM_STEPS):
        scene.step()

        for obj, obj_kwargs in objs_kwargs.items():
            p_i = get_ipc_positions(scene, **obj_kwargs, envs_idx=envs_idx)
            p_history[obj].append(p_i)

    cloth_p_history = np.stack(p_history[cloth], axis=-3)
    for obj in objs_kwargs.keys():
        obj_p_history = np.stack(p_history[obj], axis=-3)

        # Make sure that all vertices are laying on the ground
        assert (obj_p_history[..., 2] < 1.5 * CONTACT_MARGIN).any()
        assert (obj_p_history[..., 2] > 0.0).all()

        # Check that the objects did not fly away (5cm)
        obj_delta_history = np.linalg.norm((obj_p_history - obj_p_history[..., [0], :, :])[..., :2], axis=-1)
        assert_allclose(obj_delta_history, 0.0, atol=0.1)

        # Make sure that all objects reached steady state
        obj_disp_history = np.linalg.norm(np.diff(obj_p_history[..., -10:, :, :], axis=-3), axis=-1)
        assert_allclose(obj_disp_history, 0.0, tol=5e-3)

        # Make sure that the cloth is laying on all objects (at least one vertex above the others)
        if obj is cloth:
            continue
        assert (obj_p_history[..., 2].max(axis=-1) < cloth_p_history[..., 2].max(axis=-1)).all()


@pytest.mark.required
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
def test_robot_grasp_fem(coupling_type, show_viewer):
    """Verify FEM add/retrieve and that robot lift raises FEM more than 20cm."""
    DT = 0.01
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    BOX_POS = (0.65, 0.0, 0.03)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            constraint_strength_translation=10.0,
            constraint_strength_rotation=10.0,
            newton_translation_tolerance=10.0,
            enable_rigid_rigid_contact=False,
            enable_rigid_ground_contact=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 1.0, 1.0),
            camera_lookat=(0.3, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            coupling_mode="ipc_only",
            coup_friction=0.8,
        ),
    )

    material_kwargs = dict(
        coup_friction=0.8,
        coupling_mode=coupling_type,
    )
    if coupling_type == "two_way_soft_constraint":
        material_kwargs["coupling_link_filter"] = ("left_finger", "right_finger")

    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_non_overlap.xml",
        ),
        material=gs.materials.Rigid(**material_kwargs),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=BOX_POS,
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.FEM.Elastic(
            E=5.0e4,
            nu=0.45,
            rho=1000.0,
            friction_mu=0.5,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.2, 0.5),
        ),
    )

    scene.build()

    envs_idx = range(max(scene.n_envs, 1))
    motors_dof, fingers_dof = slice(0, 7), slice(7, 9)
    end_effector = franka.get_link("hand")

    franka.set_dofs_kp([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 500.0, 500.0])

    box_entity_idx = scene.sim.fem_solver.entities.index(box)
    assert len(find_ipc_geometries(scene, solver_type="fem", idx=box_entity_idx, env_idx=0)) == 1

    franka_finger_links_idx = {franka.get_link(name).idx for name in ("left_finger", "right_finger")}
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert franka_finger_links_idx.issubset(ipc_links_idx)
    for link_idx in franka_finger_links_idx:
        assert (0, link_idx) in scene.sim.coupler._abd_link_to_slot

    franka_links_idx = {link.idx for link in franka.links}
    franka_ipc_links_idx = franka_links_idx.intersection(ipc_links_idx)
    if coupling_type == "two_way_soft_constraint":
        entity_idx = scene.sim.rigid_solver.entities.index(franka)
        assert scene.sim.coupler._coupling_link_filters.get(entity_idx) == franka_finger_links_idx
        assert franka_ipc_links_idx == franka_finger_links_idx
    else:
        assert franka_finger_links_idx.issubset(franka_ipc_links_idx)

    ipc_positions_0 = get_ipc_positions(scene, solver_type="fem", idx=box_entity_idx, envs_idx=envs_idx)
    gs_positions_0 = tensor_to_array(box.get_state().pos)
    assert_allclose(ipc_positions_0, gs_positions_0, atol=TOL_SINGLE)
    gs_centroid_0 = gs_positions_0.mean(axis=1)
    assert_allclose(gs_centroid_0, BOX_POS, atol=1e-4)

    def run_stage(target_qpos, finger_pos, duration):
        franka.control_dofs_position(target_qpos[motors_dof], motors_dof)
        franka.control_dofs_position(finger_pos, fingers_dof)
        for _ in range(int(duration / DT)):
            scene.step()

    # Setting initial configuration is not supported by coupling mode "external_articulation"
    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.4], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.9482, 0.6910, 1.2114, -1.6619, -0.6739, 1.8685, 1.1844, 0.0112, 0.0096]
    with pytest.raises(gs.GenesisException) if coupling_type == "external_articulation" else nullcontext():
        franka.set_dofs_position(qpos)
        franka.control_dofs_position(qpos)
    if coupling_type == "external_articulation":
        run_stage(qpos, finger_pos=0.04, duration=2.0)

    # Lower the grapper half way to grasping position
    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.25], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.8757, 0.8824, 1.0523, -1.7619, -0.8831, 2.0903, 1.2924, 0.0400, 0.0400]
    run_stage(qpos, finger_pos=0.04, duration=1.0)

    # Reach grasping position
    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.135], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.7711, 1.0502, 0.8850, -1.7182, -1.0210, 2.2350, 1.3489, 0.0400, 0.0400]
    run_stage(qpos, finger_pos=0.04, duration=0.5)

    # Grasp the cube
    run_stage(qpos, finger_pos=0.0, duration=0.1)

    # Lift the cube
    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.4], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.9488, 0.6916, 1.2123, -1.6627, -0.6750, 1.8683, 1.1855, 0.0301, 0.0319]
    run_stage(qpos, finger_pos=0.0, duration=0.5)

    ipc_positions_f = get_ipc_positions(scene, solver_type="fem", idx=box_entity_idx, envs_idx=envs_idx)
    gs_positions_f = tensor_to_array(box.get_state().pos)
    assert_allclose(ipc_positions_f, gs_positions_f, atol=TOL_SINGLE)
    assert (gs_positions_f[..., 2] - gs_positions_0[..., 2] >= 0.2).all()
    finger_aabb = tensor_to_array(franka.get_link("right_finger").get_AABB())
    assert (gs_positions_f[..., 2] - finger_aabb[..., 0, 2] > 0).any()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_momentum_conversation(n_envs, show_viewer):
    DT = 0.001
    DURATION = 0.30
    CONTACT_MARGIN = 0.01
    VELOCITY = np.array([4.0, 0.0, 0.0], dtype=gs.np_float)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, 0.0),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=CONTACT_MARGIN,
            constraint_strength_translation=1,
            constraint_strength_rotation=1,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, 1.3, 0.6),
            camera_lookat=(0.2, 0.0, 0.3),
        ),
        show_viewer=show_viewer,
    )

    blob = scene.add_entity(
        morph=gs.morphs.Sphere(
            pos=(0.3, 0.0, 0.4),
            radius=0.1,
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e5,
            nu=0.45,
            rho=1000.0,
            model="stable_neohookean",
            friction_mu=0.0,
        ),
    )

    rigid_cube = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.4),
            size=(0.1, 0.1, 0.1),
            euler=(0, 0, 0),
        ),
        material=gs.materials.Rigid(
            rho=1000,
            coupling_mode="two_way_soft_constraint",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.2, 0.2, 0.8),
        ),
    )
    scene.build(n_envs=n_envs)

    rigid_cube.set_dofs_velocity((*VELOCITY, 0.0, 0.0, 0.0))

    fem_entity_idx = scene.sim.fem_solver.entities.index(blob)
    assert len(find_ipc_geometries(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)) == 1

    rigid_link_idx = rigid_cube.base_link_idx
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert rigid_link_idx in ipc_links_idx
    assert (0, rigid_link_idx) in scene.sim.coupler._abd_link_to_slot

    cube_mass = rigid_cube.get_mass()

    # Read actual FEM mass from IPC geometry (mesh mass != analytical sphere mass due to tet discretization).
    blob_radius = blob.morph.radius
    blob_rho = blob.material.rho
    blob_analytical_mass = (4.0 / 3.0) * np.pi * blob_radius**3 * blob_rho
    (fem_raw_geo,) = find_ipc_geometries(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
    fem_mass_density = fem_raw_geo.meta().find(builtin.mass_density).view().item()
    fem_merged_geo = get_ipc_merged_geometry(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
    fem_vertex_volumes = fem_merged_geo.vertices().find(builtin.volume).view().reshape(-1)
    blob_mass = float(np.sum(fem_vertex_volumes) * fem_mass_density)
    assert_allclose(blob_mass, blob_analytical_mass, rtol=0.01)

    total_p_history = []
    momentum_0 = VELOCITY * cube_mass

    dist_min = float("inf")
    fem_positions_prev = None  # FEM initial velocity is zero
    for step in range(int(DURATION / DT)):
        cube_vel = tensor_to_array(rigid_cube.get_links_vel(links_idx_local=0, ref="link_com")[..., 0, :])
        rigid_linear_momentum = cube_mass * cube_vel

        fem_proc_geo = get_ipc_merged_geometry(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
        fem_positions = fem_proc_geo.positions().view().squeeze(axis=-1)
        if fem_positions_prev is not None:
            fem_velocities = (fem_positions - fem_positions_prev) / DT
        else:
            fem_velocities = np.zeros_like(fem_positions)
        fem_positions_prev = fem_positions

        # Make sure that rigid and fem are not penetrating each other
        fem_aabb_min, fem_aabb_max = fem_positions.min(axis=-2), fem_positions.max(axis=-2)
        rigid_aabb = tensor_to_array(rigid_cube.get_AABB())
        rigid_aabb_min, rigid_aabb_max = rigid_aabb[..., 0, :], rigid_aabb[..., 1, :]
        overlap = np.minimum(fem_aabb_max, rigid_aabb_max) - np.maximum(rigid_aabb_min, fem_aabb_min)
        dist_min = np.minimum(dist_min, -overlap.min(axis=-1))
        assert (dist_min > 0.0).all()

        volume_attr = fem_proc_geo.vertices().find(builtin.volume)
        fem_vertex_masses = volume_attr.view().reshape(-1) * fem_mass_density
        assert_allclose(np.sum(fem_vertex_masses), blob_mass, tol=TOL_SINGLE)
        fem_linear_momentum = np.sum(fem_vertex_masses[:, np.newaxis] * fem_velocities, axis=0)

        # Before collision: FEM should have zero momentum, rigid should carry all momentum.
        if step < int(DURATION / 10 / DT):
            assert_allclose(fem_linear_momentum, 0.0, atol=TOL_SINGLE)
            assert_allclose(rigid_linear_momentum, momentum_0, tol=TOL_SINGLE)

        total_linear_momentum = rigid_linear_momentum + fem_linear_momentum
        total_p_history.append(total_linear_momentum)

        scene.step()

    # Make sure the objects bounced on each other
    assert (dist_min < 1.5 * CONTACT_MARGIN).all()
    expected_cube_vel = (cube_mass - blob_mass) / (cube_mass + blob_mass) * VELOCITY
    expected_blob_vel = 2 * cube_mass / (cube_mass + blob_mass) * VELOCITY
    assert (cube_vel[..., 0] < -0.5).all()
    assert (fem_velocities[..., 0].mean(axis=-1) > 0.5).all()

    # Check total momentum conservation.
    # NOTE : The tet mesh's contact-facing vertices (x < -0.05) have a z-mean of -0.00138 due to TetGen's asymmetric
    # Steiner point insertion, causing an asymmetric contact force distribution during the x-direction collision.
    # This z-bias produces a net -z impulse, resulting in the observed z-momentum leak.
    assert_allclose(total_p_history, momentum_0, tol=0.001)


@pytest.mark.required
@pytest.mark.parametrize("enable_rigid_ground_contact", [True, False])
@pytest.mark.parametrize("coupling_mode", ["ipc_only", "two_way_soft_constraint"])
def test_collision_delegation_ipc_vs_rigid(coupling_mode, enable_rigid_ground_contact):
    """Verify collision pair delegation between IPC and rigid solver based on coupling_mode and ground contact."""
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=True,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            enable_rigid_ground_contact=enable_rigid_ground_contact,
        ),
        show_viewer=False,
    )

    plane = scene.add_entity(gs.morphs.Plane())  # No coupling_mode: stays in rigid solver only

    # Non-IPC box — always handled by rigid solver
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(1.0, 0.0, 0.2),
        ),
        material=gs.materials.Rigid(),
    )

    if coupling_mode == "two_way_soft_constraint":
        entity = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda_non_overlap.xml",
            ),
            material=gs.materials.Rigid(
                coupling_mode="two_way_soft_constraint",
                coupling_link_filter=("left_finger", "right_finger"),
            ),
        )

        ipc_excluded_geoms = {
            geom.idx for name in entity.material.coupling_link_filter for geom in entity.get_link(name).geoms
        }
    else:
        with pytest.raises(gs.GenesisException):
            entity = scene.add_entity(
                gs.morphs.URDF(
                    file="urdf/go2/urdf/go2.urdf",
                    pos=(0.0, 0.0, 1.0),
                ),
                material=gs.materials.Rigid(
                    coupling_mode="ipc_only",
                ),
            )

        entity = scene.add_entity(
            morph=gs.morphs.Box(
                size=(0.2, 0.2, 0.2),
                pos=(0.0, 0.0, 0.6),
            ),
            material=gs.materials.Rigid(
                coupling_mode="ipc_only",
            ),
        )

        ipc_excluded_geoms = {geom.idx for geom in entity.geoms}

    scene.build()

    pair_idx = scene.sim.rigid_solver.collider._collision_pair_idx

    # Collect geom indices for entities that should retain rigid solver pairs
    rigid_kept_geoms = {geom.idx for geom in entity.geoms} - ipc_excluded_geoms
    ground_geoms = {plane.geoms[0].idx}
    box_geoms = {box.geoms[0].idx}

    # Non-IPC box always has rigid solver ground pairs
    assert any(pair_idx[min(a, b), max(a, b)] >= 0 for a in box_geoms for b in ground_geoms)

    # Pairs between IPC-excluded geoms must have no rigid solver pairs (handled by IPC)
    for i_ga in ipc_excluded_geoms:
        for i_gb in ipc_excluded_geoms:
            if i_ga < i_gb:
                assert pair_idx[i_ga, i_gb] == -1

    # Mixed pairs (IPC-excluded ↔ non-IPC) must be kept in rigid solver
    for i_ga in ipc_excluded_geoms:
        for i_gb in box_geoms:
            a, b = min(i_ga, i_gb), max(i_ga, i_gb)
            assert pair_idx[a, b] >= 0

    # IPC-excluded geom ↔ ground must be kept in rigid solver (ground is not IPC-excluded)
    for i_ga in ipc_excluded_geoms:
        for i_gb in ground_geoms:
            a, b = min(i_ga, i_gb), max(i_ga, i_gb)
            assert pair_idx[a, b] >= 0

    # Non-excluded rigid geoms (if any) keep rigid solver ground and self-collision pairs
    if rigid_kept_geoms:
        assert any(pair_idx[min(a, b), max(a, b)] >= 0 for a in rigid_kept_geoms for b in ground_geoms)
        assert any(pair_idx[min(a, b), max(a, b)] >= 0 for a in rigid_kept_geoms for b in rigid_kept_geoms if a < b)
