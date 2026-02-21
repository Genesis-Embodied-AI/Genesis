import numpy as np
import pytest

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

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


def get_entity_base_z(entity):
    pos = tensor_to_array(entity.get_pos())
    if pos.ndim == 1:
        return float(pos[2])
    return float(pos.reshape(-1, 3)[0, 2])


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_cloth_free_falling(n_envs, show_viewer):
    """Test IPC cloth simulation with gravity physics validation."""
    DT = 0.002
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=DT,
            gravity=GRAVITY.tolist(),
            contact_d_hat=0.01,
            contact_friction_mu=0.3,
            two_way_coupling=True,
            IPC_self_contact=False,
            disable_genesis_contact=True,
            enable_ipc_gui=False,
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=2.0,
            pos=(0.0, 0.0, 1.5),
            euler=(0, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e6,
            nu=0.499,
            rho=200,
            thickness=0.001,
            bending_stiffness=50.0,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
            double_sided=True,
        ),
    )

    box = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0, 0, 0.3),
        ),
        material=gs.materials.Rigid(
            rho=500,
            friction=0.3,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.8, 0.3, 0.2, 0.8),
        ),
    )
    scene.sim.coupler.set_entity_coupling_type(
        entity=box,
        coupling_type="two_way_soft_constraint",
    )
    scene.add_entity(
        morph=gs.morphs.Sphere(
            radius=0.08,
            pos=(0.5, 0.0, 0.1),
        ),
        material=gs.materials.FEM.Elastic(
            E=1.0e3,
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

    # Verify cloth geometry is present in IPC for each environment
    cloth_entity_idx = scene.sim.fem_solver.entities.index(cloth)
    for env_idx in envs_idx:
        assert len(find_ipc_geometries(scene, solver_type="cloth", idx=cloth_entity_idx, env_idx=env_idx)) == 1

    # Get initial state
    p_prev = get_ipc_positions(scene, solver_type="cloth", idx=cloth_entity_idx, envs_idx=envs_idx)
    v_prev = np.zeros_like(p_prev)

    # Run simulation and validate dynamics equations at each step
    for _i in range(20):
        # Move forward in time
        scene.step()

        # Get new position
        p_i = get_ipc_positions(scene, solver_type="cloth", idx=cloth_entity_idx, envs_idx=envs_idx)

        # Estimate velocity by finite difference: v_{n+1} = (x_{n+1} - x_n) / DT
        v_i = (p_i - p_prev) / DT

        # Validate velocity
        expected_v_next = v_prev + GRAVITY * DT
        assert_allclose(v_i, expected_v_next, atol=2e-4)

        # Validate displacement assuming Euler scheme
        expected_p = p_prev + v_prev * DT + GRAVITY * DT**2
        assert_allclose(p_i, expected_p, tol=TOL_SINGLE)

        # Update for next iteration
        p_prev, v_prev = p_i, v_i


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0])  # FIXME: batching is not supported for now
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
@pytest.mark.parametrize("fixed", [True, False])
def test_joint_revolute(n_envs, coupling_type, fixed, show_viewer):
    """Test two-way coupling with revolute joint."""
    DT = 0.01
    GRAVITY = np.array([0.0, 0.0, -9.8], dtype=gs.np_float)
    POS = (0, 0, 0.5)
    OMEGA = 2.0 * np.pi  # 1 Hz oscillation

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=GRAVITY,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=DT,
            gravity=GRAVITY.tolist(),
            contact_friction_mu=0.5,
            ipc_constraint_strength=(1, 1),
            newton_velocity_tol=1e-2,
            newton_transrate_tol=1e-2,
            linear_system_tol_rate=1e-3,
            newton_semi_implicit_enable=False,
            two_way_coupling=True,
            disable_genesis_contact=True,
            IPC_self_contact=False,
            sync_dof_enable=False,
            disable_ipc_logging=False,
            enable_ipc_gui=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            pos=POS,
            fixed=fixed,
        ),
    )
    scene.sim.coupler.set_entity_coupling_type(
        entity=robot,
        coupling_type=coupling_type,
    )

    scene.build(n_envs=n_envs)
    envs_idx = range(max(scene.n_envs, 1))

    robot.set_dofs_kp(500.0, dofs_idx_local=-1)
    robot.set_dofs_kv(50.0, dofs_idx_local=-1)

    moving_link_idx = robot.get_link("moving").idx
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert moving_link_idx in ipc_links_idx
    assert (0, moving_link_idx) in scene.sim.coupler._link_to_abd_slot
    if coupling_type == "two_way_soft_constraint":
        assert moving_link_idx in scene.sim.coupler.abd_data_by_link
        assert set(envs_idx) == set(scene.sim.coupler.abd_data_by_link[moving_link_idx])
    elif fixed:
        assert not scene.sim.coupler.abd_data_by_link

    cur_dof_pos_history, target_dof_pos_history = [], []
    for i in range(100):
        # Apply sinusoidal target position to revolute joint
        t = i * scene.sim_options.dt
        target_dof_pos, target_dof_vel = 0.5 * np.sin(OMEGA * t), 0.5 * OMEGA * np.cos(OMEGA * t)
        robot.control_dofs_position_velocity(target_dof_pos, target_dof_vel, dofs_idx_local=-1)

        # Store the current and target position / velocity
        cur_dof_pos = tensor_to_array(robot.get_dofs_position(dofs_idx_local=-1)[..., 0])
        cur_dof_pos_history.append(cur_dof_pos)
        target_dof_pos_history.append(target_dof_pos)

        scene.step()

        if coupling_type == "two_way_soft_constraint" or not fixed:
            for env_idx in envs_idx:
                abd_data = scene.sim.coupler.abd_data_by_link[moving_link_idx][env_idx]
                gs_transform, ipc_transform = abd_data["aim_transform"], abd_data["transform"]
                # FIXME: Why the tolerance is must so large if no fixed ?!
                assert_allclose(gs_transform[:3, 3], ipc_transform[:3, 3], atol=TOL_SINGLE if fixed else 0.2)
                assert_allclose(
                    gu.R_to_xyz(gs_transform[:3, :3], rpy=True),
                    gu.R_to_xyz(ipc_transform[:3, :3], rpy=True),
                    atol=1e-4 if fixed else 0.3,
                )
    cur_dof_pos_history = np.stack(cur_dof_pos_history, axis=-1)
    target_dof_pos_history = np.stack(target_dof_pos_history, axis=-1)

    assert np.corrcoef(cur_dof_pos_history, target_dof_pos_history)[0, 1] > 1.0 - 5e-3
    # FIXME: Why is it necessary to skip many steps if not fixed ?!
    start_idx = 0 if fixed else 50
    assert_allclose(
        cur_dof_pos_history[start_idx:] - cur_dof_pos_history[start_idx],
        target_dof_pos_history[start_idx:] - target_dof_pos_history[start_idx],
        tol=0.03,
    )
    assert_allclose(np.ptp(cur_dof_pos_history), 1.0, tol=0.05)

    final_base_pos = robot.get_pos()
    if fixed:
        assert_allclose(final_base_pos, POS, atol=TOL_SINGLE)
    else:
        assert POS[2] - final_base_pos[..., 2] > 0.2


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0])
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
@pytest.mark.parametrize("fixed_base", [True, False])
def test_ipc_two_way_prismatic(n_envs, coupling_type, fixed_base, show_viewer):
    """Test two-way coupling with prismatic joint."""
    DT = 1e-2
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.8),
            contact_friction_mu=0.5,
            ipc_constraint_strength=(1, 1),
            IPC_self_contact=False,
            two_way_coupling=True,
            disable_genesis_contact=True,
            disable_ipc_logging=True,
            newton_velocity_tol=1e-2,
            newton_transrate_tol=1e-2,
            linear_system_tol_rate=1e-3,
            sync_dof_enable=False,
            newton_semi_implicit_enable=False,
            enable_ipc_gui=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_prismatic.urdf",
            pos=(0, 0, 0.5),
            fixed=fixed_base,
        ),
    )
    scene.sim.coupler.set_entity_coupling_type(
        entity=robot,
        coupling_type=coupling_type,
    )
    scene.build(n_envs=n_envs)

    moving_link_idx = robot.get_link("moving").idx
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert moving_link_idx in ipc_links_idx
    assert (0, moving_link_idx) in scene.sim.coupler._link_to_abd_slot
    initial_base_z = get_entity_base_z(robot)
    initial_base_pos = tensor_to_array(robot.get_pos()).reshape(-1, 3)[0].copy()

    max_steps = 100
    omega = 2.0 * np.pi
    settle_steps = 10
    qpos_history = []
    target_history = []

    zero_target = np.zeros(robot.n_dofs, dtype=np.float32)
    for _ in range(settle_steps):
        robot.control_dofs_position(zero_target)
        scene.step()

    for i in range(max_steps):
        t = i * scene.sim_options.dt
        target_qpos = 0.15 + 0.1 * np.sin(omega * t)
        robot.control_dofs_position([target_qpos], [robot.n_dofs - 1])
        scene.step()
        current_qpos = float(tensor_to_array(robot.get_qpos()).reshape(-1)[-1])
        qpos_history.append(current_qpos)
        target_history.append(float(target_qpos))

        if i > 50:
            link_idx = moving_link_idx
            env_idx = 0
            if (
                link_idx in scene.sim.coupler.abd_data_by_link
                and env_idx in scene.sim.coupler.abd_data_by_link[link_idx]
            ):
                abd_data = scene.sim.coupler.abd_data_by_link[link_idx][env_idx]
                gs_transform = abd_data["aim_transform"]
                ipc_transform = abd_data["transform"]
                if gs_transform is not None and ipc_transform is not None:
                    if coupling_type == "external_articulation":
                        breakpoint()
                        assert_allclose(gs_transform[:3, 3], ipc_transform[:3, 3], atol=5e-3)
                        assert_allclose(
                            gu.R_to_xyz(gs_transform[:3, :3], rpy=True),
                            gu.R_to_xyz(ipc_transform[:3, :3], rpy=True),
                            atol=0.1,
                        )

    if coupling_type == "two_way_soft_constraint":
        corr = np.corrcoef(qpos_history, target_history)[0, 1]
        assert corr > 0.4
        assert np.ptp(qpos_history) > 0.12

    if not fixed_base:
        if coupling_type == "external_articulation":
            final_base_z = get_entity_base_z(robot)
            assert final_base_z <= initial_base_z - 0.03
        else:
            final_base_pos = tensor_to_array(robot.get_pos()).reshape(-1, 3)[0]
            with pytest.raises(AssertionError):
                assert_allclose(final_base_pos, initial_base_pos, atol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0])
def test_ipc_cloth_gravity_freefall(n_envs, show_viewer):
    """Test cloth free fall physics validation + IPC<->Genesis retrieve consistency."""
    DT = 2e-3
    GRAVITY = 9.8
    z0 = 2.0
    num_steps = 50

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
            contact_d_hat=0.01,
            contact_friction_mu=0.3,
            IPC_self_contact=False,
            two_way_coupling=True,
            disable_genesis_contact=True,
            enable_ipc_gui=False,
        ),
        show_viewer=show_viewer,
    )

    asset_path = get_hf_dataset(pattern="IPC/grid20x20.obj")
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=f"{asset_path}/IPC/grid20x20.obj",
            scale=2.0,
            pos=(0.0, 0.0, z0),
            euler=(0, 0, 0),
        ),
        material=gs.materials.FEM.Cloth(
            E=1e6,
            nu=0.499,
            rho=200,
            thickness=0.001,
            bending_stiffness=50.0,
        ),
        surface=gs.surfaces.Plastic(
            color=(0.3, 0.5, 0.8, 1.0),
            double_sided=True,
        ),
    )

    scene.build(n_envs=n_envs)

    cloth_entity_idx = scene.sim.fem_solver.entities.index(cloth)
    assert len(find_ipc_geometries(scene, solver_type="cloth", idx=cloth_entity_idx, env_idx=0)) == 1

    initial_positions = get_ipc_positions(scene, solver_type="cloth", idx=cloth_entity_idx, env_idx=0)
    assert initial_positions is not None
    z_initial = initial_positions[0, 2]

    for _ in range(num_steps):
        scene.step()

    final_positions = get_ipc_positions(scene, solver_type="cloth", idx=cloth_entity_idx, env_idx=0)
    assert final_positions is not None
    z_final = final_positions[0, 2]

    # Validate displacement: 0.5 * GRAVITY * t * (t + DT)
    t_total = num_steps * DT
    actual_displacement = z_initial - z_final
    expected_displacement = 0.5 * GRAVITY * t_total * (t_total + DT)
    assert_allclose(actual_displacement, expected_displacement, rtol=0.01)

    # Validate IPC<->Genesis centroid consistency
    ipc_centroid = final_positions.mean(axis=0)
    gs_positions = tensor_to_array(cloth.get_state().pos)[0]
    gs_centroid = gs_positions.mean(axis=0)
    assert_allclose(ipc_centroid, gs_centroid, atol=1e-3)


@pytest.mark.required
def test_ipc_link_filter_strict(show_viewer):
    """Verify that IPC link filter controls which links are actually added to IPC."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1e-3),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=0.01,
            contact_friction_mu=0.3,
            IPC_self_contact=False,
            two_way_coupling=True,
            disable_genesis_contact=True,
            enable_ipc_gui=False,
        ),
        show_viewer=show_viewer,
    )

    robot = scene.add_entity(
        morph=gs.morphs.URDF(
            file="urdf/simple/two_cube_revolute.urdf",
            pos=(0, 0, 0.2),
            fixed=True,
        ),
    )

    scene.sim.coupler.set_entity_coupling_type(entity=robot, coupling_type="two_way_soft_constraint")
    scene.sim.coupler.set_ipc_coupling_link_filter(entity=robot, link_names=["moving"])
    scene.build(n_envs=0)

    coupler = scene.sim.coupler
    entity_idx = robot._idx_in_solver
    base_link_idx = robot.get_link("base").idx
    moving_link_idx = robot.get_link("moving").idx

    assert entity_idx in coupler._ipc_link_filters
    assert coupler._ipc_link_filters[entity_idx] == {moving_link_idx}

    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert moving_link_idx in ipc_links_idx
    assert base_link_idx not in ipc_links_idx

    assert (0, moving_link_idx) in coupler._link_to_abd_slot
    assert (0, base_link_idx) not in coupler._link_to_abd_slot


@pytest.mark.required
def test_ipc_ipc_only_cube_freefall_height_drop(show_viewer):
    """Verify ipc_only rigid cube is in IPC and falls down under gravity."""
    DT = 1e-3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.8),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.8),
            contact_d_hat=0.01,
            contact_friction_mu=0.3,
            IPC_self_contact=False,
            two_way_coupling=True,
            disable_genesis_contact=True,
            enable_ipc_gui=False,
        ),
        show_viewer=show_viewer,
    )

    cube = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.05, 0.05, 0.05),
            pos=(0.0, 0.0, 0.4),
        ),
        material=gs.materials.Rigid(
            rho=500.0,
            friction=0.3,
        ),
    )
    scene.sim.coupler.set_entity_coupling_type(entity=cube, coupling_type="ipc_only")
    scene.build(n_envs=0)

    base_link_idx = cube.base_link_idx
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert base_link_idx in ipc_links_idx
    assert (0, base_link_idx) in scene.sim.coupler._link_to_abd_slot

    initial_z = get_entity_base_z(cube)
    for _ in range(120):
        scene.step()
    final_z = get_entity_base_z(cube)
    assert final_z <= initial_z - 0.04


@pytest.mark.required
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
def test_ipc_robot_fem_grasp_retrieve_lift_strict(coupling_type, show_viewer):
    """Verify FEM add/retrieve and that robot lift raises FEM by >= 0.1m."""
    DT = 1e-2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.8),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=DT,
            gravity=(0.0, 0.0, -9.8),
            ipc_constraint_strength=(100, 100),
            contact_friction_mu=0.8,
            newton_transrate_tol=10,
            IPC_self_contact=False,
            disable_ipc_ground_contact=True,
            disable_ipc_logging=False,
            enable_ipc_gui=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_non_overlap.xml",
        ),
    )
    scene.sim.coupler.set_entity_coupling_type(
        entity=franka,
        coupling_type=coupling_type,
    )
    scene.sim.coupler.set_ipc_coupling_link_filter(
        entity=franka,
        link_names=("left_finger", "right_finger"),
    )

    fem_box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.65, 0.0, 0.03),
            size=(0.05, 0.05, 0.05),
        ),
        material=gs.materials.FEM.Elastic(
            E=5.0e4,
            nu=0.45,
            rho=1000.0,
            model="stable_neohookean",
        ),
        surface=gs.surfaces.Plastic(
            color=(0.2, 0.8, 0.2, 0.5),
        ),
    )

    scene.build()

    fem_entity_idx = scene.sim.fem_solver.entities.index(fem_box)
    assert len(find_ipc_geometries(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)) == 1

    franka_finger_links_idx = {franka.get_link(name).idx for name in ("left_finger", "right_finger")}
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert franka_finger_links_idx.issubset(ipc_links_idx)
    for link_idx in franka_finger_links_idx:
        assert (0, link_idx) in scene.sim.coupler._link_to_abd_slot

    franka_links_idx = {link.idx for link in franka.links}
    franka_ipc_links_idx = franka_links_idx.intersection(ipc_links_idx)
    if coupling_type == "two_way_soft_constraint":
        entity_idx = scene.sim.rigid_solver.entities.index(franka)
        assert scene.sim.coupler._ipc_link_filters.get(entity_idx) == franka_finger_links_idx
        assert franka_ipc_links_idx == franka_finger_links_idx
    else:
        assert franka_finger_links_idx.issubset(franka_ipc_links_idx)

    initial_ipc_centroid = get_ipc_positions(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0).mean(axis=0)
    initial_gs_positions = tensor_to_array(fem_box.get_state().pos)[0]
    initial_gs_centroid = initial_gs_positions.mean(axis=0)
    assert_allclose(initial_ipc_centroid, initial_gs_centroid, atol=1e-3)
    initial_z = initial_gs_centroid[2]

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    current_kp = franka.get_dofs_kp()
    new_kp = current_kp
    new_kp[fingers_dof] = current_kp[fingers_dof] * 5.0
    franka.set_dofs_kp(new_kp)
    # end_effector = franka.get_link("hand")

    def run_stage(target_qpos, finger_pos, duration):
        finger_cmd = np.array([finger_pos, finger_pos], dtype=np.float32)
        for _ in range(max(int(duration / DT), 1)):
            franka.control_dofs_position(target_qpos[:-2], motors_dof)
            franka.control_dofs_position(finger_cmd, fingers_dof)
            scene.step()

    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.4], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.9482, 0.6910, 1.2114, -1.6619, -0.6739, 1.8685, 1.1844, 0.0112, 0.0096]
    run_stage(qpos, finger_pos=0.04, duration=2.0)

    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.25], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.8757, 0.8824, 1.0523, -1.7619, -0.8831, 2.0903, 1.2924, 0.0400, 0.0400]
    run_stage(qpos, finger_pos=0.04, duration=1.0)

    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.135], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.7711, 1.0502, 0.8850, -1.7182, -1.0210, 2.2350, 1.3489, 0.0400, 0.0400]
    run_stage(qpos, finger_pos=0.04, duration=0.5)

    run_stage(qpos, finger_pos=0.0, duration=0.1)

    # qpos = franka.inverse_kinematics(link=end_effector, pos=[0.65, 0.0, 0.4], quat=[0.0, 1.0, 0.0, 0.0])
    qpos = [-0.9488, 0.6916, 1.2123, -1.6627, -0.6750, 1.8683, 1.1855, 0.0301, 0.0319]
    run_stage(qpos, finger_pos=0.0, duration=0.2)

    final_ipc_centroid = get_ipc_positions(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0).mean(axis=0)
    final_gs_positions = tensor_to_array(fem_box.get_state().pos)[0]
    final_gs_centroid = final_gs_positions.mean(axis=0)

    assert_allclose(final_ipc_centroid, final_gs_centroid, atol=1e-3)

    z_gain = final_gs_centroid[2] - initial_z
    assert z_gain >= 0.1


@pytest.mark.required
def test_ipc_motion_final_relative_error_below_2pct(show_viewer):
    """IPC momentum test: final relative error must be <= 2%."""
    DT = 1e-3
    initial_velocity = 4.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=DT,
            gravity=(0.0, 0.0, 0.0),
            ipc_constraint_strength=(1, 1),
            enable_ipc_gui=False,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())

    blob = scene.add_entity(
        morph=gs.morphs.Sphere(pos=(0.3, 0.0, 0.4), radius=0.1),
        material=gs.materials.FEM.Elastic(E=1.0e5, nu=0.45, rho=1000.0, model="stable_neohookean"),
    )
    rigid_cube = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.4), size=(0.1, 0.1, 0.1), euler=(0, 0, 0)),
        material=gs.materials.Rigid(rho=1000, friction=0.3),
        surface=gs.surfaces.Plastic(color=(0.8, 0.2, 0.2, 0.8)),
    )
    scene.sim.coupler.set_entity_coupling_type(entity=rigid_cube, coupling_type="two_way_soft_constraint")
    scene.build(n_envs=1)

    fem_entity_idx = scene.sim.fem_solver.entities.index(blob)
    assert len(find_ipc_geometries(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)) == 1

    rigid_link_idx = rigid_cube.base_link_idx
    ipc_links_idx = get_ipc_rigid_links_idx(scene, env_idx=0)
    assert rigid_link_idx in ipc_links_idx
    assert (0, rigid_link_idx) in scene.sim.coupler._link_to_abd_slot

    rigid_cube.set_dofs_velocity((initial_velocity, 0.0, 0.0, 0.0, 0.0, 0.0))

    blob_radius = blob.morph.radius
    blob_rho = blob.material.rho
    fem_total_mass = (4.0 / 3.0) * np.pi * (blob_radius**3) * blob_rho

    rigid_mass = float(scene.sim.rigid_solver.get_links_inertial_mass(links_idx=rigid_link_idx).item())

    fem_prev_pos = None
    rigid_prev_pos = None
    total_p_history = []

    test_time = 0.30
    n_steps = int(test_time / DT)

    for _ in range(n_steps):
        rigid_pos = tensor_to_array(
            scene.sim.rigid_solver.get_links_pos(links_idx=rigid_link_idx, ref="link_com")
        ).reshape(-1, 3)[0]
        if rigid_prev_pos is None:
            rigid_vel = np.array([initial_velocity, 0.0, 0.0])
        else:
            rigid_vel = (rigid_pos - rigid_prev_pos) / DT
        rigid_prev_pos = rigid_pos.copy()
        rigid_linear_momentum = rigid_mass * rigid_vel

        fem_proc_geo = get_ipc_merged_geometry(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
        assert fem_proc_geo is not None
        fem_vertex_positions = fem_proc_geo.positions().view().reshape(-1, 3)

        volume_attr = fem_proc_geo.vertices().find(builtin.volume)
        mass_density_attr = fem_proc_geo.vertices().find(builtin.mass_density)
        if volume_attr and mass_density_attr:
            volumes = volume_attr.view().reshape(-1)
            mass_densities = mass_density_attr.view().reshape(-1)
            fem_vertex_masses = volumes * mass_densities
        else:
            n_vertices = len(fem_vertex_positions)
            fem_vertex_masses = np.ones(n_vertices) * (fem_total_mass / n_vertices)

        if fem_prev_pos is None:
            fem_vertex_velocities = np.zeros_like(fem_vertex_positions)
        else:
            fem_vertex_velocities = (fem_vertex_positions - fem_prev_pos) / DT
        fem_prev_pos = fem_vertex_positions.copy()

        fem_linear_momentum = np.sum(fem_vertex_masses[:, np.newaxis] * fem_vertex_velocities, axis=0)
        total_linear_momentum = rigid_linear_momentum + fem_linear_momentum
        total_p_history.append(total_linear_momentum)

        scene.step()

    total_p_history = np.array(total_p_history)
    expected_momentum = np.array([initial_velocity * rigid_mass, 0.0, 0.0])
    assert_allclose(total_p_history[-1], expected_momentum, rtol=0.01, atol=0.001)
