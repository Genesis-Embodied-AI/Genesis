import numpy as np
import pytest

import genesis as gs

from .utils import get_hf_dataset

try:
    from uipc import builtin
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot, apply_transform, merge
except ImportError:
    pytest.skip("IPC Coupler is not supported because 'uipc' module is not available.", allow_module_level=True)


def _iter_ipc_simplicial_geometries(scene):
    visitor = SceneVisitor(scene.sim.coupler._ipc_scene)
    for geo_slot in visitor.geometries():
        if isinstance(geo_slot, SimplicialComplexSlot):
            yield geo_slot.geometry()


def _get_processed_geo(geo):
    proc_geo = geo
    if geo.instances().size() >= 1:
        proc_geo = merge(apply_transform(geo))
    return proc_geo


def _collect_ipc_geometry_entries(scene):
    entries = []
    for geo in _iter_ipc_simplicial_geometries(scene):
        meta = _read_ipc_geometry_metadata(geo)
        if meta is None:
            continue
        solver_type, env_idx, idx = meta
        entries.append(
            {
                "solver_type": solver_type,
                "env_idx": env_idx,
                "idx": idx,
                "geo": geo,
            }
        )
    return entries


def _read_ipc_geometry_metadata(geo):
    """Read (solver_type, env_idx, idx) from IPC geometry metadata."""
    try:
        meta_attrs = geo.meta()

        solver_type_attr = meta_attrs.find("solver_type")
        if not solver_type_attr or solver_type_attr.name() != "solver_type":
            return None
        solver_type_view = solver_type_attr.view()
        if len(solver_type_view) == 0:
            return None
        solver_type = str(solver_type_view[0])

        env_idx_attr = meta_attrs.find("env_idx")
        if not env_idx_attr:
            return None
        env_idx = int(str(env_idx_attr.view()[0]))

        if solver_type == "rigid":
            idx_attr = meta_attrs.find("link_idx")
        elif solver_type in ("fem", "cloth"):
            idx_attr = meta_attrs.find("entity_idx")
        else:
            return None

        if not idx_attr:
            return None
        idx = int(str(idx_attr.view()[0]))
        return (solver_type, env_idx, idx)
    except Exception:
        return None


def _find_ipc_geometries(scene, *, solver_type, idx, env_idx=None):
    matches = []
    for entry in _collect_ipc_geometry_entries(scene):
        if entry["solver_type"] != solver_type:
            continue
        if entry["idx"] != idx:
            continue
        if env_idx is not None and entry["env_idx"] != env_idx:
            continue
        matches.append(entry["geo"])
    return matches


def _get_ipc_processed_geometry(scene, *, solver_type, idx, env_idx):
    matches = _find_ipc_geometries(scene, solver_type=solver_type, idx=idx, env_idx=env_idx)
    if not matches:
        return None
    return _get_processed_geo(matches[0])


def _get_ipc_positions(scene, *, solver_type, idx, env_idx):
    proc_geo = _get_ipc_processed_geometry(scene, solver_type=solver_type, idx=idx, env_idx=env_idx)
    if proc_geo is None:
        return None
    return proc_geo.positions().view().reshape(-1, 3)


def _get_ipc_centroid(scene, *, solver_type, idx, env_idx):
    positions = _get_ipc_positions(scene, solver_type=solver_type, idx=idx, env_idx=env_idx)
    if positions is None or len(positions) == 0:
        return None
    return positions.mean(axis=0)


def _get_ipc_rigid_links(scene, env_idx=0):
    links = set()
    for entry in _collect_ipc_geometry_entries(scene):
        if entry["solver_type"] == "rigid" and entry["env_idx"] == env_idx:
            links.add(entry["idx"])
    return links


def _get_fem_solver_entity_idx(scene, fem_entity):
    for idx, entity in enumerate(scene.sim.fem_solver._entities):
        if entity is fem_entity:
            return idx
    raise AssertionError(f"FEM entity {fem_entity} not found in FEM solver entity list.")


def _get_entity_base_z(entity):
    pos = np.asarray(entity.get_pos().detach().cpu().numpy())
    if pos.ndim == 1:
        return float(pos[2])
    return float(pos.reshape(-1, 3)[0, 2])


def _rotmat_to_euler(rot_mat):
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        y = np.arctan2(-rot_mat[2, 0], sy)
        z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        y = np.arctan2(-rot_mat[2, 0], sy)
        z = 0.0
    return np.array([x, y, z])


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_ipc_cloth(n_envs, show_viewer):
    """Test IPC cloth simulation with gravity physics validation."""
    dt = 2e-3
    g = 9.8

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -g),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -g),
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

    cloth_entity_idx = _get_fem_solver_entity_idx(scene, cloth)
    for env_idx in range(scene.sim._B):
        cloth_matches = _find_ipc_geometries(scene, solver_type="cloth", idx=cloth_entity_idx, env_idx=env_idx)
        assert len(cloth_matches) == 1, (
            f"Expected exactly one IPC cloth geometry for entity={cloth_entity_idx}, env={env_idx}, "
            f"found {len(cloth_matches)}."
        )

    x_n = _get_ipc_positions(scene, solver_type="cloth", idx=cloth_entity_idx, env_idx=0)
    assert x_n is not None, "Could not retrieve cloth vertex positions from IPC."
    x_n = x_n[0, 2]
    v_n = 0.0

    num_validation_steps = 10
    tolerance = 0.01

    for step in range(num_validation_steps):
        scene.step()
        x_next = _get_ipc_positions(scene, solver_type="cloth", idx=cloth_entity_idx, env_idx=0)
        assert x_next is not None, "Could not retrieve cloth vertex positions after stepping."
        x_next = x_next[0, 2]

        expected_dx = v_n * dt - g * dt * dt
        expected_x_next = x_n + expected_dx
        pos_error = abs(x_next - expected_x_next) / abs(expected_dx) if abs(expected_dx) > 1e-6 else 0.0
        assert pos_error < tolerance, f"Step {step}: Position error {pos_error * 100:.2f}% exceeds tolerance."

        v_next = (x_next - x_n) / dt
        expected_v_next = v_n - g * dt
        vel_error = abs(v_next - expected_v_next) / abs(expected_v_next) if abs(expected_v_next) > 1e-6 else 0.0
        assert vel_error < tolerance, f"Step {step}: Velocity error {vel_error * 100:.2f}% exceeds tolerance."

        x_n = x_next
        v_n = v_next


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0])
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
@pytest.mark.parametrize("fixed_base", [True, False])
def test_ipc_two_way_revolute(n_envs, coupling_type, fixed_base, show_viewer):
    """Test two-way coupling with revolute joint."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
        ),
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
            fixed=fixed_base,
        ),
    )
    scene.sim.coupler.set_entity_coupling_type(
        entity=robot,
        coupling_type=coupling_type,
    )
    scene.build(n_envs=n_envs)

    moving_link_idx = robot.get_link("moving").idx
    ipc_links = _get_ipc_rigid_links(scene, env_idx=0)
    assert moving_link_idx in ipc_links, "Moving link was not added to IPC rigid geometries."
    assert (0, moving_link_idx) in scene.sim.coupler._link_to_abd_slot, "Missing _link_to_abd_slot for moving link."
    initial_base_z = _get_entity_base_z(robot)

    max_steps = 100
    omega = 2.0 * np.pi
    dt = scene.sim_options.dt

    for i in range(max_steps):
        t = i * dt
        target_qpos = 0.5 * np.sin(omega * t)
        robot.set_dofs_position([target_qpos], zero_velocity=False)
        scene.step()

        if i > 50:
            link_idx = moving_link_idx
            env_idx = 0
            if (
                hasattr(scene.sim.coupler, "abd_data_by_link")
                and link_idx in scene.sim.coupler.abd_data_by_link
                and env_idx in scene.sim.coupler.abd_data_by_link[link_idx]
            ):
                abd_data = scene.sim.coupler.abd_data_by_link[link_idx][env_idx]
                genesis_transform = abd_data["aim_transform"]
                ipc_transform = abd_data["transform"]
                if genesis_transform is not None and ipc_transform is not None:
                    genesis_pos = genesis_transform[:3, 3]
                    ipc_pos = ipc_transform[:3, 3]
                    pos_diff = np.linalg.norm(genesis_pos - ipc_pos)
                    assert pos_diff < 0.001, f"Position difference too large: {pos_diff}"

                    genesis_euler = _rotmat_to_euler(genesis_transform[:3, :3])
                    ipc_euler = _rotmat_to_euler(ipc_transform[:3, :3])
                    rot_diff = np.linalg.norm(genesis_euler - ipc_euler)
                    assert rot_diff < 0.1, f"Rotation difference too large: {rot_diff}"

    if not fixed_base:
        final_base_z = _get_entity_base_z(robot)
        assert final_base_z <= initial_base_z - 0.03, (
            f"Free-base revolute robot base did not drop enough: dz={initial_base_z - final_base_z:.6f} m."
        )


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0])
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
@pytest.mark.parametrize("fixed_base", [True, False])
def test_ipc_two_way_prismatic(n_envs, coupling_type, fixed_base, show_viewer):
    """Test two-way coupling with prismatic joint."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
        ),
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
            file="urdf/simple/two_cube_prismatic.urdf",
            pos=(0, 0, 0.2),
            fixed=fixed_base,
        ),
    )
    scene.sim.coupler.set_entity_coupling_type(
        entity=robot,
        coupling_type=coupling_type,
    )
    scene.build(n_envs=n_envs)

    moving_link_idx = robot.get_link("moving").idx
    ipc_links = _get_ipc_rigid_links(scene, env_idx=0)
    assert moving_link_idx in ipc_links, "Slider link was not added to IPC rigid geometries."
    assert (0, moving_link_idx) in scene.sim.coupler._link_to_abd_slot, "Missing _link_to_abd_slot for slider link."
    initial_base_z = _get_entity_base_z(robot)

    max_steps = 100
    omega = 2.0 * np.pi
    dt = scene.sim_options.dt

    for i in range(max_steps):
        t = i * dt
        target_qpos = 0.15 + 0.1 * np.sin(omega * t)
        robot.set_dofs_position([target_qpos], zero_velocity=False)
        scene.step()

        if i > 50:
            link_idx = moving_link_idx
            env_idx = 0
            if (
                hasattr(scene.sim.coupler, "abd_data_by_link")
                and link_idx in scene.sim.coupler.abd_data_by_link
                and env_idx in scene.sim.coupler.abd_data_by_link[link_idx]
            ):
                abd_data = scene.sim.coupler.abd_data_by_link[link_idx][env_idx]
                genesis_transform = abd_data["aim_transform"]
                ipc_transform = abd_data["transform"]
                if genesis_transform is not None and ipc_transform is not None:
                    genesis_pos = genesis_transform[:3, 3]
                    ipc_pos = ipc_transform[:3, 3]
                    pos_diff = np.linalg.norm(genesis_pos - ipc_pos)
                    assert pos_diff < 0.001, f"Position difference too large: {pos_diff}"

                    genesis_euler = _rotmat_to_euler(genesis_transform[:3, :3])
                    ipc_euler = _rotmat_to_euler(ipc_transform[:3, :3])
                    rot_diff = np.linalg.norm(genesis_euler - ipc_euler)
                    assert rot_diff < 0.1, f"Rotation difference too large: {rot_diff}"

    if not fixed_base:
        final_base_z = _get_entity_base_z(robot)
        assert final_base_z <= initial_base_z - 0.03, (
            f"Free-base prismatic robot base did not drop enough: dz={initial_base_z - final_base_z:.6f} m."
        )


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0])
def test_ipc_cloth_gravity_freefall(n_envs, show_viewer):
    """Test cloth free fall physics validation + IPC->Genesis retrieve consistency."""
    dt = 2e-3
    g = 9.8
    z0 = 2.0
    num_steps = 50

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -g),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -g),
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

    cloth_entity_idx = _get_fem_solver_entity_idx(scene, cloth)
    cloth_matches = _find_ipc_geometries(scene, solver_type="cloth", idx=cloth_entity_idx, env_idx=0)
    assert len(cloth_matches) == 1, "Cloth entity was not uniquely added to IPC."

    initial_positions = _get_ipc_positions(scene, solver_type="cloth", idx=cloth_entity_idx, env_idx=0)
    assert initial_positions is not None, "Could not retrieve initial cloth vertex positions."
    z_initial = initial_positions[0, 2]

    for _ in range(num_steps):
        scene.step()

    final_positions = _get_ipc_positions(scene, solver_type="cloth", idx=cloth_entity_idx, env_idx=0)
    assert final_positions is not None, "Could not retrieve final cloth vertex positions."
    z_final = final_positions[0, 2]

    t_total = num_steps * dt
    actual_displacement = z_initial - z_final
    expected_displacement = 0.5 * g * t_total * (t_total + dt)
    relative_error = abs(actual_displacement - expected_displacement) / expected_displacement

    tolerance = 0.01
    assert relative_error < tolerance, (
        f"Physics validation failed: {relative_error * 100:.4f}% error (tolerance: {tolerance * 100:.2f}%)."
    )

    ipc_centroid = final_positions.mean(axis=0)
    genesis_positions = cloth.get_state().pos.detach().cpu().numpy()[0]
    genesis_centroid = genesis_positions.mean(axis=0)
    retrieve_error = np.linalg.norm(ipc_centroid - genesis_centroid)
    assert retrieve_error < 1e-3, (
        f"IPC->Genesis cloth retrieve mismatch too large: {retrieve_error:.6e} m (tolerance: 1e-3 m)."
    )


@pytest.mark.required
def test_ipc_link_filter_strict(show_viewer):
    """Strictly verify that IPC link filter controls which links are actually added to IPC."""
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

    assert entity_idx in coupler._ipc_link_filters, "Missing link filter registration for entity."
    assert coupler._ipc_link_filters[entity_idx] == {moving_link_idx}, "Unexpected link filter content."

    ipc_links = _get_ipc_rigid_links(scene, env_idx=0)
    assert moving_link_idx in ipc_links, "Filtered moving link was not added to IPC."
    assert base_link_idx not in ipc_links, "Base link should have been excluded by link filter."

    assert (0, moving_link_idx) in coupler._link_to_abd_slot, "Missing ABD slot mapping for moving link."
    assert (0, base_link_idx) not in coupler._link_to_abd_slot, "Base link unexpectedly has an ABD slot mapping."


@pytest.mark.required
def test_ipc_ipc_only_cube_freefall_height_drop(show_viewer):
    """Verify ipc_only rigid cube is in IPC and falls down under gravity."""
    dt = 1e-3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
        ),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=dt,
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
    ipc_links = _get_ipc_rigid_links(scene, env_idx=0)
    assert base_link_idx in ipc_links, "ipc_only cube base link was not added to IPC rigid geometries."
    assert (0, base_link_idx) in scene.sim.coupler._link_to_abd_slot, "Missing ABD slot mapping for ipc_only cube."

    initial_z = _get_entity_base_z(cube)
    for _ in range(120):
        scene.step()
    final_z = _get_entity_base_z(cube)
    assert final_z <= initial_z - 0.04, (
        f"ipc_only cube did not fall enough: dz={initial_z - final_z:.6f} m (expected >= 0.04 m)."
    )


@pytest.mark.required
@pytest.mark.parametrize("coupling_type", ["two_way_soft_constraint", "external_articulation"])
def test_ipc_robot_fem_grasp_retrieve_lift_strict(coupling_type, show_viewer):
    """Strict grasp test: verify FEM add/retrieve and that robot lift raises FEM by >= 0.1m."""
    dt = 1e-2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, -9.8)),
        rigid_options=None,
        coupler_options=gs.options.IPCCouplerOptions(
            dt=dt,
            gravity=(0.0, 0.0, -9.8),
            ipc_constraint_strength=(100, 100),
            disable_ipc_ground_contact=True,
            disable_ipc_logging=True,
            IPC_self_contact=False,
            contact_friction_mu=0.8,
            enable_ipc_gui=False,
            newton_transrate_tol=10,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda_non_overlap.xml"),
    )
    scene.sim.coupler.set_entity_coupling_type(
        entity=franka,
        coupling_type=coupling_type,
    )
    scene.sim.coupler.set_ipc_coupling_link_filter(
        entity=franka,
        link_names=["left_finger", "right_finger"],
    )

    fem_box = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.65, 0.0, 0.03), size=(0.05, 0.05, 0.05)),
        material=gs.materials.FEM.Elastic(E=5.0e4, nu=0.45, rho=1000.0, model="stable_neohookean"),
        surface=gs.surfaces.Plastic(color=(0.2, 0.8, 0.2, 0.5)),
    )

    scene.build(n_envs=0)

    coupler = scene.sim.coupler
    fem_entity_idx = _get_fem_solver_entity_idx(scene, fem_box)

    fem_matches = _find_ipc_geometries(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
    assert len(fem_matches) == 1, "FEM object was not uniquely added to IPC."

    left_finger_idx = franka.get_link("left_finger").idx
    right_finger_idx = franka.get_link("right_finger").idx
    expected_finger_links = {left_finger_idx, right_finger_idx}

    ipc_links = _get_ipc_rigid_links(scene, env_idx=0)
    assert expected_finger_links.issubset(ipc_links), "Required finger links were not added to IPC."
    for link_idx in expected_finger_links:
        assert (0, link_idx) in coupler._link_to_abd_slot, f"Missing ABD slot mapping for finger link {link_idx}."

    franka_link_indices = {link.idx for link in franka.links}
    franka_ipc_links = ipc_links.intersection(franka_link_indices)
    if coupling_type == "two_way_soft_constraint":
        entity_idx = franka._idx_in_solver
        assert coupler._ipc_link_filters.get(entity_idx) == expected_finger_links, "Unexpected franka IPC link filter."
        assert franka_ipc_links == expected_finger_links, "Franka IPC links should exactly match filter links."
    else:
        assert expected_finger_links.issubset(franka_ipc_links), (
            "external_articulation must include at least required finger links in IPC."
        )

    initial_ipc_centroid = _get_ipc_centroid(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
    assert initial_ipc_centroid is not None, "Could not retrieve initial FEM centroid from IPC."
    initial_genesis_positions = fem_box.get_state().pos.detach().cpu().numpy()[0]
    initial_genesis_centroid = initial_genesis_positions.mean(axis=0)
    init_retrieve_error = np.linalg.norm(initial_ipc_centroid - initial_genesis_centroid)
    assert init_retrieve_error < 1e-3, (
        f"Initial FEM retrieve mismatch too large: {init_retrieve_error:.6e} m (tolerance: 1e-3 m)."
    )
    initial_z = initial_genesis_centroid[2]

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    current_kp = franka.get_dofs_kp()
    new_kp = current_kp
    new_kp[fingers_dof] = current_kp[fingers_dof] * 5.0
    franka.set_dofs_kp(new_kp)
    end_effector = franka.get_link("hand")

    def run_stage(target_qpos, finger_pos, duration):
        num_steps = max(1, int(duration / dt))
        finger_cmd = np.array([finger_pos, finger_pos], dtype=np.float32)
        for _ in range(num_steps):
            franka.control_dofs_position(target_qpos[:-2], motors_dof)
            franka.control_dofs_position(finger_cmd, fingers_dof)
            scene.step()

    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.4]),
        quat=np.array([0.0, 1.0, 0.0, 0.0]),
    )
    run_stage(qpos, finger_pos=0.04, duration=2.0)

    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.25]),
        quat=np.array([0.0, 1.0, 0.0, 0.0]),
    )
    run_stage(qpos, finger_pos=0.04, duration=1.0)

    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.135]),
        quat=np.array([0.0, 1.0, 0.0, 0.0]),
    )
    run_stage(qpos, finger_pos=0.04, duration=0.5)

    run_stage(qpos, finger_pos=0.0, duration=0.1)

    qpos = franka.inverse_kinematics(
        link=end_effector,
        pos=np.array([0.65, 0.0, 0.4]),
        quat=np.array([0.0, 1.0, 0.0, 0.0]),
    )
    run_stage(qpos, finger_pos=0.0, duration=0.2)

    final_ipc_centroid = _get_ipc_centroid(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
    assert final_ipc_centroid is not None, "Could not retrieve final FEM centroid from IPC."
    final_genesis_positions = fem_box.get_state().pos.detach().cpu().numpy()[0]
    final_genesis_centroid = final_genesis_positions.mean(axis=0)

    retrieve_error = np.linalg.norm(final_ipc_centroid - final_genesis_centroid)
    assert retrieve_error < 1e-3, f"Final FEM retrieve mismatch too large: {retrieve_error:.6e} m (tolerance: 1e-3 m)."

    z_gain = final_genesis_centroid[2] - initial_z
    assert z_gain >= 0.1, f"FEM lift is too small: dz={z_gain:.6f} m, expected at least 0.1 m."


@pytest.mark.required
def test_ipc_motion_final_relative_error_below_2pct(show_viewer):
    """IPC momentum test: final relative error must be <= 2%."""
    dt = 1e-3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.IPCCouplerOptions(
            dt=dt,
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

    fem_entity_idx = _get_fem_solver_entity_idx(scene, blob)
    fem_matches = _find_ipc_geometries(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
    assert len(fem_matches) == 1, "FEM blob was not uniquely added to IPC."

    rigid_link_idx = rigid_cube.base_link_idx
    ipc_links = _get_ipc_rigid_links(scene, env_idx=0)
    assert rigid_link_idx in ipc_links, "Rigid cube base link was not added to IPC."
    assert (0, rigid_link_idx) in scene.sim.coupler._link_to_abd_slot, "Missing ABD slot for rigid cube base link."

    rigid_cube.set_dofs_velocity((4.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    blob_radius = blob.morph.radius
    blob_rho = blob.material.rho
    fem_total_mass = (4.0 / 3.0) * np.pi * (blob_radius**3) * blob_rho

    mass_field = scene.sim.rigid_solver.links_info.inertial_mass
    if hasattr(mass_field, "to_numpy"):
        rigid_mass = float(mass_field.to_numpy()[rigid_link_idx])
    else:
        rigid_mass = float(mass_field[rigid_link_idx])

    fem_prev_pos = None
    rigid_prev_pos = None
    total_p_history = []

    test_time = 0.30
    n_steps = int(test_time / dt)

    for _ in range(n_steps):
        rigid_pos = (
            scene.sim.rigid_solver.get_links_pos(links_idx=rigid_link_idx, ref="link_com").detach().cpu().numpy()
        )
        rigid_pos = rigid_pos.flatten()[:3]
        if rigid_prev_pos is None:
            rigid_vel = np.array([4.0, 0.0, 0.0])
        else:
            rigid_vel = (rigid_pos - rigid_prev_pos) / dt
        rigid_prev_pos = rigid_pos.copy()
        rigid_linear_momentum = rigid_mass * rigid_vel

        fem_proc_geo = _get_ipc_processed_geometry(scene, solver_type="fem", idx=fem_entity_idx, env_idx=0)
        assert fem_proc_geo is not None, "Could not retrieve FEM geometry from IPC."
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
            fem_vertex_velocities = (fem_vertex_positions - fem_prev_pos) / dt
        fem_prev_pos = fem_vertex_positions.copy()

        fem_linear_momentum = np.sum(fem_vertex_masses[:, np.newaxis] * fem_vertex_velocities, axis=0)
        total_linear_momentum = rigid_linear_momentum + fem_linear_momentum
        total_p_history.append(np.asarray(total_linear_momentum).flatten().copy())

        scene.step()

    total_p_history = np.array(total_p_history)
    expected_momentum = np.array([4.0 * rigid_mass, 0.0, 0.0])
    final_relative_error = np.linalg.norm(total_p_history[-1] - expected_momentum) / np.linalg.norm(expected_momentum)

    assert final_relative_error <= 0.02, (
        f"Final momentum relative error too large: {final_relative_error * 100:.4f}% (expected <= 2.0%)."
    )
