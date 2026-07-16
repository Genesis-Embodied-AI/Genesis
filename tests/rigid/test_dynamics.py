import math

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from ..utils import (
    assert_allclose,
    init_simulators,
)


@pytest.mark.required
def test_gravity(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
    )

    sphere = scene.add_entity(gs.morphs.Sphere())
    scene.build(n_envs=3)

    scene.sim.set_gravity(torch.tensor([0.0, 0.0, 0.0]))
    scene.sim.set_gravity(torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), envs_idx=[0, 1])
    scene.sim.set_gravity(torch.tensor([0.0, 0.0, 3.0]), envs_idx=2)
    with np.testing.assert_raises(RuntimeError):
        scene.sim.set_gravity(torch.tensor([0.0, -10.0]))
    with np.testing.assert_raises(RuntimeError):
        scene.sim.set_gravity(torch.tensor([[0.0, 0.0, -10.0], [0.0, 0.0, -10.0]]), envs_idx=1)

    scene.step()

    assert_allclose(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        sphere.get_links_acc()[..., 0, :],
        tol=tol,
    )


@pytest.mark.required
def test_all_fixed(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3, 1, 1.5),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.0),
            fixed=True,
        ),
    )
    scene.build()
    scene.step()

    assert_allclose(cube.get_pos(), 0, tol=gs.EPS)
    assert_allclose(cube.get_quat(), (1.0, 0.0, 0.0, 0.0), tol=gs.EPS)
    assert_allclose(cube.get_vel(), 0, tol=gs.EPS)
    assert_allclose(cube.get_ang(), 0, tol=gs.EPS)
    assert_allclose(scene.rigid_solver.get_links_acc(), 0, tol=gs.EPS)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["box_box"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_box_box_dynamics(gs_sim):
    (gs_robot,) = gs_sim.entities
    for _ in range(20):
        cube1_pos = np.array([0.0, 0.0, 0.2])
        cube1_quat = np.array([1.0, 0.0, 0.0, 0.0])
        cube2_pos = np.array([0.0, 0.0, 0.65 + 0.1 * np.random.rand()])
        cube2_quat = gu.xyz_to_quat(
            np.array([*(0.15 * np.random.rand(2)), np.pi * np.random.rand()]),
        )
        init_simulators(gs_sim, qpos=np.concatenate((cube1_pos, cube1_quat, cube2_pos, cube2_quat)))
        for i in range(110):
            gs_sim.scene.step()
            if i > 100:
                qvel = gs_robot.get_dofs_velocity()
                assert_allclose(qvel, 0, atol=1e-2)

        qpos = gs_robot.get_dofs_position()
        assert_allclose(qpos[8], 0.6, atol=2e-3)


@pytest.mark.debug(False)  # Disable debug for speedup
@pytest.mark.parametrize(
    "box_box_detection, gjk_collision, dynamics",
    [
        (True, False, False),
        (False, False, False),
        (False, False, True),
        (False, True, False),
    ],
)
@pytest.mark.parametrize("backend", [gs.cpu])  # TODO: Cannot afford GPU test for this one
def test_many_boxes_dynamics(box_box_detection, gjk_collision, dynamics, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
        ),
        rigid_options=gs.options.RigidOptions(
            max_collision_pairs=1000,
            box_box_detection=box_box_detection,
            use_gjk_collision=gjk_collision,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(10, 10, 10),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    for n in range(5**3):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        scene.add_entity(
            gs.morphs.Box(
                pos=(i * (1.0 - 1e-3), j * (1.0 - 1e-3), k * (1.0 - 1e-3) + 0.5),
                size=(1.0, 1.0, 1.0),
            ),
            surface=gs.surfaces.Default(
                color=(*np.random.rand(3), 0.7),
            ),
        )
    scene.build()

    if dynamics:
        for entity in scene.entities[1:]:
            entity.set_dofs_velocity(4.0 * np.random.rand(6))
    num_steps = 850 if dynamics else 150
    for i in range(num_steps):
        scene.step()
        if i > num_steps - 50:
            qvel = scene.rigid_solver.get_dofs_velocity().reshape((6, -1))
            # Checking the average velocity because is always one cube moving depending on the machine.
            assert_allclose(torch.linalg.norm(qvel, dim=0).mean(), 0, atol=0.05)

    for n, entity in enumerate(scene.entities[1:]):
        i, j, k = int(n / 25), int(n / 5) % 5, n % 5
        qpos = entity.get_dofs_position()
        if dynamics:
            assert qpos[:2].norm() < 20.0
            assert qpos[2] < 5.0
        else:
            qpos0 = np.array((i * (1.0 - 1e-3), j * (1.0 - 1e-3), k * (1.0 - 1e-3) + 0.5))
            assert_allclose(qpos[:3], qpos0, atol=0.05)
            assert_allclose(qpos[3:], 0, atol=0.03)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["double_ball_pendulum"])
def test_apply_external_forces(xml_path, show_viewer):
    GRAVITY = 2.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=2,
            gravity=(0, 0, -GRAVITY),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0, -3.5, 2.5),
            camera_lookat=(0.0, 0.0, 1.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )

    scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            quat=(1.0, 0, 1.0, 0),
        ),
    )
    duck = scene.add_entity(
        morph=gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.04,
            pos=(1.0, 0.0, 1.0),
            euler=(90, 0, 0),
            collision=False,
        ),
    )
    scene.build()
    rigid_solver = scene.rigid_solver

    end_effector_link_idx = robot.links[-1].idx
    duck_link_idx = duck.links[0].idx
    duck_mass = duck.get_mass()
    duck_init_link_pos, duck_init_link_R = duck.base_link.pos, gu.quat_to_R(duck.base_link.quat)
    for step in range(801):
        ee_pos = rigid_solver.get_links_pos([end_effector_link_idx])[0]
        duck_pos = rigid_solver.get_links_pos([duck_link_idx])[0]
        if step == 0:
            assert_allclose(ee_pos, (0.8, 0.0, 0.02), tol=1e-4)
        elif step in (500, 600):
            assert_allclose(ee_pos, (0.0, 0.0, 0.82), tol=0.01)
        elif step == 800:
            assert_allclose(ee_pos, (-0.8 / math.sqrt(2), 0.8 / math.sqrt(2), 0.02), tol=0.02)
        assert_allclose(duck_pos, duck_init_link_pos, tol=1e-3)

        if step >= 600:
            force = [-4.0, 4.0, 0.0]
            torque = [0.0, 0.0, 0.0]
        elif step >= 500:
            force = [0.0, 0.0, 0.0]
            torque = [0.0, 0.0, 2.0]
        elif step >= 50:
            force = [0.0, 0.0, 10.0]
            torque = [0.0, 0.0, 0.0]
        else:
            force = [0.0, 0.0, 0.0]
            torque = [0.0, 0.0, 0.0]

        rigid_solver.apply_links_external_force(
            force=duck_mass * GRAVITY * duck_init_link_R[2], links_idx=[duck_link_idx], ref="link_com", local=True
        )
        rigid_solver.apply_links_external_force(
            force=force, links_idx=[end_effector_link_idx], ref="link_origin", local=False
        )
        rigid_solver.apply_links_external_torque(
            torque=torque, links_idx=[end_effector_link_idx], ref="link_origin", local=False
        )
        scene.step()

    rigid_solver.apply_links_external_torque(torque=(0, 1, 0), links_idx=[duck_link_idx], ref="link_com", local=True)
    assert_allclose(rigid_solver.links_state.cfrc_applied_vel[duck_link_idx, 0], 0, tol=gs.EPS)
    assert_allclose(rigid_solver.links_state.cfrc_applied_ang[duck_link_idx, 0], -duck_init_link_R[:, 1], tol=gs.EPS)

    with np.testing.assert_raises(ValueError):
        rigid_solver.apply_links_external_force(force=(0, 0, 0), links_idx=[duck_link_idx], ref="root_com", local=True)
    with np.testing.assert_raises(ValueError):
        rigid_solver.apply_links_external_torque(
            torque=(0, 0, 0), links_idx=[duck_link_idx], ref="root_com", local=True
        )


@pytest.mark.required
@pytest.mark.parametrize("integrator", [gs.integrator.Euler, gs.integrator.approximate_implicitfast])
def test_energy_analytical_and_conservation(show_viewer, tol, integrator):
    g = 9.81
    dt = 0.001
    h0 = 0.5
    radius = 0.1
    n_steps = 400
    undamped_sol_params = [10.0, 0.001, 0.9, 0.95, 0.001, 0.5, 2.0]

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            gravity=(0, 0, -g),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.25, 1.5, 0.7),
            camera_lookat=(0.25, 0.0, 0.2),
        ),
        rigid_options=gs.options.RigidOptions(
            integrator=integrator,
        ),
        show_viewer=show_viewer,
    )
    plane = scene.add_entity(gs.morphs.Plane())
    sphere_a = scene.add_entity(
        gs.morphs.Sphere(
            radius=radius,
            pos=(0, 0, h0),
        ),
    )
    sphere_b = scene.add_entity(
        gs.morphs.Sphere(
            radius=radius,
            pos=(0.5, 0, h0),
        ),
    )
    scene.build()

    # Nearly undamped contact for sphere_a: small dampratio gives very stiff elastic spring with minimal damping.
    # Contact sol_params are averaged: 0.5*(geom_a + geom_b), so both geoms must share the same params.
    plane.geoms[0].set_sol_params(undamped_sol_params)
    sphere_a.geoms[0].set_sol_params(undamped_sol_params)

    mass = sphere_a.get_links_inertial_mass()
    te_initial = sphere_a.get_total_energy()

    ke_a, pe_a, ke_b, pe_b = [], [], [], []
    impact_step = -1
    for i in range(n_steps):
        scene.step()
        ke_a.append(sphere_a.get_kinetic_energy())
        pe_a.append(sphere_a.get_potential_energy())
        ke_b.append(sphere_b.get_kinetic_energy())
        pe_b.append(sphere_b.get_potential_energy())
        if impact_step < 0 and scene.rigid_solver.collider._collider_state.n_contacts.to_numpy().any():
            impact_step = i
    assert impact_step > 0

    # Free fall: verify analytical KE and PE (semi-implicit Euler)
    # After step n: v_n = n*g*dt, z_n = h0 - g*dt^2*n*(n+1)/2
    for i in range(impact_step):
        n = i + 1
        expected_ke = 0.5 * mass * (n * g * dt) ** 2
        expected_pe = mass * g * (h0 - g * dt**2 * n * (n + 1) / 2)
        assert_allclose(ke_a[i], expected_ke, tol=tol)
        assert_allclose(pe_a[i], expected_pe, tol=tol)
        assert_allclose(ke_b[i], expected_ke, tol=tol)
        assert_allclose(pe_b[i], expected_pe, tol=tol)

    # Undamped sphere_a: energy conserved after bouncing (drift < 1%)
    te_a_final = ke_a[-1] + pe_a[-1]
    assert_allclose(te_a_final, te_initial, tol=0.01)

    # Damped sphere_b: energy strictly decreased
    te_b_final = ke_b[-1] + pe_b[-1]
    assert te_b_final < te_initial


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("model_name", ["long_chain"])
def test_mass_mat(xml_path, show_viewer, tol):
    # Create and build the scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=1,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    franka1 = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 0, 0)),
        vis_mode="collision",
        visualize_contact=True,
    )
    franka2 = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0, 2, 0)),
        vis_mode="collision",
        visualize_contact=True,
    )
    # High-DOF single tree: its mass submatrix exceeds GPU shared memory, exercising the cooperative >shared-cap
    # assemble (the low-DOF frankas exercise the under-cap shared-memory factor instead).
    long_chain = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
            pos=(5, 0, 2),
        ),
    )
    scene.build()

    # Two identical entities must yield identical mass matrices, and the LTDL factor must reconstruct it.
    mass_mat_1 = franka1.get_mass_mat(decompose=False)
    mass_mat_2 = franka2.get_mass_mat(decompose=False)
    assert mass_mat_1.shape == (franka1.n_dofs, franka1.n_dofs)
    assert_allclose(mass_mat_1, mass_mat_2, tol=tol)

    mass_mat_L, mass_mat_D_inv = franka1.get_mass_mat(decompose=True)
    mass_mat = mass_mat_L.T @ torch.diag(1.0 / mass_mat_D_inv) @ mass_mat_L
    assert_allclose(mass_mat, mass_mat_1, tol=tol)

    # The cooperative >shared-cap assemble maps a flat lane index to a lower-triangular (row, col) via a float sqrt;
    # on GPUs whose sqrt undershoots perfect squares (Apple Metal: sqrt(15129) -> 122.999 instead of 123) a naive
    # inversion lands one row short on every j=0 boundary and silently drops the long-range coupling entries, leaving
    # the assembled mass matrix indefinite. A real joint-space mass matrix is always symmetric positive-definite.
    mass_mat_chain = tensor_to_array(long_chain.get_mass_mat(decompose=False))
    assert_allclose(mass_mat_chain, mass_mat_chain.T, tol=tol)
    assert np.linalg.eigvalsh(0.5 * (mass_mat_chain + mass_mat_chain.T)).min() > 0.0

    # On GPU the high-DOF chain factors through the register-tiled path (auto-enabled above the shared-memory cap when
    # RigidOptions.register_tiled_mass is left to its default); its LTDL factor must reconstruct the mass matrix to the
    # same accuracy as the under-cap path.
    mass_mat_chain_L, mass_mat_chain_D_inv = long_chain.get_mass_mat(decompose=True)
    mass_mat_chain_rec = mass_mat_chain_L.T @ torch.diag(1.0 / mass_mat_chain_D_inv) @ mass_mat_chain_L
    assert_allclose(mass_mat_chain_rec, mass_mat_chain, tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["two_fixed_branches"])
def test_mass_block_partition(xml_path, show_viewer, tol):
    # Two chains rigidly attached to the fixed world are kinematically independent: the mass matrix is block-diagonal,
    # so it must partition into one mass block per branch (factoring two n/2 blocks instead of one dense n block).
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
        ),
        show_viewer=show_viewer,
    )
    entity = scene.add_entity(
        gs.morphs.MJCF(
            file=xml_path,
        ),
    )
    scene.build(n_envs=0)

    n_dofs = entity.n_dofs
    branch = n_dofs // 2
    block_start = qd_to_numpy(scene.rigid_solver._rigid_global_info.dofs_mass_block_start)
    block_end = qd_to_numpy(scene.rigid_solver._rigid_global_info.dofs_mass_block_end)
    assert_allclose(block_start, [0] * branch + [branch] * branch, tol=0)
    assert_allclose(block_end, [branch] * branch + [n_dofs] * branch, tol=0)

    # The two branches do not couple, and the LTDL factor reconstructs the (block-diagonal) mass matrix.
    mass_mat = tensor_to_array(entity.get_mass_mat(decompose=False))
    assert_allclose(mass_mat[:branch, branch:], 0.0, tol=tol)
    mass_mat_L, mass_mat_D_inv = entity.get_mass_mat(decompose=True)
    assert_allclose(mass_mat_L.T @ torch.diag(1.0 / mass_mat_D_inv) @ mass_mat_L, mass_mat, tol=tol)


@pytest.mark.slow  # ~500s
@pytest.mark.required
@pytest.mark.parametrize("precision", ["32", "64"])
@pytest.mark.parametrize("backend", [gs.gpu])
def test_cholesky_tiling(monkeypatch, tol):
    import genesis.engine.solvers

    rigid_solver_build_orig = genesis.engine.solvers.RigidSolver.build

    values = []
    for enable_tiled_cholesky in (True, False):

        def rigid_solver_build(self):
            nonlocal enable_tiled_cholesky

            rigid_solver_build_orig(self)
            self._static_rigid_sim_config.enable_tiled_cholesky_mass_matrix = enable_tiled_cholesky
            self._static_rigid_sim_config.enable_tiled_cholesky_hessian = enable_tiled_cholesky
            if enable_tiled_cholesky:
                self._static_rigid_sim_config.tiled_n_dofs_per_entity = 32
                self._static_rigid_sim_config.tiled_n_dofs = 32

        monkeypatch.setattr("genesis.engine.solvers.RigidSolver.build", rigid_solver_build)

        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                sparse_solve=False,
                iterations=1,
            ),
            show_viewer=False,
            show_FPS=False,
        )
        scene.add_entity(gs.morphs.Plane())
        gs_robot = scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
            ),
        )
        scene.build(n_envs=2)
        assert scene.rigid_solver._static_rigid_sim_config.enable_tiled_cholesky_mass_matrix == enable_tiled_cholesky
        assert scene.rigid_solver._static_rigid_sim_config.enable_tiled_cholesky_hessian == enable_tiled_cholesky

        scene.step()
        assert not scene.rigid_solver.get_error_envs_mask().any()
        assert (scene.rigid_solver.constraint_solver.constraint_state.n_constraints.to_numpy() > 0).all()

        Mgrad = scene.rigid_solver.constraint_solver.constraint_state.Mgrad.to_numpy()
        assert np.linalg.norm(Mgrad) > 5.0
        values.append(Mgrad)

    # analysis for choice tolerance: https://github.com/Genesis-Embodied-AI/Genesis/pull/2659#discussion_r3041684256
    assert_allclose(*values, tol=5e-4)


@pytest.mark.slow  # ~200s
@pytest.mark.precision("32")
@pytest.mark.parametrize("backend", [gs.cuda])
def test_cholesky_tiling_large_shared_memory(show_viewer):
    if gs.device.type != "cuda":
        pytest.skip("Requires CUDA device")

    from cuda.bindings import runtime  # Transitive dependency of torch CUDA

    _, max_shared_mem = runtime.cudaDeviceGetAttribute(
        runtime.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerBlockOptin, gs.device.index
    )
    if max_shared_mem <= 49152:
        pytest.skip("GPU does not support opt-in shared memory beyond the default 48kB")

    # Stack 17 free boxes (6 DOFs each = 102 total) to exceed the default 48kB tiling limit of 96 DOFs for f32
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 1.0, 2.5),
            camera_lookat=(0.0, 0.0, 1.2),
        ),
        rigid_options=gs.options.RigidOptions(
            constraint_solver=gs.constraint_solver.Newton,
            sparse_solve=False,
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    for i in range(17):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(0, 0, 0.5 + i * 0.15),
            )
        )
    scene.build(n_envs=2)

    assert scene.rigid_solver.n_dofs == 102
    assert scene.rigid_solver._static_rigid_sim_config.enable_tiled_cholesky_hessian

    scene.step()
    assert not scene.rigid_solver.get_error_envs_mask().any()
