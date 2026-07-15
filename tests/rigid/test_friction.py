import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu

from ..utils import (
    assert_allclose,
    get_hf_dataset,
    simulate_and_check_mujoco_consistency,
)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["hinge_slide"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
@pytest.mark.parametrize("backend", [gs.cpu])
def test_frictionloss(gs_sim, mj_sim, tol):
    qvel = np.array([0.7, -0.9])
    simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qvel=qvel, num_steps=2000, tol=tol)

    # Check that final velocity is almost zero
    gs_qvel = gs_sim.rigid_solver.dofs_state.vel.to_numpy()
    assert_allclose(gs_qvel, 0.0, tol=1e-2)


@pytest.mark.required
@pytest.mark.parametrize("model_name", ["hinge_slide"])
@pytest.mark.parametrize("gs_solver", [gs.constraint_solver.CG, gs.constraint_solver.Newton])
@pytest.mark.parametrize("gs_integrator", [gs.integrator.implicitfast, gs.integrator.Euler])
def test_set_dofs_frictionloss_physics(gs_sim, tol):
    (robot,) = gs_sim.entities

    initial_velocity = np.array([1.0, 0.0])
    robot.set_dofs_velocity(initial_velocity)

    robot.set_dofs_frictionloss(np.array([0.0, 0.0]))
    frictionloss = robot.get_dofs_frictionloss()
    assert_allclose(frictionloss, np.array([0.0, 0.0]), atol=tol)
    for _ in range(10):
        gs_sim.step()
    velocity_zero = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

    robot.set_dofs_velocity(initial_velocity)
    robot.set_dofs_frictionloss(np.array([1.0, 0.0]))
    frictionloss = robot.get_dofs_frictionloss()
    assert_allclose(frictionloss, np.array([1.0, 0.0]), atol=tol)
    for _ in range(10):
        gs_sim.step()
    velocity_high = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

    np.testing.assert_array_less(velocity_high[0], velocity_zero[0])
    np.testing.assert_array_less(velocity_high[1], velocity_zero[1])

    robot.set_dofs_velocity(initial_velocity)
    robot.set_dofs_frictionloss(np.array([0.5]), dofs_idx_local=[0])
    frictionloss = robot.get_dofs_frictionloss(dofs_idx_local=[0])
    assert_allclose(frictionloss, np.array([0.5]), atol=tol)
    for _ in range(10):
        gs_sim.step()
    velocity_medium = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]

    np.testing.assert_array_less(velocity_high[0], velocity_medium[0])
    np.testing.assert_array_less(velocity_medium[0], velocity_zero[0])

    friction_effect = velocity_zero[0] - velocity_high[0]
    np.testing.assert_array_less(tol, friction_effect)

    slide_friction_effect = velocity_zero[1] - velocity_high[1]
    np.testing.assert_array_less(tol, slide_friction_effect)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_frictionloss_advanced(show_viewer, tol):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.4, 0.7, 1.4),
            camera_lookat=(0.6, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(gs.morphs.Plane())
    asset_path = get_hf_dataset(pattern="SO101/*")
    robot = scene.add_entity(
        morph=gs.morphs.MJCF(
            file=f"{asset_path}/SO101/so101_new_calib.xml",
        ),
        # vis_mode="collision",
    )
    box = scene.add_entity(
        gs.morphs.Box(
            pos=(0.1, 0.0, 0.6),
            size=(0.025, 0.025, 0.025),
        ),
    )
    scene.build()

    scene.reset()
    for _ in range(230):
        scene.step()

    assert_allclose(robot.get_contacts()["position"][:, 2].min(), 0.0, tol=1e-4)
    assert_allclose(robot.get_AABB()[0, 2], 0.0, tol=2e-4)
    box_pos = box.get_pos()
    assert box_pos[0] > 0.4

    # This is to check collision detection is working correctly on Apple Metal.
    # The box should collide with the robot and roll on the ground within a reasonable range without not blow up.
    assert_allclose(box_pos[1:], 0.0, tol=0.05)
    assert_allclose(box.get_dofs_velocity(), 0.0, tol=50 * tol)


@pytest.mark.slow("gpu")  # gpu ~250s
@pytest.mark.debug(False)  # Disable debug for speedup
@pytest.mark.parametrize(
    "backend, mode, friction, n_boxes, solver, scale, mesh_boxes",
    [
        # Two floating boxes (the original noslip scenario): a balanced half-fraction of the backend x friction x
        # scale x geometry matrix - every axis value appears four times and every axis-value pair twice.
        pytest.param(gs.cpu, "noslip", 0.5, 2, gs.constraint_solver.Newton, 0.04, False, marks=pytest.mark.required),
        (gs.cpu, "noslip", 0.5, 2, gs.constraint_solver.Newton, 1.0, True),
        pytest.param(gs.cpu, "noslip", 2.0, 2, gs.constraint_solver.Newton, 0.04, True, marks=pytest.mark.required),
        (gs.cpu, "noslip", 2.0, 2, gs.constraint_solver.Newton, 1.0, False),
        (gs.gpu, "noslip", 0.5, 2, gs.constraint_solver.Newton, 0.04, True),
        pytest.param(gs.gpu, "noslip", 0.5, 2, gs.constraint_solver.Newton, 1.0, False, marks=pytest.mark.required),
        (gs.gpu, "noslip", 2.0, 2, gs.constraint_solver.Newton, 0.04, False),
        pytest.param(gs.gpu, "noslip", 2.0, 2, gs.constraint_solver.Newton, 1.0, True, marks=pytest.mark.required),
        # Constraint-solver coverage: the CG configs document the baseline users can expect from CG. It holds the
        # two-box chain (elliptic at the near-exact Coulomb push here, noslip on CPU below); the three-box chain at
        # the same pushes is beyond its convergence and stays on Newton.
        (gs.gpu, "elliptic", 2.0, 2, gs.constraint_solver.CG, 1.0, False),
        # Three floating boxes: the longer friction chain both mechanisms must hold. At 18 DOF the chain turns
        # islands on and, on GPU past the 16-DOF cooperative threshold, engages the decomposed arm; the islands-off
        # elliptic arms are covered by test_elliptic_cone_coulomb_isotropy. CG rides the lighter-load configs; the
        # stiff high-load cases stay on Newton, which CG cannot hold as tightly. The small-scale mesh configs cover
        # scale sensitivity and mesh contacts.
        pytest.param(gs.cpu, "elliptic", 2.0, 3, gs.constraint_solver.Newton, 1.0, False, marks=pytest.mark.required),
        (gs.cpu, "elliptic", 0.5, 3, gs.constraint_solver.Newton, 0.04, True),
        pytest.param(gs.gpu, "elliptic", 2.0, 3, gs.constraint_solver.Newton, 1.0, False, marks=pytest.mark.required),
        (gs.gpu, "elliptic", 0.5, 3, gs.constraint_solver.Newton, 0.04, True),
        pytest.param(gs.cpu, "noslip", 2.0, 3, gs.constraint_solver.Newton, 0.04, True, marks=pytest.mark.required),
        (gs.cpu, "noslip", 0.5, 3, gs.constraint_solver.CG, 1.0, False),
        pytest.param(gs.gpu, "noslip", 2.0, 3, gs.constraint_solver.Newton, 1.0, False, marks=pytest.mark.required),
    ],
)
def test_static_friction(mode, friction, n_boxes, solver, scale, mesh_boxes, show_viewer, asset_tmp_path):
    # A shear-loaded stack of n_boxes floating boxes braced against a fixed wall must stay static under either
    # creep-suppression mechanism: noslip (pyramidal cone + noslip post-iterations) or the elliptic cone (high
    # tangential impedance). Regularized friction alone lets the stack slowly creep under sustained shear; both hold.
    GRAVITY = -9.81
    # SAFETY_FACTOR scales the applied shear above the theoretical minimum (weight / mu) that braces the stack. The
    # pyramidal cone inscribes the true friction cone and its regularized friction creeps, so noslip must over-push
    # ~2.5x; the elliptic cone enforces the exact Coulomb limit and holds at nearly the theoretical force (the static
    # hold breaks down just below ~1.08, since the fixed wall braces the stack only through the inter-box friction
    # chain). Residual creep shrinks monotonically with the tangential impedance ratio impratio: 20 still creeps past
    # tolerance over this horizon, ~50 holds marginally, and the default 100 holds with margin.
    SAFETY_FACTOR = 1.1 if mode == "elliptic" else 2.5
    # The noslip iteration count is tuned per chain length to match the elliptic cone's static hold: 5 iterations
    # converge the two-box chain at every scale, while the three-box chain at small scale starves at 5 (steady
    # residual creep, solver-independent) and converges from ~15.
    NOSLIP_ITERATIONS = 5 if n_boxes == 2 else 15

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            constraint_solver=solver,
            noslip_iterations=NOSLIP_ITERATIONS if mode == "noslip" else 0,
            friction_cone=gs.friction_cone.elliptic if mode == "elliptic" else gs.friction_cone.pyramidal,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=((0.5 * n_boxes + 4) * scale, (n_boxes + 1.5) * scale, 3 * scale),
            camera_lookat=(0.5 * n_boxes * scale, 0.0, 0.0),
        ),
        show_viewer=show_viewer,
    )

    for i in range(n_boxes + 1):
        box_size = (scale, scale * (1 + 0.3 * (2 - i)), scale * (1 + 0.3 * (2 - i)))
        if mesh_boxes:
            mesh_path = str(asset_tmp_path / f"static_friction_box_{scale}_{i}.obj")
            trimesh.creation.box(extents=box_size).export(mesh_path, file_type="obj")
            morph = gs.morphs.Mesh(
                file=mesh_path,
                pos=(i * scale, 0, 0),
                fixed=(i == 0),
            )
        else:
            morph = gs.morphs.Box(
                size=box_size,
                pos=(i * (1 - 1e-3) * scale, 0, 0),
                fixed=(i == 0),
            )
        scene.add_entity(
            morph,
            material=gs.materials.Rigid(
                rho=200.0,
                friction=friction,
            ),
            visualize_contact=True,
        )

    floating_boxes = scene.entities[1:]
    scene.build()

    # The solver arms are provably exercised across the parametrization: a single floating box is one island on the
    # dense monolith path, multiple floating boxes turn islands on, and on GPU the cooperative decomposed arm - the
    # path that regressed the elliptic slip - engages once the floating chain reaches the 16-DOF threshold (3 boxes).
    # prefer_decomposed_solver is pinned by the test infra (1 on GPU, 0 on CPU) and the decomposed arm is kept only
    # where the cooperative kernels engage.
    rigid_solver = scene.sim.rigid_solver
    assert rigid_solver._use_contact_island == (n_boxes > 1)
    if gs.backend != gs.cpu:
        assert rigid_solver._static_rigid_sim_config.enable_cooperative_constraint_kernels == (6 * n_boxes >= 16)
        assert rigid_solver._static_rigid_sim_config.prefer_decomposed_solver == (6 * n_boxes >= 16)

    # Force needed to hold the floating boxes static without slipping
    total_mass = sum(box.get_mass() for box in floating_boxes)
    force_x = (total_mass * GRAVITY) / friction

    # Push the furthest floating box toward the fixed wall
    floating_boxes[-1].control_dofs_force(SAFETY_FACTOR * force_x, dofs_idx_local=0)

    # Position-based orientation control stabilizes the contacts
    for box in floating_boxes:
        box.set_dofs_kp(1000.0 * total_mass, dofs_idx_local=slice(3, 6))
        box.set_dofs_kv(100.0 * total_mass, dofs_idx_local=slice(3, 6))
        box.control_dofs_position(box.get_dofs_position(dofs_idx_local=slice(3, 6)), dofs_idx_local=slice(3, 6))

    # Record rest positions after warmup
    for _ in range(50):
        scene.step()
    boxes_pos_init = [box.get_pos() for box in floating_boxes]

    # Hold under sustained shear for 20 seconds
    for _ in range(2000):
        scene.step()

    # The floating boxes stay static
    assert_allclose([box.get_pos() for box in floating_boxes], boxes_pos_init, atol=5e-3)

    # Drop the force below the theoretical threshold; the stack loses its brace and falls
    floating_boxes[-1].control_dofs_force(0.95 * force_x, dofs_idx_local=0)
    for _ in range(300):
        scene.step()
    for box in floating_boxes:
        _, _, box_z = box.get_pos()
        assert box_z < -scale


@pytest.mark.required
@pytest.mark.parametrize(
    "sparse_solve, use_contact_island",
    [
        # Beyond the default arms, the explicit-sparse config pins the elliptic whole-env skyline factor (on CPU,
        # with islands off so the skyline envelope owns the factorization) and the GPU sparse build (which must
        # rebuild with the cone baked in each iteration since the CPU-only incremental cone update is compiled out).
        (None, True),
        (True, False),
    ],
)
def test_elliptic_cone_coulomb_isotropy(sparse_solve, use_contact_island, show_viewer):
    # With the box yaw and the tangential center-of-mass force in independent random directions across parallel envs, a
    # box on a plane must slide above the Coulomb threshold |F_t| = mu*N and hold static below it, identically per env.
    GRAVITY = -9.81
    MU = 1.0
    DT = 0.005
    N_ENVS = 16

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            sparse_solve=sparse_solve,
            use_contact_island=use_contact_island,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 1.0, 0.7),
            camera_lookat=(0.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction=MU,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.1),
        ),
        material=gs.materials.Rigid(
            friction=MU,
        ),
    )
    scene.build(n_envs=N_ENVS)
    mass = box.get_mass()
    normal_force = MU * mass * (-GRAVITY)

    yaw = 2.0 * torch.pi * torch.rand(N_ENVS, device=gs.device)
    direction = 2.0 * torch.pi * torch.rand(N_ENVS, device=gs.device)
    zeros = torch.zeros(N_ENVS, device=gs.device)
    quat = torch.stack((torch.cos(0.5 * yaw), zeros, zeros, torch.sin(0.5 * yaw)), dim=1)
    force_dir = torch.stack((torch.cos(direction), torch.sin(direction)), dim=1)

    def settle():
        box.control_dofs_force(0.0, dofs_idx_local=[0, 1])
        box.set_pos((0.0, 0.0, 0.1))
        box.set_quat(quat)
        box.set_dofs_velocity(
            torch.cat(
                (0.02 * torch.randn(N_ENVS, 2, device=gs.device), torch.zeros(N_ENVS, 4, device=gs.device)), dim=1
            )
        )
        # Hold each orientation so the CoM force slides the box instead of tipping it about the contact.
        box.set_dofs_kp(1.0e3 * mass, dofs_idx_local=slice(3, 6))
        box.set_dofs_kv(1.0e2 * mass, dofs_idx_local=slice(3, 6))
        box.control_dofs_position(box.get_dofs_position(dofs_idx_local=slice(3, 6)), dofs_idx_local=slice(3, 6))
        for _ in range(25):
            scene.step()

    # Above the Coulomb threshold: the box slides, and the elliptic cone makes the sliding acceleration identical in
    # every direction. Skip the initial transient, then measure the acceleration over a fixed window.
    settle()
    box.control_dofs_force(1.5 * normal_force * force_dir, dofs_idx_local=[0, 1])
    for _ in range(10):
        scene.step()
    vel_0 = box.get_dofs_velocity(dofs_idx_local=[0, 1])
    for _ in range(20):
        scene.step()
    vel_1 = box.get_dofs_velocity(dofs_idx_local=[0, 1])
    accel = torch.linalg.norm(vel_1 - vel_0, dim=1) / (20 * DT)
    # The elliptic spread measures ~1e-5 relative; the pyramidal cone's anisotropy spreads it to ~0.5.
    assert accel.std() < 5e-5 * accel.mean()

    # Below the Coulomb threshold: friction holds the box static in every direction, with no slow tangential creep.
    # The elliptic residual measures ~1e-5; the pyramidal cone's regularized friction creeps at ~1e-3.
    settle()
    box.control_dofs_force(0.4 * normal_force * force_dir, dofs_idx_local=[0, 1])
    for _ in range(40):
        scene.step()
    assert (torch.linalg.norm(box.get_dofs_velocity(dofs_idx_local=[0, 1]), dim=1) < 5e-5).all()


@pytest.mark.required
def test_elliptic_cone_push_isotropy(show_viewer):
    N_ENVS = 8
    FRICTION = 0.5
    BOX_POS = (0.0, 0.0, 0.05)
    # Pusher path in the box's local frame; the shared +y offset gives the push a lever arm that spins the box.
    PUSH_START_LOCAL = (-0.15, 0.03, 0.05)
    PUSH_END_LOCAL = (0.02, 0.03, 0.05)
    POSE_TOL = 2e-4

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.7, 0.7, 0.45),
            camera_lookat=(0.0, 0.0, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            pos=BOX_POS,
            size=(0.1, 0.2, 0.1),
        ),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    pusher = scene.add_entity(
        gs.morphs.Cylinder(
            pos=PUSH_START_LOCAL,
            height=0.1,
            radius=0.02,
        ),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    scene.build(n_envs=N_ENVS)

    yaw = 2.0 * torch.pi * torch.arange(N_ENVS, device=gs.device) / N_ENVS
    box_quat = gu.xyz_to_quat(torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), yaw), dim=1), rpy=True)
    box.set_quat(box_quat)

    # Rotate the local pusher path into each env's world frame by the box yaw, and PD-control the pusher's full pose.
    push_start = gu.transform_by_quat(torch.tensor(PUSH_START_LOCAL, device=gs.device).repeat(N_ENVS, 1), box_quat)
    push_end = gu.transform_by_quat(torch.tensor(PUSH_END_LOCAL, device=gs.device).repeat(N_ENVS, 1), box_quat)
    pusher.set_pos(push_start)
    pusher.set_dofs_kp(
        pusher.get_mass() * torch.tensor((2000.0, 2000.0, 2000.0, 500.0, 500.0, 500.0), device=gs.device)
    )
    pusher.set_dofs_kv(pusher.get_mass() * torch.tensor((200.0, 200.0, 200.0, 50.0, 50.0, 50.0), device=gs.device))

    # Let the box resolve its initial ground contact before the push starts, so the two transients do not couple.
    scene.step()

    # Drive the pusher forward through the box while holding its height and orientation.
    pusher.control_dofs_position(push_end, dofs_idx_local=[0, 1, 2])
    pusher.control_dofs_position(0.0, dofs_idx_local=[3, 4, 5])
    for _ in range(160):
        scene.step()

    # The box and pusher settle at rest by the end.
    assert_allclose(scene.rigid_solver.get_dofs_velocity(), 0.0, atol=0.01)

    # The final box pose in its own initial frame is identical across every initial yaw.
    rel_pos = gu.transform_by_quat(box.get_pos() - torch.tensor(BOX_POS, device=gs.device), gu.inv_quat(box_quat))
    rel_yaw = gu.quat_to_xyz(gu.transform_quat_by_quat(box.get_quat(), gu.inv_quat(box_quat)), rpy=True)[:, 2]
    assert_allclose(rel_pos, rel_pos.mean(dim=0), atol=POSE_TOL)
    assert_allclose(rel_yaw, rel_yaw.mean(), atol=POSE_TOL)
