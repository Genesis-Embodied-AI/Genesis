import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu

from ..utils import (
    assert_allclose,
    assert_equal,
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
    gs_qvel = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()
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
    velocity_zero = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]

    robot.set_dofs_velocity(initial_velocity)
    robot.set_dofs_frictionloss(np.array([1.0, 0.0]))
    frictionloss = robot.get_dofs_frictionloss()
    assert_allclose(frictionloss, np.array([1.0, 0.0]), atol=tol)
    for _ in range(10):
        gs_sim.step()
    velocity_high = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]

    np.testing.assert_array_less(velocity_high[0], velocity_zero[0])
    np.testing.assert_array_less(velocity_high[1], velocity_zero[1])

    robot.set_dofs_velocity(initial_velocity)
    robot.set_dofs_frictionloss(np.array([0.5]), dofs_idx_local=[0])
    frictionloss = robot.get_dofs_frictionloss(dofs_idx_local=[0])
    assert_allclose(frictionloss, np.array([0.5]), atol=tol)
    for _ in range(10):
        gs_sim.step()
    velocity_medium = gs_sim.rigid_solver.dyn_state.dofs.vel.to_numpy()[:, 0]

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
        assert rigid_solver.rigid_config.enable_cooperative_constraint_kernels == (6 * n_boxes >= 16)
        assert rigid_solver.rigid_config.prefer_decomposed_solver == (6 * n_boxes >= 16)

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
    # Coulomb caps the friction opposing the 1.5x push at mu * N, leaving exactly half the weight to accelerate the
    # box. A contact whose normal force answers the tangential demand instead brakes harder than that and lands
    # short, which the spread below cannot see since it scales with its own mean.
    assert_allclose(accel, 0.5 * MU * (-GRAVITY), rtol=0.01)
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
@pytest.mark.parametrize("friction_cone", [gs.friction_cone.pyramidal, gs.friction_cone.elliptic])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_torsional_friction_spin_down_rate(friction_cone, n_envs, show_viewer):
    # I * dw/dt = -friction_torsional * m * g with I = 2/5 m r^2: the saturated spin-down rate is
    # friction_torsional * g / (0.4 r^2), mass-independent. The elliptic cone tracks this exact Coulomb bound once
    # fully slipping; the pyramidal cone's regularized friction decays below it, so only the rate ordering is
    # asserted. The plane's zero coefficient is inert under the pair-by-maximum rule, and a zero-coefficient sphere
    # keeps the weight-only contact force of a torsional-free scene.
    GRAVITY = 9.81
    DT = 0.01
    RADIUS = 0.1
    W0 = 3.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=friction_cone,
            enable_torsional_friction=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.75, -1.8, 0.8),
            camera_lookat=(0.75, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction_torsional=0.0,
        ),
    )
    spheres_friction_torsional = (0.0, 0.0001, 0.002, 0.005)
    spheres = []
    for i_s, friction_torsional in enumerate(spheres_friction_torsional):
        spheres.append(
            scene.add_entity(
                gs.morphs.Sphere(
                    radius=RADIUS,
                    pos=(0.5 * i_s, 0.0, RADIUS),
                ),
                material=gs.materials.Rigid(
                    friction_torsional=friction_torsional,
                ),
            )
        )
    scene.build(n_envs=n_envs)

    for _ in range(10):
        scene.step()
    for sphere in spheres:
        sphere.set_dofs_velocity([0.0, 0.0, 0.0, 0.0, 0.0, W0])
    # Let the contact reference dynamics settle into the fully slipping regime before measuring the rate.
    for _ in range(5):
        scene.step()
    w_start = [sphere.get_dofs_velocity()[..., 5] for sphere in spheres]
    for _ in range(10):
        scene.step()
    spin_downs = []
    for sphere, friction_torsional, w_0 in zip(spheres, spheres_friction_torsional, w_start):
        spin_down = w_0 - sphere.get_dofs_velocity()[..., 5]
        spin_downs.append(spin_down)
        if friction_cone == gs.friction_cone.elliptic or friction_torsional == 0.0:
            spin_down_rate = friction_torsional * GRAVITY / (0.4 * RADIUS**2)
            assert_allclose(spin_down, spin_down_rate * 10 * DT, rtol=0.05, atol=1e-3)
    for spin_down_slow, spin_down_fast in zip(spin_downs[1:], spin_downs[2:]):
        assert (spin_down_slow < spin_down_fast).all()
    assert_allclose(
        torch.linalg.norm(spheres[0].get_links_net_contact_force(), dim=-1).sum(dim=-1),
        spheres[0].get_mass() * GRAVITY,
        rtol=0.01,
    )

    # A runtime coefficient update takes effect immediately: the zero-coefficient sphere, still spinning at W0, now
    # decays exactly like the sphere that carried the same coefficient from the start.
    spheres[0].set_friction_torsional(spheres_friction_torsional[-1])
    for _ in range(5):
        scene.step()
    w_runtime_start = spheres[0].get_dofs_velocity()[..., 5]
    for _ in range(10):
        scene.step()
    assert_allclose(w_runtime_start - spheres[0].get_dofs_velocity()[..., 5], spin_downs[-1], rtol=0.05, atol=1e-3)


@pytest.mark.required
@pytest.mark.parametrize("friction_cone", [gs.friction_cone.pyramidal, gs.friction_cone.elliptic])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_rolling_friction_deceleration_rate(friction_cone, n_envs, show_viewer):
    # A sphere rolling without slipping decelerates only through the rolling resistance torque friction_rolling *
    # m * g: with the rolling constraint v = w * r and I = 2/5 m r^2, dv/dt = -(5/7) * friction_rolling * g / r,
    # mass-independent. The elliptic cone tracks this exact Coulomb bound; the pyramidal cone decays below it, so
    # only the rate ordering is asserted. The plane's zero coefficient is inert under the pair-by-maximum rule, and
    # a zero-coefficient sphere keeps rolling freely with the weight-only contact force of a rolling-free scene.
    GRAVITY = 9.81
    DT = 0.01
    RADIUS = 0.1
    V0 = 0.5

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=friction_cone,
            enable_torsional_friction=True,
            enable_rolling_friction=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.75, -1.8, 0.8),
            camera_lookat=(0.75, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction_torsional=0.0,
            friction_rolling=0.0,
        ),
    )
    spheres_friction_rolling = (0.0, 0.001, 0.004)
    spheres = []
    for i_s, friction_rolling in enumerate(spheres_friction_rolling):
        spheres.append(
            scene.add_entity(
                gs.morphs.Sphere(
                    radius=RADIUS,
                    pos=(0.5 * i_s, 0.0, RADIUS),
                ),
                material=gs.materials.Rigid(
                    friction_rolling=friction_rolling,
                ),
            )
        )
    scene.build(n_envs=n_envs)

    for _ in range(10):
        scene.step()
    for sphere in spheres:
        sphere.set_dofs_velocity([V0, 0.0, 0.0, 0.0, V0 / RADIUS, 0.0])
    # Let the contact settle into steady rolling before measuring the deceleration.
    for _ in range(5):
        scene.step()
    v_start = [sphere.get_dofs_velocity()[..., 0] for sphere in spheres]
    for _ in range(10):
        scene.step()
    slow_downs = []
    for sphere, friction_rolling, v_0 in zip(spheres, spheres_friction_rolling, v_start):
        slow_down = v_0 - sphere.get_dofs_velocity()[..., 0]
        slow_downs.append(slow_down)
        if friction_cone == gs.friction_cone.elliptic or friction_rolling == 0.0:
            deceleration = (5.0 / 7.0) * friction_rolling * GRAVITY / RADIUS
            assert_allclose(slow_down, deceleration * 10 * DT, rtol=0.05, atol=1e-3)
    for slow_down_slow, slow_down_fast in zip(slow_downs[:-1], slow_downs[1:]):
        assert (slow_down_slow < slow_down_fast).all()
    assert_allclose(
        torch.linalg.norm(spheres[0].get_links_net_contact_force(), dim=-1).sum(dim=-1),
        spheres[0].get_mass() * GRAVITY,
        rtol=0.01,
    )


@pytest.mark.required
@pytest.mark.parametrize("contact_resolution", [gs.contact_resolution.convex, gs.contact_resolution.signorini])
def test_elliptic_cone_push_isotropy(contact_resolution, show_viewer, tol):
    N_ENVS = 8
    FRICTION = 0.5
    BOX_POS = (0.0, 0.0, 0.05)
    BOX_SIZE = (0.1, 0.2, 0.1)
    # The pillar pushes the box below its centre of mass so the box slides rather than tipping: pushing level with it
    # leaves the box on the verge of lifting a leading corner, where each env's own rounding decides where it settles.
    PILLAR_HEIGHT = 0.04
    PILLAR_RADIUS = 0.0316
    # Pusher path in the box's local frame; the shared +y offset gives the push a lever arm that spins the box.
    PUSH_START_LOCAL = (-0.15, 0.03, 0.5 * PILLAR_HEIGHT)
    PUSH_END_LOCAL = (0.02, 0.03, 0.5 * PILLAR_HEIGHT)
    # Single precision sets these: one ulp of a coordinate this far off the rotation axis is 1.5e-08, and the contact
    # stiffness carries it into a milli-newton of force, so each bound sits a few times over what rounding alone
    # reaches. Double precision agrees to 1e-15 throughout, well inside them, and anything that is not rounding - a
    # detection branch reading the world orientation, say - exceeds them by orders of magnitude at either precision.
    CONTACT_TOL = tol
    VEL_TOL = 10.0 * tol
    # Force also carries the stiffness gain, and coplanar contacts of one pair share the load with a null space the
    # solve may resolve anywhere inside.
    FORCE_TOL = 1000.0 * tol
    # How far either body may be from the plane: being pressed into it, or lifted by the bouncier contact resolution.
    GROUND_TOL = 5e-3

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            contact_resolution=contact_resolution,
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
            size=BOX_SIZE,
        ),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    # The pusher is a pillar with a triangular cross-section, so it stands on three corners fixed in its own frame. A
    # circular cross-section rests on a degenerate manifold instead, one whose sampled points follow the world frame,
    # and that alone makes the push anisotropic before the box is ever touched.
    pusher = scene.add_entity(
        gs.morphs.MeshSet(
            files=(trimesh.creation.cylinder(radius=PILLAR_RADIUS, height=PILLAR_HEIGHT, sections=3),),
            pos=PUSH_START_LOCAL,
        ),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    scene.build(n_envs=N_ENVS)

    yaw = 2.0 * torch.pi * torch.arange(N_ENVS, device=gs.device) / N_ENVS
    yaw_euler = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), yaw), dim=1)
    box_quat = gu.xyz_to_quat(yaw_euler, rpy=True)
    box.set_quat(box_quat)

    # Rotate the local pusher path into each env's world frame by the box yaw, and PD-control the pusher's full pose.
    push_start = gu.transform_by_quat(torch.tensor(PUSH_START_LOCAL, device=gs.device).repeat(N_ENVS, 1), box_quat)
    push_end = gu.transform_by_quat(torch.tensor(PUSH_END_LOCAL, device=gs.device).repeat(N_ENVS, 1), box_quat)
    pusher.set_pos(push_start)
    pusher.set_quat(box_quat)
    pusher.set_dofs_kp(
        pusher.get_mass() * torch.tensor((2000.0, 2000.0, 2000.0, 5000.0, 5000.0, 5000.0), device=gs.device)
    )
    pusher.set_dofs_kv(pusher.get_mass() * torch.tensor((200.0, 200.0, 200.0, 500.0, 500.0, 500.0), device=gs.device))

    # Let the box resolve its initial ground contact before the push starts, so the two transients do not couple.
    scene.step()

    # Drive the pusher forward through the box while holding its height and orientation.
    pusher.control_dofs_position(push_end, dofs_idx_local=[0, 1, 2])
    # The pillar is held at its env's own yaw, so every env simulates one rigidly rotated copy of the same scene.
    pusher.control_dofs_position(yaw_euler, dofs_idx_local=[3, 4, 5])
    # Every quantity is compared at every step rather than only the pose the box settles in: a difference that appears
    # once is amplified by the steps that follow, so the end state cannot say which quantity broke first.
    box_quat_inv = gu.inv_quat(box_quat)
    for i_step in range(160):
        scene.step()

        contacts = scene.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=True)
        counts = [len(positions) for positions in contacts["position"]]
        assert counts == counts[:1] * N_ENVS, f"contact count differs across envs at step {i_step}: {counts}"
        blocks = []
        for i_env in range(N_ENVS):
            quat = box_quat_inv[i_env].expand(counts[i_env], 4)
            columns = (
                gu.transform_by_quat(contacts["position"][i_env], quat),
                gu.transform_by_quat(contacts["normal"][i_env], quat),
                gu.transform_by_quat(contacts["force"][i_env], quat),
                contacts["penetration"][i_env][:, None],
                contacts["geom_a"][i_env][:, None],
                contacts["geom_b"][i_env][:, None],
            )
            block = torch.cat([column.to(gs.tc_float) for column in columns], dim=1)
            if i_env == 0:
                blocks.append(block)
                continue
            # Matched to env 0 by nearest de-rotated position within the same geom pair, which is unambiguous because
            # contacts sit millimetres apart while they agree to a fraction of that. Ordering by coordinate instead
            # lets rounding swap two contacts that share one.
            is_same_pair = (block[:, None, 10:12] == blocks[0][None, :, 10:12]).all(dim=-1)
            distance = torch.linalg.norm(block[:, None, :3] - blocks[0][None, :, :3], dim=-1)
            blocks.append(block[torch.where(is_same_pair, distance, torch.inf).argmin(dim=0)])
        paired = torch.stack(blocks)
        assert_equal(paired[:, :, 10:], paired[0, :, 10:], err_msg=f"geom pairs differ at step {i_step}")
        for key, columns, atol in (
            ("position", slice(0, 3), CONTACT_TOL),
            ("normal", slice(3, 6), CONTACT_TOL),
            ("penetration", slice(9, 10), CONTACT_TOL),
            ("force", slice(6, 9), FORCE_TOL),
        ):
            values = paired[:, :, columns]
            assert_allclose(values, values[0], atol=atol, err_msg=f"contact {key} differs at step {i_step}")
        # Net wrench per geom pair about the world origin. Coplanar contacts of one pair share the load with a null
        # space the solve may resolve anywhere inside, so the individual forces are not determined while their
        # resultant is: comparing both says which of the two any difference lives in.
        for pair in ((0, 1), (0, 2), (1, 2)):
            rows = (paired[0, :, 10] == pair[0]) & (paired[0, :, 11] == pair[1])
            if not rows.any():
                continue
            force = paired[:, rows, 6:9]
            torque = torch.cross(paired[:, rows, 0:3], force, dim=-1)
            wrench = torch.cat((force.sum(dim=1), torque.sum(dim=1)), dim=1)
            assert_allclose(wrench, wrench[0], atol=FORCE_TOL, err_msg=f"net wrench differs at step {i_step}")
        # Dropping through the plane, or leaving it altogether, would leave eight envs identically wrong with every
        # comparison above still green.
        for entity in (box, pusher):
            assert_allclose(entity.get_AABB()[:, 0, 2], 0.0, atol=GROUND_TOL, err_msg=f"off the ground, step {i_step}")

        # Every free joint carries a linear half in world axes and an angular half in its own body frame, so only
        # the former is de-rotated; the latter is already invariant under a rotation of the whole scene.
        velocity = scene.rigid_solver.get_dofs_velocity().reshape((N_ENVS, -1, 6))
        quat = box_quat_inv[:, None].expand(N_ENVS, velocity.shape[1], 4)
        linear = gu.transform_by_quat(velocity[:, :, :3], quat)
        assert_allclose(linear, linear[0], atol=VEL_TOL, err_msg=f"linear velocity differs at step {i_step}")
        angular = velocity[:, :, 3:]
        assert_allclose(angular, angular[0], atol=VEL_TOL, err_msg=f"angular velocity differs at step {i_step}")

    # The box and pusher settle at rest by the end.
    assert_allclose(scene.rigid_solver.get_dofs_velocity(), 0.0, atol=0.01)

    # The pusher holds the height and orientation it was commanded to, so the push it applies is the one intended:
    # sinking would bury it in the plane and tilting would lift a corner of its stance clear of it.
    assert_allclose(pusher.get_pos()[:, 2], 0.5 * PILLAR_HEIGHT, atol=1e-3)
    assert_allclose(
        gu.transform_quat_by_quat(pusher.get_quat(), gu.inv_quat(box_quat)), (1.0, 0.0, 0.0, 0.0), atol=1e-3
    )

    # The final box pose in its own initial frame is identical across every initial yaw.
    rel_pos = gu.transform_by_quat(box.get_pos() - torch.tensor(BOX_POS, device=gs.device), gu.inv_quat(box_quat))
    rel_yaw = gu.quat_to_xyz(gu.transform_quat_by_quat(box.get_quat(), gu.inv_quat(box_quat)), rpy=True)[:, 2]
    # A push that moved the box hardly at all would satisfy the comparison below without exercising anything.
    assert (rel_pos[:, 0] > 0.01).all() and (rel_yaw.abs() > 0.05).all()
    assert_allclose(rel_pos, rel_pos.mean(dim=0), atol=CONTACT_TOL)
    assert_allclose(rel_yaw, rel_yaw.mean(), atol=CONTACT_TOL)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_kinetic_friction(n_envs, show_viewer):
    GRAVITY = 9.81
    SIZE = 0.1
    # A cube resting flat tips once friction can torque it over its leading edge, from mu = width / (2 * com_height).
    TIP_FRICTION = SIZE / (2.0 * 0.5 * SIZE)
    # Friction coefficient and launch speed per box, spanning both sides of that threshold. 0.01 is the smallest
    # coefficient the contact model applies, and the plane wears it so the pairwise max leaves each box its own.
    BOXES = ((0.01, 6.0), (0.25, 6.0), (0.5, 2.0), (0.5, 6.0), (0.5, 20.0), (0.9, 6.0), (1.2, 6.0), (2.0, 6.0))
    N_SETTLE = 40
    N_SLIDE = 40

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.5, 0.6),
            camera_lookat=(1.0, 0.3, 0.05),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction=0.01,
        ),
    )
    # Spaced far enough apart never to interact, so each box is its own contact island.
    boxes = [
        scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 4.0 * SIZE * i_box, 0.5 * SIZE),
                size=(SIZE, SIZE, SIZE),
            ),
            material=gs.materials.Rigid(
                friction=friction,
            ),
        )
        for i_box, (friction, _) in enumerate(BOXES)
    ]
    scene.build(n_envs=n_envs)

    for _ in range(N_SETTLE):
        scene.step()

    height_0 = torch.stack([box.get_pos()[..., 2] for box in boxes])
    for box, (_, speed) in zip(boxes, BOXES):
        velocity = box.get_dofs_velocity()
        velocity[..., 0] = speed
        box.set_dofs_velocity(velocity)
    speed_0 = torch.stack([box.get_dofs_velocity()[..., 0] for box in boxes])

    height_min, height_max, tilt_max = height_0, height_0, torch.zeros_like(height_0)
    for _ in range(N_SLIDE):
        scene.step()
        height = torch.stack([box.get_pos()[..., 2] for box in boxes])
        height_min = torch.minimum(height_min, height)
        height_max = torch.maximum(height_max, height)
        # Angle between the body up-axis and world up, from the quaternion's rotation of e_z.
        quat = torch.stack([box.get_quat() for box in boxes])
        up_z = 1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)
        tilt_max = torch.maximum(tilt_max, torch.rad2deg(torch.arccos(up_z.clamp(-1.0, 1.0))))
    speed_1 = torch.stack([box.get_dofs_velocity()[..., 0] for box in boxes])

    friction = torch.tensor([friction for friction, _ in BOXES], device=gs.device).reshape(-1, *(1,) * (n_envs > 0))
    is_sliding = (friction < TIP_FRICTION).expand_as(height_0)

    # Below the threshold the box slides flat; above it, friction torques it over its leading edge.
    assert_allclose(tilt_max[is_sliding], 0.0, atol=0.1)
    assert (tilt_max[~is_sliding] > 45.0).all()

    # Penetration answers to the load alone, so a sliding box sits exactly as deep as a resting one. Every box here
    # carries the same weight, so holding all of them to their common resting height pins the penetration against
    # both the friction coefficient and the sliding speed at once. The bound is set by the box closest to the
    # tipping threshold, whose contact force genuinely leans toward its leading edge and lifts the centre of mass.
    assert_allclose(height_min[is_sliding], height_0[is_sliding], atol=5e-7)
    assert_allclose(height_max[is_sliding], height_0[is_sliding], atol=5e-7)

    # The height above measures this through the geometry; the contact force states it directly. A sliding contact
    # carries the weight and nothing more, however much tangential force the slide is asking of it.
    normal_force = torch.stack([box.get_links_net_contact_force().sum(dim=-2)[..., 2] for box in boxes])
    assert_allclose(normal_force[is_sliding], boxes[0].get_mass() * GRAVITY, rtol=0.01)

    # Each box decelerates at its own mu * g, independently of how fast it was launched.
    deceleration = (speed_0 - speed_1) / (N_SLIDE * scene.sim.dt)
    assert_allclose(deceleration[is_sliding], (friction * GRAVITY).expand_as(height_0)[is_sliding], rtol=0.01)
