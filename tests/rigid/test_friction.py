import numpy as np
import pytest
import torch
import trimesh

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array

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
    # The noslip sweep needs enough passes to propagate the brace along the chain to match the elliptic cone's static
    # hold: short of that it leaves a residual creep that responds to the count without shrinking with it (the longest
    # chain creeps by a millimetre anywhere between 5 and 15 passes), and from 20 the creep collapses below the lateral
    # bound on every backend and chain. The count is the one where it converges rather than the fewest that holds the
    # bound, since it doubles as the recommendation to anyone reading it here.
    NOSLIP_ITERATIONS = 20

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

    boxes_pos_init = []
    for i in range(n_boxes + 1):
        box_size = (scale, scale * (1 + 0.3 * (2 - i)), scale * (1 + 0.3 * (2 - i)))
        if mesh_boxes:
            mesh_path = str(asset_tmp_path / f"static_friction_box_{scale}_{i}.obj")
            trimesh.creation.box(extents=box_size).export(mesh_path, file_type="obj")
            morph = gs.morphs.Mesh(
                file=mesh_path,
                pos=(i * (1 - 5e-4) * scale, 0, 0),
                fixed=(i == 0),
            )
        else:
            morph = gs.morphs.Box(
                size=box_size,
                pos=(i * (1 - 5e-4) * scale, 0, 0),
                fixed=(i == 0),
            )
        boxes_pos_init.append((i * scale, 0, 0))
        scene.add_entity(
            morph,
            material=gs.materials.Rigid(
                rho=200.0,
                friction=friction,
            ),
            vis_mode="collision",
            visualize_contact=True,
        )
        scene.add_entity(
            morph,
            material=gs.materials.Kinematic(),
        )

    boxes = scene.rigid_solver.entities
    floating_boxes = boxes[1:]
    contacts_link_a = torch.arange(n_boxes, device=gs.device).repeat_interleave(4)
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

    # FIXME: Adding pitch torque is necessary to stabilize the contacts for some reason.
    # This approach is not reliable for now due to intermittent collision detection failure.
    # for i, box in enumerate(floating_boxes):
    #     box.control_dofs_force(scale * sum(box.get_mass() for box in floating_boxes[i:]) * GRAVITY, dofs_idx_local=4)

    # FIXME: Position-based orientation control is necessary to stabilize the contacts for some reason.
    # Note that roll and yaw control is needed for some parametrization to pass.
    for box in floating_boxes:
        box.set_dofs_kp(1000.0 * total_mass, dofs_idx_local=slice(3, 6))
        box.set_dofs_kv(100.0 * total_mass, dofs_idx_local=slice(3, 6))
        box.control_dofs_position(0.0, dofs_idx_local=slice(3, 6))

    # Hold under sustained shear for 20 seconds
    for _ in range(2000):
        scene.step()
        # FIXME: The contact manifold is not stable
        # assert (rigid_solver.collider.get_contacts()["link_a"] == contacts_link_a).all()

    # The floating boxes stay static. Drift is measured per contact - each box against the one bracing it, the fixed
    # wall for the first - so a slip is charged to the contact it happens at instead of aggregating every contact
    # behind it down the chain. Each axis carries its own bound: the compression against the wall settles to an
    # absolute length whatever the scene scale, the lateral creep scales with the scene, and the slip down the
    # contact faces scales with an absolute floor on top. CG holds the noslip chain an order looser than Newton,
    # which is the baseline its configs document.
    boxes_pos_ref = torch.as_tensor(boxes_pos_init, dtype=gs.tc_float, device=gs.device)
    drift = torch.stack([box.get_pos() for box in boxes]) - boxes_pos_ref
    drift = torch.diff(drift, dim=0)
    if mode == "noslip":
        atol_x = 1e-2 if solver == gs.constraint_solver.Newton else 1e-3
        atol_y = (5e-4 if solver == gs.constraint_solver.Newton else 2e-4) * scale
        atol_z = (5e-3 if solver == gs.constraint_solver.Newton else 1e-2) * scale + 1e-3
    else:
        atol_x = 5e-3
        atol_y = (1e-5 if solver == gs.constraint_solver.Newton else 2e-5) * scale + 1e-6
        atol_z = 1e-2 * scale + 1e-3
    assert_allclose(drift[..., 0], 0.0, atol=atol_x)
    assert_allclose(drift[..., 1], 0.0, atol=atol_y)
    assert_allclose(drift[..., 2], 0.0, atol=atol_z)

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


# The mesh box at the larger scale is the demanding case, so it is the required one: it reaches the support of a face
# normal through the sampled table rather than analytically, and it is where the measured spreads sit closest to their
# bounds. Holding it holds the primitive and the unit scale with it. The two ends of the sweep are four orders of
# magnitude apart, which is what holds every tolerance the step goes through to being relative to the quantity it
# bounds: an absolute one is invisible at unit scale and takes over the answer at one end or the other.
@pytest.mark.parametrize(
    "is_box_mesh, scale",
    [
        # FIXME: A mesh geom reads its support from a table sampled on a spherical grid, and a direction along a
        # geometric feature ties several vertices there, so which one anchors a contact follows the rounding of the
        # direction and the reported manifold differs between a scene and a rotated copy of it. Restore these cases
        # with the support-table canonicalization follow-up, which resolves every such tie to a representative that
        # is a property of the mesh alone.
        # (True, 0.01),
        # (True, 0.02),
        # (True, 0.05),
        # (True, 0.1),
        # (True, 1.0),
        # pytest.param(True, 100.0, marks=pytest.mark.required),
        pytest.param(False, 1.0, marks=pytest.mark.required),
    ],
)
@pytest.mark.parametrize("contact_resolution", [gs.contact_resolution.convex, gs.contact_resolution.signorini])
def test_elliptic_cone_push_isotropy(contact_resolution, is_box_mesh, scale, precision, show_viewer, tol):
    N_ENVS = 8
    FRICTION = 0.5
    is_signorini = contact_resolution == gs.contact_resolution.signorini
    # Every length below is quoted at unit scale and multiplied by it, and gravity with it. That leaves the motion
    # geometrically similar at any scale, over the same timestep and the same number of steps, so one scene serves the
    # whole sweep and each bound only takes the power of the scale its own quantity carries. What the sweep holds is
    # that the tolerances contact detection compares against are relative to the pair of geoms it is given: it spans
    # two orders of magnitude of geometry, and eight of contact force.
    GRAVITY = 9.81 * scale
    BOX_POS = (0.0, 0.0, 0.02 * scale)
    BOX_SIZE = (0.1 * scale, 0.2 * scale, 0.04 * scale)
    # The pillar pushes the box below its centre of mass so the box slides rather than tipping: pushing level with it
    # leaves the box on the verge of lifting a leading corner, where each env's own rounding decides where it settles.
    PILLAR_HEIGHT = 0.04 * scale
    PILLAR_RADIUS = 0.0316 * scale
    # Pusher path in the box's local frame; the shared +y offset gives the push a lever arm that spins the box. The
    # height places the pillar rather than driving it: its plane contact carrying its weight holds it there.
    PUSH_START_LOCAL = (-0.15 * scale, 0.03 * scale, 0.5 * PILLAR_HEIGHT)
    PUSH_END_LOCAL = (0.02 * scale, 0.03 * scale, 0.5 * PILLAR_HEIGHT)
    # Each bound is a small factor over the worst spread measured across every backend, both array layouts and debug on
    # and off, calibrated per precision: a length holds ten times tighter in double, a direction two times tighter in
    # single, so one shared factor would leave the sharper precision unguarded; anything that is not rounding exceeds
    # these by orders of magnitude. Rounding being relative, each bound carries its quantity's power of the scale, none
    # for a direction or an angular rate with time held fixed; a bound shared by several quantities covers the largest.
    LENGTH_TOL = 0.5 * tol * scale
    DIRECTION_TOL = (2.0 if gs.np_float == np.float64 else 1.0) * tol
    LIN_VEL_TOL = (2.0 if gs.np_float == np.float64 else 5.0) * tol * scale
    # The angular rate is where the constraint solve leaves its residual: two orientations each converge to within the
    # solver tolerance, and how far apart that leaves the rate measures several tens of it. In single precision the
    # coupled cone's solve runs at its rounding floor through the whole sliding phase and the worst backend's sweep
    # spreads the rate near a hundred times the solve's tolerance, so the single-precision velocity bounds are quoted
    # over that backend with headroom of two; double precision holds orders of magnitude tighter.
    ANG_VEL_TOL = (30.0 if gs.np_float == np.float64 else 200.0) * tol
    # Force carries the stiffness gain on top, and coplanar contacts of one pair share the load with a null space the
    # solve may resolve anywhere inside, so the bound covers the split. A mass times an acceleration takes three powers
    # of the scale from the mass and one from gravity; a torque one more from its lever arm.
    FORCE_TOL = 500.0 * tol * scale**4
    TORQUE_TOL = 50.0 * tol * scale**5
    # How far either body may sit from the plane resting under its own weight, how still it must end, and how far the
    # pusher may sit from the height its stance gives it and the yaw it was commanded to. Resolving the penetration
    # depth on its own holds a body on the plane whether or not it slides, so each bound is met to a fraction of a
    # thousandth; coupling it to the friction solve charges a sliding contact's tangential residual to the normal
    # direction instead, throwing the body off the plane to fall back, and both bodies slide for the whole push, so
    # where in that cycle the run ends costs the second set an order or two.
    GROUND_TOL = (5e-4 if is_signorini else 2e-2) * scale
    REST_LIN_VEL_TOL = (1e-3 if is_signorini else 2e-3) * scale
    REST_ANG_VEL_TOL = 5e-3 if is_signorini else 5e-2
    REST_LENGTH_TOL = (2e-4 if is_signorini else 2e-2) * scale
    REST_TILT_TOL = 1e-3 if is_signorini else 5e-2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.005,
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=gs.friction_cone.elliptic,
            contact_resolution=contact_resolution,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.7 * scale, 0.7 * scale, 0.8 * scale),
            camera_lookat=(0.0, 0.0, 0.05 * scale),
        ),
        show_viewer=show_viewer,
    )
    ground = scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
    )
    # The box is swept over both geometries the push can act through, because they reach the support of a face normal -
    # the direction whose support vertices tie - by different means: the primitive resolves it analytically from the
    # sign of each component, the mesh reads the sampled table. Either may pick a different one of the tied vertices
    # for a rotated copy of the same scene, and only sweeping both holds each to the same manifold.
    box_morph = (
        gs.morphs.MeshSet(
            files=(trimesh.creation.box(extents=BOX_SIZE),),
            pos=BOX_POS,
        )
        if is_box_mesh
        else gs.morphs.Box(
            pos=BOX_POS,
            size=BOX_SIZE,
        )
    )
    box = scene.add_entity(
        box_morph,
        material=gs.materials.Rigid(
            friction=FRICTION,
        ),
        visualize_contact=True,
        vis_mode="collision",
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
        surface=gs.surfaces.Default(
            smooth=False,
        ),
        visualize_contact=True,
        vis_mode="collision",
    )
    scene.build(n_envs=N_ENVS, env_spacing=(0.3 * scale, 0.3 * scale))

    yaw = 2.0 * torch.pi * torch.arange(N_ENVS, device=gs.device) / N_ENVS
    yaw_euler = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), yaw), dim=1)
    box_quat = gu.xyz_to_quat(yaw_euler, rpy=True)
    box.set_quat(box_quat)

    # Rotate the local pusher path into each env's world frame by the box yaw, and PD-control the pusher's full pose.
    push_start = gu.transform_by_quat(torch.tensor(PUSH_START_LOCAL, device=gs.device).repeat(N_ENVS, 1), box_quat)
    push_end = gu.transform_by_quat(torch.tensor(PUSH_END_LOCAL, device=gs.device).repeat(N_ENVS, 1), box_quat)
    pusher.set_pos(push_start)
    pusher.set_quat(box_quat)
    # Quoted per unit mass, the linear gains are accelerations per unit error, and holding those fixed is what keeps the
    # pusher tracking the same path at any scale. The angular ones act on the inertia instead, which grows two powers of
    # length faster than the mass they are quoted against, so they carry that difference.
    gains_scale = pusher.get_mass() * torch.tensor((1.0, 1.0, 0.0, 0.0, 0.0, scale**2), device=gs.device)
    pusher.set_dofs_kp(gains_scale * torch.tensor((2000.0, 2000.0, 0.0, 0.0, 0.0, 5000.0), device=gs.device))
    pusher.set_dofs_kv(gains_scale * torch.tensor((200.0, 200.0, 0.0, 0.0, 0.0, 500.0), device=gs.device))

    # Let the box resolve its initial ground contact before the push starts, so the two transients do not couple.
    scene.step()

    # Only the horizontal path and the yaw are driven. Holding the height would carry the plane contact whatever the
    # solve does with it, and holding roll and pitch would keep the stance flat however the contacts load it, which is
    # what the manifold below is there to check.
    pusher.control_dofs_position(push_end[:, :2], dofs_idx_local=[0, 1])
    # The pillar is held at its env's own yaw, so every env simulates one rigidly rotated copy of the same scene.
    pusher.control_dofs_position(yaw_euler[:, 2:], dofs_idx_local=[5])
    # Every quantity is compared at every step rather than only the pose the box settles in: a difference that appears
    # once is amplified by the steps that follow, so the end state cannot say which quantity broke first.
    box_quat_inv = gu.inv_quat(box_quat)
    expected_manifold = {
        (ground.geoms[0].idx, box.geoms[0].idx): 4,
        (ground.geoms[0].idx, pusher.geoms[0].idx): 3,
        (box.geoms[0].idx, pusher.geoms[0].idx): 2,
    }
    is_manifold_complete = False
    N_STEPS = 160
    for i_step in range(N_STEPS):
        scene.step()

        contacts = scene.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=True)
        counts = [len(positions) for positions in contacts["position"]]
        assert counts == counts[:1] * N_ENVS, f"contact count differs across envs at step {i_step}: {counts}"
        # Which contacts the scene has, once it has them all, not merely how many: each geom pair stands on the
        # corners of one of its geoms, so the per-pair count is a property of the shapes alone and holds for the rest
        # of the run. Only the resolution holding a sliding body on the plane earns this (see the rest-state bounds);
        # in flight there is no manifold to hold.
        if is_signorini:
            pair_counts = [
                {
                    pair: int(((contacts["geom_a"][i_env] == pair[0]) & (contacts["geom_b"][i_env] == pair[1])).sum())
                    for pair in expected_manifold
                }
                for i_env in range(N_ENVS)
            ]
            is_manifold_complete = is_manifold_complete or pair_counts[0] == expected_manifold
            if is_manifold_complete:
                for i_env, pair_count in enumerate(pair_counts):
                    assert pair_count == expected_manifold, (
                        f"the manifold lost a contact at step {i_step}, env {i_env}: "
                        f"{pair_count} against {expected_manifold}"
                    )
        # The box leaves the ground for a step at the smallest scale, which leaves the whole scene with nothing
        # to compare while the pose and the velocities below still say the envs agree.
        if counts[0]:
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
                blocks.append(torch.cat([column.to(gs.tc_float) for column in columns], dim=1))
            paired = torch.stack(blocks)
            # Compared row by row in the order the collider reports them: the contacts of every geom pair are ordered
            # by their position in one of the pair's own frames, so a scene and any rotated copy of it report the same
            # contacts in the same order. That the rows line up at all is itself part of what is checked here.
            assert_equal(
                paired[:, :, 10:], paired[0, :, 10:], err_msg=f"contact pairs or their order differ at step {i_step}"
            )
            for key, columns, atol in (
                ("position", slice(0, 3), LENGTH_TOL),
                ("normal", slice(3, 6), DIRECTION_TOL),
                ("penetration", slice(9, 10), LENGTH_TOL),
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
                net_force = paired[:, rows, 6:9].sum(dim=1)
                net_torque = torch.cross(paired[:, rows, 0:3], paired[:, rows, 6:9], dim=-1).sum(dim=1)
                assert_allclose(net_force, net_force[0], atol=FORCE_TOL, err_msg=f"net force differs at step {i_step}")
                assert_allclose(
                    net_torque, net_torque[0], atol=TORQUE_TOL, err_msg=f"net torque differs at step {i_step}"
                )
        # Dropping through the plane, or leaving it altogether, would leave eight envs identically wrong with every
        # comparison above still green.
        for entity in (box, pusher):
            assert_allclose(entity.get_AABB()[:, 0, 2], 0.0, atol=GROUND_TOL, err_msg=f"off the ground, step {i_step}")

        # Every free joint carries a linear half in world axes and an angular half in its own body frame, so only
        # the former is de-rotated; the latter is already invariant under a rotation of the whole scene.
        velocity = scene.rigid_solver.get_dofs_velocity().reshape((N_ENVS, -1, 6))
        quat = box_quat_inv[:, None].expand(N_ENVS, velocity.shape[1], 4)
        linear = gu.transform_by_quat(velocity[:, :, :3], quat)
        assert_allclose(linear, linear[0], atol=LIN_VEL_TOL, err_msg=f"linear velocity differs at step {i_step}")
        angular = velocity[:, :, 3:]
        assert_allclose(angular, angular[0], atol=ANG_VEL_TOL, err_msg=f"angular velocity differs at step {i_step}")

    # The box and pusher come to rest on the plane by the end, the pusher at the height its stance gives it and flat
    # on the yaw it was commanded to: sinking would bury it in the plane, tilting would lift a corner of its stance.
    # The per-resolution slack is the rest-state bounds' (see above); under the coupled resolution both bodies are
    # still in flight whenever the run ends.
    velocity = scene.rigid_solver.get_dofs_velocity().reshape((N_ENVS, -1, 6))
    assert_allclose(velocity[:, :, :3], 0.0, atol=REST_LIN_VEL_TOL)
    assert_allclose(velocity[:, :, 3:], 0.0, atol=REST_ANG_VEL_TOL)
    assert_allclose(pusher.get_pos()[:, 2], 0.5 * PILLAR_HEIGHT, atol=REST_LENGTH_TOL)
    assert_allclose(
        gu.transform_quat_by_quat(pusher.get_quat(), gu.inv_quat(box_quat)), (1.0, 0.0, 0.0, 0.0), atol=REST_TILT_TOL
    )

    # The final box pose in its own initial frame is identical across every initial yaw.
    rel_pos = gu.transform_by_quat(box.get_pos() - torch.tensor(BOX_POS, device=gs.device), gu.inv_quat(box_quat))
    rel_yaw = gu.quat_to_xyz(gu.transform_quat_by_quat(box.get_quat(), gu.inv_quat(box_quat)), rpy=True)[:, 2]
    # A push that moved the box hardly at all would satisfy the comparison below without exercising anything.
    assert (rel_pos[:, 0] > 0.01 * scale).all() and (rel_yaw.abs() > 0.05).all()
    assert_allclose(rel_pos, rel_pos.mean(dim=0), atol=LENGTH_TOL)
    assert_allclose(rel_yaw, rel_yaw.mean(), atol=DIRECTION_TOL)


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
