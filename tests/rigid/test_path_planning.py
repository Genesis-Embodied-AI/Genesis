import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import qd_to_torch

from ..utils import assert_allclose, assert_equal


def _max_true_penetration(scene, entity, qpos_waypoints, obstacles, show_viewer):
    # Ground truth through the real collider, independent of the planner's sphere proxy. Mutates the scene, so
    # callers restore the configuration afterward. Doubles as the interactive replay of the returned path, since
    # planning tests never step the scene.
    collider_state = scene.rigid_solver.collider._collider_state
    obstacle_geoms = {geom.idx for obstacle in obstacles for link in obstacle.links for geom in link.geoms}
    max_penetration = 0.0
    for waypoint in qpos_waypoints:
        entity.set_qpos(waypoint, zero_velocity=False)
        if show_viewer:
            scene.visualizer.update()
        scene.rigid_solver.collider.detection()
        n_contacts = qd_to_torch(collider_state.n_contacts)
        geom_a = qd_to_torch(collider_state.contact_data.geom_a)
        geom_b = qd_to_torch(collider_state.contact_data.geom_b)
        penetration = qd_to_torch(collider_state.contact_data.penetration)
        for i_b in range(max(scene.n_envs, 1)):
            for i_c in range(int(n_contacts[i_b])):
                if int(geom_a[i_c, i_b]) in obstacle_geoms or int(geom_b[i_c, i_b]) in obstacle_geoms:
                    max_penetration = max(max_penetration, float(penetration[i_c, i_b]))
    return max_penetration


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_plan_to_qpos_goal_avoids_obstacles(n_envs, show_viewer, tol):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 1.5, 1.5),
            camera_lookat=(0.3, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    pillar = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 1.2),
            pos=(0.45, 0.0, 0.6),
            fixed=True,
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=n_envs)
    franka.set_dofs_vel_limit(2.0)

    qpos_goal = np.array([0.9, 0.6, 0.0, -1.4, 0.0, 2.0, 0.8, 0.02, 0.02])
    qpos_before = franka.get_qpos().clone()
    links_pos_before = scene.rigid_solver.get_links_pos([link.idx for link in franka.links]).clone()

    # The straight joint-space interpolation to this goal sweeps through the pillar, so a certified plan proves
    # obstacle avoidance; waypoints are spaced exactly at the scene dt by default.
    path = franka.plan_path(qpos_goal, seed=11)
    assert path.is_valid.all()
    assert_allclose(path.dt, scene.dt, tol=tol)
    assert_allclose(path.qpos[0], qpos_before, tol=1e-5)
    assert_allclose(path.qpos[-1], torch.as_tensor(qpos_goal, dtype=gs.tc_float), tol=1e-4)

    # Planning is a pure query: the whole kinematic state is bit-identical afterward.
    assert_equal(franka.get_qpos(), qpos_before)
    assert_equal(scene.rigid_solver.get_links_pos([link.idx for link in franka.links]), links_pos_before)

    # Same seed reproduces the exact plan; a different seed explores differently.
    path_same = franka.plan_path(qpos_goal, seed=11)
    assert_equal(path_same.qpos, path.qpos)
    assert_equal(path_same.dt, path.dt)
    path_other = franka.plan_path(qpos_goal, seed=12)
    assert not torch.equal(path_other.qpos, path.qpos)

    # Retiming respects the model velocity limit, and velocities are consistent with the waypoint spacing.
    assert (path.dofs_vel.abs() <= 2.0 + 1e-5).all()
    dt_bc = path.dt if n_envs == 0 else path.dt[:, None]
    fd_vel = (path.qpos[2:] - path.qpos[:-2]) / (2.0 * dt_bc)
    assert_allclose(fd_vel, path.dofs_vel[1:-1], atol=0.1)
    assert_allclose(path.dofs_vel[0], 0.0, tol=tol)
    assert_allclose(path.dofs_vel[-1], 0.0, tol=tol)

    # A fixed waypoint count is honored, with the true per-env spacing returned.
    path_fixed = franka.plan_path(qpos_goal, num_waypoints=50, seed=11)
    assert path_fixed.qpos.shape[0] == 50
    assert (path_fixed.dt > 0).all()

    # ignore_collision yields the retimed straight-line interpolation.
    path_line = franka.plan_path(qpos_goal, ignore_collision=True)
    assert path_line.is_valid.all()
    assert_allclose(path_line.qpos[-1], torch.as_tensor(qpos_goal, dtype=gs.tc_float), tol=1e-4)

    # An unreachable goal (below the floor) reports failure instead of raising, and leaves the state untouched.
    qpos_bad = np.array([0.0, 1.2, 0.0, -0.4, 0.0, 3.4, 0.8, 0.02, 0.02])
    path_bad = franka.plan_path(qpos_bad, max_retry=0)
    assert not path_bad.is_valid.any()
    assert_equal(franka.get_qpos(), qpos_before)

    # Conflicting or incomplete goal specifications raise.
    hand = franka.get_link("hand")
    with pytest.raises(gs.GenesisException):
        franka.plan_path()
    with pytest.raises(gs.GenesisException):
        franka.plan_path(qpos_goal, goal_link=hand, goal_pos=[0.4, 0.0, 0.5])
    with pytest.raises(gs.GenesisException):
        franka.plan_path(goal_pos=[0.4, 0.0, 0.5])
    with pytest.raises(gs.GenesisException):
        franka.plan_path(qpos_goal, ee_link_name="hand")

    if n_envs > 0:
        # Batched planning with per-env goals, and env subsets.
        goals = np.stack([qpos_goal] * n_envs)
        goals[-1, 0] = -0.9
        path_multi = franka.plan_path(goals)
        assert path_multi.is_valid.all()
        assert_allclose(path_multi.qpos[-1], torch.as_tensor(goals, dtype=gs.tc_float), tol=1e-4)
        path_sub = franka.plan_path(qpos_goal, envs_idx=[0])
        assert path_sub.qpos.shape[1] == 1 and path_sub.is_valid.all()

    # Ground truth: sweeping the returned waypoints through the real collider never penetrates the obstacles.
    max_penetration = _max_true_penetration(scene, franka, path.qpos, (pillar,), show_viewer)
    assert max_penetration < 5e-3


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_plan_path_with_attached_entity(n_envs, show_viewer):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 1.5, 1.5),
            camera_lookat=(0.2, 0.0, 0.7),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    pillar = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 1.2),
            pos=(0.45, 0.0, 0.6),
            fixed=True,
        ),
    )
    # A cube held between the open fingers (squeezed by the fingertip pads), and a bystander cube on the floor.
    held = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.088, 0.0, 0.81),
        ),
    )
    bystander = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.5, 0.35, 0.02),
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=n_envs)

    qpos_goal = np.array([0.9, 0.6, 0.0, -1.4, 0.0, 2.0, 0.8, 0.02, 0.02])
    held_pos_before = held.get_pos().clone()
    held_quat_before = held.get_quat().clone()
    bystander_pos_before = bystander.get_pos().clone()

    # Explicit attachment: the carried cube is certified collision-free and its live pose is never touched.
    path = franka.plan_path(qpos_goal, ee_link_name="hand", with_entity=held, seed=3)
    assert path.is_valid.all()
    assert_equal(held.get_pos(), held_pos_before)
    assert_equal(held.get_quat(), held_quat_before)

    # Ground truth on the carried volume: compose the cube pose from the hand pose at every waypoint and check
    # its corners stay clear of the pillar.
    hand = franka.get_link("hand")
    hand_pos_0 = hand.get_pos()
    hand_quat_0 = hand.get_quat()
    grasp_pos, grasp_quat = gu.inv_transform_pos_quat_by_trans_quat(
        held_pos_before, held_quat_before, hand_pos_0, hand_quat_0
    )
    corners = 0.02 * torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=gs.tc_float,
    )
    min_clearance = torch.inf
    for waypoint in path.qpos:
        franka.set_qpos(waypoint, zero_velocity=False)
        cube_pos, cube_quat = gu.transform_pos_quat_by_trans_quat(
            grasp_pos, grasp_quat, hand.get_pos(), hand.get_quat()
        )
        corners_world = cube_pos[..., None, :] + gu.transform_by_quat(
            corners.expand(*cube_pos.shape[:-1], 8, 3), cube_quat[..., None, :].expand(*cube_quat.shape[:-1], 8, 4)
        )
        delta = (corners_world - torch.tensor([0.45, 0.0, 0.6], dtype=gs.tc_float)).abs() - torch.tensor(
            [0.05, 0.05, 0.6], dtype=gs.tc_float
        )
        min_clearance = min(min_clearance, float(delta.amax(dim=-1).min()))
    franka.set_qpos(path.qpos[0], zero_velocity=False)
    assert min_clearance > -1e-3

    # Auto-grasp: with no explicit arguments, the squeezed cube is detected and carried; the bystander cube
    # (touching nothing) stays a plain obstacle.
    path_auto = franka.plan_path(qpos_goal, seed=3)
    assert path_auto.is_valid.all()
    assert_equal(held.get_pos(), held_pos_before)
    assert_equal(bystander.get_pos(), bystander_pos_before)
    context = scene.rigid_solver.planner._entity_contexts[franka.idx]
    assert qd_to_torch(context.planner_info.fk.attach_is_active).any()

    # Auto-grasp off: nothing is attached, and the cube between the fingers becomes an obstacle whose start
    # contacts are excluded, so planning still succeeds away from it.
    franka.plan_path(qpos_goal, attach_held_entities=False, seed=3)
    assert not qd_to_torch(context.planner_info.fk.attach_is_active).any()

    # Planning toward a grasp, hands free: a pre-grasp pose with the open fingers straddling the bystander cube
    # is proxy-colliding by nature, and certifies through the goal-contact allowance.
    path_pregrasp = franka.plan_path(
        max_retry=2,
        attach_held_entities=False,
        goal_link=franka.get_link("hand"),
        goal_pos=[0.5, 0.35, 0.12],
        goal_quat=[0.0, 1.0, 0.0, 0.0],
        seed=3,
    )
    assert path_pregrasp.is_valid.all()
    franka.set_qpos(path_pregrasp.qpos[-1], zero_velocity=False)
    assert_allclose(hand.get_pos(), [0.5, 0.35, 0.12], tol=6e-3)


@pytest.mark.slow
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_plan_path_clearance_and_narrow_passage(n_envs, show_viewer):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, 1.5, 1.5),
            camera_lookat=(0.4, 0.0, 0.5),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    # A radial wall between the start and goal azimuths: the straight sweep hits it, the detour lifts over.
    wall = scene.add_entity(
        gs.morphs.Box(
            size=(0.6, 0.04, 0.85),
            pos=(0.42, 0.42, 0.425),
            euler=(0.0, 0.0, 45.0),
            fixed=True,
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=n_envs)

    qpos_start = np.array([0.0, 0.7, 0.0, -1.4, 0.0, 2.1, 0.8, 0.02, 0.02])
    qpos_goal = np.array([1.9, 0.7, 0.0, -1.4, 0.0, 2.1, 0.8, 0.02, 0.02])

    # The detour over the wall is beyond local optimization from the straight seed, so this exercises the
    # sampling fallback; a couple of retries must be enough.
    path = franka.plan_path(qpos_goal, qpos_start=qpos_start, max_retry=2, seed=5)
    assert path.is_valid.all()
    max_penetration = _max_true_penetration(scene, franka, path.qpos, (wall,), show_viewer)
    assert max_penetration < 5e-3

    # A requested clearance is honored along the whole path (up to the proxy conservatism), and an impossible
    # clearance reports failure rather than returning a violating path.
    path_margin = franka.plan_path(qpos_goal, qpos_start=qpos_start, max_retry=2, safety_margin=0.03, seed=5)
    if path_margin.is_valid.all():
        collider_state = scene.rigid_solver.collider._collider_state
        wall_geoms = {geom.idx for link in wall.links for geom in link.geoms}
        for waypoint in path_margin.qpos:
            franka.set_qpos(waypoint, zero_velocity=False)
            scene.rigid_solver.collider.detection()
            n_contacts = qd_to_torch(collider_state.n_contacts)
            geom_a = qd_to_torch(collider_state.contact_data.geom_a)
            for i_c in range(int(n_contacts[0])):
                assert int(geom_a[i_c, 0]) not in wall_geoms
    path_impossible = franka.plan_path(qpos_goal, qpos_start=qpos_start, max_retry=0, safety_margin=0.15)
    assert not path_impossible.is_valid.any()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 16])
def test_plan_to_pregrasp_goal_in_clutter(n_envs, show_viewer):
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, -1.3, 1.5),
            camera_lookat=(0.05, 0.1, 0.45),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    # Four corner groups of four fixed boxes in an L pointing away from the robot, the fourth box right below
    # the L corner: tight spacing and tall groups leave only narrow corridors down to the free box.
    obstacles = []
    for i_grp, height in enumerate((0.7, 0.6, 0.8, 0.6)):
        azimuth = np.pi / 4 + i_grp * np.pi / 2
        u_rad = np.array([np.cos(azimuth), np.sin(azimuth)])
        u_tan = np.array([-np.sin(azimuth), np.cos(azimuth)])
        corner = 0.35 * u_rad
        for xy, z in (
            (corner, height),
            (corner + 0.15 * u_rad, height),
            (corner + 0.15 * u_tan, height),
            (corner, height - 0.1),
        ):
            obstacles.append(
                scene.add_entity(
                    gs.morphs.Box(
                        size=(0.1, 0.1, 0.1),
                        pos=(xy[0], xy[1], z),
                        fixed=True,
                    ),
                )
            )
    free_box = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            pos=(0.35, 0.13, 0.02),
        ),
    )
    franka = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=n_envs)

    # One challenging pre-grasp spot per env: the mid-gap corridors between the groups, the inner ring under the
    # L corners, tight spots beside the radial arms, and the pocket mouths.
    box_xy = [
        (0.40, 0.00),
        (0.00, 0.40),
        (-0.40, 0.00),
        (0.00, -0.40),
        (0.22, 0.22),
        (-0.22, 0.22),
        (-0.22, -0.22),
        (0.05, -0.30),
        (0.52, 0.18),
        (-0.18, 0.52),
        (-0.52, -0.18),
        (0.18, -0.52),
        (0.35, 0.13),
        (-0.13, 0.35),
        (-0.35, -0.13),
        (0.13, -0.35),
    ][: max(n_envs, 1)]
    box_pos = torch.tensor([[x, y, 0.02] for x, y in box_xy], dtype=gs.tc_float)
    # Top-down pre-grasps with the wrist yaw facing each box's azimuth: the natural approach orientation, so
    # every spot stays kinematically comfortable while the surroundings do the stressing.
    goal_quat = torch.as_tensor(
        np.stack(
            [
                gu.transform_quat_by_quat(
                    np.array([0.0, 1.0, 0.0, 0.0]), gu.rotvec_to_quat(np.array([0.0, 0.0, np.arctan2(y, x)]))
                )
                for x, y in box_xy
            ]
        ),
        dtype=gs.tc_float,
    )
    goal_pos = box_pos + torch.tensor([0.0, 0.0, 0.11], dtype=gs.tc_float)
    if n_envs == 0:
        box_pos, goal_pos, goal_quat = box_pos[0], goal_pos[0], goal_quat[0]
    free_box.set_pos(box_pos)
    franka.set_qpos([0.0, -0.3, 0.0, -1.0, 0.0, 1.5, 0.785, 0.04, 0.04])

    # Every pre-grasp is contact-rich (fingers within proxy padding of the box and the floor), so this exercises
    # the goal-contact allowance, the multi-restart goal resolution and its per-retry branch reconsideration,
    # and convergence through the corridors, in every env at once.
    hand = franka.get_link("hand")
    for seed in (0, 1, 2):
        path = franka.plan_path(
            max_retry=2,
            goal_link=hand,
            goal_pos=goal_pos,
            goal_quat=goal_quat,
            seed=seed,
        )
        assert path.is_valid.all()

    # Ground truth on the last plan: the real collider reports no penetration of the fixed clutter anywhere
    # along the returned path, and the final waypoint reaches the pre-grasp pose in every env.
    max_penetration = _max_true_penetration(scene, franka, path.qpos, obstacles, show_viewer)
    assert max_penetration < 5e-3
    franka.set_qpos(path.qpos[-1], zero_velocity=False)
    assert_allclose(hand.get_pos(), goal_pos, tol=6e-3)
