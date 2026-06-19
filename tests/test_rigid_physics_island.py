from itertools import product

import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .utils import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_partition_logics(show_viewer, n_envs):
    # The welded pair never touches, so only the equality edge couples them: without it the partition would split them
    # and the weld would be solved across two islands. A fixed body carries no dofs and joins no island.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -4.0, 2.5),
            camera_lookat=(1.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_bottom = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.05),
        )
    )
    box_top = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.16),
        )
    )
    box_weld_a = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(1.0, 0.0, 0.05),
        )
    )
    box_weld_b = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(1.3, 0.0, 0.05),
        )
    )
    box_alone = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(2.0, 0.0, 0.05),
        )
    )
    scene.build(n_envs=n_envs)

    scene.rigid_solver.add_weld_constraint(box_weld_a.base_link_idx, box_weld_b.base_link_idx)

    for _ in range(45):
        scene.step()

    # The partition is rebuilt inside every step; inspect the one the solver actually used this step.
    solver = scene.rigid_solver
    island_state = solver.constraint_solver.island_state

    island_idx = qd_to_numpy(island_state.entities_island_idx)
    island_of = {
        name: island_idx[entity.idx]
        for name, entity in (
            ("bottom", box_bottom),
            ("top", box_top),
            ("weld_a", box_weld_a),
            ("weld_b", box_weld_b),
            ("alone", box_alone),
        )
    }
    assert all((v >= 0).all() for v in island_of.values())
    assert_equal(island_of["top"], island_of["bottom"])
    assert_equal(island_of["weld_a"], island_of["weld_b"])
    # The stack, the welded pair and the lone box land in three distinct islands in every env.
    assert (island_of["bottom"] != island_of["weld_a"]).all()
    assert (island_of["bottom"] != island_of["alone"]).all()
    assert (island_of["weld_a"] != island_of["alone"]).all()
    assert_equal(qd_to_numpy(island_state.n_islands), 3)

    # Per env: each free box has 6 dofs (stack and welded pair hold 12 each, lone box 6); per-island contact and
    # constraint counts sum back to the env total; and the lone island holds exactly the lone box's dofs.
    n_islands = qd_to_numpy(island_state.n_islands)
    island_dof_n = qd_to_numpy(island_state.dof_slices.n)
    island_dof_start = qd_to_numpy(island_state.dof_slices.start)
    dof_id = qd_to_numpy(island_state.dof_id)
    island_contact_n = qd_to_numpy(island_state.contact_slices.n)
    island_constraint_n = qd_to_numpy(island_state.constraint_slices.n)
    n_contacts = qd_to_numpy(solver.collider._collider_state.n_contacts)
    n_constraints = qd_to_numpy(solver.constraint_solver.constraint_state.n_constraints)
    alone_dofs = list(range(box_alone.dof_start, box_alone.dof_start + box_alone.n_dofs))
    for i_env in range(island_idx.shape[1]):
        n = n_islands[i_env]
        assert sorted(island_dof_n[:n, i_env].tolist()) == [6, 12, 12]
        assert island_contact_n[:n, i_env].sum() == n_contacts[i_env]
        assert island_constraint_n[:n, i_env].sum() == n_constraints[i_env]
        assert island_contact_n[island_of["bottom"][i_env], i_env] >= 1
        assert island_constraint_n[island_of["weld_a"][i_env], i_env] >= 1
        k = island_of["alone"][i_env]
        seg = dof_id[island_dof_start[k, i_env] : island_dof_start[k, i_env] + island_dof_n[k, i_env], i_env]
        assert sorted(seg.tolist()) == alone_dofs


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_partition_track_changes(show_viewer, n_envs):
    # The partition is rebuilt every step, so it must track contacts forming (merge) and breaking (split).
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -4.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_lower = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.05),
        )
    )
    box_upper = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.40),
        )
    )
    scene.build(n_envs=n_envs)

    # The step rebuilds the partition; read the island count the solver actually used this step.
    island_state = scene.rigid_solver.constraint_solver.island_state

    def n_islands_now():
        return qd_to_numpy(island_state.n_islands)

    scene.step()
    assert_equal(n_islands_now(), 2)
    for _ in range(45):
        scene.step()
    assert_equal(n_islands_now(), 1)
    box_upper.set_pos([0.0, 0.0, 0.40])
    scene.step()
    assert_equal(n_islands_now(), 2)


@pytest.mark.required
@pytest.mark.parametrize("noslip_iterations", [0, 5])
@pytest.mark.parametrize("n_envs", [0, 2])
def test_solve_correctness(show_viewer, noslip_iterations, n_envs):
    # Partitioning the solve into per-island blocks must not change the result (the global Hessian is block-diagonal by
    # island). The noslip pass is a global post-solve refinement reading the island-solved accelerations, so it
    # composes too.
    positions = []
    for use_contact_island in (False, True):
        scene = gs.Scene(
            rigid_options=gs.options.RigidOptions(
                use_contact_island=use_contact_island,
                noslip_iterations=noslip_iterations,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.0, -4.0, 2.5),
                camera_lookat=(1.0, 0.0, 0.1),
            ),
            show_viewer=show_viewer,
        )
        scene.add_entity(gs.morphs.Plane())
        box_bottom = scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(0.0, 0.0, 0.05),
            )
        )
        box_top = scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(0.0, 0.0, 0.16),
            )
        )
        box_weld_a = scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(1.0, 0.0, 0.05),
            )
        )
        box_weld_b = scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(1.3, 0.0, 0.05),
            )
        )
        box_alone = scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(2.0, 0.0, 0.05),
            )
        )
        scene.build(n_envs=n_envs)

        scene.rigid_solver.add_weld_constraint(box_weld_a.base_link_idx, box_weld_b.base_link_idx)
        for _ in range(45):
            scene.step()
        boxes = (box_bottom, box_top, box_weld_a, box_weld_b, box_alone)
        positions.append(np.stack([tensor_to_array(b.get_pos()) for b in boxes]))

    # Loose tol: the monolith's incremental Cholesky vs the island path's direct rebuild are both exact in theory, but
    # 80 steps of a chaotic stack drift apart at fp-accumulation level.
    assert_allclose(positions[1], positions[0], tol=5e-3)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_pruning(show_viewer, n_envs):
    # A convex-decomposed box is a compound body (27 sub-box geoms on one link), so its ground contacts pile up per
    # link-pair and pruning collapses them. The island construction reads contacts through contact_sort_idx, so pruning
    # and islands run together; each box then settles with its bottom face on the plane, center at its half-height.
    half = 0.1
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -4.0, 2.5),
            camera_lookat=(0.5, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    sub_meshes = []
    for sx, sy, sz in product((-1, 0, 1), repeat=3):
        mesh = trimesh.creation.box(extents=(2 / 3 * half,) * 3)
        mesh.apply_translation((2 / 3 * sx * half, 2 / 3 * sy * half, 2 / 3 * sz * half))
        sub_meshes.append(mesh)
    boxes = [
        scene.add_entity(
            gs.morphs.MeshSet(
                files=sub_meshes,
                pos=(i * 0.5, 0.0, 0.3),
            )
        )
        for i in range(3)
    ]
    scene.build(n_envs=n_envs)

    for _ in range(60):
        scene.step()
    for box in boxes:
        assert_allclose(box.get_pos()[..., 2], half, atol=5e-3)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_weld_coupling(show_viewer, n_envs):
    # box2 hangs from a weld onto the anchored box1 at a horizontal offset, never touching it. Without the equality
    # edge in the partition the two land in different islands and the weld is dropped, letting box2 free-fall.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.15, -4.0, 2.5),
            camera_lookat=(0.15, 0.0, 0.9),
        ),
        show_viewer=show_viewer,
    )
    box1 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 1.0),
            fixed=True,
        )
    )
    box2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.3, 0.0, 1.0),
        )
    )
    scene.build(n_envs=n_envs)

    scene.rigid_solver.add_weld_constraint(box1.base_link_idx, box2.base_link_idx)

    z_start = box2.get_pos()[..., 2]
    for _ in range(50):
        scene.step()
    # A dropped weld would free-fall ~1 m in 1 s; the weld holds box2 near its start height.
    assert_allclose(box2.get_pos()[..., 2], z_start, tol=0.15)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_sparsity(show_viewer, n_envs):
    # On CPU the sparse Jacobian and the per-island solve exploit the same block-diagonal structure and must compose
    # (islands own the per-block factorization, the sparse jac makes products and the constraint-to-island lookup
    # O(nonzeros)). On GPU the dense tiled path wins, so sparse is dropped and islands stand alone.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            sparse_solve=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.5, -4.0, 2.5),
            camera_lookat=(0.5, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_a = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.3),
        )
    )
    box_b = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(1.0, 0.0, 0.3),
        )
    )
    scene.build(n_envs=n_envs)

    for _ in range(50):
        scene.step()

    assert_allclose(box_a.get_pos()[..., 2], 0.05, atol=2e-3)
    assert_allclose(box_b.get_pos()[..., 2], 0.05, atol=2e-3)


@pytest.mark.parametrize("n_envs", [0, 2])
def test_hibernation_wakes_on_user_input(show_viewer, n_envs):
    # Every user input that drives a sleeping body must wake it (and only its island) AND take effect: a hibernated
    # body's dofs are skipped by forward dynamics and integration, so the motion checks catch a body that wakes but
    # stays frozen (e.g. gravity cancelled by a neighbour's stale constraint force). Seven separated boxes are seven
    # islands, so each input wakes exactly one.
    G = 9.8
    DT = 1.0 / 60.0
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, -G),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, -4.0, 2.5),
            camera_lookat=(3.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_force = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.1),
        )
    )
    box_pos = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(1.0, 0.0, 0.1),
        )
    )
    box_vel = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(2.0, 0.0, 0.1),
        )
    )
    box_qpos = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(3.0, 0.0, 0.1),
        )
    )
    box_cforce = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(4.0, 0.0, 0.1),
        )
    )
    box_cvel = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(5.0, 0.0, 0.1),
        )
    )
    box_cpos = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(6.0, 0.0, 0.1),
        )
    )
    scene.build(n_envs=n_envs)

    solver = scene.rigid_solver

    def asleep(entity):
        return qd_to_numpy(solver.entities_state.is_hibernated, entity.idx).all()

    def z_of(entity):
        return entity.get_pos()[..., 2]

    # Velocity/position control need PD gains to produce a force; index 2 of each free joint is the world-z dof.
    for box in (box_cvel, box_cpos):
        box.set_dofs_kp([0.0, 0.0, 400.0, 0.0, 0.0, 0.0])
        box.set_dofs_kv([40.0, 40.0, 40.0, 4.0, 4.0, 4.0])

    n_fall = 8
    free_fall_drop = 0.5 * G * (n_fall * DT) ** 2

    for _ in range(90):
        scene.step()
    assert all(map(asleep, (box_force, box_pos, box_vel, box_qpos, box_cforce, box_cvel, box_cpos)))

    z0 = z_of(box_force)
    for _ in range(6):
        solver.apply_links_external_force([0.0, 0.0, 40.0], links_idx=[box_force.base_link_idx])
        scene.step()
    assert not asleep(box_force) and (z_of(box_force) > z0 + 0.02).all()
    assert all(map(asleep, (box_pos, box_vel, box_qpos, box_cforce, box_cvel, box_cpos)))

    box_pos.set_dofs_position([1.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    assert not asleep(box_pos)
    z0 = z_of(box_pos)
    for _ in range(n_fall):
        scene.step()
    assert_allclose(z0 - z_of(box_pos), free_fall_drop, rtol=0.2)
    assert all(map(asleep, (box_vel, box_qpos, box_cforce, box_cvel, box_cpos)))

    box_vel.set_dofs_velocity([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    assert not asleep(box_vel)
    z0 = z_of(box_vel)
    for _ in range(5):
        scene.step()
    assert (z_of(box_vel) > z0 + 0.05).all()
    assert all(map(asleep, (box_qpos, box_cforce, box_cvel, box_cpos)))

    box_qpos.set_qpos([3.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0])
    assert not asleep(box_qpos)
    z0 = z_of(box_qpos)
    for _ in range(n_fall):
        scene.step()
    assert_allclose(z0 - z_of(box_qpos), free_fall_drop, rtol=0.2)
    assert all(map(asleep, (box_cforce, box_cvel, box_cpos)))

    z0 = z_of(box_cforce)
    for _ in range(8):
        box_cforce.control_dofs_force([0.0, 0.0, 30.0, 0.0, 0.0, 0.0])
        scene.step()
    assert not asleep(box_cforce) and (z_of(box_cforce) > z0 + 0.02).all()
    assert all(map(asleep, (box_cvel, box_cpos)))

    z0 = z_of(box_cvel)
    for _ in range(8):
        box_cvel.control_dofs_velocity([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        scene.step()
    assert not asleep(box_cvel) and (z_of(box_cvel) > z0 + 0.02).all()
    assert asleep(box_cpos)

    z0 = z_of(box_cpos)
    for _ in range(12):
        box_cpos.control_dofs_position([6.0, 0.0, 0.6, 0.0, 0.0, 0.0])
        scene.step()
    assert not asleep(box_cpos) and (z_of(box_cpos) > z0 + 0.05).all()


@pytest.mark.parametrize("n_envs", [0, 2])
def test_hibernation_wakes_on_collision(show_viewer, n_envs):
    # An awake body striking a sleeping one must wake it so it responds instead of acting as an immovable obstacle.
    # This needs the broad-phase sort-buffer refresh of awake geoms (so the contact is detected) and the wake-on-contact
    # pass.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.1, -4.0, 2.5),
            camera_lookat=(0.1, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_rest = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.05),
        )
    )
    box_hit = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.22, 0.0, 0.05),
        )
    )
    scene.build(n_envs=n_envs)

    solver = scene.rigid_solver

    def asleep(entity):
        return qd_to_numpy(solver.entities_state.is_hibernated, entity.idx).all()

    for _ in range(50):
        scene.step()
    assert asleep(box_rest) and asleep(box_hit)
    rest_x0 = box_rest.get_pos()[..., 0]

    box_hit.set_dofs_velocity([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(30):
        scene.step()

    # The struck sleeper woke and was knocked; the striker was stopped by it (did not tunnel through).
    assert not asleep(box_rest)
    rest_x1 = box_rest.get_pos()[..., 0]
    hit_x1 = box_hit.get_pos()[..., 0]
    assert (rest_x1 < rest_x0 - 1e-3).all()
    assert (hit_x1 > rest_x1).all()


@pytest.mark.parametrize("n_envs", [0, 2])
def test_hibernation_wakes_on_daisy_chain(show_viewer, n_envs):
    # Two welded bodies sleep as ONE island. Disturbing only box_a must wake the WHOLE island via the daisy chain, else
    # its coupled partner stays frozen and the weld is solved against a sleeping body. A weld is used (not a contact
    # stack, whose micro-settling keeps it awake); a separated third box is its own island and stays asleep.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -4.0, 2.5),
            camera_lookat=(1.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_a = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.05),
        )
    )
    box_b = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.3, 0.0, 0.05),
        )
    )
    box_far = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(2.0, 0.0, 0.05),
        )
    )
    scene.build(n_envs=n_envs)

    solver = scene.rigid_solver

    solver.add_weld_constraint(box_a.base_link_idx, box_b.base_link_idx)

    def asleep(entity):
        return qd_to_numpy(solver.entities_state.is_hibernated, entity.idx).all()

    for _ in range(50):
        scene.step()
    assert asleep(box_a) and asleep(box_b) and asleep(box_far)

    solver.apply_links_external_force([20.0, 0.0, 0.0], links_idx=[box_a.base_link_idx])
    scene.step()
    assert not asleep(box_a)
    assert not asleep(box_b)
    assert asleep(box_far)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_hibernation_repartitioning(show_viewer, n_envs):
    # Full lifecycle of hibernation and the partition together: two boxes sleep apart (2 islands); moving one onto the
    # other wakes it, it collides, and the stack sleeps as one merged island; moving a box off the hibernated stack
    # must wake the WHOLE merged island (else the stale daisy chain keeps re-connecting both); they then split back.
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -4.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.15),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box1 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(-0.3, 0.0, 0.15),
        )
    )
    box2 = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.3, 0.0, 0.15),
        )
    )
    scene.build(n_envs=n_envs)

    solver = scene.sim.rigid_solver
    box1_idx = box1._idx_in_solver
    box2_idx = box2._idx_in_solver
    island_state = solver.constraint_solver.island_state

    def asleep(idx):
        return qd_to_numpy(solver.entities_state.is_hibernated, idx).all()

    def awake(idx):
        return not qd_to_numpy(solver.entities_state.is_hibernated, idx).any()

    for _ in range(60):
        scene.step()
        if asleep(box1_idx) and asleep(box2_idx):
            break
    assert asleep(box1_idx)
    assert asleep(box2_idx)
    assert_equal(qd_to_numpy(island_state.n_islands), 2)

    box2_pos = tensor_to_array(box2.get_pos())
    box1_target = box2_pos.copy()
    box1_target[..., 0] += 0.01
    box1_target[..., 1] += 0.01
    box1_target[..., 2] = 0.3
    box1.set_pos(box1_target)
    assert awake(box1_idx)
    assert (box1.get_pos()[..., 2] > 0.2).all()

    for _ in range(30):
        scene.step()
    assert awake(box1_idx)
    assert awake(box2_idx)

    for _ in range(60):
        scene.step()
        if asleep(box1_idx) and asleep(box2_idx):
            break
    assert asleep(box1_idx)
    assert asleep(box2_idx)
    assert_equal(qd_to_numpy(island_state.n_islands), 1)

    box1.set_pos([1.0, 0.0, 0.15])
    assert awake(box1_idx)
    assert awake(box2_idx)

    for _ in range(120):
        scene.step()
        if asleep(box1_idx) and asleep(box2_idx):
            break
    assert asleep(box1_idx)
    assert asleep(box2_idx)
    assert_equal(qd_to_numpy(island_state.n_islands), 2)
