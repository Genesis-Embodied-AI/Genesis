from itertools import product

import numpy as np
import pytest
import trimesh

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .utils import assert_allclose


@pytest.mark.required
def test_partition_logics(show_viewer):
    # The welded pair never touches, so only the equality edge couples them: without it the partition would split them
    # and the weld would be solved across two islands. A fixed body carries no dofs and joins no island.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -4.0, 2.5),
            camera_lookat=(1.0, 0.0, 0.1),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_bottom = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05)))
    box_top = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.16)))
    box_weld_a = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, 0.05)))
    box_weld_b = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.3, 0.0, 0.05)))
    box_alone = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(2.0, 0.0, 0.05)))
    scene.build(n_envs=1)

    scene.rigid_solver.add_weld_constraint(box_weld_a.base_link_idx, box_weld_b.base_link_idx)

    for _ in range(50):
        scene.step()

    # Deferred import: the solver package executes module-scope dtype code (gs.qd_float) that only exists after gs.init.
    from genesis.engine.solvers.rigid.island import kernel_build_islands, kernel_group_constraints_by_island

    solver = scene.rigid_solver
    island_state = array_class.get_island_state(solver, solver.collider)
    kernel_build_islands(
        solver.entities_info,
        solver.entities_state,
        solver.links_info,
        solver.joints_info,
        solver.equalities_info,
        solver.constraint_solver.constraint_state,
        solver.collider._collider_state,
        island_state,
        solver._static_rigid_sim_config,
    )

    entities_island_idx = qd_to_numpy(island_state.entities_island_idx, transpose=True)[0]
    n_islands = int(qd_to_numpy(island_state.n_islands)[0])
    isl = {
        name: entities_island_idx[ent.idx]
        for name, ent in {
            "bottom": box_bottom,
            "top": box_top,
            "weld_a": box_weld_a,
            "weld_b": box_weld_b,
            "alone": box_alone,
        }.items()
    }
    assert all(v >= 0 for v in isl.values())
    assert isl["top"] == isl["bottom"]
    assert isl["weld_a"] == isl["weld_b"]
    assert len({isl["bottom"], isl["weld_a"], isl["alone"]}) == 3
    assert n_islands == 3

    # Each free box has 6 dofs, so the stack and welded pair hold 12 each and the lone box 6, covering every dof once.
    island_dof_n = qd_to_numpy(island_state.dof_slices.n, transpose=True)[0]
    assert sorted(island_dof_n[:n_islands].tolist()) == [6, 12, 12]
    island_dof_start = qd_to_numpy(island_state.dof_slices.start, transpose=True)[0]
    dof_id = qd_to_numpy(island_state.dof_id, transpose=True)[0]
    k = isl["alone"]
    seg = sorted(dof_id[island_dof_start[k] : island_dof_start[k] + island_dof_n[k]].tolist())
    assert seg == list(range(box_alone.dof_start, box_alone.dof_start + box_alone.n_dofs))

    n_contacts = int(qd_to_numpy(solver.collider._collider_state.n_contacts)[0])
    island_contact_n = qd_to_numpy(island_state.contact_slices.n, transpose=True)[0]
    assert int(island_contact_n[:n_islands].sum()) == n_contacts
    assert island_contact_n[isl["bottom"]] >= 1

    kernel_group_constraints_by_island(
        island_state,
        solver.constraint_solver.constraint_state,
        solver._rigid_global_info,
        solver._static_rigid_sim_config,
    )
    n_constraints = int(qd_to_numpy(solver.constraint_solver.constraint_state.n_constraints)[0])
    island_constraint_n = qd_to_numpy(island_state.constraint_slices.n, transpose=True)[0]
    assert int(island_constraint_n[:n_islands].sum()) == n_constraints
    assert island_constraint_n[isl["weld_a"]] >= 1


@pytest.mark.required
def test_partition_track_changes(show_viewer):
    # The partition is rebuilt every step, so it must track contacts forming (merge) and breaking (split).
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -4.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.2),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    box_lower = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05)))
    box_upper = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.40)))
    scene.build(n_envs=1)

    from genesis.engine.solvers.rigid.island import kernel_build_islands

    solver = scene.rigid_solver
    island_state = array_class.get_island_state(solver, solver.collider)

    def n_islands_now():
        kernel_build_islands(
            solver.entities_info,
            solver.entities_state,
            solver.links_info,
            solver.joints_info,
            solver.equalities_info,
            solver.constraint_solver.constraint_state,
            solver.collider._collider_state,
            island_state,
            solver._static_rigid_sim_config,
        )
        return int(qd_to_numpy(island_state.n_islands)[0])

    scene.step()
    assert n_islands_now() == 2
    for _ in range(120):
        scene.step()
    assert n_islands_now() == 1
    box_upper.set_pos(np.array([0.0, 0.0, 0.40]))
    scene.step()
    assert n_islands_now() == 2


@pytest.mark.required
@pytest.mark.parametrize("noslip_iterations", [0, 5])
def test_solve_correctness(show_viewer, noslip_iterations):
    # Partitioning the solve into per-island blocks must not change the result (the global Hessian is block-diagonal by
    # island). The noslip pass is a global post-solve refinement reading the island-solved accelerations, so it
    # composes too. sparse_solve=False so the dense per-island Hessian gets the full jac it needs.
    positions = []
    for use_contact_island in (False, True):
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=1.0 / 60.0,
                gravity=(0.0, 0.0, -9.8),
            ),
            rigid_options=gs.options.RigidOptions(
                use_contact_island=use_contact_island,
                constraint_solver=gs.constraint_solver.Newton,
                sparse_solve=False,
                noslip_iterations=noslip_iterations,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(1.0, -4.0, 2.5),
                camera_lookat=(1.0, 0.0, 0.1),
            ),
            show_viewer=show_viewer,
        )
        scene.add_entity(gs.morphs.Plane())
        box_bottom = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05)))
        box_top = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.16)))
        box_weld_a = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, 0.05)))
        box_weld_b = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.3, 0.0, 0.05)))
        box_alone = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(2.0, 0.0, 0.05)))
        scene.build(n_envs=1)
        scene.rigid_solver.add_weld_constraint(box_weld_a.base_link_idx, box_weld_b.base_link_idx)
        for _ in range(80):
            scene.step()
        boxes = (box_bottom, box_top, box_weld_a, box_weld_b, box_alone)
        positions.append(np.stack([tensor_to_array(b.get_pos()).reshape(-1) for b in boxes]))

    # Loose tol: the monolith's incremental Cholesky vs the island path's direct rebuild are both exact in theory, but
    # 80 steps of a chaotic stack drift apart at fp-accumulation level.
    assert_allclose(positions[1], positions[0], tol=5e-3)


@pytest.mark.required
def test_pruning(show_viewer):
    # A convex-decomposed box is a compound body (27 sub-box geoms on one link), so its ground contacts pile up per
    # link-pair and pruning collapses them. The island construction reads contacts through contact_sort_idx, so pruning
    # and islands run together; each box then settles with its bottom face on the plane, center at its half-height.
    half = 0.1
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
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
    boxes = [scene.add_entity(gs.morphs.MeshSet(files=sub_meshes, pos=(i * 0.5, 0.0, 0.3))) for i in range(3)]
    scene.build(n_envs=1)

    solver = scene.rigid_solver
    assert solver.collider._collider_static_config.has_prunable_contacts
    assert solver._use_contact_island

    for _ in range(150):
        scene.step()
    for box in boxes:
        assert_allclose(box.get_pos()[..., 2], half, atol=5e-3)


@pytest.mark.required
def test_weld_coupling(show_viewer):
    # box2 hangs from a weld onto the anchored box1 at a horizontal offset, never touching it. Without the equality
    # edge in the partition the two land in different islands and the weld is dropped, letting box2 free-fall.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.15, -4.0, 2.5),
            camera_lookat=(0.15, 0.0, 0.9),
        ),
        show_viewer=show_viewer,
    )
    box1 = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 1.0), fixed=True))
    box2 = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.3, 0.0, 1.0)))
    scene.build(n_envs=1)

    scene.rigid_solver.add_weld_constraint(box1.base_link_idx, box2.base_link_idx)

    z_start = float(box2.get_pos()[..., 2])
    for _ in range(120):
        scene.step()
    # A dropped weld would free-fall ~2 m in 2 s; the weld holds box2 near its start height.
    assert_allclose(box2.get_pos()[..., 2], z_start, tol=0.15)


@pytest.mark.required
def test_sparsity(show_viewer):
    # On CPU the sparse Jacobian and the per-island solve exploit the same block-diagonal structure and must compose
    # (islands own the per-block factorization, the sparse jac makes products and the constraint-to-island lookup
    # O(nonzeros)). On GPU the dense tiled path wins, so sparse is dropped and islands stand alone.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
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
    box_a = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.3)))
    box_b = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, 0.3)))
    scene.build(n_envs=1)
    for _ in range(200):
        scene.step()

    solver = scene.rigid_solver
    if gs.backend == gs.cpu:
        assert solver._static_rigid_sim_config.sparse_solve and solver._use_contact_island
    else:
        assert solver._use_contact_island and not solver._static_rigid_sim_config.sparse_solve

    assert_allclose(box_a.get_pos()[..., 2], 0.05, atol=2e-3)
    assert_allclose(box_b.get_pos()[..., 2], 0.05, atol=2e-3)


def test_hibernation_wakes_on_user_input(show_viewer):
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
    box_force = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.1)))
    box_pos = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, 0.1)))
    box_vel = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(2.0, 0.0, 0.1)))
    box_qpos = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(3.0, 0.0, 0.1)))
    box_cforce = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(4.0, 0.0, 0.1)))
    box_cvel = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(5.0, 0.0, 0.1)))
    box_cpos = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(6.0, 0.0, 0.1)))
    scene.build(n_envs=1)
    solver = scene.rigid_solver

    def asleep(entity):
        return bool(qd_to_numpy(solver.entities_state.is_hibernated, transpose=True).reshape(-1)[entity.idx])

    def z_of(entity):
        return float(entity.get_pos()[..., 2])

    # Velocity/position control need PD gains to produce a force; index 2 of each free joint is the world-z dof.
    for box in (box_cvel, box_cpos):
        box.set_dofs_kp([0.0, 0.0, 400.0, 0.0, 0.0, 0.0])
        box.set_dofs_kv([40.0, 40.0, 40.0, 4.0, 4.0, 4.0])

    n_fall = 8
    free_fall_drop = 0.5 * G * (n_fall * DT) ** 2

    for _ in range(300):
        scene.step()
    assert all(asleep(b) for b in (box_force, box_pos, box_vel, box_qpos, box_cforce, box_cvel, box_cpos))

    z0 = z_of(box_force)
    for _ in range(6):
        solver.apply_links_external_force([[0.0, 0.0, 40.0]], links_idx=[box_force.base_link_idx])
        scene.step()
    assert not asleep(box_force) and z_of(box_force) > z0 + 0.02
    assert all(asleep(b) for b in (box_pos, box_vel, box_qpos, box_cforce, box_cvel, box_cpos))

    box_pos.set_dofs_position([1.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    assert not asleep(box_pos)
    z0 = z_of(box_pos)
    for _ in range(n_fall):
        scene.step()
    assert_allclose(z0 - z_of(box_pos), free_fall_drop, rtol=0.2)
    assert all(asleep(b) for b in (box_vel, box_qpos, box_cforce, box_cvel, box_cpos))

    box_vel.set_dofs_velocity([0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    assert not asleep(box_vel)
    z0 = z_of(box_vel)
    for _ in range(5):
        scene.step()
    assert z_of(box_vel) > z0 + 0.05
    assert all(asleep(b) for b in (box_qpos, box_cforce, box_cvel, box_cpos))

    box_qpos.set_qpos([3.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0])
    assert not asleep(box_qpos)
    z0 = z_of(box_qpos)
    for _ in range(n_fall):
        scene.step()
    assert_allclose(z0 - z_of(box_qpos), free_fall_drop, rtol=0.2)
    assert all(asleep(b) for b in (box_cforce, box_cvel, box_cpos))

    z0 = z_of(box_cforce)
    for _ in range(8):
        box_cforce.control_dofs_force([0.0, 0.0, 30.0, 0.0, 0.0, 0.0])
        scene.step()
    assert not asleep(box_cforce) and z_of(box_cforce) > z0 + 0.02
    assert all(asleep(b) for b in (box_cvel, box_cpos))

    z0 = z_of(box_cvel)
    for _ in range(8):
        box_cvel.control_dofs_velocity([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        scene.step()
    assert not asleep(box_cvel) and z_of(box_cvel) > z0 + 0.02
    assert asleep(box_cpos)

    z0 = z_of(box_cpos)
    for _ in range(12):
        box_cpos.control_dofs_position([6.0, 0.0, 0.6, 0.0, 0.0, 0.0])
        scene.step()
    assert not asleep(box_cpos) and z_of(box_cpos) > z0 + 0.05


def test_hibernation_wakes_on_collision(show_viewer):
    # An awake body striking a sleeping one must wake it so it responds instead of acting as an immovable obstacle.
    # This needs the broad-phase sort-buffer refresh of awake geoms (so the contact is detected) and the wake-on-contact
    # pass.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
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
    box_rest = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05)))
    box_hit = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.22, 0.0, 0.05)))
    scene.build(n_envs=1)
    solver = scene.rigid_solver

    def asleep(entity):
        return bool(qd_to_numpy(solver.entities_state.is_hibernated, transpose=True).reshape(-1)[entity.idx])

    for _ in range(300):
        scene.step()
    assert asleep(box_rest) and asleep(box_hit)
    rest_x0 = float(box_rest.get_pos()[..., 0])

    box_hit.set_dofs_velocity(np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    for _ in range(30):
        scene.step()

    # The struck sleeper woke and was knocked; the striker was stopped by it (did not tunnel through).
    assert not asleep(box_rest)
    rest_x1 = float(box_rest.get_pos()[..., 0])
    hit_x1 = float(box_hit.get_pos()[..., 0])
    assert rest_x1 < rest_x0 - 1e-3
    assert hit_x1 > rest_x1


def test_hibernation_wakes_on_daisy_chain(show_viewer):
    # Two welded bodies sleep as ONE island. Disturbing only box_a must wake the WHOLE island via the daisy chain, else
    # its coupled partner stays frozen and the weld is solved against a sleeping body. A weld is used (not a contact
    # stack, whose micro-settling keeps it awake); a separated third box is its own island and stays asleep.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
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
    box_a = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05)))
    box_b = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.3, 0.0, 0.05)))
    box_far = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(2.0, 0.0, 0.05)))
    scene.build(n_envs=1)
    solver = scene.rigid_solver

    solver.add_weld_constraint(box_a.base_link_idx, box_b.base_link_idx)

    def asleep(entity):
        return bool(qd_to_numpy(solver.entities_state.is_hibernated, transpose=True).reshape(-1)[entity.idx])

    for _ in range(400):
        scene.step()
    assert asleep(box_a) and asleep(box_b) and asleep(box_far)

    solver.apply_links_external_force(np.array([[20.0, 0.0, 0.0]]), links_idx=[box_a.base_link_idx])
    scene.step()
    assert not asleep(box_a)
    assert not asleep(box_b)
    assert asleep(box_far)


@pytest.mark.required
def test_hibernation_repartitioning(show_viewer):
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
    box1 = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(-0.3, 0.0, 0.15)))
    box2 = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.3, 0.0, 0.15)))
    scene.build()

    solver = scene.sim.rigid_solver
    box1_idx = box1._idx_in_solver
    box2_idx = box2._idx_in_solver

    for _ in range(200):
        scene.step()
        if solver.entities_state.is_hibernated[box1_idx, 0] and solver.entities_state.is_hibernated[box2_idx, 0]:
            break
    assert solver.entities_state.is_hibernated[box1_idx, 0]
    assert solver.entities_state.is_hibernated[box2_idx, 0]
    assert solver.constraint_solver.island_state.n_islands[0] == 2

    box2_pos = box2.get_pos()
    box1.set_pos(np.array([float(box2_pos[0]) + 0.01, float(box2_pos[1]) + 0.01, 0.3]))
    assert not solver.entities_state.is_hibernated[box1_idx, 0]
    assert float(box1.get_pos()[2]) > 0.2

    for _ in range(25):
        scene.step()
    assert not solver.entities_state.is_hibernated[box1_idx, 0]
    assert not solver.entities_state.is_hibernated[box2_idx, 0]

    for _ in range(200):
        scene.step()
        if solver.entities_state.is_hibernated[box1_idx, 0] and solver.entities_state.is_hibernated[box2_idx, 0]:
            break
    assert solver.entities_state.is_hibernated[box1_idx, 0]
    assert solver.entities_state.is_hibernated[box2_idx, 0]
    assert solver.constraint_solver.island_state.n_islands[0] == 1

    box1.set_pos(np.array([1.0, 0.0, 0.15]))
    assert not solver.entities_state.is_hibernated[box1_idx, 0]
    assert not solver.entities_state.is_hibernated[box2_idx, 0]

    for _ in range(500):
        scene.step()
        if solver.entities_state.is_hibernated[box1_idx, 0] and solver.entities_state.is_hibernated[box2_idx, 0]:
            break
    assert solver.entities_state.is_hibernated[box1_idx, 0]
    assert solver.entities_state.is_hibernated[box2_idx, 0]
    assert solver.constraint_solver.island_state.n_islands[0] == 2
