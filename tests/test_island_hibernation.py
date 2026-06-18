import numpy as np
import pytest

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .utils import assert_allclose


@pytest.mark.required
def test_island_partition_groups_contacts_and_equalities():
    # The partitioner must group entities by ALL inter-entity coupling, not just contacts. A weld
    # between two non-contacting bodies must land them in the same island (the equality-edge fix);
    # bodies coupled only by contact must group; an isolated body must be its own island; fixed
    # bodies (the plane) carry no dofs and belong to no island. Driven on a monolith scene so it does
    # not depend on the (separate) per-island solve path.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=False,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    # Stack: box_top rests on box_bottom -> coupled by contact.
    box_bottom = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05)))
    box_top = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.16)))
    # Welded pair: 0.3 m apart (never touching) -> coupled only by the equality constraint.
    box_weld_a = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, 0.05)))
    box_weld_b = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.3, 0.0, 0.05)))
    # Isolated free body -> its own island.
    box_alone = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(2.0, 0.0, 0.05)))
    scene.build(n_envs=1)

    scene.rigid_solver.add_weld_constraint(box_weld_a.base_link_idx, box_weld_b.base_link_idx)

    for _ in range(50):
        scene.step()

    # Deferred import: pulling in the solver package executes module-scope dtype code (gs.qd_float)
    # that only exists after gs.init, which the scene above has now triggered.
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

    entity_island = qd_to_numpy(island_state.entity_island, transpose=True)[0]
    n_islands = int(qd_to_numpy(island_state.n_islands)[0])

    isl = {
        name: entity_island[ent.idx]
        for name, ent in {
            "bottom": box_bottom,
            "top": box_top,
            "weld_a": box_weld_a,
            "weld_b": box_weld_b,
            "alone": box_alone,
        }.items()
    }

    # Every dof-entity is assigned an island; the fixed plane is not.
    assert all(v >= 0 for v in isl.values())
    # Contact coupling groups the stack.
    assert isl["top"] == isl["bottom"]
    # Equality coupling groups the welded pair even though they never touch (the soundness fix).
    assert isl["weld_a"] == isl["weld_b"]
    # The three groups are distinct, and the isolated body stands alone.
    assert len({isl["bottom"], isl["weld_a"], isl["alone"]}) == 3
    assert n_islands == 3

    # Per-island dof list (block-gather map): each free box has 6 dofs, so the stack and welded pair
    # hold 12 dofs each and the lone box 6; together they cover every dof exactly once.
    island_dof_n = qd_to_numpy(island_state.island_dof.n, transpose=True)[0]
    assert sorted(island_dof_n[:n_islands].tolist()) == [6, 12, 12]
    island_dof_start = qd_to_numpy(island_state.island_dof.start, transpose=True)[0]
    dof_id = qd_to_numpy(island_state.dof_id, transpose=True)[0]
    # The lone box's island dof segment is exactly that entity's global dof range.
    k = isl["alone"]
    seg = sorted(dof_id[island_dof_start[k] : island_dof_start[k] + island_dof_n[k]].tolist())
    assert seg == list(range(box_alone.dof_start, box_alone.dof_start + box_alone.n_dofs))

    # Per-island contact list: every detected contact is assigned to exactly one island, and the
    # stack island holds at least its box-on-box contact.
    n_contacts = int(qd_to_numpy(solver.collider._collider_state.n_contacts)[0])
    island_contact_n = qd_to_numpy(island_state.island_contact.n, transpose=True)[0]
    assert int(island_contact_n[:n_islands].sum()) == n_contacts
    assert island_contact_n[isl["bottom"]] >= 1

    # Constraint grouping (post-assembly): every assembled constraint is routed to exactly one island.
    kernel_group_constraints_by_island(
        island_state,
        solver.constraint_solver.constraint_state,
        solver._rigid_global_info,
        solver._static_rigid_sim_config,
    )
    n_constraints = int(qd_to_numpy(solver.constraint_solver.constraint_state.n_constraints)[0])
    island_constraint_n = qd_to_numpy(island_state.island_constraint.n, transpose=True)[0]
    assert int(island_constraint_n[:n_islands].sum()) == n_constraints
    # The welded pair couples two entities, so its island carries the weld's equality constraints.
    assert island_constraint_n[isl["weld_a"]] >= 1


# The island partition must be built from ALL inter-entity coupling - contacts AND equality constraints - so a
# weld coupling two non-contacting entities groups them into one island, and islands ON vs OFF match.
def _run_mixed_scene(use_contact_island, n_steps):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=use_contact_island,
            constraint_solver=gs.constraint_solver.Newton,
            # The per-island Hessian is dense, so it needs the dense jac the sparse_solve=False
            # assembly produces (sparse mode only populates the relevant-dof jac entries).
            sparse_solve=False,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box_bottom = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.05)))
    box_top = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.16)))
    box_weld_a = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, 0.05)))
    box_weld_b = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.3, 0.0, 0.05)))
    box_alone = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(2.0, 0.0, 0.05)))
    scene.build(n_envs=1)
    scene.rigid_solver.add_weld_constraint(box_weld_a.base_link_idx, box_weld_b.base_link_idx)
    for _ in range(n_steps):
        scene.step()
    boxes = (box_bottom, box_top, box_weld_a, box_weld_b, box_alone)
    return np.stack([b.get_pos().cpu().numpy().reshape(-1) for b in boxes])


@pytest.mark.required
def test_island_solve_matches_monolith_end_to_end():
    # The per-island Newton solve must reproduce the monolith physics: the global Hessian is
    # block-diagonal by island, so factoring/solving each island's block is identical to the single
    # global solve. Same scene, islands off vs on -> matching body trajectories.
    pos_monolith = _run_mixed_scene(use_contact_island=False, n_steps=80)
    pos_island = _run_mixed_scene(use_contact_island=True, n_steps=80)
    # FIXME: tolerance reflects the monolith's incremental Cholesky vs the island path's direct
    # rebuild (both exact in theory); over 80 steps of a chaotic stack the lateral position drifts at
    # fp-accumulation level. Per-solve equivalence is asserted rigorously in
    # test_island_newton_solve_matches_dense. Tighten once the iteration uses a shared factorization.
    assert_allclose(pos_island, pos_monolith, tol=5e-3)


@pytest.mark.required
def test_island_partition_tracks_contact_changes():
    # The partitioner is recomputed every step, so it must track a changing contact graph: two bodies
    # falling separately are two islands; once one lands on the other they merge into one.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=False,
        ),
        show_viewer=False,
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

    # Early on, box_upper is still in the air -> the two boxes are separate islands.
    scene.step()
    assert n_islands_now() == 2
    # After box_upper falls and settles onto box_lower, the contact merges them into one island.
    for _ in range(120):
        scene.step()
    assert n_islands_now() == 1


@pytest.mark.required
def test_island_honors_weld_between_noncontacting_entities():
    # box1 is anchored in the air; box2 hangs from a weld at a horizontal offset, never touching
    # box1. With the weld honored, box2 cannot free-fall. With islands enabled the weld must still
    # apply even though the two entities share no contact (different islands without the equality
    # edge).
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
        ),
        show_viewer=False,
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
    scene.build(n_envs=1)

    scene.rigid_solver.add_weld_constraint(box1.base_link_idx, box2.base_link_idx)

    z_start = np.atleast_1d(box2.get_pos().cpu().numpy().reshape(-1, 3)[..., 2])[0]
    for _ in range(120):
        scene.step()
    z_end = np.atleast_1d(box2.get_pos().cpu().numpy().reshape(-1, 3)[..., 2])[0]

    # The weld holds box2 near its start height; a dropped weld would free-fall ~2 m in 2 s.
    assert_allclose(z_end, z_start, tol=0.15)


@pytest.mark.required
def test_island_single_entity_disables_islands_and_settles():
    # A scene with a single DOF-carrying entity has no multi-island structure, so use_contact_island is
    # disabled in computation even though the user opted in (the lone island would be pure overhead and would
    # not fit the cooperative tile). The body must still free-fall and settle via the non-island solve.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            constraint_solver=gs.constraint_solver.Newton,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.0, 0.0, 0.5),
        )
    )
    scene.build(n_envs=1)

    # One dof-entity -> no multi-island structure -> islands auto-disabled regardless of the opt-in.
    assert scene.rigid_solver._use_contact_island is False

    for _ in range(150):
        scene.step()
    z_end = np.atleast_1d(box.get_pos().cpu().numpy().reshape(-1, 3)[..., 2])[0]
    # Settled on the plane at the box half-height.
    assert_allclose(z_end, 0.05, tol=0.02)


@pytest.mark.required
def test_island_and_sparse_solve_compose_on_cpu():
    # The sparse Jacobian representation and per-island solve exploit the block-diagonal Hessian from
    # complementary angles, so on CPU they compose: islands own the per-block factorization while the sparse
    # Jacobian makes the per-iteration products and the constraint-to-island lookup O(nonzeros). The skyline
    # envelope (the global-Hessian factorization) is the part islands supersede, so it is dropped. On GPU the
    # dense tiled path is faster, so sparse is dropped and islands stand alone.
    box_z0 = 0.3
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            sparse_solve=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box_a = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, box_z0)))
    box_b = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, box_z0)))
    scene.build(n_envs=1)
    for _ in range(200):
        scene.step()

    solver = scene.rigid_solver
    sparse_active = solver._static_rigid_sim_config.sparse_solve
    islands_active = solver._use_contact_island
    if gs.backend == gs.cpu:
        # Both block-structure exploiters compose on CPU.
        assert sparse_active and islands_active
    else:
        assert islands_active and not sparse_active

    # The two well-separated boxes (each its own island) settle on the plane: bottom face at z=0, so the
    # 0.1-cube centers rest at z=0.05, with no ground penetration.
    z_a = float(np.atleast_1d(tensor_to_array(box_a.get_pos())[..., 2])[0])
    z_b = float(np.atleast_1d(tensor_to_array(box_b.get_pos())[..., 2])[0])
    assert_allclose(z_a, 0.05, atol=2e-3)
    assert_allclose(z_b, 0.05, atol=2e-3)


def test_hibernation_settles_sleeps_and_wakes_per_island():
    # Hibernation runs on the unified IslandState: each well-separated box is its own island, so once it
    # settles it sleeps independently, and disturbing one island wakes only that island. Hibernation requires
    # performance_mode (field storage), so this test runs only under GS_ENABLE_NDARRAY=0 and skips otherwise.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box_a = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.1)))
    box_b = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, 0.1)))
    scene.build(n_envs=1)
    solver = scene.rigid_solver

    def hibernated():
        # [B, n_entities] -> per-entity flag for env 0; entity 0 is the fixed plane.
        return qd_to_numpy(solver.entities_state.is_hibernated, transpose=True).reshape(-1).astype(bool)

    for _ in range(300):
        scene.step()

    # Both boxes settled on the plane (0.1-cube center rests at z=0.05) and went to sleep.
    flags = hibernated()
    assert flags[box_a.idx] and flags[box_b.idx]
    z_a = float(np.atleast_1d(tensor_to_array(box_a.get_pos())[..., 2])[0])
    z_b = float(np.atleast_1d(tensor_to_array(box_b.get_pos())[..., 2])[0])
    assert_allclose(z_a, 0.05, atol=2e-3)
    assert_allclose(z_b, 0.05, atol=2e-3)

    # An upward control force on box_a wakes only its island; box_b stays asleep.
    box_a.control_dofs_force(np.array([0.0, 0.0, 50.0, 0.0, 0.0, 0.0]))
    scene.step()
    flags = hibernated()
    assert not flags[box_a.idx]
    assert flags[box_b.idx]

    # The woken box responds to the force and lifts off; the sleeping one does not move.
    for _ in range(20):
        scene.step()
    assert float(np.atleast_1d(tensor_to_array(box_a.get_pos())[..., 2])[0]) > 0.1
    assert_allclose(float(np.atleast_1d(tensor_to_array(box_b.get_pos())[..., 2])[0]), 0.05, atol=2e-3)


def test_hibernation_wakeup_on_state_setters():
    # Each user state-setter that mutates a sleeping body must revive it (and only its island) AND leave it
    # dynamically correct afterwards. The motion checks are the real regression: a body woken by set_dofs_position
    # that hangs frozen in mid-air - its gravity cancelled by a sleeping neighbour's stale constraint force leaking
    # into its dofs - would still read "awake". Four well-separated boxes each form their own island, so a setter
    # wakes exactly one; the others must neither move nor perturb it.
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
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box_force = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.1)))
    box_pos = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, 0.1)))
    box_vel = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(2.0, 0.0, 0.1)))
    box_qpos = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(3.0, 0.0, 0.1)))
    scene.build(n_envs=1)
    solver = scene.rigid_solver

    def asleep(entity):
        return bool(qd_to_numpy(solver.entities_state.is_hibernated, transpose=True).reshape(-1)[entity.idx])

    def z_of(entity):
        return float(np.atleast_1d(tensor_to_array(entity.get_pos())[..., 2])[0])

    # Free-fall from rest over n steps with semi-implicit Euler: each box stays its own column above empty floor,
    # so a woken body must drop by this much. A frozen (force-cancelled) body drops ~0, a healthy one ~free-fall.
    n_fall = 8
    free_fall_drop = 0.5 * G * (n_fall * DT) ** 2

    for _ in range(300):
        scene.step()
    assert asleep(box_force) and asleep(box_pos) and asleep(box_vel) and asleep(box_qpos)

    # apply_links_external_force wakes box_force's island; a sustained upward thrust then lifts it off the floor,
    # confirming the force reaches the dynamics rather than being absorbed by a still-sleeping body.
    z_force0 = z_of(box_force)
    for _ in range(6):
        solver.apply_links_external_force(np.array([[0.0, 0.0, 40.0]]), links_idx=[box_force.base_link_idx])
        scene.step()
    assert not asleep(box_force)
    assert z_of(box_force) > z_force0 + 0.02
    assert asleep(box_pos) and asleep(box_vel) and asleep(box_qpos)

    # set_dofs_position lifts box_pos into the air; with its island awake and isolated it must free-fall, not hang.
    box_pos.set_dofs_position(np.array([1.0, 0.0, 0.5, 0.0, 0.0, 0.0]))
    assert not asleep(box_pos)
    z_pos0 = z_of(box_pos)
    for _ in range(n_fall):
        scene.step()
    assert_allclose(z_pos0 - z_of(box_pos), free_fall_drop, rtol=0.2)
    assert asleep(box_vel) and asleep(box_qpos)

    # set_dofs_velocity gives box_vel an upward velocity; it must lift off, proving the velocity took effect.
    box_vel.set_dofs_velocity(np.array([0.0, 0.0, 2.0, 0.0, 0.0, 0.0]))
    assert not asleep(box_vel)
    z_vel0 = z_of(box_vel)
    for _ in range(5):
        scene.step()
    assert z_of(box_vel) > z_vel0 + 0.05
    assert asleep(box_qpos)

    # set_qpos teleports box_qpos up; it must then free-fall from the new height.
    box_qpos.set_qpos(np.array([3.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0]))
    assert not asleep(box_qpos)
    z_qpos0 = z_of(box_qpos)
    for _ in range(n_fall):
        scene.step()
    assert_allclose(z_qpos0 - z_of(box_qpos), free_fall_drop, rtol=0.2)


def test_hibernation_wakeup_on_collision():
    # An awake body colliding with a sleeping body must wake it so it responds dynamically instead of acting as an
    # immovable obstacle (or being tunnelled through). This needs both the broad-phase sort-buffer refresh of awake
    # geoms - so the awake-vs-sleeping contact is detected at all - and the wake-on-contact pass.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=True,
        ),
        show_viewer=False,
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
    rest_x0 = float(np.atleast_1d(tensor_to_array(box_rest.get_pos())[..., 0])[0])

    # Ram box_hit leftwards into the sleeping box_rest at a moderate speed (no tunnelling).
    box_hit.set_dofs_velocity(np.array([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    for _ in range(30):
        scene.step()

    # The struck sleeper woke and was knocked; the striker was stopped by it (did not pass through).
    assert not asleep(box_rest)
    rest_x1 = float(np.atleast_1d(tensor_to_array(box_rest.get_pos())[..., 0])[0])
    hit_x1 = float(np.atleast_1d(tensor_to_array(box_hit.get_pos())[..., 0])[0])
    assert rest_x1 < rest_x0 - 1e-3
    assert hit_x1 > rest_x1


def test_hibernation_wakes_whole_island_through_daisy_chain():
    # Two coupled bodies sleep as ONE island, chained together so the partition survives across steps. Disturbing
    # just one of them (an external force on box_a) must wake the WHOLE island via the daisy chain: waking only the
    # directly addressed body would leave its coupled partner frozen, so a constraint would be solved against a
    # sleeping body. The pair is joined by a weld (a stable coupling that reliably sleeps, unlike a contact stack
    # whose micro-settling keeps it awake); a separated third box is its own island and stays asleep.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, -9.8),
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=True,
        ),
        show_viewer=False,
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

    # The welded pair and the lone box all settle and go to sleep.
    for _ in range(400):
        scene.step()
    assert asleep(box_a) and asleep(box_b) and asleep(box_far)

    # A shove on box_a must wake both members of its weld-coupled island.
    solver.apply_links_external_force(np.array([[20.0, 0.0, 0.0]]), links_idx=[box_a.base_link_idx])
    scene.step()
    assert not asleep(box_a)
    assert not asleep(box_b)
    # The separated box is a different island and stays asleep.
    assert asleep(box_far)
