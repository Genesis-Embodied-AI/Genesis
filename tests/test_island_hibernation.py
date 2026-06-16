import numpy as np
import pytest

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.utils.misc import qd_to_numpy

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
def test_island_and_sparse_solve_route_to_one_block_solver():
    # sparse-skyline and per-island both exploit the block-diagonal Hessian, so requesting both must NOT error
    # and must NOT run both at once. The solver routes to the backend's winner: sparse on CPU (it supersedes
    # islands), islands on GPU (sparse is dropped there).
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
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.1)))
    scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(1.0, 0.0, 0.1)))
    scene.build(n_envs=1)
    for _ in range(10):
        scene.step()

    solver = scene.rigid_solver
    sparse_active = solver._static_rigid_sim_config.sparse_solve
    islands_active = solver._use_contact_island
    # Exactly one block-structure exploiter is active - never both, never an error.
    assert sparse_active != islands_active
    if gs.backend == gs.cpu:
        assert sparse_active and not islands_active
    else:
        assert islands_active and not sparse_active
