import xml.etree.ElementTree as ET
from itertools import product

import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .utils import assert_allclose, assert_equal


@pytest.fixture
def multi_free_body_path(tmp_path):
    # A single MJCF entity holding several free bodies (b0, b1, b2) is common. b1 carries a hinge child b1c, so a
    # kinematic edge must keep the child in its parent's island. The ground plane lives in the entity's worldbody: the
    # entity owns a 0-dof static link that every free body contacts, yet each free body must remain its own island -
    # the shared static link must not couple them.
    mjcf = ET.Element("mujoco", model="multi_free_body")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="plane", size="5 5 0.01")
    b0 = ET.SubElement(worldbody, "body", name="b0", pos="0.0 0.0 0.3")
    ET.SubElement(b0, "freejoint")
    ET.SubElement(b0, "geom", type="box", size="0.05 0.05 0.05")
    b1 = ET.SubElement(worldbody, "body", name="b1", pos="0.5 0.0 0.3")
    ET.SubElement(b1, "freejoint")
    ET.SubElement(b1, "geom", type="box", size="0.05 0.05 0.05")
    b1c = ET.SubElement(b1, "body", name="b1c", pos="0.0 0.0 0.12")
    ET.SubElement(b1c, "joint", type="hinge", axis="0 0 1")
    ET.SubElement(b1c, "geom", type="box", size="0.05 0.05 0.05")
    b2 = ET.SubElement(worldbody, "body", name="b2", pos="1.0 0.0 0.3")
    ET.SubElement(b2, "freejoint")
    ET.SubElement(b2, "geom", type="box", size="0.05 0.05 0.05")
    file_path = str(tmp_path / "multi_free_body.xml")
    ET.ElementTree(mjcf).write(file_path, encoding="utf-8", xml_declaration=True)
    return file_path


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_partition_logics(show_viewer, n_envs, multi_free_body_path, monkeypatch):
    # The welded pair never touches, so only the equality edge couples them: without it the partition would split them
    # and the weld would be solved across two islands. A fixed body carries no dofs and joins no island. The
    # multi-free-body MJCF entity (offset clear of the boxes) is a single Genesis entity that must split into one island
    # per free-body subtree, never one dense block - its hinge child stays in its parent's island via a kinematic edge.
    #
    # This scene is small and fits-shared, so in production the GPU solve runs whole-env and never builds the island
    # partition this test asserts (enable_per_island_solve is False without hibernation). Force the per-island path on
    # so the partition is built; this patch only exists to keep this partition-structure test backend-agnostic.
    from genesis.utils.array_class import RigidSimStaticConfig

    _orig_static_config_init = RigidSimStaticConfig.__init__

    def _force_per_island_solve(self, *args, **kwargs):
        if kwargs.get("use_contact_island"):
            kwargs["enable_per_island_solve"] = True
        _orig_static_config_init(self, *args, **kwargs)

    monkeypatch.setattr(RigidSimStaticConfig, "__init__", _force_per_island_solve)

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
    multibody = scene.add_entity(
        gs.morphs.MJCF(
            file=multi_free_body_path,
            pos=(0.0, 5.0, 0.0),
        )
    )
    scene.build(n_envs=n_envs)

    scene.rigid_solver.add_weld_constraint(box_weld_a.base_link_idx, box_weld_b.base_link_idx)

    for _ in range(45):
        scene.step()

    # The partition is rebuilt inside every step; inspect the one the solver actually used this step.
    solver = scene.rigid_solver
    island_state = solver.constraint_solver.island_state

    island_idx = qd_to_numpy(island_state.links_island_idx)
    island_of = {
        name: island_idx[entity.base_link_idx]
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

    # The multi-free-body entity splits into one island per free body (b0, b1, b2), each distinct from the others and
    # from the box islands; the entity's static ground link joins no island. Its hinge child b1c lands in its parent
    # b1's island via the kinematic edge.
    multibody_bases = [link for link in multibody.links if link.parent_idx == -1 and link.n_dofs > 0]
    assert len(multibody_bases) == 3
    base_islands = [island_idx[link.idx] for link in multibody_bases]
    assert all((isl >= 0).all() for isl in base_islands)
    for i, isl_a in enumerate(base_islands):
        for isl_b in base_islands[i + 1 :]:
            assert (isl_a != isl_b).all()
        for box_island in island_of.values():
            assert (isl_a != box_island).all()
    b1c = next(link for link in multibody.links if link.parent_idx != -1)
    assert_equal(island_idx[b1c.idx], island_idx[b1c.parent_idx])

    # Three box islands plus three free-body islands.
    assert_equal(qd_to_numpy(island_state.n_islands), 6)

    # Per env: each free box has 6 dofs (stack and welded pair hold 12 each, lone box 6; the free bodies hold 6, 7 with
    # the hinge child, and 6); per-island contact and constraint counts sum back to the env total; and the lone island
    # holds exactly the lone box's dofs.
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
        assert sorted(island_dof_n[:n, i_env].tolist()) == [6, 6, 6, 7, 12, 12]
        assert island_contact_n[:n, i_env].sum() == n_contacts[i_env]
        assert island_constraint_n[:n, i_env].sum() == n_constraints[i_env]
        assert island_contact_n[island_of["bottom"][i_env], i_env] >= 1
        assert island_constraint_n[island_of["weld_a"][i_env], i_env] >= 1
        k = island_of["alone"][i_env]
        seg = dof_id[island_dof_start[k, i_env] : island_dof_start[k, i_env] + island_dof_n[k, i_env], i_env]
        assert sorted(seg.tolist()) == alone_dofs

    # The per-component solve keeps the free bodies stable: they settle on the plane (half-extent 0.05) rather than
    # exploding or sinking through it.
    free_body_z = np.stack(
        [np.atleast_1d(tensor_to_array(link.get_pos())[..., 2]) for link in multibody.links if link.n_dofs > 0]
    )
    assert ((free_body_z > 0.0) & (free_body_z < 0.5)).all()


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_partition_track_changes(show_viewer, n_envs, monkeypatch):
    # The partition is rebuilt every step, so it must track contacts forming (merge) and breaking (split).
    #
    # This scene is small and fits-shared, so in production the GPU solve runs whole-env and never builds the island
    # partition this test asserts (enable_per_island_solve is False without hibernation). Force the per-island path on
    # so the partition is built; this patch only exists to keep this partition-structure test backend-agnostic.
    from genesis.utils.array_class import RigidSimStaticConfig

    _orig_static_config_init = RigidSimStaticConfig.__init__

    def _force_per_island_solve(self, *args, **kwargs):
        if kwargs.get("use_contact_island"):
            kwargs["enable_per_island_solve"] = True
        _orig_static_config_init(self, *args, **kwargs)

    monkeypatch.setattr(RigidSimStaticConfig, "__init__", _force_per_island_solve)

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
@pytest.mark.parametrize("backend", [gs.gpu])
def test_island_monolith_seed_oversaturated(show_viewer, monkeypatch):
    # The GPU island monolith leaves its initial factor + gradient to func_solve_init, except when its body self-inits
    # them per-env - which it only does when the cooperative kernels are disabled. With them enabled, func_solve_init
    # must seed: for a shared-fitting Hessian at an env count that oversaturates the GPU, neither the fused nor the
    # per-island seed branch fires, so the fallback factor+gradient must run, else Mgrad stays stale and the solve
    # makes no progress (the boxes fall through the floor). Faking GPU saturation (get_gpu_core_count -> 1) reaches the
    # oversaturated path at 2 envs instead of the thousands a real GPU would need.
    import genesis.engine.solvers.rigid.rigid_solver as rigid_solver
    from genesis.utils.array_class import RigidSimStaticConfig

    monkeypatch.setattr(rigid_solver, "get_gpu_core_count", lambda: 1)
    # The gap is in the monolith arm's init path, so pin the arm (otherwise the autotuner might pick the decomposed
    # arm, which always seeds and would hide the regression).
    _orig_static_config_init = RigidSimStaticConfig.__init__

    def _force_monolith(self, *args, **kwargs):
        kwargs["prefer_decomposed_solver"] = 0
        _orig_static_config_init(self, *args, **kwargs)

    monkeypatch.setattr(RigidSimStaticConfig, "__init__", _force_monolith)

    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    # Four spaced free boxes: 24 dofs (>= 16, so the cooperative kernels engage) that fit the shared tile, and four
    # independent islands - exactly the shared-fitting multi-island case the seed branches gate out.
    boxes = [
        scene.add_entity(
            gs.morphs.Box(
                size=(0.1, 0.1, 0.1),
                pos=(0.4 * i, 0.0, 0.2),
            )
        )
        for i in range(4)
    ]
    scene.build(n_envs=2)

    cfg = scene.rigid_solver._static_rigid_sim_config
    # Guard against the test silently ceasing to exercise the gap (e.g. if the saturation heuristic changes).
    assert cfg.enable_cooperative_constraint_kernels
    assert not cfg.enable_fused_factor_solve_init
    assert not cfg.enable_per_island_solve

    for _ in range(150):
        scene.step()

    z = np.stack([tensor_to_array(box.get_pos())[..., 2] for box in boxes])
    vel = tensor_to_array(scene.rigid_solver.get_dofs_velocity())
    assert not np.isnan(z).any()
    # The boxes rest on the floor at half their size; a stale seed would never apply the contact response.
    assert_allclose(z, 0.05, tol=5e-3)
    assert np.abs(vel).max() < 0.1


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
def test_hibernation_with_pruning(show_viewer, n_envs):
    # A convexified duck is a compound body whose many ground contacts pile into one (duck, plane) link-pair bucket,
    # so link-pair pruning collapses them into a logical permutation in contact_sort_idx. Islands read contacts through
    # that permutation and hibernation advects the resting contacts while the body sleeps, so pruning, islands and
    # hibernation all run together. contact_pruning_tolerance is set explicitly to keep pruning on alongside islands.
    # The duck must reach the plane without tunnelling, hibernate, and then stay frozen in place.
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1.0 / 100.0,
        ),
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=True,
            contact_pruning_tolerance=0.02,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    duck = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/duck.obj",
            scale=0.02,
            pos=(0.0, 0.0, 0.1),
            euler=(90.0, 0.0, 0.0),
        ),
    )
    scene.build(n_envs=n_envs)
    solver = scene.rigid_solver

    def asleep():
        return qd_to_numpy(solver.entities_state.is_hibernated, duck.idx).all()

    for _ in range(200):
        scene.step()
        assert (tensor_to_array(duck.get_pos())[..., 2] > -0.05).all()
        if asleep():
            break
    assert asleep()
    z_rest = tensor_to_array(duck.get_pos())[..., 2]

    for _ in range(100):
        scene.step()
    assert asleep()
    assert_allclose(duck.get_pos()[..., 2], z_rest, atol=1e-5)


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
@pytest.mark.parametrize("broadphase_traversal", [None, gs.broadphase_traversal.ALL_VS_ALL], ids=["sap", "allvsall"])
def test_hibernation_wakes_on_collision(show_viewer, n_envs, broadphase_traversal, multi_free_body_path):
    # An awake body striking a sleeping one must wake it so it responds instead of acting as an immovable obstacle.
    # This needs the broad-phase sort-buffer refresh of awake geoms (so the contact is detected) and the wake-on-contact
    # pass. The multi-free-body entity (offset clear of the boxes) checks per-component hibernation: its free bodies
    # sleep independently and disturbing one wakes only its island, which needs the wake/daisy chain to act per link
    # rather than per entity. ALL_VS_ALL exercises hibernation under the non-default traversal: it advects and skips
    # hibernated-fixed pairs exactly like SAP, and the hibernated-vs-hibernated pairs it traverses instead of skipping
    # are inert (both bodies frozen).
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_contact_island=True,
            use_hibernation=True,
            broadphase_traversal=broadphase_traversal,
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
    multibody = scene.add_entity(
        gs.morphs.MJCF(
            file=multi_free_body_path,
            pos=(0.0, 5.0, 0.0),
        )
    )
    scene.build(n_envs=n_envs)

    solver = scene.rigid_solver

    def asleep(entity):
        return qd_to_numpy(solver.entities_state.is_hibernated, entity.idx).all()

    def link_asleep(link):
        return qd_to_numpy(solver.links_state.is_hibernated, link.idx).all()

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

    # The undisturbed entity's free bodies all settled and slept independently; disturbing one wakes only its island.
    multibody_bases = [link for link in multibody.links if link.parent_idx == -1 and link.n_dofs > 0]
    assert all(link_asleep(link) for link in multibody_bases)
    disturbed = multibody_bases[0]
    solver.set_dofs_velocity(
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        list(range(disturbed.dof_start, disturbed.dof_start + disturbed.n_dofs)),
    )
    assert not link_asleep(disturbed)
    assert all(link_asleep(link) for link in multibody_bases[1:])


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
