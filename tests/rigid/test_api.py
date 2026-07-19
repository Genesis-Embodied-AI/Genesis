from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pytest
import torch

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.states.solvers import RigidSolverState
from genesis.utils.misc import qd_to_torch

from ..utils import (
    assert_allclose,
)


@pytest.mark.slow  # ~200s
@pytest.mark.parametrize(
    "n_envs, batched, backend",
    [
        (0, False, gs.cpu),
        (0, False, gs.gpu),
        (3, False, gs.cpu),
        # (3, True, gs.cpu),  # FIXME: Must refactor the unit test to support batching
    ],
)
def test_data_accessor(n_envs, batched, tol):
    # Create and build the scene
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            batch_dofs_info=batched,
            batch_joints_info=batched,
            batch_links_info=batched,
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
    gs_link = gs_robot.get_link("RR_thigh")
    gs_geom = gs_link.geoms[0]
    gs_vgeom = gs_link.vgeoms[0]
    scene.build(n_envs=n_envs)
    gs_s = scene.sim.rigid_solver

    # Initialize the simulation
    np.random.seed(0)
    dof_bounds = gs_s.dyn_info.dofs.limit.to_numpy()
    dof_bounds[..., :2, :] = np.array((-1.0, 1.0))
    dof_bounds[..., 2, :] = np.array((0.7, 1.0))
    dof_bounds[..., 3:6, :] = np.array((-np.pi / 2, np.pi / 2))
    for i in range(max(n_envs, 1)):
        qpos = dof_bounds[:, 0] + (dof_bounds[:, 1] - dof_bounds[:, 0]) * np.random.rand(gs_robot.n_dofs)
        gs_robot.set_dofs_position(qpos, envs_idx=([i] if n_envs else None))

    # Simulate for a while, until they collide with something
    for _ in range(400):
        scene.step()

        gs_n_contacts = gs_s.collider._collider_state.n_contacts.to_numpy()
        assert len(gs_n_contacts) == max(n_envs, 1)
        for as_tensor in (False, True):
            for to_torch in (False, True):
                contacts_info = gs_s.collider.get_contacts(as_tensor, to_torch)
                for value in contacts_info.values():
                    if n_envs > 0:
                        assert n_envs == len(value)
                    else:
                        assert gs_n_contacts[0] == len(value)
                        value = value[None] if as_tensor else (value,)

                    for i_b in range(n_envs):
                        n_contacts = gs_n_contacts[i_b]
                        if as_tensor:
                            assert isinstance(value, torch.Tensor if to_torch else np.ndarray)
                            if value.dtype in (gs.tc_int, gs.np_int):
                                assert (value[i_b, :n_contacts] != -1).all()
                                assert (value[i_b, n_contacts:] == -1).all()
                            else:
                                assert_allclose(value[i_b, n_contacts:], 0.0, tol=0)
                        else:
                            assert isinstance(value, (list, tuple))
                            assert value[i_b].shape[0] == n_contacts
                            if value[i_b].dtype in (gs.tc_int, gs.np_int):
                                assert (value[i_b] != -1).all()

        if (gs_n_contacts > 0).all():
            break
    else:
        assert False
    gs_s._func_forward_dynamics()
    gs_s._func_constraint_force()

    # Make sure that all the robots ends up in the different state
    qposs = gs_robot.get_qpos()
    for i in range(n_envs - 1):
        with np.testing.assert_raises(AssertionError):
            assert_allclose(qposs[i], qposs[i + 1], tol=tol)

    # Check attribute getters / setters.
    # First, without any any row or column masking:
    # * Call 'Get' -> Call 'Set' with random value -> Call 'Get'
    # * Compare first 'Get' ouput with Quadrants value
    # Then, for any possible combinations of row and column masking:
    # * Call 'Get' -> Call 'Set' with 'Get' output -> Call 'Get'
    # * Compare first 'Get' output with last 'Get' output
    # * Compare last 'Get' output with corresponding slice of non-masking 'Get' output
    def get_all_supported_masks(i, max_length):
        if max_length <= 0 or i > max_length - 1:
            return (None,)
        if i == max_length - 1:
            return (
                i,
                [i],
                slice(i, i + 1),
                range(i, i + 1),
                np.array([i], dtype=np.int32),
                torch.tensor([i], dtype=torch.int64),
                torch.tensor([i], dtype=gs.tc_int, device=gs.device),
            )
        return (
            [i, i + 1],
            slice(i, i + 2),
            range(i, i + 2),
            np.array([i, i + 1], dtype=np.int32),
            torch.tensor([i, i + 1], dtype=torch.int64),
            torch.tensor([i, i + 1], dtype=gs.tc_int, device=gs.device),
        )

    def must_cast(value, dtype):
        return not (
            isinstance(value, torch.Tensor)
            and value.is_contiguous()
            and value.dtype == dtype
            and value.device == gs.device
        )

    for arg1_max, arg2_max, getter_or_spec, setter, qd_data in (
        # SOLVER
        (gs_s.n_links, n_envs, gs_s.get_links_pos, None, gs_s.dyn_state.links.pos),
        (gs_s.n_links, n_envs, gs_s.get_links_quat, None, gs_s.dyn_state.links.quat),
        (gs_s.n_links, n_envs, gs_s.get_links_vel, None, None),
        (gs_s.n_links, n_envs, gs_s.get_links_ang, None, gs_s.dyn_state.links.cd_ang),
        (gs_s.n_links, n_envs, gs_s.get_links_acc, None, None),
        (gs_s.n_links, n_envs, gs_s.get_links_root_COM, None, gs_s.dyn_state.links.root_COM),
        (gs_s.n_links, n_envs, gs_s.get_links_mass_shift, gs_s.set_links_mass_shift, gs_s.dyn_state.links.mass_shift),
        (gs_s.n_links, n_envs, gs_s.get_links_COM_shift, gs_s.set_links_COM_shift, gs_s.dyn_state.links.i_pos_shift),
        (
            gs_s.n_links,
            -1,
            gs_s.get_links_inertial_mass,
            gs_s.set_links_inertial_mass,
            gs_s.dyn_info.links.inertial_mass,
        ),
        (gs_s.n_links, -1, gs_s.get_links_invweight, None, gs_s.dyn_info.links.invweight),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_control_force, gs_s.control_dofs_force, None),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_force, None, gs_s.dyn_state.dofs.force),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_velocity, gs_s.set_dofs_velocity, gs_s.dyn_state.dofs.vel),
        (gs_s.n_dofs, n_envs, gs_s.get_dofs_position, gs_s.set_dofs_position, gs_s.dyn_state.dofs.pos),
        (gs_s.n_dofs, -1, gs_s.get_dofs_force_range, gs_s.set_dofs_force_range, gs_s.dyn_info.dofs.force_range),
        (gs_s.n_dofs, -1, gs_s.get_dofs_limit, gs_s.set_dofs_limit, gs_s.dyn_info.dofs.limit),
        (gs_s.n_dofs, -1, gs_s.get_dofs_stiffness, gs_s.set_dofs_stiffness, gs_s.dyn_info.dofs.stiffness),
        (gs_s.n_dofs, -1, gs_s.get_dofs_invweight, None, gs_s.dyn_info.dofs.invweight),
        (gs_s.n_dofs, -1, gs_s.get_dofs_armature, gs_s.set_dofs_armature, gs_s.dyn_info.dofs.armature),
        (gs_s.n_dofs, -1, gs_s.get_dofs_damping, gs_s.set_dofs_damping, gs_s.dyn_info.dofs.damping),
        (gs_s.n_dofs, -1, gs_s.get_dofs_frictionloss, gs_s.set_dofs_frictionloss, gs_s.dyn_info.dofs.frictionloss),
        (gs_s.n_dofs, -1, gs_s.get_dofs_kp, gs_s.set_dofs_kp, gs_s.dyn_info.dofs.act_gain),
        (gs_s.n_dofs, -1, gs_s.get_dofs_kv, gs_s.set_dofs_kv, None),
        (gs_s.n_dofs, -1, gs_s.get_dofs_act_bias, gs_s.set_dofs_act_bias, gs_s.dyn_info.dofs.act_bias),
        (gs_s.n_dofs, -1, gs_s.get_dofs_act_gain, gs_s.set_dofs_act_gain, gs_s.dyn_info.dofs.act_gain),
        (gs_s.n_geoms, n_envs, gs_s.get_geoms_pos, None, gs_s.dyn_state.geoms.pos),
        (gs_s.n_geoms, n_envs, gs_s.get_geoms_quat, None, gs_s.dyn_state.geoms.quat),
        (
            gs_s.n_geoms,
            n_envs,
            gs_s.get_geoms_friction_ratio,
            gs_s.set_geoms_friction_ratio,
            gs_s.dyn_state.geoms.friction_ratio,
        ),
        (gs_s.n_geoms, -1, gs_s.get_geoms_friction, gs_s.set_geoms_friction, gs_s.dyn_info.geoms.friction),
        (gs_s.n_qs, n_envs, gs_s.get_qpos, gs_s.set_qpos, gs_s.qpos),
        # ROBOT
        (gs_robot.n_links, n_envs, gs_robot.get_links_pos, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_quat, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_vel, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_ang, None, None),
        (gs_robot.n_links, n_envs, gs_robot.get_links_acc, None, None),
        (gs_robot.n_links, n_envs, (), gs_robot.set_mass_shift, None),
        (gs_robot.n_links, n_envs, (3,), gs_robot.set_COM_shift, None),
        (gs_robot.n_links, n_envs, (), gs_robot.set_friction_ratio, None),
        (gs_robot.n_links, -1, gs_robot.get_links_inertial_mass, gs_robot.set_links_inertial_mass, None),
        (gs_robot.n_links, -1, gs_robot.get_links_invweight, None, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_control_force, None, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_force, None, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_velocity, gs_robot.set_dofs_velocity, None),
        (gs_robot.n_dofs, n_envs, gs_robot.get_dofs_position, gs_robot.set_dofs_position, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_force_range, gs_robot.set_dofs_force_range, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_limit, None, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_stiffness, None, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_invweight, None, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_armature, None, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_damping, None, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_frictionloss, gs_robot.set_dofs_frictionloss, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_kp, gs_robot.set_dofs_kp, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_kv, gs_robot.set_dofs_kv, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_act_bias, gs_robot.set_dofs_act_bias, None),
        (gs_robot.n_dofs, -1, gs_robot.get_dofs_act_gain, gs_robot.set_dofs_act_gain, None),
        (gs_robot.n_qs, n_envs, gs_robot.get_qpos, gs_robot.set_qpos, None),
        (-1, n_envs, gs_robot.get_mass_mat, None, None),
        (-1, n_envs, gs_robot.get_links_net_contact_force, None, None),
        (-1, n_envs, gs_robot.get_pos, gs_robot.set_pos, None),
        (-1, n_envs, gs_robot.get_quat, gs_robot.set_quat, None),
        (-1, -1, gs_robot.get_mass, gs_robot.set_mass, None),
        (-1, -1, gs_robot.get_verts, None, None),
        (-1, -1, gs_robot.get_AABB, None, None),
        (-1, -1, gs_robot.get_vAABB, None, None),
        # LINK
        (-1, -1, gs_link.get_pos, None, None),
        (-1, -1, gs_link.get_quat, None, None),
        (-1, -1, gs_link.get_mass, gs_link.set_mass, None),
        (-1, -1, gs_link.get_verts, None, None),
        (-1, -1, gs_link.get_AABB, None, None),
        (-1, -1, gs_link.get_vAABB, None, None),
        # GEOM
        (-1, -1, gs_geom.get_pos, None, None),
        (-1, -1, gs_geom.get_quat, None, None),
        (-1, -1, gs_geom.get_verts, None, None),
        (-1, -1, gs_geom.get_AABB, None, None),
        # VGEOM
        (-1, -1, gs_vgeom.get_pos, None, None),
        (-1, -1, gs_vgeom.get_quat, None, None),
        (-1, -1, gs_vgeom.get_vAABB, None, None),
    ):
        getter, spec = (getter_or_spec, None) if callable(getter_or_spec) else (None, getter_or_spec)

        # Restore PD consistency before each iteration (act_gain/act_bias setters may have broken it)
        gs_s.set_dofs_kp(0.0)
        gs_s.set_dofs_kv(0.0)

        # Check getter and setter without row or column masking
        if getter is not None:
            datas = deepcopy(getter())
            is_tuple = isinstance(datas, (tuple, list))
            if arg1_max > 0:
                assert_allclose(getter(range(arg1_max)), datas, tol=tol)
        else:
            batch_shape = []
            if arg2_max > 0:
                batch_shape.append(arg2_max)
            if arg1_max > 0:
                batch_shape.append(arg1_max)
            is_tuple = spec and isinstance(spec[0], (tuple, list))
            if is_tuple:
                datas = [torch.ones((*batch_shape, *shape)) for shape in spec]
            else:
                datas = torch.ones((*batch_shape, *spec))
        if qd_data is not None:
            true = qd_to_torch(qd_data)
            qd_ndim = getattr(qd_data, "ndim", len(getattr(qd_data, "element_shape", ())))
            true = true.movedim(true.ndim - qd_ndim - 1, 0)
            if is_tuple:
                true = torch.unbind(true, dim=-1)
                true = [val.reshape(data.shape) for data, val in zip(datas, true)]
            else:
                true = true.reshape(datas.shape)
            assert_allclose(datas, true, tol=tol)
        if setter is not None:
            if is_tuple:
                datas = [torch.as_tensor(val) for val in datas]
            else:
                datas = torch.as_tensor(datas, dtype=gs.tc_float)
            datas_tp = datas if is_tuple else (datas,)
            if getter is not None:
                # Randomly sample new data that are strictly positive and normalized,
                # as this may be required for some setters (mass, quaternion, ...).
                for val in datas_tp:
                    val[()] = torch.abs(torch.randn(val.shape, dtype=gs.tc_float, device=gs.device)) + gs.EPS
                    val /= torch.linalg.norm(val, dim=-1, keepdims=True)
            setter(*datas_tp)
            if getter is not None:
                assert_allclose(getter(), datas, tol=tol)

        # Early return if neither rows or columns can be masked
        if not (arg1_max > 0 or arg2_max > 0):
            continue

        # Check getter and setter for all possible combinations of row and column masking
        for i in range(arg1_max) if arg1_max > 0 else (None,):
            if i is not None:
                mask_i = [i, i + 1] if i < arg1_max - 1 else [i]
            for arg1 in get_all_supported_masks(i, arg1_max):
                for j in range(max(arg2_max, 1)) if arg2_max >= 0 else (None,):
                    if j is not None:
                        mask_j = [j, j + 1] if j < arg2_max - 1 else [j]
                    for arg2 in get_all_supported_masks(j, arg2_max):
                        if arg1 is None and arg2 is not None:
                            if getter is not None:
                                data = deepcopy(getter(arg2))
                            else:
                                if is_tuple:
                                    data = [torch.ones((len(mask_j), *shape)) for shape in spec]
                                else:
                                    data = torch.ones((len(mask_j), *spec))
                            if setter is not None:
                                setter(data, arg2)
                            if n_envs:
                                if is_tuple:
                                    data_ = [val[mask_j] for val in datas]
                                else:
                                    data_ = datas[mask_j]
                            else:
                                data_ = datas
                        elif arg1 is not None and arg2 is None:
                            if getter is not None:
                                data = deepcopy(getter(arg1))
                            else:
                                if is_tuple:
                                    data = [torch.ones((len(mask_i), *shape)) for shape in spec]
                                else:
                                    data = torch.ones((len(mask_i), *spec))
                            if setter is not None:
                                if is_tuple:
                                    setter(*data, arg1)
                                else:
                                    setter(data, arg1)
                            if is_tuple:
                                data_ = [val[mask_i] for val in datas]
                            else:
                                data_ = datas[mask_i]
                        else:
                            if getter is not None:
                                data = deepcopy(getter(arg1, arg2))
                            else:
                                if is_tuple:
                                    data = [torch.ones((len(mask_j), len(mask_i), *shape)) for shape in spec]
                                else:
                                    data = torch.ones((len(mask_j), len(mask_i), *spec))
                            if setter is not None:
                                setter(data, arg1, arg2)
                            if is_tuple:
                                data_ = [val[mask_j, :][:, mask_i] for val in datas]
                            else:
                                data_ = datas[mask_j, :][:, mask_i]
                        # FIXME: Not sure why tolerance must be increased for tests to pass
                        assert_allclose(data_, data, tol=(5.0 * tol))

    for dofs_idx in (*get_all_supported_masks(0, gs_s.n_dofs), None):
        for envs_idx in (*(get_all_supported_masks(0, gs_s.n_dofs) if n_envs > 0 else ()), None):
            dofs_pos = gs_s.get_dofs_position(dofs_idx, envs_idx)
            dofs_vel = gs_s.get_dofs_velocity(dofs_idx, envs_idx)
            gs_s.control_dofs_position(dofs_pos, dofs_idx, envs_idx)
            gs_s.control_dofs_velocity(dofs_vel, dofs_idx, envs_idx)

    # Must be tested independently because of non-trival return type
    gs_robot.get_contacts()


@pytest.mark.required
@pytest.mark.parametrize("enable_mujoco_compatibility", [True, False])
def test_getter_vs_state_post_step_consistency(enable_mujoco_compatibility):
    DT = 0.01
    GRAVITY = 10.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=DT,
            gravity=(0.0, 0.0, GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_mujoco_compatibility=enable_mujoco_compatibility,
        ),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 0.0),
        )
    )
    (box_link,) = box.links
    scene.build()

    scene.step()
    dof_vel = scene.rigid_solver.get_dofs_velocity()
    assert_allclose(dof_vel[:3], (0.0, 0.0, GRAVITY * DT), atol=gs.EPS)
    vel = box_link.get_vel()
    with pytest.raises(AssertionError) if enable_mujoco_compatibility else nullcontext():
        assert_allclose(dof_vel[:3], vel, atol=gs.EPS)
    dof_pos = scene.rigid_solver.get_qpos()
    assert_allclose(dof_pos[:3], (0.0, 0.0, GRAVITY * DT**2), atol=gs.EPS)
    pos = box_link.get_pos()
    with pytest.raises(AssertionError) if enable_mujoco_compatibility else nullcontext():
        assert_allclose(dof_pos[:3], pos, atol=gs.EPS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_extended_broadcasting():
    scene = gs.Scene(
        show_viewer=False,
    )
    for i in range(4):
        scene.add_entity(
            gs.morphs.Box(
                size=(1.0, 1.0, 1.0),
                pos=(0.0, 0.0, i),
            )
        )
    scene.build(n_envs=2)

    envs_idx = torch.tensor([0, 1], dtype=gs.tc_int, device=gs.device)
    for entity in scene.entities:
        entity.zero_all_dofs_velocity(envs_idx)
    assert_allclose(entity.get_dofs_velocity(), 0.0, tol=gs.EPS)
    entity.set_dofs_velocity(1.0)
    assert_allclose(entity.get_dofs_velocity(), 1.0, tol=gs.EPS)
    entity.set_dofs_velocity((1.0, 2.0))
    assert_allclose(entity.get_dofs_velocity(), np.array([(1.0,) * 6, (2.0,) * 6]), tol=gs.EPS)
    entity.set_dofs_velocity((3.0,) * 6)
    assert_allclose(entity.get_dofs_velocity(), 3.0, tol=gs.EPS)
    entity.zero_all_dofs_velocity(torch.tensor([False, True], dtype=torch.bool, device=gs.device))
    assert_allclose(entity.get_dofs_velocity(), np.array([(3.0,) * 6, (0.0,) * 6]), tol=gs.EPS)


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("batch_links_info", [False, True])
@pytest.mark.parametrize("batch_joints_info", [False, True])
@pytest.mark.parametrize("batch_dofs_info", [False, True])
def test_batched_info(batch_links_info, batch_joints_info, batch_dofs_info):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            batch_links_info=batch_links_info,
            batch_joints_info=batch_joints_info,
            batch_dofs_info=batch_dofs_info,
        ),
    )
    terrain = scene.add_entity(gs.morphs.Terrain())
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build(n_envs=2)

    links_info = terrain.solver.data_manager.dyn_info.links
    entity_idx = links_info.entity_idx.to_numpy()
    assert entity_idx.shape == (12, 2) if batch_links_info else (12,)

    joints_info = terrain.solver.data_manager.dyn_info.joints
    pos = joints_info.pos.to_numpy()
    assert pos.shape == (10, 2, 3) if batch_joints_info else (10, 3)

    dofs_info = terrain.solver.data_manager.dyn_info.dofs
    act_gain = dofs_info.act_gain.to_numpy()
    assert act_gain.shape == (9, 2) if batch_dofs_info else (9,)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_info_batching(tol):
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            batch_dofs_info=True,
            batch_joints_info=True,
            batch_links_info=True,
        ),
        show_viewer=False,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene.build(n_envs=2)

    scene.step()
    qposs = robot.get_qpos()
    assert_allclose(qposs[0], qposs[1], tol=tol)


@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 2])
def test_geom_pos_quat(n_envs, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -10.0),
        ),
        show_viewer=show_viewer,
    )

    box = scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 2.0),
        )
    )
    scene.build(n_envs=n_envs)
    batch_shape = (n_envs,) if n_envs > 0 else ()

    box.set_dofs_position(np.random.rand(*batch_shape, 6))
    scene.rigid_solver.update_vgeoms()

    for link in box.links:
        for vgeom, geom in zip(link.vgeoms, link.geoms):
            geom_pos, geom_quat = geom.get_pos(), geom.get_quat()
            assert geom_pos.shape == (*batch_shape, 3)
            assert geom_quat.shape == (*batch_shape, 4)
            vgeom_pos, vgeom_quat = vgeom.get_pos(), vgeom.get_quat()
            assert vgeom_pos.shape == (*batch_shape, 3)
            assert vgeom_quat.shape == (*batch_shape, 4)
            assert_allclose(geom_pos, vgeom_pos, atol=gs.EPS)
            assert_allclose(geom_quat, vgeom_quat, atol=gs.EPS)


@pytest.mark.slow  # ~200s
@pytest.mark.required
@pytest.mark.parametrize("batch_fixed_verts", [False, True])
@pytest.mark.parametrize("relative", [False, True])
def test_set_root_pose(batch_fixed_verts, relative, show_viewer, tol):
    ROBOT_POS_ZERO = (0.0, 0.4, 0.1)
    ROBOT_EULER_ZERO = (0.0, 0.0, 90.0)
    CUBE_POS_ZERO = (0.65, 0.0, 0.02)
    CUBE_EULER_ZERO = (0.0, 90.0, 0.0)

    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            offset_pos=ROBOT_POS_ZERO,
            offset_euler=ROBOT_EULER_ZERO,
            batch_fixed_verts=batch_fixed_verts,
        ),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.04,
            batch_fixed_verts=False,
            fixed=True,
        ),
    )
    cube = scene.add_entity(
        gs.morphs.Box(
            size=(0.04, 0.04, 0.04),
            offset_pos=CUBE_POS_ZERO,
            offset_euler=CUBE_EULER_ZERO,
        ),
    )
    plain_box = scene.add_entity(
        gs.morphs.Box(
            pos=(2.0, 0.0, 0.2),
            size=(0.04, 0.04, 0.04),
        ),
    )
    POSED_BOX_POS = (2.0, 0.5, 0.3)
    POSED_BOX_OFFSET_EULER = (0.0, 0.0, 45.0)
    posed_box = scene.add_entity(
        gs.morphs.Box(
            pos=POSED_BOX_POS,
            size=(0.04, 0.04, 0.04),
            offset_pos=(0.0, 0.0, 0.5),
            offset_euler=POSED_BOX_OFFSET_EULER,
        ),
    )
    scene.build(n_envs=2)

    # A no-offset entity reports the same pose in the user and world frames.
    assert_allclose(plain_box.get_pos(relative=True), plain_box.get_pos(relative=False), tol=tol)
    assert_allclose(plain_box.get_pos(), (2.0, 0.0, 0.2), tol=tol)

    # With both a morph pose and an offset, the relative getter returns the morph pose while the world getter carries
    # the offset composed onto it (the offset position adds in z since the user orientation is identity).
    assert_allclose(posed_box.get_pos(relative=True), POSED_BOX_POS, tol=tol)
    assert_allclose(posed_box.get_quat(relative=True), gu.identity_quat(), tol=tol)
    assert_allclose(posed_box.get_pos(relative=False), (2.0, 0.5, 0.8), tol=tol)
    assert_allclose(
        posed_box.get_quat(relative=False),
        gu.xyz_to_quat(np.array(POSED_BOX_OFFSET_EULER), rpy=True, degrees=True),
        tol=tol,
    )

    # Setting the orientation in the user frame keeps the user-frame position fixed: the offset position rotates with
    # the orientation, so the world position is rewritten to preserve the reported relative position. Rotating about x
    # while the offset position is along z makes that offset contribution change, exercising the rewrite.
    new_quat = gu.xyz_to_quat(np.array((90.0, 0.0, 0.0)), rpy=True, degrees=True)
    posed_box.set_quat(new_quat, relative=True)
    assert_allclose(posed_box.get_pos(relative=True), POSED_BOX_POS, tol=tol)
    assert_allclose(posed_box.get_quat(relative=True), new_quat, tol=tol)

    robot_aabb_init, robot_base_aabb_init = robot.get_AABB(), robot.geoms[0].get_AABB()
    cube_aabb_init, cube_base_aabb_init = cube.get_AABB(), cube.geoms[0].get_AABB()

    # Make sure that it is not possible to end up in an inconsistent state for fixed geometries. These place entities
    # at absolute world positions, so they bypass the pose offset (relative=False).
    pos_delta = np.random.rand(2, 3)
    with nullcontext() if batch_fixed_verts else pytest.raises(gs.GenesisException):
        robot.set_pos(pos_delta, relative=False)
        if show_viewer:
            scene.visualizer.update()
    with nullcontext() if batch_fixed_verts else pytest.raises(gs.GenesisException):
        robot.set_pos(pos_delta[[0]], envs_idx=[0], relative=False)
        if show_viewer:
            scene.visualizer.update()
    cube.set_pos(pos_delta[[0]] + (0.0, 0.0, 0.16), envs_idx=[0], relative=False)
    cube.set_pos(pos_delta[[1]] + (0.0, 0.0, 0.11), envs_idx=[1], relative=False)
    sphere.set_pos(np.tile(pos_delta[[0]], (2, 1)) + 1.0, relative=False)
    quat_delta = np.random.rand(2, 4)
    with nullcontext() if batch_fixed_verts else pytest.raises(gs.GenesisException):
        robot.set_quat(quat_delta, relative=False)
        if show_viewer:
            scene.visualizer.update()
    with nullcontext() if batch_fixed_verts else pytest.raises(gs.GenesisException):
        robot.set_quat(quat_delta[[0]], envs_idx=[0], relative=False)
        if show_viewer:
            scene.visualizer.update()
    cube.set_quat(quat_delta, relative=False)
    if show_viewer:
        scene.visualizer.update()

    sphere_aabb, sphere_base_aabb = sphere.get_AABB(), sphere.geoms[0].get_AABB()
    assert_allclose(sphere_aabb.mean(dim=-2), pos_delta[0] + 1.0, tol=tol)
    assert_allclose(sphere_aabb, sphere_base_aabb, tol=tol)

    # Simulate for a while to check if the dynamic object is colliding with the static one
    if batch_fixed_verts:
        has_collided = torch.tensor([False, False], dtype=torch.bool, device=gs.device)
        for _ in range(20):
            scene.step()
            contacts_state = cube.get_contacts(with_entity=robot, exclude_self_contact=True)
            has_collided |= contacts_state["valid_mask"].any(dim=-1)
            if has_collided.all():
                break
        else:
            raise AssertionError("Cube never collided with robot for at least one of the environments.")

    for _ in range(2):
        scene.reset()

        for entity, pos_zero, euler_zero, entity_aabb_init, base_aabb_init in (
            (robot, ROBOT_POS_ZERO, ROBOT_EULER_ZERO, robot_aabb_init, robot_base_aabb_init),
            (cube, CUBE_POS_ZERO, CUBE_EULER_ZERO, cube_aabb_init, cube_base_aabb_init),
        ):
            pos_zero = torch.tensor(pos_zero, device=gs.device, dtype=gs.tc_float)
            euler_zero = torch.deg2rad(torch.tensor(euler_zero, dtype=gs.tc_float))
            quat_zero = gu.xyz_to_quat(euler_zero, rpy=True)
            # The pose lives in the offset, so the world frame (relative=False) carries it; the user frame is identity.
            assert_allclose(entity.get_pos(relative=False), pos_zero, tol=tol)
            assert_allclose(entity.get_pos(relative=True), 0.0, tol=tol)
            # Use quaternion for comparison to avoid gymbal lock issue in euler angles
            quat = entity.get_quat(relative=False)
            assert_allclose(quat, quat_zero, tol=tol)
            base_aabb = entity.geoms[0].get_AABB()
            assert base_aabb.shape == ((2, 2, 3) if not entity.geoms[0].is_fixed or batch_fixed_verts else (2, 3))
            assert_allclose(base_aabb, base_aabb_init, tol=tol)
            assert_allclose(entity.get_AABB(), entity_aabb_init, tol=tol)

            pos_delta = torch.as_tensor(np.random.rand(3), dtype=gs.tc_float, device=gs.device).expand((2, 3))
            entity.set_pos(pos_delta, relative=relative)

            pos_ref = pos_delta + pos_zero if relative else pos_delta
            # Round-trip in the frame it was set in: the getter must report back exactly what set_pos received.
            assert_allclose(entity.get_pos(relative=relative), pos_delta, tol=tol)
            assert_allclose(entity.geoms[0].get_AABB(), base_aabb_init + (pos_ref - pos_zero), tol=tol)
            assert_allclose(entity.get_AABB(), entity_aabb_init + (pos_ref - pos_zero), tol=tol)

            quat_delta = torch.tile(torch.as_tensor(np.random.rand(4), dtype=gs.tc_float, device=gs.device), (2, 1))
            quat_delta /= torch.linalg.norm(quat_delta, axis=1, keepdim=True)
            entity.set_quat(quat_delta, relative=relative)
            assert_allclose(entity.get_quat(relative=relative), quat_delta, tol=tol)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_normalized_quat(show_viewer, tol):
    scene = gs.Scene(
        show_viewer=show_viewer,
        show_FPS=False,
    )
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
        ),
    )
    scene.build()

    # Make sure that the simulation state is not sensitive to qpos normalization
    quat = torch.randn((4,), dtype=gs.tc_float, device=gs.device)

    qpos = robot.get_qpos()
    qpos[3:7] = quat / torch.linalg.norm(quat)
    robot.set_qpos(qpos)
    scene.step()
    qpos_post = robot.get_qpos()
    assert_allclose(torch.linalg.norm(qpos_post[3:7]), 1.0, tol=tol)

    qpos[3:7] = quat
    scene.reset()
    robot.set_qpos(qpos)
    # assert_allclose(qpos, robot.get_qpos(), tol=tol)  # True, but not specification requirement
    scene.step()
    assert_allclose(qpos_post, robot.get_qpos(), tol=tol)

    scene.reset()
    robot.set_quat(quat)
    # assert_allclose(quat, qpos[3:7], tol=tol)  # True, but not specification requirement
    scene.step()
    assert_allclose(qpos_post, robot.get_qpos(), tol=tol)

    # Make sure that entity, link and geom quaternions are normalized.
    # "RigidEntity.set_quat" is calling 'kernel_forward_kinematics_links_geoms', which is relying on
    # 'func_update_cartesian_space' under the hood.
    # Let's check that everything is properly normalized at this stage already. If so, it means that all quaternions of
    # interest are guaranteed to be always normalized, since 'func_update_cartesian_space' is called internally during
    # forward dynamics 'step_1' at the very beginning of 'RigidSolver.step'.
    scene.reset()
    robot.set_quat(quat)
    assert_allclose(torch.linalg.norm(robot.get_quat()), 1.0, tol=tol)
    for link in robot.links:
        assert_allclose(torch.linalg.norm(link.get_quat()), 1.0, tol=tol)
    for geom in robot.geoms:
        assert_allclose(torch.linalg.norm(geom.get_quat()), 1.0, tol=tol)
    assert_allclose(torch.linalg.norm(scene.rigid_solver.get_links_quat(), dim=-1), 1.0, tol=tol)
    assert_allclose(torch.linalg.norm(scene.rigid_solver.get_geoms_quat(), dim=-1), 1.0, tol=tol)


@pytest.mark.required
def test_mass_setters(tol):
    # Batched links info (default): entity- and link-level set_mass apply, link masses may differ per env, and a
    # wrong-length array is rejected. The heterogeneous entity gives each env a distinct starting mass.
    scene = gs.Scene(
        show_viewer=False,
    )
    het_obj = scene.add_entity(
        morph=[
            gs.morphs.Box(size=(0.01, 0.01, 0.01)),
            gs.morphs.Box(size=(0.02, 0.02, 0.02)),
            gs.morphs.Sphere(radius=0.01),
            gs.morphs.Sphere(radius=0.02),
        ],
    )
    scene.build(n_envs=4)
    link = next(link for link in het_obj.links if not link.is_fixed)
    with pytest.raises(gs.GenesisException):
        link.set_mass((1.0, 2.0))
    het_obj.set_mass(1.0)
    assert_allclose(het_obj.get_mass(), 1.0, tol=tol)
    target_mass = (0.2, 0.4, 0.6, 0.8)
    link.set_mass(target_mass)
    assert_allclose(link.get_mass(), target_mass, tol=tol)

    # Non-batched links info: link mass is shared across envs, so a scalar applies uniformly and a per-env array raises.
    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            batch_links_info=False,
        ),
    )
    obj = scene.add_entity(
        morph=gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
        )
    )
    scene.build(n_envs=4)
    link = next(link for link in obj.links if not link.is_fixed)
    link.set_mass(2.0)
    assert_allclose(link.get_mass(), 2.0, tol=tol)
    with pytest.raises(gs.GenesisException):
        link.set_mass((1.0, 2.0, 3.0, 4.0))


@pytest.mark.slow  # ~250s
@pytest.mark.required
@pytest.mark.parametrize("n_envs", [0, 3])
def test_axis_aligned_bounding_boxes(n_envs):
    scene = gs.Scene()
    scene.add_entity(
        gs.morphs.Plane(
            normal=(0, 0, 1),
            pos=(0, 0, 0),
        ),
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0.5, 0, 0.05),
        ),
    )
    scene.add_entity(
        gs.morphs.Cylinder(
            height=0.8,
            radius=0.06,
            pos=(1.0, 0, 0.5),
        ),
    )
    scene.add_entity(
        gs.morphs.Sphere(
            radius=0.05,
            pos=(-0.5, 0, 0.05),
        ),
    )
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
        ),
    )
    scene.build(n_envs=n_envs)

    batch_shape = (n_envs,) if n_envs > 0 else ()
    aabb_shape = (*batch_shape, 2, 3)

    qpos = np.random.rand(*(*batch_shape, robot.n_dofs))
    robot.set_dofs_position(qpos)

    robot_aabb = robot.get_AABB()
    robot_geoms_aabb = torch.stack([geom.get_AABB().expand(aabb_shape) for geom in robot.geoms], dim=0)
    assert_allclose(torch.min(robot_geoms_aabb[..., 0, :], dim=0).values, robot_aabb[..., 0, :], tol=gs.EPS)
    assert_allclose(torch.max(robot_geoms_aabb[..., 1, :], dim=0).values, robot_aabb[..., 1, :], tol=gs.EPS)
    for link in robot.links:
        link_aabb = link.get_AABB()
        link_geoms_aabb = torch.stack([geom.get_AABB().expand(aabb_shape) for geom in link.geoms], dim=0)
        assert_allclose(torch.min(link_geoms_aabb[..., 0, :], dim=0).values, link_aabb[..., 0, :], tol=gs.EPS)
        assert_allclose(torch.max(link_geoms_aabb[..., 1, :], dim=0).values, link_aabb[..., 1, :], tol=gs.EPS)

    all_aabbs = scene.sim.rigid_solver.get_AABB()
    aabbs = [geom.get_AABB().expand(aabb_shape) for entity in scene.entities for geom in entity.geoms]
    if n_envs > 0:
        assert all_aabbs.ndim == 4 and len(all_aabbs) == n_envs
    else:
        assert all_aabbs.ndim == 3
    assert all_aabbs.shape[-3:] == (len(aabbs), 2, 3)
    assert_allclose(aabbs[:4], all_aabbs.swapaxes(-3, 0)[:4], atol=gs.EPS)
    with pytest.raises(AssertionError):
        assert_allclose(aabbs[4:], all_aabbs.swapaxes(-3, 0)[4:], atol=gs.EPS)

    box_aabb_min, box_aabb_max = aabbs[1].split(1, dim=-2)
    assert_allclose(box_aabb_min, (0.45, -0.05, 0.0), atol=gs.EPS)
    assert_allclose(box_aabb_max, (0.55, 0.05, 0.1), atol=gs.EPS)
    sphere_aabb_min, sphere_aabb_max = aabbs[3].split(1, dim=-2)
    assert_allclose(sphere_aabb_min, (-0.55, -0.05, 0.0), atol=gs.EPS)
    assert_allclose(sphere_aabb_max, (-0.45, 0.05, 0.1), atol=gs.EPS)

    vaabbs = [vgeom.get_vAABB().expand(aabb_shape) for entity in scene.entities for vgeom in entity.vgeoms]
    if n_envs > 0:
        for entity in scene.entities:
            for vgeom in entity.vgeoms:
                assert_allclose(vgeom.get_vAABB(), [vgeom.get_vAABB(i)[0] for i in range(n_envs)], tol=gs.EPS)
    box_aabb_min, box_aabb_max = vaabbs[1].split(1, dim=-2)
    assert_allclose(box_aabb_min, (0.45, -0.05, 0.0), atol=gs.EPS)
    assert_allclose(box_aabb_max, (0.55, 0.05, 0.1), atol=gs.EPS)
    sphere_aabb_min, sphere_aabb_max = vaabbs[3].split(1, dim=-2)
    assert_allclose(sphere_aabb_min, (-0.55, -0.05, 0.0), atol=1e-3)
    assert_allclose(sphere_aabb_max, (-0.45, 0.05, 0.1), atol=1e-3)

    robot_vaabb = robot.get_vAABB()
    assert_allclose(robot_vaabb, robot_aabb, atol=1e-3)


@pytest.mark.slow  # ~200s
@pytest.mark.required
def test_reset(show_viewer):
    BOOL_MASK = torch.tensor([True, False, True, False], dtype=torch.bool, device=gs.device)

    scene = gs.Scene(
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.URDF(
            file="urdf/plane/plane.urdf",
            fixed=True,
        )
    )
    scene.add_entity(
        gs.morphs.Box(
            size=(0.1, 0.1, 0.1),
            pos=(0, 0, 0.5),
        )
    )
    scene.build(n_envs=4)

    init_state = scene.get_state()
    init_rigid_state = next(s for s in init_state.solvers_state if isinstance(s, RigidSolverState))
    for _ in range(50):
        scene.step()
    fallen_state = scene.get_state()
    fallen_rigid_state = next(s for s in fallen_state.solvers_state if isinstance(s, RigidSolverState))

    for envs_idx in (BOOL_MASK, torch.where(BOOL_MASK)[0]):
        scene.reset(state=fallen_state)
        scene.reset(state=init_state, envs_idx=envs_idx)
        for actual, init_ref, fallen_ref in (
            (
                qd_to_torch(scene.rigid_solver.rigid_info.qpos, transpose=True, copy=True),
                init_rigid_state.qpos,
                fallen_rigid_state.qpos,
            ),
            (
                qd_to_torch(scene.rigid_solver.dyn_state.dofs.vel, transpose=True, copy=True),
                init_rigid_state.dofs_vel,
                fallen_rigid_state.dofs_vel,
            ),
            (
                qd_to_torch(scene.rigid_solver.dyn_state.links.pos, transpose=True, copy=True),
                init_rigid_state.links_pos,
                fallen_rigid_state.links_pos,
            ),
        ):
            assert_allclose(actual[BOOL_MASK], init_ref[BOOL_MASK], tol=gs.EPS)
            assert_allclose(actual[~BOOL_MASK], fallen_ref[~BOOL_MASK], tol=gs.EPS)

    # After reset, simulation from init_state should reproduce the original fallen_state trajectory
    for _ in range(50):
        scene.step()
    for actual, fallen_ref in (
        (qd_to_torch(scene.rigid_solver.rigid_info.qpos, transpose=True, copy=True), fallen_rigid_state.qpos),
        (qd_to_torch(scene.rigid_solver.dyn_state.dofs.vel, transpose=True, copy=True), fallen_rigid_state.dofs_vel),
        (
            qd_to_torch(scene.rigid_solver.dyn_state.links.pos, transpose=True, copy=True),
            fallen_rigid_state.links_pos,
        ),
    ):
        assert_allclose(actual[BOOL_MASK], fallen_ref[BOOL_MASK], tol=gs.EPS)


@pytest.mark.slow  # ~350s
@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.cpu, gs.gpu])
def test_scene_saver_franka(tmp_path, show_viewer, tol):
    scene1 = gs.Scene(
        show_viewer=show_viewer,
        profiling_options=gs.options.ProfilingOptions(
            show_FPS=False,
        ),
    )
    franka1 = scene1.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene1.build()

    dof_idx = [j.dofs_idx_local[0] for j in franka1.joints]

    franka1.set_dofs_kp(np.full(len(dof_idx), 3000), dof_idx)
    franka1.set_dofs_kv(np.full(len(dof_idx), 300), dof_idx)

    target_pose = np.array([0.3, -0.8, 0.4, -1.6, 0.5, 1.0, -0.6, 0.03, 0.03], dtype=float)
    franka1.control_dofs_position(target_pose, dof_idx)

    for _ in range(100):
        scene1.step()

    pose_ref = franka1.get_dofs_position(dof_idx)

    ckpt_path = tmp_path / "franka_unit.pkl"
    scene1.save_checkpoint(ckpt_path)

    scene2 = gs.Scene(show_viewer=show_viewer)
    franka2 = scene2.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )
    scene2.build()
    scene2.load_checkpoint(ckpt_path)

    pose_loaded = franka2.get_dofs_position(dof_idx)

    # FIXME: It should be possible to achieve better accuracy with 64bits precision
    assert_allclose(pose_ref, pose_loaded, tol=2e-6)


@pytest.mark.required
def test_deprecated_properties(caplog):
    scene = gs.Scene(
        show_viewer=False,
        show_FPS=False,
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(1.0, 1.0, 1.0),
            pos=(0.0, 0.0, 0.0),
        )
    )
    scene.build()

    joint = box.joints[0]

    # Verify introspection doesn't trigger warnings
    caplog.clear()
    with caplog.at_level("WARNING"):
        repr(joint)
        vars(joint)
    assert len(caplog.records) == 0

    for name_old, name_new in (
        ("dof_idx", "dofs_idx"),
        ("dof_idx_local", "dofs_idx_local"),
        ("q_idx", "qs_idx"),
        ("q_idx_local", "qs_idx_local"),
    ):
        # Make sure that deprecated properties are hidden
        assert name_old not in dir(joint)

        # Verify deprecated properties emit warnings but work correctly
        caplog.clear()
        with caplog.at_level("WARNING"):
            deprecated_value = getattr(joint, name_old)
        assert len(caplog.records) > 0
        assert_allclose(deprecated_value, getattr(joint, name_new), tol=gs.EPS)
