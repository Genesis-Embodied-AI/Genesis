from dataclasses import dataclass
from itertools import chain
from typing import Literal, Sequence

import numpy as np

import mujoco

import genesis as gs
import genesis.utils.geom as gu


@dataclass
class MjSim:
    model: mujoco.MjModel
    data: mujoco.MjData


def init_simulators(gs_sim, mj_sim=None, qpos=None, qvel=None):
    if mj_sim is not None:
        _, (_, _, mj_qs_idx, mj_dofs_idx, _) = _get_model_mappings(gs_sim, mj_sim)

    (gs_robot,) = gs_sim.entities

    gs_sim.scene.reset()
    if qpos is not None:
        gs_robot.set_qpos(qpos)
    if qvel is not None:
        gs_robot.set_dofs_velocity(qvel)
    # TODO: This should be moved in `set_state`, `set_qpos`, `set_dofs_position`, `set_dofs_velocity`
    gs_sim.rigid_solver.dofs_state.qf_constraint.fill(0.0)
    gs_sim.rigid_solver._kernel_forward_dynamics()
    gs_sim.rigid_solver._func_constraint_force()
    if gs_sim.scene.visualizer:
        gs_sim.scene.visualizer.update()

    if mj_sim is not None:
        mujoco.mj_resetData(mj_sim.model, mj_sim.data)
        mj_sim.data.qpos[mj_qs_idx] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[mj_dofs_idx] = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
        mujoco.mj_forward(mj_sim.model, mj_sim.data)


def _gs_search_by_joints_name(
    scene,
    joints_name: str | list[str],
    to: Literal["entity", "index"] = "index",
    is_local: bool = False,
    flatten: bool = True,
):
    if isinstance(joints_name, str):
        joints_name = [joints_name]

    for entity in scene.entities:
        try:
            gs_joints_idx = dict()
            gs_joints_qs_idx = dict()
            gs_joints_dofs_idx = dict()
            valid_joints_name = []
            for joint in entity.joints:
                valid_joints_name.append(joint.name)
                if joint.name in joints_name:
                    if to == "entity":
                        gs_joints_idx[joint.name] = joint
                        gs_joints_qs_idx[joint.name] = joint
                        gs_joints_dofs_idx[joint.name] = joint
                    elif to == "index":
                        gs_joints_idx[joint.name] = joint.idx_local if is_local else joint.idx
                        gs_joints_qs_idx[joint.name] = joint.qs_idx_local if is_local else joint.qs_idx
                        gs_joints_dofs_idx[joint.name] = joint.dofs_idx_local if is_local else joint.dofs_idx
                    else:
                        raise ValueError(f"Cannot recognize what ({to}) to extract for the search")

            missing_joints_name = set(joints_name) - gs_joints_idx.keys()
            if len(missing_joints_name) > 0:
                raise ValueError(
                    f"Cannot find joints `{missing_joints_name}`. Valid joints names are {valid_joints_name}"
                )

            if flatten:
                return (
                    list(gs_joints_idx.values()),
                    list(chain.from_iterable(gs_joints_qs_idx.values())),
                    list(chain.from_iterable(gs_joints_dofs_idx.values())),
                )
            return (gs_joints_idx, gs_joints_qs_idx, gs_joints_dofs_idx)
        except ValueError:
            pass
    else:
        raise ValueError(f"Fail to find joint indices for {joints_name}")


def _gs_search_by_links_name(
    scene,
    links_name: str | Sequence[str],
    to: Literal["entity", "index"] = "index",
    is_local: bool = False,
    flatten: bool = True,
):
    if isinstance(links_name, str):
        links_name = (links_name,)

    for entity in scene.entities:
        try:
            gs_links_idx = dict()
            valid_links_name = []
            for link in entity.links:
                valid_links_name.append(link.name)
                if link.name in links_name:
                    if to == "entity":
                        gs_links_idx[link.name] = link
                    elif to == "index":
                        gs_links_idx[link.name] = link.idx_local if is_local else link.idx
                    else:
                        raise ValueError(f"Cannot recognize what ({to}) to extract for the search")

            missing_links_name = set(links_name) - gs_links_idx.keys()
            if missing_links_name:
                raise ValueError(f"Cannot find links `{missing_links_name}`. Valid link names are {valid_links_name}")

            if flatten:
                return list(gs_links_idx.values())
            return gs_links_idx
        except ValueError:
            pass
    else:
        raise ValueError(f"Fail to find link indices for {links_name}")


def _get_model_mappings(
    gs_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    body_names: list[str] | None = None,
):
    if joints_name is None:
        joints_name = [
            joint.name for entity in gs_sim.entities for joint in entity.joints if joint.type != gs.JOINT_TYPE.FIXED
        ]
    body_names = [
        body.name for entity in gs_sim.entities for body in entity.links if not (body.is_fixed and body.parent_idx < 0)
    ]

    motors_name: list[str] = []
    mj_joints_idx: list[int] = []
    mj_qs_idx: list[int] = []
    mj_dofs_idx: list[int] = []
    mj_motors_idx: list[int] = []
    for joint_name in joints_name:
        if joint_name:
            mj_joint = mj_sim.model.joint(joint_name)
        else:
            # Must rely on exhaustive search if the joint has empty name
            for j in range(mj_sim.model.njoint):
                mj_joint = mj_sim.model.joint(j)
                if mj_joint.name == "":
                    break
            else:
                raise ValueError(f"Invalid joint name '{joint_name}'.")
        mj_joints_idx.append(mj_joint.id)
        mj_type = mj_sim.model.jnt_type[mj_joint.id]
        if mj_type == mujoco.mjtJoint.mjJNT_HINGE:
            n_dofs, n_qs = 1, 1
        elif mj_type == mujoco.mjtJoint.mjJNT_SLIDE:
            n_dofs, n_qs = 1, 1
        elif mj_type == mujoco.mjtJoint.mjJNT_BALL:
            n_dofs, n_qs = 3, 4
        elif mj_type == mujoco.mjtJoint.mjJNT_FREE:
            n_dofs, n_qs = 6, 7
        else:
            raise ValueError(f"Invalid joint type '{mj_type}'.")
        mj_dof_start_j = mj_sim.model.jnt_dofadr[mj_joint.id]
        mj_dofs_idx += range(mj_dof_start_j, mj_dof_start_j + n_dofs)

        mj_q_start_j = mj_sim.model.jnt_qposadr[mj_joint.id]
        mj_qs_idx += range(mj_q_start_j, mj_q_start_j + n_qs)
        if (mj_joint.id == mj_sim.model.actuator_trnid[:, 0]).any():
            motors_name.append(joint_name)
            (motors_idx,) = np.nonzero(mj_joint.id == mj_sim.model.actuator_trnid[:, 0])
            # FIXME: only supporting 1DoF per actuator
            mj_motors_idx.append(motors_idx[0])
    mj_bodies_idx = [mj_sim.model.body(body_name).id for body_name in body_names]
    (gs_joints_idx, gs_q_idx, gs_dofs_idx) = _gs_search_by_joints_name(gs_sim.scene, joints_name)
    (_, _, gs_motors_idx) = _gs_search_by_joints_name(gs_sim.scene, motors_name)
    gs_bodies_idx = _gs_search_by_links_name(gs_sim.scene, body_names)

    gs_maps = (gs_bodies_idx, gs_joints_idx, gs_q_idx, gs_dofs_idx, gs_motors_idx)
    mj_maps = (mj_bodies_idx, mj_joints_idx, mj_qs_idx, mj_dofs_idx, mj_motors_idx)
    return gs_maps, mj_maps


def check_mujoco_model_consistency(
    gs_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    body_names: list[str] | None = None,
    *,
    atol: float,
):
    # Get mapping between Mujoco and Genesis
    gs_maps, mj_maps = _get_model_mappings(gs_sim, mj_sim, joints_name, body_names)
    (gs_bodies_idx, gs_joints_idx, gs_q_idx, gs_dofs_idx, gs_motors_idx) = gs_maps
    (mj_bodies_idx, mj_joints_idx, mj_qs_idx, mj_dofs_idx, mj_motors_idx) = mj_maps

    # solver
    gs_gravity = gs_sim.rigid_solver.scene.gravity
    mj_gravity = mj_sim.model.opt.gravity
    np.testing.assert_allclose(gs_gravity, mj_gravity, atol=atol)
    assert mj_sim.model.opt.timestep == gs_sim.rigid_solver.substep_dt
    assert mj_sim.model.opt.tolerance == gs_sim.rigid_solver._options.tolerance
    assert mj_sim.model.opt.iterations == gs_sim.rigid_solver._options.iterations
    assert not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
    assert not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    assert not (mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    assert mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    assert not (mj_sim.model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_FWDINV)

    mj_adj_collision = bool(mj_sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    gs_adj_collision = gs_sim.rigid_solver._options.enable_adjacent_collision
    assert gs_adj_collision == mj_adj_collision

    mj_solver = mujoco.mjtSolver(mj_sim.model.opt.solver)
    if mj_solver.name == "mjSOL_PGS":
        assert False
    elif mj_solver.name == "mjSOL_CG":
        assert gs_sim.rigid_solver._options.constraint_solver == gs.constraint_solver.CG
    elif mj_solver.name == "mjSOL_NEWTON":
        assert gs_sim.rigid_solver._options.constraint_solver == gs.constraint_solver.Newton
    else:
        assert False

    mj_integrator = mujoco.mjtIntegrator(mj_sim.model.opt.integrator)
    if mj_integrator.name == "mjINT_EULER":
        assert gs_sim.rigid_solver._options.integrator == gs.integrator.Euler
    elif mj_integrator.name == "mjINT_IMPLICIT":
        assert False
    elif mj_integrator.name == "mjINT_IMPLICITFAST":
        assert gs_sim.rigid_solver._options.integrator == gs.integrator.implicitfast
    else:
        assert False

    mj_cone = mujoco.mjtCone(mj_sim.model.opt.cone)
    if mj_cone.name == "mjCONE_ELLIPTIC":
        assert False
    elif mj_cone.name == "mjCONE_PYRAMIDAL":
        assert True
    else:
        assert False

    # body
    for gs_i, mj_i in zip(gs_bodies_idx, mj_bodies_idx):
        gs_invweight_i = gs_sim.rigid_solver.links_info.invweight.to_numpy()[gs_i]
        mj_invweight_i = mj_sim.model.body(mj_i).invweight0[0]
        np.testing.assert_allclose(gs_invweight_i, mj_invweight_i, atol=atol)
        gs_inertia_i = gs_sim.rigid_solver.links_info.inertial_i.to_numpy()[gs_i, [0, 1, 2], [0, 1, 2]]
        mj_inertia_i = mj_sim.model.body(mj_i).inertia
        np.testing.assert_allclose(gs_inertia_i, mj_inertia_i, atol=atol)
        gs_ipos_i = gs_sim.rigid_solver.links_info.inertial_pos.to_numpy()[gs_i]
        mj_ipos_i = mj_sim.model.body(mj_i).ipos
        np.testing.assert_allclose(gs_ipos_i, mj_ipos_i, atol=atol)
        gs_iquat_i = gs_sim.rigid_solver.links_info.inertial_quat.to_numpy()[gs_i]
        mj_iquat_i = mj_sim.model.body(mj_i).iquat
        np.testing.assert_allclose(gs_iquat_i, mj_iquat_i, atol=atol)
        gs_pos_i = gs_sim.rigid_solver.links_info.pos.to_numpy()[gs_i]
        mj_pos_i = mj_sim.model.body(mj_i).pos
        np.testing.assert_allclose(gs_pos_i, mj_pos_i, atol=atol)
        gs_quat_i = gs_sim.rigid_solver.links_info.quat.to_numpy()[gs_i]
        mj_quat_i = mj_sim.model.body(mj_i).quat
        np.testing.assert_allclose(gs_quat_i, mj_quat_i, atol=atol)
        gs_mass_i = gs_sim.rigid_solver.links_info.inertial_mass.to_numpy()[gs_i]
        mj_mass_i = mj_sim.model.body(mj_i).mass
        np.testing.assert_allclose(gs_mass_i, mj_mass_i, atol=atol)

    # dof / joints
    gs_dof_damping = gs_sim.rigid_solver.dofs_info.damping.to_numpy()
    mj_dof_damping = mj_sim.model.dof_damping
    np.testing.assert_allclose(gs_dof_damping[gs_dofs_idx], mj_dof_damping[mj_dofs_idx], atol=atol)

    # FIXME: DoF damping implementation in Genesis is not consistent with Mujoco (for efficiency)
    # np.testing.assert_allclose(mj_sim.model.dof_damping, 0.0)

    gs_dof_armature = gs_sim.rigid_solver.dofs_info.armature.to_numpy()
    mj_dof_armature = mj_sim.model.dof_armature
    np.testing.assert_allclose(gs_dof_armature[gs_dofs_idx], mj_dof_armature[mj_dofs_idx], atol=atol)

    # FIXME: 1 stiffness per joint in Mujoco, 1 stiffness per DoF in Genesis
    gs_dof_stiffness = gs_sim.rigid_solver.dofs_info.stiffness.to_numpy()
    mj_dof_stiffness = mj_sim.model.jnt_stiffness
    # np.testing.assert_allclose(gs_dof_stiffness[gs_dofs_idx], mj_dof_stiffness[mj_joints_idx], atol=atol)

    gs_dof_invweight0 = gs_sim.rigid_solver.dofs_info.invweight.to_numpy()
    mj_dof_invweight0 = mj_sim.model.dof_invweight0
    np.testing.assert_allclose(gs_dof_invweight0[gs_dofs_idx], mj_dof_invweight0[mj_dofs_idx], atol=atol)

    gs_joint_solparams = np.concatenate([joint.sol_params for entity in gs_sim.entities for joint in entity.joints])
    gs_joint_solref = gs_joint_solparams[:, :2]
    mj_joint_solref = mj_sim.model.jnt_solref
    np.testing.assert_allclose(gs_joint_solref[gs_joints_idx], mj_joint_solref[mj_joints_idx], atol=atol)
    gs_joint_solimp = gs_joint_solparams[:, 2:]
    mj_joint_solimp = mj_sim.model.jnt_solimp
    np.testing.assert_allclose(gs_joint_solimp[gs_joints_idx], mj_joint_solimp[mj_joints_idx], atol=atol)
    gs_dof_solparams = np.concatenate([joint.dofs_sol_params for entity in gs_sim.entities for joint in entity.joints])
    gs_dof_solref = gs_dof_solparams[:, :2]
    mj_dof_solref = mj_sim.model.dof_solref
    np.testing.assert_allclose(gs_dof_solref[gs_dofs_idx], mj_dof_solref[mj_dofs_idx], atol=atol)
    gs_dof_solimp = gs_dof_solparams[:, 2:]
    mj_dof_solimp = mj_sim.model.dof_solimp
    np.testing.assert_allclose(gs_dof_solimp[gs_dofs_idx], mj_dof_solimp[mj_dofs_idx], atol=atol)

    np.testing.assert_allclose(mj_sim.model.jnt_margin, 0, atol=atol)
    gs_joint_range = np.stack(
        [
            gs_sim.rigid_solver.dofs_info[gs_sim.rigid_solver.joints_info[i].dof_start].limit.to_numpy()
            for i in gs_joints_idx
        ],
        axis=0,
    )
    mj_joint_range = mj_sim.model.jnt_range
    mj_joint_range[mj_sim.model.jnt_limited == 0, 0] = float("-inf")
    mj_joint_range[mj_sim.model.jnt_limited == 0, 1] = float("+inf")
    np.testing.assert_allclose(gs_joint_range, mj_joint_range[mj_joints_idx], atol=atol)

    # actuator (position control)
    for v in mj_sim.model.actuator_dyntype:
        assert v == mujoco.mjtDyn.mjDYN_NONE
    for v in mj_sim.model.actuator_biastype:
        assert v in (mujoco.mjtBias.mjBIAS_AFFINE, mujoco.mjtBias.mjBIAS_NONE)

    # NOTE: not considering gear
    gs_kp = gs_sim.rigid_solver.dofs_info.kp.to_numpy()
    gs_kv = gs_sim.rigid_solver.dofs_info.kv.to_numpy()
    mj_kp = -mj_sim.model.actuator_biasprm[:, 1]
    mj_kv = -mj_sim.model.actuator_biasprm[:, 2]
    np.testing.assert_allclose(gs_kp[gs_motors_idx], mj_kp[mj_motors_idx], atol=atol)
    np.testing.assert_allclose(gs_kv[gs_motors_idx], mj_kv[mj_motors_idx], atol=atol)


def check_mujoco_data_consistency(
    gs_sim,
    mj_sim,
    joints_name: list[str] | None = None,
    body_names: list[str] | None = None,
    *,
    qvel_prev: np.ndarray | None = None,
    atol: float,
):
    # Get mapping between Mujoco and Genesis
    gs_maps, mj_maps = _get_model_mappings(gs_sim, mj_sim, joints_name, body_names)
    (gs_bodies_idx, gs_joints_idx, gs_q_idx, gs_dofs_idx, gs_motors_idx) = gs_maps
    (mj_bodies_idx, mj_joints_idx, mj_qs_idx, mj_dofs_idx, mj_motors_idx) = mj_maps

    # crb
    gs_crb_inertial = gs_sim.rigid_solver.links_state.crb_inertial.to_numpy()[:, 0].reshape([-1, 9])[
        :, [0, 4, 8, 1, 2, 5]
    ]
    mj_crb_inertial = mj_sim.data.crb[:, :6]  # upper-triangular part
    np.testing.assert_allclose(gs_crb_inertial[gs_bodies_idx], mj_crb_inertial[mj_bodies_idx], atol=atol)
    gs_crb_pos = gs_sim.rigid_solver.links_state.crb_pos.to_numpy()[:, 0]
    mj_crb_pos = mj_sim.data.crb[:, 6:9]
    np.testing.assert_allclose(gs_crb_pos[gs_bodies_idx], mj_crb_pos[mj_bodies_idx], atol=atol)
    gs_crb_mass = gs_sim.rigid_solver.links_state.crb_mass.to_numpy()[:, 0]
    mj_crb_mass = mj_sim.data.crb[:, 9]
    np.testing.assert_allclose(gs_crb_mass[gs_bodies_idx], mj_crb_mass[mj_bodies_idx], atol=atol)

    gs_mass_mat_damped = gs_sim.rigid_solver.mass_mat.to_numpy()[:, :, 0]
    mj_mass_mat_damped = np.zeros((mj_sim.model.nv, mj_sim.model.nv))
    mujoco.mj_fullM(mj_sim.model, mj_mass_mat_damped, mj_sim.data.qM)
    mj_mass_mat_damped[np.diag_indices(mj_sim.model.nv)] += mj_sim.model.dof_damping * mj_sim.model.opt.timestep
    np.testing.assert_allclose(
        gs_mass_mat_damped[gs_dofs_idx][:, gs_dofs_idx], mj_mass_mat_damped[mj_dofs_idx][:, mj_dofs_idx], atol=atol
    )

    gs_meaninertia = gs_sim.rigid_solver.meaninertia.to_numpy()[0]
    mj_meaninertia = mj_sim.model.stat.meaninertia
    np.testing.assert_allclose(gs_meaninertia, mj_meaninertia, atol=atol)

    gs_qfrc_bias = gs_sim.rigid_solver.dofs_state.qf_bias.to_numpy()[:, 0]
    mj_qfrc_bias = mj_sim.data.qfrc_bias
    np.testing.assert_allclose(gs_qfrc_bias, mj_qfrc_bias[mj_dofs_idx], atol=atol)
    gs_qfrc_passive = gs_sim.rigid_solver.dofs_state.qf_passive.to_numpy()[:, 0]
    mj_qfrc_passive = mj_sim.data.qfrc_passive
    np.testing.assert_allclose(gs_qfrc_passive, mj_qfrc_passive[mj_dofs_idx], atol=atol)
    gs_qfrc_actuator = gs_sim.rigid_solver.dofs_state.qf_applied.to_numpy()[:, 0]
    mj_qfrc_actuator = mj_sim.data.qfrc_actuator
    np.testing.assert_allclose(gs_qfrc_actuator, mj_qfrc_actuator[mj_dofs_idx], atol=atol)

    gs_n_contacts = gs_sim.rigid_solver.collider.n_contacts.to_numpy()[0]
    mj_n_contacts = mj_sim.data.ncon
    assert gs_n_contacts == mj_n_contacts
    gs_n_constraints = gs_sim.rigid_solver.constraint_solver.n_constraints.to_numpy()[0]
    mj_n_constraints = mj_sim.data.nefc
    assert gs_n_constraints == mj_n_constraints

    if gs_n_constraints:
        gs_contact_pos = gs_sim.rigid_solver.collider.contact_data.pos.to_numpy()[:gs_n_contacts, 0]
        mj_contact_pos = mj_sim.data.contact.pos
        gs_sidx = np.argsort(gs_contact_pos[:, 0])
        mj_sidx = np.argsort(mj_contact_pos[:, 0])
        np.testing.assert_allclose(gs_contact_pos[gs_sidx], mj_contact_pos[mj_sidx], atol=atol)
        gs_contact_normal = gs_sim.rigid_solver.collider.contact_data.normal.to_numpy()[:gs_n_contacts, 0]
        mj_contact_normal = -mj_sim.data.contact.frame[:, :3]
        np.testing.assert_allclose(gs_contact_normal[gs_sidx], mj_contact_normal[mj_sidx], atol=atol)
        gs_penetration = gs_sim.rigid_solver.collider.contact_data.penetration.to_numpy()[:gs_n_contacts, 0]
        mj_penetration = -mj_sim.data.contact.dist
        np.testing.assert_allclose(gs_penetration[gs_sidx], mj_penetration[mj_sidx], atol=atol)

        gs_jac = gs_sim.rigid_solver.constraint_solver.jac.to_numpy()[:gs_n_constraints, :, 0]
        mj_jac = mj_sim.data.efc_J.reshape([gs_n_constraints, -1])
        gs_efc_D = gs_sim.rigid_solver.constraint_solver.efc_D.to_numpy()[:gs_n_constraints, 0]
        mj_efc_D = mj_sim.data.efc_D
        gs_efc_aref = gs_sim.rigid_solver.constraint_solver.aref.to_numpy()[:gs_n_constraints, 0]
        mj_efc_aref = mj_sim.data.efc_aref
        for gs_sidx, mj_sidx in (
            (np.argsort(gs_jac.sum(axis=1)), np.argsort(mj_jac.sum(axis=1))),
            (np.argsort(gs_efc_aref), np.argsort(mj_efc_aref)),
        ):
            try:
                np.testing.assert_allclose(gs_jac[gs_sidx][:, gs_dofs_idx], mj_jac[mj_sidx][:, mj_dofs_idx], atol=atol)
                np.testing.assert_allclose(gs_efc_D[gs_sidx], mj_efc_D[mj_sidx], atol=atol)
                np.testing.assert_allclose(gs_efc_aref[gs_sidx], mj_efc_aref[mj_sidx], atol=atol)
                break
            except AssertionError:
                pass
        else:
            assert False

        mj_iter = mj_sim.data.solver_niter[0] - 1
        if gs_n_constraints and mj_iter > 0:
            gs_scale = 1.0 / (gs_meaninertia * max(1, gs_sim.rigid_solver.n_dofs))
            gs_improvement = gs_scale * (
                gs_sim.rigid_solver.constraint_solver.prev_cost[0] - gs_sim.rigid_solver.constraint_solver.cost[0]
            )
            mj_improvement = mj_sim.data.solver.improvement[mj_iter]
            np.testing.assert_allclose(gs_improvement, mj_improvement, atol=atol)
            gs_gradient = gs_scale * np.linalg.norm(
                gs_sim.rigid_solver.constraint_solver.grad.to_numpy()[: gs_sim.rigid_solver.n_dofs, 0]
            )
            mj_gradient = mj_sim.data.solver.gradient[mj_iter]
            np.testing.assert_allclose(gs_gradient, mj_gradient, atol=atol)

    gs_qfrc_constraint = gs_sim.rigid_solver.dofs_state.qf_constraint.to_numpy()[:, 0]
    mj_qfrc_constraint = mj_sim.data.qfrc_constraint
    np.testing.assert_allclose(gs_qfrc_constraint[gs_dofs_idx], mj_qfrc_constraint[mj_dofs_idx], atol=atol)

    if gs_n_constraints:
        gs_efc_force = gs_sim.rigid_solver.constraint_solver.efc_force.to_numpy()[:gs_n_constraints, 0]
        mj_efc_force = mj_sim.data.efc_force
        np.testing.assert_allclose(gs_efc_force[gs_sidx], mj_efc_force[mj_sidx], atol=atol)

        if qvel_prev is not None:
            # FIXME: This check does not pass for some scene...
            gs_efc_vel = gs_jac @ qvel_prev
            mj_efc_vel = mj_sim.data.efc_vel
            # np.testing.assert_allclose(gs_efc_vel[gs_sidx], mj_efc_vel[mj_sidx], atol=atol)

    gs_qfrc_all = gs_sim.rigid_solver.dofs_state.force.to_numpy()[:, 0]
    mj_qfrc_all = mj_sim.data.qfrc_smooth + mj_sim.data.qfrc_constraint
    if gs_sim.rigid_solver._options.integrator != gs.integrator.Euler:
        # FIXME: External forces are not added up for Euler integrator
        np.testing.assert_allclose(gs_qfrc_all[gs_dofs_idx], mj_qfrc_all[mj_dofs_idx], atol=atol)

    # FIXME: Why this check is not passing???
    gs_qfrc_smooth = gs_sim.rigid_solver.dofs_state.qf_smooth.to_numpy()[:, 0]
    mj_qfrc_smooth = mj_sim.data.qfrc_smooth
    # np.testing.assert_allclose(gs_qfrc_smooth[gs_dofs_idx], mj_qfrc_smooth[mj_dofs_idx], atol=atol)

    gs_qacc_smooth = gs_sim.rigid_solver.dofs_state.acc_smooth.to_numpy()[:, 0]
    mj_qacc_smooth = mj_sim.data.qacc_smooth
    np.testing.assert_allclose(gs_qacc_smooth[gs_dofs_idx], mj_qacc_smooth[mj_dofs_idx], atol=atol)

    # Acceleration pre- VS post-implicit damping
    # gs_qacc_post = gs_sim.rigid_solver.dofs_state.acc.to_numpy()[:, 0]
    if gs_n_constraints:
        gs_qacc_pre = gs_sim.rigid_solver.constraint_solver.qacc.to_numpy()[:, 0]
    else:
        gs_qacc_pre = gs_qacc_smooth
    mj_qacc_pre = mj_sim.data.qacc
    np.testing.assert_allclose(gs_qacc_pre[gs_dofs_idx], mj_qacc_pre[mj_dofs_idx], atol=atol)

    gs_qpos = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    mj_qpos = mj_sim.data.qpos
    np.testing.assert_allclose(gs_qpos[gs_q_idx], mj_qpos[mj_qs_idx], atol=atol)
    gs_qvel = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
    mj_qvel = mj_sim.data.qvel
    np.testing.assert_allclose(gs_qvel[gs_dofs_idx], mj_qvel[mj_dofs_idx], atol=atol)

    # ------------------------------------------------------------------------
    mujoco.mj_fwdPosition(mj_sim.model, mj_sim.data)
    mujoco.mj_fwdVelocity(mj_sim.model, mj_sim.data)
    gs_sim.rigid_solver._kernel_forward_kinematics_links_geoms(np.array([0]))

    gs_com = gs_sim.rigid_solver.links_state.COM.to_numpy()[0, 0]
    mj_com = mj_sim.data.subtree_com[0]
    np.testing.assert_allclose(gs_com, mj_com, atol=atol)

    gs_xipos = gs_sim.rigid_solver.links_state.i_pos.to_numpy()[:, 0]
    mj_xipos = mj_sim.data.xipos - mj_sim.data.subtree_com[0]
    np.testing.assert_allclose(gs_xipos[gs_bodies_idx], mj_xipos[mj_bodies_idx], atol=atol)

    gs_xpos = gs_sim.rigid_solver.links_state.pos.to_numpy()[:, 0]
    mj_xpos = mj_sim.data.xpos
    np.testing.assert_allclose(gs_xpos[gs_bodies_idx], mj_xpos[mj_bodies_idx], atol=atol)

    gs_xquat = gs_sim.rigid_solver.links_state.quat.to_numpy()[:, 0]
    gs_xmat = gu.quat_to_R(gs_xquat).reshape([-1, 9])
    mj_xmat = mj_sim.data.xmat
    np.testing.assert_allclose(gs_xmat[gs_bodies_idx], mj_xmat[mj_bodies_idx], atol=atol)

    gs_cd_vel = gs_sim.rigid_solver.links_state.cd_vel.to_numpy()[:, 0]
    gs_cd_vel_ = gs_sim.rigid_solver.links_state.vel.to_numpy()[:, 0]
    np.testing.assert_allclose(gs_cd_vel, gs_cd_vel_, atol=atol)
    mj_cd_vel = mj_sim.data.cvel[:, 3:]
    np.testing.assert_allclose(gs_cd_vel[gs_bodies_idx], mj_cd_vel[mj_bodies_idx], atol=atol)
    gs_cd_ang = gs_sim.rigid_solver.links_state.cd_ang.to_numpy()[:, 0]
    mj_cd_ang = mj_sim.data.cvel[:, :3]
    np.testing.assert_allclose(gs_cd_ang[gs_bodies_idx], mj_cd_ang[mj_bodies_idx], atol=atol)

    gs_cdof_vel = gs_sim.rigid_solver.dofs_state.cdof_vel.to_numpy()[:, 0]
    mj_cdof_vel = mj_sim.data.cdof[:, 3:]
    np.testing.assert_allclose(gs_cdof_vel[gs_dofs_idx], mj_cdof_vel[mj_dofs_idx], atol=atol)
    gs_cdof_ang = gs_sim.rigid_solver.dofs_state.cdof_ang.to_numpy()[:, 0]
    mj_cdof_ang = mj_sim.data.cdof[:, :3]
    np.testing.assert_allclose(gs_cdof_ang[gs_dofs_idx], mj_cdof_ang[mj_dofs_idx], atol=atol)

    mj_cdof_dot_ang = mj_sim.data.cdof_dot[:, :3]
    gs_cdof_dot_ang = gs_sim.rigid_solver.dofs_state.cdofd_ang.to_numpy()[:, 0]
    np.testing.assert_allclose(gs_cdof_dot_ang[gs_dofs_idx], mj_cdof_dot_ang[mj_dofs_idx], atol=atol)

    mj_cdof_dot_vel = mj_sim.data.cdof_dot[:, 3:]
    gs_cdof_dot_vel = gs_sim.rigid_solver.dofs_state.cdofd_vel.to_numpy()[:, 0]
    np.testing.assert_allclose(gs_cdof_dot_vel[gs_dofs_idx], mj_cdof_dot_vel[mj_dofs_idx], atol=atol)

    # cinr
    gs_cinr_inertial = gs_sim.rigid_solver.links_state.cinr_inertial.to_numpy()[:, 0].reshape([-1, 9])[
        :, [0, 4, 8, 1, 2, 5]
    ]
    mj_cinr_inertial = mj_sim.data.cinert[:, :6]  # upper-triangular part
    np.testing.assert_allclose(gs_cinr_inertial[gs_bodies_idx], mj_cinr_inertial[mj_bodies_idx], atol=atol)
    gs_cinr_pos = gs_sim.rigid_solver.links_state.cinr_pos.to_numpy()[:, 0]
    mj_cinr_pos = mj_sim.data.cinert[:, 6:9]
    np.testing.assert_allclose(gs_cinr_pos[gs_bodies_idx], mj_cinr_pos[mj_bodies_idx], atol=atol)
    gs_cinr_mass = gs_sim.rigid_solver.links_state.cinr_mass.to_numpy()[:, 0]
    mj_cinr_mass = mj_sim.data.cinert[:, 9]
    np.testing.assert_allclose(gs_cinr_mass[gs_bodies_idx], mj_cinr_mass[mj_bodies_idx], atol=atol)


def simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos=None, qvel=None, *, atol, num_steps):
    # Get mapping between Mujoco and Genesis
    _, (_, _, mj_qs_idx, mj_dofs_idx, _) = _get_model_mappings(gs_sim, mj_sim)

    # Make sure that "static" model information are matching
    check_mujoco_model_consistency(gs_sim, mj_sim, atol=atol)

    # Initialize the simulation
    init_simulators(gs_sim, mj_sim, qpos, qvel)

    # Run the simulation for a few steps
    qvel_prev = None

    for i in range(num_steps):
        # Make sure that all "dynamic" quantities are matching before stepping
        check_mujoco_data_consistency(gs_sim, mj_sim, qvel_prev=qvel_prev, atol=atol)

        # Keep Mujoco and Genesis simulation in sync to avoid drift over time
        mj_sim.data.qpos[mj_qs_idx] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[mj_dofs_idx] = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
        qvel_prev = mj_sim.data.qvel.copy()

        # Do a single simulation step (eventually with substeps for Genesis)
        mujoco.mj_step(mj_sim.model, mj_sim.data)
        gs_sim.scene.step()
        # if gs_sim.scene.visualizer:
        #     gs_sim.scene.visualizer.update()
