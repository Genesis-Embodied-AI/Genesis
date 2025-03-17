from typing import Literal
from itertools import chain
from dataclasses import dataclass

import numpy as np
import mujoco
import genesis as gs


@dataclass
class MjSim:
    model: mujoco.MjModel
    data: mujoco.MjData


def init_simulators(gs_sim, mj_sim, qpos, qvel):
    (gs_robot,) = gs_sim.entities

    gs_sim.scene.reset()
    gs_robot.set_qpos(qpos)
    gs_robot.set_dofs_velocity(qvel)
    # TODO: This should be moved in `set_state`, `set_qpos`, `set_dofs_position`, `set_dofs_velocity`
    gs_sim.rigid_solver._kernel_forward_dynamics()
    gs_sim.rigid_solver._func_constraint_force()
    if gs_sim.scene.visualizer:
        gs_sim.scene.visualizer.update()

    mujoco.mj_resetData(mj_sim.model, mj_sim.data)
    mj_sim.data.qpos[:] = qpos
    mj_sim.data.qvel[:] = qvel
    mujoco.mj_forward(mj_sim.model, mj_sim.data)


def _gs_search_by_joint_names(
    scene,
    joint_names: str | list[str],
    to: Literal["entity", "index"] = "index",
    is_local: bool = False,
    flatten: bool = True,
):
    if isinstance(joint_names, str):
        joint_names = [joint_names]

    for entity in scene.entities:
        try:
            gs_dof_idcs = dict()
            valid_joint_names = []
            for v in entity.joints:
                valid_joint_names.append(v.name)
                if v.name in joint_names:
                    if to == "entity":
                        gs_dof_idcs[v.name] = v
                    elif to == "index":
                        gs_dof_idcs[v.name] = v.dof_idx_local if is_local else v.dof_idx
                    else:
                        raise ValueError(f"Cannot recognize what ({to}) to extract for the search")

            missing_joint_names = set(joint_names) - gs_dof_idcs.keys()
            if len(missing_joint_names) > 0:
                raise ValueError(
                    f"Cannot find joints `{missing_joint_names}`. Valid joints names are {valid_joint_names}"
                )

            if flatten:
                out_fl = []
                for k in joint_names:
                    if isinstance(gs_dof_idcs[k], list):
                        out_fl.extend(gs_dof_idcs[k])
                    else:
                        out_fl.append(gs_dof_idcs[k])
                return out_fl

            return gs_dof_idcs
        except ValueError:
            pass
    else:
        raise ValueError(f"Fail to find joint indices for {joint_names}")


def _gs_search_by_link_names(
    scene,
    link_names: str | list[str],
    to: Literal["entity", "index"] = "index",
    is_local: bool = False,
    flatten: bool = True,
):
    if isinstance(link_names, str):
        link_names = [link_names]

    for entity in scene.entities:
        try:
            gs_link_idcs = dict()
            valid_link_names = []
            for v in entity.links:
                valid_link_names.append(v.name)
                if v.name in link_names:
                    if to == "entity":
                        gs_link_idcs[v.name] = v
                    elif to == "index":
                        gs_link_idcs[v.name] = v.idx_local if is_local else v.idx
                    else:
                        raise ValueError(f"Cannot recognize what ({to}) to extract for the search")

            missing_link_names = set(link_names) - gs_link_idcs.keys()
            if len(missing_link_names) > 0:
                raise ValueError(f"Cannot find links `{missing_link_names}`. Valid link names are {valid_link_names}")

            if flatten:
                return [gs_link_idcs[k] for k in link_names]

            return gs_link_idcs
        except ValueError:
            pass
    else:
        raise ValueError(f"Fail to find link indices for {link_names}")


def check_mujoco_model_consistency(
    gs_sim,
    mj_sim,
    joint_names: list[str] | None = None,
    body_names: list[str] | None = None,
    atol: float = 1e-9,
):
    # Extract joint and dof mapping from names to indices
    if joint_names is None:
        joint_names = [
            joint.name
            for entity in gs_sim.entities
            for joint in entity.joints
            if joint.name != "world" and joint.type != gs.JOINT_TYPE.FIXED
        ]
    if body_names is None:
        body_names = [body.name for entity in gs_sim.entities for body in entity.links]
    mj_dof_idcs: list[int] = []
    for joint_name in joint_names:
        mj_joint_j = mj_sim.model.joint(joint_name)
        mj_type_j = mj_sim.model.jnt_type[mj_joint_j.id]
        if mj_type_j == mujoco.mjtJoint.mjJNT_HINGE:
            n_dofs_j = 1
        elif mj_type_j == mujoco.mjtJoint.mjJNT_SLIDE:
            n_dofs_j = 1
        elif mj_type_j == mujoco.mjtJoint.mjJNT_BALL:
            n_dofs_j = 3
        elif mj_type_j == mujoco.mjtJoint.mjJNT_FREE:
            n_dofs_j = 6
        else:
            assert False
        mj_dof_start_j = mj_sim.model.jnt_dofadr[mj_joint_j.id]
        mj_dof_idcs += range(mj_dof_start_j, mj_dof_start_j + n_dofs_j)
    mj_body_idcs = [mj_sim.model.body(body_name).id for body_name in body_names]
    gs_dof_idcs = _gs_search_by_joint_names(gs_sim.scene, joint_names)
    gs_body_idcs = _gs_search_by_link_names(gs_sim.scene, body_names)

    # solver
    gs_gravity = gs_sim.rigid_solver.scene.gravity
    mj_gravity = mj_sim.model.opt.gravity
    np.testing.assert_allclose(gs_gravity, mj_gravity, atol=atol)
    assert mj_sim.model.opt.timestep == gs_sim.rigid_solver.substep_dt
    assert mj_sim.model.opt.tolerance == gs_sim.rigid_solver._options.tolerance
    assert mj_sim.model.opt.iterations == gs_sim.rigid_solver._options.iterations

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
    for gs_i, mj_i in zip(gs_body_idcs, mj_body_idcs):
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
    np.testing.assert_allclose(gs_dof_damping[gs_dof_idcs], mj_dof_damping[mj_dof_idcs], atol=atol)

    gs_dof_armature = gs_sim.rigid_solver.dofs_info.armature.to_numpy()
    mj_dof_armature = mj_sim.model.dof_armature
    np.testing.assert_allclose(gs_dof_armature[gs_dof_idcs], mj_dof_armature[mj_dof_idcs], atol=atol)

    # FIXME: 1 stiffness per joint in Mujoco, 1 stiffness per DoF in Genesis
    gs_dof_stiffness = gs_sim.rigid_solver.dofs_info.stiffness.to_numpy()
    mj_dof_stiffness = mj_sim.model.jnt_stiffness
    # np.testing.assert_allclose(gs_dof_stiffness[gs_dof_idcs], mj_dof_stiffness[mj_jnt_idcs], atol=atol)

    gs_dof_invweight0 = gs_sim.rigid_solver.dofs_info.invweight.to_numpy()
    mj_dof_invweight0 = mj_sim.model.dof_invweight0
    np.testing.assert_allclose(gs_dof_invweight0[gs_dof_idcs], mj_dof_invweight0[mj_dof_idcs], atol=atol)

    gs_solparams = gs_sim.rigid_solver.dofs_info.sol_params.to_numpy()
    gs_solref = gs_solparams[:, :2]
    mj_solref = mj_sim.model.dof_solref
    np.testing.assert_allclose(gs_solref[gs_dof_idcs], mj_solref[mj_dof_idcs], atol=atol)
    gs_solimp = gs_solparams[:, 2:]
    mj_solimp = mj_sim.model.dof_solimp
    np.testing.assert_allclose(gs_solimp[gs_dof_idcs], mj_solimp[mj_dof_idcs], atol=atol)

    # actuator (position control)
    for v in mj_sim.model.actuator_dyntype:
        assert mujoco.mjtDyn(v) == mujoco.mjtDyn.mjDYN_NONE
    for v in mj_sim.model.actuator_biastype:
        assert mujoco.mjtBias(v) == mujoco.mjtBias.mjBIAS_AFFINE

    # FIXME: kp/kv are defined for all DoFs in Genesis, only actuated joints in Mujoco
    assert (mj_sim.model.actuator_biastype == mujoco.mjtBias.mjBIAS_AFFINE).all()
    mj_kp = -mj_sim.model.actuator_biasprm[:, 1]  # NOTE: not considering gear
    mj_kv = -mj_sim.model.actuator_biasprm[:, 2]
    gs_kp = gs_sim.rigid_solver.dofs_info.kp.to_numpy()
    gs_kv = gs_sim.rigid_solver.dofs_info.kv.to_numpy()
    # np.testing.assert_allclose(gs_kp[gs_dof_idcs], mj_kp[mj_act_idcs], atol=atol)
    # np.testing.assert_allclose(gs_kv[gs_dof_idcs], mj_kv[mj_act_idcs], atol=atol)


def _mj_get_mass_matrix_from_sparse(mj_model, mj_data):
    is_, js, madr_ijs = [], [], []
    for i in range(mj_model.nv):
        madr_ij, j = mj_model.dof_Madr[i], i
        while True:
            madr_ij, j = madr_ij + 1, mj_model.dof_parentid[j]
            if j == -1:
                break
            is_, js, madr_ijs = (
                is_ + [i],
                js + [j],
                madr_ijs + [madr_ij],
            )
    i, j, madr_ij = (np.array(x, dtype=np.int32) for x in (is_, js, madr_ijs))
    mat = np.zeros((mj_model.nv, mj_model.nv))
    mat[i, j] = mj_data.qM[madr_ij]
    mat = np.diag(mj_data.qM[np.array(mj_model.dof_Madr)]) + mat + mat.T
    return mat


def check_mujoco_data_consistency(gs_sim, mj_sim, is_first_step, qvel_prev, atol: float = 1e-9):
    gs_meaninertia = gs_sim.rigid_solver.meaninertia.to_numpy()[0]
    mj_meaninertia = mj_sim.model.stat.meaninertia
    np.testing.assert_allclose(gs_meaninertia, mj_meaninertia, atol=atol)

    # crb
    gs_crb_inertial = gs_sim.rigid_solver.links_state.crb_inertial.to_numpy()[:-1, 0].reshape([-1, 9])[
        :, [0, 4, 8, 1, 2, 5]
    ]
    mj_crb_inertial = mj_sim.data.crb[1:, :6]  # upper-triangular part
    np.testing.assert_allclose(gs_crb_inertial, mj_crb_inertial, atol=atol)
    gs_crb_pos = gs_sim.rigid_solver.links_state.crb_pos.to_numpy()[:-1, 0]
    mj_crb_pos = mj_sim.data.crb[1:, 6:9]
    np.testing.assert_allclose(gs_crb_pos, mj_crb_pos, atol=atol)
    gs_crb_mass = gs_sim.rigid_solver.links_state.crb_mass.to_numpy()[:-1, 0]
    mj_crb_mass = mj_sim.data.crb[1:, 9]
    np.testing.assert_allclose(gs_crb_mass, mj_crb_mass, atol=atol)

    gs_mass_mat_damped = gs_sim.rigid_solver.mass_mat.to_numpy()[:, :, 0]
    mj_mass_mat_damped = _mj_get_mass_matrix_from_sparse(mj_sim.model, mj_sim.data)
    np.testing.assert_allclose(gs_mass_mat_damped, mj_mass_mat_damped, atol=atol)

    # FIXME: Why this check is not passing???
    gs_cd_vel = gs_sim.rigid_solver.links_state.cd_vel.to_numpy()[:-1, 0]
    mj_cd_vel = mj_sim.data.cvel[1:, :3]
    # np.testing.assert_allclose(gs_cd_vel, mj_cd_vel, atol=atol)
    gs_cd_ang = gs_sim.rigid_solver.links_state.cd_ang.to_numpy()[:-1, 0]
    mj_cd_ang = mj_sim.data.cvel[1:, 3:]
    # np.testing.assert_allclose(gs_cd_ang, mj_cd_ang, atol=atol)

    gs_qfrc_bias = gs_sim.rigid_solver.dofs_state.qf_bias.to_numpy()[:, 0]
    mj_qfrc_bias = mj_sim.data.qfrc_bias
    np.testing.assert_allclose(gs_qfrc_bias, mj_qfrc_bias, atol=atol)
    gs_qfrc_passive = gs_sim.rigid_solver.dofs_state.qf_passive.to_numpy()[:, 0]
    mj_qfrc_passive = mj_sim.data.qfrc_passive
    np.testing.assert_allclose(gs_qfrc_passive, mj_qfrc_passive, atol=atol)
    gs_qfrc_actuator = gs_sim.rigid_solver.dofs_state.qf_applied.to_numpy()[:, 0]
    mj_qfrc_actuator = mj_sim.data.qfrc_actuator
    np.testing.assert_allclose(gs_qfrc_actuator, mj_qfrc_actuator, atol=atol)

    if is_first_step:
        gs_n_constraints = 0
        mj_n_constraints = 0
    else:
        gs_n_contacts = gs_sim.rigid_solver.collider.n_contacts.to_numpy()[0]
        mj_n_contacts = mj_sim.data.ncon
        assert gs_n_contacts == mj_n_contacts
        gs_n_constraints = gs_sim.rigid_solver.constraint_solver.n_constraints.to_numpy()[0]
        mj_n_constraints = mj_sim.data.nefc
        assert gs_n_constraints == mj_n_constraints

    if gs_n_constraints:
        gs_contact_pos = gs_sim.rigid_solver.collider.contact_data.pos.to_numpy()[:gs_n_contacts, 0]
        mj_contact_pos = mj_sim.data.contact.pos
        # gs_sim.scene.draw_debug_sphere(pos=gs_contact_pos[0], radius=0.001, color=(1, 0, 0, 1))
        # gs_sim.scene.draw_debug_sphere(pos=mj_contact_pos[0], radius=0.001, color=(0, 1, 0, 1))
        # gs_sim.scene.viewer._pyrender_viewer._run()
        gs_sidx = np.argsort(gs_contact_pos[:, 0])
        mj_sidx = np.argsort(mj_contact_pos[:, 0])
        np.testing.assert_allclose(gs_contact_pos[gs_sidx], mj_contact_pos[mj_sidx], atol=atol)
        gs_contact_normal = gs_sim.rigid_solver.collider.contact_data.normal.to_numpy()[:gs_n_contacts, 0]
        mj_contact_normal = -mj_sim.data.contact.frame[:, :3]
        np.testing.assert_allclose(gs_contact_normal[gs_sidx], mj_contact_normal[mj_sidx], atol=atol)
        gs_penetration = gs_sim.rigid_solver.collider.contact_data.penetration.to_numpy()[:gs_n_contacts, 0]
        mj_penetration = -mj_sim.data.contact.dist
        np.testing.assert_allclose(gs_penetration[gs_sidx], mj_penetration[mj_sidx], atol=atol)

        gs_efc_D = gs_sim.rigid_solver.constraint_solver.efc_D.to_numpy()[:gs_n_constraints, 0]
        mj_efc_D = mj_sim.data.efc_D
        np.testing.assert_allclose(
            gs_efc_D[[j for i in gs_sidx for j in range(i * 4, (i + 1) * 4)]],
            mj_efc_D[[j for i in mj_sidx for j in range(i * 4, (i + 1) * 4)]],
            atol=atol,
        )

        gs_jac = gs_sim.rigid_solver.constraint_solver.jac.to_numpy()[:gs_n_constraints, :, 0]
        mj_jac = mj_sim.data.efc_J.reshape([gs_n_constraints, -1])
        gs_sidx = np.argsort(gs_jac.sum(axis=1))
        mj_sidx = np.argsort(mj_jac.sum(axis=1))
        np.testing.assert_allclose(gs_jac[gs_sidx], mj_jac[mj_sidx], atol=atol)

        gs_efc_aref = gs_sim.rigid_solver.constraint_solver.aref.to_numpy()[:gs_n_constraints, 0]
        mj_efc_aref = mj_sim.data.efc_aref
        np.testing.assert_allclose(gs_efc_aref[gs_sidx], mj_efc_aref[mj_sidx], atol=atol)

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
    np.testing.assert_allclose(gs_qfrc_constraint, mj_qfrc_constraint, atol=atol)

    if gs_n_constraints:
        gs_efc_force = gs_sim.rigid_solver.constraint_solver.efc_force.to_numpy()[:gs_n_constraints, 0]
        mj_efc_force = mj_sim.data.efc_force
        np.testing.assert_allclose(gs_efc_force[gs_sidx], mj_efc_force[mj_sidx], atol=atol)

        if qvel_prev is not None:
            gs_efc_vel = gs_jac @ qvel_prev
            mj_efc_vel = mj_sim.data.efc_vel
            np.testing.assert_allclose(gs_efc_vel[gs_sidx], mj_efc_vel[mj_sidx], atol=atol)

    gs_qfrc_all = gs_sim.rigid_solver.dofs_state.force.to_numpy()[:, 0]
    mj_qfrc_all = mj_sim.data.qfrc_smooth + mj_sim.data.qfrc_constraint
    np.testing.assert_allclose(gs_qfrc_all, mj_qfrc_all, atol=atol)

    # FIXME: Why this check is not passing???
    gs_qfrc_smooth = gs_sim.rigid_solver.dofs_state.qf_smooth.to_numpy()[:, 0]
    mj_qfrc_smooth = mj_sim.data.qfrc_smooth
    # np.testing.assert_allclose(gs_qfrc_smooth, mj_qfrc_smooth, atol=atol)

    gs_qacc_smooth = gs_sim.rigid_solver.dofs_state.acc_smooth.to_numpy()[:, 0]
    mj_qacc_smooth = mj_sim.data.qacc_smooth
    np.testing.assert_allclose(gs_qacc_smooth, mj_qacc_smooth, atol=atol)

    # Acceleration pre- VS post-implicit damping
    # gs_qacc_post = gs_sim.rigid_solver.dofs_state.acc.to_numpy()[:, 0]
    gs_qacc_pre = gs_sim.rigid_solver.constraint_solver.qacc.to_numpy()[:, 0]
    mj_qacc_pre = mj_sim.data.qacc
    np.testing.assert_allclose(gs_qacc_pre, mj_qacc_pre, atol=atol)

    gs_qpos = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
    mj_qpos = mj_sim.data.qpos
    np.testing.assert_allclose(gs_qpos, mj_qpos, atol=atol)
    gs_qvel = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
    mj_qvel = mj_sim.data.qvel
    np.testing.assert_allclose(gs_qvel, mj_qvel, atol=atol)

    # ------------------------------------------------------------------------
    mujoco.mj_fwdPosition(mj_sim.model, mj_sim.data)

    gs_xpos = gs_sim.rigid_solver.links_state.pos.to_numpy()[:-1, 0]
    mj_xpos = mj_sim.data.xpos[1:]
    np.testing.assert_allclose(gs_xpos, mj_xpos, atol=atol)

    gs_cdof_vel = gs_sim.rigid_solver.dofs_state.cdof_vel.to_numpy()[:, 0]
    mj_cdof_vel = mj_sim.data.cdof[:, 3:]
    np.testing.assert_allclose(gs_cdof_vel, mj_cdof_vel, atol=atol)
    gs_cdof_ang = gs_sim.rigid_solver.dofs_state.cdof_ang.to_numpy()[:, 0]
    mj_cdof_ang = mj_sim.data.cdof[:, :3]
    np.testing.assert_allclose(gs_cdof_ang, mj_cdof_ang, atol=atol)

    # cinr
    gs_cinr_inertial = gs_sim.rigid_solver.links_state.cinr_inertial.to_numpy()[:-1, 0].reshape([-1, 9])[
        :, [0, 4, 8, 1, 2, 5]
    ]
    mj_cinr_inertial = mj_sim.data.cinert[1:, :6]  # upper-triangular part
    np.testing.assert_allclose(gs_cinr_inertial, mj_cinr_inertial, atol=atol)
    gs_cinr_pos = gs_sim.rigid_solver.links_state.cinr_pos.to_numpy()[:-1, 0]
    mj_cinr_pos = mj_sim.data.cinert[1:, 6:9]
    np.testing.assert_allclose(gs_cinr_pos, mj_cinr_pos, atol=atol)
    gs_cinr_mass = gs_sim.rigid_solver.links_state.cinr_mass.to_numpy()[:-1, 0]
    mj_cinr_mass = mj_sim.data.cinert[1:, 9]
    np.testing.assert_allclose(gs_cinr_mass, mj_cinr_mass, atol=atol)


def simulate_and_check_mujoco_consistency(gs_sim, mj_sim, qpos, qvel, num_steps):
    check_mujoco_model_consistency(gs_sim, mj_sim)

    init_simulators(gs_sim, mj_sim, qpos, qvel)

    qvel_prev = None
    for i in range(num_steps):
        is_first_step = i == 0
        check_mujoco_data_consistency(gs_sim, mj_sim, is_first_step, qvel_prev)

        mj_sim.data.qpos[:] = gs_sim.rigid_solver.qpos.to_numpy()[:, 0]
        mj_sim.data.qvel[:] = gs_sim.rigid_solver.dofs_state.vel.to_numpy()[:, 0]
        qvel_prev = mj_sim.data.qvel.copy()
        mujoco.mj_step(mj_sim.model, mj_sim.data)
        gs_sim.scene.step()
        if gs_sim.scene.visualizer:
            gs_sim.scene.visualizer.update()
