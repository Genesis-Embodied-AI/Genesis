import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
import genesis.utils.geom as gu


@qd.func
def func_wakeup_entity_and_its_temp_island(
    i_e,
    i_b,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    contact_island_state: array_class.ContactIslandState,
):
    # Note: Original function handled non-hibernated & fixed entities.
    # Now, we require a properly hibernated entity to be passed in.
    island_idx = contact_island_state.entity_island[i_e, i_b]

    for ei in range(contact_island_state.island_entity.n[island_idx, i_b]):
        entity_ref = contact_island_state.island_entity.start[island_idx, i_b] + ei
        entity_idx = contact_island_state.entity_id[entity_ref, i_b]

        is_entity_hibernated = entities_state.hibernated[entity_idx, i_b]

        if is_entity_hibernated:
            contact_island_state.entity_idx_to_next_entity_idx_in_hibernated_island[entity_idx, i_b] = -1

            entities_state.hibernated[entity_idx, i_b] = False
            n_awake_entities = qd.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
            rigid_global_info.awake_entities[n_awake_entities, i_b] = entity_idx

            n_dofs = entities_info.n_dofs[entity_idx]
            base_entity_dof_idx = entities_info.dof_start[entity_idx]
            base_awake_dof_idx = qd.atomic_add(rigid_global_info.n_awake_dofs[i_b], n_dofs)
            for i in range(n_dofs):
                i_d = base_entity_dof_idx + i
                dofs_state.hibernated[i_d, i_b] = False
                rigid_global_info.awake_dofs[base_awake_dof_idx + i, i_b] = i_d

            n_links = entities_info.n_links[entity_idx]
            base_entity_link_idx = entities_info.link_start[entity_idx]
            base_awake_link_idx = qd.atomic_add(rigid_global_info.n_awake_links[i_b], n_links)
            for i in range(n_links):
                i_l = base_entity_link_idx + i
                links_state.hibernated[i_l, i_b] = False
                rigid_global_info.awake_links[base_awake_link_idx + i, i_b] = i_l

            for i_g in range(entities_info.geom_start[entity_idx], entities_info.geom_end[entity_idx]):
                geoms_state.hibernated[i_g, i_b] = False


# --------------------------------------------------------------------------------------
# Initialization kernels
# --------------------------------------------------------------------------------------


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_init_invweight(
    envs_idx: qd.types.ndarray(),
    links_invweight: qd.types.ndarray(),
    dofs_invweight: qd.types.ndarray(),
    links_info: array_class.LinksInfo,
    dofs_info: array_class.DofsInfo,
    force_update: qd.template(),
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]

    if qd.static(static_rigid_sim_config.batch_links_info):
        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_l, i_b_ in qd.ndrange(links_info.parent_idx.shape[0], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            for j in qd.static(range(2)):
                if force_update or links_info.invweight[i_l, i_b][j] < EPS:
                    links_info.invweight[i_l, i_b][j] = links_invweight[i_b_, i_l, j]
    else:
        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_l in range(links_info.parent_idx.shape[0]):
            for j in qd.static(range(2)):
                if force_update or links_info.invweight[i_l][j] < EPS:
                    links_info.invweight[i_l][j] = links_invweight[i_l, j]

    if qd.static(static_rigid_sim_config.batch_dofs_info):
        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_d, i_b_ in qd.ndrange(dofs_info.invweight.shape[0], envs_idx.shape[0]):
            i_b = envs_idx[i_b_]
            if force_update or dofs_info.invweight[i_d, i_b] < EPS:
                dofs_info.invweight[i_d, i_b] = dofs_invweight[i_b_, i_d]
    else:
        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_d in range(dofs_info.invweight.shape[0]):
            if force_update or dofs_info.invweight[i_d] < EPS:
                dofs_info.invweight[i_d] = dofs_invweight[i_d]


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_init_meaninertia(
    envs_idx: qd.types.ndarray(),
    rigid_global_info: array_class.RigidGlobalInfo,
    entities_info: array_class.EntitiesInfo,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = rigid_global_info.mass_mat.shape[0]
    n_entities = entities_info.n_links.shape[0]
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_b_ in range(envs_idx.shape[0]):
        i_b = envs_idx[i_b_]
        if n_dofs > 0:
            rigid_global_info.meaninertia[i_b] = 0.0
            for i_e in range(n_entities):
                for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    rigid_global_info.meaninertia[i_b] = (
                        rigid_global_info.meaninertia[i_b] + rigid_global_info.mass_mat[i_d, i_d, i_b]
                    )
                rigid_global_info.meaninertia[i_b] = rigid_global_info.meaninertia[i_b] / n_dofs
        else:
            rigid_global_info.meaninertia[i_b] = 1.0


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_init_dof_fields(
    # input np array
    entity_idx: qd.types.ndarray(),
    dofs_motion_ang: qd.types.ndarray(),
    dofs_motion_vel: qd.types.ndarray(),
    dofs_limit: qd.types.ndarray(),
    dofs_invweight: qd.types.ndarray(),
    dofs_stiffness: qd.types.ndarray(),
    dofs_damping: qd.types.ndarray(),
    dofs_frictionloss: qd.types.ndarray(),
    dofs_armature: qd.types.ndarray(),
    dofs_kp: qd.types.ndarray(),
    dofs_kv: qd.types.ndarray(),
    dofs_force_range: qd.types.ndarray(),
    # Quadrants variables
    dofs_info: array_class.DofsInfo,
    dofs_state: array_class.DofsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    n_dofs = dofs_state.ctrl_mode.shape[0]
    _B = dofs_state.ctrl_mode.shape[1]

    for I_d in qd.grouped(dofs_info.invweight):
        i_d = I_d[0]  # batching (if any) will be the second dim

        for j in qd.static(range(3)):
            dofs_info.motion_ang[I_d][j] = dofs_motion_ang[i_d, j]
            dofs_info.motion_vel[I_d][j] = dofs_motion_vel[i_d, j]

        for j in qd.static(range(2)):
            dofs_info.limit[I_d][j] = dofs_limit[i_d, j]
            dofs_info.force_range[I_d][j] = dofs_force_range[i_d, j]

        dofs_info.armature[I_d] = dofs_armature[i_d]
        dofs_info.invweight[I_d] = dofs_invweight[i_d]
        dofs_info.stiffness[I_d] = dofs_stiffness[i_d]
        dofs_info.damping[I_d] = dofs_damping[i_d]
        dofs_info.frictionloss[I_d] = dofs_frictionloss[i_d]
        dofs_info.kp[I_d] = dofs_kp[i_d]
        dofs_info.kv[I_d] = dofs_kv[i_d]
        dofs_info.entity_idx[I_d] = entity_idx[i_d]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        dofs_state.ctrl_mode[i_d, i_b] = gs.CTRL_MODE.FORCE
        dofs_state.ctrl_force[i_d, i_b] = gs.qd_float(0.0)

    if qd.static(static_rigid_sim_config.use_hibernation):
        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_d, i_b in qd.ndrange(n_dofs, _B):
            dofs_state.hibernated[i_d, i_b] = False
            rigid_global_info.awake_dofs[i_d, i_b] = i_d

        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_b in range(_B):
            rigid_global_info.n_awake_dofs[i_b] = n_dofs


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_init_link_fields(
    links_parent_idx: qd.types.ndarray(),
    links_root_idx: qd.types.ndarray(),
    links_q_start: qd.types.ndarray(),
    links_dof_start: qd.types.ndarray(),
    links_joint_start: qd.types.ndarray(),
    links_q_end: qd.types.ndarray(),
    links_dof_end: qd.types.ndarray(),
    links_joint_end: qd.types.ndarray(),
    links_invweight: qd.types.ndarray(),
    links_is_fixed: qd.types.ndarray(),
    links_pos: qd.types.ndarray(),
    links_quat: qd.types.ndarray(),
    links_inertial_pos: qd.types.ndarray(),
    links_inertial_quat: qd.types.ndarray(),
    links_inertial_i: qd.types.ndarray(),
    links_inertial_mass: qd.types.ndarray(),
    links_entity_idx: qd.types.ndarray(),
    links_geom_start: qd.types.ndarray(),
    links_geom_end: qd.types.ndarray(),
    links_vgeom_start: qd.types.ndarray(),
    links_vgeom_end: qd.types.ndarray(),
    # Quadrants variables
    links_info: array_class.LinksInfo,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    n_links = links_parent_idx.shape[0]
    _B = links_state.pos.shape[1]

    for I_l in qd.grouped(links_info.invweight):
        i_l = I_l[0]

        links_info.parent_idx[I_l] = links_parent_idx[i_l]
        links_info.root_idx[I_l] = links_root_idx[i_l]
        links_info.q_start[I_l] = links_q_start[i_l]
        links_info.joint_start[I_l] = links_joint_start[i_l]
        links_info.dof_start[I_l] = links_dof_start[i_l]
        links_info.q_end[I_l] = links_q_end[i_l]
        links_info.dof_end[I_l] = links_dof_end[i_l]
        links_info.joint_end[I_l] = links_joint_end[i_l]
        links_info.n_dofs[I_l] = links_dof_end[i_l] - links_dof_start[i_l]
        links_info.is_fixed[I_l] = links_is_fixed[i_l]
        links_info.entity_idx[I_l] = links_entity_idx[i_l]
        links_info.geom_start[I_l] = links_geom_start[i_l]
        links_info.geom_end[I_l] = links_geom_end[i_l]
        links_info.vgeom_start[I_l] = links_vgeom_start[i_l]
        links_info.vgeom_end[I_l] = links_vgeom_end[i_l]

        for j in qd.static(range(2)):
            links_info.invweight[I_l][j] = links_invweight[i_l, j]

        for j in qd.static(range(4)):
            links_info.quat[I_l][j] = links_quat[i_l, j]
            links_info.inertial_quat[I_l][j] = links_inertial_quat[i_l, j]

        for j in qd.static(range(3)):
            links_info.pos[I_l][j] = links_pos[i_l, j]
            links_info.inertial_pos[I_l][j] = links_inertial_pos[i_l, j]

        links_info.inertial_mass[I_l] = links_inertial_mass[i_l]
        for j1, j2 in qd.static(qd.ndrange(3, 3)):
            links_info.inertial_i[I_l][j1, j2] = links_inertial_i[i_l, j1, j2]

    for i_l, i_b in qd.ndrange(n_links, _B):
        I_l = [i_l, i_b] if qd.static(static_rigid_sim_config.batch_links_info) else i_l

        # Update state for root fixed link. Their state will not be updated in forward kinematics later but can be manually changed by user.
        if links_info.parent_idx[I_l] == -1 and links_info.is_fixed[I_l]:
            for j in qd.static(range(4)):
                links_state.quat[i_l, i_b][j] = links_quat[i_l, j]

            for j in qd.static(range(3)):
                links_state.pos[i_l, i_b][j] = links_pos[i_l, j]

        for j in qd.static(range(3)):
            links_state.i_pos_shift[i_l, i_b][j] = 0.0
        links_state.mass_shift[i_l, i_b] = 0.0

    if qd.static(static_rigid_sim_config.use_hibernation):
        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_l, i_b in qd.ndrange(n_links, _B):
            links_state.hibernated[i_l, i_b] = False
            rigid_global_info.awake_links[i_l, i_b] = i_l

        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_b in range(_B):
            rigid_global_info.n_awake_links[i_b] = n_links


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_update_heterogeneous_link_info(
    i_l: qd.i32,
    links_geom_start: qd.types.ndarray(),
    links_geom_end: qd.types.ndarray(),
    links_vgeom_start: qd.types.ndarray(),
    links_vgeom_end: qd.types.ndarray(),
    links_inertial_mass: qd.types.ndarray(),
    links_inertial_pos: qd.types.ndarray(),
    links_inertial_i: qd.types.ndarray(),
    # Quadrants variables
    links_info: array_class.LinksInfo,
):
    """Update per-environment link info for heterogeneous entities."""
    _B = links_geom_start.shape[0]

    for i_b in range(_B):
        links_info.geom_start[i_l, i_b] = links_geom_start[i_b]
        links_info.geom_end[i_l, i_b] = links_geom_end[i_b]
        links_info.vgeom_start[i_l, i_b] = links_vgeom_start[i_b]
        links_info.vgeom_end[i_l, i_b] = links_vgeom_end[i_b]
        links_info.inertial_mass[i_l, i_b] = links_inertial_mass[i_b]

        for j in qd.static(range(3)):
            links_info.inertial_pos[i_l, i_b][j] = links_inertial_pos[i_b, j]

        for j1, j2 in qd.static(qd.ndrange(3, 3)):
            links_info.inertial_i[i_l, i_b][j1, j2] = links_inertial_i[i_b, j1, j2]


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_init_joint_fields(
    joints_type: qd.types.ndarray(),
    joints_sol_params: qd.types.ndarray(),
    joints_q_start: qd.types.ndarray(),
    joints_dof_start: qd.types.ndarray(),
    joints_q_end: qd.types.ndarray(),
    joints_dof_end: qd.types.ndarray(),
    joints_pos: qd.types.ndarray(),
    # Quadrants variables
    joints_info: array_class.JointsInfo,
    static_rigid_sim_config: qd.template(),
):
    for I_j in qd.grouped(joints_info.type):
        i_j = I_j[0]

        joints_info.type[I_j] = joints_type[i_j]
        joints_info.q_start[I_j] = joints_q_start[i_j]
        joints_info.dof_start[I_j] = joints_dof_start[i_j]
        joints_info.q_end[I_j] = joints_q_end[i_j]
        joints_info.dof_end[I_j] = joints_dof_end[i_j]
        joints_info.n_dofs[I_j] = joints_dof_end[i_j] - joints_dof_start[i_j]

        for j in qd.static(range(7)):
            joints_info.sol_params[I_j][j] = joints_sol_params[i_j, j]
        for j in qd.static(range(3)):
            joints_info.pos[I_j][j] = joints_pos[i_j, j]


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_init_vert_fields(
    verts: qd.types.ndarray(),
    faces: qd.types.ndarray(),
    edges: qd.types.ndarray(),
    normals: qd.types.ndarray(),
    verts_geom_idx: qd.types.ndarray(),
    init_center_pos: qd.types.ndarray(),
    verts_state_idx: qd.types.ndarray(),
    is_fixed: qd.types.ndarray(),
    # Quadrants variables
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    edges_info: array_class.EdgesInfo,
    static_rigid_sim_config: qd.template(),
):
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]
    n_edges = edges.shape[0]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.PARTIAL))
    for i_v in range(n_verts):
        for j in qd.static(range(3)):
            verts_info.init_pos[i_v][j] = verts[i_v, j]
            verts_info.init_normal[i_v][j] = normals[i_v, j]
            verts_info.init_center_pos[i_v][j] = init_center_pos[i_v, j]

        verts_info.geom_idx[i_v] = verts_geom_idx[i_v]
        verts_info.verts_state_idx[i_v] = verts_state_idx[i_v]
        verts_info.is_fixed[i_v] = is_fixed[i_v]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_f in range(n_faces):
        for j in qd.static(range(3)):
            faces_info.verts_idx[i_f][j] = faces[i_f, j]
        faces_info.geom_idx[i_f] = verts_geom_idx[faces[i_f, 0]]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_ed in range(n_edges):
        edges_info.v0[i_ed] = edges[i_ed, 0]
        edges_info.v1[i_ed] = edges[i_ed, 1]
        # minus = verts_info.init_pos[edges[i_ed, 0]] - verts_info.init_pos[edges[i_ed, 1]]
        # edges_info.length[i_ed] = minus.norm()
        # FIXME: the line below does not work
        edges_info.length[i_ed] = (verts_info.init_pos[edges[i_ed, 0]] - verts_info.init_pos[edges[i_ed, 1]]).norm()


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_init_vvert_fields(
    vverts: qd.types.ndarray(),
    vfaces: qd.types.ndarray(),
    vnormals: qd.types.ndarray(),
    vverts_vgeom_idx: qd.types.ndarray(),
    # Quadrants variables
    vverts_info: array_class.VVertsInfo,
    vfaces_info: array_class.VFacesInfo,
    static_rigid_sim_config: qd.template(),
):
    n_vverts = vverts.shape[0]
    n_vfaces = vfaces.shape[0]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_vv in range(n_vverts):
        for j in qd.static(range(3)):
            vverts_info.init_pos[i_vv][j] = vverts[i_vv, j]
            vverts_info.init_vnormal[i_vv][j] = vnormals[i_vv, j]

        vverts_info.vgeom_idx[i_vv] = vverts_vgeom_idx[i_vv]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_vf in range(n_vfaces):
        for j in qd.static(range(3)):
            vfaces_info.vverts_idx[i_vf][j] = vfaces[i_vf, j]
        vfaces_info.vgeom_idx[i_vf] = vverts_vgeom_idx[vfaces[i_vf, 0]]


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_init_geom_fields(
    geoms_pos: qd.types.ndarray(),
    geoms_center: qd.types.ndarray(),
    geoms_quat: qd.types.ndarray(),
    geoms_link_idx: qd.types.ndarray(),
    geoms_type: qd.types.ndarray(),
    geoms_friction: qd.types.ndarray(),
    geoms_sol_params: qd.types.ndarray(),
    geoms_vert_start: qd.types.ndarray(),
    geoms_face_start: qd.types.ndarray(),
    geoms_edge_start: qd.types.ndarray(),
    geoms_verts_state_start: qd.types.ndarray(),
    geoms_vert_end: qd.types.ndarray(),
    geoms_face_end: qd.types.ndarray(),
    geoms_edge_end: qd.types.ndarray(),
    geoms_verts_state_end: qd.types.ndarray(),
    geoms_data: qd.types.ndarray(),
    geoms_is_convex: qd.types.ndarray(),
    geoms_needs_coup: qd.types.ndarray(),
    geoms_contype: qd.types.ndarray(),
    geoms_conaffinity: qd.types.ndarray(),
    geoms_coup_softness: qd.types.ndarray(),
    geoms_coup_friction: qd.types.ndarray(),
    geoms_coup_restitution: qd.types.ndarray(),
    geoms_is_fixed: qd.types.ndarray(),
    geoms_is_decomp: qd.types.ndarray(),
    # Quadrants variables
    geoms_info: array_class.GeomsInfo,
    geoms_state: array_class.GeomsState,
    verts_info: array_class.VertsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,  # TODO: move to rigid global info
    static_rigid_sim_config: qd.template(),
):
    n_geoms = geoms_pos.shape[0]
    _B = geoms_state.friction_ratio.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g in range(n_geoms):
        for j in qd.static(range(3)):
            geoms_info.pos[i_g][j] = geoms_pos[i_g, j]
            geoms_info.center[i_g][j] = geoms_center[i_g, j]

        for j in qd.static(range(4)):
            geoms_info.quat[i_g][j] = geoms_quat[i_g, j]

        for j in qd.static(range(7)):
            geoms_info.data[i_g][j] = geoms_data[i_g, j]
            geoms_info.sol_params[i_g][j] = geoms_sol_params[i_g, j]

        geoms_info.vert_start[i_g] = geoms_vert_start[i_g]
        geoms_info.vert_end[i_g] = geoms_vert_end[i_g]
        geoms_info.vert_num[i_g] = geoms_vert_end[i_g] - geoms_vert_start[i_g]

        geoms_info.face_start[i_g] = geoms_face_start[i_g]
        geoms_info.face_end[i_g] = geoms_face_end[i_g]
        geoms_info.face_num[i_g] = geoms_face_end[i_g] - geoms_face_start[i_g]

        geoms_info.edge_start[i_g] = geoms_edge_start[i_g]
        geoms_info.edge_end[i_g] = geoms_edge_end[i_g]
        geoms_info.edge_num[i_g] = geoms_edge_end[i_g] - geoms_edge_start[i_g]

        geoms_info.verts_state_start[i_g] = geoms_verts_state_start[i_g]
        geoms_info.verts_state_end[i_g] = geoms_verts_state_end[i_g]

        geoms_info.link_idx[i_g] = geoms_link_idx[i_g]
        geoms_info.type[i_g] = geoms_type[i_g]
        geoms_info.friction[i_g] = geoms_friction[i_g]

        geoms_info.is_convex[i_g] = geoms_is_convex[i_g]
        geoms_info.needs_coup[i_g] = geoms_needs_coup[i_g]
        geoms_info.contype[i_g] = geoms_contype[i_g]
        geoms_info.conaffinity[i_g] = geoms_conaffinity[i_g]

        geoms_info.coup_softness[i_g] = geoms_coup_softness[i_g]
        geoms_info.coup_friction[i_g] = geoms_coup_friction[i_g]
        geoms_info.coup_restitution[i_g] = geoms_coup_restitution[i_g]

        geoms_info.is_fixed[i_g] = geoms_is_fixed[i_g]
        geoms_info.is_decomposed[i_g] = geoms_is_decomp[i_g]

        # compute init AABB.
        # Beware the ordering the this corners is critical and MUST NOT be changed as this order is used elsewhere
        # in the codebase, e.g. overlap estimation between two convex geometries using there bounding boxes.
        lower = gu.qd_vec3(qd.math.inf)
        upper = gu.qd_vec3(-qd.math.inf)
        for i_v in range(geoms_vert_start[i_g], geoms_vert_end[i_g]):
            lower = qd.min(lower, verts_info.init_pos[i_v])
            upper = qd.max(upper, verts_info.init_pos[i_v])
        geoms_init_AABB[i_g, 0] = qd.Vector([lower[0], lower[1], lower[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 1] = qd.Vector([lower[0], lower[1], upper[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 2] = qd.Vector([lower[0], upper[1], lower[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 3] = qd.Vector([lower[0], upper[1], upper[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 4] = qd.Vector([upper[0], lower[1], lower[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 5] = qd.Vector([upper[0], lower[1], upper[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 6] = qd.Vector([upper[0], upper[1], lower[2]], dt=gs.qd_float)
        geoms_init_AABB[i_g, 7] = qd.Vector([upper[0], upper[1], upper[2]], dt=gs.qd_float)

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_g, i_b in qd.ndrange(n_geoms, _B):
        geoms_state.friction_ratio[i_g, i_b] = 1.0


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_adjust_link_inertia(
    link_idx: qd.i32,
    ratio: qd.f32,
    links_info: array_class.LinksInfo,
    static_rigid_sim_config: qd.template(),
):
    if qd.static(static_rigid_sim_config.batch_links_info):
        for i_b in range(links_info.root_idx.shape[0]):
            for j in qd.static(range(2)):
                links_info.invweight[link_idx, i_b][j] /= ratio
            links_info.inertial_mass[link_idx, i_b] *= ratio
            for j1, j2 in qd.static(qd.ndrange(3, 3)):
                links_info.inertial_i[link_idx, i_b][j1, j2] *= ratio
    else:
        for j in qd.static(range(2)):
            links_info.invweight[link_idx][j] /= ratio
        links_info.inertial_mass[link_idx] *= ratio
        for j1, j2 in qd.static(qd.ndrange(3, 3)):
            links_info.inertial_i[link_idx][j1, j2] *= ratio


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_init_vgeom_fields(
    vgeoms_pos: qd.types.ndarray(),
    vgeoms_quat: qd.types.ndarray(),
    vgeoms_link_idx: qd.types.ndarray(),
    vgeoms_vvert_start: qd.types.ndarray(),
    vgeoms_vface_start: qd.types.ndarray(),
    vgeoms_vvert_end: qd.types.ndarray(),
    vgeoms_vface_end: qd.types.ndarray(),
    vgeoms_color: qd.types.ndarray(),
    # Quadrants variables
    vgeoms_info: array_class.VGeomsInfo,
    static_rigid_sim_config: qd.template(),
):
    n_vgeoms = vgeoms_pos.shape[0]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_vg in range(n_vgeoms):
        for j in qd.static(range(3)):
            vgeoms_info.pos[i_vg][j] = vgeoms_pos[i_vg, j]

        for j in qd.static(range(4)):
            vgeoms_info.quat[i_vg][j] = vgeoms_quat[i_vg, j]

        vgeoms_info.vvert_start[i_vg] = vgeoms_vvert_start[i_vg]
        vgeoms_info.vvert_end[i_vg] = vgeoms_vvert_end[i_vg]
        vgeoms_info.vvert_num[i_vg] = vgeoms_vvert_end[i_vg] - vgeoms_vvert_start[i_vg]

        vgeoms_info.vface_start[i_vg] = vgeoms_vface_start[i_vg]
        vgeoms_info.vface_end[i_vg] = vgeoms_vface_end[i_vg]
        vgeoms_info.vface_num[i_vg] = vgeoms_vface_end[i_vg] - vgeoms_vface_start[i_vg]

        vgeoms_info.link_idx[i_vg] = vgeoms_link_idx[i_vg]
        for j in qd.static(range(4)):
            vgeoms_info.color[i_vg][j] = vgeoms_color[i_vg, j]


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_init_entity_fields(
    entities_dof_start: qd.types.ndarray(),
    entities_dof_end: qd.types.ndarray(),
    entities_link_start: qd.types.ndarray(),
    entities_link_end: qd.types.ndarray(),
    entities_geom_start: qd.types.ndarray(),
    entities_geom_end: qd.types.ndarray(),
    entities_gravity_compensation: qd.types.ndarray(),
    entities_is_local_collision_mask: qd.types.ndarray(),
    # Quadrants variables
    entities_info: array_class.EntitiesInfo,
    entities_state: array_class.EntitiesState,
    links_info: array_class.LinksInfo,
    dofs_info: array_class.DofsInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    n_entities = entities_dof_start.shape[0]
    _B = entities_state.hibernated.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_e in range(n_entities):
        entities_info.dof_start[i_e] = entities_dof_start[i_e]
        entities_info.dof_end[i_e] = entities_dof_end[i_e]
        entities_info.n_dofs[i_e] = entities_dof_end[i_e] - entities_dof_start[i_e]

        entities_info.link_start[i_e] = entities_link_start[i_e]
        entities_info.link_end[i_e] = entities_link_end[i_e]
        entities_info.n_links[i_e] = entities_link_end[i_e] - entities_link_start[i_e]

        entities_info.geom_start[i_e] = entities_geom_start[i_e]
        entities_info.geom_end[i_e] = entities_geom_end[i_e]
        entities_info.n_geoms[i_e] = entities_geom_end[i_e] - entities_geom_start[i_e]

        entities_info.gravity_compensation[i_e] = entities_gravity_compensation[i_e]
        entities_info.is_local_collision_mask[i_e] = entities_is_local_collision_mask[i_e]

    if qd.static(static_rigid_sim_config.use_hibernation):
        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_e, i_b in qd.ndrange(n_entities, _B):
            entities_state.hibernated[i_e, i_b] = False
            rigid_global_info.awake_entities[i_e, i_b] = i_e

        qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
        for i_b in range(_B):
            rigid_global_info.n_awake_entities[i_b] = n_entities


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_init_equality_fields(
    equalities_type: qd.types.ndarray(),
    equalities_eq_obj1id: qd.types.ndarray(),
    equalities_eq_obj2id: qd.types.ndarray(),
    equalities_eq_data: qd.types.ndarray(),
    equalities_eq_type: qd.types.ndarray(),
    equalities_sol_params: qd.types.ndarray(),
    # Quadrants variables
    equalities_info: array_class.EqualitiesInfo,
    static_rigid_sim_config: qd.template(),
):
    n_equalities = equalities_eq_obj1id.shape[0]
    _B = equalities_info.eq_obj1id.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_eq, i_b in qd.ndrange(n_equalities, _B):
        equalities_info.eq_obj1id[i_eq, i_b] = equalities_eq_obj1id[i_eq]
        equalities_info.eq_obj2id[i_eq, i_b] = equalities_eq_obj2id[i_eq]
        equalities_info.eq_type[i_eq, i_b] = equalities_eq_type[i_eq]
        for j in qd.static(range(11)):
            equalities_info.eq_data[i_eq, i_b][j] = equalities_eq_data[i_eq, j]
        for j in qd.static(range(7)):
            equalities_info.sol_params[i_eq, i_b][j] = equalities_sol_params[i_eq, j]


# --------------------------------------------------------------------------------------
# External force kernels
# --------------------------------------------------------------------------------------


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_apply_links_external_force(
    force: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    ref: qd.template(),
    local: qd.template(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        force_i = qd.Vector([force[i_b_, i_l_, 0], force[i_b_, i_l_, 1], force[i_b_, i_l_, 2]], dt=gs.qd_float)
        func_apply_link_external_force(force_i, links_idx[i_l_], envs_idx[i_b_], ref, local, links_state)


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_apply_links_external_torque(
    torque: qd.types.ndarray(),
    links_idx: qd.types.ndarray(),
    envs_idx: qd.types.ndarray(),
    ref: qd.template(),
    local: qd.template(),
    links_state: array_class.LinksState,
    static_rigid_sim_config: qd.template(),
):
    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_l_, i_b_ in qd.ndrange(links_idx.shape[0], envs_idx.shape[0]):
        torque_i = qd.Vector([torque[i_b_, i_l_, 0], torque[i_b_, i_l_, 1], torque[i_b_, i_l_, 2]], dt=gs.qd_float)
        func_apply_link_external_torque(torque_i, links_idx[i_l_], envs_idx[i_b_], ref, local, links_state)


@qd.func
def func_apply_coupling_force(pos, force, link_idx, env_idx, links_state: array_class.LinksState):
    torque = (pos - links_state.root_COM[link_idx, env_idx]).cross(force)
    links_state.cfrc_coupling_ang[link_idx, env_idx] -= torque
    links_state.cfrc_coupling_vel[link_idx, env_idx] -= force


@qd.func
def func_apply_link_external_force(
    force,
    link_idx,
    env_idx,
    ref: qd.template(),
    local: qd.template(),
    links_state: array_class.LinksState,
):
    torque = qd.Vector.zero(gs.qd_float, 3)
    if qd.static(ref == 1):  # link's CoM
        if qd.static(local == 1):
            force = gu.qd_transform_by_quat(force, links_state.i_quat[link_idx, env_idx])
        torque = links_state.i_pos[link_idx, env_idx].cross(force)
    if qd.static(ref == 2):  # link's origin
        if qd.static(local == 1):
            force = gu.qd_transform_by_quat(force, links_state.i_quat[link_idx, env_idx])
        torque = (links_state.pos[link_idx, env_idx] - links_state.root_COM[link_idx, env_idx]).cross(force)

    links_state.cfrc_applied_vel[link_idx, env_idx] -= force
    links_state.cfrc_applied_ang[link_idx, env_idx] -= torque


@qd.func
def func_apply_external_torque(self, torque, link_idx, env_idx):
    self.links_state.cfrc_applied_ang[link_idx, env_idx] -= torque


@qd.func
def func_apply_link_external_torque(
    torque,
    link_idx,
    env_idx,
    ref: qd.template(),
    local: qd.template(),
    links_state: array_class.LinksState,
):
    if qd.static(ref == 1 and local == 1):  # link's CoM
        torque = gu.qd_transform_by_quat(torque, links_state.i_quat[link_idx, env_idx])
    if qd.static(ref == 2 and local == 1):  # link's origin
        torque = gu.qd_transform_by_quat(torque, links_state.quat[link_idx, env_idx])

    links_state.cfrc_applied_ang[link_idx, env_idx] -= torque


@qd.func
def func_clear_external_force(
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    n_links = links_state.pos.shape[0]
    _B = links_state.pos.shape[1]

    qd.loop_config(serialize=qd.static(static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL))
    for i_0, i_b in (
        qd.ndrange(1, _B) if qd.static(static_rigid_sim_config.use_hibernation) else qd.ndrange(n_links, _B)
    ):
        for i_1 in (
            range(rigid_global_info.n_awake_links[i_b])
            if qd.static(static_rigid_sim_config.use_hibernation)
            else qd.static(range(1))
        ):
            i_l = rigid_global_info.awake_links[i_1, i_b] if qd.static(static_rigid_sim_config.use_hibernation) else i_0
            links_state.cfrc_applied_ang[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)
            links_state.cfrc_applied_vel[i_l, i_b] = qd.Vector.zero(gs.qd_float, 3)


# --------------------------------------------------------------------------------------
# Render transform kernels
# --------------------------------------------------------------------------------------


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_update_geoms_render_T(
    geoms_render_T: qd.types.ndarray(),
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_geoms = geoms_state.pos.shape[0]
    _B = geoms_state.pos.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_g, i_b in qd.ndrange(n_geoms, _B):
        geom_T = gu.qd_trans_quat_to_T(
            geoms_state.pos[i_g, i_b] + rigid_global_info.envs_offset[i_b], geoms_state.quat[i_g, i_b], EPS
        )
        if (qd.abs(geom_T) < 1e20).all():
            for J in qd.static(qd.grouped(qd.ndrange(4, 4))):
                geoms_render_T[(i_g, i_b, *J)] = qd.cast(geom_T[J], qd.float32)


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_update_vgeoms_render_T(
    vgeoms_render_T: qd.types.ndarray(),
    vgeoms_info: array_class.VGeomsInfo,
    vgeoms_state: array_class.VGeomsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    EPS = rigid_global_info.EPS[None]

    n_vgeoms = vgeoms_info.link_idx.shape[0]
    _B = links_state.pos.shape[1]
    qd.loop_config(serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL)
    for i_g, i_b in qd.ndrange(n_vgeoms, _B):
        geom_T = gu.qd_trans_quat_to_T(
            vgeoms_state.pos[i_g, i_b] + rigid_global_info.envs_offset[i_b], vgeoms_state.quat[i_g, i_b], EPS
        )
        if (qd.abs(geom_T) < 1e20).all():
            for J in qd.static(qd.grouped(qd.ndrange(4, 4))):
                vgeoms_render_T[(i_g, i_b, *J)] = qd.cast(geom_T[J], qd.float32)


# --------------------------------------------------------------------------------------
# Utility kernels and functions
# --------------------------------------------------------------------------------------


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_bit_reduction(tensor: array_class.V_ANNOTATION) -> qd.i32:
    flag = qd.i32(0)
    for i in range(tensor.shape[0]):
        flag = qd.atomic_or(flag, tensor[i])
    return flag


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_set_zero(envs_idx: qd.types.ndarray(), tensor: array_class.V_ANNOTATION):
    for i_b in range(envs_idx.shape[0]):
        tensor[i_b] = 0


@qd.func
def func_atomic_add_if(field: array_class.V_ANNOTATION, I, value, cond: qd.template()):
    if qd.static(cond):
        qd.atomic_add(field[I], value)
    return value


@qd.func
def func_add_safe_backward(field: array_class.V_ANNOTATION, I, value, cond: qd.template()):
    # Use (expensive) atomic add in backward for differentiability -- when there is race condition on the field to
    # write, use atomic add directly. For reference, see official Quadrants documentation:
    # https://docs.taichi-lang.org/docs/differentiable_programming#global-data-access-rules
    if qd.static(cond):
        qd.atomic_add(field[I], value)
    else:
        field[I] = field[I] + value


@qd.func
def func_read_field_if(field: array_class.V_ANNOTATION, I, value, cond: qd.template()):
    return field[I] if qd.static(cond) else value


@qd.func
def func_write_field_if(field: array_class.V_ANNOTATION, I, value, cond: qd.template()):
    if qd.static(cond):
        field[I] = value
    return value


@qd.func
def func_write_and_read_field_if(field: array_class.V_ANNOTATION, I, value, cond: qd.template()):
    if qd.static(cond):
        field[I] = value
    return field[I] if qd.static(cond) else value


@qd.func
def func_check_index_range(idx: qd.i32, min: qd.i32, max: qd.i32, cond: qd.template()):
    # Conditionally check if the index is in the range [min, max) to save computational cost
    return (idx >= min and idx < max) if qd.static(cond) else True


@qd.kernel(fastcache=gs.use_fastcache)
def kernel_clear_external_force(
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    func_clear_external_force(
        links_state=links_state,
        rigid_global_info=rigid_global_info,
        static_rigid_sim_config=static_rigid_sim_config,
    )


from genesis.utils.deprecated_module_wrapper import create_virtual_deprecated_module

create_virtual_deprecated_module(__name__, "genesis.engine.solvers.rigid.rigid_solver_util_decomp")
