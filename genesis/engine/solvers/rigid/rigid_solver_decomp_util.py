import taichi as ti

import genesis.utils.array_class as array_class


@ti.func
def func_wakeup_entity(
    i_e,
    i_b,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    if entities_state.hibernated[i_e, i_b]:
        entities_state.hibernated[i_e, i_b] = False
        n_awake_entities = ti.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
        rigid_global_info.awake_entities[n_awake_entities, i_b] = i_e

        for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
            dofs_state.hibernated[i_d, i_b] = False
            n_awake_dofs = ti.atomic_add(rigid_global_info.n_awake_dofs[i_b], 1)
            rigid_global_info.awake_dofs[n_awake_dofs, i_b] = i_d

        for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
            links_state.hibernated[i_l, i_b] = False
            n_awake_links = ti.atomic_add(rigid_global_info.n_awake_links[i_b], 1)
            rigid_global_info.awake_links[n_awake_links, i_b] = i_l