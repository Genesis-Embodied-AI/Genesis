import gstaichi as ti

import genesis.utils.array_class as array_class

from genesis.engine.solvers.rigid.contact_island import INVALID_NEXT_HIBERNATED_ENTITY_IDX


@ti.func
def func_wakeup_entity_and_its_temp_island(
    i_e,
    i_b,
    entities_state: array_class.EntitiesState,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    links_state: array_class.LinksState,
    geoms_state: array_class.GeomsState,
    rigid_global_info: array_class.RigidGlobalInfo,
    contact_island: ti.template(),
):
    # Note: Original function handled non-hibernated & fixed entities.
    # Now, we require a properly hibernated entity to be passed in.
    island_idx = contact_island.entity_island[i_e, i_b]

    entity_ref_range = contact_island.island_entity[island_idx, i_b]
    for ei in range(entity_ref_range.n):
        entity_ref = entity_ref_range.start + ei
        entity_idx = contact_island.entity_id[entity_ref, i_b]

        is_entity_hibernated = entities_state.hibernated[entity_idx, i_b]

        if is_entity_hibernated:
            contact_island.entity_idx_to_next_entity_idx_in_hibernated_island[entity_idx, i_b] = (
                INVALID_NEXT_HIBERNATED_ENTITY_IDX
            )

            entities_state.hibernated[entity_idx, i_b] = False
            n_awake_entities = ti.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
            rigid_global_info.awake_entities[n_awake_entities, i_b] = entity_idx

            n_dofs = entities_info.n_dofs[entity_idx]
            base_entity_dof_idx = entities_info.dof_start[entity_idx]
            base_awake_dof_idx = ti.atomic_add(rigid_global_info.n_awake_dofs[i_b], n_dofs)
            for i in range(n_dofs):
                i_d = base_entity_dof_idx + i
                dofs_state.hibernated[i_d, i_b] = False
                rigid_global_info.awake_dofs[base_awake_dof_idx + i, i_b] = i_d

            n_links = entities_info.n_links[entity_idx]
            base_entity_link_idx = entities_info.link_start[entity_idx]
            base_awake_link_idx = ti.atomic_add(rigid_global_info.n_awake_links[i_b], n_links)
            for i in range(n_links):
                i_l = base_entity_link_idx + i
                links_state.hibernated[i_l, i_b] = False
                rigid_global_info.awake_links[base_awake_link_idx + i, i_b] = i_l

            for i_g in range(entities_info.geom_start[entity_idx], entities_info.geom_end[entity_idx]):
                geoms_state.hibernated[i_g, i_b] = False
