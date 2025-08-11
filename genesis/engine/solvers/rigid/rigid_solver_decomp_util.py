import taichi as ti

import genesis.utils.array_class as array_class
from genesis.engine.solvers.rigid.contact_island import INVALID_HIBERNATED_ISLAND_ID
from genesis.engine.solvers.rigid.rigid_debug import Debug
from genesis.engine.solvers.rigid.rigid_validate import validate_temp_island_contains_both_hibernated_and_awake_entities




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
    contact_island: ti.template()
):
    # Note: Original function handled non-hibernated & fixed entities.
    # Now, we require a properly hibernated entity to be passed in.
    island_idx = contact_island.entity_island[i_e, i_b]
    is_entity_fixed = island_idx == -1
    Debug.assertf(0x0ad00007, not is_entity_fixed)  # Fixed entities are excluded from hibernation logic
    Debug.assertf(0x0ad0000b, entities_state.hibernated[i_e, i_b])

    # Note: temporarily, we have duplicated logic for hibernation: the entity_state.hibernated,
    # and the new contact_island's arrays that are used to store persistent_hibernated_island_id.
    hibernated_island_id = contact_island.hibernated_entity_idx_to_hibernated_island_id[i_e, i_b]
    is_entity_hibernated = hibernated_island_id != INVALID_HIBERNATED_ISLAND_ID
    Debug.assertf(0x0ad0000c, is_entity_hibernated)  # Entityt must belong to a persistent hibernated island

    if ti.static(Debug.validate):
        validate_temp_island_contains_both_hibernated_and_awake_entities(island_idx, i_b, entities_state, contact_island)

    n_entities = entities_info.n_links.shape[0]
    for entity_idx in range(n_entities): 


        if contact_island.hibernated_entity_idx_to_hibernated_island_id[entity_idx, i_b] == hibernated_island_id:
            Debug.assertf(0x0ad0000b, entities_state.hibernated[entity_idx, i_b])  # Entity expected to be hibernated

            contact_island.hibernated_entity_idx_to_hibernated_island_id[entity_idx, i_b] = INVALID_HIBERNATED_ISLAND_ID

            if entities_state.hibernated[entity_idx, i_b]:
                entities_state.hibernated[entity_idx, i_b] = False
                n_awake_entities = ti.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
                rigid_global_info.awake_entities[n_awake_entities, i_b] = entity_idx

                # todo: do single atomic add
                for i_d in range(entities_info.dof_start[entity_idx], entities_info.dof_end[entity_idx]):
                    dofs_state.hibernated[i_d, i_b] = False
                    n_awake_dofs = ti.atomic_add(rigid_global_info.n_awake_dofs[i_b], 1)
                    rigid_global_info.awake_dofs[n_awake_dofs, i_b] = i_d

                # todo: do single atomic add
                for i_l in range(entities_info.link_start[entity_idx], entities_info.link_end[entity_idx]):
                    links_state.hibernated[i_l, i_b] = False
                    n_awake_links = ti.atomic_add(rigid_global_info.n_awake_links[i_b], 1)
                    rigid_global_info.awake_links[n_awake_links, i_b] = i_l

                for i_g in range(entities_info.geom_start[entity_idx], entities_info.geom_end[entity_idx]):
                    geoms_state.hibernated[i_g, i_b] = False

                # validation only: un-hibernate the island
                non_persistent_hibernated_island_idx = contact_island.entity_island[entity_idx, i_b]
                Debug.assertf(0x0ad00008, non_persistent_hibernated_island_idx != -1)  # Entity being hibernated has invalid temp island index
                Debug.assertf(0x0ad00009, non_persistent_hibernated_island_idx == island_idx)  # Not matching island indices

                contact_island.island_hibernated[non_persistent_hibernated_island_idx, i_b] = False
