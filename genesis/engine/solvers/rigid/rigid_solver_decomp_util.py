import taichi as ti

import genesis.utils.array_class as array_class
from genesis.engine.solvers.rigid.contact_island import INVALID_HIBERNATED_ISLAND_ID

@ti.func
def func_wakeup_entity(
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
    island_idx = contact_island.entity_island[i_e, i_b]
    if island_idx != -1 and contact_island.island_hibernated[island_idx, i_b]:
        print(f"Internal error!?: Island {island_idx} hibernated but calling awake.")

    # Warning: can't run it as is -- this is run 30 timers for a stack of 4 boxes
    if entities_state.hibernated[i_e, i_b]:

        # check how many objects in the island
        island_idx = contact_island.entity_island[i_e, i_b]
        hibernated_island_id = contact_island.hibernated_entity_idx_to_hibernated_island_id[i_e, i_b]

        is_entity_hibernated = hibernated_island_id != INVALID_HIBERNATED_ISLAND_ID
        if ti.static(True): # Debug.config
            is_entity_fixed = island_idx == -1
            if is_entity_fixed and is_entity_hibernated:
                print(f"Internal error!?: Fixed entity in a hibernated island")
        
        if is_entity_hibernated:
            # check if either all hibernated or all awake
            all_hibernated = True
            all_awake = True
            for i in range(contact_island.island_entity[island_idx, i_b].n):
                entity_ref = contact_island.island_entity[island_idx, i_b].start + i
                i_e = contact_island.entity_id[entity_ref, i_b]
                is_entity_awake = not entities_state.hibernated[i_e, i_b]
                all_hibernated &= not is_entity_awake
                all_awake &= is_entity_awake
            if all_hibernated or all_awake:
                print(f"Internal error!?: All entities in island {island_idx} are already all hibernated or awake")

            # for i in range(contact_island.island_entity[island_idx, i_b].n):
            #     entity_ref = contact_island.island_entity[island_idx, i_b].start + i
            #     entity_idx = contact_island.entity_id[entity_ref, i_b]
            n_entities = entities_info.n_links.shape[0]
            n_awaken_entities = 0
            for entity_idx in range(n_entities): 
                if contact_island.hibernated_entity_idx_to_hibernated_island_id[entity_idx, i_b] == hibernated_island_id:
                    n_awaken_entities += 1
                    if ti.static(True): # Debug.config
                        if not entities_state.hibernated[entity_idx, i_b]:
                            print(f"Internal error!?: Entity {entity_idx} is not hibernated")

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

                        if non_persistent_hibernated_island_idx == -1:
                            print(f"Internal error!?: Non-persistent hibernated island idx is -1")

                        # note size is the single hibernated body + additional new non-hiberanted bodies in contact
                        non_persistent_hibernated_island_size = contact_island.island_entity[non_persistent_hibernated_island_idx, i_b].n
                        if 0 == non_persistent_hibernated_island_size:
                            print(f"Internal error!?: Non-persistent hibernated island size is zero")

                        contact_island.island_hibernated[non_persistent_hibernated_island_idx, i_b] = False
                        print(f"//Un-hibernate entity {entity_idx}")
                        
            print(f"Reactivate island id {hibernated_island_id} of {n_awaken_entities} entities")

        else:
            # check if entity is free or fixed

            # !? how to check if entity is fixed? looking at n_dofs only? or iterating through links and checking if their info is StructLinksInfo.is_fixed
            if entities_info.n_dofs[i_e] > 0:
                print(f"Entity {i_e} is not in any island")

    # if entities_state.hibernated[i_e, i_b]:
    #     entities_state.hibernated[i_e, i_b] = False
    #     n_awake_entities = ti.atomic_add(rigid_global_info.n_awake_entities[i_b], 1)
    #     rigid_global_info.awake_entities[n_awake_entities, i_b] = i_e

    #     for i_d in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
    #         dofs_state.hibernated[i_d, i_b] = False
    #         n_awake_dofs = ti.atomic_add(rigid_global_info.n_awake_dofs[i_b], 1)
    #         rigid_global_info.awake_dofs[n_awake_dofs, i_b] = i_d

    #     for i_l in range(entities_info.link_start[i_e], entities_info.link_end[i_e]):
    #         links_state.hibernated[i_l, i_b] = False
    #         n_awake_links = ti.atomic_add(rigid_global_info.n_awake_links[i_b], 1)
    #         rigid_global_info.awake_links[n_awake_links, i_b] = i_l

