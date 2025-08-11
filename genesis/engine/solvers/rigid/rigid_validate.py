import taichi as ti

import genesis.utils.array_class as array_class

from genesis.engine.solvers.rigid.rigid_debug import Debug


@ti.func
def validate_entity_hibernation_state_for_all_entities_in_temp_island(
    temp_island_idx: ti.i32,
    i_b: ti.i32,
    entities_state: array_class.EntitiesState,
    contact_island: ti.template(), # ContactIsland,
    expected_hibernation_state: ti.u1,
):
    ci = contact_island
    entity_ref_range = ci.island_entity[temp_island_idx, i_b]
    all_okay = True
    error_count = 0
    for i in range(entity_ref_range.n):
        entity_ref = entity_ref_range.start + i
        entity_idx = ci.entity_id[entity_ref, i_b]
        all_okay = all_okay and entities_state.hibernated[entity_idx, i_b] == expected_hibernation_state
        if entities_state.hibernated[entity_idx, i_b] != expected_hibernation_state:
            error_count += 1
    Debug.assertf(0x7ad0000a, all_okay)  # Entity expected to be matching the expected hibernation state
    if not all_okay:
        print(f"Entity hibernation state mismatch for temp island {temp_island_idx} of {entity_ref_range.n} entities, expected: {expected_hibernation_state}, error_count: {error_count}")


@ti.func
def validate_temp_island_contains_both_hibernated_and_awake_entities(
    temp_island_idx: ti.i32,
    i_b: ti.i32,
    entities_state: array_class.EntitiesState,
    contact_island: ti.template()
):
    all_hibernated = True
    all_awake = True

    ci = contact_island
    entity_ref_range = ci.island_entity[temp_island_idx, i_b]
    for i in range(entity_ref_range.n):
        entity_ref = entity_ref_range.start + i
        entity_idx = ci.entity_id[entity_ref, i_b]
        is_entity_hibernated = entities_state.hibernated[entity_idx, i_b]
        all_awake = ti.u1(all_awake and not is_entity_hibernated)
        all_hibernated = ti.u1(all_hibernated and is_entity_hibernated)

    Debug.assertf(0x7ad00009, not (all_hibernated or all_awake))  # Island being woken up is expected to contain both hibernated and awake entities
