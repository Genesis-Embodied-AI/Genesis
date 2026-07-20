"""Coulomb breakaway under both friction cones, documenting the pyramidal cone's limitations.

A box and a torsional-friction sphere rest on a plane, so each contact carries the exact normal load m * g. A
generalized force (box, tangential) and torque (sphere, about the vertical axis) ramp up linearly across the Coulomb
limits mu * m * g and mu_torsional * m * g; the load at which each entity starts moving is the measured breakaway.
The pyramidal cone at its default 'impratio' of 1 mixes the normal direction into every friction row, so it creeps
well below the Coulomb limit and breaks away early; the elliptic cone (auto-resolved 'impratio' of 100) sticks until
the limit and tracks it closely. See the 'friction_cone' and 'impratio' options of 'gs.options.RigidOptions'.
"""

import argparse

import genesis as gs


GRAVITY = 9.81
# SAFETY_FACTOR scales the ramp end past the analytic Coulomb limit so the breakaway is bracketed on both cones,
# whether it lands below the limit (regularized pyramid) or slightly above it.
SAFETY_FACTOR = 1.5
SLIP_THRESHOLD = 0.05
N_STEPS = 300


def measure_breakaway(friction_cone, show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, -GRAVITY),
        ),
        rigid_options=gs.options.RigidOptions(
            friction_cone=friction_cone,
            enable_torsional_friction=True,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        gs.morphs.Plane(),
    )
    box = scene.add_entity(
        gs.morphs.Box(
            size=(0.2, 0.2, 0.2),
            pos=(0.0, 0.0, 0.1),
        ),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=0.1,
            pos=(0.6, 0.0, 0.1),
        ),
        material=gs.materials.Rigid(
            friction_torsional=0.05,
        ),
    )
    scene.build()

    box_load_coulomb = box.geoms[0].friction * float(box.get_mass()) * GRAVITY
    sphere_load_coulomb = sphere.geoms[0].friction_torsional * float(sphere.get_mass()) * GRAVITY
    for _ in range(50):
        scene.step()

    breakaway_ratios = [None, None]
    creep_speeds_half_load = [None, None]
    for i_step in range(N_STEPS):
        load_ratio = SAFETY_FACTOR * (i_step + 1) / N_STEPS
        box.control_dofs_force([load_ratio * box_load_coulomb, 0.0, 0.0, 0.0, 0.0, 0.0])
        sphere.control_dofs_force([0.0, 0.0, 0.0, 0.0, 0.0, load_ratio * sphere_load_coulomb])
        scene.step()
        speeds = (float(box.get_dofs_velocity()[..., 0]), float(sphere.get_dofs_velocity()[..., 5]))
        for i_e, speed in enumerate(speeds):
            if creep_speeds_half_load[i_e] is None and load_ratio >= 0.5:
                creep_speeds_half_load[i_e] = speed
            if breakaway_ratios[i_e] is None and speed > SLIP_THRESHOLD:
                breakaway_ratios[i_e] = load_ratio
    return breakaway_ratios, creep_speeds_half_load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    for friction_cone in (gs.friction_cone.pyramidal, gs.friction_cone.elliptic):
        breakaway_ratios, creep_speeds_half_load = measure_breakaway(friction_cone, args.vis)
        print(f"{friction_cone.name} cone:")
        for name, breakaway_ratio, creep_speed in zip(
            ("tangential (box)", "torsional (sphere)"), breakaway_ratios, creep_speeds_half_load
        ):
            breakaway = f"{breakaway_ratio:.2f} x Coulomb limit" if breakaway_ratio is not None else "above the ramp"
            print(f"  {name}: breakaway at {breakaway}, creep speed at half load {creep_speed:.4f}")


if __name__ == "__main__":
    main()
