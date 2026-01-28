"""
Entity Naming Tutorial
======================

This tutorial demonstrates the Genesis entity naming system:
1. Auto-generated names based on morph type + UID
2. User-specified custom names
3. Scene-level entity lookup APIs

Works with both Rigid and PBD (Position-Based Dynamics) entities.
"""

import argparse

import genesis as gs


def demo_rigid_naming(show_viewer: bool = False):
    """Demonstrate entity naming with Rigid bodies."""
    print("\n=== Rigid Entity Naming Demo ===\n")

    scene = gs.Scene(show_viewer=show_viewer)

    # Auto-generated names: {morph_type}_{uid_prefix}
    box = scene.add_entity(gs.morphs.Box(pos=(0, 0, 0.5), size=(0.2, 0.2, 0.2)))
    sphere = scene.add_entity(gs.morphs.Sphere(pos=(0.5, 0, 0.5), radius=0.1))
    print(f"Auto-generated names: box='{box.name}', sphere='{sphere.name}'")

    # User-specified names
    ground = scene.add_entity(gs.morphs.Plane(), name="ground")
    obstacle = scene.add_entity(gs.morphs.Cylinder(pos=(-0.5, 0, 0.3), radius=0.1, height=0.6), name="obstacle")
    print(f"User-specified names: ground='{ground.name}', obstacle='{obstacle.name}'")

    # Lookup by name
    retrieved = scene.get_entity(name="ground")
    print(f"Lookup by name: scene.get_entity(name='ground') -> {retrieved.name}")

    # Lookup by UID prefix
    uid_prefix = str(box.uid)[:4]
    retrieved = scene.get_entity(uid=uid_prefix)
    print(f"Lookup by UID prefix: scene.get_entity(uid='{uid_prefix}') -> {retrieved.name}")

    # List all entity names (creation order)
    print(f"All entity names: {scene.entity_names}")

    scene.build()
    scene.step()

    scene.destroy()
    print("Rigid demo complete.")


def demo_pbd_naming(show_viewer: bool = False):
    """Demonstrate entity naming with PBD (soft body) entities."""
    print("\n=== PBD Entity Naming Demo ===\n")

    scene = gs.Scene(
        pbd_options=gs.options.PBDOptions(particle_size=0.02),
        show_viewer=show_viewer,
    )

    # Rigid ground with user name
    ground = scene.add_entity(gs.morphs.Plane(), name="ground")

    # PBD entity: auto-name has 'pbd_' prefix
    cloth = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/cloth.obj",
            pos=(0, 0, 1.0),
            euler=(90, 0, 0),
        ),
        material=gs.materials.PBD.Cloth(),
        name="soft_cloth",
    )
    print(f"PBD entity with custom name: '{cloth.name}'")

    # List all names
    print(f"All entity names: {scene.entity_names}")

    # Lookup by name works for both rigid and PBD
    for name in scene.entity_names:
        entity = scene.get_entity(name=name)
        print(f"  {name}: {type(entity).__name__}")

    scene.build()
    scene.step()

    scene.destroy()
    print("PBD demo complete.")


def demo_file_based_naming(show_viewer: bool = False):
    """Demonstrate naming for file-based morphs (URDF, MJCF, Mesh)."""
    print("\n=== File-Based Morph Naming Demo ===\n")

    scene = gs.Scene(show_viewer=show_viewer)

    # File-based morphs use file stem as name prefix
    robot = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0, 0, 0.5)))
    print(f"URDF auto-name uses file stem: '{robot.name}'")

    bunny = scene.add_entity(gs.morphs.Mesh(file="meshes/bunny.obj", pos=(1, 0, 0.2), scale=0.3))
    print(f"Mesh auto-name uses file stem: '{bunny.name}'")

    # User can override with custom name
    ground = scene.add_entity(gs.morphs.Plane(), name="floor")

    print(f"All entity names: {scene.entity_names}")

    scene.build()
    scene.step()

    scene.destroy()
    print("File-based naming demo complete.")


def main():
    parser = argparse.ArgumentParser(description="Entity Naming Tutorial")
    parser.add_argument("--vis", action="store_true", help="Show viewer")
    parser.add_argument(
        "--demo",
        choices=["rigid", "pbd", "file", "all"],
        default="all",
        help="Which demo to run",
    )
    args = parser.parse_args()

    gs.init(backend=gs.cpu)

    if args.demo in ("rigid", "all"):
        demo_rigid_naming(args.vis)

    if args.demo in ("pbd", "all"):
        demo_pbd_naming(args.vis)

    if args.demo in ("file", "all"):
        demo_file_based_naming(args.vis)


if __name__ == "__main__":
    main()
