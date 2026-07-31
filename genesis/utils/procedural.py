"""Procedurally generate simple articulated bodies as inline URDF or MJCF, for reuse in tests and examples."""

import math
import xml.etree.ElementTree as ET


def build_articulated_chain(n_links, link_radius, link_length, *, format="mjcf", density=1000.0):
    """Build an ``n_links`` pendulum chain of equal links joined by hinge joints, as an inline XML string.

    The returned string is ready to pass to ``gs.morphs.MJCF(file=...)`` / ``gs.morphs.URDF(file=...)``. Each link
    is ``link_length`` long and ``link_radius`` thick; consecutive links are connected by a revolute joint about
    the y axis. Both forms use cylinder links at the given ``density`` with matching mass and inertia.

    The MJCF form hinges its first link to the world, so it is a fixed-base ``n_links`` pendulum. The URDF form
    keeps an explicit ``base`` root link whose fixing follows the morph: pass ``gs.morphs.URDF(file=..., fixed=True)``
    for the same fixed-base pendulum, otherwise the base is free and the whole chain falls under gravity.

    Parameters
    ----------
    n_links : int
        Number of links in the chain (must be >= 1).
    link_radius : float
        Radius of each link.
    link_length : float
        Length of each link along its local axis.
    format : str
        Either "mjcf" or "urdf".
    density : float
        Solid density of each link.
    """
    if n_links < 1:
        raise ValueError(f"`n_links` must be >= 1, got {n_links}.")

    if format == "mjcf":
        root = ET.Element("mujoco", model="chain")
        worldbody = ET.SubElement(root, "worldbody")
        parent = worldbody
        for i in range(n_links):
            body = ET.SubElement(parent, "body", name=f"link_{i}", pos=f"0 0 {0.0 if i == 0 else -link_length}")
            ET.SubElement(body, "joint", name=f"joint_{i}", type="hinge", axis="0 1 0")
            ET.SubElement(
                body,
                "geom",
                type="cylinder",
                fromto=f"0 0 0 0 0 {-link_length}",
                size=f"{link_radius}",
                density=f"{density}",
            )
            parent = body
        return ET.tostring(root, encoding="unicode")

    if format == "urdf":
        root = ET.Element("robot", name="chain")
        ET.SubElement(root, "link", name="base")
        # Solid-cylinder inertial (URDF requires an explicit one).
        mass = density * math.pi * link_radius**2 * link_length
        i_transverse = mass * (3.0 * link_radius**2 + link_length**2) / 12.0
        i_axial = mass * link_radius**2 / 2.0
        parent_name = "base"
        for i in range(n_links):
            link_name = f"link_{i}"
            joint = ET.SubElement(root, "joint", name=f"joint_{i}", type="continuous")
            ET.SubElement(joint, "parent", link=parent_name)
            ET.SubElement(joint, "child", link=link_name)
            ET.SubElement(joint, "origin", xyz=f"0 0 {0.0 if i == 0 else -link_length}", rpy="0 0 0")
            ET.SubElement(joint, "axis", xyz="0 1 0")
            link = ET.SubElement(root, "link", name=link_name)
            for tag in ("visual", "collision"):
                node = ET.SubElement(link, tag)
                ET.SubElement(node, "origin", xyz=f"0 0 {-0.5 * link_length}", rpy="0 0 0")
                geometry = ET.SubElement(node, "geometry")
                ET.SubElement(geometry, "cylinder", radius=f"{link_radius}", length=f"{link_length}")
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz=f"0 0 {-0.5 * link_length}", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value=f"{mass}")
            ET.SubElement(
                inertial,
                "inertia",
                ixx=f"{i_transverse}",
                ixy="0",
                ixz="0",
                iyy=f"{i_transverse}",
                iyz="0",
                izz=f"{i_axial}",
            )
            parent_name = link_name
        return ET.tostring(root, encoding="unicode")

    raise ValueError(f"`format` must be 'mjcf' or 'urdf', got {format!r}.")
