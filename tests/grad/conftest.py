import xml.etree.ElementTree as ET

import pytest


def _add_hinge_arm(parent, body_name, pos, **joint_kwargs):
    """Add a 1-DOF hinge arm link (y-axis hinge, capsule geom, explicit inertia) and return its body element."""
    body = ET.SubElement(parent, "body", name=body_name, pos=pos)
    ET.SubElement(body, "joint", type="hinge", axis="0 1 0", **joint_kwargs)
    ET.SubElement(body, "inertial", mass="0.5", pos="0.1 0 0", diaginertia="0.01 0.01 0.01")
    ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.02", contype="0", conaffinity="0")
    return body


@pytest.fixture(scope="session")
def grad_free():
    mjcf = ET.Element("mujoco", model="free")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="chassis", pos="0 0 0")
    ET.SubElement(body, "freejoint")
    ET.SubElement(body, "inertial", mass="1.0", pos="0 0 0", diaginertia="0.1 0.1 0.1")
    ET.SubElement(body, "geom", type="box", size="0.1 0.1 0.1", contype="0", conaffinity="0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_revolute():
    mjcf = ET.Element("mujoco", model="revolute")
    worldbody = ET.SubElement(mjcf, "worldbody")
    _add_hinge_arm(worldbody, "arm", "0 0 0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_revolute_frictionloss():
    mjcf = ET.Element("mujoco", model="revolute_frictionloss")
    worldbody = ET.SubElement(mjcf, "worldbody")
    _add_hinge_arm(worldbody, "arm", "0 0 0", frictionloss="0.5")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_prismatic():
    mjcf = ET.Element("mujoco", model="prismatic")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="slider", pos="0 0 0")
    ET.SubElement(body, "joint", type="slide", axis="1 0 0")
    ET.SubElement(body, "inertial", mass="0.5", pos="0 0 0", diaginertia="0.01 0.01 0.01")
    ET.SubElement(body, "geom", type="box", size="0.05 0.05 0.05", contype="0", conaffinity="0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_spherical():
    mjcf = ET.Element("mujoco", model="spherical")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="ball", pos="0 0 0")
    ET.SubElement(body, "joint", type="ball")
    ET.SubElement(body, "inertial", mass="0.5", pos="0.1 0 0", diaginertia="0.01 0.01 0.01")
    ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.02", contype="0", conaffinity="0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_capsule():
    mjcf = ET.Element("mujoco", model="capsule")
    ET.SubElement(mjcf, "compiler", angle="degree")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="capsule", pos="0 0 0")
    ET.SubElement(body, "geom", type="capsule", size="0.1 0.2")
    ET.SubElement(body, "joint", name="capsule_joint", type="free")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_free_with_revolute():
    mjcf = ET.Element("mujoco", model="free_with_child")
    worldbody = ET.SubElement(mjcf, "worldbody")
    chassis = ET.SubElement(worldbody, "body", name="chassis", pos="0 0 0")
    ET.SubElement(chassis, "freejoint")
    ET.SubElement(chassis, "inertial", mass="1.0", pos="0 0 0", diaginertia="0.1 0.1 0.1")
    ET.SubElement(chassis, "geom", type="box", size="0.1 0.1 0.1", contype="0", conaffinity="0")
    arm = ET.SubElement(chassis, "body", name="arm", pos="0.2 0 0")
    ET.SubElement(arm, "joint", type="hinge", axis="0 1 0")
    ET.SubElement(arm, "inertial", mass="0.5", pos="0.1 0 0", diaginertia="0.01 0.01 0.01")
    ET.SubElement(arm, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.02", contype="0", conaffinity="0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_revolute_chain3():
    mjcf = ET.Element("mujoco", model="chain3")
    worldbody = ET.SubElement(mjcf, "worldbody")
    parent = worldbody
    for name in ("l1", "l2", "l3"):
        body = ET.SubElement(parent, "body", name=name, pos="0 0 0" if name == "l1" else "0.2 0 0")
        ET.SubElement(body, "joint", type="hinge", axis="0 1 0")
        ET.SubElement(body, "inertial", mass="0.3", pos="0.1 0 0", diaginertia="0.005 0.005 0.005")
        ET.SubElement(body, "geom", type="capsule", fromto="0 0 0 0.2 0 0", size="0.02", contype="0", conaffinity="0")
        parent = body
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_slider_limit():
    mjcf = ET.Element("mujoco", model="slider_limit")
    ET.SubElement(mjcf, "option", gravity="0 0 0")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name="cart", pos="0 0 0")
    ET.SubElement(body, "joint", name="slider", type="slide", axis="1 0 0", range="-4 4", damping="0.0")
    ET.SubElement(body, "inertial", pos="0 0 0", mass="1.0", diaginertia="1.0 1.0 1.0")
    ET.SubElement(body, "geom", type="box", size="0.25 0.25 0.1", contype="0", conaffinity="0")
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_hinge_pair_joint_eq_linear():
    mjcf = ET.Element("mujoco", model="hinge_pair_joint_eq_linear")
    worldbody = ET.SubElement(mjcf, "worldbody")
    arm1 = _add_hinge_arm(worldbody, "arm1", "0 0 0", name="j1")
    _add_hinge_arm(arm1, "arm2", "0.2 0 0", name="j2")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(
        equality, "joint", joint1="j1", joint2="j2", polycoef="0 1 0 0 0", solimp="0.95 0.99 0.001", solref="0.005 1"
    )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_hinge_pair_joint_eq_quadratic():
    mjcf = ET.Element("mujoco", model="hinge_pair_joint_eq_quadratic")
    worldbody = ET.SubElement(mjcf, "worldbody")
    arm1 = _add_hinge_arm(worldbody, "arm1", "0 0 0", name="j1")
    _add_hinge_arm(arm1, "arm2", "0.2 0 0", name="j2")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(
        equality, "joint", joint1="j1", joint2="j2", polycoef="0 1 0.5 0 0", solimp="0.95 0.99 0.001", solref="0.005 1"
    )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_connect_loop():
    mjcf = ET.Element("mujoco", model="connect_loop")
    worldbody = ET.SubElement(mjcf, "worldbody")
    _add_hinge_arm(worldbody, "arm1", "0 0 0", name="j1")
    _add_hinge_arm(worldbody, "arm2", "0 0.3 0", name="j2")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(
        equality, "connect", body1="arm1", body2="arm2", anchor="0.2 0 0", solimp="0.95 0.99 0.001", solref="0.005 1"
    )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_weld_pair():
    mjcf = ET.Element("mujoco", model="weld_pair")
    worldbody = ET.SubElement(mjcf, "worldbody")
    _add_hinge_arm(worldbody, "arm1", "0 0 0", name="j1")
    _add_hinge_arm(worldbody, "arm2", "0 0.3 0", name="j2")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(
        equality,
        "weld",
        body1="arm1",
        body2="arm2",
        relpose="0 -0.3 0 1 0 0 0",
        solimp="0.95 0.99 0.001",
        solref="0.005 1",
    )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_all_eq_fric():
    # Integration scene exercising every differentiated constraint group: frictionloss on j1, equality JOINT between
    # j1 and j2, equality CONNECT between arm3 and arm4, equality WELD between arm5 and arm6. Each group acts on a
    # disjoint pair of links so the constraint solver faces a well-posed system within every pair.
    mjcf = ET.Element("mujoco", model="all_eq_fric")
    worldbody = ET.SubElement(mjcf, "worldbody")
    _add_hinge_arm(worldbody, "arm1", "0 0 0", name="j1", frictionloss="0.5")
    for i_arm in range(2, 7):
        _add_hinge_arm(worldbody, f"arm{i_arm}", f"0 {0.2 * (i_arm - 1):.1f} 0", name=f"j{i_arm}")
    equality = ET.SubElement(mjcf, "equality")
    ET.SubElement(
        equality, "joint", joint1="j1", joint2="j2", polycoef="0 1 0 0 0", solimp="0.95 0.99 0.001", solref="0.005 1"
    )
    ET.SubElement(
        equality, "connect", body1="arm3", body2="arm4", anchor="0.2 0 0", solimp="0.95 0.99 0.001", solref="0.005 1"
    )
    ET.SubElement(
        equality,
        "weld",
        body1="arm5",
        body2="arm6",
        relpose="0 -0.2 0 1 0 0 0",
        solimp="0.95 0.99 0.001",
        solref="0.005 1",
    )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_cartpole():
    mjcf = ET.Element("mujoco", model="cartpole")
    ET.SubElement(mjcf, "option", gravity="0 0 -9.81")
    worldbody = ET.SubElement(mjcf, "worldbody")
    cart = ET.SubElement(worldbody, "body", name="cart", pos="0 0 0")
    ET.SubElement(cart, "joint", name="slider", type="slide", axis="1 0 0", range="-4 4", damping="0.0")
    ET.SubElement(cart, "inertial", pos="0 0 0", mass="1.0", diaginertia="1.0 1.0 1.0")
    ET.SubElement(cart, "geom", type="box", size="0.25 0.25 0.1", contype="0", conaffinity="0", rgba="0 0 0.8 1")
    pole = ET.SubElement(cart, "body", name="pole", pos="0 0 0")
    ET.SubElement(pole, "joint", name="hinge", type="hinge", axis="0 1 0", damping="0.0")
    ET.SubElement(pole, "inertial", pos="0 0 0.5", mass="10.0", diaginertia="1.0 1.0 1.0")
    ET.SubElement(
        pole, "geom", type="box", pos="0 0 0.5", size="0.025 0.025 0.5", contype="0", conaffinity="0", rgba="1 1 1 1"
    )
    return ET.tostring(mjcf, encoding="unicode")


@pytest.fixture(scope="session")
def grad_hopper():
    mjcf = ET.Element("mujoco", model="hopper")
    ET.SubElement(mjcf, "compiler", angle="radian")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "joint", limited="true", armature="1", damping="1")
    ET.SubElement(default, "geom", condim="3", friction="0.9 0.005 0.0001")
    worldbody = ET.SubElement(mjcf, "worldbody")
    torso = ET.SubElement(worldbody, "body", name="torso", pos="0 0 1.25")
    root_kwargs = {"pos": "0 0 0", "limited": "false", "armature": "0", "damping": "0"}
    ET.SubElement(torso, "joint", name="rootx", axis="1 0 0", type="slide", **root_kwargs)
    ET.SubElement(torso, "joint", name="rootz", axis="0 0 1", type="slide", **root_kwargs)
    ET.SubElement(torso, "joint", name="rooty", axis="0 1 0", type="hinge", **root_kwargs)
    ET.SubElement(torso, "geom", name="torso_geom", type="capsule", size="0.05 0.2")
    thigh = ET.SubElement(torso, "body", name="thigh", pos="0 0 -0.2")
    ET.SubElement(thigh, "joint", name="thigh_joint", pos="0 0 0", axis="0 -1 0", type="hinge", range="-2.61799 0")
    ET.SubElement(thigh, "geom", name="thigh_geom", type="capsule", size="0.05 0.225", pos="0 0 -0.225")
    leg = ET.SubElement(thigh, "body", name="leg", pos="0 0 -0.7")
    ET.SubElement(leg, "joint", name="leg_joint", pos="0 0 0.25", axis="0 -1 0", type="hinge", range="-2.61799 0")
    ET.SubElement(leg, "geom", name="leg_geom", type="capsule", size="0.04 0.25")
    foot = ET.SubElement(leg, "body", name="foot", pos="0 0 -0.25")
    ET.SubElement(
        foot, "joint", name="foot_joint", pos="0 0 0", axis="0 -1 0", type="hinge", range="-0.785398 0.785398"
    )
    ET.SubElement(
        foot,
        "geom",
        name="foot_geom",
        type="capsule",
        size="0.06 0.195",
        pos="0.06 0 0",
        quat="0.707107 0 -0.707107 0",
        friction="2 0.005 0.0001",
    )
    return ET.tostring(mjcf, encoding="unicode")
