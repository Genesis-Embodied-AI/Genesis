import os
import xml.etree.ElementTree as ET

import pytest

from genesis.utils.misc import get_assets_dir

from ..utils import get_hf_dataset


@pytest.fixture
def mesh_path(mesh_file):
    path = os.path.join(get_assets_dir(), mesh_file)
    if not os.path.exists(path):
        path = os.path.join(get_hf_dataset(pattern=mesh_file), mesh_file)
    return path


@pytest.fixture
def mesh_urdf(mesh_path):
    """URDF content wrapping the mesh as the visual geometry of a one-link robot."""
    robot = ET.Element("robot", name="model")
    geometry = ET.SubElement(ET.SubElement(ET.SubElement(robot, "link", name="base"), "visual"), "geometry")
    ET.SubElement(geometry, "mesh", filename=mesh_path)
    return ET.tostring(robot, encoding="unicode")
