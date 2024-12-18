import os

import numpy as np

from ..constants import log, tol
from ..decomposition import convex_decomposition
from ..version import __version__


def export_urdf(mesh, directory, scale=1.0, color=None, **kwargs):
    """
    Convert a Trimesh object into a URDF package for physics
    simulation. This breaks the mesh into convex pieces and
    writes them to the same directory as the .urdf file.

    Parameters
    ---------
    mesh : trimesh.Trimesh
      Input geometry
    directory : str
      The directory path for the URDF package

    Returns
    ---------
    mesh : Trimesh
      Multi-body mesh containing convex decomposition
    """

    import lxml.etree as et

    # TODO: fix circular import
    from .export import export_mesh
    from ..resources import get

    # Extract the save directory and the file name
    fullpath = os.path.abspath(directory)
    name = os.path.basename(fullpath)
    _, ext = os.path.splitext(name)

    if ext != "":
        raise ValueError("URDF path must be a directory!")

    # Create directory if needed
    if not os.path.exists(fullpath):
        os.mkdir(fullpath)
    elif not os.path.isdir(fullpath):
        raise ValueError("URDF path must be a directory!")

    # Perform a convex decomposition
    try:
        convex_pieces = convex_decomposition(mesh, **kwargs)
        if not isinstance(convex_pieces, list):
            convex_pieces = [convex_pieces]
    except BaseException:
        log.error("problem with convex decomposition, using hull", exc_info=True)
        convex_pieces = [mesh.convex_hull]

    # Get the effective density of the mesh
    effective_density = mesh.volume / sum([m.volume for m in convex_pieces])

    # open an XML tree
    root = et.Element("robot", name="root")

    # Loop through all pieces, adding each as a link
    prev_link_name = None
    for i, piece in enumerate(convex_pieces):

        # Save each nearly convex mesh out to a file
        piece_name = "{}_convex_piece_{}".format(name, i)
        piece_filename = "{}.obj".format(piece_name)
        piece_filepath = os.path.join(fullpath, piece_filename)
        export_mesh(piece, piece_filepath)

        # Set the mass properties of the piece
        piece.center_mass = mesh.center_mass
        piece.density = effective_density * mesh.density

        link_name = "link_{}".format(piece_name)
        geom_name = "{}".format(piece_filename)
        I = [["{:.2E}".format(y) for y in x] for x in piece.moment_inertia]  # NOQA

        # Write the link out to the XML Tree
        link = et.SubElement(root, "link", name=link_name)

        # Inertial information
        inertial = et.SubElement(link, "inertial")
        et.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
        et.SubElement(inertial, "mass", value="{:.2E}".format(piece.mass))
        et.SubElement(inertial, "inertia", ixx=I[0][0], ixy=I[0][1], ixz=I[0][2], iyy=I[1][1], iyz=I[1][2], izz=I[2][2])
        # Visual Information
        visual = et.SubElement(link, "visual")
        et.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
        geometry = et.SubElement(visual, "geometry")
        et.SubElement(geometry, "mesh", filename=geom_name, scale="{:.4E} {:.4E} {:.4E}".format(scale, scale, scale))
        material = et.SubElement(visual, "material", name="")
        if color is not None:
            et.SubElement(material, "color", rgba="{:.2E} {:.2E} {:.2E} 1".format(color[0], color[1], color[2]))

        # Collision Information
        collision = et.SubElement(link, "collision")
        et.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
        geometry = et.SubElement(collision, "geometry")
        et.SubElement(geometry, "mesh", filename=geom_name, scale="{:.4E} {:.4E} {:.4E}".format(scale, scale, scale))

        # Create rigid joint to previous link
        if prev_link_name is not None:
            joint_name = "{}_joint".format(link_name)
            joint = et.SubElement(root, "joint", name=joint_name, type="fixed")
            et.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
            et.SubElement(joint, "parent", link=prev_link_name)
            et.SubElement(joint, "child", link=link_name)

        prev_link_name = link_name

    # Write URDF file
    tree = et.ElementTree(root)
    urdf_filename = "{}.urdf".format(name)
    tree.write(os.path.join(fullpath, urdf_filename), pretty_print=True)

    # Write Gazebo config file
    root = et.Element("model")
    model = et.SubElement(root, "name")
    model.text = name
    version = et.SubElement(root, "version")
    version.text = "1.0"
    sdf = et.SubElement(root, "sdf", version="1.4")
    sdf.text = "{}.urdf".format(name)

    author = et.SubElement(root, "author")
    et.SubElement(author, "name").text = "trimesh {}".format(__version__)
    et.SubElement(author, "email").text = "blank@blank.blank"

    description = et.SubElement(root, "description")
    description.text = name
    tree = et.ElementTree(root)

    if tol.strict:
        # todo : we don't pass the URDF schema validation
        schema = et.XMLSchema(file=get("schema/urdf.xsd", as_stream=True))
        if not schema.validate(tree):
            # actual error isn't raised by validate
            log.debug(schema.error_log)

    tree.write(os.path.join(fullpath, "model.config"))
    return np.sum(convex_pieces)
