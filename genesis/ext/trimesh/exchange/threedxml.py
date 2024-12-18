"""
threedxml.py
-------------

Load 3DXML files, a scene format from Dassault products like Solidworks, Abaqus, Catia
"""

import numpy as np

try:
    # `pip install pillow`
    # optional: used for textured meshes
    from PIL import Image
except BaseException as E:
    # if someone tries to use Image re-raise
    # the import error so they can debug easily
    from ..exceptions import ExceptionWrapper

    Image = ExceptionWrapper(E)

import json
import collections

from .. import util
from ..visual.texture import TextureVisuals


def load_3DXML(file_obj, *args, **kwargs):
    """
    Load a 3DXML scene into kwargs. 3DXML is a CAD format
    that can be exported from Solidworks

    Parameters
    ------------
    file_obj : file object
      Open and containing 3DXML data

    Returns
    -----------
    kwargs : dict
      Can be passed to trimesh.exchange.load.load_kwargs
    """
    archive = util.decompress(file_obj, file_type="zip")

    # a dictionary of file name : lxml etree
    as_etree = {}
    for k, v in archive.items():
        # wrap in try statement, as sometimes 3DXML
        # contains non- xml files, like JPG previews
        try:
            as_etree[k] = etree.XML(v.read())
        except etree.XMLSyntaxError:
            # move the file object back to the file start
            v.seek(0)

    # the file name of the root scene
    root_file = as_etree["Manifest.xml"].find("{*}Root").text
    # the etree of the scene layout
    tree = as_etree[root_file]
    # index of root element of directed acyclic graph
    root_id = tree.find("{*}ProductStructure").attrib["root"]

    # load the materials library from the materials elements
    colors = {}
    images = {}
    # but only if it exists
    material_key = "CATMaterialRef.3dxml"
    if material_key in as_etree:
        material_tree = as_etree[material_key]
        for MaterialDomain in material_tree.iter("{*}MaterialDomain"):
            material_id = MaterialDomain.attrib["id"]
            material_file = MaterialDomain.attrib["associatedFile"].split("urn:3DXML:")[-1]
            rend = as_etree[material_file].find("{*}Feature[@Alias='RenderingFeature']")
            diffuse = rend.find("{*}Attr[@Name='DiffuseColor']")
            # specular = rend.find("{*}Attr[@Name='SpecularColor']")
            # emissive = rend.find("{*}Attr[@Name='EmissiveColor']")
            if diffuse is not None:
                rgb = (np.array(json.loads(diffuse.attrib["Value"])) * 255).astype(np.uint8)
                colors[material_id] = rgb
            texture = rend.find("{*}Attr[@Name='TextureImage']")
            if texture is not None:
                tex_file, tex_id = texture.attrib["Value"].split(":")[-1].split("#")
                rep_image = as_etree[tex_file].find("{*}CATRepImage/{*}CATRepresentationImage[@id='%s']" % tex_id)
                if rep_image is not None:
                    image_file = rep_image.get("associatedFile", "").split(":")[-1]
                    images[material_id] = Image.open(archive[image_file])

        # copy indexes for instances of colors
        for MaterialDomainInstance in material_tree.iter("{*}MaterialDomainInstance"):
            instance = MaterialDomainInstance.find("{*}IsInstanceOf")
            # colors[b.attrib['id']] = colors[instance.text]
            for aggregate in MaterialDomainInstance.findall("{*}IsAggregatedBy"):
                colors[aggregate.text] = colors.get(instance.text)
                images[aggregate.text] = images.get(instance.text)

    # references which hold the 3DXML scene structure as a dict
    # element id : {key : value}
    references = collections.defaultdict(dict)

    def get_rgba(color):
        """
        Return (4,) uint8 color array defined by Color element attributes.

        Parameters
        -----------
        color : lxml.Element
          Element containing RGBA colors.

        Returns
        -----------
        as_int : (4,) np.uint8
          Colors as uint8 RGBA.
        """
        assert "RGBAColorType" in color.attrib.values()
        # colors will be float 0.0 - 1.0
        rgba = np.array([color.get(channel, 1.0) for channel in ("red", "green", "blue", "alpha")], dtype=np.float64)
        # convert to int colors
        return (rgba * 255).astype(np.uint8)

    # the 3DXML can specify different visual properties for occurrences
    view = tree.find("{*}DefaultView")
    if view is not None:
        for ViewProp in view.iter("{*}DefaultViewProperty"):
            color = ViewProp.find("{*}GraphicProperties/" + "{*}SurfaceAttributes/{*}Color")
            if color is None:
                continue
            rgba = get_rgba(color)
            for occurrence in ViewProp.findall("{*}OccurenceId/{*}id"):
                reference_id = occurrence.text.split("#")[-1]
                references[reference_id]["color"] = rgba

    # geometries will hold meshes
    geometries = dict()

    # get geometry
    for ReferenceRep in tree.iter(tag="{*}ReferenceRep"):
        # the str of an int that represents this meshes unique ID
        part_id = ReferenceRep.attrib["id"]
        # which part file in the archive contains the geometry we care about
        part_file = ReferenceRep.attrib["associatedFile"].split(":")[-1]
        # the format of the geometry file
        part_format = ReferenceRep.attrib["format"]
        if part_format not in ("TESSELLATED",):
            util.log.warning("ReferenceRep %r unsupported format %r" % (part_file, part_format))
            continue

        # load actual geometry
        mesh_faces = []
        mesh_colors = []
        mesh_normals = []
        mesh_vertices = []
        mesh_uv = []
        mesh_image = None

        if part_file not in as_etree and part_file in archive:
            # the data is stored in some binary format
            util.log.warning("unable to load Rep %r" % part_file)
            # data = archive[part_file]
            continue

        # the geometry is stored in a Rep
        for Rep in as_etree[part_file].iter("{*}Rep"):
            rep_faces = []  # faces sharing the same list of vertices
            vertices = Rep.find("{*}VertexBuffer/{*}Positions")
            if vertices is None:
                continue

            # they mix delimiters like we couldn't figure it out from the
            # shape :(
            # load vertices into (n, 3) float64
            mesh_vertices.append(
                np.fromstring(vertices.text.replace(",", " "), sep=" ", dtype=np.float64).reshape((-1, 3))
            )

            # load vertex normals into (n, 3) float64
            normals = Rep.find("{*}VertexBuffer/{*}Normals")
            mesh_normals.append(
                np.fromstring(normals.text.replace(",", " "), sep=" ", dtype=np.float64).reshape((-1, 3))
            )

            uv = Rep.find("{*}VertexBuffer/{*}TextureCoordinates")
            if uv is not None:  # texture coordinates are available
                rep_uv = np.fromstring(uv.text.replace(",", " "), sep=" ", dtype=np.float64)
                if "1D" == uv.get("dimension"):
                    mesh_uv.append(np.stack([rep_uv, np.zeros(len(rep_uv))], axis=1))
                else:  # 2D
                    mesh_uv.append(rep_uv.reshape(-1, 2))

            material = Rep.find("{*}SurfaceAttributes/" + "{*}MaterialApplication/" + "{*}MaterialId")
            if material is None:
                material_id = None
            else:
                (material_file, material_id) = material.attrib["id"].split("urn:3DXML:")[-1].split("#")
                mesh_image = images.get(material_id)  # texture for this Rep, if any

            for faces in Rep.iter("{*}Faces"):
                triangles = []  # mesh triangles for this Faces element
                for face in faces.iter("{*}Face"):
                    # Each Face may have optional strips, triangles or fans attributes
                    if "strips" in face.attrib:
                        # triangle strips, sequence of arbitrary length lists
                        # np.fromstring is substantially faster than np.array(i.split())
                        # inside the list comprehension
                        strips = [np.fromstring(i, sep=" ", dtype=np.int64) for i in face.attrib["strips"].split(",")]
                        # convert strips to (m, 3) int triangles
                        triangles.extend(util.triangle_strips_to_faces(strips))

                    if "triangles" in face.attrib:
                        triangles.extend(
                            np.fromstring(face.attrib["triangles"], sep=" ", dtype=np.int64).reshape((-1, 3))
                        )

                    if "fans" in face.attrib:
                        fans = [np.fromstring(i, sep=" ", dtype=np.int64) for i in face.attrib["fans"].split(",")]
                        # convert fans to (m, 3) int triangles
                        triangles.extend(util.triangle_fans_to_faces(fans))

                    rep_faces.extend(triangles)

                    # store the material information as (m, 3) uint8 FACE COLORS
                    faceColor = colors.get(material_id, [128, 128, 128])
                    # each Face may have its own color
                    colorElement = face.find("{*}SurfaceAttributes/{*}Color")
                    if colorElement is not None:
                        faceColor = get_rgba(colorElement)[:3]
                    mesh_colors.append(np.tile(faceColor, (len(triangles), 1)))
            mesh_faces.append(rep_faces)

        # save each mesh as the kwargs for a trimesh.Trimesh constructor
        # aka, a Trimesh object can be created with trimesh.Trimesh(**mesh)
        # this avoids needing trimesh- specific imports in this IO function
        mesh = dict()
        (mesh["vertices"], mesh["faces"]) = util.append_faces(mesh_vertices, mesh_faces)
        mesh["vertex_normals"] = np.vstack(mesh_normals)
        if mesh_uv and mesh_image:
            mesh["visual"] = TextureVisuals(uv=np.vstack(mesh_uv), image=mesh_image)
        else:
            mesh["face_colors"] = np.vstack(mesh_colors)

        # as far as I can tell, all 3DXML files are exported as
        # implicit millimeters (it isn't specified in the file)
        mesh["metadata"] = {"units": "mm"}
        mesh["class"] = "Trimesh"

        geometries[part_id] = mesh
        references[part_id]["geometry"] = part_id

    # a Reference3D maps to a subassembly or assembly
    for Reference3D in tree.iter("{*}Reference3D"):
        references[Reference3D.attrib["id"]] = {"name": Reference3D.attrib["name"], "type": "Reference3D"}

    # a node that is the connectivity between a geometry and the Reference3D
    for InstanceRep in tree.iter("{*}InstanceRep"):
        current = InstanceRep.attrib["id"]
        instance = InstanceRep.find("{*}IsInstanceOf").text
        aggregate = InstanceRep.find("{*}IsAggregatedBy").text

        references[current].update({"aggregate": aggregate, "instance": instance, "type": "InstanceRep"})

    # an Instance3D maps basically to a part
    for Instance3D in tree.iter("{*}Instance3D"):
        matrix = np.eye(4)
        relative = Instance3D.find("{*}RelativeMatrix")
        if relative is not None:
            relative = np.array(relative.text.split(), dtype=np.float64)

            # rotation component
            matrix[:3, :3] = relative[:9].reshape((3, 3)).T
            # translation component
            matrix[:3, 3] = relative[9:]

        current = Instance3D.attrib["id"]
        name = Instance3D.attrib["name"]
        instance = Instance3D.find("{*}IsInstanceOf").text
        aggregate = Instance3D.find("{*}IsAggregatedBy").text

        references[current].update(
            {"aggregate": aggregate, "instance": instance, "matrix": matrix, "name": name, "type": "Instance3D"}
        )

    # turn references into directed graph for path finding
    graph = nx.DiGraph()
    for k, v in references.items():
        # IsAggregatedBy points up to a parent
        if "aggregate" in v:
            graph.add_edge(v["aggregate"], k)
        # IsInstanceOf indicates a child
        if "instance" in v:
            graph.add_edge(k, v["instance"])

    # the 3DXML format is stored as a directed acyclic graph that needs all
    # paths from the root to a geometry to generate the tree of the scene
    paths = []
    for geometry_id in geometries.keys():
        paths.extend(nx.all_simple_paths(graph, source=root_id, target=geometry_id))

    # the name of the root frame
    root_name = references[root_id]["name"]
    # create a list of kwargs to send to the scene.graph.update function
    # start with a transform from the graphs base frame to our root name

    graph_kwargs = [{"frame_to": root_name, "matrix": np.eye(4)}]

    # we are going to collect prettier geometry names as we traverse paths
    geom_names = {}
    # loop through every simple path and generate transforms tree
    # note that we are flattening the transform tree here
    for path in paths:
        name = ""
        if "name" in references[path[-3]]:
            name = references[path[-3]]["name"]
            geom_names[path[-1]] = name
        # we need a unique node name for our geometry instance frame
        # due to the nature of the DAG names specified by the file may not
        # be unique, so we add an Instance3D name then append the path ids
        node_name = name + "#" + ":".join(path)

        # pull all transformations in the path
        matrices = [references[i]["matrix"] for i in path if "matrix" in references[i]]
        if len(matrices) == 0:
            matrix = np.eye(4)
        elif len(matrices) == 1:
            matrix = matrices[0]
        else:
            matrix = util.multi_dot(matrices)

        graph_kwargs.append({"matrix": matrix, "frame_from": root_name, "frame_to": node_name, "geometry": path[-1]})

    # remap geometry names from id numbers to the name string
    # we extracted from the 3DXML tree
    geom_final = {}
    for key, value in geometries.items():
        if key in geom_names:
            geom_final[geom_names[key]] = value
    # change geometry names in graph kwargs in place
    for kwarg in graph_kwargs:
        if "geometry" not in kwarg:
            continue
        kwarg["geometry"] = geom_names[kwarg["geometry"]]

    # create the kwargs for load_kwargs
    result = {"class": "Scene", "geometry": geom_final, "graph": graph_kwargs}

    return result


def print_element(element):
    """
    Pretty-print an lxml.etree element.

    Parameters
    ------------
    element : etree element
    """
    pretty = etree.tostring(element, pretty_print=True).decode("utf-8")
    return pretty


try:
    # soft dependencies
    from lxml import etree
    import networkx as nx

    _threedxml_loaders = {"3dxml": load_3DXML}
except BaseException as E:
    # set loader to exception wrapper
    from ..exceptions import ExceptionWrapper

    _threedxml_loaders = {"3dxml": ExceptionWrapper(E)}
