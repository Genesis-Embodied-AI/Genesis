import collections
import io
import sys
import uuid
import zipfile

import numpy as np

from .. import graph, util
from ..constants import log


def load_3MF(file_obj, postprocess=True, **kwargs):
    """
    Load a 3MF formatted file into a Trimesh scene.

    Parameters
    ------------
    file_obj : file-like
      Contains 3MF formatted data

    Returns
    ------------
    kwargs : dict
      Constructor arguments for `trimesh.Scene`
    """
    # dict, {name in archive: BytesIo}
    archive = util.decompress(file_obj, file_type="zip")
    # get model with case-insensitive keys
    model = next(iter(v for k, v in archive.items() if "3d/3dmodel.model" in k.lower()))

    # read root attributes only from XML first
    event, root = next(etree.iterparse(model, tag=("{*}model"), events=("start",)))
    # collect unit information from the tree
    if "unit" in root.attrib:
        metadata = {"units": root.attrib["unit"]}
    else:
        # the default units, defined by the specification
        metadata = {"units": "millimeters"}

    # { mesh id : mesh name}
    id_name = {}
    # { mesh id: (n,3) float vertices}
    v_seq = {}
    # { mesh id: (n,3) int faces}
    f_seq = {}
    # components are objects that contain other objects
    # {id : [other ids]}
    components = collections.defaultdict(list)
    # load information about the scene graph
    # each instance is a single geometry
    build_items = []

    consumed_names = set()
    # iterate the XML object and build elements with an LXML iterator
    # loaded elements are cleared to avoid ballooning memory
    model.seek(0)
    for _, obj in etree.iterparse(model, tag=("{*}object", "{*}build")):
        # parse objects
        if "object" in obj.tag:
            # id is mandatory
            index = obj.attrib["id"]

            # start with stored name
            name = obj.attrib.get("name", str(index))
            # apparently some exporters name multiple meshes
            # the same thing so check to see if it's been used
            if name in consumed_names:
                name = name + str(index)
            consumed_names.add(name)
            # store name reference on the index
            id_name[index] = name

            # if the object has actual geometry data parse here
            for mesh in obj.iter("{*}mesh"):
                vertices = mesh.find("{*}vertices")
                v_seq[index] = np.array(
                    [[i.attrib["x"], i.attrib["y"], i.attrib["z"]] for i in vertices.iter("{*}vertex")],
                    dtype=np.float64,
                )
                vertices.clear()
                vertices.getparent().remove(vertices)

                faces = mesh.find("{*}triangles")
                f_seq[index] = np.array(
                    [[i.attrib["v1"], i.attrib["v2"], i.attrib["v3"]] for i in faces.iter("{*}triangle")],
                    dtype=np.int64,
                )
                faces.clear()
                faces.getparent().remove(faces)

            # components are references to other geometries
            for c in obj.iter("{*}component"):
                mesh_index = c.attrib["objectid"]
                transform = _attrib_to_transform(c.attrib)
                components[index].append((mesh_index, transform))

        # parse build
        if "build" in obj.tag:
            # scene graph information stored here, aka "build" the scene
            for item in obj.iter("{*}item"):
                # get a transform from the item's attributes
                transform = _attrib_to_transform(item.attrib)
                # the index of the geometry this item instantiates
                build_items.append((item.attrib["objectid"], transform))

        # free resources
        obj.clear()
        obj.getparent().remove(obj)
        del obj

    # have one mesh per 3MF object
    # one mesh per geometry ID, store as kwargs for the object
    meshes = {}
    for gid in v_seq.keys():
        name = id_name[gid]
        meshes[name] = {"vertices": v_seq[gid], "faces": f_seq[gid], "metadata": metadata.copy()}
        meshes[name].update(kwargs)

    # turn the item / component representation into
    # a MultiDiGraph to compound our pain
    g = nx.MultiDiGraph()
    # build items are the only things that exist according to 3MF
    # so we accomplish that by linking them to the base frame
    for gid, tf in build_items:
        g.add_edge("world", gid, matrix=tf)
    # components are instances which need to be linked to base
    # frame by a build_item
    for start, group in components.items():
        for gid, tf in group:
            g.add_edge(start, gid, matrix=tf)

    # turn the graph into kwargs for a scene graph
    # flatten the scene structure and simplify to
    # a single unique node per instance
    graph_args = []
    parents = collections.defaultdict(set)
    for path in graph.multigraph_paths(G=g, source="world"):
        # collect all the transform on the path
        transforms = graph.multigraph_collect(G=g, traversal=path, attrib="matrix")
        # combine them into a single transform
        if len(transforms) == 1:
            transform = transforms[0]
        else:
            transform = util.multi_dot(transforms)

        # the last element of the path should be the geometry
        last = path[-1][0]
        # if someone included an undefined component, skip it
        if last not in id_name:
            log.debug("id {} included but not defined!".format(last))
            continue
        # frame names unique
        name = id_name[last] + util.unique_id()
        # index in meshes
        geom = id_name[last]

        # collect parents if we want to combine later
        if len(path) > 2:
            parent = path[-2][0]
            parents[parent].add(last)

        graph_args.append({"frame_from": "world", "frame_to": name, "matrix": transform, "geometry": geom})

    # solidworks will export each body as its own mesh with the part
    # name as the parent so optionally rename and combine these bodies
    if postprocess and all("body" in i.lower() for i in meshes.keys()):
        # don't rename by default
        rename = {k: k for k in meshes.keys()}
        for parent, mesh_name in parents.items():
            # only handle the case where a parent has a single child
            # if there are multiple children we would do a combine op
            if len(mesh_name) != 1:
                continue
            # rename the part
            rename[id_name[next(iter(mesh_name))]] = id_name[parent].split("(")[0]

        # apply the rename operation meshes
        meshes = {rename[k]: m for k, m in meshes.items()}
        # rename geometry references in the scene graph
        for arg in graph_args:
            if "geometry" in arg:
                arg["geometry"] = rename[arg["geometry"]]

    # construct the kwargs to load the scene
    kwargs = {"base_frame": "world", "graph": graph_args, "geometry": meshes, "metadata": metadata}

    return kwargs


def export_3MF(mesh, batch_size=4096, compression=zipfile.ZIP_DEFLATED, compresslevel=5):
    """
    Converts a Trimesh object into a 3MF file.

    Parameters
    ---------
    mesh trimesh.trimesh
      Mesh or Scene to export.
    batch_size : int
      Number of nodes to write per batch.
    compression : zipfile.ZIP_*
      Type of zip compression to use in this export.
    compresslevel : int
      For Python > 3.7 specify the 0-9 compression level.

    Returns
    ---------
    export : bytes
      Represents geometry as a 3MF file.
    """

    if sys.version_info < (3, 6):
        # Python only added 'w' mode to `zipfile` in Python 3.6
        # and it is not worth the effort to work around
        raise NotImplementedError("3MF export requires Python >= 3.6")
    from ..scene.scene import Scene

    if not isinstance(mesh, Scene):
        mesh = Scene(mesh)

    geometry = mesh.geometry
    graph = mesh.graph.to_networkx()
    base_frame = mesh.graph.base_frame

    # xml namespaces
    model_nsmap = {
        None: "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
        "m": "http://schemas.microsoft.com/3dmanufacturing/material/2015/02",
        "p": "http://schemas.microsoft.com/3dmanufacturing/production/2015/06",
        "b": "http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02",
        "s": "http://schemas.microsoft.com/3dmanufacturing/slice/2015/07",
        "sc": "http://schemas.microsoft.com/3dmanufacturing/securecontent/2019/04",
    }

    rels_nsmap = {None: "http://schemas.openxmlformats.org/package/2006/relationships"}

    # model ids
    models = []

    def model_id(x):
        if x not in models:
            models.append(x)
        return str(models.index(x) + 1)

    # 3mf archive dict {path: BytesIO}
    file_obj = io.BytesIO()

    # specify the parameters for the zip container
    zip_kwargs = {"compression": compression}
    # compresslevel was added in Python 3.7
    if sys.version_info >= (3, 7):
        zip_kwargs["compresslevel"] = compresslevel

    with zipfile.ZipFile(file_obj, mode="w", **zip_kwargs) as z:
        # 3dmodel.model
        with z.open("3D/3dmodel.model", mode="w") as f, etree.xmlfile(f, encoding="utf-8") as xf:
            xf.write_declaration()

            # stream elements
            with xf.element("model", {"unit": "millimeter"}, nsmap=model_nsmap):
                # objects with mesh data and/or references to other objects
                with xf.element("resources"):
                    # stream objects with actual mesh data
                    for i, (name, m) in enumerate(geometry.items()):
                        # attributes for object
                        attribs = {"id": model_id(name), "name": name, "type": "model", "p:UUID": ID()}
                        with xf.element("object", **attribs):
                            with xf.element("mesh"):
                                with xf.element("vertices"):
                                    # vertex nodes are writed directly to the file
                                    # so make sure lxml's buffer is flushed
                                    xf.flush()
                                    for i in range(0, len(m.vertices), batch_size):
                                        batch = m.vertices[i : i + batch_size]
                                        fragment = '<vertex x="{}" y="{}" z="{}" />' * len(batch)
                                        f.write(fragment.format(*batch.flatten()).encode("utf-8"))
                                with xf.element("triangles"):
                                    xf.flush()
                                    for i in range(0, len(m.faces), batch_size):
                                        batch = m.faces[i : i + batch_size]
                                        fragment = '<triangle v1="{}" v2="{}" v3="{}" />' * len(batch)
                                        f.write(fragment.format(*batch.flatten()).encode("utf-8"))

                    # stream components
                    for node in graph.nodes:
                        if node == base_frame or node.startswith("camera"):
                            continue
                        if len(graph[node]) == 0:
                            continue

                        attribs = {"id": model_id(node), "name": node, "type": "model", "p:UUID": ID()}
                        with xf.element("object", **attribs):
                            with xf.element("components"):
                                for next, data in graph[node].items():
                                    transform = " ".join(str(i) for i in np.array(data["matrix"])[:3, :4].T.flatten())
                                    xf.write(
                                        etree.Element(
                                            "component",
                                            {
                                                "objectid": (
                                                    model_id(data["geometry"]) if "geometry" in data else model_id(next)
                                                ),
                                                "transform": transform,
                                            },
                                        )
                                    )

                # stream build (objects on base_frame)
                with xf.element("build", {"p:UUID": ID()}):
                    for node, data in graph[base_frame].items():
                        if node.startswith("camera"):
                            continue
                        transform = " ".join(str(i) for i in np.array(data["matrix"])[:3, :4].T.flatten())
                        uuid_tag = "{{{}}}UUID".format(model_nsmap["p"])
                        xf.write(
                            etree.Element(
                                "item",
                                {"objectid": model_id(node), "transform": transform, uuid_tag: ID()},
                                nsmap=model_nsmap,
                            )
                        )

        # .rels
        with z.open("_rels/.rels", "w") as f, etree.xmlfile(f, encoding="utf-8") as xf:
            xf.write_declaration()
            # stream elements
            with xf.element("Relationships", nsmap=rels_nsmap):
                rt = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
                xf.write(
                    etree.Element(
                        "Relationship",
                        Type=rt,
                        Target="/3D/3dmodel.model",
                        Id="rel0",
                    )
                )

        # [Content_Types].xml
        with z.open("[Content_Types].xml", "w") as f, etree.xmlfile(f, encoding="utf-8") as xf:
            xf.write_declaration()
            # xml namespaces
            nsmap = {None: "http://schemas.openxmlformats.org/package/2006/content-types"}

            # stream elements
            types = [
                ("jpeg", "image/jpeg"),
                ("jpg", "image/jpeg"),
                ("model", "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"),
                ("png", "image/png"),
                ("rels", "application/vnd.openxmlformats-package.relationships+xml"),
                (
                    "texture",
                    "application/vnd.ms-package.3dmanufacturing-3dmodeltexture",
                ),
            ]
            with xf.element("Types", nsmap=nsmap):
                for ext, ctype in types:
                    xf.write(etree.Element("Default", Extension=ext, ContentType=ctype))

    return file_obj.getvalue()


def _attrib_to_transform(attrib):
    """
    Extract a homogeneous transform from a dictionary.

    Parameters
    ------------
    attrib: dict, optionally containing 'transform'

    Returns
    ------------
    transform: (4, 4) float, homogeonous transformation
    """

    transform = np.eye(4, dtype=np.float64)
    if "transform" in attrib:
        # wangle their transform format
        values = np.array(attrib["transform"].split(), dtype=np.float64).reshape((4, 3)).T
        transform[:3, :4] = values
    return transform


# do import here to keep lxml a soft dependency
try:
    import networkx as nx
    from lxml import etree

    _three_loaders = {"3mf": load_3MF}
    if sys.version_info < (3, 6):
        # Python only added 'w' mode to `zipfile` in Python 3.6
        # and it is not worth the effort to work around
        from ..exceptions import ExceptionWrapper

        _3mf_exporters = {"3mf": ExceptionWrapper(NotImplementedError("3MF export requires Python >= 3.6"))}
    else:
        _3mf_exporters = {"3mf": export_3MF}
except BaseException as E:
    from ..exceptions import ExceptionWrapper

    _three_loaders = {"3mf": ExceptionWrapper(E)}
    _3mf_exporters = {"3mf": ExceptionWrapper(E)}
