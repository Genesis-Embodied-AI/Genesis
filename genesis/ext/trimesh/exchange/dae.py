import copy
import io

import numpy as np

from .. import util, visual
from ..constants import log
from ..util import unique_name

_EYE = np.eye(4)
_EYE.flags.writeable = False


def load_collada(file_obj, resolver=None, ignore_broken=True, **kwargs):
    """
    Load a COLLADA (.dae) file into a list of trimesh kwargs.

    Parameters
    ----------
    file_obj : file object
      Containing a COLLADA file
    resolver : trimesh.visual.Resolver or None
      For loading referenced files, like texture images
    ignore_broken: bool
      Ignores broken references during loading:
        [collada.common.DaeUnsupportedError,
         collada.common.DaeBrokenRefError]
    kwargs : **
      Passed to trimesh.Trimesh.__init__

    Returns
    -------
    loaded : list of dict
      kwargs for Trimesh constructor
    """
    import collada

    if ignore_broken:
        ignores = [
            collada.common.DaeError,
            collada.common.DaeIncompleteError,
            collada.common.DaeMalformedError,
            collada.common.DaeBrokenRefError,
            collada.common.DaeUnsupportedError,
            collada.common.DaeIncompleteError,
        ]
    else:
        ignores = None

    # load scene using pycollada
    c = collada.Collada(file_obj, ignore=ignores)

    # Create material map from Material ID to trimesh material
    material_map = {}
    for m in c.materials:
        effect = m.effect
        material_map[m.id] = _parse_material(effect, resolver)

    # name : kwargs
    meshes = {}
    # increments to enable `unique_name` to avoid n^2 behavior
    meshes_count = {}
    # list of dict
    graph = []
    for node in c.scene.nodes:
        _parse_node(
            node=node,
            parent_matrix=_EYE,
            material_map=material_map,
            meshes=meshes,
            meshes_count=meshes_count,
            graph=graph,
            resolver=resolver,
        )

    # create kwargs for load_kwargs
    result = {"class": "Scene", "graph": graph, "geometry": meshes}

    return result


def export_collada(mesh, **kwargs):
    """
    Export a mesh or a list of meshes as a COLLADA .dae file.

    Parameters
    -----------
    mesh: Trimesh object or list of Trimesh objects
        The mesh(es) to export.

    Returns
    -----------
    export: str, string of COLLADA format output
    """
    import collada

    meshes = mesh
    if not isinstance(mesh, (list, tuple, set, np.ndarray)):
        meshes = [mesh]

    c = collada.Collada()
    nodes = []
    for i, m in enumerate(meshes):

        # Load uv, colors, materials
        uv = None
        colors = None
        mat = _unparse_material(None)
        if m.visual.defined:
            if m.visual.kind == "texture":
                mat = _unparse_material(m.visual.material)
                uv = m.visual.uv
            elif m.visual.kind == "vertex":
                colors = (m.visual.vertex_colors / 255.0)[:, :3]
        c.effects.append(mat.effect)
        c.materials.append(mat)

        # Create geometry object
        vertices = collada.source.FloatSource("verts-array", m.vertices.flatten(), ("X", "Y", "Z"))
        normals = collada.source.FloatSource("normals-array", m.vertex_normals.flatten(), ("X", "Y", "Z"))
        input_list = collada.source.InputList()
        input_list.addInput(0, "VERTEX", "#verts-array")
        input_list.addInput(1, "NORMAL", "#normals-array")
        arrays = [vertices, normals]
        if (uv is not None) and (len(uv) > 0):
            texcoords = collada.source.FloatSource("texcoords-array", uv.flatten(), ("U", "V"))
            input_list.addInput(2, "TEXCOORD", "#texcoords-array")
            arrays.append(texcoords)
        if colors is not None:
            idx = 2
            if uv:
                idx = 3
            colors = collada.source.FloatSource("colors-array", colors.flatten(), ("R", "G", "B"))
            input_list.addInput(idx, "COLOR", "#colors-array")
            arrays.append(colors)
        geom = collada.geometry.Geometry(c, uuid.uuid4().hex, uuid.uuid4().hex, arrays)
        indices = np.repeat(m.faces.flatten(), len(arrays))

        matref = "material{}".format(i)
        triset = geom.createTriangleSet(indices, input_list, matref)
        geom.primitives.append(triset)
        c.geometries.append(geom)

        matnode = collada.scene.MaterialNode(matref, mat, inputs=[])
        geomnode = collada.scene.GeometryNode(geom, [matnode])
        node = collada.scene.Node("node{}".format(i), children=[geomnode])
        nodes.append(node)
    scene = collada.scene.Scene("scene", nodes)
    c.scenes.append(scene)
    c.scene = scene

    b = io.BytesIO()
    c.write(b)
    b.seek(0)
    return b.read()


def _parse_node(node, parent_matrix, material_map, meshes, meshes_count, graph, resolver=None):
    """
    Recursively parse COLLADA scene nodes.
    """
    import collada

    # Parse mesh node
    if isinstance(node, collada.scene.GeometryNode):
        geometry = node.geometry

        # Create local material map from material symbol to actual material
        local_material_map = {}
        for mn in node.materials:
            symbol = mn.symbol
            m = mn.target
            if m.id in material_map:
                local_material_map[symbol] = material_map[m.id]
            else:
                local_material_map[symbol] = _parse_material(m, resolver)

        # Iterate over primitives of geometry
        for primitive in geometry.primitives:
            if isinstance(primitive, collada.polylist.Polylist):
                primitive = primitive.triangleset()
            if isinstance(primitive, collada.triangleset.TriangleSet):
                vertex = primitive.vertex
                vertex_index = primitive.vertex_index
                vertices = vertex[vertex_index].reshape(len(vertex_index) * 3, 3)

                # Get normals if present
                normals = None
                if primitive.normal is not None:
                    normal = primitive.normal
                    normal_index = primitive.normal_index
                    normals = normal[normal_index].reshape(len(normal_index) * 3, 3)

                # Get colors if present
                colors = None
                s = primitive.sources
                if "COLOR" in s and len(s["COLOR"]) > 0 and len(primitive.index) > 0:
                    color = s["COLOR"][0][4].data
                    color_index = primitive.index[:, :, s["COLOR"][0][0]]
                    colors = color[color_index].reshape(len(color_index) * 3, -1)

                faces = np.arange(vertices.shape[0]).reshape(vertices.shape[0] // 3, 3)

                # Get UV coordinates if possible
                vis = None
                if primitive.material in local_material_map:
                    material = copy.copy(local_material_map[primitive.material])
                    uv = None
                    if len(primitive.texcoordset) > 0:
                        texcoord = primitive.texcoordset[0]
                        texcoord_index = primitive.texcoord_indexset[0]
                        uv = texcoord[texcoord_index].reshape((len(texcoord_index) * 3, 2))
                    vis = visual.texture.TextureVisuals(uv=uv, material=material)

                geom_name = unique_name(geometry.id, contains=meshes, counts=meshes_count)
                meshes[geom_name] = {
                    "vertices": vertices,
                    "faces": faces,
                    "vertex_normals": normals,
                    "vertex_colors": colors,
                    "visual": vis,
                }

                graph.append({"frame_to": geom_name, "matrix": parent_matrix, "geometry": geom_name})

    # recurse down tree for nodes with children
    elif isinstance(node, collada.scene.Node):
        if node.children is not None:
            for child in node.children:
                # create the new matrix
                matrix = np.dot(parent_matrix, node.matrix)
                # parse the child node
                _parse_node(
                    node=child,
                    parent_matrix=matrix,
                    material_map=material_map,
                    meshes=meshes,
                    meshes_count=meshes_count,
                    graph=graph,
                    resolver=resolver,
                )

    elif isinstance(node, collada.scene.CameraNode):
        # TODO: convert collada cameras to trimesh cameras
        pass
    elif isinstance(node, collada.scene.LightNode):
        # TODO: convert collada lights to trimesh lights
        pass


def _load_texture(file_name, resolver):
    """
    Load a texture from a file into a PIL image.
    """
    from PIL import Image

    file_data = resolver.get(file_name)
    image = Image.open(util.wrap_as_stream(file_data))
    return image


def _parse_material(effect, resolver):
    """
    Turn a COLLADA effect into a trimesh material.
    """
    import collada

    # Compute base color
    baseColorFactor = np.ones(4)
    baseColorTexture = None
    if isinstance(effect.diffuse, collada.material.Map):
        try:
            baseColorTexture = _load_texture(effect.diffuse.sampler.surface.image.path, resolver)
        except BaseException:
            log.debug("unable to load base texture", exc_info=True)
    elif effect.diffuse is not None:
        baseColorFactor = effect.diffuse

    # Compute emission color
    emissiveFactor = np.zeros(3)
    emissiveTexture = None
    if isinstance(effect.emission, collada.material.Map):
        try:
            emissiveTexture = _load_texture(effect.diffuse.sampler.surface.image.path, resolver)
        except BaseException:
            log.warning("unable to load emissive texture", exc_info=True)
    elif effect.emission is not None:
        emissiveFactor = effect.emission[:3]

    # Compute roughness
    roughnessFactor = 1.0
    if not isinstance(effect.shininess, collada.material.Map) and effect.shininess is not None:
        roughnessFactor = np.sqrt(2.0 / (2.0 + effect.shininess))

    # Compute metallic factor
    metallicFactor = 0.0

    # Compute normal texture
    normalTexture = None
    if effect.bumpmap is not None:
        try:
            normalTexture = _load_texture(effect.bumpmap.sampler.surface.image.path, resolver)
        except BaseException:
            log.warning("unable to load bumpmap", exc_info=True)

    # Compute opacity
    if effect.transparent is not None and not isinstance(effect.transparent, collada.material.Map):
        baseColorFactor = tuple(np.append(baseColorFactor[:3], float(effect.transparent[3])))

    return visual.material.PBRMaterial(
        emissiveFactor=emissiveFactor,
        emissiveTexture=emissiveTexture,
        normalTexture=normalTexture,
        baseColorTexture=baseColorTexture,
        baseColorFactor=baseColorFactor,
        metallicFactor=metallicFactor,
        roughnessFactor=roughnessFactor,
    )


def _unparse_material(material):
    """
    Turn a trimesh material into a COLLADA material.
    """
    import collada

    # TODO EXPORT TEXTURES
    if isinstance(material, visual.material.PBRMaterial):
        diffuse = material.baseColorFactor
        diffuse = diffuse / 255.0
        if diffuse is not None:
            diffuse = list(diffuse)

        emission = material.emissiveFactor
        if emission is not None:
            emission = [float(emission[0]), float(emission[1]), float(emission[2]), 1.0]

        shininess = material.roughnessFactor
        if shininess is not None:
            shininess = 2.0 / shininess**2 - 2.0

        effect = collada.material.Effect(
            uuid.uuid4().hex,
            params=[],
            shadingtype="phong",
            diffuse=diffuse,
            emission=emission,
            specular=[1.0, 1.0, 1.0, 1.0],
            shininess=float(shininess),
        )
        material = collada.material.Material(uuid.uuid4().hex, "pbrmaterial", effect)
    else:
        effect = collada.material.Effect(uuid.uuid4().hex, params=[], shadingtype="phong")
        material = collada.material.Material(uuid.uuid4().hex, "defaultmaterial", effect)
    return material


def load_zae(file_obj, resolver=None, **kwargs):
    """
    Load a ZAE file, which is just a zipped DAE file.

    Parameters
    -------------
    file_obj : file object
      Contains ZAE data
    resolver : trimesh.visual.Resolver
      Resolver to load additional assets
    kwargs : dict
      Passed to load_collada

    Returns
    ------------
    loaded : dict
      Results of loading
    """

    # a dict, {file name : file object}
    archive = util.decompress(file_obj, file_type="zip")

    # load the first file with a .dae extension
    file_name = next(i for i in archive.keys() if i.lower().endswith(".dae"))

    # a resolver so the loader can load textures / etc
    resolver = visual.resolvers.ZipResolver(archive)

    # run the regular collada loader
    loaded = load_collada(archive[file_name], resolver=resolver, **kwargs)
    return loaded


# only provide loaders if `pycollada` is installed
_collada_loaders = {}
_collada_exporters = {}
if util.has_module("collada"):

    _collada_loaders["dae"] = load_collada
    _collada_loaders["zae"] = load_zae
    _collada_exporters["dae"] = export_collada
else:
    # store an exception to raise later
    from ..exceptions import ExceptionWrapper

    _exc = ExceptionWrapper(ImportError("missing `pip install pycollada`"))
    _collada_loaders.update({"dae": _exc, "zae": _exc})
    _collada_exporters["dae"] = _exc
