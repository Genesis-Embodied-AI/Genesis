"""
rendering.py
--------------

Functions to convert trimesh objects to pyglet/opengl objects.
"""

import numpy as np

from . import util

# avoid importing pyglet or pyglet.gl
# as pyglet does things on import
GL_POINTS, GL_LINES, GL_TRIANGLES = (0, 1, 4)


def convert_to_vertexlist(geometry, **kwargs):
    """
    Try to convert various geometry objects to the constructor
    args for a pyglet indexed vertex list.

    Parameters
    ------------
    obj : Trimesh, Path2D, Path3D, (n,2) float, (n,3) float
      Object to render

    Returns
    ------------
    args : tuple
      Args to be passed to pyglet indexed vertex list
      constructor.
    """
    if util.is_instance_named(geometry, "Trimesh"):
        return mesh_to_vertexlist(geometry, **kwargs)
    elif util.is_instance_named(geometry, "Path"):
        # works for Path3D and Path2D
        # both of which inherit from Path
        return path_to_vertexlist(geometry, **kwargs)
    elif util.is_instance_named(geometry, "PointCloud"):
        # pointcloud objects contain colors
        return points_to_vertexlist(geometry.vertices, colors=geometry.colors, **kwargs)
    elif util.is_instance_named(geometry, "ndarray"):
        # (n,2) or (n,3) points
        return points_to_vertexlist(geometry, **kwargs)
    elif util.is_instance_named(geometry, "VoxelGrid"):
        # for voxels view them as a bunch of boxes
        return mesh_to_vertexlist(geometry.as_boxes(**kwargs), **kwargs)
    else:
        raise ValueError("Geometry passed is not a viewable type!")


def mesh_to_vertexlist(mesh, group=None, smooth=True, smooth_threshold=60000):
    """
    Convert a Trimesh object to arguments for an
    indexed vertex list constructor.

    Parameters
    -------------
    mesh : trimesh.Trimesh
      Mesh to be rendered
    group : str
      Rendering group for the vertex list
    smooth : bool
      Should we try to smooth shade the mesh
    smooth_threshold : int
      Maximum number of faces to smooth shade

    Returns
    --------------
    args : (7,) tuple
      Args for vertex list constructor
    """

    if hasattr(mesh.visual, "uv"):
        # if the mesh has texture defined pass it to pyglet
        vertex_count = len(mesh.vertices)
        normals = mesh.vertex_normals.reshape(-1).tolist()
        faces = mesh.faces.reshape(-1).tolist()
        vertices = mesh.vertices.reshape(-1).tolist()

        # get the per-vertex UV coordinates
        uv = mesh.visual.uv

        # shortcut for the material
        material = mesh.visual.material
        if hasattr(material, "image"):
            # does the material actually have an image specified
            no_image = material.image is None
        elif hasattr(material, "baseColorTexture"):
            no_image = material.baseColorTexture is None
        else:
            no_image = True

        # didn't get valid texture so skip it
        if uv is None or no_image or len(uv) != vertex_count:
            # if no UV coordinates on material, just set face colors
            # to the diffuse color of the material
            color_gl = colors_to_gl(material.main_color, vertex_count)
        else:
            # if someone passed (n, 3) UVR cut it off here
            if uv.shape[1] > 2:
                uv = uv[:, :2]
            # texcoord as (2,) float
            color_gl = ("t2f/static", uv.astype(np.float64).reshape(-1).tolist())

    elif smooth and len(mesh.faces) < smooth_threshold:
        # if we have a small number of faces and colors defined
        # smooth the  mesh by merging vertices of faces below
        # the threshold angle
        mesh = mesh.smoothed()
        vertex_count = len(mesh.vertices)
        normals = mesh.vertex_normals.reshape(-1).tolist()
        faces = mesh.faces.reshape(-1).tolist()
        vertices = mesh.vertices.reshape(-1).tolist()
        color_gl = colors_to_gl(mesh.visual.vertex_colors, vertex_count)
    else:
        # we don't have textures or want to smooth so
        # send a polygon soup of disconnected triangles to opengl
        vertex_count = len(mesh.triangles) * 3
        normals = np.tile(mesh.face_normals, (1, 3)).reshape(-1).tolist()
        vertices = mesh.triangles.reshape(-1).tolist()
        faces = np.arange(vertex_count).tolist()
        colors = np.tile(mesh.visual.face_colors, (1, 3)).reshape((-1, 4))
        color_gl = colors_to_gl(colors, vertex_count)

    # create the ordered tuple for pyglet, use like:
    # `batch.add_indexed(*args)`
    args = (
        vertex_count,  # number of vertices
        GL_TRIANGLES,  # mode
        group,  # group
        faces,  # indices
        ("v3f/static", vertices),
        ("n3f/static", normals),
        color_gl,
    )
    return args


def path_to_vertexlist(path, group=None, **kwargs):
    """
    Convert a Path3D object to arguments for a
    pyglet indexed vertex list constructor.

    Parameters
    -------------
    path : trimesh.path.Path3D object
      Mesh to be rendered
    group : str
      Rendering group for the vertex list

    Returns
    --------------
    args : (7,) tuple
      Args for vertex list constructor
    """
    # avoid cache check inside tight loop
    vertices = path.vertices

    # get (n, 2, (2|3)) lines
    stacked = [util.stack_lines(e.discrete(vertices)) for e in path.entities]
    lines = util.vstack_empty(stacked)
    count = len(lines)

    # stack zeros for 2D lines
    if util.is_shape(vertices, (-1, 2)):
        lines = lines.reshape((-1, 2))
        lines = np.column_stack((lines, np.zeros(len(lines))))
    # index for GL is one per point
    index = np.arange(count).tolist()
    # convert from entity color to the color of
    # each vertex in the line segments
    colors = path.colors
    if colors is not None:
        colors = np.vstack([(np.ones((len(s), 4)) * c).astype(np.uint8) for s, c in zip(stacked, path.colors)])
    # convert to gl-friendly colors
    gl_colors = colors_to_gl(colors, count=count)

    # collect args for vertexlist constructor
    args = (
        count,  # number of lines
        GL_LINES,  # mode
        group,  # group
        index,  # indices
        ("v3f/static", lines.reshape(-1)),
        gl_colors,
    )
    return args


def points_to_vertexlist(points, colors=None, group=None, **kwargs):
    """
    Convert a numpy array of 3D points to args for
    a vertex list constructor.

    Parameters
    -------------
    points : (n, 3) float
      Points to be rendered
    colors : (n, 3) or (n, 4) float
      Colors for each point
    group : str
      Rendering group for the vertex list

    Returns
    --------------
    args : (7,) tuple
      Args for vertex list constructor
    """
    points = np.asanyarray(points, dtype=np.float64)

    if util.is_shape(points, (-1, 2)):
        points = np.column_stack((points, np.zeros(len(points))))
    elif not util.is_shape(points, (-1, 3)):
        raise ValueError("Pointcloud must be (n,3)!")

    index = np.arange(len(points)).tolist()

    args = (
        len(points),  # number of vertices
        GL_POINTS,  # mode
        group,  # group
        index,  # indices
        ("v3f/static", points.reshape(-1)),
        colors_to_gl(colors, len(points)),
    )
    return args


def colors_to_gl(colors, count):
    """
    Given a list of colors (or None) return a GL-acceptable
    list of colors.

    Parameters
    ------------
    colors: (count, (3 or 4)) float
      Input colors as an array

    Returns
    ---------
    colors_type : str
      Color type
    colors_gl : (count,) list
      Colors to pass to pyglet
    """

    colors = np.asanyarray(colors)
    count = int(count)
    # get the GL kind of color we have
    colors_dtypes = {"f": "f", "i": "B", "u": "B"}

    if colors.dtype.kind in colors_dtypes:
        dtype = colors_dtypes[colors.dtype.kind]
    else:
        dtype = None

    if dtype is not None and util.is_shape(colors, (count, (3, 4))):
        # save the shape and dtype for opengl color string
        colors_type = "c{}{}/static".format(colors.shape[1], dtype)
        # reshape the 2D array into a 1D one and then convert to a python list
        gl_colors = colors.reshape(-1).tolist()
    elif dtype is not None and colors.shape in [(3,), (4,)]:
        # we've been passed a single color so tile them
        gl_colors = (np.ones((count, colors.size), dtype=colors.dtype) * colors).reshape(-1).tolist()
        # we know we're tiling
        colors_type = "c{}{}/static".format(colors.size, dtype)
    else:
        # case where colors are wrong shape
        # use black as the default color
        gl_colors = np.tile([0.0, 0.0, 0.0], (count, 1)).reshape(-1).tolist()
        # we're returning RGB float colors
        colors_type = "c3f/static"

    return colors_type, gl_colors


def material_to_texture(material, upsize=True):
    """
    Convert a trimesh.visual.texture.Material object into
    a pyglet-compatible texture object.

    Parameters
    --------------
    material : trimesh.visual.texture.Material
      Material to be converted
    upsize: bool
      If True, will upscale textures to their nearest power
      of two resolution to avoid weirdness

    Returns
    ---------------
    texture : pyglet.image.Texture
      Texture loaded into pyglet form
    """
    import pyglet

    # try to extract a PIL image from material
    if hasattr(material, "image"):
        img = material.image
    elif hasattr(material, "baseColorTexture"):
        img = material.baseColorTexture
    else:
        return None

    # if no images in texture return now
    if img is None:
        return None

    # if we're not powers of two upsize
    if upsize:
        from .visual.texture import power_resize

        img = power_resize(img)

    # use a PNG export to exchange into pyglet
    # probably a way to do this with a PIL converter
    with util.BytesIO() as f:
        # export PIL image as PNG
        img.save(f, format="png")
        f.seek(0)
        # filename used for format guess
        gl_image = pyglet.image.load(filename=".png", file=f)

    # turn image into pyglet texture
    texture = gl_image.get_texture()

    return texture


def matrix_to_gl(matrix):
    """
    Convert a numpy row-major homogeneous transformation matrix
    to a flat column-major GLfloat transformation.

    Parameters
    -------------
    matrix : (4,4) float
      Row-major homogeneous transform

    Returns
    -------------
    glmatrix : (16,) gl.GLfloat
      Transform in pyglet format
    """
    from pyglet import gl

    # convert to GLfloat, switch to column major and flatten to (16,)
    return (gl.GLfloat * 16)(*np.array(matrix, dtype=np.float32).T.ravel())


def vector_to_gl(array, *args):
    """
    Convert an array and an optional set of args into a
    flat vector of gl.GLfloat
    """
    from pyglet import gl

    array = np.array(array)
    if len(args) > 0:
        array = np.append(array, args)
    vector = (gl.GLfloat * len(array))(*array)
    return vector


def light_to_gl(light, transform, lightN):
    """
    Convert trimesh.scene.lighting.Light objects into
    args for gl.glLightFv calls

    Parameters
    --------------
    light : trimesh.scene.lighting.Light
      Light object to be converted to GL
    transform : (4, 4) float
      Transformation matrix of light
    lightN : int
      Result of gl.GL_LIGHT0, gl.GL_LIGHT1, etc

    Returns
    --------------
    multiarg : [tuple]
      List of args to pass to gl.glLightFv eg:
      [gl.glLightfb(*a) for a in multiarg]
    """
    from pyglet import gl

    # convert color to opengl
    gl_color = vector_to_gl(light.color.astype(np.float64) / 255.0)
    assert len(gl_color) == 4

    # cartesian translation from matrix
    gl_position = vector_to_gl(transform[:3, 3])

    # create the different position and color arguments
    args = [
        (lightN, gl.GL_POSITION, gl_position),
        (lightN, gl.GL_SPECULAR, gl_color),
        (lightN, gl.GL_DIFFUSE, gl_color),
        (lightN, gl.GL_AMBIENT, gl_color),
    ]
    return args
