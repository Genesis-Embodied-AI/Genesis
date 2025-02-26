"""
windowed.py
---------------

Provides a pyglet- based windowed viewer to preview
Trimesh, Scene, PointCloud, and Path objects.

Works on all major platforms: Windows, Linux, and OSX.
"""

import collections
import numpy as np

import pyglet

# pyglet 2.0 is close to a re-write moving from fixed-function
# to shaders and we will likely support it by forking an entirely
# new viewer `trimesh.viewer.shaders` and then basically keeping
# `windowed` around for backwards-compatibility with no changes
if int(pyglet.version.split(".")[0]) >= 2:
    raise ImportError('`trimesh.viewer.windowed` requires `pip install "pyglet<2"`')

from .trackball import Trackball

from .. import util
from .. import rendering

from ..visual import to_rgba
from ..transformations import translation_matrix

pyglet.options["shadow_window"] = False

import pyglet.gl as gl  # NOQA

# smooth only when fewer faces than this
_SMOOTH_MAX_FACES = 100000


class SceneViewer(pyglet.window.Window):

    def __init__(
        self,
        scene,
        smooth=True,
        flags=None,
        visible=True,
        resolution=None,
        start_loop=True,
        callback=None,
        callback_period=None,
        caption=None,
        fixed=None,
        offset_lines=True,
        line_settings=None,
        background=None,
        window_conf=None,
        profile=False,
        record=False,
        **kwargs,
    ):
        """
        Create a window that will display a trimesh.Scene object
        in an OpenGL context via pyglet.

        Parameters
        ---------------
        scene : trimesh.scene.Scene
          Scene with geometry and transforms
        smooth : bool
          If True try to smooth shade things
        flags : dict
          If passed apply keys to self.view:
          ['cull', 'wireframe', etc]
        visible : bool
          Display window or not
        resolution : (2,) int
          Initial resolution of window
        start_loop : bool
          Call pyglet.app.run() at the end of init
        callback : function
          A function which can be called periodically to
          update things in the scene
        callback_period : float
          How often to call the callback, in seconds
        fixed : None or iterable
          List of keys in scene.geometry to skip view
          transform on to keep fixed relative to camera
        offset_lines : bool
          If True, will offset lines slightly so if drawn
          coplanar with mesh geometry they will be visible
        background : None or (4,) uint8
          Color for background
        window_conf : None, or gl.Config
          Passed to window init
        profile : bool
          If set will run a `pyinstrument` profile for
          every call to `on_draw` and print the output.
        record : bool
          If True, will save a list of `png` bytes to
          a list located in `scene.metadata['recording']`
        kwargs : dict
          Additional arguments to pass, including
          'background' for to set background color
        """
        self.scene = self._scene = scene

        self.callback = callback
        self.callback_period = callback_period
        self.scene._redraw = self._redraw
        self.offset_lines = bool(offset_lines)
        self.background = background
        # save initial camera transform
        self._initial_camera_transform = scene.camera_transform.copy()

        # a transform to offset lines slightly to avoid Z-fighting
        self._line_offset = translation_matrix([0, 0, scene.scale / 1000 if self.offset_lines else 0])

        self.reset_view()
        self.batch = pyglet.graphics.Batch()
        self._smooth = smooth

        self._profile = bool(profile)
        if self._profile:
            from pyinstrument import Profiler

            self.Profiler = Profiler

        self._record = bool(record)
        if self._record:
            # will save bytes here
            self.scene.metadata["recording"] = []

        # store kwargs
        self.kwargs = kwargs

        # store a vertexlist for an axis marker
        self._axis = None
        # store a vertexlist for a grid display
        self._grid = None
        # store scene geometry as vertex lists
        self.vertex_list = {}
        # store geometry hashes
        self.vertex_list_hash = {}
        # store geometry rendering mode
        self.vertex_list_mode = {}
        # store meshes that don't rotate relative to viewer
        self.fixed = fixed
        # store a hidden (don't not display) node.
        self._nodes_hidden = set()
        # name : texture
        self.textures = {}

        # if resolution isn't defined set a default value
        if resolution is None:
            resolution = scene.camera.resolution
        else:
            scene.camera.resolution = resolution

        # set the default line settings to a fraction
        # of our resolution so the points aren't tiny
        scale = max(resolution)
        self.line_settings = {"point_size": scale / 200, "line_width": scale / 400}
        # if we've been passed line settings override the default
        if line_settings is not None:
            self.line_settings.update(line_settings)

        # no window conf was passed so try to get the best looking one
        if window_conf is None:
            try:
                # try enabling antialiasing
                # if you have a graphics card this will probably work
                conf = gl.Config(sample_buffers=1, samples=4, depth_size=24, double_buffer=True)
                super(SceneViewer, self).__init__(
                    config=conf,
                    visible=visible,
                    resizable=True,
                    width=resolution[0],
                    height=resolution[1],
                    caption=caption,
                )
            except pyglet.window.NoSuchConfigException:
                conf = gl.Config(double_buffer=True)
                super(SceneViewer, self).__init__(
                    config=conf,
                    resizable=True,
                    visible=visible,
                    width=resolution[0],
                    height=resolution[1],
                    caption=caption,
                )
        else:
            # window config was manually passed
            super(SceneViewer, self).__init__(
                config=window_conf,
                resizable=True,
                visible=visible,
                width=resolution[0],
                height=resolution[1],
                caption=caption,
            )

        # add scene geometry to viewer geometry
        self._update_vertex_list()

        # call after geometry is added
        self.init_gl()
        self.set_size(*resolution)
        if flags is not None:
            self.reset_view(flags=flags)
        self.update_flags()

        # someone has passed a callback to be called periodically
        if self.callback is not None:
            # if no callback period is specified set it to default
            if callback_period is None:
                # 30 times per second
                callback_period = 1.0 / 30.0
            # set up a do-nothing periodic task which will
            # trigger `self.on_draw` every `callback_period`
            # seconds if someone has passed a callback
            pyglet.clock.schedule_interval(lambda x: x, callback_period)
        if start_loop:
            pyglet.app.run()

    def _redraw(self):
        self.on_draw()

    def _update_vertex_list(self):
        # update vertex_list if needed
        for name, geom in self.scene.geometry.items():
            if geom.is_empty:
                continue
            if geometry_hash(geom) == self.vertex_list_hash.get(name):
                continue
            self.add_geometry(name=name, geometry=geom, smooth=bool(self._smooth))

    def _update_meshes(self):
        # call the callback if specified
        if self.callback is not None:
            self.callback(self.scene)
            self._update_vertex_list()
            self._update_perspective(self.width, self.height)

    def add_geometry(self, name, geometry, **kwargs):
        """
        Add a geometry to the viewer.

        Parameters
        --------------
        name : hashable
          Name that references geometry
        geometry : Trimesh, Path2D, Path3D, PointCloud
          Geometry to display in the viewer window
        kwargs **
          Passed to rendering.convert_to_vertexlist
        """
        try:
            # convert geometry to constructor args
            args = rendering.convert_to_vertexlist(geometry, **kwargs)
        except BaseException:
            util.log.warning("failed to add geometry `{}`".format(name), exc_info=True)
            return

        # create the indexed vertex list
        self.vertex_list[name] = self.batch.add_indexed(*args)
        # save the hash of the geometry
        self.vertex_list_hash[name] = geometry_hash(geometry)
        # save the rendering mode from the constructor args
        self.vertex_list_mode[name] = args[1]

        # get the visual if the element has it
        visual = getattr(geometry, "visual", None)
        if hasattr(visual, "uv") and hasattr(visual, "material"):
            try:
                tex = rendering.material_to_texture(visual.material)
                if tex is not None:
                    self.textures[name] = tex
            except BaseException:
                util.log.warning("failed to load texture", exc_info=True)

    def cleanup_geometries(self):
        """
        Remove any stored vertex lists that no longer
        exist in the scene.
        """
        # shorthand to scene graph
        graph = self.scene.graph
        # which parts of the graph still have geometry
        geom_keep = set([graph[node][1] for node in graph.nodes_geometry])
        # which geometries no longer need to be kept
        geom_delete = [geom for geom in self.vertex_list if geom not in geom_keep]
        for geom in geom_delete:
            # remove stored vertex references
            self.vertex_list.pop(geom, None)
            self.vertex_list_hash.pop(geom, None)
            self.vertex_list_mode.pop(geom, None)
            self.textures.pop(geom, None)

    def unhide_geometry(self, node):
        """
        If a node is hidden remove the flag and show the
        geometry on the next draw.

        Parameters
        -------------
        node : str
          Node to display
        """
        self._nodes_hidden.discard(node)

    def hide_geometry(self, node):
        """
        Don't display the geometry contained at a node on
        the next draw.

        Parameters
        -------------
        node : str
          Node to not display
        """
        self._nodes_hidden.add(node)

    def reset_view(self, flags=None):
        """
        Set view to the default view.

        Parameters
        --------------
        flags : None or dict
          If any view key passed override the default
          e.g. {'cull': False}
        """
        self.view = {
            "cull": True,
            "axis": False,
            "grid": False,
            "fullscreen": False,
            "wireframe": False,
            "ball": Trackball(
                pose=self._initial_camera_transform,
                size=self.scene.camera.resolution,
                scale=self.scene.scale,
                target=self.scene.centroid,
            ),
        }
        try:
            # if any flags are passed override defaults
            if isinstance(flags, dict):
                for k, v in flags.items():
                    if k in self.view:
                        self.view[k] = v
                self.update_flags()
        except BaseException:
            pass

    def init_gl(self):
        """
        Perform the magic incantations to create an
        OpenGL scene using pyglet.
        """

        # if user passed a background color use it
        if self.background is None:
            # default background color is white
            background = np.ones(4)
        else:
            # convert to (4,) uint8 RGBA
            background = to_rgba(self.background)
            # convert to 0.0-1.0 float
            background = background.astype(np.float64) / 255.0

        self._gl_set_background(background)
        # use camera setting for depth
        self._gl_enable_depth(self.scene.camera)
        self._gl_enable_color_material()
        self._gl_enable_blending()
        self._gl_enable_smooth_lines(**self.line_settings)
        self._gl_enable_lighting(self.scene)

    @staticmethod
    def _gl_set_background(background):
        gl.glClearColor(*background)

    @staticmethod
    def _gl_unset_background():
        gl.glClearColor(*[0, 0, 0, 0])

    @staticmethod
    def _gl_enable_depth(camera):
        """
        Enable depth test in OpenGL using distances
        from `scene.camera`.
        """
        gl.glClearDepth(1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

    @staticmethod
    def _gl_enable_color_material():
        # do some openGL things
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glShadeModel(gl.GL_SMOOTH)

        gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, rendering.vector_to_gl(0.192250, 0.192250, 0.192250))
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, rendering.vector_to_gl(0.507540, 0.507540, 0.507540))
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, rendering.vector_to_gl(0.5082730, 0.5082730, 0.5082730))

        gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 0.4 * 128.0)

    @staticmethod
    def _gl_enable_blending():
        # enable blending for transparency
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    @staticmethod
    def _gl_enable_smooth_lines(line_width=4, point_size=4):
        # make the lines from Path3D objects less ugly
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
        # set the width of lines to 4 pixels
        gl.glLineWidth(line_width)
        # set PointCloud markers to 4 pixels in size
        gl.glPointSize(point_size)

    @staticmethod
    def _gl_enable_lighting(scene):
        """
        Take the lights defined in scene.lights and
        apply them as openGL lights.
        """
        gl.glEnable(gl.GL_LIGHTING)
        # opengl only supports 7 lights?
        for i, light in enumerate(scene.lights[:7]):
            # the index of which light we have
            lightN = eval("gl.GL_LIGHT{}".format(i))

            # get the transform for the light by name
            matrix = scene.graph.get(light.name)[0]

            # convert light object to glLightfv calls
            multiargs = rendering.light_to_gl(light=light, transform=matrix, lightN=lightN)

            # enable the light in question
            gl.glEnable(lightN)
            # run the glLightfv calls
            for args in multiargs:
                gl.glLightfv(*args)

    def toggle_culling(self):
        """
        Toggle back face culling.

        It is on by default but if you are dealing with
        non- watertight meshes you probably want to be able
        to see the back sides.
        """
        self.view["cull"] = not self.view["cull"]
        self.update_flags()

    def toggle_wireframe(self):
        """
        Toggle wireframe mode

        Good for  looking inside meshes, off by default.
        """
        self.view["wireframe"] = not self.view["wireframe"]
        self.update_flags()

    def toggle_fullscreen(self):
        """
        Toggle between fullscreen and windowed mode.
        """
        self.view["fullscreen"] = not self.view["fullscreen"]
        self.update_flags()

    def toggle_axis(self):
        """
        Toggle a rendered XYZ/RGB axis marker:
        off, world frame, every frame
        """
        # cycle through three axis states
        states = [False, "world", "all", "without_world"]
        # the state after toggling
        index = (states.index(self.view["axis"]) + 1) % len(states)
        # update state to next index
        self.view["axis"] = states[index]
        # perform gl actions
        self.update_flags()

    def toggle_grid(self):
        """
        Toggle a rendered grid.
        """
        # update state to next index
        self.view["grid"] = not self.view["grid"]
        # perform gl actions
        self.update_flags()

    def update_flags(self):
        """
        Check the view flags, and call required GL functions.
        """
        # view mode, filled vs wirefrom
        if self.view["wireframe"]:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        else:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

        # set fullscreen or windowed
        self.set_fullscreen(fullscreen=self.view["fullscreen"])

        # backface culling on or off
        if self.view["cull"]:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

        # case where we WANT an axis and NO vertexlist
        # is stored internally
        if self.view["axis"] and self._axis is None:
            from .. import creation

            # create an axis marker sized relative to the scene
            axis = creation.axis(origin_size=self.scene.scale / 100)
            # create ordered args for a vertex list
            args = rendering.mesh_to_vertexlist(axis)
            # store the axis as a reference
            self._axis = self.batch.add_indexed(*args)
        # case where we DON'T want an axis but a vertexlist
        # IS stored internally
        elif not self.view["axis"] and self._axis is not None:
            # remove the axis from the rendering batch
            self._axis.delete()
            # set the reference to None
            self._axis = None

        if self.view["grid"] and self._grid is None:
            try:
                # create a grid marker
                from ..path.creation import grid

                bounds = self.scene.bounds
                center = bounds.mean(axis=0)
                # set the grid to the lowest Z position
                # also offset by the scale to avoid interference
                center[2] = bounds[0][2] - (np.ptp(bounds[:, 2]) / 100)
                # choose the side length by maximum XY length
                side = np.ptp(bounds, axis=0)[:2].max()
                # create an axis marker sized relative to the scene
                grid_mesh = grid(side=side, count=4, transform=translation_matrix(center))
                # convert the path to vertexlist args
                args = rendering.convert_to_vertexlist(grid_mesh)
                # create ordered args for a vertex list
                self._grid = self.batch.add_indexed(*args)
            except BaseException:
                util.log.warning("failed to create grid!", exc_info=True)
        elif not self.view["grid"] and self._grid is not None:
            self._grid.delete()
            self._grid = None

    def _update_perspective(self, width, height):
        try:
            # for high DPI screens viewport size
            # will be different then the passed size
            width, height = self.get_viewport_size()
        except BaseException:
            # older versions of pyglet may not have this
            pass

        # set the new viewport size
        gl.glViewport(0, 0, width, height)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        # get field of view and Z range from camera
        camera = self.scene.camera

        # set perspective from camera data
        gl.gluPerspective(camera.fov[1], width / float(height), camera.z_near, camera.z_far)
        gl.glMatrixMode(gl.GL_MODELVIEW)

        return width, height

    def on_resize(self, width, height):
        """
        Handle resized windows.
        """
        width, height = self._update_perspective(width, height)
        self.scene.camera.resolution = (width, height)
        self.view["ball"].resize(self.scene.camera.resolution)
        self.scene.camera_transform = self.view["ball"].pose

    def on_mouse_press(self, x, y, buttons, modifiers):
        """
        Set the start point of the drag.
        """
        self.view["ball"].set_state(Trackball.STATE_ROTATE)
        if buttons == pyglet.window.mouse.LEFT:
            ctrl = modifiers & pyglet.window.key.MOD_CTRL
            shift = modifiers & pyglet.window.key.MOD_SHIFT
            if ctrl and shift:
                self.view["ball"].set_state(Trackball.STATE_ZOOM)
            elif shift:
                self.view["ball"].set_state(Trackball.STATE_ROLL)
            elif ctrl:
                self.view["ball"].set_state(Trackball.STATE_PAN)
        elif buttons == pyglet.window.mouse.MIDDLE:
            self.view["ball"].set_state(Trackball.STATE_PAN)
        elif buttons == pyglet.window.mouse.RIGHT:
            self.view["ball"].set_state(Trackball.STATE_ZOOM)

        self.view["ball"].down(np.array([x, y]))
        self.scene.camera_transform = self.view["ball"].pose

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """
        Pan or rotate the view.
        """
        self.view["ball"].drag(np.array([x, y]))
        self.scene.camera_transform = self.view["ball"].pose

    def on_mouse_scroll(self, x, y, dx, dy):
        """
        Zoom the view.
        """
        self.view["ball"].scroll(dy)
        self.scene.camera_transform = self.view["ball"].pose

    def on_key_press(self, symbol, modifiers):
        """
        Call appropriate functions given key presses.
        """
        magnitude = 10
        if symbol == pyglet.window.key.W:
            self.toggle_wireframe()
        elif symbol == pyglet.window.key.Z:
            self.reset_view()
        elif symbol == pyglet.window.key.C:
            self.toggle_culling()
        elif symbol == pyglet.window.key.A:
            self.toggle_axis()
        elif symbol == pyglet.window.key.G:
            self.toggle_grid()
        elif symbol == pyglet.window.key.Q:
            self.on_close()
        elif symbol == pyglet.window.key.M:
            self.maximize()
        elif symbol == pyglet.window.key.F:
            self.toggle_fullscreen()

        if symbol in [pyglet.window.key.LEFT, pyglet.window.key.RIGHT, pyglet.window.key.DOWN, pyglet.window.key.UP]:
            self.view["ball"].down([0, 0])
            if symbol == pyglet.window.key.LEFT:
                self.view["ball"].drag([-magnitude, 0])
            elif symbol == pyglet.window.key.RIGHT:
                self.view["ball"].drag([magnitude, 0])
            elif symbol == pyglet.window.key.DOWN:
                self.view["ball"].drag([0, -magnitude])
            elif symbol == pyglet.window.key.UP:
                self.view["ball"].drag([0, magnitude])
            self.scene.camera_transform = self.view["ball"].pose

    def on_draw(self):
        """
        Run the actual draw calls.
        """

        if self._profile:
            profiler = self.Profiler()
            profiler.start()

        self._update_meshes()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glLoadIdentity()

        # pull the new camera transform from the scene
        transform_camera = np.linalg.inv(self.scene.camera_transform)

        # apply the camera transform to the matrix stack
        gl.glMultMatrixf(rendering.matrix_to_gl(transform_camera))

        # we want to render fully opaque objects first,
        # followed by objects which have transparency
        node_names = collections.deque(self.scene.graph.nodes_geometry)
        # how many nodes did we start with
        count_original = len(node_names)
        count = -1

        # if we are rendering an axis marker at the world
        if self._axis and not self.view["axis"] == "without_world":
            # we stored it as a vertex list
            self._axis.draw(mode=gl.GL_TRIANGLES)
        if self._grid:
            self._grid.draw(mode=gl.GL_LINES)

        # save a reference outside of the loop
        geometry = self.scene.geometry
        graph = self.scene.graph

        while len(node_names) > 0:
            count += 1
            current_node = node_names.popleft()

            if current_node in self._nodes_hidden:
                continue

            # get the transform from world to geometry and mesh name
            transform, geometry_name = graph.get(current_node)

            # if no geometry at this frame continue without rendering
            if geometry_name is None or geometry_name not in self.vertex_list_mode:
                continue

            # if a geometry is marked as fixed apply the inverse view transform
            if self.fixed is not None and geometry_name in self.fixed:
                # remove altered camera transform from fixed geometry
                transform_fix = np.linalg.inv(np.dot(self._initial_camera_transform, transform_camera))
                # apply the transform so the fixed geometry doesn't move
                transform = np.dot(transform, transform_fix)

            # get a reference to the mesh so we can check transparency
            mesh = geometry[geometry_name]
            if mesh.is_empty:
                continue

            # get the GL mode of the current geometry
            mode = self.vertex_list_mode[geometry_name]

            # if you draw a coplanar line with a triangle it will z-fight
            # the best way to do this is probably a shader but this works fine
            if mode == gl.GL_LINES:
                # apply the offset in camera space
                transform = util.multi_dot(
                    [transform, np.linalg.inv(transform_camera), self._line_offset, transform_camera]
                )

            # add a new matrix to the model stack
            gl.glPushMatrix()
            # transform by the nodes transform
            gl.glMultMatrixf(rendering.matrix_to_gl(transform))

            # draw an axis marker for each mesh frame
            if self.view["axis"] == "all":
                self._axis.draw(mode=gl.GL_TRIANGLES)
            elif self.view["axis"] == "without_world":
                if not util.allclose(transform, np.eye(4), atol=1e-5):
                    self._axis.draw(mode=gl.GL_TRIANGLES)

            # transparent things must be drawn last
            if hasattr(mesh, "visual") and hasattr(mesh.visual, "transparency") and mesh.visual.transparency:
                # put the current item onto the back of the queue
                if count < count_original:
                    # add the node to be drawn last
                    node_names.append(current_node)
                    # pop the matrix stack for now
                    gl.glPopMatrix()
                    # come back to this mesh later
                    continue

            # if we have texture enable the target texture
            texture = None
            if geometry_name in self.textures:
                texture = self.textures[geometry_name]
                gl.glEnable(texture.target)
                gl.glBindTexture(texture.target, texture.id)

            # draw the mesh with its transform applied
            self.vertex_list[geometry_name].draw(mode=mode)
            # pop the matrix stack as we drew what we needed to draw
            gl.glPopMatrix()

            # disable texture after using
            if texture is not None:
                gl.glDisable(texture.target)

        if self._profile:
            profiler.stop()
            util.log.debug(profiler.output_text(unicode=True, color=True))

    def flip(self):
        super(SceneViewer, self).flip()
        if self._record:
            # will save a PNG-encoded bytes
            img = self.save_image(util.BytesIO())
            # seek start of file-like object
            img.seek(0)
            # save the bytes from the file object
            self.scene.metadata["recording"].append(img.read())

    def save_image(self, file_obj):
        """
        Save the current color buffer to a file object
        in PNG format.

        Parameters
        -------------
        file_obj: file name, or file- like object
        """
        manager = pyglet.image.get_buffer_manager()
        colorbuffer = manager.get_color_buffer()
        # if passed a string save by name
        if hasattr(file_obj, "write"):
            colorbuffer.save(file=file_obj)
        else:
            colorbuffer.save(filename=file_obj)
        return file_obj


def geometry_hash(geometry):
    """
    Get a hash for a geometry object

    Parameters
    ------------
    geometry : object

    Returns
    ------------
    hash : str
    """
    h = str(hash(geometry))
    if hasattr(geometry, "visual"):
        # if visual properties are defined
        h += str(hash(geometry.visual))

    return h


def render_scene(scene, resolution=None, visible=True, **kwargs):
    """
    Render a preview of a scene to a PNG. Note that
    whether this works or not highly variable based on
    platform and graphics driver.

    Parameters
    ------------
    scene : trimesh.Scene
      Geometry to be rendered
    resolution : (2,) int or None
      Resolution in pixels or set from scene.camera
    visible : bool
      Show a window during rendering. Note that MANY
      platforms refuse to render with hidden windows
      and will likely return a blank image; this is a
      platform issue and cannot be fixed in Python.
    kwargs : **
      Passed to SceneViewer

    Returns
    ---------
    render : bytes
      Image in PNG format
    """
    window = SceneViewer(scene, start_loop=False, visible=visible, resolution=resolution, **kwargs)

    from ..util import BytesIO

    # need to run loop twice to display anything
    for save in [False, False, True]:
        pyglet.clock.tick()
        window.switch_to()
        window.dispatch_events()
        window.dispatch_event("on_draw")
        window.flip()
        if save:
            # save the color buffer data to memory
            file_obj = BytesIO()
            window.save_image(file_obj)
            file_obj.seek(0)
            render = file_obj.read()
    window.close()

    return render
