from .options import Options


class ViewerInteraction(Options):
    """
    Base class for viewer interaction options.

    All viewer interaction option classes should inherit from this base class.
    """

    pass


class ViewerDefaultControls(ViewerInteraction):
    """
    Default viewer interaction controls with keyboard shortcuts for recording, changing render modes, etc.

    Parameters
    ----------
    keybindings : dict[str, int]
        Override the default mapping of action names to keyboard key codes (pyglet.window.key.*).
    """

    keybindings: dict[str, int] = None


class MouseSpringViewerPlugin(ViewerDefaultControls):
    """
    Options for the interactive viewer plugin that allows mouse-based object manipulation.
    """

    pass


class MeshPointSelectorPlugin(ViewerDefaultControls):
    """
    Options for the mesh point selector plugin that allows selecting points on a mesh.

    Parameters
    ----------
    sphere_radius : float
        The radius of the sphere used to visualize selected points.
    sphere_color : tuple
        The color of the sphere used to visualize selected points.
    hover_color : tuple
        The color of the sphere used to visualize the point and normal when hovering over a mesh.
    grid_snap : tuple[float, float, float]
        Grid snap spacing for each axis (x, y, z). Any negative value disables snapping for that axis.
        Default is (-1.0, -1.0, -1.0) which means no snapping.
    """

    sphere_radius: float = 0.005
    sphere_color: tuple = (0.1, 0.3, 1.0, 1.0)
    hover_color: tuple = (0.3, 0.5, 1.0, 1.0)
    grid_snap: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    output_file: str = "selected_points.csv"
