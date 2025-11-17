from .options import Options


class ViewerPlugin(Options):
    """
    Base class for viewer plugin options.

    All viewer plugin option classes should inherit from this base class.
    """

    pass


class MouseSpringViewerPlugin(ViewerPlugin):
    """
    Options for the interactive viewer plugin that allows mouse-based object manipulation.
    """

    pass


class MeshPointSelectorPlugin(ViewerPlugin):
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
    """

    sphere_radius: float = 0.005
    sphere_color: tuple = (1.0, 0.0, 0.0, 1.0)
    hover_color: tuple = (1.0, 0.0, 0.0, 1.0)
    output_file: str = "selected_points.csv"
