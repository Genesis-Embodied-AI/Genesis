from .options import Options


class ViewerPlugin(Options):
    """
    Base class for viewer interaction options.

    All viewer interaction option classes should inherit from this base class.
    """


class HelpTextPlugin(ViewerPlugin):
    """
    Displays keyboard instructions in the viewer.
    """

    display_instructions: bool = True
    font_size: int = 26


class DefaultControlsPlugin(HelpTextPlugin):
    """
    Default viewer interaction controls with keyboard shortcuts for recording, changing render modes, etc.
    """
