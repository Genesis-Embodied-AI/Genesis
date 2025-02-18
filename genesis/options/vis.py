from typing import Optional

import genesis as gs

from .options import Options


class ViewerOptions(Options):
    """
    Options configuring preperties of the interactive viewer.

    Note
    ----
    The viewer's camera uses the `Rasterizer` backend regardless of `gs.renderers.*` when creating the scene.

    Parameters
    ----------
    res : tuple, shape (2,), optional
        The resolution of the viewer. If not set, will auto-compute using resolution of the connected display.
    refresh_rate : int
        The refresh rate of the viewer.
    max_FPS : int | None
        The FPS (frames per second) the viewer will be capped at. Note that this will also synchronize the simulation speed. If not set, the viewer will render at maximum speed.
    camera_pos : tuple of float, shape (3,)
        The position of the viewer's camera.
    camera_lookat : tuple of float, shape (3,)
        The lookat position that the camera.
    camera_up : tuple of float, shape (3,)
        The up vector of the camera's extrinsic pose.
    camera_fov : float
        The field of view (in degrees) of the camera.
    """

    res: Optional[tuple] = None
    refresh_rate: int = 60
    max_FPS: Optional[int] = 60
    camera_pos: tuple = (3.5, 0.5, 2.5)
    camera_lookat: tuple = (0.0, 0.0, 0.5)
    camera_up: tuple = (0.0, 0.0, 1.0)
    camera_fov: float = 40


class VisOptions(Options):
    """
    This configures visualization-related properties that are independent of the viewer or camera.

    Parameters
    ----------
    show_world_frame : bool
        Whether to visualize the world frame.
    world_frame_size : float
        The length (in meters) of the world frame's axes.
    show_link_frame : bool
        Whether to visualize the frames of each RigidLink.
    link_frame_size : float
        The length (in meters) of the link frames' axes.
    show_cameras : bool
        Whether to render the cameras added to the scene, together with their frustums.
    shadow : bool
        Whether to render shadow. Defaults to True.
    plane_reflection : bool
        Whether to render plane reflection. Defaults to False.
    env_separate_rigid : bool
        Whether to share rigid objects across environments. Disabled when shown by the viewer. Defaults to False.
    background_color : tuple of float, shape (3,)
        The color of the scene background.
    ambient_light : tuple of float, shape (3,)
        The color of the scene's ambient light.
    visualize_mpm_boundary : bool
        Whether to visualize the boundary of the MPM Solver.
    visualize_sph_boundary : bool
        Whether to visualize the boundary of the SPH Solver.
    visualize_pbd_boundary : bool
        Whether to visualize the boundary of the PBD Solver.
    segmentation_level : str
        The segmentation level used for segmentation mask rendering. Should be one of ['entity', 'link', 'geom']. Defaults to 'link'.
    render_particle_as : str
        How particles in the scene should be rendered. Should be one of ['sphere', 'tet']. Defaults to 'sphere'.
    particle_size_scale : float
        Scale applied to actual particle size for rendering. Defaults to 1.0.
    contact_force_scale : float = 0.02
        Scale for contact arrow visualization, m/N. E.g. the force arrow representing 10N will be 0.2m long if scale is 0.02. Defaults to 0.02.
    n_support_neighbors : int
        Number of supporting neighbor particles used to compute vertex position of the visual mesh. Used for rendering deformable bodies. Defaults to 12.
    n_rendered_envs : int, optional
        Number of environments with being rendered. If None, all environments will be rendered. Defaults to None.
    lights  : list of dict.
        Lights added to the scene.
    """

    show_world_frame: bool = True
    world_frame_size: float = 1.0
    show_link_frame: bool = False
    link_frame_size: float = 0.2
    show_cameras: bool = False
    shadow: bool = True
    plane_reflection: bool = False
    env_separate_rigid: bool = False
    background_color: tuple = (0.04, 0.08, 0.12)
    ambient_light: tuple = (0.1, 0.1, 0.1)
    visualize_mpm_boundary: bool = False
    visualize_sph_boundary: bool = False
    visualize_pbd_boundary: bool = False
    segmentation_level: str = "link"  # ['entity', 'link', 'geom']
    render_particle_as: str = "sphere"  # ['sphere', 'tet']
    particle_size_scale: float = 1.0  # scale applied to actual particle size for rendering
    contact_force_scale: float = (
        0.01  # scale of force visualization, m/N. E.g. the force arrow representing 10N wille be 0.1m long if scale is 0.01.
    )
    n_support_neighbors: int = (
        12  # number of neighbor particles used to compute vertex position of the visual mesh. Used for rendering deformable bodies.
    )
    n_rendered_envs: Optional[int] = None  # number of environments being rendered
    lights: list = [
        {"type": "directional", "dir": (-1, -1, -1), "color": (1.0, 1.0, 1.0), "intensity": 5.0},
    ]

    def __init__(self, **data):
        super().__init__(**data)

        assert self.segmentation_level in ["entity", "link", "geom"]

        if self.render_particle_as not in ["sphere", "tet"]:
            gs.raise_exception(
                f"Unsupported `render_particle_as`: {self.render_particle_as}, must be one of ['sphere', 'tet']"
            )
