from typing import Annotated, Literal

from pydantic import StrictBool, StrictInt, Field, model_validator

import genesis as gs
from genesis.datatypes import List
from genesis.typing import IArrayType, PositiveFloat, PositiveInt, PositiveVec2IType, Vec3FType, Color3Type

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
    run_in_thread : bool
        Whether to run the viewer in a background thread. This option is not supported on MacOS. True by default if
        available.
    refresh_rate : int
        The refresh rate of the viewer.
    max_FPS : int | None
        The FPS (frames per second) the viewer will be capped at. Note that this will also synchronize the simulation
        speed. If not set, the viewer will render at maximum speed.
    camera_pos : tuple of float, shape (3,)
        The position of the viewer's camera.
    camera_lookat : tuple of float, shape (3,)
        The lookat position that the camera.
    camera_up : tuple of float, shape (3,)
        The up vector of the camera's extrinsic pose.
    camera_fov : float
        The field of view (in degrees) of the camera.
    enable_help_text : bool
        Whether to enable the rendering of instructions text in the viewer.
    enable_default_keybinds : bool
        Whether to enable the default keyboard controls in the viewer.
    """

    res: PositiveVec2IType | None = None
    run_in_thread: StrictBool | None = None
    refresh_rate: PositiveInt = 60
    max_FPS: PositiveInt | None = 60
    camera_pos: Vec3FType = (3.5, 0.5, 2.5)
    camera_lookat: Vec3FType = (0.0, 0.0, 0.5)
    camera_up: Vec3FType = (0.0, 0.0, 1.0)
    camera_fov: float = 40
    enable_help_text: StrictBool = True
    enable_default_keybinds: StrictBool = True


class DirectionalLight(Options):
    type: Literal["directional"] = "directional"
    dir: Vec3FType
    color: Color3Type
    intensity: float


class PointLight(Options):
    type: Literal["point"] = "point"
    pos: Vec3FType
    color: Color3Type
    intensity: float


class AmbientLight(Options):
    type: Literal["ambient"] = "ambient"
    color: Color3Type
    intensity: float


LightType = Annotated[DirectionalLight | PointLight | AmbientLight, Field(discriminator="type")]


class VisOptions(Options):
    """
    This configures visualization-related properties that are independent of the viewer or camera.

    Parameters
    ----------
    show_world_frame : bool
        Whether to visualize the world frame. Default to False.
    world_frame_size : float
        The length (in meters) of the world frame's axes.
    show_link_frame : bool
        Whether to visualize the frames of each RigidLink. Default to False.
    link_frame_size : float
        The length (in meters) of the link frames' axes.
    show_cameras : bool
        Whether to render the cameras added to the scene, together with their frustums. Default to False.
    shadow : bool
        Whether to render shadow. Defaults to True.
    plane_reflection : bool
        Whether to render plane reflection. Defaults to False.
    env_separate_rigid : bool
        Whether to render all the rigid objects in batched environments in isolation or as part of the same scene.
        This is only an option for Rasterizer. This behavior is enforced for BatchRender. Defaults to False.
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
        The segmentation level used for segmentation mask rendering. Should be one of ['entity', 'link', 'geom'].
        Defaults to 'link'.
    render_particle_as : str
        How particles in the scene should be rendered. Should be one of ['sphere', 'tet']. Defaults to 'sphere'.
    particle_size_scale : float
        Scale applied to actual particle size for rendering. Defaults to 1.0.
    contact_force_scale : float = 0.02
        Scale in m.N^{-1} for contact arrow visualization, e.g. the force arrow representing 10N will be 0.2m long if
        scale is 0.02. Defaults to 0.01.
    n_support_neighbors : int
        Number of supporting neighbor particles used to compute vertex position of the visual mesh. Used for rendering
        deformable bodies. Defaults to 12.
    rendered_envs_idx : list, optional
        Indices of the environments that will be rendered. If not provided, all the environments will be considered.
        Defaults to None.
    n_rendered_envs : int, optional
        This option is deprecated. Please use `rendered_envs_idx` instead.
    lights : list of dict.
        Lights added to the scene.
    """

    show_world_frame: StrictBool = False
    world_frame_size: float = 1.0
    show_link_frame: StrictBool = False
    link_frame_size: float = 0.2
    show_cameras: StrictBool = False
    shadow: StrictBool = True
    plane_reflection: StrictBool = False
    env_separate_rigid: StrictBool = False
    background_color: Color3Type = (0.04, 0.08, 0.12)
    ambient_light: Color3Type = (0.1, 0.1, 0.1)
    visualize_mpm_boundary: StrictBool = False
    visualize_sph_boundary: StrictBool = False
    visualize_pbd_boundary: StrictBool = False
    segmentation_level: Literal["entity", "link", "geom"] = "link"
    render_particle_as: Literal["sphere", "tet"] = "sphere"
    particle_size_scale: PositiveFloat = 1.0
    contact_force_scale: PositiveFloat = 0.01
    n_support_neighbors: StrictInt = 12
    rendered_envs_idx: IArrayType | None = None
    lights: Annotated[List[LightType], Field(validate_default=True, strict=False)] = List(
        (DirectionalLight(dir=(-1, -1, -1), color=(1.0, 1.0, 1.0), intensity=5.0),)
    )

    @model_validator(mode="before")
    @classmethod
    def _handle_deprecated_n_rendered_envs(cls, data: dict) -> dict:
        if "n_rendered_envs" in data:
            gs.logger.warning(
                "Viewer option 'n_rendered_envs' is deprecated and will be removed in a future release. "
                "Please use 'rendered_envs_idx' instead."
            )
            if data.get("rendered_envs_idx") is not None:
                raise ValueError("Cannot specify both 'n_rendered_envs' and 'rendered_envs_idx'.")
            data["rendered_envs_idx"] = tuple(range(data.pop("n_rendered_envs")))
        return data
