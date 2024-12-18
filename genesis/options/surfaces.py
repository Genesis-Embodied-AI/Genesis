from typing import Optional

import numpy as np

import genesis as gs

from .misc import FoamOptions
from .options import Options
from .textures import ColorTexture, ImageTexture, Texture


############################ Base ############################
class Surface(Options):
    """
    Base class for all surfaces types in Genesis.
    A ``Surface`` object encapsulates all visual information used for rendering an entity or its sub-components (links, geoms, etc.)
    The surface contains different types textures: diffuse_texture, specular_texture, roughness_texture, metallic_texture, transmission_texture, normal_texture, and emissive_texture. Each one of them is a `gs.textures.Texture` object.

    Tip
    ---
    If any of the textures only has single value (instead of a map), you can use the shortcut attribute (e.g., `color`, `roughness`, `metallic`, `emissive`) instead of creating a texture object.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    color : tuple | None, optional
        Diffuse color of the surface. Shortcut for `diffuse_texture` with a single color.
    opacity : float | None, optional
        Opacity of the surface. Shortcut for `opacity_texture` with a single value.
    roughness : float | None, optional
        Roughness of the surface. Shortcut for `roughness_texture` with a single value.
    metallic : float | None, optional
        Metallicness of the surface. Shortcut for `metallic_texture` with a single value.
    emissive : tuple | None, optional
        Emissive color of the surface. Shortcut for `emissive_texture` with a single color.
    ior : float, optional
        Index of Refraction.
    opacity_texture : gs.textures.Texture | None, optional
        Opacity texture of the surface.
    roughness_texture : gs.textures.Texture | None, optional
        Roughness texture of the surface.
    metallic_texture : gs.textures.Texture | None, optional
        Metallic texture of the surface.
    normal_texture : gs.textures.Texture | None, optional
        Normal texture of the surface.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    default_roughness : float, optional
        Default roughness value when `roughness` is not set and the asset does not have a roughness texture. Defaults to 1.0.
    vis_mode : str | None, optional
        How the entity should be visualized. Possible values are ['visual', 'particle', 'collision', 'sdf', 'recon'].

        - 'visual': Render the entity's visual geometry.
        - 'collision': Render the entity's collision geometry.
        - 'particle': Render the entity's particle representation (if applicable).
        - 'sdf': Render the reconstructed surface mesh of the entity's sdf.
        - 'recon': Render the reconstructed surface mesh of the entity's particle representation.

    smooth : bool, optional
        Whether to smooth face normals by interpolating vertex normals.
    double_sided : bool | None, optional
        Whether to render both sides of the surface. Useful for non-watertight 2D objects. Defaults to True for Cloth material and False for others.
    beam_angle : float
        The beam angle of emission. Defaults to 180.0.
    normal_diff_clamp : float, optional
        Controls the threshold for computing surface normals by interpolating vertex normals.
    recon_backend : str, optional
        Backend for surface reconstruction. Possible values are ['splashsurf', 'openvdb'].
    generate_foam : bool, optional
        Whether to generate foam particles for visual effects for particle-based entities.
    foam_options : gs.options.FoamOptions, optional
        Options for foam generation.
    """

    # shortcuts
    color: Optional[tuple] = None
    opacity: Optional[float] = None
    roughness: Optional[float] = None
    metallic: Optional[float] = None
    emissive: Optional[tuple] = None
    ior: Optional[float] = None

    # textures (can be either ColorTexture or ImageTexture)
    opacity_texture: Optional[Texture] = None
    roughness_texture: Optional[Texture] = None
    metallic_texture: Optional[Texture] = None
    normal_texture: Optional[Texture] = None
    emissive_texture: Optional[Texture] = None

    default_roughness: float = 1.0

    vis_mode: Optional[str] = None  # ['visual', 'particle', 'collision', 'sdf', 'recon']
    smooth: bool = True  # whether to smooth face normals by interpolating vertex normals
    double_sided: Optional[bool] = (
        None  # whether to render both sides of the surface. Defaults to True for Cloth material and False for others.
    )
    beam_angle: float = 180
    normal_diff_clamp: float = 180
    recon_backend: str = "splashsurf"  # backend for surface recon
    generate_foam: bool = False
    foam_options: Optional[FoamOptions] = None

    @staticmethod
    def shortcut_info(name, map_name):
        return f"`{name}` is a shortcut for texture. " f"When {name} is set, {map_name} setting is not allowed."

    def __init__(self, **data):
        super().__init__(**data)

        if self.foam_options is None:
            self.foam_options = FoamOptions()

        if self.color is not None:
            if self.get_texture() is not None:
                gs.raise_exception(self.shortcut_info("color", "texture"))
            self.set_texture(ColorTexture(color=self.color))

        if self.opacity is not None:
            if self.opacity_texture is not None:
                gs.raise_exception(self.shortcut_info("opacity", "opacity_texture"))
            self.opacity_texture = ColorTexture(color=(self.opacity,))

        if self.roughness is not None:
            if self.roughness_texture is not None:
                gs.raise_exception(self.shortcut_info("roughness", "roughness_texture"))
            self.roughness_texture = ColorTexture(color=(self.roughness,))

        if self.metallic is not None:
            if self.metallic_texture is not None:
                gs.raise_exception(self.shortcut_info("metallic", "metallic_texture"))
            self.metallic_texture = ColorTexture(color=(self.metallic,))

        if self.emissive is not None:
            if self.emissive_texture is not None:
                gs.raise_exception(self.shortcut_info("emissive", "emissive_texture"))
            self.emissive_texture = ColorTexture(color=self.emissive)

        color_texture = self.get_texture()
        if color_texture is not None:
            opacity_texture = color_texture.check_dim(3)
            if self.opacity_texture is None:
                self.opacity_texture = opacity_texture
        if self.emissive_texture is not None:
            opacity_texture = self.emissive_texture.check_dim(3)
            if self.opacity_texture is None:
                self.opacity_texture = opacity_texture

    def update_texture(
        self,
        color_texture=None,
        opacity_texture=None,
        roughness_texture=None,
        metallic_texture=None,
        normal_texture=None,
        emissive_texture=None,
        ior=None,
        double_sided=None,
        force=False,
    ):
        """
        Update the surface textures using given attributes.
        Note that if the surface already contains corresponding textures, the existing one have a higher priority and won't be overridden. Force overriding can be enable by setting force=True,
        """
        # diffuse map (or specular for glass and emissive for emission)
        if self.get_texture() is None or force:
            if color_texture is not None:
                self.set_texture(color_texture)
            elif not force:
                self.set_texture(ColorTexture())

        # update opacity
        if self.opacity_texture is None or force:
            if opacity_texture is not None:
                self.opacity_texture = opacity_texture
            elif not force:
                self.opacity_texture = ColorTexture(color=(1.0,))

        # update roughness
        if self.roughness_texture is None or force:
            if roughness_texture is not None:
                self.roughness_texture = roughness_texture
            elif not force:
                self.roughness_texture = ColorTexture(color=(self.default_roughness,))

        # update metallic
        if self.metallic_texture is None or force:
            if metallic_texture is not None:
                self.metallic_texture = metallic_texture

        # update normal
        if self.normal_texture is None or force:
            if normal_texture is not None:
                self.normal_texture = normal_texture

        # update emissive
        if self.emissive_texture is None or force:
            if emissive_texture is not None:
                self.emissive_texture = emissive_texture

        # update ior
        if self.ior is None or force:
            if ior is not None:
                self.ior = ior
            elif not force:
                self.ior = 1.5

        # update double sided
        if self.double_sided is None or force:
            if double_sided is not None:
                self.double_sided = double_sided

    def requires_uv(self):
        return (
            isinstance(self.get_texture(), ImageTexture)
            or isinstance(self.opacity_texture, ImageTexture)
            or isinstance(self.roughness_texture, ImageTexture)
            or isinstance(self.metallic_texture, ImageTexture)
            or isinstance(self.normal_texture, ImageTexture)
            or isinstance(self.emissive_texture, ImageTexture)
        )

    def get_rgba(self):
        texture = self.get_texture() if self.emissive_texture is None else self.emissive_texture
        opacity_texture = self.opacity_texture

        if isinstance(texture, ColorTexture):
            if isinstance(opacity_texture, ColorTexture):
                return ColorTexture(color=(*texture.color, *opacity_texture.color))

            elif isinstance(opacity_texture, ImageTexture) and opacity_texture.image_array is not None:
                rgb_color = np.round(np.array(texture.color) * 255).astype(np.uint8)
                rgb_array = np.full((*opacity_texture.image_array.shape[:2], 3), rgb_color, dtype=np.uint8)
                rgba_array = np.dstack((rgb_array, opacity_texture.image_array))
                rgba_scale = (1.0, 1.0, 1.0, *opacity_texture.image_color)
                return ImageTexture(image_array=rgba_array, image_color=rgba_scale)

            else:
                return ColorTexture(color=(*texture.color, 1.0))

        elif isinstance(texture, ImageTexture) and texture.image_array is not None:
            if isinstance(opacity_texture, ColorTexture):
                a_color = np.round(np.array(opacity_texture.color) * 255).astype(np.uint8)
                a_array = np.full((*texture.image_array.shape[:2],), a_color, dtype=np.uint8)
                rgba_array = np.dstack((texture.image_array, a_array))
                rgba_scale = (*texture.image_color, 1.0)

            elif (
                isinstance(opacity_texture, ImageTexture)
                and opacity_texture.image_array is not None
                and opacity_texture.image_array.shape[:2] == texture.image_array.shape[:2]
            ):
                rgba_array = np.dstack((texture.image_array, opacity_texture.image_array))
                rgba_scale = (*texture.image_color, *opacity_texture.image_color)
            else:
                if isinstance(opacity_texture, ImageTexture) and opacity_texture.image_array is not None:
                    gs.logger.warning("Color and opacity image shapes do not match. Fall back to fully opaque texture.")
                a_array = np.full((*texture.image_array.shape[:2],), 255, dtype=np.uint8)
                rgba_array = np.dstack((texture.image_array, a_array))
                rgba_scale = (*texture.image_color, 1.0)

            return ImageTexture(image_array=rgba_array, image_color=rgba_scale)

        else:
            return ColorTexture(color=(1.0, 1.0, 1.0, 1.0))

    def set_texture(self, texture):
        raise NotImplementedError

    def get_texture(self):
        raise NotImplementedError

    def get_emission(self):
        return self.emissive_texture


############################ Three surface types ############################
class Glass(Surface):
    """
    Base class for all surfaces types in Genesis.
    A ``Surface`` object encapsulates all visual information used for rendering an entity or its sub-components (links, geoms, etc.)
    The surface contains different types textures: diffuse_texture, specular_texture, roughness_texture, metallic_texture, normal_texture, and emissive_texture. Each one of them is a `gs.textures.Texture` object.

    Tip
    ---
    If any of the textures only has single value (instead of a map), you can use the shortcut attribute (e.g., `color`, `roughness`, `metallic`, `emissive`) instead of creating a texture object.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    color : tuple | None, optional
        Diffuse color of the surface. Shortcut for `diffuse_texture` with a single color.
    roughness : float | None, optional
        Roughness of the surface. Shortcut for `roughness_texture` with a single value.
    ior : float, optional
        Index of Refraction.
    subsurface : bool
        Whether apply a simple BSSRDF subsurface to the glass material.
    thickness : float | None, optional
        The thickness of the top surface when 'subsurface' is set to True, that is, the maximum distance of subsurface scattering. Shortcut for `thickness_texture` with a single value.
    metallic : float | None, optional
        Metallicness of the surface. Shortcut for `metallic_texture` with a single value.
    emissive : tuple | None, optional
        Emissive color of the surface. Shortcut for `emissive_texture` with a single color.
    specular_texture : gs.textures.Texture | None, optional
        Specular texture of the surface.
    transmission_texture : gs.textures.Texture | None, optional
        Transmission texture of the surface.
    opacity_texture : gs.textures.Texture | None, optional
        Opacity texture of the surface.
    roughness_texture : gs.textures.Texture | None, optional
        Roughness texture of the surface.
    metallic_texture : gs.textures.Texture | None, optional
        Metallic texture of the surface.
    normal_texture : gs.textures.Texture | None, optional
        Normal texture of the surface.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    thickness_texture : gs.textures.Texture | None, optional
        The thickness of the top surface.
    default_roughness : float, optional
        Default roughness value when `roughness` is not set and the asset does not have a roughness texture. Defaults to 1.0.
    vis_mode : str | None, optional
        How the entity should be visualized. Possible values are ['visual', 'particle', 'collision', 'sdf', 'recon'].

        - 'visual': Render the entity's visual geometry.
        - 'collision': Render the entity's collision geometry.
        - 'particle': Render the entity's particle representation (if applicable).
        - 'sdf': Render the reconstructed surface mesh of the entity's sdf.
        - 'recon': Render the reconstructed surface mesh of the entity's particle representation.

    smooth : bool, optional
        Whether to smooth face normals by interpolating vertex normals.
    double_sided : bool | None, optional
        Whether to render both sides of the surface. Useful for non-watertight 2D objects. Defaults to True for Cloth material and False for others.
    normal_diff_clamp : float, optional
        Controls the threshold for computing surface normals by interpolating vertex normals.
    recon_backend : str, optional
        Backend for surface reconstruction. Possible values are ['splashsurf', 'openvdb'].
    generate_foam : bool, optional
        Whether to generate foam particles for visual effects for particle-based entities.
    foam_options : gs.options.FoamOptions, optional
        Options for foam generation.
    """

    roughness: float = 0.0
    ior: float = 1.5
    subsurface: bool = False
    thickness: Optional[float] = None

    thickness_texture: Optional[Texture] = None
    specular_texture: Optional[Texture] = None
    transmission_texture: Optional[Texture] = None

    def __init__(self, **data):
        super().__init__(**data)

        if self.thickness is not None:
            if self.thickness_texture is not None:
                gs.raise_exception(self.shortcut_info("thickness", "thickness_texture"))
            self.thickness_texture = ColorTexture(color=(self.thickness,))

    def get_texture(self):
        return self.specular_texture

    def set_texture(self, texture):
        # for simplicity, let's use the same texture for specular and transmission
        self.specular_texture = texture
        self.transmission_texture = texture


class Metal(Surface):
    """
    Metal surface.

    Parameters
    ----------
    color : tuple | None, optional
        Diffuse color of the surface. Shortcut for `diffuse_texture` with a single color.
    roughness : float | None, optional
        Roughness of the surface. Shortcut for `roughness_texture` with a single value.
    metallic : float | None, optional
        Metallicness of the surface. Shortcut for `metallic_texture` with a single value.
    emissive : tuple | None, optional
        Emissive color of the surface. Shortcut for `emissive_texture` with a single color.
    metal_type : str, optional
        Type of metal, indicating a specific index of refraction (IOR). Possible values are ['aluminium', 'gold', 'copper', 'brass', 'iron', 'titanium', 'vanadium', 'lithium'].
    diffuse_texture : gs.textures.Texture | None, optional
        Diffuse (basic color) texture of the surface.
    opacity_texture : gs.textures.Texture | None, optional
        Opacity texture of the surface.
    roughness_texture : gs.textures.Texture | None, optional
        Roughness texture of the surface.
    metallic_texture : gs.textures.Texture | None, optional
        Metallic texture of the surface.
    normal_texture : gs.textures.Texture | None, optional
        Normal texture of the surface.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    default_roughness : float, optional
        Default roughness value when `roughness` is not set and the asset does not have a roughness texture. Defaults to 1.0.
    vis_mode : str | None, optional
        How the entity should be visualized. Possible values are ['visual', 'particle', 'collision', 'sdf', 'recon'].

        - 'visual': Render the entity's visual geometry.
        - 'collision': Render the entity's collision geometry.
        - 'particle': Render the entity's particle representation (if applicable).
        - 'sdf': Render the reconstructed surface mesh of the entity's sdf.
        - 'recon': Render the reconstructed surface mesh of the entity's particle representation.

    smooth : bool, optional
        Whether to smooth face normals by interpolating vertex normals.
    double_sided : bool | None, optional
        Whether to render both sides of the surface. Useful for non-watertight 2D objects. Defaults to True for Cloth material and False for others.
    normal_diff_clamp : float, optional
        Controls the threshold for computing surface normals by interpolating vertex normals.
    recon_backend : str, optional
        Backend for surface reconstruction. Possible values are ['splashsurf', 'openvdb'].
    generate_foam : bool, optional
        Whether to generate foam particles for visual effects for particle-based entities.
    foam_options : gs.options.FoamOptions, optional
        Options for foam generation.
    """

    roughness: Optional[float] = 0.1
    metal_type: Optional[str] = "iron"
    diffuse_texture: Optional[Texture] = None

    def __init__(self, **data):
        super().__init__(**data)

        if self.metal_type not in ["aluminium", "gold", "copper", "brass", "iron", "titanium", "vanadium", "lithium"]:
            gs.raise_exception(f"Invalid metal metal_type: {self.metal_type}.")

    def get_texture(self):
        return self.diffuse_texture

    def set_texture(self, texture):
        self.diffuse_texture = texture


class Plastic(Surface):
    """
    Plastic surface is the most basic type of surface.

    Parameters
    ----------
    color : tuple | None, optional
        Diffuse color of the surface. Shortcut for `diffuse_texture` with a single color.
    roughness : float | None, optional
        Roughness of the surface. Shortcut for `roughness_texture` with a single value.
    metallic : float | None, optional
        Metallicness of the surface. Shortcut for `metallic_texture` with a single value.
    emissive : tuple | None, optional
        Emissive color of the surface. Shortcut for `emissive_texture` with a single color.
    ior : float, optional
        Index of Refraction.
    diffuse_texture : gs.textures.Texture | None, optional
        Diffuse (basic color) texture of the surface.
    specular_texture : gs.textures.Texture | None, optional
        Specular texture of the surface.
    opacity_texture : gs.textures.Texture | None, optional
        Opacity texture of the surface.
    roughness_texture : gs.textures.Texture | None, optional
        Roughness texture of the surface.
    metallic_texture : gs.textures.Texture | None, optional
        Metallic texture of the surface.
    normal_texture : gs.textures.Texture | None, optional
        Normal texture of the surface.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    default_roughness : float, optional
        Default roughness value when `roughness` is not set and the asset does not have a roughness texture. Defaults to 1.0.
    vis_mode : str | None, optional
        How the entity should be visualized. Possible values are ['visual', 'particle', 'collision', 'sdf', 'recon'].

        - 'visual': Render the entity's visual geometry.
        - 'collision': Render the entity's collision geometry.
        - 'particle': Render the entity's particle representation (if applicable).
        - 'sdf': Render the reconstructed surface mesh of the entity's sdf.
        - 'recon': Render the reconstructed surface mesh of the entity's particle representation.

    smooth : bool, optional
        Whether to smooth face normals by interpolating vertex normals.
    double_sided : bool | None, optional
        Whether to render both sides of the surface. Useful for non-watertight 2D objects. Defaults to True for Cloth material and False for others.
    normal_diff_clamp : float, optional
        Controls the threshold for computing surface normals by interpolating vertex normals.
    recon_backend : str, optional
        Backend for surface reconstruction. Possible values are ['splashsurf', 'openvdb'].
    generate_foam : bool, optional
        Whether to generate foam particles for visual effects for particle-based entities.
    foam_options : gs.options.FoamOptions, optional
        Options for foam generation.
    """

    ior: Optional[float] = 1.0
    diffuse_texture: Optional[Texture] = None
    specular_texture: Optional[Texture] = None

    def get_texture(self):
        return self.diffuse_texture

    def set_texture(self, texture):
        self.diffuse_texture = texture


class BSDF(Surface):
    """
    Plastic surface is the most basic type of surface.

    Parameters
    ----------
    color : tuple | None, optional
        Diffuse color of the surface. Shortcut for `diffuse_texture` with a single color.
    roughness : float | None, optional
        Roughness of the surface. Shortcut for `roughness_texture` with a single value.
    metallic : float | None, optional
        Metallicness of the surface. Shortcut for `metallic_texture` with a single value.
    emissive : tuple | None, optional
        Emissive color of the surface. Shortcut for `emissive_texture` with a single color.
    ior : float, optional
        Index of Refraction.
    diffuse_texture : gs.textures.Texture | None, optional
        Diffuse (basic color) texture of the surface.
    specular_tint : gs.textures.Texture | None, optional
        Specular texture of the surface.
    opacity_texture : gs.textures.Texture | None, optional
        Opacity texture of the surface.
    roughness_texture : gs.textures.Texture | None, optional
        Roughness texture of the surface.
    metallic_texture : gs.textures.Texture | None, optional
        Metallic texture of the surface.
    normal_texture : gs.textures.Texture | None, optional
        Normal texture of the surface.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    default_roughness : float, optional
        Default roughness value when `roughness` is not set and the asset does not have a roughness texture. Defaults to 1.0.
    vis_mode : str | None, optional
        How the entity should be visualized. Possible values are ['visual', 'particle', 'collision', 'sdf', 'recon'].

        - 'visual': Render the entity's visual geometry.
        - 'collision': Render the entity's collision geometry.
        - 'particle': Render the entity's particle representation (if applicable).
        - 'sdf': Render the reconstructed surface mesh of the entity's sdf.
        - 'recon': Render the reconstructed surface mesh of the entity's particle representation.

    smooth : bool, optional
        Whether to smooth face normals by interpolating vertex normals.
    double_sided : bool | None, optional
        Whether to render both sides of the surface. Useful for non-watertight 2D objects. Defaults to True for Cloth material and False for others.
    normal_diff_clamp : float, optional
        Controls the threshold for computing surface normals by interpolating vertex normals.
    recon_backend : str, optional
        Backend for surface reconstruction. Possible values are ['splashsurf', 'openvdb'].
    generate_foam : bool, optional
        Whether to generate foam particles for visual effects for particle-based entities.
    foam_options : gs.options.FoamOptions, optional
        Options for foam generation.
    """

    diffuse_texture: Optional[Texture] = None
    specular_trans: Optional[float] = 0.0
    diffuse_trans: Optional[float] = 0.0

    def get_texture(self):
        return self.diffuse_texture

    def set_texture(self, texture):
        self.diffuse_texture = texture


class Emission(Surface):
    """
    Emission surface. This surface emits light. Note that in Genesis's ray tracing pipeline, lights are not a special type of objects, but simply entities with emission surfaces.

    Parameters
    ----------
    emissive : tuple | None, optional
        Emissive color of the surface. Shortcut for `emissive_texture` with a single color.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    """

    def get_texture(self):
        return self.emissive_texture

    def set_texture(self, texture):
        self.emissive_texture = texture


############################ Handy shortcuts ############################
class Default(BSDF):
    """
    The default surface type used in Genesis. This is just an alias for `Plastic`.
    """

    pass


class Rough(Plastic):
    """
    Shortcut for a rough plastic surface.
    """

    roughness: float = 1.0
    ior: float = 1.5


class Smooth(Plastic):
    """
    Shortcut for a smooth plastic surface.
    """

    roughness: float = 0.1
    ior: float = 1.5


class Reflective(Plastic):
    """
    Shortcut for a reflective (smoother than `Smooth`) plastic surface.
    """

    roughness: float = 0.01
    ior: float = 2.0


class Collision(Plastic):
    """
    Default surface type for collision geometry with a grey color.
    """

    def __init__(self, **data):
        super().__init__(**data)
        self.diffuse_texture = ColorTexture(color=(0.5, 0.5, 0.5))


class Water(Glass):
    """
    Shortcut for a water surface (using Glass surface with proper values).
    """

    color: tuple = (0.61, 0.98, 0.93)
    roughness: float = 0.2
    ior: float = 1.2

    def __init__(self, **data):
        super().__init__(**data)


class Iron(Metal):
    """
    Shortcut for an metallic surface with `metal_type = 'iron'`.
    """

    metal_type: str = "iron"


class Aluminium(Metal):
    """
    Shortcut for an metallic surface with `metal_type = 'aluminium'`.
    """

    metal_type: str = "aluminium"


class Copper(Metal):
    """
    Shortcut for an metallic surface with `metal_type = 'copper'`.
    """

    metal_type: str = "copper"


class Gold(Metal):
    """
    Shortcut for an metallic surface with `metal_type = 'gold'`.
    """

    metal_type: str = "gold"
