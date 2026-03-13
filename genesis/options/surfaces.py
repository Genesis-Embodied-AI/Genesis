import math
from typing import ClassVar, Literal
from typing_extensions import Self

import numpy as np
from pydantic import Field, StrictBool, model_validator

import genesis as gs
from genesis.typing import UnitInterval
from genesis.utils import mesh as mu

from .misc import FoamOptions
from .options import Options
from .textures import Texture, ColorTexture, ImageTexture, BatchTexture

MetalType = Literal["aluminium", "gold", "copper", "brass", "iron", "titanium", "vanadium", "lithium"]


############################ Base ############################
class Surface(Options):
    """
    Base class for all surfaces types in Genesis.

    A ``Surface`` object encapsulates all visual information used for rendering an entity or its sub-components (links,
    geoms, ...). The surface contains different types of textures depending on the surface type (e.g. diffuse, specular,
    roughness, metallic, normal, emissive). Each one of them is a `gs.textures.Texture` object.

    Tip
    ---
    If any of the textures only has single value (instead of a map), you can use the shortcut parameter (e.g., `color`,
    `roughness`, `metallic`, `emissive`) instead of creating a texture object.

    Note
    ----
    This class should *not* be instantiated directly.

    Parameters
    ----------
    color : tuple | None, optional
        Color of the surface. Shortcut for the primary texture with a single color.
    opacity : float | None, optional
        Opacity of the surface. Shortcut for `opacity_texture` with a single value.
    roughness : float | None, optional
        Roughness of the surface. Shortcut for `roughness_texture` with a single value.
    metallic : float | None, optional
        Metalness of the surface. Shortcut for `metallic_texture` with a single value.
    emissive : tuple | None, optional
        Emissive color of the surface. Shortcut for `emissive_texture` with a single color.
    ior : float, optional
        Index of Refraction.
    default_roughness : float, optional
        Default roughness value when `roughness` is not set and the asset does not have a roughness texture. Defaults
        to 1.0.
    vis_mode : str | None, optional
        How the entity should be visualized, e.g.
        - 'visual': Render the entity's visual geometry.
        - 'collision': Render the entity's collision geometry.
        - 'particle': Render the entity's particle representation (if applicable).
        - 'sdf': Render the reconstructed surface mesh of the entity's sdf.
        - 'recon': Render the reconstructed surface mesh of the entity's particle representation.
    smooth : bool, optional
        Whether to smooth face normals by interpolating vertex normals.
    double_sided : bool | None, optional
        Whether to render both sides of the surface. Useful for non-watertight 2D objects. Defaults to True for Cloth
        material and False for others.
    cutoff : float
        The cutoff angle of emission. Defaults to 180.0.
    normal_diff_clamp : float, optional
        Controls the threshold for computing surface normals by interpolating vertex normals.
    recon_backend : str, optional
        Backend for surface reconstruction. Possible values are ['splashsurf', 'openvdb']. Defaults to 'splashsurf'.
    generate_foam : bool, optional
        Whether to generate foam particles for visual effects for particle-based entities.
    foam_options : gs.options.FoamOptions, optional
        Options for foam generation.
    """

    _color_target: ClassVar[str] = "diffuse_texture"

    ior: float | None = None
    default_roughness: UnitInterval = 1.0
    vis_mode: Literal["visual", "collision", "particle", "sdf", "recon"] | None = None
    smooth: StrictBool = True
    double_sided: StrictBool | None = None
    cutoff: float = 180.0
    normal_diff_clamp: float = 180.0
    recon_backend: Literal["splashsurf", "openvdb"] = "splashsurf"
    generate_foam: StrictBool = False
    foam_options: FoamOptions = Field(default_factory=FoamOptions)

    def __init__(
        self,
        *,
        color: tuple | None = None,
        opacity: float | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
        emissive: tuple | None = None,
        **data,
    ) -> None:
        super().__init__(
            color=color,
            opacity=opacity,
            roughness=roughness,
            metallic=metallic,
            emissive=emissive,
            **data,
        )

    @model_validator(mode="before")
    @classmethod
    def _resolve_shortcuts(cls, data: dict) -> dict:
        color_target = cls._color_target
        color = data.pop("color", None)
        if color is not None:
            if data.get(color_target) is not None:
                gs.raise_exception(f"'color' and '{color_target}' cannot both be set.")
            data[color_target] = ColorTexture(color=tuple(color))

        for shortcut, texture_field in (
            ("opacity", "opacity_texture"),
            ("roughness", "roughness_texture"),
            ("metallic", "metallic_texture"),
        ):
            value = data.pop(shortcut, None)
            if value is not None:
                if data.get(texture_field) is not None:
                    gs.raise_exception(f"'{shortcut}' and '{texture_field}' cannot both be set.")
                data[texture_field] = ColorTexture(color=(float(value),))

        emissive = data.pop("emissive", None)
        if emissive is not None:
            if data.get("emissive_texture") is not None:
                gs.raise_exception("'emissive' and 'emissive_texture' cannot both be set.")
            data["emissive_texture"] = ColorTexture(color=tuple(emissive))

        return data

    @property
    def texture(self) -> Texture | None:
        raise NotImplementedError

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        raise NotImplementedError

    @property
    def emission(self) -> Texture | None:
        return None

    @property
    def requires_uv(self) -> bool:
        return False

    def get_rgba(self, batch: bool = False) -> BatchTexture | Texture:
        return self._make_rgba(self.texture, None, batch)

    def update_texture(
        self,
        *,
        color_texture: Texture | None = None,
        ior: float | None = None,
        double_sided: bool | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        """
        Update the surface textures using given attributes.

        If the surface already contains corresponding textures, the existing ones have higher priority and won't be
        overridden. Force overriding can be enabled by setting force=True.
        """
        # update primary texture
        if self.texture is None or force:
            if color_texture is not None:
                self.texture = color_texture
            elif not force:
                self.texture = ColorTexture()

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

    @staticmethod
    def _update_field(
        current: Texture | None, new: Texture | None, default: Texture | None, force: bool
    ) -> Texture | None:
        if current is None or force:
            if new is not None:
                return new
            elif not force and default is not None:
                return default
        return current

    @staticmethod
    def _extract_opacity_from(
        texture: Texture | None, emissive: Texture | None, opacity: Texture | None
    ) -> Texture | None:
        if texture is not None:
            tex = texture.check_dim(3)
            if opacity is None and tex is not None:
                opacity = tex
        if emissive is not None:
            tex = emissive.check_dim(3)
            if opacity is None and tex is not None:
                opacity = tex
        return opacity

    @staticmethod
    def _make_rgba(
        color_texture: Texture | None, opacity_texture: Texture | None, batch: bool
    ) -> BatchTexture | Texture:
        all_textures = []
        for texture in (color_texture, opacity_texture):
            textures = texture.textures if isinstance(texture, BatchTexture) else [texture]
            all_textures.append(textures if batch else textures[:1])
        color_textures, opacity_textures = all_textures

        rgba_textures = []
        num_colors = len(color_textures)
        num_opacities = len(opacity_textures)
        num_rgba = num_colors * num_opacities // math.gcd(num_colors, num_opacities)

        for i in range(num_rgba):
            color_texture = color_textures[i % num_colors]
            opacity_texture = opacity_textures[i % num_opacities]

            if isinstance(color_texture, ColorTexture):
                if isinstance(opacity_texture, ColorTexture):
                    rgba_texture = ColorTexture(color=(*color_texture.color, *opacity_texture.color))
                elif isinstance(opacity_texture, ImageTexture) and opacity_texture.image_array is not None:
                    rgb_color = mu.color_f32_to_u8(color_texture.color)
                    rgb_array = np.full((*opacity_texture.image_array.shape[:2], 3), rgb_color, dtype=np.uint8)
                    rgba_array = np.dstack((rgb_array, opacity_texture.image_array))
                    rgba_scale = (1.0, 1.0, 1.0, *opacity_texture.image_color)
                    rgba_texture = ImageTexture(image_array=rgba_array, image_color=rgba_scale)
                else:
                    rgba_texture = ColorTexture(color=(*color_texture.color, 1.0))

            elif isinstance(color_texture, ImageTexture) and color_texture.image_array is not None:
                if isinstance(opacity_texture, ColorTexture):
                    a_color = mu.color_f32_to_u8(opacity_texture.color)
                    a_array = np.full((*color_texture.image_array.shape[:2],), a_color, dtype=np.uint8)
                    rgba_array = np.dstack((color_texture.image_array, a_array))
                    rgba_scale = (*color_texture.image_color, 1.0)
                elif (
                    isinstance(opacity_texture, ImageTexture)
                    and opacity_texture.image_array is not None
                    and opacity_texture.image_array.shape[:2] == color_texture.image_array.shape[:2]
                ):
                    rgba_array = np.dstack((color_texture.image_array, opacity_texture.image_array))
                    rgba_scale = (*color_texture.image_color, *opacity_texture.image_color)
                else:
                    if isinstance(opacity_texture, ImageTexture) and opacity_texture.image_array is not None:
                        gs.logger.warning(
                            "Color and opacity image shapes do not match. Fall back to fully opaque texture."
                        )
                    a_array = np.full(color_texture.image_array.shape[:2], 255, dtype=np.uint8)
                    rgba_array = np.dstack((color_texture.image_array, a_array))
                    rgba_scale = (*color_texture.image_color, 1.0)
                rgba_texture = ImageTexture(image_array=rgba_array, image_color=rgba_scale)

            else:
                rgba_texture = ColorTexture(color=(1.0, 1.0, 1.0, 1.0))

            rgba_textures.append(rgba_texture)

        return BatchTexture(textures=rgba_textures) if batch else rgba_textures[0]


############################ Surface types ############################
class Glass(Surface):
    """
    Glass surface with specular reflection and transmission.

    Parameters
    ----------
    color : tuple | None, optional
        Specular color of the surface. Shortcut for `specular_texture` with a single color.
    roughness : float, optional
        Roughness of the surface. Defaults to 0.0.
    ior : float, optional
        Index of Refraction. Defaults to 1.5.
    subsurface : bool
        Whether to apply a simple BSSRDF subsurface to the glass material.
    thickness : float | None, optional
        The thickness of the top surface when 'subsurface' is set to True. Shortcut for `thickness_texture`.
    specular_texture : gs.textures.Texture | None, optional
        Specular texture of the surface.
    diffuse_texture : gs.textures.Texture | None, optional
        Diffuse texture of the surface.
    transmission_texture : gs.textures.Texture | None, optional
        Transmission texture of the surface.
    thickness_texture : gs.textures.Texture | None, optional
        The thickness texture of the top surface.
    roughness_texture : gs.textures.Texture | None, optional
        Roughness texture of the surface.
    normal_texture : gs.textures.Texture | None, optional
        Normal texture of the surface.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    """

    _color_target: ClassVar[str] = "specular_texture"

    subsurface: StrictBool = False
    specular_texture: Texture | None = None
    diffuse_texture: Texture | None = None
    transmission_texture: Texture | None = None
    thickness_texture: Texture | None = None
    roughness_texture: Texture | None = None
    normal_texture: Texture | None = None
    emissive_texture: Texture | None = None

    def __init__(self, *, roughness: float = 0.0, ior: float = 1.5, thickness: float | None = None, **data) -> None:
        super().__init__(
            roughness=roughness,
            ior=ior,
            thickness=thickness,
            default_roughness=data.pop("default_roughness", roughness),
            **data,
        )

    @model_validator(mode="before")
    @classmethod
    def _resolve_glass_shortcuts(cls, data: dict) -> dict:
        thickness = data.pop("thickness", None)
        if thickness is not None:
            if data.get("thickness_texture") is not None:
                gs.raise_exception("'thickness' and 'thickness_texture' cannot both be set.")
            data["thickness_texture"] = ColorTexture(color=(float(thickness),))
        return data

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        # Truncate specular/emissive textures to 3 channels (discard alpha for Glass which has no opacity_texture)
        if self.specular_texture is not None:
            self.specular_texture.check_dim(3)
        if self.emissive_texture is not None:
            self.emissive_texture.check_dim(3)
        if self.specular_texture is not None and self.transmission_texture is None:
            self.transmission_texture = self.specular_texture
        return self

    @property
    def texture(self) -> Texture | None:
        return self.specular_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.specular_texture = value
        self.transmission_texture = value

    @property
    def emission(self) -> Texture | None:
        return self.emissive_texture

    @property
    def requires_uv(self) -> bool:
        return any(
            t is not None and t.requires_uv
            for t in (
                self.specular_texture,
                self.diffuse_texture,
                self.transmission_texture,
                self.thickness_texture,
                self.roughness_texture,
                self.normal_texture,
                self.emissive_texture,
            )
        )

    def get_rgba(self, batch: bool = False) -> BatchTexture | Texture:
        color = self.emissive_texture if self.emissive_texture is not None else self.specular_texture
        return self._make_rgba(color, None, batch)

    def update_texture(
        self,
        *,
        roughness_texture: Texture | None = None,
        normal_texture: Texture | None = None,
        emissive_texture: Texture | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        super().update_texture(force=force, **kwargs)
        self.roughness_texture = self._update_field(
            self.roughness_texture, roughness_texture, ColorTexture(color=(self.default_roughness,)), force
        )
        self.normal_texture = self._update_field(self.normal_texture, normal_texture, None, force)
        self.emissive_texture = self._update_field(self.emissive_texture, emissive_texture, None, force)


class Metal(Surface):
    """
    Metal surface.

    Parameters
    ----------
    color : tuple | None, optional
        Diffuse color of the surface. Shortcut for `diffuse_texture` with a single color.
    metal_type : str, optional
        Type of metal, indicating a specific index of refraction (IOR). Possible values are ['aluminium', 'gold',
        'copper', 'brass', 'iron', 'titanium', 'vanadium', 'lithium']. Defaults to 'iron'.
    diffuse_texture : gs.textures.Texture | None, optional
        Diffuse (basic color) texture of the surface.
    opacity_texture : gs.textures.Texture | None, optional
        Opacity texture of the surface.
    roughness_texture : gs.textures.Texture | None, optional
        Roughness texture of the surface.
    normal_texture : gs.textures.Texture | None, optional
        Normal texture of the surface.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    """

    metal_type: MetalType = "iron"
    diffuse_texture: Texture | None = None
    opacity_texture: Texture | None = None
    roughness_texture: Texture | None = None
    normal_texture: Texture | None = None
    emissive_texture: Texture | None = None

    def __init__(self, *, roughness: float = 0.1, **data) -> None:
        super().__init__(roughness=roughness, default_roughness=data.pop("default_roughness", roughness), **data)

    @property
    def texture(self) -> Texture | None:
        return self.diffuse_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.diffuse_texture = value

    @property
    def emission(self) -> Texture | None:
        return self.emissive_texture

    @property
    def requires_uv(self) -> bool:
        return any(
            t is not None and t.requires_uv
            for t in (
                self.diffuse_texture,
                self.opacity_texture,
                self.roughness_texture,
                self.normal_texture,
                self.emissive_texture,
            )
        )

    def get_rgba(self, batch: bool = False) -> BatchTexture | Texture:
        color = self.emissive_texture if self.emissive_texture is not None else self.diffuse_texture
        return self._make_rgba(color, self.opacity_texture, batch)

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        self.opacity_texture = self._extract_opacity_from(
            self.diffuse_texture, self.emissive_texture, self.opacity_texture
        )
        return self

    def update_texture(
        self,
        *,
        opacity_texture: Texture | None = None,
        roughness_texture: Texture | None = None,
        normal_texture: Texture | None = None,
        emissive_texture: Texture | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        super().update_texture(force=force, **kwargs)
        self.opacity_texture = self._update_field(
            self.opacity_texture, opacity_texture, ColorTexture(color=(1.0,)), force
        )
        self.roughness_texture = self._update_field(
            self.roughness_texture, roughness_texture, ColorTexture(color=(self.default_roughness,)), force
        )
        self.normal_texture = self._update_field(self.normal_texture, normal_texture, None, force)
        self.emissive_texture = self._update_field(self.emissive_texture, emissive_texture, None, force)


class Plastic(Surface):
    """
    Plastic surface is the most basic type of surface.

    Parameters
    ----------
    color : tuple | None, optional
        Diffuse color of the surface. Shortcut for `diffuse_texture` with a single color.
    ior : float, optional
        Index of Refraction. Defaults to 1.0.
    diffuse_texture : gs.textures.Texture | None, optional
        Diffuse (basic color) texture of the surface.
    specular_texture : gs.textures.Texture | None, optional
        Specular texture of the surface.
    opacity_texture : gs.textures.Texture | None, optional
        Opacity texture of the surface.
    roughness_texture : gs.textures.Texture | None, optional
        Roughness texture of the surface.
    normal_texture : gs.textures.Texture | None, optional
        Normal texture of the surface.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    """

    diffuse_texture: Texture | None = None
    specular_texture: Texture | None = None
    opacity_texture: Texture | None = None
    roughness_texture: Texture | None = None
    normal_texture: Texture | None = None
    emissive_texture: Texture | None = None

    def __init__(self, *, ior: float = 1.0, **data) -> None:
        super().__init__(ior=ior, **data)

    @property
    def texture(self) -> Texture | None:
        return self.diffuse_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.diffuse_texture = value

    @property
    def emission(self) -> Texture | None:
        return self.emissive_texture

    @property
    def requires_uv(self) -> bool:
        return any(
            t is not None and t.requires_uv
            for t in (
                self.diffuse_texture,
                self.specular_texture,
                self.opacity_texture,
                self.roughness_texture,
                self.normal_texture,
                self.emissive_texture,
            )
        )

    def get_rgba(self, batch: bool = False) -> BatchTexture | Texture:
        color = self.emissive_texture if self.emissive_texture is not None else self.diffuse_texture
        return self._make_rgba(color, self.opacity_texture, batch)

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        self.opacity_texture = self._extract_opacity_from(
            self.diffuse_texture, self.emissive_texture, self.opacity_texture
        )
        return self

    def update_texture(
        self,
        *,
        opacity_texture: Texture | None = None,
        roughness_texture: Texture | None = None,
        normal_texture: Texture | None = None,
        emissive_texture: Texture | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        super().update_texture(force=force, **kwargs)
        self.opacity_texture = self._update_field(
            self.opacity_texture, opacity_texture, ColorTexture(color=(1.0,)), force
        )
        self.roughness_texture = self._update_field(
            self.roughness_texture, roughness_texture, ColorTexture(color=(self.default_roughness,)), force
        )
        self.normal_texture = self._update_field(self.normal_texture, normal_texture, None, force)
        self.emissive_texture = self._update_field(self.emissive_texture, emissive_texture, None, force)


class BSDF(Surface):
    """
    Disney BSDF surface with principled shading.

    Parameters
    ----------
    color : tuple | None, optional
        Diffuse color of the surface. Shortcut for `diffuse_texture` with a single color.
    ior : float, optional
        Index of Refraction. Defaults to 1.0.
    specular_trans : float, optional
        Specular transmission. Defaults to 0.0.
    diffuse_trans : float, optional
        Diffuse transmission. Defaults to 0.0.
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
    """

    diffuse_texture: Texture | None = None
    opacity_texture: Texture | None = None
    roughness_texture: Texture | None = None
    metallic_texture: Texture | None = None
    normal_texture: Texture | None = None
    emissive_texture: Texture | None = None
    specular_trans: float = 0.0
    diffuse_trans: float = 0.0

    def __init__(self, *, ior: float = 1.0, **data) -> None:
        super().__init__(ior=ior, **data)

    @property
    def texture(self) -> Texture | None:
        return self.diffuse_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.diffuse_texture = value

    @property
    def emission(self) -> Texture | None:
        return self.emissive_texture

    @property
    def requires_uv(self) -> bool:
        return any(
            t is not None and t.requires_uv
            for t in (
                self.diffuse_texture,
                self.opacity_texture,
                self.roughness_texture,
                self.metallic_texture,
                self.normal_texture,
                self.emissive_texture,
            )
        )

    def get_rgba(self, batch: bool = False) -> BatchTexture | Texture:
        color = self.emissive_texture if self.emissive_texture is not None else self.diffuse_texture
        return self._make_rgba(color, self.opacity_texture, batch)

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        self.opacity_texture = self._extract_opacity_from(
            self.diffuse_texture, self.emissive_texture, self.opacity_texture
        )
        return self

    def update_texture(
        self,
        *,
        opacity_texture: Texture | None = None,
        roughness_texture: Texture | None = None,
        metallic_texture: Texture | None = None,
        normal_texture: Texture | None = None,
        emissive_texture: Texture | None = None,
        force: bool = False,
        **kwargs,
    ) -> None:
        super().update_texture(force=force, **kwargs)
        self.opacity_texture = self._update_field(
            self.opacity_texture, opacity_texture, ColorTexture(color=(1.0,)), force
        )
        self.roughness_texture = self._update_field(
            self.roughness_texture, roughness_texture, ColorTexture(color=(self.default_roughness,)), force
        )
        self.metallic_texture = self._update_field(self.metallic_texture, metallic_texture, None, force)
        self.normal_texture = self._update_field(self.normal_texture, normal_texture, None, force)
        self.emissive_texture = self._update_field(self.emissive_texture, emissive_texture, None, force)


class Emission(Surface):
    """
    Emission surface. This surface emits light. Note that in Genesis's ray tracing pipeline, lights are not a special
    type of objects, but simply entities with emission surfaces.

    Parameters
    ----------
    color : tuple | None, optional
        Emissive color. Shortcut for `emissive_texture` with a single color.
    emissive : tuple | None, optional
        Emissive color. Shortcut for `emissive_texture` with a single color.
    emissive_texture : gs.textures.Texture | None, optional
        Emissive texture of the surface.
    """

    _color_target: ClassVar[str] = "emissive_texture"

    emissive_texture: Texture | None = None

    @property
    def texture(self) -> Texture | None:
        return self.emissive_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.emissive_texture = value

    @property
    def emission(self) -> Texture | None:
        return self.emissive_texture

    @property
    def requires_uv(self) -> bool:
        return self.emissive_texture is not None and self.emissive_texture.requires_uv

    def get_rgba(self, batch: bool = False) -> BatchTexture | Texture:
        return self._make_rgba(self.emissive_texture, None, batch)

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        if self.emissive_texture is not None:
            self.emissive_texture.check_dim(3)
        return self

    def update_texture(self, *, emissive_texture: Texture | None = None, force: bool = False, **kwargs) -> None:
        super().update_texture(force=force, **kwargs)
        self.emissive_texture = self._update_field(self.emissive_texture, emissive_texture, None, force)


############################ Handy shortcuts ############################
class Default(BSDF):
    """
    The default surface type used in Genesis. This is an alias for `BSDF`.
    """

    pass


class Rough(Plastic):
    """
    Shortcut for a rough plastic surface.
    """

    def __init__(self, *, roughness: float = 1.0, ior: float = 1.5, **data) -> None:
        super().__init__(
            roughness=roughness, ior=ior, default_roughness=data.pop("default_roughness", roughness), **data
        )


class Smooth(Plastic):
    """
    Shortcut for a smooth plastic surface.
    """

    def __init__(self, *, roughness: float = 0.1, ior: float = 1.5, **data) -> None:
        super().__init__(
            roughness=roughness, ior=ior, default_roughness=data.pop("default_roughness", roughness), **data
        )


class Reflective(Plastic):
    """
    Shortcut for a reflective (smoother than `Smooth`) plastic surface.
    """

    def __init__(self, *, roughness: float = 0.01, ior: float = 2.0, **data) -> None:
        super().__init__(
            roughness=roughness, ior=ior, default_roughness=data.pop("default_roughness", roughness), **data
        )


class Collision(Plastic):
    """
    Default surface type for collision geometry with a grey color by default.
    """

    def __init__(self, *, color: tuple = (0.5, 0.5, 0.5), **data) -> None:
        super().__init__(color=color, **data)


class Water(Glass):
    """
    Shortcut for a water surface (using Glass surface with proper values).
    """

    def __init__(self, *, color: tuple = (0.61, 0.98, 0.93), roughness: float = 0.2, ior: float = 1.2, **data) -> None:
        super().__init__(
            color=color,
            roughness=roughness,
            ior=ior,
            default_roughness=data.pop("default_roughness", roughness),
            **data,
        )


class Iron(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'iron'`.
    """

    pass


class Aluminium(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'aluminium'`.
    """

    def __init__(self, *, metal_type: MetalType = "aluminium", **data) -> None:
        super().__init__(metal_type=metal_type, **data)


class Copper(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'copper'`.
    """

    def __init__(self, *, metal_type: MetalType = "copper", **data) -> None:
        super().__init__(metal_type=metal_type, **data)


class Gold(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'gold'`.
    """

    def __init__(self, *, metal_type: MetalType = "gold", **data) -> None:
        super().__init__(metal_type=metal_type, **data)
