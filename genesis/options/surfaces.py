import functools
import math
from typing import Any, ClassVar, Literal
from typing_extensions import Self

import numpy as np
from pydantic import Field, InstanceOf, PrivateAttr, StrictBool, model_validator

import genesis as gs
from genesis.typing import FArrayType, UnitInterval, ValidFloat
from genesis.utils import mesh as mu

from .misc import FoamOptions
from .options import Options
from .textures import Texture, ColorTexture, ImageTexture, BatchTexture

MetalType = Literal["aluminium", "gold", "copper", "brass", "iron", "titanium", "vanadium", "lithium"]

METAL_COLOR: dict[MetalType, tuple[float, float, float]] = {
    "iron": (0.530, 0.513, 0.494),
    "aluminium": (0.916, 0.923, 0.924),
    "copper": (0.932, 0.623, 0.522),
    "gold": (1.000, 0.773, 0.307),
    "brass": (0.910, 0.778, 0.423),
    "titanium": (0.441, 0.400, 0.361),
    "vanadium": (0.534, 0.526, 0.546),
    "lithium": (0.916, 0.890, 0.807),
}


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

    # Set True by `finalize_texture()`. Consumers that need fully-populated defaults (e.g.
    # `surface_uvs_to_trimesh_visual`) should assert this is True before reading texture fields.
    _finalized: StrictBool = PrivateAttr(default=False)

    # Shortcut fields — resolved to texture fields by _resolve_shortcuts, excluded from serialization.
    color: FArrayType | None = Field(default=None, exclude=True, repr=False)
    opacity: UnitInterval | None = Field(default=None, exclude=True, repr=False)
    roughness: UnitInterval | None = Field(default=None, exclude=True, repr=False)
    metallic: UnitInterval | None = Field(default=None, exclude=True, repr=False)
    emissive: FArrayType | None = Field(default=None, exclude=True, repr=False)

    ior: float | None = None
    default_ior: float = 1.0
    default_roughness: UnitInterval = 1.0
    default_color: FArrayType = (1.0, 1.0, 1.0)
    default_opacity: UnitInterval = 1.0
    vis_mode: Literal["visual", "collision", "particle", "sdf", "recon"] | None = None
    smooth: StrictBool = True
    double_sided: StrictBool | None = None
    cutoff: float = 180.0
    normal_diff_clamp: float = 180.0
    recon_backend: Literal["splashsurf", "openvdb"] = "splashsurf"
    generate_foam: StrictBool = False
    foam_options: FoamOptions = Field(default_factory=FoamOptions)

    @model_validator(mode="before")
    @classmethod
    def _resolve_shortcuts(cls, data: Any) -> Any:
        # Route each shortcut into its texture counterpart. Subclasses that don't expose a given texture (e.g. Glass has
        # no opacity_texture) are skipped via the model_fields guard, and class defaults like `Rough.roughness = 1.0`
        # are honored.
        for shortcut, texture_field in (
            ("color", cls._color_target),
            ("opacity", "opacity_texture"),
            ("roughness", "roughness_texture"),
            ("metallic", "metallic_texture"),
            ("thickness", "thickness_texture"),
            ("emissive", "emissive_texture"),
        ):
            if texture_field not in cls.model_fields:
                continue
            field = cls.model_fields.get(shortcut)
            value = data.get(shortcut, field.default if field is not None else None)
            if value is None:
                continue
            if data.get(texture_field) is not None:
                gs.raise_exception(f"'{shortcut}' and '{texture_field}' cannot both be set.")
            data[texture_field] = ColorTexture(color=value)

        return data

    @property
    def texture(self) -> Texture | None:
        raise NotImplementedError

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        raise NotImplementedError

    @property
    def emissive_tex(self) -> Texture | None:
        raise NotImplementedError

    @property
    def opacity_tex(self) -> Texture | None:
        raise NotImplementedError

    @property
    def roughness_tex(self) -> Texture | None:
        raise NotImplementedError

    @property
    def metallic_tex(self) -> Texture | None:
        raise NotImplementedError

    @property
    def requires_uv(self) -> bool:
        return False

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        if len(self.default_color) > 3:
            self.default_opacity = self.default_color[3]
            self.default_color = self.default_color[:3]
        return self

    def get_rgba(self, batch: bool = False) -> BatchTexture | Texture:
        return self._make_rgba(
            self.emissive_tex if self.emissive_tex is not None else self.texture, self.opacity_tex, batch
        )

    def get_arm(self, batch: bool = False) -> BatchTexture | Texture:
        return self._make_arm(self.roughness_tex, self.metallic_tex, batch)

    def update_texture(
        self,
        *,
        color_texture: Texture | None = None,
        ior: float | None = None,
        double_sided: bool | None = None,
        **kwargs,
    ) -> None:
        """
        Layer texture fields onto the surface: fill `None` fields ONLY from explicit values.

        Existing (already-set) fields are preserved. Fields with no provided value remain `None` so further
        `update_texture` calls (e.g. asset material after user surface) can still fill them. Defaults are NOT
        populated here — call `finalize_texture()` once when layering is complete.

        Raises if the surface has already been finalized — layering after finalize is a programming error.
        """
        if self._finalized:
            gs.raise_exception(
                f"Cannot `update_texture` on a finalized {type(self).__name__}. "
                "All layering must happen before `finalize_texture()`."
            )

        self.texture = self._default_if_none(self.texture, color_texture)
        self.ior = self._default_if_none(self.ior, ior)
        self.double_sided = self._default_if_none(self.double_sided, double_sided)

    def finalize_texture(self) -> None:
        """
        Populate any still-unset texture fields with class defaults and mark the surface finalized.

        Call once when the surface is committed and no further layering will happen. Sets `_finalized = True`
        so downstream consumers (e.g. `surface_uvs_to_trimesh_visual`) can guard against unfinalized surfaces.
        Idempotent.
        """
        if self._finalized:
            gs.raise_exception(f"Cannot `finalize_texture` on a finalized {type(self).__name__}. ")
        self._finalized = True
        self.texture = self._default_if_none(self.texture, ColorTexture(color=self.default_color))
        self.ior = self._default_if_none(self.ior, self.default_ior)
        self.roughness = self._default_if_none(self.roughness, self.default_roughness)

    @staticmethod
    def _default_if_none(current, default):
        return default if current is None else current

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
    def _ref_shape(component_dict: dict) -> "tuple[int, int] | None":
        """
        Return the common 2D `(h, w)` shape of all `ImageTexture` inputs, or `None` if none are images.

        Warns when shapes mismatch and returns the first image's shape; callers should re-check each
        ImageTexture against the returned shape and treat mismatched ones as scalar fallback.
        """
        shape: tuple[int, int] | None = None
        for name, component in component_dict.items():
            if component[1]:  # has image
                cur_shape = component[0].image_array.shape[:2]
                if shape is None:
                    shape = cur_shape
                elif cur_shape != shape:
                    gs.logger.warning(
                        f"Texture `{name}` shapes do not match: {shape} vs {cur_shape}. "
                        "It will fall back to scalar value."
                    )
                    component[1] = False
        return shape

    @staticmethod
    @functools.lru_cache(maxsize=128)
    def _combine_textures(input_specs, batch: bool) -> "BatchTexture | Texture":
        """
        Pack channel-grouped components into a single output Texture.

        ``input_specs`` is a tuple of ``(name, texture, default_scale, default_fill)`` tuples; each
        tuple contributes its channels to the output in order. Used by both `_make_rgba` (color RGB
        + opacity A) and `_make_arm` (AO + roughness + metallic).

        Memoized on the input texture instances so surfaces sharing the same textures (e.g. all
        textured submeshes of a GLB) reuse one merged array instead of allocating one per surface.
        Callers must pass ``input_specs`` as a tuple (not a list) so the cache key is hashable.

        - All-scalar inputs → `ColorTexture` with concatenated channel values.
        - Any input is an `ImageTexture` → `ImageTexture` whose channels are dstacked; for scalar
          inputs, the image array is filled with `default_fill` (uint8) and the actual value is
          carried in the per-channel `image_color` scale.
        """
        # Expand each BatchTexture into a list of per-frame Textures; pin to length 1 in non-batch mode.
        expanded = [
            (
                name,
                (t.textures if isinstance(t, BatchTexture) else [t])[: None if batch else 1],
                default_scale,
                default_fill,
            )
            for name, t, default_scale, default_fill in input_specs
        ]

        num_iter = math.lcm(*(len(textures) for _, textures, _, _ in expanded)) if batch else 1

        results = []
        for i in range(num_iter):
            # Build the per-iteration component_dict that `_ref_shape` and the packing logic share.
            component_dict = {
                name: [
                    textures[i % len(textures)],
                    textures[i % len(textures)] is not None and textures[i % len(textures)].has_image,
                    default_scale,
                    default_fill,
                ]
                for name, textures, default_scale, default_fill in expanded
            }
            ref_shape = Surface._ref_shape(component_dict)

            # Per-channel provenance: True iff the user supplied an explicit input for this
            # component. Stamped on every result texture so downstream consumers (e.g. the Nyx
            # exporter's matOverride clearing path) can tell "user override" from "default fill".
            explicit = {name: (c[0] is not None) for name, c in component_dict.items()}

            # All-scalar: pack ColorTexture.color or default into one ColorTexture.
            if ref_shape is None:
                packed = sum(
                    (c[0].color if isinstance(c[0], ColorTexture) else c[2] for c in component_dict.values()),
                    start=(),
                )
                results.append(ColorTexture(color=packed, explicit_channels=explicit))
                continue

            scale = sum(
                (
                    tuple(c[0].image_color[: len(c[2])])
                    if c[1]
                    else tuple(c[0].color[: len(c[2])])
                    if isinstance(c[0], ColorTexture)
                    else c[2]
                    for c in component_dict.values()
                ),
                start=(),
            )
            array = np.dstack(
                tuple(
                    (c[0].image_array[:, :, : len(c[2])] if c[0].image_array.ndim == 3 else c[0].image_array)
                    if c[1]
                    else np.tile(np.asarray(c[3], dtype=np.uint8), (*ref_shape, 1))
                    for c in component_dict.values()
                )
            )
            results.append(ImageTexture(image_array=array, image_color=scale, explicit_channels=explicit))

        return BatchTexture(textures=results) if batch else results[0]

    @staticmethod
    def _make_rgba(
        color_texture: Texture | None, opacity_texture: Texture | None, batch: bool
    ) -> BatchTexture | Texture:
        return Surface._combine_textures(
            (
                ("color", color_texture, (1.0, 1.0, 1.0), (255, 255, 255)),
                ("opacity", opacity_texture, (1.0,), (255,)),
            ),
            batch,
        )

    @staticmethod
    def _make_arm(
        roughness_texture: Texture | None, metallic_texture: Texture | None, batch: bool
    ) -> BatchTexture | Texture:
        """
        Pack (AO=1.0, roughness, metallic) into a single 3-channel ARM texture.

        Matches Nyx's ARM convention (R=AO, G=roughness, B=metallic) and glTF's
        metallicRoughnessTexture layout. AO is always 1.0 (Genesis has no AO field).
        """
        return Surface._combine_textures(
            (
                ("ao", None, (1.0,), (255,)),
                ("roughness", roughness_texture, (1.0,), (255,)),
                ("metallic", metallic_texture, (0.0,), (255,)),
            ),
            batch,
        )


def clear_rgba_cache() -> None:
    """Clear the memoization cache used by ``Surface._combine_textures``.

    Mirrors upstream's ``_make_rgba.cache_clear()`` entry point invoked by ``genesis.destroy()``.
    """
    Surface._combine_textures.cache_clear()


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

    roughness: UnitInterval | None = Field(default=0.0, exclude=True, repr=False)
    ior: float | None = 1.5
    thickness: ValidFloat | None = Field(default=None, exclude=True, repr=False)

    subsurface: StrictBool = False
    specular_texture: Texture | None = None
    transmission_texture: Texture | None = None
    thickness_texture: Texture | None = None
    roughness_texture: Texture | None = None
    normal_texture: Texture | None = None
    emissive_texture: Texture | None = None

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        super()._post_init()
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
    def emissive_tex(self) -> Texture | None:
        return self.emissive_texture

    @property
    def opacity_tex(self) -> Texture | None:
        return None  # Glass has no opacity_texture field

    @property
    def roughness_tex(self) -> Texture | None:
        return self.roughness_texture

    @property
    def metallic_tex(self) -> Texture | None:
        return None  # Glass isn't metallic

    @property
    def requires_uv(self) -> bool:
        return any(
            t is not None and t.requires_uv
            for t in (
                self.specular_texture,
                self.transmission_texture,
                self.thickness_texture,
                self.roughness_texture,
                self.normal_texture,
                self.emissive_texture,
            )
        )

    def update_texture(
        self,
        *,
        roughness_texture: Texture | None = None,
        normal_texture: Texture | None = None,
        emissive_texture: Texture | None = None,
        **kwargs,
    ) -> None:
        self._extract_opacity_from(kwargs.get("color_texture"), emissive_texture, None)
        super().update_texture(**kwargs)
        self.roughness_texture = self._default_if_none(self.roughness_texture, roughness_texture)
        self.normal_texture = self._default_if_none(self.normal_texture, normal_texture)
        self.emissive_texture = self._default_if_none(self.emissive_texture, emissive_texture)

    def finalize_texture(self) -> None:
        super().finalize_texture()
        self.roughness_texture = self._default_if_none(
            self.roughness_texture, ColorTexture(color=(self.default_roughness,))
        )


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

    roughness: UnitInterval | None = Field(default=0.1, exclude=True, repr=False)
    metallic: UnitInterval | None = Field(default=1.0, exclude=True, repr=False)

    metal_type: MetalType = "iron"
    diffuse_texture: Texture | None = None
    opacity_texture: Texture | None = None
    roughness_texture: Texture | None = None
    normal_texture: Texture | None = None
    emissive_texture: Texture | None = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_shortcuts(cls, data: Any) -> Any:
        if isinstance(data, dict) and "default_color" not in data:
            metal_type = data.get("metal_type") or cls.model_fields["metal_type"].default
            if metal_type in METAL_COLOR:
                data["default_color"] = METAL_COLOR[metal_type]

        return super()._resolve_shortcuts(data)

    @property
    def texture(self) -> Texture | None:
        return self.diffuse_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.diffuse_texture = value

    @property
    def emissive_tex(self) -> Texture | None:
        return self.emissive_texture

    @property
    def opacity_tex(self) -> Texture | None:
        return self.opacity_texture

    @property
    def roughness_tex(self) -> Texture | None:
        return self.roughness_texture

    @property
    def metallic_tex(self) -> Texture | None:
        # Metal stores metallic as a scalar field; wrap as ColorTexture for the ARM channel.
        return ColorTexture(color=(self.metallic,))

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

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        super()._post_init()
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
        **kwargs,
    ) -> None:
        # Pre-truncate incoming RGBA color/emissive into RGB + a candidate opacity,
        # so the invariant holds before `_default_if_none` runs and so an explicit
        # `opacity_texture=` still wins (first-writer-wins). Parent's diffuse arrives
        # as `color_texture` in kwargs.
        opacity_texture = self._extract_opacity_from(kwargs.get("color_texture"), emissive_texture, opacity_texture)
        super().update_texture(**kwargs)
        self.opacity_texture = self._default_if_none(self.opacity_texture, opacity_texture)
        self.roughness_texture = self._default_if_none(self.roughness_texture, roughness_texture)
        self.normal_texture = self._default_if_none(self.normal_texture, normal_texture)
        self.emissive_texture = self._default_if_none(self.emissive_texture, emissive_texture)

    def finalize_texture(self) -> None:
        super().finalize_texture()
        self.opacity_texture = self._default_if_none(self.opacity_texture, ColorTexture(color=(self.default_opacity,)))
        self.roughness_texture = self._default_if_none(
            self.roughness_texture, ColorTexture(color=(self.default_roughness,))
        )


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

    ior: float | None = 1.0

    diffuse_texture: Texture | None = None
    specular_texture: Texture | None = None
    opacity_texture: Texture | None = None
    roughness_texture: Texture | None = None
    normal_texture: Texture | None = None
    emissive_texture: Texture | None = None

    @property
    def texture(self) -> Texture | None:
        return self.diffuse_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.diffuse_texture = value

    @property
    def emissive_tex(self) -> Texture | None:
        return self.emissive_texture

    @property
    def opacity_tex(self) -> Texture | None:
        return self.opacity_texture

    @property
    def roughness_tex(self) -> Texture | None:
        return self.roughness_texture

    @property
    def metallic_tex(self) -> Texture | None:
        return None  # Plastic isn't metallic

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

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        super()._post_init()
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
        **kwargs,
    ) -> None:
        opacity_texture = self._extract_opacity_from(kwargs.get("color_texture"), emissive_texture, opacity_texture)
        super().update_texture(**kwargs)
        self.opacity_texture = self._default_if_none(self.opacity_texture, opacity_texture)
        self.roughness_texture = self._default_if_none(self.roughness_texture, roughness_texture)
        self.normal_texture = self._default_if_none(self.normal_texture, normal_texture)
        self.emissive_texture = self._default_if_none(self.emissive_texture, emissive_texture)

    def finalize_texture(self) -> None:
        super().finalize_texture()
        self.opacity_texture = self._default_if_none(self.opacity_texture, ColorTexture(color=(self.default_opacity,)))
        self.roughness_texture = self._default_if_none(
            self.roughness_texture, ColorTexture(color=(self.default_roughness,))
        )


class BSDF(Surface):
    """
    Disney BSDF surface with principled shading.

    Parameters
    ----------
    color : tuple | None, optional
        Diffuse color of the surface. Shortcut for `diffuse_texture` with a single color.
    ior : float | None, optional
        Index of Refraction. Defaults to ``None`` so the GLB-baked value
        (``KHR_materials_ior``) — or, if absent, the class-level
        ``default_ior`` (1.5) — fills it in via ``update_texture``. Set
        explicitly to lock the surface to a fixed IOR (assets cannot then
        override).
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

    ior: float | None = None

    diffuse_texture: Texture | None = None
    opacity_texture: Texture | None = None
    roughness_texture: Texture | None = None
    metallic_texture: Texture | None = None
    normal_texture: Texture | None = None
    emissive_texture: Texture | None = None
    specular_trans: float = 0.0
    diffuse_trans: float = 0.0

    @property
    def texture(self) -> Texture | None:
        return self.diffuse_texture

    @texture.setter
    def texture(self, value: Texture | None) -> None:
        self.diffuse_texture = value

    @property
    def emissive_tex(self) -> Texture | None:
        return self.emissive_texture

    @property
    def opacity_tex(self) -> Texture | None:
        return self.opacity_texture

    @property
    def roughness_tex(self) -> Texture | None:
        return self.roughness_texture

    @property
    def metallic_tex(self) -> Texture | None:
        return self.metallic_texture

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

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        super()._post_init()
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
        **kwargs,
    ) -> None:
        opacity_texture = self._extract_opacity_from(kwargs.get("color_texture"), emissive_texture, opacity_texture)
        super().update_texture(**kwargs)
        self.opacity_texture = self._default_if_none(self.opacity_texture, opacity_texture)
        self.roughness_texture = self._default_if_none(self.roughness_texture, roughness_texture)
        self.metallic_texture = self._default_if_none(self.metallic_texture, metallic_texture)
        self.normal_texture = self._default_if_none(self.normal_texture, normal_texture)
        self.emissive_texture = self._default_if_none(self.emissive_texture, emissive_texture)

    def finalize_texture(self) -> None:
        super().finalize_texture()
        self.opacity_texture = self._default_if_none(self.opacity_texture, ColorTexture(color=(self.default_opacity,)))
        self.roughness_texture = self._default_if_none(
            self.roughness_texture, ColorTexture(color=(self.default_roughness,))
        )


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
    def emissive_tex(self) -> Texture | None:
        return self.emissive_texture

    @property
    def opacity_tex(self) -> Texture | None:
        return None  # Emission has no opacity_texture field

    @property
    def roughness_tex(self) -> Texture | None:
        return None  # Emission has no roughness_texture field

    @property
    def metallic_tex(self) -> Texture | None:
        return None  # Emission isn't metallic

    @property
    def requires_uv(self) -> bool:
        return self.emissive_texture is not None and self.emissive_texture.requires_uv

    @model_validator(mode="after")
    def _post_init(self) -> Self:
        super()._post_init()
        if self.emissive_texture is not None:
            self.emissive_texture.check_dim(3)
        return self

    def update_texture(self, *, emissive_texture: Texture | None = None, **kwargs) -> None:
        # Pre-truncate incoming RGBA color/emissive to RGB; Emission has no
        # `opacity_texture` field, so the extracted alpha is discarded.
        self._extract_opacity_from(kwargs.get("color_texture"), emissive_texture, None)
        super().update_texture(**kwargs)
        self.emissive_texture = self._default_if_none(self.emissive_texture, emissive_texture)


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

    roughness: UnitInterval | None = Field(default=1.0, exclude=True, repr=False)
    ior: float | None = 1.5


class Smooth(Plastic):
    """
    Shortcut for a smooth plastic surface.
    """

    roughness: UnitInterval | None = Field(default=0.1, exclude=True, repr=False)
    ior: float | None = 1.5


class Reflective(Plastic):
    """
    Shortcut for a reflective (smoother than `Smooth`) plastic surface.
    """

    roughness: UnitInterval | None = Field(default=0.01, exclude=True, repr=False)
    ior: float | None = 2.0


class Collision(Plastic):
    """
    Default surface type for collision geometry.

    Each instance gets a freshly randomized color at construction (when the user does
    not supply an explicit ``color=`` kwarg), making convex-decomposition pieces
    visually distinguishable. Pass ``color=...`` to override.
    """

    color: FArrayType | None = Field(default=(0.5, 0.5, 0.5), exclude=True, repr=False)


class Water(Glass):
    """
    Shortcut for a water surface (using Glass surface with proper values).
    """

    color: FArrayType | None = Field(default=(0.61, 0.98, 0.93), exclude=True, repr=False)
    roughness: UnitInterval | None = Field(default=0.2, exclude=True, repr=False)
    ior: float | None = 1.2


class Iron(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'iron'`.
    """


class Aluminium(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'aluminium'`.
    """

    metal_type: MetalType = "aluminium"


class Copper(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'copper'`.
    """

    metal_type: MetalType = "copper"


class Gold(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'gold'`.
    """

    metal_type: MetalType = "gold"


class Brass(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'brass'`.
    """

    metal_type: MetalType = "brass"


class Titanium(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'titanium'`.
    """

    metal_type: MetalType = "titanium"


class Vanadium(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'vanadium'`.
    """

    metal_type: MetalType = "vanadium"


class Lithium(Metal):
    """
    Shortcut for a metallic surface with `metal_type = 'lithium'`.
    """

    metal_type: MetalType = "lithium"
