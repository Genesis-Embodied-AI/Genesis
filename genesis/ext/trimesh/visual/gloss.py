import numpy as np

from ..constants import log
from ..exceptions import ExceptionWrapper

try:
    from PIL.Image import Image, fromarray
except BaseException as E:
    Image = ExceptionWrapper(E)
    fromarray = ExceptionWrapper(E)


def specular_to_pbr(
    specularFactor=None,
    glossinessFactor=None,
    specularGlossinessTexture=None,
    diffuseTexture=None,
    diffuseFactor=None,
    **kwargs,
):
    """
    Convert the KHR_materials_pbrSpecularGlossiness to a
    metallicRoughness visual.

    Parameters
    -----------
    specularFactor : list[float]
        Specular color values. Ignored if specularGlossinessTexture
        is present and defaults to [1.0, 1.0, 1.0].
    glossinessFactor : float
        glossiness factor in range [0, 1], Ignored if
        specularGlossinessTexture is present and defaults to 1.0.
    specularGlossinessTexture : PIL.Image
        Texture with 4 color channels. With [0,1,2] representing
        specular RGB and 3 glossiness.
    diffuseTexture : PIL.Image
        Texture with 4 color channels. With [0,1,2] representing diffuse
        RGB and 3 opacity.
    diffuseFactor: float
        Diffuse RGBA color. Ignored if diffuseTexture is present
        Defaults to [1.0, 1.0, 1.0, 1.0].

    Returns
    ----------
    kwargs : dict
      Constructor args for a PBRMaterial object.
      Containing:
        - either baseColorTexture or baseColorFactor
        - either metallicRoughnessTexture or metallicFactor and roughnessFactor
    """
    # based on:
    # https://github.com/KhronosGroup/glTF/blob/89427b26fcac884385a2e6d5803d917ab5d1b04f/extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness/examples/convert-between-workflows-bjs/js/babylon.pbrUtilities.js#L33-L64

    if isinstance(Image, ExceptionWrapper):
        log.debug("unable to convert specular-glossy material without pillow!")
        result = {}
        if isinstance(diffuseTexture, dict):
            result["baseColorTexture"] = diffuseTexture
        if diffuseFactor is not None:
            result["baseColorFactor"] = diffuseFactor
        return result

    dielectric_specular = np.array([0.04, 0.04, 0.04], dtype=np.float32)
    epsilon = 1e-6

    def solve_metallic(diffuse, specular, one_minus_specular_strength):
        if isinstance(specular, float) and specular < dielectric_specular[0]:
            return 0.0

        if len(diffuse.shape) == 2:
            diffuse = diffuse[..., None]
        if len(specular.shape) == 2:
            specular = specular[..., None]

        a = dielectric_specular[0]
        b = (
            diffuse * one_minus_specular_strength / (1.0 - dielectric_specular[0])
            + specular
            - 2.0 * dielectric_specular[0]
        )
        c = dielectric_specular[0] - specular
        D = b * b - 4.0 * a * c
        D = np.clip(D, epsilon, None)
        metallic = np.clip((-b + np.sqrt(D)) / (2.0 * a), 0.0, 1.0)
        if isinstance(metallic, np.ndarray):
            metallic[specular < dielectric_specular[0]] = 0.0
        return metallic

    def get_perceived_brightness(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

    def toPIL(img):
        if isinstance(img, Image):
            return img
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        return fromarray(img)

    def get_float(val):
        if isinstance(val, float):
            return val
        if isinstance(val, np.ndarray) and len(val.shape) == 1:
            return val[0]
        return val.tolist()

    def get_diffuse(diffuseFactor, diffuseTexture):
        diffuseFactor = diffuseFactor if diffuseFactor is not None else [1.0, 1.0, 1.0, 1.0]
        diffuseFactor = np.array(diffuseFactor, dtype=np.float32)

        if diffuseTexture is not None:
            diffuse = np.array(diffuseTexture) / 255.0
            if len(diffuse.shape) == 2:
                diffuse = diffuse[..., None]
            if diffuse.shape[-1] == 1:
                diffuse = diffuse * diffuseFactor
            elif diffuse.shape[-1] == 2:
                alpha = diffuse[..., 1:2]
                diffuse = diffuse[..., :1] * diffuseFactor
                diffuse[..., -1:] *= alpha
            elif diffuse.shape[-1] == diffuseFactor.shape[-1]:
                diffuse = diffuse * diffuseFactor
            elif diffuse.shape[-1] == 3 and diffuseFactor.shape[-1] == 4:
                diffuse = np.concatenate([diffuse, np.ones_like(diffuse[..., :1])], axis=-1) * diffuseFactor
            else:
                log.warning(
                    "`diffuseFactor` and `diffuseTexture` have incompatible shapes: "
                    + "{0} and {1}".format(diffuseFactor.shape, diffuse.shape)
                )
        else:
            diffuse = diffuseFactor if diffuseFactor is not None else [1, 1, 1, 1]
            diffuse = np.array(diffuse, dtype=np.float32)
        return diffuse

    def get_specular_glossiness(specularFactor, glossinessFactor, specularGlossinessTexture):
        if specularFactor is None:
            specularFactor = [1.0, 1.0, 1.0]
        specularFactor = np.array(specularFactor, dtype=np.float32)
        if glossinessFactor is None:
            glossinessFactor = 1.0
        glossinessFactor = np.array([glossinessFactor], dtype=np.float32)

        # specularGlossinessTexture should be a texture with 4 channels,
        # 3 sRGB channels for specular and 1 linear channel for glossiness.
        # in practice, it can also have just 1, 2, or 3 channels which are then to
        # be multiplied with the provided factors

        if specularGlossinessTexture is not None:
            specularGlossinessTexture = np.array(specularGlossinessTexture)
            specularTexture, glossinessTexture = None, None

            if (len(specularGlossinessTexture.shape) == 2 or specularGlossinessTexture.shape[-1]) == 1:
                # use the one channel as a multiplier for specular and glossiness
                specularTexture = glossinessTexture = specularGlossinessTexture.reshape((-1, -1, 1))
            elif specularGlossinessTexture.shape[-1] == 3:
                # all channels are specular, glossiness is only a factor
                specularTexture = specularGlossinessTexture[..., :3]
            elif specularGlossinessTexture.shape[-1] == 2:
                # first channel is specular, last channel is glossiness
                specularTexture = specularGlossinessTexture[..., :1]
                glossinessTexture = specularGlossinessTexture[..., 1:2]
            elif specularGlossinessTexture.shape[-1] == 4:
                # first 3 channels are specular, last channel is glossiness
                specularTexture = specularGlossinessTexture[..., :3]
                glossinessTexture = specularGlossinessTexture[..., 3:]

            if specularTexture is not None:
                # convert into [0,1] range. Does this require conversion of sRGB values?
                specular = specularTexture / 255.0
                specular = specular * specularFactor
            else:
                specular = specularFactor

            if glossinessTexture is not None:
                # convert into [0,1] range. Does this require conversion of sRGB values?
                glossiness = glossinessTexture / 255.0
                glossiness = glossiness * glossinessFactor
            else:
                glossiness = glossinessFactor

            one_minus_specular_strength = 1.0 - np.max(specular, axis=-1, keepdims=True)
        else:
            specular = specularFactor if specularFactor is not None else [1.0, 1.0, 1.0]
            specular = np.array(specular, dtype=np.float32)
            glossiness = glossinessFactor if glossinessFactor is not None else 1.0
            glossiness = np.array(glossiness, dtype=np.float32)
            one_minus_specular_strength = 1.0 - max(specular[:3])

        return specular, glossiness, one_minus_specular_strength

    if diffuseTexture is not None and specularGlossinessTexture is not None:
        # reshape to the size of the largest texture
        max_shape = [max(diffuseTexture.size[i], specularGlossinessTexture.size[i]) for i in range(2)]
        if diffuseTexture.size[0] != max_shape[0] or diffuseTexture.size[1] != max_shape[1]:
            diffuseTexture = diffuseTexture.resize(max_shape)
        if specularGlossinessTexture.size[0] != max_shape[0] or specularGlossinessTexture.size[1] != max_shape[1]:
            specularGlossinessTexture = specularGlossinessTexture.resize(max_shape)

    diffuse = get_diffuse(diffuseFactor, diffuseTexture)
    specular, glossiness, one_minus_specular_strength = get_specular_glossiness(
        specularFactor, glossinessFactor, specularGlossinessTexture
    )

    metallic = solve_metallic(
        get_perceived_brightness(diffuse), get_perceived_brightness(specular), one_minus_specular_strength
    )
    if not isinstance(metallic, np.ndarray):
        metallic = np.array(metallic, dtype=np.float32)

    diffuse_rgb = diffuse[..., :3]
    opacity = diffuse[..., -1] if diffuse.shape[-1] == 4 else None
    base_color_from_diffuse = diffuse_rgb * (
        one_minus_specular_strength / (1.0 - dielectric_specular[0]) / np.clip((1.0 - metallic), epsilon, None)
    )
    base_color_from_specular = (specular - dielectric_specular * (1.0 - metallic)) * (
        1.0 / np.clip(metallic, epsilon, None)
    )
    mm = metallic * metallic
    base_color = mm * base_color_from_specular + (1.0 - mm) * base_color_from_diffuse
    base_color = np.clip(base_color, 0.0, 1.0)

    if opacity is not None and np.any(opacity < 1.0):
        base_color = np.concatenate([base_color, opacity[..., None]], axis=-1)

    result = {}
    if len(base_color.shape) > 1:
        result["baseColorTexture"] = toPIL(base_color)
    else:
        result["baseColorFactor"] = base_color.tolist()

    if len(metallic.shape) > 1 or len(glossiness.shape) > 1:
        if len(glossiness.shape) == 1:
            glossiness = np.tile(glossiness, (metallic.shape[0], metallic.shape[1], 1))
        if len(metallic.shape) == 1:
            metallic = np.tile(metallic, (glossiness.shape[0], glossiness.shape[1], 1))

        result["metallicRoughnessTexture"] = toPIL(np.concatenate([metallic, 1.0 - glossiness], axis=-1))
    else:
        result["metallicFactor"] = get_float(metallic)
        result["roughnessFactor"] = get_float(1.0 - glossiness)

    return result
