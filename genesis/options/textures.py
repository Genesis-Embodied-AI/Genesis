import os
from typing import Optional, List, Union

import Imath
import numpy as np
import OpenEXR
from PIL import Image

import genesis as gs

from .options import Options


class Texture(Options):
    """
    Base class for Genesis's texture objects.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    def __init__(self, **data):
        super().__init__(**data)

    def check_dim(self, dim):
        raise NotImplementedError

    def check_simplify(self):
        raise NotImplementedError

    def apply_cutoff(self, cutoff):
        raise NotImplementedError


class ColorTexture(Texture):
    """
    A texture that consists of a single color.

    Parameters
    ----------
    color : list of float
        A list of color values, stored as tuple, supporting any number of channels within the range [0.0, 1.0].
        Default is (1.0, 1.0, 1.0).
    """

    color: Union[float, List[float]] = (1.0, 1.0, 1.0)

    def __init__(self, **data):
        super().__init__(**data)
        if isinstance(self.color, float):
            self.color = (self.color,)
        else:
            self.color = tuple(self.color)  # Use tuple to store image color since it is more efficient

    def check_dim(self, dim):
        if len(self.color) > dim:
            self.color, res = self.color[:dim], self.color[dim]
            return ColorTexture(color=res)
        return None

    def check_simplify(self):
        return self

    def apply_cutoff(self, cutoff):
        if cutoff is None:
            return
        self.color = tuple(1.0 if c >= cutoff else 0.0 for c in self.color)


class ImageTexture(Texture):
    """
    A texture with a texture map (image).

    Parameters
    ----------
    image_path : str, optional
        Path to the image file.
    image_array : np.ndarray, optional
        Image array.
    image_color : float or list of float, optional
        The factor that will be multiplied with the base color, stored as tuple. Default is None.
    encoding : str, optional
        The encoding way of the image. Possible values are ['srgb', 'linear']. Default is 'srgb'.

        - 'srgb': Encoding of some RGB images.
        - 'linear': All generic images, such as opacity, roughness and normal, should be encoded with 'linear'.
    """

    image_path: Optional[str] = None
    image_array: Optional[np.ndarray] = None
    image_color: Optional[Union[float, List[float]]] = None
    encoding: str = "srgb"

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)

        if not (self.image_path is None) ^ (self.image_array is None):
            gs.raise_exception("Please set either `image_path` or `image_array`.")

        if self.image_path is not None:
            input_image_path = self.image_path
            if not os.path.exists(self.image_path):
                self.image_path = os.path.join(gs.utils.get_assets_dir(), self.image_path)

            if not os.path.exists(self.image_path):
                gs.raise_exception(
                    f"File not found in either current directory or assets directory: '{input_image_path}'."
                )

            # Load image_path as actual image_array, unless for special texture images (e.g. `.hdr` and `.exr`) that are only supported by raytracers
            if self.image_path.endswith((".hdr", ".exr")):
                self.encoding = "linear"  # .exr or .hdr images should be encoded with 'linear'
                if self.image_path.endswith((".exr")):
                    exr_file = OpenEXR.InputFile(self.image_path)
                    exr_header = exr_file.header()

                    if exr_header["compression"].v > Imath.Compression.PIZ_COMPRESSION:
                        new_image_path = f"{self.image_path[:-4]}_ZIP.exr"
                        gs.logger.warning(
                            f"EXR image {self.image_path}'s compression type {exr_header['compression']} is not supported. "
                            f"Converting to compression type ZIP_COMPRESSION and saving to {new_image_path}."
                        )
                        self.image_path = new_image_path

                        if not os.path.exists(new_image_path):
                            channel_data = {channel: exr_file.channel(channel) for channel in exr_header["channels"]}
                            exr_header["compression"] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
                            new_exr_file = OpenEXR.OutputFile(new_image_path, exr_header)
                            new_exr_file.writePixels(channel_data)
                            new_exr_file.close()

                    exr_file.close()
            else:
                self.image_array = np.array(Image.open(self.image_path))
                self.image_path = None

        elif self.image_array is not None:
            if not isinstance(self.image_array, np.ndarray):
                gs.raise_exception("`image_array` needs to be a numpy array.")
            arr = self.image_array
            if arr.dtype != np.uint8:
                if np.issubdtype(arr.dtype, np.floating):
                    if arr.max() <= 1.0:
                        arr = (arr * 255.0).round()
                    arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
                elif arr.dtype == np.bool_:
                    arr = arr.astype(np.uint8) * 255
                elif np.issubdtype(arr.dtype, np.integer):
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                else:
                    gs.raise_exception(
                        f"Unsupported image dtype {arr.dtype}. "
                        "Only uint8, integer, floating-point, or bool types are supported."
                    )
            self.image_array = arr

        # calculate channel
        if self.image_array is None:
            if isinstance(self.resolution, (tuple, list)):
                H, W = self.resolution
            else:
                H = W = self.resolution

            # Default to 3-channel RGB
            white = np.array([255, 255, 255], dtype=np.uint8)
            self.image_array = np.full((H, W, 3), white, dtype=np.uint8)
            self._mean_color = np.array([1.0, 1.0, 1.0], dtype=np.float16)
            self._channel = 3
        else:
            self._mean_color = (np.mean(self.image_array, axis=(0, 1), dtype=np.float32) / 255.0).astype(np.float16)
            self._channel = self.image_array.shape[2] if self.image_array.ndim == 3 else 1

        # build image color
        if self.image_color is None:
            self.image_color = (1.0,) * self._channel
        else:
            if isinstance(self.image_color, float):
                self.image_color = (self.image_color,) * self._channel
            else:
                self.image_color = tuple(self.image_color[: self._channel])

        self.encoding = self.encoding.lower()
        if self.encoding not in ["srgb", "linear"]:
            gs.raise_exception(f"Invalid image encoding: {self.encoding}.")

    def check_dim(self, dim):
        if self.image_array is not None:
            if self._channel > dim:
                self.image_array, res_array = self.image_array[:, :, :dim], self.image_array[:, :, dim]
                self.image_color, res_color = self.image_color[:dim], self.image_color[dim:]
                self._channel = dim
                return ImageTexture(image_array=res_array, image_color=res_color, encoding="linear").check_simplify()
        return None

    def check_simplify(self):
        if self.image_array is None:
            return self
        max_color = np.max(self.image_array, axis=(0, 1))
        min_color = np.min(self.image_array, axis=(0, 1))
        if np.all(min_color == max_color):
            return ColorTexture(color=max_color.reshape(-1) / 255.0 * self.image_color)
        else:
            return self

    def mean_color(self):
        return self._mean_color

    def channel(self):
        return self._channel

    def apply_cutoff(self, cutoff):
        if cutoff is None or self.image_array is None:  # Cutoff does not apply on image file.
            return
        self.image_array = np.where(self.image_array >= 255.0 * cutoff, 255, 0).astype(np.uint8)
