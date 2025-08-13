import ctypes

import torch

import genesis as gs


class TensorRingBuffer:
    def __init__(
        self,
        N: int,
        shape: tuple[int, ...],
        dtype=torch.float32,
        buffer: torch.Tensor | None = None,
        idx_ptr: int | ctypes.c_int = 0,
    ):
        if buffer is None:
            self.buffer = torch.empty((N, *shape), dtype=dtype, device=gs.device)
        else:
            assert buffer.shape == (N, *shape)
            self.buffer = buffer
        self.N = N
        if isinstance(idx_ptr, int):
            self._idx_ptr = ctypes.c_int(idx_ptr)
        else:
            self._idx_ptr = idx_ptr

    def append(self, tensor: torch.Tensor):
        self.buffer[self._idx_ptr.value].copy_(tensor)
        self._idx_ptr.value = (self._idx_ptr.value + 1) % self.N

    def get(self, idx: int, clone: bool = True):
        """
        Parameters
        ----------
        idx : int
            Index of the element to get, where 0 is the latest element, 1 is the second latest, etc.
        clone : bool
            Whether to clone the tensor.
        """
        tensor = self.buffer[(self._idx_ptr.value - idx) % self.N]
        return tensor.clone() if clone else tensor

    def clone(self):
        return TensorRingBuffer(
            self.N,
            self.buffer.shape[1:],
            dtype=self.buffer.dtype,
            buffer=self.buffer.clone(),
            idx_ptr=self._idx_ptr,
        )

    def __getitem__(self, key: int | slice | tuple):
        """
        Enable slicing of the tensor ring buffer.

        Parameters
        ----------
        key : int | slice | tuple
            Slice object (e.g., 3:6) or integer index or tuple of indices

        Returns
        -------
        TensorRingBuffer
            A new ring buffer containing a view of the sliced data
        """
        if isinstance(key, int):
            sliced_buffer = self.buffer[:, key : key + 1]
        elif isinstance(key, slice):
            sliced_buffer = self.buffer[:, key]
        elif isinstance(key, tuple):
            indexes = (slice(None),) + key
            sliced_buffer = self.buffer[indexes]
        else:
            raise TypeError(f"Unsupported key type: {type(key)}")

        return TensorRingBuffer(
            self.N,
            sliced_buffer.shape[1:],
            dtype=sliced_buffer.dtype,
            buffer=sliced_buffer,
            idx_ptr=self._idx_ptr,
        )
