import torch

import genesis as gs


class TensorRingBuffer:
    def __init__(
        self, N: int, shape: tuple[int, ...], dtype=torch.float32, buffer: torch.Tensor | None = None, idx_ptr: int = 0
    ):
        if buffer is None:
            self.buffer = torch.empty((N, *shape), dtype=dtype, device=gs.device)
        else:
            assert buffer.shape == (N, *shape)
            self.buffer = buffer
        self.N = N
        self._idx_ptr = idx_ptr  # idx_ptr points to the next free slot

    def append(self, tensor: torch.Tensor):
        self.buffer[self._idx_ptr].copy_(tensor)
        self._idx_ptr = (self._idx_ptr + 1) % self.N

    def get(self, idx: int):
        """
        Parameters
        ----------
        idx : int
            Index of the element to get, where 0 is the latest element, 1 is the second latest, etc.
        """
        return self.buffer[(self._idx_ptr - idx) % self.N]

    def clone(self):
        return TensorRingBuffer(
            self.N, self.buffer.shape[1:], dtype=self.buffer.dtype, buffer=self.buffer.clone(), idx_ptr=self._idx_ptr
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
            self.N, sliced_buffer.shape[1:], dtype=sliced_buffer.dtype, buffer=sliced_buffer, idx_ptr=self._idx_ptr
        )
