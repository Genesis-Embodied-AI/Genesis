import genesis as gs
import torch


class TensorRingBuffer:
    def __init__(self, N, shape, dtype=torch.float32):
        self.buffer = torch.empty((N, *shape), dtype=dtype, device=gs.device)
        self.N = N
        self._idx_ptr = 0  # idx_ptr points to the next free slot

    def append(self, tensor):
        self.buffer[self._idx_ptr].copy_(tensor)
        self._idx_ptr = (self._idx_ptr + 1) % self.N

    def get(self, idx):
        """
        Parameters
        ----------
        idx : int
            Index of the element to get, where 0 is the latest element, 1 is the second latest, etc.
        """
        return self.buffer[(self._idx_ptr - idx) % self.N]
