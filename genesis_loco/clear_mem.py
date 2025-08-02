import gc 
import torch

gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

