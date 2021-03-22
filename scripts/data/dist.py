import torch
import torch.distributed as dist

def sync_tensor(rank: int, tag: int, tensor: torch.Tensor, world_size=1) -> torch.Tensor:
    if rank != 0:
        dist.send(tensor, dst=0, tag=tag)
        return tensor
    # receive tensor from other ranks & log
    elif world_size > 1:
        new_tensor = torch.zeros(1)
        dist.recv(new_tensor, src=1, tag=tag)
        return tensor + new_tensor
    else:
        return tensor