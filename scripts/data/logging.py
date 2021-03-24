from tensorboardX import SummaryWriter
import wandb

def log_scalar_wandb(rank: int, data: dict):
    if rank != 0:
        return
    wandb.log(data)

def log_scalar_tb(rank: int, scalar, key: 'str', idx: int, writer: SummaryWriter):
    if rank != 0:
        return
    writer.add_scalar(key, scalar, idx)

def log_image_wandb(rank: int, data: dict):
    if rank != 0:
        return
    wandb.log(data)

def log_image_tb(rank: int, image, key:str, idx: int, writer: SummaryWriter):
    if rank != 0:
        return
    writer.add_image(key, image, idx)

def log_string(rank: int, message: str, end: str = '\n'):
    if rank == 0:
        print(message, end=end)