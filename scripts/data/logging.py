from tensorboardX import SummaryWriter
import wandb

def log_scalar(rank: int, scalar, key: 'str', idx: int, writer: SummaryWriter, logger: str):
    if rank != 0:
        return
    if logger == 'tensorboard':
        writer.add_scalar(key, scalar, idx)
    else:
        wandb.log({key.split('/')[-1]: scalar}, step=idx)

def log_string(rank: int, message: str, end: str = '\n'):
    if rank == 0:
        print(message, end=end)

def log_image(image, key:str,  idx: int, writer: SummaryWriter, logger: str):
    if logger == 'tensorboard':
        writer.add_image(key, image, idx)
    else:
        wandb.log({'images': [wandb.Image(image, caption=key.split('/')[-1])]}, step=idx)
