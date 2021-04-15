import wandb
from tensorboardX import SummaryWriter

from data.config import TrainingConfig

def log_wandb(rank: int, data: dict):
    if rank != 0:
        return
    wandb.log(data)

def log_scalar_tb(rank: int, scalar, key: 'str', idx: int, writer: SummaryWriter):
    if rank != 0:
        return
    writer.add_scalar(key, scalar, idx)

def log_image_tb(rank: int, image, key:str, idx: int, writer: SummaryWriter):
    if rank != 0:
        return
    writer.add_image(key, image, idx)

def log_string(rank: int, message: str, end: str = '\n'):
    if rank == 0:
        print(message, end=end)


def init_wandb(name: str, train_cfg: TrainingConfig):
    wandb.init(
        project='birdseye-semseg',
        entity='ais-birdseye',
        group=train_cfg.group,
        name=name,
        resume=train_cfg.resume_training,
        dir=train_cfg.log_dir,
        config={
            'model_name': train_cfg.model_name,
            'segmentation_loss': train_cfg.loss_function,
            'epochs': train_cfg.epochs,
            'learning_rate': train_cfg.learning_rate,
            'mask_det_threshold': train_cfg.mask_detection_thresh,
            'agent_drop_probability': train_cfg.drop_prob,
            'initial_difficulty': train_cfg.initial_difficulty,
            'maximum_difficulty': train_cfg.maximum_difficulty,
            'strategy': train_cfg.strategy,
            'strategy_parameter': train_cfg.strategy_parameter,
            'torch_seed': train_cfg.torch_seed,
            'shuffle_data': train_cfg.shuffle_data
    })