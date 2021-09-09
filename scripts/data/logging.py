import wandb
from data.config import TrainingConfig

def init_wandb(name: str, train_cfg: TrainingConfig):
    wandb.init(
        project='birdseye-semseg',
        entity='ais-birdseye',
        group=train_cfg.group,
        name=name,
        dir=train_cfg.log_dir,
        config={
            'model_name': train_cfg.model_name,
            'segmentation_loss': train_cfg.loss_function,
            'epochs': train_cfg.epochs,
            'learning_rate': train_cfg.learning_rate,
            'resume': train_cfg.resume_training,
            'mask_det_threshold': train_cfg.mask_detection_thresh,
            'initial_difficulty': train_cfg.initial_difficulty,
            'maximum_difficulty': train_cfg.maximum_difficulty,
            'strategy': train_cfg.strategy,
            'strategy_parameter': train_cfg.strategy_parameter,
            'torch_seed': train_cfg.torch_seed,
            'shuffle_data': train_cfg.shuffle_data
    })