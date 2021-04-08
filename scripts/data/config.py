import yaml

class SemanticCloudConfig:
    def __init__(self, file_path: str):
        yaml_file = open(file_path)
        conf = yaml.load(yaml_file, Loader=yaml.FullLoader)
        yaml_file.close()
        self.cloud_max_x = float(conf['cloud']['max_point_x'])
        self.cloud_min_x = float(conf['cloud']['min_point_x'])
        self.cloud_max_y = float(conf['cloud']['max_point_y'])
        self.cloud_min_y = float(conf['cloud']['min_point_y'])
        self.cloud_x_span = self.cloud_max_x - self.cloud_min_x
        self.cloud_y_span = self.cloud_max_y - self.cloud_min_y
        self.image_rows = int(conf['bev']['image_rows'])
        self.image_cols = int(conf['bev']['image_cols'])

    def pix_per_m(self, rows=0, cols=0):
        """
        returns pixels per meter given the image size
        """
        if rows == 0 and cols == 0:
            return  ((self.image_rows / self.cloud_x_span) + (self.image_cols / self.cloud_y_span)) / 2.0    
        return  ((rows / self.cloud_x_span) + (cols / self.cloud_y_span)) / 2.0

    def center_x(self, cols=0):
        """
        returns center of the car in the image coordinates system
        """
        if cols == 0:
            return int((self.cloud_max_y / self.cloud_y_span) * self.image_cols)    
        return int((self.cloud_max_y / self.cloud_y_span) * cols)
    
    def center_y(self, rows=0):
        """
        returns center of the car in the image coordinates system
        """
        if rows == 0:
            return int((self.cloud_max_x / self.cloud_x_span) * self.image_rows)
        return int((self.cloud_max_x / self.cloud_x_span) * rows)

class TrainingConfig:
    def __init__(self, file_path: str):
        yaml_file = open(file_path)
        conf = yaml.load(yaml_file, Loader=yaml.FullLoader)
        yaml_file.close()
        # dataset config
        self.dset_dir = conf['dataset']['dataset-dir']
        self.dset_file = conf['dataset']['dataset-file']
        self.dset_name = conf['dataset']['dataset-name']
        # logging config
        self.training_name = conf['logging']['name']
        self.log_dir = conf['logging']['log-dir']
        self.log_every = conf['logging']['log-every']
        self.snapshot_dir = conf['logging']['snapshot-dir']
        # training config
        self.device = conf['training']['device']
        self.world_size = int(conf['training']['world-size'])
        self.torch_seed = int(conf['training']['torch-seed'])
        self.distributed = conf['training']['distributed']
        # dataloader config
        self.loader_workers = int(conf['dataloader']['dataloder-workers'])
        self.pin_memory = bool(conf['dataloader']['pin-memory'])
        self.shuffle_data = bool(conf['dataloader']['shuffle-data'])
        # network
        self.model_name = conf['network']['model-name']
        self.batchnorm_keep_stats = conf['network']['batchnorm-keep-stats']
        # curriculum config
        self.initial_difficulty = int(conf['curriculum']['initial-difficulty'])
        self.maximum_difficulty = int(conf['curriculum']['maximum-difficulty'])
        self.max_agent_count = int(conf['curriculum']['maximum-agent-count'])
        self.strategy = conf['curriculum']['strategy']
        self.strategy_parameter = conf['curriculum']['strategy-parameter']
        # hyperparameters
        self.drop_prob = float(conf['hyperparameters']['drop-prob'])
        self.output_h = int(conf['hyperparameters']['output-h'])
        self.output_w = int(conf['hyperparameters']['output-w'])
        self.classes = conf['hyperparameters']['classes']
        self.num_classes = int(conf['hyperparameters']['num-classes'])
        self.learning_rate = float(conf['hyperparameters']['learning-rate'])
        self.epochs = int(conf['hyperparameters']['epochs'])
        self.loss_function = conf['hyperparameters']['loss']
        # validation parameters
        self.mask_detection_thresh = float(conf['validation']['mask-det-threshold'])
        # evaluation parameters
        self.eval_dataset = conf['evaluation']['dataset']
        self.eval_run = conf['evaluation']['run']
        self.eval_model_version = conf['evaluation']['model-version']
        self.eval_random_samples = conf['evaluation']['random-samples']
        self.eval_plot = conf['evaluation']['plot']
        self.eval_difficulty = conf['evaluation']['difficulty']
        self.eval_plot_count = conf['evaluation']['count']
        self.eval_plot_dir = conf['evaluation']['plot-dir']
        self.eval_plot_tag = conf['evaluation']['plot-tag']
        self.eval_model_name = conf['evaluation']['model-name']
        self.eval_batchnorm_keep_stats = conf['evaluation']['batchnorm-keep-stats']