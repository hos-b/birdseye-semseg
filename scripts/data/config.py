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
        # logging config
        self.training_name = str(conf['logging']['name'])
        self.group = str(conf['logging']['group'])
        self.log_dir = str(conf['logging']['log-dir'])
        self.log_every = int(conf['logging']['log-every'])
        self.snapshot_dir = str(conf['logging']['snapshot-dir'])
        # training config
        self.device = str(conf['training']['device'])
        self.torch_seed = int(conf['training']['torch-seed'])
        self.loss_function = str(conf['training']['loss'])
        self.weight_losses = bool(conf['training']['weight-losses'])
        # network
        self.model_name = str(conf['network']['model-name'])
        self.aggregation_type = str(conf['network']['aggregation-type'])
        # resume
        self.resume_tag = str(conf['resume']['tag'])
        self.resume_training = bool(conf['resume']['flag'])
        self.resume_model_version = str(conf['resume']['model-version'])
        self.resume_starting_epoch = int(conf['resume']['starting-epoch'])
        self.resume_difficulty = int(conf['resume']['difficulty'])
        self.resume_decoder_only = bool(conf['resume']['decoder-only'])
        self.resume_optimizer_state = bool(conf['resume']['resume-optimizer-state'])
        # curriculum config
        self.curriculum_activate = bool(conf['curriculum']['activate'])
        self.initial_difficulty = int(conf['curriculum']['initial-difficulty'])
        self.maximum_difficulty = int(conf['curriculum']['maximum-difficulty'])
        self.max_agent_count = int(conf['curriculum']['maximum-agent-count'])
        self.strategy = str(conf['curriculum']['strategy'])
        self.strategy_parameter = conf['curriculum']['strategy-parameter']
        # hyperparameters
        self.drop_prob = float(conf['hyperparameters']['drop-prob'])
        self.learning_rate = float(conf['hyperparameters']['learning-rate'])
        self.epochs = int(conf['hyperparameters']['epochs'])
        self.color_jitter = list(conf['hyperparameters']['color-jitter'])
        # validation parameters
        self.mask_detection_thresh = float(conf['validation']['mask-det-threshold'])
        # dataloader config
        self.loader_workers = int(conf['dataloader']['dataloder-workers'])
        self.shuffle_data = bool(conf['dataloader']['shuffle-data'])
        # dataset config
        self.output_h = int(conf['dataset']['output-h'])
        self.output_w = int(conf['dataset']['output-w'])
        self.classes = str(conf['dataset']['classes'])
        self.num_classes = int(conf['dataset']['num-classes'])
        self.dset_dir = str(conf['dataset']['dataset-dir'])
        self.trainset_file = str(conf['dataset']['trainset-file'])
        self.validset_file = str(conf['dataset']['validset-file'])
        self.trainset_name = str(conf['dataset']['trainset-name'])
        self.validset_name = str(conf['dataset']['validset-name'])

class EvaluationConfig:
    def __init__(self, file_path: str):
        yaml_file = open(file_path)
        conf = yaml.load(yaml_file, Loader=yaml.FullLoader)
        yaml_file.close()
        # evaluation parameters
        self.device = str(conf['device'])
        self.run = str(conf['run'])
        self.torch_seed = int(conf['torch-seed'])
        self.snapshot_dir = str(conf['snapshot-dir'])
        # model parameters
        self.model_name = str(conf['model']['model-name'])
        self.model_version = str(conf['model']['model-version'])
        self.aggregation_type = str(conf['model']['aggregation-type'])
        # plotting parameters
        self.plot_count = int(conf['plot']['count'])
        self.plot_type = str(conf['plot']['plot-type'])
        self.plot_dir = str(conf['plot']['plot-dir'])
        self.plot_tag = str(conf['plot']['plot-tag'])
        # dataset parameters
        self.random_samples = bool(conf['dataset']['random-samples'])
        self.dset_dir = str(conf['dataset']['dataset-dir'])
        self.dset_file = str(conf['dataset']['dataset-file'])
        self.dset_name = str(conf['dataset']['dataset-name'])
        self.output_h = int(conf['dataset']['output-h'])
        self.output_w = int(conf['dataset']['output-w'])
        self.classes = str(conf['dataset']['classes'])
        self.num_classes = int(conf['dataset']['num-classes'])
        # curriculum parameters
        self.difficulty = int(conf['curriculum']['difficulty'])
        self.max_agent_count = int(conf['curriculum']['maximum-agent-count'])
