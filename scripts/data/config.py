import yaml

num_classes_dict = {
    'ours': 7,
    'carla': 23,
    'diminished': 3
}

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
        # noise parameters
        self.se3_noise_enable = conf['se3-noise']['enable']
        if self.se3_noise_enable:
            self.se3_noise_th_std = float(conf['se3-noise']['se3-noise-theta-std'])
            self.se3_noise_dx_std = float(conf['se3-noise']['se3-noise-dx-std'])
            self.se3_noise_dy_std = float(conf['se3-noise']['se3-noise-dy-std'])
        else:
            self.se3_noise_th_std = 0.0
            self.se3_noise_dx_std = 0.0
            self.se3_noise_dy_std = 0.0
        # network
        self.model_name = str(conf['network']['model-name'])
        self.aggregation_type = str(conf['network']['aggregation-type'])
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
        self.visualize_hard_batches = bool(conf['validation']['visualize-hard-batches'])
        self.hard_batches_indices = list(conf['validation']['hard-batch-indices'])
        # dataloader config
        self.loader_workers = int(conf['dataloader']['dataloder-workers'])
        self.shuffle_data = bool(conf['dataloader']['shuffle-data'])
        # dataset config
        self.output_h = int(conf['dataset']['output-h'])
        self.output_w = int(conf['dataset']['output-w'])
        self.classes = str(conf['dataset']['classes'])
        self.num_classes = num_classes_dict[self.classes]
        self.dset_dir = str(conf['dataset']['dataset-dir'])
        self.trainset_file = str(conf['dataset']['trainset-file'])
        self.validset_file = str(conf['dataset']['validset-file'])
        self.trainset_name = str(conf['dataset']['trainset-name'])
        self.validset_name = str(conf['dataset']['validset-name'])
        # resume
        self.resume_training = bool(conf['resume']['flag'])
        self.resume_tag = str(conf['resume']['tag'])
        self.resume_model_version = str(conf['resume']['model-version'])
        self.resume_starting_epoch = int(conf['resume']['starting-epoch'])
        self.resume_difficulty = int(conf['resume']['difficulty'])
        self.resume_optimizer_state = bool(conf['resume']['resume-optimizer-state'])
        self.perform_sanity_check()

    def perform_sanity_check(self):
        """
        performs a sanity check to make sure the given parameters are not
        absolutely wrong
        """
        if self.log_every < 10:
            print(f'sanity-check-warning: logging every {self.log_every} batches is suboptimal.')
        if self.device != 'cuda' and self.device != 'cpu':
            print(f'sanity-check-error: unknown device {self.device}.')
            exit()
        if self.loss_function != 'cross-entropy' and self.loss_function != 'focal':
            print(f'sanity-check-error: unkown loss function {self.loss_function}.')
            exit()
        if self.aggregation_type != 'bilinear' and self.aggregation_type != 'nearest':
            print(f'sanity-check-error: unkown aggregation type {self.aggregation_type}.')
            exit()
        if self.se3_noise_enable:
            if self.se3_noise_dx_std == 0 and self.se3_noise_dy_std == 0 and self.se3_noise_th_std == 0:
                print(f'sanity-check-error: noise std cannot be 0 if se3 noise is enabled.')
                exit()
        if self.initial_difficulty < 1 or self.initial_difficulty > self.max_agent_count:
            print(f'sanity-check-error: invalid initial difficulty {self.initial_difficulty}.')
            exit()
        if self.maximum_difficulty < 1 or self.maximum_difficulty > self.max_agent_count:
            print(f'sanity-check-error: invalid maximum difficulty {self.maximum_difficulty}.')
            exit()
        if self.maximum_difficulty < self.initial_difficulty:
            print(f'sanity-check-error: maximum difficulty cannot be smaller than initial difficulty.')
            exit()
        if self.strategy != 'metric' and self.strategy != 'every-x-epoch':
            print(f'sanity-check-error: unknown curriculum strategy {self.strategy}.')
            exit()
        if self.drop_prob < 0.0 or self.drop_prob > 1:
            print(f'sanity-check-error: connection drop probability must be between 0 and 1.')
            exit()
        if len(self.color_jitter) != 4:
            print(f'sanity-check-error: color jitter list must include 4 elements.')
            exit()
        if self.mask_detection_thresh < 0.0 or self.mask_detection_thresh > 1.0:
            print(f'sanity-check-error: mask detection threshold must be between 0 and 1.')
            exit()
        if self.classes != 'ours' and self.classes != 'carla' and self.classes != 'diminished':
            print(f'sanity-check-error: unknown segmentation classes {self.classes}.')
            exit()
        if self.resume_training and self.resume_tag == '':
            print(f'sanity-check-error: resuming a run requires a resume-tag.')
            exit()
        if self.output_h > 500 or self.output_w > 400:
            print(f'sanity-check-error: output size {self.output_h}x{self.output_w} is invalid.')
            exit()

class EvaluationConfig:
    def __init__(self, file_path: str):
        yaml_file = open(file_path)
        conf = yaml.load(yaml_file, Loader=yaml.FullLoader)
        yaml_file.close()
        # evaluation parameters
        self.device = str(conf['device'])
        self.torch_seed = int(conf['torch-seed'])
        self.snapshot_dir = str(conf['snapshot-dir'])
        # model parameters
        self.runs = list(conf['models']['runs'])
        self.model_names = list(conf['models']['model-names'])
        self.model_versions = list(conf['models']['model-versions'])
        self.aggregation_types = list(conf['models']['aggregation-types'])
        # gui baseline parameteres
        self.evaluate_at_start = bool(conf['gui']['evalutate-at-start'])
        self.baseline_run = str(conf['gui']['baseline-run'])
        self.baseline_model_name = str(conf['gui']['baseline-model-name'])
        self.baseline_model_version = str(conf['gui']['baseline-model-version'])
        # noise parameters
        self.se3_noise_enable = conf['se3-noise']['enable']
        if self.se3_noise_enable:
            self.se3_noise_th_std = float(conf['se3-noise']['se3-noise-theta-std'])
            self.se3_noise_dx_std = float(conf['se3-noise']['se3-noise-dx-std'])
            self.se3_noise_dy_std = float(conf['se3-noise']['se3-noise-dy-std'])
        else:
            self.se3_noise_th_std = 0.0
            self.se3_noise_dx_std = 0.0
            self.se3_noise_dy_std = 0.0
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
        self.num_classes = num_classes_dict[self.classes]
        # curriculum parameters
        self.difficulty = int(conf['curriculum']['difficulty'])
        self.max_agent_count = int(conf['curriculum']['maximum-agent-count'])
        self.perform_sanity_check()
    
    def perform_sanity_check(self):
        """
        performs a sanity check to make sure the given parameters are not
        absolutely wrong
        """
        if self.device != 'cuda' and self.device != 'cpu':
            print(f'sanity-check-error: unknown device {self.device}.')
            exit()
        if self.se3_noise_enable:
            if self.se3_noise_dx_std == 0 and self.se3_noise_dy_std == 0 and self.se3_noise_th_std == 0:
                print(f'sanity-check-error: noise std cannot be 0 if se3 noise is enabled.')
                exit() 
        if self.difficulty < 1 or self.difficulty > self.max_agent_count:
            print(f'sanity-check-error: invalid difficulty {self.difficulty}.')
            exit()
        if self.output_h > 500 or self.output_w > 400:
            print(f'sanity-check-error: output size {self.output_h}x{self.output_w} is invalid.')
            exit()
