import os
import yaml
import datetime

num_classes_dict = {
    'ours': 7,
    'ours+mask': 8,
    'carla': 23,
    'diminished': 3,
    'diminished+mask': 4
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
        # noise parameters
        self.se2_noise_enable = conf['se2-noise']['enable']
        if self.se2_noise_enable:
            self.se2_noise_th_std = float(conf['se2-noise']['se2-noise-theta-std'])
            self.se2_noise_dx_std = float(conf['se2-noise']['se2-noise-dx-std'])
            self.se2_noise_dy_std = float(conf['se2-noise']['se2-noise-dy-std'])
        else:
            self.se2_noise_th_std = 0.0
            self.se2_noise_dx_std = 0.0
            self.se2_noise_dy_std = 0.0
        # network
        self.model_name = str(conf['network']['model-name'])
        self.extra_model_arg = str(conf['network']['extra-arg'])
        self.aggregation_type = str(conf['network']['aggregation-type'])
        # curriculum config
        self.curriculum_activate = bool(conf['curriculum']['activate'])
        self.initial_difficulty = int(conf['curriculum']['initial-difficulty'])
        self.maximum_difficulty = int(conf['curriculum']['maximum-difficulty'])
        self.max_agent_count = int(conf['curriculum']['maximum-agent-count'])
        self.strategy = str(conf['curriculum']['strategy'])
        self.strategy_parameter = conf['curriculum']['strategy-parameter']
        self.enforce_adj_calc = bool(conf['curriculum']['enforce-adj-calc'])
        # hyperparameters
        self.drop_prob = float(conf['hyperparameters']['drop-prob'])
        self.learning_rate = float(conf['hyperparameters']['learning-rate'])
        self.epochs = int(conf['hyperparameters']['epochs'])
        self.color_jitter = list(conf['hyperparameters']['color-jitter'])
        self.gaussian_mask_std = float(conf['hyperparameters']['gaussian-blur-std'])
        self.gaussian_kernel_size = int(conf['hyperparameters']['gaussian-kernel-size'])
        self.wallhack_prob = float(conf['hyperparameters']['wallhack-prob'])
        # validation parameters
        self.mask_detection_thresh = float(conf['validation']['mask-det-threshold'])
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
        # cross entropy weights
        if self.loss_function == 'weighted-cross-entropy':
            self.ce_weights = list(conf['training']['ce-weights'])
        else:
            self.ce_weights = [1.0] * self.num_classes
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
        if self.loss_function != 'cross-entropy' and self.loss_function != 'focal' \
                and self.loss_function != 'weighted-cross-entropy':
            print(f'sanity-check-error: unkown loss function {self.loss_function}.')
            exit()
        if self.aggregation_type != 'bilinear' and self.aggregation_type != 'nearest':
            print(f'sanity-check-error: unkown aggregation type {self.aggregation_type}.')
            exit()
        if self.se2_noise_enable:
            if self.se2_noise_dx_std == 0 and self.se2_noise_dy_std == 0 and self.se2_noise_th_std == 0:
                print(f'sanity-check-error: noise std cannot be 0 if se2 noise is enabled.')
                exit()
        if self.gaussian_mask_std < 0:
            print(f'sanity-check-error: gaussian mask std cannot be negative.')
            exit()
        if self.gaussian_kernel_size < 0:
            print(f'sanity-check-error: gaussian kernel size cannot be negative.')
            exit()
        if self.gaussian_kernel_size % 2 == 0:
            print(f'sanity-check-error: gaussian kernel size must be odd.')
            exit()
        if self.wallhack_prob < 0 or self.wallhack_prob > 1:
            print(f'sanity-check-error: wallhack probability cannot be negative or greater than 1.')
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
        if self.classes != 'ours' and self.classes != 'carla' and self.classes != 'diminished' \
                                  and self.classes != 'ours+mask' and self.classes != 'diminished+mask':
            print(f'sanity-check-error: unknown segmentation classes {self.classes}.')
            exit()
        if self.resume_training and self.resume_tag == '':
            print(f'sanity-check-error: resuming a run requires a resume-tag.')
            exit()
        if self.output_h > 500 or self.output_w > 400:
            print(f'sanity-check-error: output size {self.output_h}x{self.output_w} is invalid.')
            exit()

    def print_config(self):
        print('training config\n-----------------------------------')
        # general
        print(f'torch-seed: {self.torch_seed}')
        print(f'loss-function: {self.loss_function}')
        # noise
        if not self.se2_noise_enable:
            print(f'se2-noise: disabled')
        else:
            print(f'se2-noise: enabled')
            print(f'se2-noise dx-std: {self.se2_noise_dx_std}')
            print(f'se2-noise dy-std: {self.se2_noise_dy_std}')
            print(f'se2-noise th-std: {self.se2_noise_th_std}')
        # network
        print(f'network: {self.model_name}')
        print(f'aggregation-type: {self.aggregation_type}')
        # curriculum config
        if self.curriculum_activate:
            print(f'curriculum: activated')
            print(f'curriculum strategy: {self.strategy}')
            print(f'curriculum initial difficulty: {self.initial_difficulty}')
            print(f'curriculum maximum difficulty: {self.maximum_difficulty}')
        else:
            print(f'curriculum: deactivated')
            print(f'enforce adj. calculation: {self.enforce_adj_calc}')
        # hyperparameters
        print(f'connection drop probability: {self.drop_prob}')
        if self.gaussian_mask_std == 0:
            print(f'gaussian mask smoothing: disabled')
        else:
            print(f'gaussian mask smoothing: enabled')
            print(f'gaussian mask std: {self.gaussian_mask_std}')
            print(f'gaussian kernel size: {self.gaussian_kernel_size}')
        if self.wallhack_prob == 0:
            print(f'wallhack: disabled')
        else:
            print(f'wallhack: enabled')
            print(f'wallhack probability: {self.wallhack_prob}')
        # validation parameters
        print(f'mask detection threshold: {self.mask_detection_thresh}')
        # dataloader config
        print('data loader workers: {}'.format(self.loader_workers))
        # dataset config
        print(f'classes: {self.classes}')
        print(f'output size: {self.output_h}x{self.output_w}')
        # resume
        print(f'datetime: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}')
        if self.resume_training:
            print(f'resuming training from {self.resume_model_version} checkpoint (epoch {self.resume_starting_epoch})')
            print(f'resume tag: {self.resume_tag}')
            print(f'resume difficulty: {self.resume_difficulty}')
            print(f'resume optimizer state: {self.resume_optimizer_state}')
        else:
            #print date and time without seconds
            print('starting new training')
        print('-----------------------------------')

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
        self.model_gnn_flags = list(conf['models']['graph-networks'])
        self.model_names = list(conf['models']['model-names'])
        self.model_versions = list(conf['models']['model-versions'])
        self.aggregation_types = list(conf['models']['aggregation-types'])
        self.model_extra_arg = str(conf['models']['extra-arg'])
        # gui parameteres
        self.evaluate_ious_at_start = bool(conf['gui']['evaluate-ious-at-start'])
        self.evaluate_noise_at_start = bool(conf['gui']['evaluate-noise-at-start'])
        self.profile_at_start = bool(conf['gui']['profile-at-start'])
        self.mask_thresh = float(conf['gui']['mask-threshold'])
        self.sample_save_dir = str(conf['gui']['sample-save-dir'])
        self.full_metrics_save_dir = str(conf['gui']['full-metrics-save-dir'])
        self.transparent_masks = bool(conf['gui']['transparent-masks'])
        # noise parameters
        self.se2_noise_th_std = float(conf['se2-noise']['se2-noise-theta-std'])
        self.se2_noise_dx_std = float(conf['se2-noise']['se2-noise-dx-std'])
        self.se2_noise_dy_std = float(conf['se2-noise']['se2-noise-dy-std'])
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
        self.gaussian_mask_std = float(conf['dataset']['gaussian-blur-std'])
        self.gaussian_kernel_size = int(conf['dataset']['gaussian-kernel-size'])
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
        if self.difficulty < 1 or self.difficulty > self.max_agent_count:
            print(f'sanity-check-error: invalid difficulty {self.difficulty}.')
            exit()
        if self.output_h > 500 or self.output_w > 400:
            print(f'sanity-check-error: output size {self.output_h}x{self.output_w} is invalid.')
            exit()
        if self.gaussian_mask_std < 0:
            print(f'sanity-check-error: gaussian mask std cannot be negative.')
            exit()
        if self.gaussian_kernel_size < 0:
            print(f'sanity-check-error: gaussian kernel size cannot be negative.')
            exit()
        if self.gaussian_kernel_size % 2 == 0:
            print(f'sanity-check-error: gaussian kernel size must be odd.')
            exit()
        if self.mask_thresh < 0 or self.mask_thresh > 1:
            print(f'sanity-check-error: mask threshold must be between 0 and 1.')
            exit()
        if len(self.runs) != len(self.model_names) or len(self.runs) != len(self.model_versions) or \
           len(self.runs) != len(self.aggregation_types) or len(self.runs) != len(self.model_gnn_flags):
            print(f'sanity-check-error: model lists should have the same length.')
            exit()
        if not os.path.exists(self.sample_save_dir):
            print(f'sanity-check-error: sample save directory {self.sample_save_dir} does not exist.')
            exit()
        if not os.path.exists(self.full_metrics_save_dir):
            print(f'sanity-check-error: full metrics save directory {self.full_metrics_save_dir} does not exist.')
            exit()
        for i in range(len(self.model_versions)):
            if self.model_versions[i] != 'best' and self.model_versions[i] != 'last':
                print(f'sanity-check-error: {self.model_versions[i]} is not a valid model version.')
                exit()
        for i in range(len(self.aggregation_types)):
            if self.aggregation_types[i] != 'bilinear' and self.aggregation_types[i] != 'nearest':
                print(f'sanity-check-error: {self.aggregation_types[i]} is not a valid aggregation type.')
                exit()


class ReportConfig:
    def __init__(self, file_path: str):
        yaml_file = open(file_path)
        conf = yaml.load(yaml_file, Loader=yaml.FullLoader)
        yaml_file.close()
        # evaluation parameters
        self.report_name = str(conf['report-name'])
        self.device = str(conf['device'])
        self.torch_seed = int(conf['torch-seed'])
        self.snapshot_dir = str(conf['snapshot-dir'])
        self.log_dir = str(conf['log-dir'])
        # model parameters
        self.runs = list(conf['models']['runs'])
        self.model_names = list(conf['models']['model-names'])
        self.model_versions = list(conf['models']['model-versions'])
        self.aggregation_types = list(conf['models']['aggregation-types'])
        # noise parameters
        self.se2_noise_enable = conf['se2-noise']['enable']
        if self.se2_noise_enable:
            self.se2_noise_th_std = float(conf['se2-noise']['se2-noise-theta-std'])
            self.se2_noise_dx_std = float(conf['se2-noise']['se2-noise-dx-std'])
            self.se2_noise_dy_std = float(conf['se2-noise']['se2-noise-dy-std'])
        else:
            self.se2_noise_th_std = 0.0
            self.se2_noise_dx_std = 0.0
            self.se2_noise_dy_std = 0.0
        # hard batches
        self.hard_batch_indices = list(conf['hard-batch']['indices'])
        self.hard_batch_labels = list(conf['hard-batch']['labels'])
        # curriculum parameters
        self.difficulty = int(conf['curriculum']['difficulty'])
        self.max_agent_count = int(conf['curriculum']['maximum-agent-count'])
        # dataset parameters
        self.random_samples = bool(conf['dataset']['random-samples'])
        self.dset_dir = str(conf['dataset']['dataset-dir'])
        self.dset_file = str(conf['dataset']['dataset-file'])
        self.dset_name = str(conf['dataset']['dataset-name'])
        self.output_h = int(conf['dataset']['output-h'])
        self.output_w = int(conf['dataset']['output-w'])
        self.classes = str(conf['dataset']['classes'])
        self.num_classes = num_classes_dict[self.classes]
        self.perform_sanity_check()
    
    def perform_sanity_check(self):
        """
        performs a sanity check to make sure the given parameters are not
        absolutely wrong
        """
        if self.device != 'cuda' and self.device != 'cpu':
            print(f'sanity-check-error: unknown device {self.device}.')
            exit()
        if self.difficulty < 1 or self.difficulty > self.max_agent_count:
            print(f'sanity-check-error: invalid difficulty {self.difficulty}.')
            exit()
        if self.output_h > 500 or self.output_w > 400:
            print(f'sanity-check-error: output size {self.output_h}x{self.output_w} is invalid.')
            exit()
        for i in range(len(self.model_versions)):
            if self.model_versions[i] != 'best' and self.model_versions[i] != 'last':
                print(f'sanity-check-error: {self.model_versions[i]} is not a valid model version.')
                exit()
        for i in range(len(self.aggregation_types)):
            if self.aggregation_types[i] != 'bilinear' and self.aggregation_types[i] != 'nearest':
                print(f'sanity-check-error: {self.aggregation_types[i]} is not a valid aggregation type.')
                exit()
        if self.se2_noise_enable:
            if self.se2_noise_dx_std == 0 and self.se2_noise_dy_std == 0 and self.se2_noise_th_std == 0:
                print(f'sanity-check-error: noise std cannot be 0 if se2 noise is enabled.')
                exit()
