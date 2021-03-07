import yaml

class SemanticCloudConfig:
    def __init__(self, file_path: str):
        yaml_file = open(file_path)
        conf = yaml.load(yaml_file, Loader=yaml.FullLoader)
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
        self.training_name = conf['parameters']['name']
        self.dset_dir = conf['parameters']['dataset-dir']
        self.dset_file = conf['parameters']['dataset-file']
        self.dset_name = conf['parameters']['dataset-name']
        self.tensorboard_dir = conf['parameters']['tensorboard-dir']
        self.device = conf['parameters']['device']

        self.drop_prob = conf['hyperparameters']['drop-prob']
        self.output_h = conf['hyperparameters']['output-h']
        self.output_w = conf['hyperparameters']['output-w']
        self.classes = conf['hyperparameters']['classes']
        self.num_classes = conf['hyperparameters']['num-classes']
        self.learning_rate = float(conf['hyperparameters']['learning-rate'])
        self.epochs = conf['hyperparameters']['epochs']

        self.mask_detection_thresh = conf['validation']['mask-det-threshold']