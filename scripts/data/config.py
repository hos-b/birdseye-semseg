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
        self.pix_per_m = self.image_rows / self.cloud_x_span
        # position of the center of the car in the image (not in cartesian space)
        self.center_y = int((self.cloud_max_x / self.cloud_x_span) * self.image_rows)
        self.center_x = int((self.cloud_max_y / self.cloud_y_span) * self.image_cols)