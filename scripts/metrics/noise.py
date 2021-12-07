import torch
import statistics
from agent.agent_pool import CurriculumPool
from data.config import SemanticCloudConfig
from data.utils import get_noisy_transforms
from model.base import NoiseEstimator

class NoiseMetrics:
    """
    evaluates a single network for noise correction
    """
    def __init__(self, label: str, network: NoiseEstimator, device, max_agent_count: int) -> None:
        if not hasattr(network, 'feat_matching_net'):
            raise ValueError('object does not have feature matching network')
        self.x_noise = {'pre': [], 'post': []}
        self.y_noise = {'pre': [], 'post': []}
        self.t_noise = {'pre': [], 'post': []}
        self.network = network
        self.label = label
        self.pool = CurriculumPool(
            max_agent_count, max_agent_count, max_agent_count, True, device
        )

    def update_network(self, rgbs, car_masks, fov_masks, gt_transforms,
                       noise_std_x, noise_std_y, noise_std_theta,
                       output_h, output_w, ppm, center_x, center_y):


        noisy_transforms = get_noisy_transforms(gt_transforms,
                                                noise_std_x,
                                                noise_std_y,
                                                noise_std_theta)
        solo_masks = car_masks + fov_masks
        self.pool.generate_connection_strategy(
            solo_masks, gt_transforms, ppm, output_h, output_w, center_x, center_y
        )
        # no noise, no correction
        x_noise, y_noise, t_noise = self.network.get_batch_noise_performance(
            rgbs, car_masks, noisy_transforms, gt_transforms, self.pool.adjacency_matrix
        )
        self.x_noise['pre'] += x_noise['pre']
        self.y_noise['pre'] += y_noise['pre']
        self.t_noise['pre'] += t_noise['pre']
        self.x_noise['post'] += x_noise['post']
        self.y_noise['post'] += y_noise['post']
        self.t_noise['post'] += t_noise['post']
    
    def finish(self):
        self.metrics = {
            'pre_x_mean': statistics.mean(self.x_noise['pre']),
            'pre_x_median': statistics.median(self.x_noise['pre']),
            'pre_y_mean': statistics.mean(self.y_noise['pre']),
            'pre_y_median': statistics.median(self.y_noise['pre']),
            'pre_t_mean': statistics.mean(self.t_noise['pre']),
            'pre_t_median': statistics.median(self.t_noise['pre']),
            'post_x_mean': statistics.mean(self.x_noise['post']),
            'post_x_median': statistics.median(self.x_noise['post']),
            'post_y_mean': statistics.mean(self.y_noise['post']),
            'post_y_median': statistics.median(self.y_noise['post']),
            'post_t_mean': statistics.mean(self.t_noise['post']),
            'post_t_median': statistics.median(self.t_noise['post']),
        }
        self.metrics['pre_x_std'] = statistics.stdev(self.x_noise['pre'], self.metrics['pre_x_mean'])
        self.metrics['pre_y_std'] = statistics.stdev(self.y_noise['pre'], self.metrics['pre_y_mean'])
        self.metrics['pre_t_std'] = statistics.stdev(self.t_noise['pre'], self.metrics['pre_t_mean'])
        self.metrics['post_x_std'] = statistics.stdev(self.x_noise['post'], self.metrics['post_x_mean'])
        self.metrics['post_y_std'] = statistics.stdev(self.y_noise['post'], self.metrics['post_y_mean'])
        self.metrics['post_t_std'] = statistics.stdev(self.t_noise['post'], self.metrics['post_t_mean'])


    def write_to_file(self, file_path: str):
        lines = [f'{self.label} noise metrics\n']
        lines.append('noise  | pre estimation mean | pre estimation std | post estimation mean | post estimation std\n')
        lines.append('x      | ' + f'{self.metrics["pre_x_mean"]:.3f}'.center(19)  + ' | ' \
                                 + f'{self.metrics["pre_x_std"]:.3f}'.center(18)   + ' | ' \
                                 + f'{self.metrics["post_x_mean"]:.3f}'.center(20) + ' | ' \
                                 + f'{self.metrics["post_x_std"]:.3f}'.center(19)  + '\n')
        lines.append('y      | ' + f'{self.metrics["pre_y_mean"]:.3f}'.center(19)  + ' | ' \
                                 + f'{self.metrics["pre_y_std"]:.3f}'.center(18)   + ' | ' \
                                 + f'{self.metrics["post_y_mean"]:.3f}'.center(20) + ' | ' \
                                 + f'{self.metrics["post_y_std"]:.3f}'.center(19)  + '\n')
        lines.append('theta  | ' + f'{self.metrics["pre_t_mean"]:.3f}'.center(19)  + ' | ' \
                                 + f'{self.metrics["pre_t_std"]:.3f}'.center(18)   + ' | ' \
                                 + f'{self.metrics["post_t_mean"]:.3f}'.center(20) + ' | ' \
                                 + f'{self.metrics["post_t_std"]:.3f}'.center(19)  + '\n')
        file = open(file_path, 'w')
        file.writelines(lines)
        file.close()
