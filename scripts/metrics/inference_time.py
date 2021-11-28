import time
import torch

class InferenceMetrics:
    def __init__(self, networks: dict, max_agent_count) -> None:
        self.max_agent_count = max_agent_count
        self.inference_times = {}
        # + 1 for pre-aggr calc
        self.counts = torch.zeros(max_agent_count + 1, dtype=torch.int64)
        for key in networks.keys():
            # check of inference functions
            if not hasattr(networks[key], 'infer_preaggr'):
                raise ValueError(f'{key} has not implemented the infer_preaggr function.')
            if not hasattr(networks[key], 'infer_aggregate'):
                raise ValueError(f'{key} has not implemented the infer_aggregate function.')
            self.inference_times[key] = torch.zeros(max_agent_count + 1, dtype=torch.float64)

    def update_network(self, network_label: str, network: torch.nn.Module, rgbs, car_masks, gt_transforms):
        batch_size = rgbs.shape[0]
        self.counts[0] += batch_size
        self.counts[batch_size] += 1
        semantics = []
        masks = []
        # perform pre-aggregation inference
        for i in range(batch_size):
            start_tick = time.time()
            ego_semantic, ego_mask = network.infer_preaggr(rgbs[i].unsqueeze(0), car_masks[i].unsqueeze(0))
            elapsed_ms = (time.time() - start_tick) * 1000.0
            semantics.append(ego_semantic)
            masks.append(ego_mask)
            self.inference_times[network_label][0] += elapsed_ms
        # stack outputs
        semantics = torch.cat(semantics, dim=0)
        masks = torch.cat(masks, dim=0)
        # perform aggregation inference
        for i in range(batch_size):
            start_tick = time.time()
            network.infer_aggregate(i, semantics, masks, gt_transforms)
            elapsed_ms = (time.time() - start_tick) * 1000.0
            self.inference_times[network_label][batch_size] += elapsed_ms
    
    def finish(self):
        for key in self.inference_times.keys():
            self.inference_times[key] /= self.counts
    
    def write_to_file(self, file_path: str, net_name_length = 24):
        headers = [f'aggr. {str(i + 1)}' for i in range(self.max_agent_count)]
        headers.insert(0, 'ego inf')
        file = open(file_path, 'w')
        header_str = f'{"network".ljust(net_name_length)} || {" || ".join(headers)}'
        file.write(f'{header_str}\n{"=" * len(header_str)}\n')
        for (net_label, net_dict) in self.inference_times.items():
            line = net_label[:net_name_length].ljust(net_name_length) + '    '
            for i in range(self.max_agent_count + 1):
                line += f'{net_dict[i]:.3f}'[:7].ljust(7) + '    '
            file.write(f'{line[:-4]}\n')
        file.close()
