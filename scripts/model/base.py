import torch
import kornia
import torch.nn as nn
from abc import ABC
from data.mask_warp import get_single_relative_img_transform, get_single_adjacent_aggregate_mask
from data.mask_warp import get_all_aggregate_masks
from data.utils import get_se2_diff
from metrics.iou import get_iou_per_class, get_mask_iou


"""
the following base classes offer a unified visualization
and iou caclulatin API to be used inside gui.py.dick
"""

class DoubleSemantic(nn.Module):
    """
    base class for networs that predict solo & aggregated semantics.
    implements methods for evaluation
    """
    def __init__(self):
        super().__init__()
    
    def parameter_count(self):
        """
        returns the number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def get_eval_output(self, semantic_classes: dict, graph_network: bool, rgbs: torch.Tensor,
                        car_masks: torch.Tensor, fov_masks: torch.Tensor, car_transforms: torch.Tensor,
                        adjacency_matrix: torch.Tensor, ppm, output_h, output_w, center_x, center_y,
                        agent_index: int, self_mask: bool, device: torch.device, noise_correction_en: bool):
        """
        return the solo & aggregated semantics & masks. the mask is taken from gt if not available.
        different conditions apply if the network is used as baseline.
        """
        agent_count = rgbs.shape[0]
        solo_gt_masks = car_masks + fov_masks
        # use manual aggregation if used as baseline
        if not graph_network:
            solo_sseg_preds, _ = self.forward(
                rgbs, car_transforms, torch.eye(agent_count, dtype=torch.bool),
                car_masks, noise_correction_en=noise_correction_en, evaluation=True
            )
            final_solo_sseg_pred = solo_sseg_preds[agent_index].clone()
            # remove mask from semantic classes for aggregation
            if 'Mask' in semantic_classes.values():
                # mask id is the last index of the label list
                mask_sem_id = len(semantic_classes.keys()) - 1
                # save original solo predictions before pre-aggregation mutilation
                # solo prediction is inherently masked => predicted mask = ones
                final_solo_mask_pred = torch.ones_like(solo_gt_masks[0])
                # extract mask from predictions for aggregation
                semantic_ids = torch.argmax(solo_sseg_preds, dim=1)
                solo_mask_preds = (semantic_ids == mask_sem_id).long()
                # aggregate prediction doesn't have masks, aggregate from solo preds
                final_aggr_mask_pred = get_single_adjacent_aggregate_mask(
                    solo_mask_preds, car_transforms, agent_index,
                    ppm, output_h, output_w, center_x, center_y,
                    adjacency_matrix, True
                )
                # remove mask channel from semantic predictions
                solo_sseg_preds = solo_sseg_preds[:, :mask_sem_id]
            # use gt masks since the network doesn't predcit them
            else:
                final_solo_mask_pred = solo_gt_masks[agent_index]
                final_aggr_mask_pred = get_single_adjacent_aggregate_mask(
                    solo_gt_masks, car_transforms, agent_index,
                    ppm, output_h, output_w, center_x, center_y,
                    adjacency_matrix, True
                )
                solo_mask_preds = solo_gt_masks
            
            # mask of regions outside FoV
            if self_mask:
                solo_sseg_preds *= solo_mask_preds
            # aggregate semantic predictions
            outside_fov = torch.where(adjacency_matrix[agent_index] == 0)[0]
            relative_tfs = get_single_relative_img_transform(
                car_transforms, agent_index, ppm, center_x, center_y
            ).to(rgbs.device)
            agent_relative_semantics = kornia.warp_affine(
                solo_sseg_preds, relative_tfs, dsize=(output_h, output_w),
                mode='nearest'
            )
            agent_relative_semantics[outside_fov] = 0
            final_aggr_sseg_pred = agent_relative_semantics.sum(dim=0)
        # use latent aggregation if not used as baseline
        else:
            solo_sseg_preds, aggr_sseg_preds = self.forward(
                rgbs, car_transforms, adjacency_matrix, car_masks,
                noise_correction_en=noise_correction_en, evaluation=True
            )
            final_solo_sseg_pred = solo_sseg_preds[agent_index]
            final_aggr_sseg_pred = aggr_sseg_preds[agent_index]
            if 'Mask' in semantic_classes.values():
                final_solo_mask_pred = torch.ones_like(solo_gt_masks[0])
                final_aggr_mask_pred = torch.ones_like(solo_gt_masks[0])
            else:
                final_solo_mask_pred = solo_gt_masks[agent_index]
                final_aggr_mask_pred = get_single_adjacent_aggregate_mask(
                    solo_gt_masks, car_transforms, agent_index,
                    ppm, output_h, output_w, center_x, center_y,
                    adjacency_matrix, True
                )
        # calculate aggregated mask based on adjacency matrix (either from network or gt)
        return final_solo_sseg_pred.to(device), final_solo_mask_pred.to(device), \
               final_aggr_sseg_pred.to(device), final_aggr_mask_pred.to(device)

    @torch.no_grad()
    def get_batch_ious(self, semantic_classes: dict, graph_network: bool, rgbs: torch.Tensor,
                       car_masks: torch.Tensor, fov_masks: torch.Tensor, gt_transforms: torch.Tensor,
                       car_transforms: torch.Tensor, labels: torch.Tensor, ppm, output_h, output_w,
                       center_x, center_y, mask_detect_thresh, noise_correction_en: bool):
        """
        the function returns the ious for a batch.
        """
        num_classes = len(semantic_classes.keys())
        agent_count = rgbs.shape[0]
        solo_gt_masks = car_masks + fov_masks
        # for this type of network, mask iou is only calculated for baseline
        # networks with mask as semantic id. for others, it is automtaically
        # calculated with get_iou_per_class
        mask_iou = 0.0
        aggr_gt_masks = get_all_aggregate_masks(
            solo_gt_masks, gt_transforms, ppm, output_h, output_w,
            center_x, center_y, merge_masks=True
        )
        # manual aggregation for baseline network
        if not graph_network:
            solo_sseg_preds, _ = self.forward(
                rgbs, car_transforms, torch.eye(agent_count, dtyp=torch.bool),
                car_masks, noise_correction_en=noise_correction_en, evaluation=True
            )
            if 'Mask' in semantic_classes.values():
                # mask id is the last index of the label list
                mask_sem_id = num_classes - 1
                semantic_ids = torch.argmax(solo_sseg_preds, dim=1)
                # extract mask from semantic labels for sem aggregation & mask iou calc
                solo_mask_preds = (semantic_ids == mask_sem_id).long()
                aggr_mask_preds = get_all_aggregate_masks(
                    solo_mask_preds, car_transforms, ppm,
                    output_h, output_w, center_x, center_y, merge_masks=True
                )
                mask_iou = get_mask_iou(aggr_mask_preds, aggr_gt_masks, mask_detect_thresh).item()
                # remove masks from semantic predictions
                solo_sseg_preds = solo_sseg_preds[:, :mask_sem_id]
            # use gt since the network doesn't predcit masks
            else:
                solo_mask_preds = solo_gt_masks

            # masking of regions outside FoV
            solo_sseg_preds *= solo_mask_preds
            # aggregating semantic predictions
            aggr_sseg_preds = torch.zeros_like(solo_sseg_preds)
            for i in range(agent_count):
                relative_tfs = get_single_relative_img_transform(
                    car_transforms, i, ppm, center_x, center_y
                ).to(rgbs.device)
                relative_semantics = kornia.warp_affine(
                    solo_sseg_preds, relative_tfs,
                    dsize=(output_h, output_w), mode='nearest'
                )
                aggr_sseg_preds[i] = relative_semantics.sum(dim=0)
        # latent aggregation for others
        else:
            _, aggr_sseg_preds = self.forward(
                rgbs, car_transforms, torch.ones((agent_count, agent_count), dtype=torch.bool),
                car_masks, noise_correction_en=noise_correction_en, evaluation=True
            )
            # disable masking for iou calculation if not baseline & mask is a semantic class
            if 'Mask' in semantic_classes.values():
                aggr_gt_masks = torch.ones_like(solo_gt_masks)
        
        mskd_ious = get_iou_per_class(
            aggr_sseg_preds, labels, aggr_gt_masks, num_classes
        ).to(rgbs.device)
        full_ious = get_iou_per_class(
            aggr_sseg_preds, labels, torch.ones_like(aggr_gt_masks), num_classes
        ).to(rgbs.device)
        # add the calculated iou that was removed from the semantic ids
        if 'Mask' in semantic_classes.values() and not graph_network:
            mskd_ious[-1] = full_ious [-1] = mask_iou
            mask_iou = 0
        return mskd_ious, full_ious, mask_iou


class AggrSemanticsSoloMask(nn.Module):
    """
    base class for networs that predict solo mask and aggregated semantics.
    implements methods for evaluation. the best model so far (wce-mcnnt-0noise-full-0.1)
    is of this type
    """
    def __init__(self):
        super().__init__()
    
    def parameter_count(self):
        """
        returns the number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def get_eval_output(self, semantic_classes: dict, graph_network: bool, rgbs: torch.Tensor,
                        car_masks: torch.Tensor, fov_masks: torch.Tensor, car_transforms: torch.Tensor,
                        adjacency_matrix: torch.Tensor, ppm, output_h, output_w, center_x, center_y,
                        agent_index: int, self_mask: bool, device: torch.device, noise_correction_en: bool):
        """
        return the solo & aggregated semantics & masks. the mask is taken from gt if not available.
        different conditions apply if the network is used as baseline.
        """
        agent_count = rgbs.shape[0]
        # use manual aggregation if used as baseline
        if not graph_network:
            solo_sseg_preds, solo_mask_preds = self.forward(
                rgbs, car_transforms, torch.eye(agent_count, dtype=torch.bool),
                car_masks, noise_correction_en=noise_correction_en, evaluation=True
            )
            final_solo_sseg_pred = solo_sseg_preds[agent_index].clone()
            final_solo_mask_pred = solo_mask_preds[agent_index].squeeze()
            final_aggr_mask_pred = get_single_adjacent_aggregate_mask(
                solo_mask_preds.squeeze(1), car_transforms, agent_index,
                ppm, output_h, output_w, center_x, center_y,
                adjacency_matrix, False
            )
            # mask of regions outside FoV for semantic aggregation
            if self_mask:
                solo_sseg_preds *= solo_mask_preds
            # aggregate semantic predictions
            outside_fov = torch.where(adjacency_matrix[agent_index] == 0)[0]
            relative_tfs = get_single_relative_img_transform(
                car_transforms, agent_index, ppm, center_x, center_y
            ).to(rgbs.device)
            agent_relative_semantics = kornia.warp_affine(
                solo_sseg_preds, relative_tfs, dsize=(output_h, output_w),
                mode='nearest'
            )
            agent_relative_semantics[outside_fov] = 0
            final_aggr_sseg_pred = agent_relative_semantics.sum(dim=0)
        # use latent aggregation if not used as baseline
        else:
            solo_sseg_preds, _ = self.forward(
                rgbs, car_transforms, torch.eye(agent_count, dtype=torch.bool),
                car_masks, noise_correction_en=noise_correction_en, evaluation=True
            )
            aggr_sseg_preds, solo_mask_preds = self.forward(
                rgbs, car_transforms, adjacency_matrix, car_masks,
                noise_correction_en=noise_correction_en, evaluation=True
            )
            final_solo_sseg_pred = solo_sseg_preds[agent_index]
            final_aggr_sseg_pred = aggr_sseg_preds[agent_index]
            final_solo_mask_pred = solo_mask_preds[agent_index].squeeze()
            final_aggr_mask_pred = get_single_adjacent_aggregate_mask(
                solo_mask_preds, car_transforms, agent_index,
                ppm, output_h, output_w, center_x, center_y,
                adjacency_matrix, False
            )
        # calculate aggregated mask based on adjacency matrix (either from network or gt)
        return final_solo_sseg_pred.to(device), final_solo_mask_pred.to(device), \
               final_aggr_sseg_pred.to(device), final_aggr_mask_pred.to(device)

    @torch.no_grad()
    def get_batch_ious(self, semantic_classes: dict, graph_network: bool, rgbs: torch.Tensor,
                       car_masks: torch.Tensor, fov_masks: torch.Tensor, gt_transforms: torch.Tensor,
                       car_transforms: torch.Tensor, labels: torch.Tensor, ppm, output_h, output_w,
                       center_x, center_y, mask_detect_thresh, noise_correction_en: bool):
        """
        the function returns the ious for a batch.
        """
        num_classes = len(semantic_classes.keys())
        agent_count = rgbs.shape[0]
        solo_gt_masks = car_masks + fov_masks
        aggr_gt_masks = get_all_aggregate_masks(
            solo_gt_masks, gt_transforms, ppm, output_h, output_w,
            center_x, center_y, merge_masks=True
        )
        # aggr. mask is not directly estimated but calculated
        # from warped solo masks
        mask_iou = 0.0
        # manual aggregation for baseline network
        if not graph_network:
            solo_sseg_preds, solo_mask_preds = self.forward(
                rgbs, car_transforms, torch.eye(agent_count, dtype=torch.bool),
                car_masks, noise_correction_en=noise_correction_en, evaluation=True
            )
            # masking of regions outside estimated FoV
            solo_sseg_preds *= solo_mask_preds
            # aggregating semantic predictions
            aggr_sseg_preds = torch.zeros_like(solo_sseg_preds)
            for i in range(agent_count):
                relative_tfs = get_single_relative_img_transform(
                    car_transforms, i, ppm, center_x, center_y
                ).to(rgbs.device)
                relative_semantics = kornia.warp_affine(
                    solo_sseg_preds, relative_tfs, dsize=(output_h, output_w),
                    mode='nearest'
                )
                aggr_sseg_preds[i] = relative_semantics.sum(dim=0)
        # latent aggregation for others
        else:
            aggr_sseg_preds, solo_mask_preds = self.forward(
                rgbs, car_transforms, torch.ones((agent_count, agent_count), dtype=torch.bool),
                car_masks, noise_correction_en=noise_correction_en, evaluation=True
            )
        # thresholding masks before (and after) aggregation
        solo_mask_preds[solo_mask_preds >= mask_detect_thresh] = 1
        solo_mask_preds[solo_mask_preds < mask_detect_thresh] = 0
        aggr_mask_preds = get_all_aggregate_masks(
            solo_mask_preds.squeeze(1), car_transforms, ppm,
            output_h, output_w, center_x, center_y, merge_masks=False
        )
        mask_iou  = get_mask_iou(aggr_mask_preds, aggr_gt_masks, mask_detect_thresh)
        mskd_ious = get_iou_per_class(aggr_sseg_preds, labels, aggr_gt_masks, num_classes).to(rgbs.device)
        full_ious = get_iou_per_class(
            aggr_sseg_preds, labels, torch.ones_like(aggr_gt_masks), num_classes
        ).to(rgbs.device)

        return mskd_ious, full_ious, mask_iou


class SoloAggrSemanticsMask(nn.Module):
    """
    base class for networs that predict solo & aggregated masks and semantics.
    implements methods for evaluation. must not be used as baseline.
    """
    def __init__(self):
        super().__init__()
    
    def parameter_count(self):
        """
        returns the number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def get_eval_output(self, semantic_classes: dict, graph_network: bool, rgbs: torch.Tensor,
                        car_masks: torch.Tensor, fov_masks: torch.Tensor, car_transforms: torch.Tensor,
                        adjacency_matrix: torch.Tensor, ppm, output_h, output_w, center_x, center_y,
                        agent_index: int, self_mask: bool, device: torch.device, noise_correction_en: bool):
        """
        return the solo & aggregated semantics & masks. the mask is taken from gt if not available.
        different conditions apply if the network is used as baseline.
        """
        # this type of network is explictly written for comparison with baseline
        assert graph_network, 'network not suitable as baseline. use smaller version'
        solo_sseg_preds, solo_mask_preds, aggr_sseg_preds, aggr_mask_preds =  self.forward(
            rgbs, car_transforms, adjacency_matrix, car_masks, 
            noise_correction_en=noise_correction_en, evaluation=True
        )
        final_solo_sseg_pred = solo_sseg_preds[agent_index].to(device)
        final_aggr_sseg_pred = aggr_sseg_preds[agent_index].to(device)
        final_solo_mask_pred = solo_mask_preds[agent_index].to(device).squeeze()
        final_aggr_mask_pred = aggr_mask_preds[agent_index].to(device).squeeze()
        return final_solo_sseg_pred, final_solo_mask_pred, \
               final_aggr_sseg_pred, final_aggr_mask_pred

    @torch.no_grad()
    def get_batch_ious(self, semantic_classes: dict, graph_network: bool, rgbs: torch.Tensor,
                       car_masks: torch.Tensor, fov_masks: torch.Tensor, gt_transforms: torch.Tensor,
                       car_transforms: torch.Tensor, labels: torch.Tensor, ppm, output_h, output_w,
                       center_x, center_y, mask_detect_thresh, noise_correction_en: bool):
        """
        the function returns the ious for a batch.
        """
        assert graph_network, 'network not suitable as baseline. use smaller version'
        agent_count = rgbs.shape[0]
        num_classes = len(semantic_classes.keys())
        solo_gt_masks = car_masks + fov_masks
        aggr_gt_masks = get_all_aggregate_masks(
            solo_gt_masks, gt_transforms, ppm, output_h, output_w,
            center_x, center_y, merge_masks=True
        )
        _, _, aggr_sseg_preds, aggr_mask_preds = self.forward(
            rgbs, car_transforms, torch.ones((agent_count, agent_count), dtype=torch.bool),
            car_masks, noise_correction_en=noise_correction_en, evaluation=True
        )
        mask_iou  = get_mask_iou(aggr_mask_preds.squeeze(1), aggr_gt_masks, mask_detect_thresh).item()
        mskd_ious = get_iou_per_class(
            aggr_sseg_preds, labels, aggr_gt_masks, num_classes
        ).to(rgbs.device)
        full_ious = get_iou_per_class(
            aggr_sseg_preds, labels, torch.ones_like(aggr_gt_masks), num_classes
        ).to(rgbs.device)

        return mskd_ious, full_ious, mask_iou


class NoiseEstimator(ABC):
    @torch.no_grad()
    def get_batch_noise_performance(self, rgbs: torch.Tensor, car_masks: torch.Tensor, car_transforms: torch.Tensor,
                                    gt_transforms: torch.Tensor, adjacency_matrix: torch.Tensor):
        """
        returns accumulated x, y and theta  absolute error before and after noise correction.
        also returns number of estimated noise instances.
        """        
        agent_count = rgbs.shape[0]
        x_noise = {'pre': [], 'post': []}
        y_noise = {'pre': [], 'post': []}
        t_noise = {'pre': [], 'post': []}

        # ignore single batches
        if agent_count == 1:
            return x_noise, y_noise, t_noise
        self.forward(
            rgbs, car_transforms, adjacency_matrix, car_masks,
            noise_correction_en=True, evaluation=True
        )
        for i in range(agent_count):
            # T_j w.r.t. T_i
            gt_relative_tfs = gt_transforms[i].inverse() @ gt_transforms
            # T_j' w.r.t. T_i
            nz_relative_tfs = car_transforms[i].inverse() @ car_transforms
            # hopefully T_j w.r.t. T_i
            co_relative_tfs = nz_relative_tfs @ self.feat_matching_net.estimated_noise[i].inverse()
            # present relative noise, pre correction
            noise_pre = get_se2_diff(nz_relative_tfs, gt_relative_tfs)
            # present relative noise, post correction
            noise_post = get_se2_diff(co_relative_tfs, gt_relative_tfs)
            # get absolute noise
            xpre = torch.abs(noise_pre[:, 0]).cpu().tolist()
            ypre = torch.abs(noise_pre[:, 1]).cpu().tolist()
            tpre = torch.abs(noise_pre[:, 2]).cpu().tolist()
            xpost = torch.abs(noise_post[:, 0]).cpu().tolist()
            ypost = torch.abs(noise_post[:, 1]).cpu().tolist()
            tpost = torch.abs(noise_post[:, 2]).cpu().tolist()
            # mark and discard invalid predictions (ego & out-of-view)
            unwanted = torch.where(adjacency_matrix[i] == False)[0].cpu().tolist()
            unwanted.append(i)
            for unwanted_idx in sorted(unwanted, reverse=True):
                xpre.pop(unwanted_idx); ypre.pop(unwanted_idx); tpre.pop(unwanted_idx)
                xpost.pop(unwanted_idx); ypost.pop(unwanted_idx); tpost.pop(unwanted_idx)
            # add to the list
            x_noise['pre'] += xpre
            y_noise['pre'] += ypre
            t_noise['pre'] += tpre
            x_noise['post'] += xpost
            y_noise['post'] += ypost
            t_noise['post'] += tpost

        return x_noise, y_noise, t_noise