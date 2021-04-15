import torch

def iou_per_class(predictions: torch.Tensor, labels: torch.Tensor, target_sseg_mask: torch.Tensor, num_classes=7) -> torch.Tensor:
    """
    returns a [num_classes x 1] tensor containing the sum iou of each class for all images.
    in the end, the aggregated ious should be devided by the number of images, skipped here
    because the batch sizes differ.
    """
    assert len(predictions.shape) == 4, f"expected [B x num_classes x H x W], got {predictions.shape}"
    assert len(labels.shape) == 3, f"expected [B x H x W], got {labels.shape}"
    assert predictions.shape[1] == num_classes, f"expected second dim to be {num_classes}, got {predictions.shape[1]}"
    bool_mask = target_sseg_mask == 1
    pred_argmax = torch.argmax(predictions, dim=1)
    ious = torch.zeros((num_classes, 1), dtype=torch.float64)
    for i in range(num_classes):
        pred_class_i = pred_argmax == i
        labl_class_i = labels == i
        # image lvl iou for each class
        intersection = (pred_class_i & labl_class_i & bool_mask).sum(dim=(1, 2))
        union = ((pred_class_i | labl_class_i) & bool_mask).sum(dim=(1, 2))
        iou = intersection / union
        # set NaNs to zero
        iou[iou != iou] = 0
        ious[i] = iou.sum()
    return ious

def mask_iou(predictions: torch.Tensor, gt_masks: torch.Tensor, detection_tresh):
    assert len(predictions.shape) == len(gt_masks.shape), \
           f"dimensions of predictions {predictions.shape} != ground truth {gt_masks.shape}"
    preds = predictions.clone()
    preds[preds >= detection_tresh] = 1.0
    preds[preds < detection_tresh] = 0.0
    preds = preds.long()
    labels = gt_masks.long()
    # image lvl iou for mask
    intersection = (preds & labels).sum(dim=(1, 2))
    union = (preds | labels).sum(dim=(1, 2))
    iou = intersection / union
    # set NaNs to zero
    iou[iou != iou] = 0
    return iou.sum()