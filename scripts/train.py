import os
from matplotlib.pyplot import plot_date
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import subprocess


from data.dataset import get_datasets
from data.config import SemanticCloudConfig, TrainingConfig
from data.color_map import our_semantics_to_cityscapes_rgb
from data.color_map import __our_classes as classes
from data.mask_warp import get_all_aggregate_masks
from data.utils import drop_agent_data, squeeze_all, get_matplotlib_image
from model.mass_cnn import MassCNN
from agent.agent_pool import AgentPool
from metrics.iou import iou_per_class, mask_iou

def to_device(rgbs, labels, masks, car_transforms, device):
    return rgbs.to(device), labels.to(device), \
           masks.to(device), car_transforms.to(device)

# opening semantic cloud settings file
geom_cfg = SemanticCloudConfig('../mass_data_collector/param/sc_settings.yaml')
train_cfg = TrainingConfig('config/training.yml')
DATASET_DIR = train_cfg.dset_dir
PKG_NAME = train_cfg.dset_file
DATASET = train_cfg.dset_name
TENSORBOARD_DIR = train_cfg.tensorboard_dir
NEW_SIZE = (train_cfg.output_h, train_cfg.output_w)

# image size and center coordinates
CENTER = (geom_cfg.center_x(NEW_SIZE[1]), geom_cfg.center_y(NEW_SIZE[0]))
PPM = geom_cfg.pix_per_m(NEW_SIZE[0], NEW_SIZE[1])

# dataset
device = torch.device(train_cfg.device)
train_set, test_set = get_datasets(DATASET, DATASET_DIR, PKG_NAME, (0.8, 0.2), NEW_SIZE, train_cfg.classes)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=4)

# logging
name = train_cfg.training_name + '-'
name += subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8')[:-1]
writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, name))

# saving model
last_metric = 0.0

# network stuff
model = MassCNN(geom_cfg,
                num_classes=train_cfg.num_classes,
                device=device,
                output_size=NEW_SIZE).to(device)
epochs = train_cfg.epochs
agent_pool = AgentPool(model, device, NEW_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
semseg_loss = nn.CrossEntropyLoss(reduction='none')
mask_loss = nn.L1Loss(reduction='none')
print(f"{(model.parameter_count() / 1e6):.2f}M trainable parameters")

for ep in range(epochs):
    total_train_m_loss = 0.0
    total_train_s_loss = 0.0
    total_valid_m_loss = 0.0
    total_valid_s_loss = 0.0
    # training
    # model.train()
    # for batch_idx, (_, rgbs, labels, masks, car_transforms) in enumerate(train_loader):
    #     print(f'\repoch: {ep + 1}/{epochs}, training batch: {batch_idx + 1} / {len(train_loader)}', end='')
    #     rgbs, labels, masks, car_transforms = to_device(rgbs, labels, masks, car_transforms, device)
    #     # simulate connection drops
    #     rgbs, labels, masks, car_transforms = drop_agent_data(rgbs, labels, masks, car_transforms, train_cfg.drop_prob)
    #     # masked loss
    #     aggregate_masks = get_all_aggregate_masks(masks, car_transforms, PPM, NEW_SIZE[0], \
    #                                               NEW_SIZE[1], CENTER[0], CENTER[1], device)
    #     optimizer.zero_grad()
    #     agent_pool.calculate_detached_messages(rgbs)
    #     for i in range(agent_pool.agent_count):
    #         mask_pred = agent_pool.calculate_agent_mask(rgbs[i])
    #         sseg_pred = agent_pool.aggregate_messages(i, car_transforms)
    #         m_loss = torch.mean(mask_loss(mask_pred.squeeze(), masks[i]) * aggregate_masks[i], dim=(0, 1))
    #         s_loss = torch.mean(semseg_loss(sseg_pred, labels[i].unsqueeze(0)) * masks[i], dim=(0, 1, 2))
    #         (m_loss + s_loss).backward()
    #         optimizer.step()
    #         batch_train_m_loss = m_loss.item()
    #         batch_train_s_loss = s_loss.item()
    #     writer.add_scalar("loss/batch_train_msk", batch_train_m_loss, ep * len(train_loader) + batch_idx)
    #     writer.add_scalar("loss/batch_train_seg", batch_train_s_loss, ep * len(train_loader) + batch_idx)
    #     total_train_m_loss += batch_train_m_loss
    #     total_train_s_loss += batch_train_s_loss

    # writer.add_scalar("loss/total_train_msk", total_train_m_loss / len(train_loader), ep + 1)
    # writer.add_scalar("loss/total_train_seg", total_train_s_loss / len(train_loader), ep + 1)
    # print(f'\nepoch loss: {total_train_m_loss / len(train_loader)} mask, '
    #                     f'{total_train_s_loss / len(train_loader)} segmentation')

    # validation
    model.eval()
    visaulized = False
    sseg_ious = torch.zeros((train_cfg.num_classes, 1), dtype=torch.float64)
    mask_ious = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (_, rgbs, labels, masks, car_transforms) in enumerate(test_loader):
            print(f'\repoch: {ep + 1}/{epochs}, validation batch: {batch_idx + 1} / {len(test_loader)}', end='')
            sample_count += rgbs.shape[1]
            rgbs, labels, masks, car_transforms = squeeze_all(rgbs, labels, masks, car_transforms)
            rgbs, labels, masks, car_transforms = to_device(rgbs, labels, masks, car_transforms, device)
            mask_preds, sseg_preds = model(rgbs, car_transforms)
            sseg_ious += iou_per_class(sseg_preds, labels)
            mask_ious += mask_iou(mask_preds.squeeze(), masks, train_cfg.mask_detection_thresh).item()
            m_loss = mask_loss(mask_preds.squeeze(), masks.squeeze())
            s_loss = semseg_loss(sseg_preds, labels)
            total_valid_m_loss += torch.mean(m_loss.view(1, -1)).item()
            total_valid_s_loss += torch.mean(s_loss.view(1, -1)).item()
            # visaluize the first agent from the first batch
            if not visaulized:
                aggregate_masks = get_all_aggregate_masks(masks, car_transforms, PPM, NEW_SIZE[0], \
                                                          NEW_SIZE[1], CENTER[0], CENTER[1], device)
                ss_trgt_img = our_semantics_to_cityscapes_rgb(labels[0].cpu()).transpose(2, 0, 1)
                ss_mask = aggregate_masks[0].cpu()
                ss_trgt_img[:, ss_mask == 0] = 0
                _, ss_pred = torch.max(sseg_preds[0], dim=0)
                ss_pred_img = our_semantics_to_cityscapes_rgb(ss_pred.cpu()).transpose(2, 0, 1)
                pred_mask_img = get_matplotlib_image(mask_preds[0].squeeze().cpu())
                trgt_mask_img = get_matplotlib_image(masks[0].cpu())
                writer.add_image("validation/input_rgb", rgbs[0], ep + 1)
                writer.add_image("validation/mask_predicted", torch.from_numpy(pred_mask_img).permute(2, 0, 1), ep + 1)
                writer.add_image("validation/mask_target", torch.from_numpy(trgt_mask_img).permute(2, 0, 1), ep + 1)
                writer.add_image("validation/segmentation_predicted", ss_pred_img, ep + 1)
                writer.add_image("validation/segmentation_target", torch.from_numpy(ss_trgt_img), ep + 1)
                visaulized = True
    
    new_metric = 0.0
    writer.add_scalar("loss/total_valid_msk", total_valid_m_loss / len(test_loader), ep + 1)
    writer.add_scalar("loss/total_valid_seg", total_valid_s_loss / len(test_loader), ep + 1)
    writer.add_scalar("iou/mask_iou", mask_ious / sample_count, ep + 1)
    for key, val in classes.items():
        writer.add_scalar(f"iou/{val.lower()}_iou", sseg_ious[key] / sample_count, ep + 1)
        new_metric += sseg_ious[key] / sample_count
    new_metric += mask_ious / sample_count
    print(f'\nepoch loss: {total_valid_m_loss / len(test_loader)} mask, {total_valid_s_loss / len(test_loader)} segmentation')

    if new_metric > last_metric:
        print(f'saving snapshot at epoch {ep}')
        last_metric = new_metric
        if not os.path.exists(train_cfg.snapshot_dir):
            os.makedirs(train_cfg.snapshot_dir)
        torch.save(optimizer.state_dict(), train_cfg.snapshot_dir + '/optimizer_dict')
        torch.save(model.state_dict(), train_cfg.snapshot_dir + '/model_snapshot.pth')

writer.close()