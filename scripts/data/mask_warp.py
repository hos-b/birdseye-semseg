import numpy as np
import torch
import kornia
import matplotlib.pyplot as plt

def get_centered_img_transforms(transforms: torch.Tensor, pixels_per_meter, h, w, center_x, center_y) -> torch.Tensor:
    """
    turns a location-based transform into an image-centered pixel-based transform,
    to be used for warping. the x & y
    input: tensor of shape A x 3 x 3, relative transforms of agents
    output: tensor of shape A x 2 x 3, image-centered transforms of agents w.r.t. a specific agent
    """
    assert len(transforms.shape) == 3, f"transforms should have the dimensions A x 3 x 3 but got {transforms.shape}"
    assert transforms.shape[1:3] == torch.Size([3, 3]), f"transforms should be 3 x 3 but they're {transforms.shape[1:3]}"
    rectified_tf = transforms.clone()
    # image rotation = inverse of cartesian rotation. for some reason tranpose doesn't work
    rectified_tf[:, :2, :2] = rectified_tf[:, :2, :2].inverse()
    # image +x = cartesian -y, image +y = cartesian -x
    rectified_tf[:, 0, 2] = -transforms[:, 1, 2]
    rectified_tf[:, 1, 2] = -transforms[:, 0, 2]
    # changing the translation from meters to pixels
    rectified_tf[:, 0, 2] *= pixels_per_meter
    rectified_tf[:, 1, 2] *= pixels_per_meter
    porg = torch.tensor([[1.0, 0.0, center_x],
                         [0.0, 1.0, center_y],
                         [0.0, 0.0,      1.0]]).unsqueeze(0)
    norg = torch.tensor([[1.0, 0.0, -center_x],
                         [0.0, 1.0, -center_y],
                         [0.0, 0.0,       1.0]])
    return (porg @ rectified_tf @ norg)[:, :2, :]

def get_single_relative_img_transform(transforms: torch.Tensor, agent_id, pixels_per_meter, h, w, center_x, center_y) -> torch.Tensor:
    """
    input: 
        - tensor of shape A x 4 x 4, transforms of agents w.r.t. origin
        - index of the agent to be considered
        - pixels per meter
        - image height
        - image width
        - x_center in image frame
        - y_center in image frame
    output: 
        - tensor of shape A x 3 x 2, transforms of agents w.r.t. given agent
    """
    assert len(transforms.shape) == 3, f"transforms should have the dimensions Ax3x3 but got {transforms.shape}"
    assert transforms.shape[1:3] == torch.Size([4, 4]), f"transforms should be 4x4 but they're {transforms.shape[1:3]}"
    agent_count, _, _ = transforms.shape
    # 3D transforms w.r.t. origin -> 3D transforms w.r.t. one agent
    rel_3d_tf = transforms[agent_id].inverse() @ transforms
    # 3D relative transforms -> 2D relative transforms
    rel_2d_tf = torch.eye(3).unsqueeze(0).repeat(agent_count, 1, 1)
    # copying over the translation
    rel_2d_tf[:, :2,  2] = rel_3d_tf[:, :2,  3]
    # copying over the [yaw] rotation
    rel_2d_tf[:, :2, :2] = rel_3d_tf[:, :2, :2]
    # top-left corner transform -> center transform + conversion from meters to pixels
    return get_centered_img_transforms(rel_2d_tf, pixels_per_meter, h, w, center_x, center_y)

def get_all_relative_img_transforms(transforms: torch.Tensor, pixels_per_meter, h, w, center_x, center_y) -> torch.Tensor:
    """
    input: 
        - tensor of shape A x 4 x 4, transforms of agents w.r.t. origin
        - pixels per meter
        - image height
        - image width
        - x_center in image frame
        - y_center in image frame
    output: 
        - tensor of shape (A*A) x 3 x 2, transforms of agents w.r.t. all other agents
    """
    assert len(transforms.shape) == 3, f"transforms should have the dimensions Ax3x3 but got {transforms.shape}"
    assert transforms.shape[1:3] == torch.Size([4, 4]), f"transforms should be 4x4 but they're {transforms.shape[1:3]}"
    agent_count = transforms.shape[0]
    # new dimension where all 3D transforms w.r.t. origin are repeated
    rel_3d_tf = transforms.repeat(agent_count, 1, 1)
    # 3D transforms w.r.t. origin -> 3D relative transforms
    rel_3d_tf = transforms.inverse().repeat_interleave(agent_count, dim=0) @ rel_3d_tf
    # 3D relative transforms -> 2D relative transforms
    rel_2d_tf = torch.eye(3).unsqueeze(0).repeat(agent_count * agent_count, 1, 1)
    # copying over the translation
    rel_2d_tf[:, :2,  2] = rel_3d_tf[:, :2,  3]
    # copying over the [yaw] rotation
    rel_2d_tf[:, :2, :2] = rel_3d_tf[:, :2, :2]
    # top-left corner transform -> center transform + conversion from meters to pixels
    return get_centered_img_transforms(rel_2d_tf, pixels_per_meter, h, w, center_x, center_y)

def get_single_aggregate_mask(masks, transforms, agent_id, pixels_per_meter, h, w, center_x, center_y, merge_masks=False):
    """
    input: masks & transforms of all agents, target agent id, extra info
    output: accumulative mask for that agent
    """
    assert len(masks.shape) == 3, f"masks should have the dimensions AxHxW but got {masks.shape}"
    assert agent_id < masks.shape[0], f"given agent index {agent_id} does not exist"
    relative_tfs = get_single_relative_img_transform(transforms, agent_id, pixels_per_meter, h, w, center_x, center_y)
    warped_mask = kornia.warp_affine(masks.unsqueeze(1), relative_tfs, dsize=(h, w), flags='bilinear')
    warped_mask = warped_mask.sum(dim=0)
    if merge_masks:
        warped_mask[warped_mask > 1] = 1
        warped_mask[warped_mask < 1] = 0
    return warped_mask

def get_all_aggregate_masks_deprecated(masks, transforms, pixels_per_meter, h, w, center_x, center_y):
    """
    input: masks & transforms of all agents, target agent id, extra info
    output: accumulative mask for all agents
    deprecated. for loop replaced with big brain version
    """
    assert len(masks.shape) == 3, f"masks should have the dimensions AxHxW but got {masks.shape}"
    agent_count = masks.shape[0]
    all_masks = torch.zeros_like(masks)
    for i in range(agent_count):
        relative_tfs = get_single_relative_img_transform(transforms, i, pixels_per_meter, h, w, center_x, center_y).to(masks.device)
        warped_mask = kornia.warp_affine(masks.unsqueeze(1), relative_tfs, dsize=(h, w), flags='nearest')
        all_masks[i] = warped_mask.sum(dim=0)
    return all_masks

def get_all_aggregate_masks(masks, transforms, pixels_per_meter, h, w, center_x, center_y):
    """
    input:
        - all agent masks
        - all agent transforms
        - extra info about the geometry
    output:
        - accumulative identified masks for each agent
    """
    assert len(masks.shape) == 3, f"masks should have the dimensions AxHxW but got {masks.shape}"
    agent_count = masks.shape[0]
    relative_tfs = get_all_relative_img_transforms(transforms, pixels_per_meter, h, w, center_x, center_y).to(masks.device)
    warped_masks = kornia.warp_affine(masks.unsqueeze(1).repeat(agent_count, 1, 1, 1),
                                      relative_tfs, dsize=(h, w), flags='nearest')
    warped_masks = warped_masks.reshape(agent_count, agent_count, h, w)
    return warped_masks.sum(dim=1)

def test_transforms(transforms, pixels_per_meter, h, w, center_x, center_y):
    agent_count = transforms.shape[0]
    relative_tfs_2 = get_all_relative_img_transforms(transforms, pixels_per_meter, h, w, center_x, center_y)
    relative_tfs_1 = torch.zeros_like(relative_tfs_2)
    for i in range(agent_count):
        relative_tfs_1[i * agent_count : (i + 1) * agent_count] = get_single_relative_img_transform(transforms, i, pixels_per_meter, h, w, center_x, center_y)
    if (torch.abs(relative_tfs_1 - relative_tfs_2) < 1e-15).unique() != torch.tensor([True]):
        print('\nrelative transforms are not equal')
    else:
        print('relative transforms are equal')


if __name__ == "__main__":
    # creating 20 x 20 checkerboard
    npimg = np.zeros((400, 400))
    for i in range(20):
        for j in range(20):
            if i % 2 == 0 and j % 2 == 0:
                npimg[i * 20 : (i + 1) * 20, j * 20 : (j + 1) * 20] = 1.0
            elif (i + 1) % 2 == 0 and (j + 1) % 2 == 0:
                npimg[i * 20 : (i + 1) * 20, j * 20 : (j + 1) * 20] = 1.0

    # B x C x H x W
    img = torch.from_numpy(npimg).unsqueeze(0).unsqueeze(0).float()
    # B x 2 x 3
    alpha = np.pi / 8
    x_transform = 0
    y_transform = 0
    x_mid = 400
    y_mid = 400
    # centered rotation
    positive_tf = torch.tensor([[1.0, 0.0, x_mid],
                                [0.0, 1.0, y_mid],
                                [0.0, 0.0,   1.0]]).float()
    tfrm        = torch.tensor([[np.cos(alpha), -np.sin(alpha), x_transform],
                                [np.sin(alpha),  np.cos(alpha), y_transform],
                                [0.0          , 0.0           ,        1.0]]).float()
    negative_tf = torch.tensor([[1.0, 0.0, -x_mid],
                                [0.0, 1.0, -y_mid],
                                [0.0, 0.0,    1.0]]).float()
    tfrm = positive_tf @ tfrm @ negative_tf
    # import pdb; pdb.set_trace()
    tfrm = tfrm[:2, :].unsqueeze(0)
    # tfrm = get_centered_img_transforms(tfrm, 1.0, 400, 400, 200, 200)
    transformed_img = kornia.warp_affine(img, tfrm, dsize=(400, 400), flags='bilinear')
    # transformed_img[transformed_img < 1] = 0
    plt.imshow(transformed_img.float().squeeze().numpy())
    plt.show()