import numpy as np
import torch
import kornia
import matplotlib.pyplot as plt

#pylint: disable=E1101
#pylint: disable=not-callable

def get_centered_img_transform(transforms: torch.Tensor, pixels_per_meter, h, w, center_x, center_y) -> torch.Tensor:
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

def get_relative_img_transform(transforms: torch.Tensor, agent_id, pixels_per_meter, h, w, center_x, center_y) -> torch.Tensor:
    """
    input: tensor of shape A x 4 x 4, transforms of agents w.r.t. origin
    output: tensor of shape A x 3 x 2, transforms of agents w.r.t. given agent
    """
    assert len(transforms.shape) == 3, f"transforms should have the dimensions Ax3x3 but got {transforms.shape}"
    assert transforms.shape[1:3] == torch.Size([4, 4]), f"transforms should be 4x4 but they're {transforms.shape[1:3]}"
    agent_count, _, _ = transforms.shape
    # 3D transforms w.r.t. origin -> 2D transforms w.r.t. one agent
    rel_3d_tf = transforms[agent_id].inverse() @ transforms
    rel_2d_tf = torch.eye(3).unsqueeze(0).repeat(agent_count, 1, 1)
    # copying over the translation
    rel_2d_tf[:, :2,  2] = rel_3d_tf[:, :2,  3]
    # copying over the [yaw] rotation
    rel_2d_tf[:, :2, :2] = rel_3d_tf[:, :2, :2]
    # top-left corner transform -> center transform + conversion from meters to pixels
    return get_centered_img_transform(rel_2d_tf, pixels_per_meter, h, w, center_x, center_y)

def get_aggregate_mask(masks, transforms, agent_id, pixels_per_meter, h, w, center_x, center_y):
    """
    input: masks & transforms of all agents, target agent id, extra info
    output: accumulative mask for that agent
    """
    assert len(masks.shape) == 3, f"masks should have the dimensions AxHxW but got {masks.shape}"
    assert agent_id < masks.shape[0], f"given agent index {agent_id} does not exist"
    relative_tfs = get_relative_img_transform(transforms, agent_id, pixels_per_meter, h, w, center_x, center_y)
    masks_cp = masks.clone().unsqueeze(1)
    warped_mask = kornia.warp_affine(masks_cp, relative_tfs, dsize=(h, w), flags='bilinear')
    warped_mask = warped_mask.sum(dim=0)
    # warped_mask[warped_mask > 1] = 1
    # warped_mask[warped_mask < 1] = 0
    return warped_mask

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
    # tfrm = get_centered_img_transform(tfrm, 1.0, 400, 400, 200, 200)
    transformed_img = kornia.warp_affine(img, tfrm, dsize=(400, 400), flags='bilinear')
    # transformed_img[transformed_img < 1] = 0
    plt.imshow(transformed_img.float().squeeze().numpy())
    plt.show()