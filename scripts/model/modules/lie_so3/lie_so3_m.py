import torch

class LieSE3(torch.nn.Module):
    """
        computes the Lie transform based on 3 given parameters
        estimated from a nerual network. the transform is computed
        per sample in the batch. the output is a 4x4 matrix.
    """
    def __init__(self, translation_scale=1, rotation_scale=0.1, eps=1e-5):
        super(LieSE3, self).__init__()
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.eps = eps

    def forward(self, uv):
        """
        uv -> Bx3: [x, y, p_yaw]
        """
        device = uv.device
        complete_transformation = torch.zeros((uv.size(0), 4, 4)).to(device)
        complete_transformation[:, 3, 3] = 1.
        for b in range(uv.size(0)):
            omega_z = uv[b, 2] * self.rotation_scale
            omega_skew = torch.zeros(3, 3).to(device)
            omega_skew[0, 1] = -omega_z
            omega_skew[1, 0] = omega_z
            u_vec = torch.zeros((3, 1)).to(device)
            u_vec[0, :] = uv[b, 0] * self.translation_scale
            u_vec[1, :] = uv[b, 1] * self.translation_scale
            u_vec[2, :] = 0
            omega_skew_squared = torch.mm(omega_skew, omega_skew)
            theta_sqr = omega_z * omega_z
            theta = torch.sqrt(theta_sqr)
            sin_theta_div_theta = torch.sin(theta) / (theta + self.eps)
            one_minus_cos_theta = 1 - torch.cos(theta)
            one_minus_cos_div_theta_sqr = one_minus_cos_theta / (theta_sqr + self.eps)
            # SO(3) rotation
            one_minus_A_div_theta_sqr = (1. - sin_theta_div_theta) / (theta_sqr + self.eps)
            complete_transformation[b, 0:3, 0:3] = torch.eye(3).to(device) + torch.mul(omega_skew, sin_theta_div_theta) + torch.mul(omega_skew_squared, one_minus_cos_div_theta_sqr)
            tmp = (torch.eye(3).to(device) + torch.mul(omega_skew, one_minus_cos_div_theta_sqr) + torch.mul(omega_skew_squared, one_minus_A_div_theta_sqr))
            complete_transformation[b, 0:3, 3] = torch.mm(tmp, u_vec).squeeze()

        return complete_transformation

class OrgLieSE3(torch.nn.Module):
    """
        computes the Lie transform based on 3 given parameters
        estimated from a nerual network. the transform is computed
        per sample in the batch. the output is a 4x4 matrix.
    """
    def __init__(self, translation_scale=1, rotation_scale=0.1, eps=1e-5):
        super(OrgLieSE3, self).__init__()
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.eps = eps

    def forward(self, uv):
        """
        uv -> Bx3: [x, y, p_yaw]
        """
        batch_size = uv.size(0)
        device = uv.device
        complete_transformation = torch.zeros((batch_size, 4, 4)).to(device)
        complete_transformation[:, 3, 3] = 1.

        for b in range(batch_size):
            omega_x = 0
            omega_y = 0
            omega_z = uv[b, 2] * self.rotation_scale

            u_x = uv[b, 0] * self.translation_scale
            u_y = uv[b, 1] * self.translation_scale
            u_z = 0

            omega_skew = torch.zeros(3, 3).to(device)

            omega_skew[0, 1] = -omega_z
            omega_skew[0, 2] = omega_y
            omega_skew[1, 0] = omega_z
            omega_skew[1, 2] = -omega_x
            omega_skew[2, 0] = -omega_y
            omega_skew[2, 1] = omega_x

            u_vec = torch.zeros((3, 1)).to(device)
            u_vec[0, :] = u_x
            u_vec[1, :] = u_y
            u_vec[2, :] = u_z

            omega_skew_squared = torch.mm(omega_skew, omega_skew)

            theta_sqr = omega_x * omega_x + omega_y * omega_y + omega_z * omega_z

            theta = torch.sqrt(theta_sqr)

            sin_theta = torch.sin(theta)
            sin_theta_div_theta = sin_theta / (theta + self.eps)

            one_minus_cos_theta = 1 - torch.cos(theta)
            one_minus_cos_div_theta_sqr = one_minus_cos_theta / (theta_sqr + self.eps)

            # SO(3) rotation
            one_minus_A_div_theta_sqr = (1. - sin_theta_div_theta) / (theta_sqr + self.eps)

            complete_transformation[b, 0:3, 0:3] = torch.eye(3).to(device) + torch.mul(omega_skew, sin_theta_div_theta) + torch.mul(omega_skew_squared, one_minus_cos_div_theta_sqr)

            tmp = (torch.eye(3).to(device) + torch.mul(omega_skew, one_minus_cos_div_theta_sqr) + torch.mul(
                omega_skew_squared, one_minus_A_div_theta_sqr))
            complete_transformation[b, 0:3, 3] = torch.mm(tmp, u_vec).squeeze()

        return complete_transformation


class LieSE2(torch.nn.Module):
    """
        computes the Lie transform based on 6 given parameters
        estimated from a nerual network. the transform is computed
        per sample in the batch.
    """
    def __init__(self, eps=1e-5):
        super(LieSE2, self).__init__()
        self.eps = eps

    def forward(self, uv):
        """
        uv -> Bx3: [x, y, p_yaw]
        """
        complete_transformation = torch.zeros((uv.size(0), 3, 3)).to(device)
        complete_transformation[:, 2, 2] = 1.

        for b in range(uv.size(0)):
            # u_x = uv[b, 0]
            # u_y = uv[b, 1]
            omega_z = uv[b, 2]
            omega_skew = torch.zeros(2, 2).to(device)
            omega_skew[0, 1] = -omega_z
            omega_skew[1, 0] = omega_z
            u_vec = torch.zeros((2, 1)).to(device)
            # u_vec[0, :] = u_x
            # u_vec[1, :] = u_y
            omega_skew_squared = torch.mm(omega_skew, omega_skew)
            theta_sqr = omega_z * omega_z
            theta = torch.sqrt(theta_sqr)
            sin_theta_div_theta = torch.sin(theta) / (theta + self.eps)
            one_minus_cos_theta = 1 - torch.cos(theta)
            one_minus_cos_div_theta_sqr = one_minus_cos_theta / (theta_sqr + self.eps)
            # SO(2) rotation
            one_minus_A_div_theta_sqr = (1. - sin_theta_div_theta) / (theta_sqr + self.eps)
            complete_transformation[b, 0:2, :2] = torch.eye(2).to(device) + torch.mul(omega_skew, sin_theta_div_theta) + torch.mul(omega_skew_squared, one_minus_cos_div_theta_sqr)
            tmp = (torch.eye(2).to(device) + torch.mul(omega_skew, one_minus_cos_div_theta_sqr) + torch.mul(omega_skew_squared, one_minus_A_div_theta_sqr))
            complete_transformation[b, 0:2, 2] = torch.mm(tmp, uv[:2].reshape(2, 1)).squeeze()

        return complete_transformation

if __name__ == '__main__':
    device = torch.device('cuda')
    lie_se3 = LieSE3().to(device)

    test_vec = torch.tensor([[0.12, 0.15, -1.0], [0.74, 0.88, 0.11]], device=device)
    print(lie_se3(test_vec))
