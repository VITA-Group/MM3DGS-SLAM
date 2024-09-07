#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchmetrics.functional.regression import pearson_corrcoef
from utils.pose_utils import quadmultiply


def rel_pose_loss(camera_pose, initial_pose):
    """
    Calculate the loss between two poses.
    This is done with taking the L2 loss between the translation and
    rotation angle
    Return:
        Translation loss
        Angle loss
    """
    # Translation error
    t_err = ((camera_pose[4:] - initial_pose[4:]) ** 2).sum()

    # Rotation error
    gtconj = torch.tensor(
        [initial_pose[0], -initial_pose[1], -initial_pose[2], -initial_pose[3]]
    )
    diff = quadmultiply(camera_pose[:4], gtconj)
    diff = torch.nn.functional.normalize(diff.unsqueeze(0), dim=1).squeeze(0)
    angle_err = 2 * torch.acos(torch.abs(diff[0]))

    return t_err, angle_err


def pearson_loss(render, estimate, mask=None, invert_estimate=True):
    """
    Calculate loss using the pearson correlation coefficient.
    We want the pearson correlation to approach 1.
    """
    if mask is not None:
        render_masked = render[mask]
        estimate_masked = estimate[mask]
    else:
        render_masked = render
        estimate_masked = estimate
    if invert_estimate:
        loss = min(
            (1 - pearson_corrcoef(-estimate_masked, render_masked)).mean(),
            (1 - pearson_corrcoef(1 / (estimate_masked + 200.0), render_masked)).mean(),
        )
    else:
        loss = (1 - pearson_corrcoef(estimate_masked, render_masked)).mean()
    return loss


def l1_loss(network_output, gt, mask=None):
    if mask is None:
        return torch.abs((network_output - gt)).mean()
    else:
        return torch.abs((network_output - gt))[:, mask].mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.Tensor([w, x, y, z])


def quat_loss(network_output, gt):
    # Return rotation angle of the difference quaternion
    gtconj = torch.Tensor([gt[0], -gt[1], -gt[2], -gt[3]])
    diff = multiply_quaternions(network_output, gtconj)
    diff = diff / torch.norm(diff)
    return 2 * torch.acos(abs(diff[0]))


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
