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

import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from slam.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix2
from utils.pose_utils import get_camera_from_tensor, quadmultiply
from utils.sh_utils import eval_sh


def get_depth_and_silhouette(means3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame
    means_homo = torch.cat((means3D, torch.ones_like(means3D[:, :1])), dim=-1)
    means_c = (w2c @ means_homo.transpose(0, 1)).transpose(0, 1)
    depth_z = means_c[:, 2].unsqueeze(-1)  # [num_gaussians, 1]
    depth_z_sq = torch.square(depth_z)  # [num_gaussians, 1]

    # Depth and Silhouette
    depth_silhouette = torch.zeros((means3D.shape[0], 3)).cuda().float()
    depth_silhouette[:, 0] = depth_z.squeeze(-1)
    depth_silhouette[:, 1] = 1.0
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1)

    return depth_silhouette


class Renderer:
    def __init__(self, cfg):
        self.cfg = cfg

        # Rasterization settings
        self.zfar = 100.0
        self.znear = 0.01
        self.image_height = int(self.cfg["desired_height"])
        self.image_width = int(self.cfg["desired_width"])
        self.cx = self.cfg["cam"]["cx"]
        self.cy = self.cfg["cam"]["cy"]
        self.fovx = self.cfg["cam"]["fx"]
        self.fovy = self.cfg["cam"]["fy"]
        # self.tanfovx = math.tan(self.fovx * 0.5)
        # self.tanfovy = math.tan(self.fovy * 0.5)
        self.tanfovx = self.image_width / (2 * self.fovx)
        self.tanfovy = self.image_height / (2 * self.fovy)
        # Calculate projection matrix
        self.projection_matrix = (
            getProjectionMatrix2(
                znear=self.znear,
                zfar=self.zfar,
                fx=self.fovx,
                fy=self.fovy,
                cx=self.cx,
                cy=self.cy,
                h=self.image_height,
                w=self.image_width,
            )
            .transpose(0, 1)
            .to(self.cfg["device"])
        )

        # Background color
        bg_color = [1, 1, 1] if self.cfg["white_background"] else [0, 0, 0]
        self.background = torch.tensor(
            bg_color, dtype=torch.float32, device=self.cfg["device"]
        )

    def render(
        self,
        pc: GaussianModel,
        camera_pose: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                pc.get_xyz,
                dtype=pc.get_xyz.dtype,
                requires_grad=True,
                device=self.cfg["device"],
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration

        # Get viewmatrix and projmatrix from camera_tensor in a differentiable way
        # Transpose to account for row/col-major conventions
        if self.cfg["pipeline"]["transform_means_python"]:
            w2c = torch.eye(4, device=self.cfg["device"])
        else:
            w2c = get_camera_from_tensor(camera_pose).transpose(0, 1)
        projmatrix = (
            w2c.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_pos = w2c.inverse()[3, :3]
        raster_settings = GaussianRasterizationSettings(
            image_height=self.image_height,
            image_width=self.image_width,
            tanfovx=self.tanfovx,
            tanfovy=self.tanfovy,
            bg=self.background,
            scale_modifier=scaling_modifier,
            viewmatrix=w2c,
            projmatrix=projmatrix,
            sh_degree=pc.active_sh_degree,
            campos=camera_pos,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        if self.cfg["pipeline"]["transform_means_python"]:
            # Apply w2c transform to gaussians
            rel_w2c = get_camera_from_tensor(camera_pose)
            # Transform mean and rot of Gaussians to camera frame
            gaussians_xyz = pc._xyz.clone()
            gaussians_rot = pc._rotation.clone()

            xyz_ones = torch.ones(gaussians_xyz.shape[0], 1).cuda().float()
            xyz_homo = torch.cat((gaussians_xyz, xyz_ones), dim=1)
            gaussians_xyz_trans = (rel_w2c @ xyz_homo.T).T[:, :3]
            gaussians_rot_trans = quadmultiply(camera_pose[:4], gaussians_rot)
            means3D = gaussians_xyz_trans
        else:
            means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.cfg["pipeline"]["compute_cov3D_python"]:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            if self.cfg["pipeline"]["force_isotropic"]:
                scales = torch.exp(torch.tile(pc._scaling[:, 0].unsqueeze(1), (1, 3)))
            else:
                scales = pc.get_scaling
            if self.cfg["pipeline"]["transform_means_python"]:
                # rotations = gaussians_rot_trans
                rotations = pc.get_rotation
            else:
                rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if self.cfg["pipeline"]["convert_SHs_python"]:
                shs_view = pc.get_features.transpose(1, 2).view(
                    -1, 3, (pc.max_sh_degree + 1) ** 2
                )
                dir_pp = pc.get_xyz - camera_pos.repeat(pc.get_features.shape[0], 1)
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
        )

        # add depth + alpha rendering
        rendered_depth, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            colors_precomp=get_depth_and_silhouette(means3D, w2c),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
