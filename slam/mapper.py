import os
import time
from collections import defaultdict, deque
from random import randint

import cv2
import numpy as np
import pyiqa
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from utils.depth_utils import depth_to_rgb
from utils.loss_utils import l1_loss, pearson_loss, ssim
from utils.pose_utils import get_camera_from_tensor
from utils.sh_utils import RGB2SH


class KeyFrame:
    """
    Container holding the image, estimated pose, and idx
    """

    def __init__(
        self, idx, gt_color, est_pose, gt_depth=None, est_depth=None, niqe=None
    ):
        self.idx = idx
        self.gt_color = gt_color
        self.pose = est_pose
        self.gt_depth = gt_depth
        self.est_depth = est_depth
        self.niqe = niqe


class Mapper:
    """
    Mapping thread
    """

    def __init__(self, slam):
        self.cfg = slam.cfg

        # Inherit global variables
        self.gaussians = slam.gaussians
        self.n_img = slam.n_img
        self.estimate_pose_list = slam.estimate_pose_list
        self.gt_pose_list = slam.gt_pose_list
        self.renderer = slam.renderer
        if slam.use_imu:
            self.tf = slam.tf
            self.tstamps = slam.tstamps

        # Optimization params
        self.num_iter = self.cfg["mapping"]["iters"]
        self.camera_extent = 0  # This will get set by SLAM on first frame

        # Initialize Keyframes
        self.keyframes = []
        self.covisibility_graph = defaultdict(
            set
        )  # Adjacency list. Edges define covisibility

        # Load checkpoint
        if "iteration" in self.cfg:
            results = np.load(
                os.path.join(self.cfg["outputdir"], "results.npz"), allow_pickle=True
            )
            self.keyframes = [KeyFrame(**kf) for kf in results["keyframes"]]
            for kf_idx in range(len(self.keyframes)):
                self.update_covisibility_graph(kf_idx)

        # Set up NIQE metric evaluator
        self.niqe = pyiqa.create_metric(
            "niqe", device="cpu"
        )  # CPU since GPU is not working...
        if self.cfg["mapping"]["niqe_kf"]:
            self.niqe_window = deque(maxlen=self.cfg["mapping"]["niqe_window_size"])

        # Set up logging
        if self.cfg["debug"]["create_video"]:
            self.video_writer = slam.video_writer_full
        if self.cfg["debug"]["get_runtime_stats"]:
            # Log the mapping per-iteration runtime
            self.mapping_time_sum = 0
            self.mapping_iter_count = 0

    def add_keyframe(self, idx, est_pose, gt_color, gt_depth=None, est_depth=None):
        """
        Add a new keyframe to the keyframe list
        Returns the new KeyFrame added
        """
        # Get the lowest NIQE keyframe from the sliding window
        if self.cfg["mapping"]["niqe_kf"]:
            new_kf = self.niqe_window[0]
        else:
            new_kf = KeyFrame(idx, gt_color, est_pose, gt_depth, est_depth)
        self.keyframes.append(new_kf)
        if idx > 0:
            self.update_covisibility_graph(len(self.keyframes) - 1)
        return new_kf

    def need_new_keyframe(
        self, idx, est_pose, gt_color, gt_depth=None, est_depth=None
    ) -> bool:
        """
        Return whether a new keyframe is needed.
        """
        if self.cfg["method"].lower() == "splatam":
            # Add keyframe every n frames
            # Add keyframe if 2nd to last frame and gt pose is stable
            return (
                (idx == 0)
                or ((idx + 1) % self.cfg["mapping"]["kf_every"] == 0)
                or (idx == self.n_img - 2)
            )

        else:
            if self.cfg["mapping"]["niqe_kf"]:
                # Keep a sliding window of minimum NIQE keyframes
                frame_niqe = self.niqe(gt_color.unsqueeze(0))
                curr_kf = KeyFrame(
                    idx, gt_color, est_pose, gt_depth, est_depth, frame_niqe
                )
                # Remove the elements which are out of the window
                if idx >= self.cfg["mapping"]["niqe_window_size"]:
                    while (
                        self.niqe_window
                        and self.niqe_window[0].idx
                        <= idx - self.cfg["mapping"]["niqe_window_size"]
                    ):
                        self.niqe_window.popleft()
                # For every element, the previous smaller elements are useless
                while self.niqe_window and frame_niqe < self.niqe_window[-1].niqe:
                    self.niqe_window.pop()
                self.niqe_window.append(curr_kf)

            if len(self.keyframes) == 0 or idx == 0:
                return True

            # If still have covisibility with current KF, don't spawn new KF
            # Get the depth rendering of key
            with torch.no_grad():
                result = self.renderer.render(
                    self.gaussians,
                    camera_pose=self.keyframes[-1].pose,
                )
                depth = result["depth"][0, :, :]
                silhouette = result["depth"][1, :, :]
                presence_sil_mask = silhouette > 0.99
                depth[~presence_sil_mask] = 0
                valid_depth_indices = torch.where(depth > 0)
                valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
                sampled_indices = valid_depth_indices
                # Back Project the selected pixels to 3D Pointcloud
                w2c = get_camera_from_tensor(self.keyframes[-1].pose)
                pts = self.get_depth_pointcloud(depth, w2c, sampled_indices)
                height, width = depth.shape

            if self.is_covisible(
                pts,
                self.estimate_pose_list[idx],
                height,
                width,
                threshold=self.cfg["mapping"]["min_covisibility"],
            ):
                return False

            # Spawn new KF every n frames
            if idx - self.keyframes[-1].idx >= self.cfg["mapping"]["kf_every"]:
                return True

            return False

    def get_depth_pointcloud(self, depth, w2c, sampled_indices):
        FX = self.cfg["cam"]["fx"]
        FY = self.cfg["cam"]["fy"]
        CX = self.cfg["cam"]["cx"]
        CY = self.cfg["cam"]["cy"]

        # Compute indices of sampled pixels
        xx = (sampled_indices[:, 1] - CX) / FX
        yy = (sampled_indices[:, 0] - CY) / FY
        depth_z = depth[sampled_indices[:, 0], sampled_indices[:, 1]]

        # Transform point cloud
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]

        # Remove points at camera origin
        A = torch.abs(torch.round(pts, decimals=4))
        B = torch.zeros((1, 3)).cuda().float()
        _, idx, counts = torch.cat([A, B], dim=0).unique(
            dim=0, return_inverse=True, return_counts=True
        )
        mask = torch.isin(idx, torch.where(counts.gt(1))[0])
        invalid_pt_idx = mask[: len(A)]
        valid_pt_idx = ~invalid_pt_idx
        pts = pts[valid_pt_idx]

        return pts

    def is_covisible(self, depth_pcd, camera_pose, height, width, threshold=0.9):
        """
        Return true is the covisibility metric between the depth map and given keyframe is greater than the threshold.
        Covisibility is defined by the percentage of points in the depth map visible by the keyframe
        """
        # Get the estimated world2cam of the keyframe
        est_w2c = get_camera_from_tensor(camera_pose)
        # Transform the 3D pointcloud to the keyframe's camera space
        depth_pcd_homo = torch.cat(
            [depth_pcd, torch.ones_like(depth_pcd[:, :1])], dim=1
        )
        transformed_pts = (est_w2c @ depth_pcd_homo.T).T[:, :3]
        # Project the 3D pointcloud to the keyframe's image space
        intrinsics = torch.eye(3).cuda()
        intrinsics[0][2] = self.cfg["cam"]["cx"]
        intrinsics[1][2] = self.cfg["cam"]["cy"]
        intrinsics[0][0] = self.cfg["cam"]["fx"]
        intrinsics[1][1] = self.cfg["cam"]["fy"]
        points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
        points_2d = points_2d.transpose(0, 1)
        points_z = points_2d[:, 2:] + 1e-5
        points_2d = points_2d / points_z
        projected_pts = points_2d[:, :2]
        # Filter out the points that are outside the image
        edge = 0
        mask = (
            (projected_pts[:, 0] < width - edge)
            * (projected_pts[:, 0] > edge)
            * (projected_pts[:, 1] < height - edge)
            * (projected_pts[:, 1] > edge)
        )
        mask = mask & (points_z[:, 0] > 0)
        # Compute the percentage of points that are inside the image
        percent_inside = mask.sum() / projected_pts.shape[0]

        return percent_inside > threshold

    def update_covisibility_graph(self, key):
        """
        Update the covisibility graph by linking current keyframe to
        other keyframes.
        Assume kf is already in self.keyframes
        """
        # Get the depth rendering of key
        with torch.no_grad():
            result = self.renderer.render(
                self.gaussians,
                camera_pose=self.keyframes[key].pose,
            )
            depth = result["depth"][0, :, :]
            silhouette = result["depth"][1, :, :]
            presence_sil_mask = silhouette > 0.99
            depth[~presence_sil_mask] = 0
            valid_depth_indices = torch.where(depth > 0)
            valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
            sampled_indices = valid_depth_indices
            # Back Project the selected pixels to 3D Pointcloud
            w2c = get_camera_from_tensor(self.keyframes[key].pose)
            pts = self.get_depth_pointcloud(depth, w2c, sampled_indices)
            height, width = depth.shape

        # Loop over all keyframes.
        for keyframeid, keyframe in enumerate(self.keyframes[:-1]):
            # If covisible, add edge in the covisibility graph
            if self.is_covisible(
                pts,
                keyframe.pose,
                height,
                width,
                threshold=self.cfg["mapping"]["kf_covisibility"],
            ):
                self.covisibility_graph[key].add(keyframeid)
                self.covisibility_graph[keyframeid].add(key)

    def get_covisible_set(self, idx, camera_pose, gt_color, gt_depth=None, N=1):
        """
        Get the set of keyframes covisible with the given frame/current KF.
        Covisibility is defined in self.is_covisible
        Returns:
            The set of keyframes to optimizer upon. Note that -1 represents the current frame
        """
        if idx == 0:
            return [], []

        # SplaTAM selects overlapping keyframes based off depth map
        if self.cfg["method"].lower() == "splatam":
            if self.cfg["use_gt_depth"]:
                depth = gt_depth
            else:
                result = self.renderer.render(
                    self.gaussians,
                    camera_pose=camera_pose,
                )
                depth = result["depth"][0, :, :]
                silhouette = result["depth"][1, :, :]
                presence_sil_mask = silhouette > 0.99
                depth[~presence_sil_mask] = 0

            pixels = 1600
            # Randomly Sample Pixel Indices from valid depth pixels

            height, width = depth.shape
            valid_depth_indices = torch.where(depth > 0)
            valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
            indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
            sampled_indices = valid_depth_indices[indices]

            # Back Project the selected pixels to 3D Pointcloud
            w2c = get_camera_from_tensor(camera_pose)
            pts = self.get_depth_pointcloud(depth, w2c, sampled_indices)

            list_keyframe = []
            for keyframeid, keyframe in enumerate(self.keyframes[:-1]):
                # Get the estimated world2cam of the keyframe
                est_w2c = get_camera_from_tensor(keyframe.pose)
                # Transform the 3D pointcloud to the keyframe's camera space
                pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
                transformed_pts = (est_w2c @ pts4.T).T[:, :3]
                # Project the 3D pointcloud to the keyframe's image space
                intrinsics = torch.eye(3).cuda()
                intrinsics[0][2] = self.cfg["cam"]["cx"]
                intrinsics[1][2] = self.cfg["cam"]["cy"]
                intrinsics[0][0] = self.cfg["cam"]["fx"]
                intrinsics[1][1] = self.cfg["cam"]["fy"]
                points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
                points_2d = points_2d.transpose(0, 1)
                points_z = points_2d[:, 2:] + 1e-5
                points_2d = points_2d / points_z
                projected_pts = points_2d[:, :2]
                # Filter out the points that are outside the image
                edge = 20
                mask = (
                    (projected_pts[:, 0] < width - edge)
                    * (projected_pts[:, 0] > edge)
                    * (projected_pts[:, 1] < height - edge)
                    * (projected_pts[:, 1] > edge)
                )
                mask = mask & (points_z[:, 0] > 0)
                # Compute the percentage of points that are inside the image
                percent_inside = mask.sum() / projected_pts.shape[0]
                list_keyframe.append(
                    {"id": keyframeid, "percent_inside": percent_inside}
                )

            # Sort the keyframes based on the percentage of points that are inside the image
            list_keyframe = sorted(
                list_keyframe, key=lambda i: i["percent_inside"], reverse=True
            )
            # Select the keyframes with percentage of points inside the image > 0
            selected_keyframe_list = [
                keyframe_dict["id"]
                for keyframe_dict in list_keyframe
                if keyframe_dict["percent_inside"] > 0.0
            ]
            selected_keyframe_list = list(
                np.random.permutation(np.array(selected_keyframe_list))[
                    : self.cfg["mapping"]["kf_window_size"] - 2
                ]
            )

            # Add current frame and most recent keyframe to the list
            if len(self.keyframes) > 0:
                # Add last keyframe to the selected keyframes
                selected_keyframe_list.append(len(self.keyframes) - 1)

            selected_time_idx = [
                self.keyframes[frame_idx].idx for frame_idx in selected_keyframe_list
            ]

            return selected_keyframe_list, selected_time_idx
        else:
            curr_kf_idx = len(self.keyframes) - 1
            covisible = set([curr_kf_idx])  # Set of covisible KFs

            for _ in range(N):  # over N levels of covisibility
                search_space = covisible.copy()
                for k in search_space:  # for each keyframe in the search space
                    # Get covisible keyframes
                    neighbors = set(self.covisibility_graph[k]) - covisible
                    if not neighbors:  # exit early if no new neighbors
                        continue
                    covisible.update(neighbors)
                if search_space == covisible:  # break if no work done
                    break

            # TODO: if doing BA, fix the poses of the edge KFs to ensure smoothness

            # Convert covisible set to list
            covisible.remove(curr_kf_idx)
            selected_keyframe_list = list(covisible)
            # Limit window size
            selected_keyframe_list = list(
                np.random.permutation(np.array(selected_keyframe_list))[
                    : self.cfg["mapping"]["kf_window_size"] - 2
                ]
            )
            # Add curr KF back
            selected_keyframe_list.append(curr_kf_idx)
            selected_time_idx = [
                self.keyframes[frame_idx].idx for frame_idx in selected_keyframe_list
            ]

            return selected_keyframe_list, selected_time_idx

    def get_pointcloud(
        self,
        color,
        depth,
        w2c,
        transform_pts=True,
        mask=None,
        compute_mean_sq_dist=False,
        mean_sq_dist_method="projective",
        add_noise=False,
        use_median=False,
        noise_scale=0.5,
    ):
        """
        Compute a new Gaussian for each empty pixel in the given frame.
        """
        width, height = color.shape[2], color.shape[1]
        FX = self.cfg["cam"]["fx"]
        FY = self.cfg["cam"]["fy"]
        CX = self.cfg["cam"]["cx"]
        CY = self.cfg["cam"]["cy"]

        # Compute indices of pixels
        x_grid, y_grid = torch.meshgrid(
            torch.arange(width).cuda().float(),
            torch.arange(height).cuda().float(),
            indexing="xy",
        )
        xx = (x_grid - CX) / FX
        yy = (y_grid - CY) / FY
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        depth_z = depth.reshape(-1)

        if add_noise and torch.any(mask):
            if use_median:
                if torch.any(~mask):
                    median_z = torch.median(depth_z[~mask])
                    std_z = torch.sqrt(torch.var(depth_z[~mask]))
                else:
                    median_z = 1
                    std_z = 1
                depth_z = torch.normal(
                    median_z * torch.ones_like(depth_z), noise_scale * std_z
                )
            else:
                std_z = torch.sqrt(torch.var(depth_z[mask]))
                depth_z = torch.normal(depth_z, noise_scale * std_z)

        # Initialize point cloud
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        if transform_pts:
            # Transform points to world frame
            pix_ones = torch.ones(height * width, 1).cuda().float()
            pts4 = torch.cat((pts_cam, pix_ones), dim=1)
            c2w = torch.inverse(w2c)
            pts = (c2w @ pts4.T).T[:, :3]
        else:
            pts = pts_cam

        # Compute mean squared distance for initializing the scale of the Gaussians
        if compute_mean_sq_dist:
            if mean_sq_dist_method == "projective":
                # Projective Geometry (this is fast, farther -> larger radius)
                scale_gaussian = depth_z / ((FX + FY) / 2)
                mean3_sq_dist = scale_gaussian**2
            else:
                raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")

        # Colorize point cloud
        cols = torch.permute(color, (1, 2, 0)).reshape(
            -1, 3
        )  # (C, H, W) -> (H, W, C) -> (H * W, C)
        point_cld = torch.cat((pts, cols), -1)

        # Select points based on mask
        if mask is not None:
            point_cld = point_cld[mask]
            if compute_mean_sq_dist:
                mean3_sq_dist = mean3_sq_dist[mask]

        if compute_mean_sq_dist:
            return point_cld, mean3_sq_dist
        else:
            return point_cld

    def initialize_new_gaussians(
        self, idx, camera_pose, gt_color, gt_depth=None, est_depth=None
    ):
        """
        Add new Gaussians to the map based off undiscovered areas of the given frame
        Reset the map optimizer after doing so.
        """
        depth = gt_depth if self.cfg["use_gt_depth"] else est_depth
        if self.cfg["method"].lower() == "splatam":
            if idx == 0 and "iteration" not in self.cfg:
                # If this is the first frame
                print("First frame. Initializing Gaussians")
                non_presence_mask = torch.ones(
                    depth.shape, dtype=bool, device=self.cfg["device"]
                ).reshape(-1)
            else:
                # Silhouette Rendering
                result = self.renderer.render(
                    self.gaussians,
                    camera_pose=camera_pose,
                )

                silhouette = result["depth"][1, :, :]
                non_presence_sil_mask = silhouette < 0.5
                # Check for new foreground objects by using GT depth

                render_depth = result["depth"][0, :, :]
                depth_error = torch.abs(depth - render_depth) * (depth > 0)
                non_presence_depth_mask = (render_depth > depth) * (
                    depth_error > 50 * depth_error.median()
                )
                # Determine non-presence mask
                non_presence_mask = non_presence_sil_mask | non_presence_depth_mask

                # Flatten mask
                non_presence_mask = non_presence_mask.reshape(-1)

            # Get the new frame Gaussians based on the Silhouette
            if torch.sum(non_presence_mask) > 0:
                # Get the new pointcloud in the world frame
                curr_w2c = get_camera_from_tensor(camera_pose)
                valid_depth_mask = depth > 0
                non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
                new_pt_cld, mean3_sq_dist = self.get_pointcloud(
                    gt_color,
                    depth,
                    curr_w2c,
                    mask=non_presence_mask,
                    compute_mean_sq_dist=True,
                )

                # Add new params to optimizer
                new_rgb = new_pt_cld[:, 3:6].float().cuda()
                fused_color = RGB2SH(new_pt_cld[:, 3:6].float().cuda())
                new_features = (
                    torch.zeros(
                        (
                            fused_color.shape[0],
                            3,
                            (self.gaussians.max_sh_degree + 1) ** 2,
                        )
                    )
                    .float()
                    .cuda()
                )
                new_features[:, :3, 0] = fused_color
                new_features[:, 3:, 1:] = 0.0
                # Identity rotation
                new_rots = torch.zeros((new_pt_cld.shape[0], 4), device="cuda")
                new_rots[:, 0] = 1
                # 0.5 opacity
                new_opacities = torch.zeros(
                    (new_pt_cld.shape[0], 1), dtype=torch.float, device="cuda"
                )
                # Spherical pixel-wide scaling
                new_scaling = torch.log(torch.sqrt(mean3_sq_dist))[..., None].repeat(
                    1, 3
                )
                self.gaussians.densification_postfix(
                    new_xyz=new_pt_cld[:, :3],
                    new_features_dc=new_features[:, :, 0:1]
                    .transpose(1, 2)
                    .contiguous(),
                    new_features_rest=new_features[:, :, 1:]
                    .transpose(1, 2)
                    .contiguous(),
                    new_opacities=new_opacities,
                    new_scaling=new_scaling,
                    new_rotation=new_rots,
                    new_rgb=new_rgb,
                )

                # return new points mask
                num_new_pts = new_pt_cld.shape[0]
                new_gaussians_mask = torch.zeros_like(
                    self.gaussians.get_xyz[:, 0],
                    dtype=torch.bool,
                    device=self.cfg["device"],
                )
                new_gaussians_mask[-num_new_pts:] = True
            else:
                new_gaussians_mask = None

            return new_gaussians_mask, non_presence_mask.reshape(depth.shape)
        else:
            if idx == 0 and "iteration" not in self.cfg:
                # If this is the first frame
                print("First frame. Initializing Gaussians")
                non_presence_mask = torch.ones(
                    depth.shape, dtype=bool, device=self.cfg["device"]
                ).reshape(-1)

                render_mask = torch.zeros(
                    depth.shape, dtype=bool, device=self.cfg["device"]
                ).reshape(-1)
            else:
                # Silhouette Rendering
                result = self.renderer.render(
                    self.gaussians,
                    camera_pose=camera_pose,
                )

                silhouette = result["depth"][1, :, :]
                non_presence_sil_mask = silhouette < 0.5

                # Check for new foreground objects by using GT depth
                render_depth = result["depth"][0, :, :]
                depth_error = torch.abs(depth - render_depth) * (depth > 0)
                non_presence_depth_mask = depth_error > 10 * depth_error.median()

                # Determine non-presence mask
                non_presence_mask = non_presence_sil_mask | non_presence_depth_mask

                # Flatten mask
                non_presence_mask = non_presence_mask.reshape(-1)

            # Get the new frame Gaussians based on the Silhouette
            # Get the new pointcloud in the world frame
            curr_w2c = get_camera_from_tensor(camera_pose)
            valid_depth_mask = depth > 0
            non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
            new_pt_cld, mean3_sq_dist = self.get_pointcloud(
                gt_color,
                depth,
                curr_w2c,
                mask=non_presence_mask,
                compute_mean_sq_dist=True,
            )

            # Add new params to optimizer
            new_rgb = new_pt_cld[:, 3:6].float().cuda()
            fused_color = RGB2SH(new_pt_cld[:, 3:6].float().cuda())
            new_features = (
                torch.zeros(
                    (
                        fused_color.shape[0],
                        3,
                        (self.gaussians.max_sh_degree + 1) ** 2,
                    )
                )
                .float()
                .cuda()
            )
            new_features[:, :3, 0] = fused_color
            new_features[:, 3:, 1:] = 0.0
            # Identity rotation
            new_rots = torch.zeros((new_pt_cld.shape[0], 4), device="cuda")
            new_rots[:, 0] = 1
            # 0.5 opacity
            new_opacities = torch.zeros(
                (new_pt_cld.shape[0], 1), dtype=torch.float, device="cuda"
            )
            # Spherical pixel-wide scaling
            new_scaling = torch.log(torch.sqrt(mean3_sq_dist))[..., None].repeat(1, 3)
            self.gaussians.densification_postfix(
                new_xyz=new_pt_cld[:, :3],
                new_features_dc=new_features[:, :, 0:1].transpose(1, 2).contiguous(),
                new_features_rest=new_features[:, :, 1:].transpose(1, 2).contiguous(),
                new_opacities=new_opacities,
                new_scaling=new_scaling,
                new_rotation=new_rots,
                new_rgb=new_rgb,
            )

            # Return new points mask
            num_new_pts = new_pt_cld.shape[0]
            new_gaussians_mask = torch.zeros_like(
                self.gaussians.get_xyz[:, 0],
                dtype=torch.bool,
                device=self.cfg["device"],
            )
            new_gaussians_mask[-num_new_pts:] = True

            return new_gaussians_mask, non_presence_mask.reshape(depth.shape)

    def get_covisible_gaussians(self, keyframe_idx_list, curr_camera_tensor, min_kf=2):
        """
        Given a covisible set, return a mask of Gaussians covisible by at least min_kf
        keyframes.
        """
        with torch.no_grad():
            visibility_sum = torch.zeros_like(
                self.gaussians.get_xyz[:, 0], device=self.cfg["device"]
            )
            # Render to get visibility_filter
            for keyframe_idx in keyframe_idx_list:
                if keyframe_idx == -1:
                    # -1 means use current frame
                    camera_pose = curr_camera_tensor
                else:
                    keyframe = self.keyframes[keyframe_idx]
                    camera_pose = keyframe.pose

                # Get visible Gaussians
                visibility_mask = self.renderer.render(
                    self.gaussians,
                    camera_pose=camera_pose,
                )["visibility_filter"]

                visibility_sum += visibility_mask.int()

        return visibility_sum >= 2

    def optimize_map(
        self,
        idx,
        num_iter,
        keyframe_idx_list,
        new_gaussians_mask,
        curr_camera_tensor,
        curr_gt_color,
        curr_gt_depth=None,
        curr_est_depth=None,
    ):
        """
        Do iterations of joint map + pose optimization over the keyframe window

        Args:
            num_iter: number of iterations to do
            keyframe_idx_list (List): list of Keyframes indices to optimize
            curr_camera_tensor_q: rotation of the current frame's pose estimate
            curr_camera_tensor_T: translation of the current frame's pose estimate
            curr_gt_color: current frame's ground truth RGB image
            curr_gt_depth: current frame's ground truth depth
            curr_est_depth: current frame's monocular estimated depth
        """
        # If no iterations, do nothing
        if num_iter == 0:
            return

        # Setup optimizers
        # Create differentiable version of keyframe poses
        curr_camera_tensor_q = curr_camera_tensor[:4]
        curr_camera_tensor_T = curr_camera_tensor[4:]
        if self.cfg["mapping"]["do_BA"] and idx > 0:
            camera_tensor_q_list = [curr_camera_tensor_q.requires_grad_()]
            camera_tensor_T_list = [curr_camera_tensor_T.requires_grad_()]
            # TODO: Keep the first frame fixed
            for keyframe_idx in keyframe_idx_list:
                camera_tensor_q_list.append(
                    self.keyframes[keyframe_idx].pose[:4].requires_grad_()
                )
                camera_tensor_T_list.append(
                    self.keyframes[keyframe_idx].pose[4:].requires_grad_()
                )
            # TODO: optimize only the poses of the current frame and current KF
            # if len(self.keyframes) > 1:
            #     camera_tensor_q_list.append(
            #         self.keyframes[-1].pose[:4].requires_grad_()
            #     )
            #     camera_tensor_T_list.append(
            #         self.keyframes[-1].pose[4:].requires_grad_()
            #     )
            pose_param_group = [
                {
                    "params": camera_tensor_q_list,
                    "lr": self.cfg["mapping"]["cam_q_lr"],
                    "name": "cam_rot",
                },
                {
                    "params": camera_tensor_T_list,
                    "lr": self.cfg["mapping"]["cam_t_lr"],
                    "name": "cam_pos",
                },
            ]
            pose_optimizer = torch.optim.Adam(pose_param_group, lr=0.0, eps=1e-15)

            # mask out Gaussians that are not covisible
            optimization_mask = self.get_covisible_gaussians(
                keyframe_idx_list, curr_camera_tensor, 2
            )
            # If new Gaussians were initialized, add these to the optimization mask
            if new_gaussians_mask is not None:
                optimization_mask |= new_gaussians_mask

        keyframe_idx_stack = None

        progress_bar = tqdm(
            range(num_iter), desc=f"Mapping Time Step: {idx}", disable=True
        )

        # Perform optimization iterations
        for iteration in range(num_iter):
            if self.cfg["debug"]["get_runtime_stats"]:
                iter_start_time = time.perf_counter()

            # Pick a random Camera
            # Stack allows each camera to be picked the same amount of times
            if not keyframe_idx_stack:
                keyframe_idx_stack = list(keyframe_idx_list)
            keyframe_idx = keyframe_idx_stack.pop(
                randint(0, len(keyframe_idx_stack) - 1)
            )

            if keyframe_idx == -1:
                # -1 means use current frame
                camera_tensor_q = curr_camera_tensor_q
                camera_tensor_T = curr_camera_tensor_T
                gt_color = curr_gt_color
                gt_depth = curr_gt_depth
                est_depth = curr_est_depth
            else:
                keyframe = self.keyframes[keyframe_idx]
                camera_tensor_q = keyframe.pose[:4]
                camera_tensor_T = keyframe.pose[4:]
                gt_color = keyframe.gt_color
                gt_depth = keyframe.gt_depth
                est_depth = keyframe.est_depth

            # Render
            result = self.renderer.render(
                self.gaussians,
                camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]),
            )

            image = result["render"]
            depth = result["depth"][0, :, :]
            silhouette = result["depth"][1, :, :]
            presence_sil_mask = silhouette > 0.5

            # Loss
            if self.cfg["method"].lower() == "splatam":
                losses = {}  # Loss dictionary
                depth_sq = result["depth"][2, :, :]
                # Depths are rendered via the rasterizer, which takes into account
                # opacity. Thus, when the range of depths present in that pixel is larger,
                # the difference between depth_sq and depth**2 should increase.
                uncertainty = depth_sq - depth**2
                uncertainty = uncertainty.detach()
                nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
                mask = gt_depth > 0
                mask = mask & nan_mask
                mask = mask.detach()
                # Depth Loss
                losses["depth"] = torch.abs(gt_depth - depth)[mask].mean()
                # RGB Loss
                losses["im"] = (1 - self.cfg["mapping"]["lambda_dssim"]) * l1_loss(
                    image, gt_color
                ) + self.cfg["mapping"]["lambda_dssim"] * (1.0 - ssim(image, gt_color))
                loss = losses["depth"] + 0.5 * losses["im"]
            else:
                loss = (1 - self.cfg["mapping"]["lambda_dssim"]) * l1_loss(
                    image, gt_color
                ) + self.cfg["mapping"]["lambda_dssim"] * (1.0 - ssim(image, gt_color))
                if (
                    not self.cfg["use_gt_depth"]
                    and self.cfg["mapping"]["use_depth_estimate_loss"]
                ):
                    loss += self.cfg["mapping"]["pearson_weight"] * pearson_loss(
                        depth, est_depth, invert_estimate=False
                    )
                elif (
                    self.cfg["use_gt_depth"]
                    and self.cfg["mapping"]["use_depth_estimate_loss"]
                ):
                    depth_mask = gt_depth > 0
                    loss += self.cfg["mapping"]["pearson_weight"] * pearson_loss(
                        depth, gt_depth, mask=depth_mask, invert_estimate=False
                    )

            loss.backward()

            # Densification and pruning
            with torch.no_grad():
                if self.cfg["method"].lower() == "splatam":
                    # Splatam does not densify---only prunes
                    if iteration <= 20:
                        if (iteration >= 0) and (iteration % 20 == 0):
                            min_opacity = self.cfg["mapping"]["min_opacity"]
                            self.gaussians.prune(min_opacity, self.camera_extent)

                else:
                    if iteration <= self.cfg["mapping"]["densify_until_iter"]:
                        # Keep track of max radii in image-space for pruning
                        viewspace_point_tensor = result["viewspace_points"]
                        visibility_filter = result["visibility_filter"]
                        radii = result["radii"]
                        self.gaussians.max_radii2D[visibility_filter] = torch.max(
                            self.gaussians.max_radii2D[visibility_filter],
                            radii[visibility_filter],
                        )
                        self.gaussians.add_densification_stats(
                            viewspace_point_tensor, visibility_filter
                        )

                        # Densification & pruning only done in certain intervals
                        if (
                            iteration >= self.cfg["mapping"]["densify_from_iter"]
                            and iteration % self.cfg["mapping"]["pruning_interval"] == 0
                        ):
                            prune_mask = self.gaussians.prune(
                                self.cfg["mapping"]["min_opacity"],
                                self.camera_extent,
                                self.cfg["mapping"]["size_threshold"],
                            )

                            if self.cfg["mapping"]["do_BA"] and idx > 0:
                                optimization_mask = optimization_mask[~prune_mask]

                            # # Periodic densification
                            # if idx % self.cfg["mapping"]["densification_interval"] == 0:
                            #     print("densify")
                            #     self.gaussians.densify_and_prune(
                            #         self.cfg["mapping"]["densify_grad_threshold"],
                            #         self.cfg["mapping"]["min_opacity"],
                            #         self.camera_extent,
                            #         size_threshold,
                            #     )
                            # else:
                            #     self.gaussians.prune(
                            #         self.cfg["mapping"]["min_opacity"],
                            #         self.camera_extent,
                            #         size_threshold,
                            #     )

                # Optimizer step
                if self.cfg["mapping"]["do_BA"] and idx > 0:
                    # Mask out gradients according to optimization_mask
                    for group in self.gaussians.optimizer.param_groups:
                        for param in group["params"]:
                            if param.grad is not None:
                                param.grad[~optimization_mask, :] = 0

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if self.cfg["mapping"]["do_BA"] and idx > 0:
                    pose_optimizer.step()
                    pose_optimizer.zero_grad(set_to_none=True)

                # Logging
                if self.cfg["debug"]["get_runtime_stats"]:
                    iter_end_time = time.perf_counter()
                    self.mapping_time_sum += iter_end_time - iter_start_time
                    self.mapping_iter_count += 1
                progress_bar.update(1)
        progress_bar.close()

    def run_frame(self, idx, gt_color, gt_depth=None, est_depth=None, imu_meas=None):
        """
        Run map optimization for the current frame over the covisibility window.
        If the current frame is eligible for keyframe, add it to keyframe list.
        Args:
            idx: The frame index
            gt_color: The ground truth RGB image
            gt_depth: The ground truth depth image
        """
        camera_pose = self.estimate_pose_list[idx]

        with torch.no_grad():
            new_points_vis_mask = None
            new_gaussians_mask = None
            # Get the current covisibility window
            keyframe_idx_list, keyframe_time_idx_list = self.get_covisible_set(
                idx,
                camera_pose,
                gt_color,
                gt_depth,
                N=self.cfg["mapping"]["covisibility_level"],
            )
            # Add current frame to the selected keyframes
            keyframe_idx_list.append(-1)
            keyframe_time_idx_list.append(idx)

            # Check if current frame should be added as a keyframe
            if self.need_new_keyframe(idx, camera_pose, gt_color, gt_depth, est_depth):
                # Add new Gaussians in current frame
                (
                    new_gaussians_mask,
                    new_points_vis_mask,
                ) = self.initialize_new_gaussians(
                    idx, camera_pose, gt_color, gt_depth, est_depth
                )
                new_kf = self.add_keyframe(
                    idx, camera_pose, gt_color, gt_depth, est_depth
                )

                if self.cfg["debug"]["save_keyframes"]:
                    # Save images
                    render_path = os.path.join(self.cfg["outputdir"], "keyframes")
                    os.makedirs(render_path, exist_ok=True)
                    torchvision.utils.save_image(
                        new_kf.gt_color,
                        os.path.join(
                            render_path, "{0:05d}".format(new_kf.idx) + ".png"
                        ),
                    )

        # Optimize the current covisible set
        self.optimize_map(
            idx,
            self.num_iter,
            keyframe_idx_list,
            new_gaussians_mask,
            camera_pose,
            gt_color,
            gt_depth,
            est_depth,
        )

        return new_points_vis_mask
