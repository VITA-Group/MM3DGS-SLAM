import time

import cv2
import torch
from tqdm import tqdm

from utils.depth_utils import depth_to_rgb
from utils.loss_utils import l1_loss, pearson_loss, rel_pose_loss
from utils.pose_utils import propagate_const_vel, propagate_imu


class Tracker:
    """
    Tracking thread
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
        self.num_iter = self.cfg["tracking"]["iters"]
        if "dynamics_model" in self.cfg["tracking"]:
            self.dyn_model = self.cfg["tracking"]["dynamics_model"]
        else:
            self.dyn_model = None

        # Set up logging
        if self.cfg["debug"]["create_video"]:
            self.video_writer = slam.video_writer_full
        if self.cfg["debug"]["get_runtime_stats"]:
            # Log the tracking per-iteration runtime
            self.tracking_time_sum = 0
            self.tracking_iter_count = 0

    def optimize_cam(
        self,
        idx,
        num_iter,
        optimizer,
        camera_tensor_q,
        camera_tensor_T,
        gt_color,
        gt_depth=None,
        est_depth=None,
    ):
        """
        Do iterations of camera pose optimization. Render depth/color, calculate loss and backpropagation.

        Args:
            num_iter: number of iterations to do
            optimizer (torch.optim): camera optimizer.
            camera_tensor_q (tensor): rotation component of camera tensor.
            camera_tensor_T (tensor): translation component of camera tensor.

        Returns:
            loss (float): The final value of loss.
            image: The final rendered image.
        """
        # If no iterations, do nothing
        if num_iter == 0:
            with torch.no_grad():
                # Render
                result = self.renderer.render(
                    self.gaussians,
                    camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]),
                )

                image = result["render"]
                loss = 0
                return loss, image

        progress_bar = tqdm(
            range(num_iter), desc=f"Tracking Time Step: {idx}", disable=True
        )

        # Copy initial pose estimate (for IMU loss)
        initial_pose = torch.cat([camera_tensor_q, camera_tensor_T]).clone().detach()

        # Keep track of best pose candidate
        candidate_q = camera_tensor_q.clone().detach()
        candidate_T = camera_tensor_T.clone().detach()
        current_min_loss = float(1e20)

        for iteration in range(num_iter):
            if self.cfg["debug"]["get_runtime_stats"]:
                iter_start_time = time.perf_counter()

            # Render
            result = self.renderer.render(
                self.gaussians,
                camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]),
            )

            image = result["render"]
            depth = result["depth"][0, :, :]
            silhouette = result["depth"][1, :, :]
            presence_sil_mask = silhouette > 0.99

            # Loss
            if self.cfg["method"].lower() == "splatam":
                losses = {}  # Loss dictionary
                depth_sq = result["depth"][2, :, :]
                uncertainty = depth_sq - depth**2
                uncertainty = uncertainty.detach()
                nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
                mask = gt_depth > 0
                mask = mask & nan_mask
                mask = mask & presence_sil_mask
                mask = mask.detach()
                # Depth Loss
                losses["depth"] = torch.abs(gt_depth - depth)[mask].sum()
                # RGB Loss
                color_mask = torch.tile(mask, (3, 1, 1))
                color_mask = color_mask.detach()
                losses["im"] = torch.abs(gt_color - image)[color_mask].sum()
                loss = losses["depth"] + 0.5 * losses["im"]
            else:
                # loss = l1_loss(image, gt_color, presence_sil_mask)
                loss = torch.abs((image - gt_color))[:, presence_sil_mask].mean()
                if (
                    not self.cfg["use_gt_depth"]
                    and self.cfg["tracking"]["use_depth_estimate_loss"]
                ):
                    loss += self.cfg["tracking"]["pearson_weight"] * pearson_loss(
                        depth, est_depth, mask=presence_sil_mask, invert_estimate=True
                    )
                elif (
                    self.cfg["use_gt_depth"]
                    and self.cfg["tracking"]["use_depth_estimate_loss"]
                ):
                    depth_mask = presence_sil_mask & (gt_depth > 0)
                    loss += self.cfg["tracking"]["pearson_weight"] * pearson_loss(
                        depth, gt_depth, mask=depth_mask, invert_estimate=True
                    )

                if self.cfg["tracking"]["use_imu_loss"]:
                    T_imu_loss, q_imu_loss = rel_pose_loss(
                        torch.cat([camera_tensor_q, camera_tensor_T]),
                        initial_pose,
                    )

                    loss += (
                        self.cfg["tracking"]["imu_T_weight"] * T_imu_loss
                        + self.cfg["tracking"]["imu_q_weight"] * q_imu_loss
                    )

            loss.backward()

            # Optimize
            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if iteration == 0:
                    initial_loss = loss

                if loss < current_min_loss:
                    current_min_loss = loss
                    candidate_q = camera_tensor_q.clone().detach()
                    candidate_T = camera_tensor_T.clone().detach()

                # Logging
                if self.cfg["debug"]["get_runtime_stats"]:
                    iter_end_time = time.perf_counter()
                    self.tracking_time_sum += iter_end_time - iter_start_time
                    self.tracking_iter_count += 1
                progress_bar.update(1)

        # Copy over best candidate pose
        camera_tensor_q = candidate_q
        camera_tensor_T = candidate_T

        progress_bar.close()
        return loss, image

    def run_frame(self, idx, gt_color, gt_depth=None, est_depth=None, imu_meas=None):
        """
        Estimate camera pose for the current frame
        Args:
            idx: The frame index
            gt_color: The ground truth RGB image
            gt_depth: The ground truth depth image
        Returns:
            image: The final rendered image.
        """

        camera_tensor = self.estimate_pose_list[idx - 1].clone().detach()

        # Propagate the camera pose based on dynamics model
        if self.dyn_model is not None:
            if self.dyn_model.lower() == "const_velocity":
                if idx - 2 >= 0:
                    camera_tensor = propagate_const_vel(
                        self.estimate_pose_list[idx - 1],
                        self.estimate_pose_list[idx - 2],
                    )
            elif self.dyn_model.lower() == "imu":
                assert imu_meas is not None, "IMU measurements must be provided"
                if idx - 2 >= 0:
                    camera_tensor = propagate_imu(
                        self.estimate_pose_list[idx - 1],
                        self.estimate_pose_list[idx - 2],
                        imu_meas,
                        self.tf["c2i"],
                        self.tstamps[idx - 1] - self.tstamps[idx - 2],
                        # 1 / self.cfg["cam"]["fps"] * self.cfg["stride"],
                        1 / 100.0,
                    )
                else:
                    # Assume 0 velocity at the start
                    camera_tensor = propagate_imu(
                        self.estimate_pose_list[idx - 1],
                        self.estimate_pose_list[idx - 1],
                        imu_meas,
                        self.tf["c2i"],
                        1.0,
                        1 / 100.0,
                    )
            else:
                raise ValueError(f"Unknown dynamics model {self.dyn_model}")

        # Create optimizer
        camera_tensor_T = camera_tensor[-3:].requires_grad_()
        camera_tensor_q = camera_tensor[:4].requires_grad_()
        pose_optimizer = torch.optim.Adam(
            [
                {
                    "params": [camera_tensor_T],
                    "lr": self.cfg["tracking"]["position_lr"],
                },
                {
                    "params": [camera_tensor_q],
                    "lr": self.cfg["tracking"]["rotation_lr"],
                },
            ]
        )

        # Optimize pose
        loss, image = self.optimize_cam(
            idx,
            self.num_iter,
            pose_optimizer,
            camera_tensor_q,
            camera_tensor_T,
            gt_color,
            gt_depth,
            est_depth,
        )

        # Save pose estimate
        with torch.no_grad():
            self.estimate_pose_list[idx] = (
                torch.cat([camera_tensor_q, camera_tensor_T]).clone().detach()
            )

        return image
