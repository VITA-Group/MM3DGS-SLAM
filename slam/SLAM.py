import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import load_config
from gradslam_datasets import ReplicaDataset, TUMDataset, UTMMDataset
from slam.gaussian_model import GaussianModel
from slam.mapper import Mapper
from slam.renderer import Renderer
from slam.tracker import Tracker
from utils.depth_utils import depth_to_rgb, get_dpt, get_scale_shift
from utils.eval_utils import evaluate_ate_rmse, evaluate_image_quality
from utils.pose_utils import (
    get_camera_from_tensor,
    get_tensor_from_camera,
    preintegrate_imu,
)


def get_dataset_type(name):
    if name.lower() == "replica":
        return ReplicaDataset
    elif name.lower() == "tum":
        return TUMDataset
    elif name.lower() == "utmm":
        return UTMMDataset
    else:
        raise ValueError(f"Unknown dataset {name}")


class SLAM:
    def __init__(self, cfg):
        """Load dataset, initialize Gaussians, instantiate mapping and tracking processes"""
        # Load config
        self.cfg = cfg
        self.device = self.cfg["device"]
        self.use_imu = self.cfg["tracking"]["dynamics_model"].lower() == "imu"

        # Load dataset
        end_idx = -1 if "early_stop_idx" not in self.cfg else self.cfg["early_stop_idx"]
        self.dataset = get_dataset_type(self.cfg["dataset"])(
            config_dict=self.cfg,
            basedir=self.cfg["inputdir"],
            sequence=self.cfg["scene"],
            start=self.cfg["start_idx"],
            end=end_idx,
            stride=self.cfg["stride"],
            desired_height=self.cfg["desired_height"],
            desired_width=self.cfg["desired_width"],
            device=self.cfg["device"],
            relative_pose=True,
            ignore_bad=False,
            use_train_split=True,
        )
        self.n_img = len(self.dataset)

        # Get the modified intrinsics
        _, _, intrinsics, _, _ = self.dataset[0]
        self.cfg["cam"]["cx"] = intrinsics[0, 2]
        self.cfg["cam"]["cy"] = intrinsics[1, 2]
        self.cfg["cam"]["fx"] = intrinsics[0, 0]
        self.cfg["cam"]["fy"] = intrinsics[1, 1]

        # Load timestamps
        if self.use_imu:
            self.tstamps = self.dataset.tstamps

        # Load transforms
        self.tf = {}
        if self.use_imu:
            self.tf["c2i"] = self.dataset.get_c2i_tf()

        # Create output directories
        self.output = self.cfg["outputdir"]
        os.makedirs(self.output, exist_ok=True)

        # Initialize global variables
        self.gaussians = GaussianModel(self.cfg)
        self.estimate_pose_list = torch.zeros(
            (self.n_img, 7), device=self.cfg["device"]
        )
        # Load checkpoint
        if "iteration" in self.cfg:
            self.gaussians.load_ply(
                os.path.join(
                    self.cfg["outputdir"],
                    "point_cloud",
                    "iteration_" + str(self.cfg["iteration"]),
                    "point_cloud.ply",
                )
            )
            results = np.load(
                os.path.join(self.cfg["outputdir"], "results.npz"), allow_pickle=True
            )
            self.estimate_pose_list = torch.tensor(
                results["pose_est"], device=self.cfg["device"]
            )

        self.gaussians.training_setup()
        self.gt_pose_list = torch.zeros((self.n_img, 7), device=self.cfg["device"])

        self.renderer = Renderer(self.cfg)

        if not self.cfg["use_gt_depth"]:
            # If not using GT depth, initialize monocular depth estimator
            self.dpt = get_dpt(self.cfg["dpt_model"], self.cfg["device"])

        # Logging
        if self.cfg["debug"]["create_video"]:
            video_filename = os.path.join(self.cfg["outputdir"], "debug_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                video_filename,
                fourcc,
                self.cfg["cam"]["fps"],
                (
                    self.cfg["desired_width"] * 3,
                    self.cfg["desired_height"] * 2,
                ),
            )
            video_filename_full = os.path.join(
                self.cfg["outputdir"], "debug_video_full.mp4"
            )
            self.video_writer_full = cv2.VideoWriter(
                video_filename_full,
                fourcc,
                self.cfg["cam"]["fps"],
                (
                    self.cfg["desired_width"] * 3,
                    self.cfg["desired_height"] * (1 if self.cfg["use_gt_depth"] else 2),
                ),
            )

        # Initialize Mapping and Tracking threads
        self.mapper = Mapper(self)
        self.tracker = Tracker(self)

    def get_scene_radius(self, gt_depth):
        return torch.max(gt_depth) / self.cfg["scene_radius_depth_ratio"]

    def render(self):
        pbar = tqdm(range(len(self.dataset)))
        rendering_time_sum = 0
        rendering_iter_count = 0

        render_path = os.path.join(self.cfg["outputdir"], "render")
        os.makedirs(render_path, exist_ok=True)
        for idx in tqdm(pbar):
            if idx % 50 != 0:
                continue
            # Preprocess data
            gt_color, gt_depth, _, gt_c2w, _ = self.dataset[idx]
            gt_depth = gt_depth.squeeze()

            gt_w2c = torch.inverse(gt_c2w)
            gt_color = (
                gt_color.permute(2, 0, 1) / 255
            )  # H,W,C -> C,H,W to match Gaussian renderer

            camera_pose = self.estimate_pose_list[idx]
            # camera_pose = get_tensor_from_camera(gt_w2c)
            iter_start_time = time.perf_counter()
            result = self.renderer.render(
                self.gaussians,
                camera_pose=camera_pose,
            )
            iter_end_time = time.perf_counter()
            rendering_time_sum += iter_end_time - iter_start_time
            rendering_iter_count += 1
            image = result["render"]
            depth = result["depth"][0, :, :]

            torchvision.utils.save_image(
                torch.cat(
                    [image, depth_to_rgb(depth)],
                    dim=1,
                ),
                os.path.join(render_path, "render" + "{0:05d}".format(idx) + ".png"),
            )
            torchvision.utils.save_image(
                torch.cat(
                    [gt_color, depth_to_rgb(gt_depth)],
                    dim=1,
                ),
                os.path.join(render_path, "gt" + "{0:05d}".format(idx) + ".png"),
            )
        rendering_time_per_iter = rendering_time_sum / rendering_iter_count
        print(f"\nAverage Rendering Time: {rendering_time_per_iter*1000} ms")

    def evaluate_images(self, last_idx):
        psnr_list = []
        ssim_list = []
        lpips_list = []
        print("Evaluating image rendering quality")

        for idx in tqdm(range(last_idx)):
            # Skip frames if not eval_every
            if idx != 0 and (idx + 1) % self.cfg["eval_every"] != 0:
                continue

            # Preprocess data
            gt_color, gt_depth, _, gt_c2w, _ = self.dataset[idx]
            if self.cfg["use_gt_depth"]:
                gt_depth = gt_depth.squeeze()

            gt_w2c = torch.inverse(gt_c2w)
            gt_color = (
                gt_color.permute(2, 0, 1) / 255
            )  # H,W,C -> C,H,W to match Gaussian renderer

            camera_pose = self.estimate_pose_list[idx]
            # camera_pose = get_tensor_from_camera(gt_w2c)
            image = self.renderer.render(
                self.gaussians,
                camera_pose=camera_pose,
            )["render"]

            psnr, ssim, lpips = evaluate_image_quality(image, gt_color, gt_depth)

            psnr_list.append(psnr.detach().cpu().numpy())
            ssim_list.append(ssim.detach().cpu().numpy())
            lpips_list.append(lpips.detach().cpu().numpy())

        return psnr_list, ssim_list, lpips_list

    def save_video_frame(self, idx, gt_color, gt_depth, est_depth_scaled, name):
        with torch.no_grad():
            camera_pose = self.estimate_pose_list[idx]
            result = self.renderer.render(
                self.gaussians,
                camera_pose=camera_pose,
            )
            image = result["render"]
            depth = result["depth"][0, :, :]

            vid_image = torch.cat(
                [
                    gt_color,
                    image,
                    torch.abs(image - gt_color),
                ],
                dim=2,
            )
            if not self.cfg["use_gt_depth"]:
                depth_image = torch.cat(
                    [
                        depth_to_rgb(gt_depth),
                        depth_to_rgb(depth),
                        depth_to_rgb(est_depth_scaled),
                    ],
                    dim=2,
                )
            else:
                depth_image = torch.cat(
                    [
                        depth_to_rgb(gt_depth),
                        depth_to_rgb(depth),
                        depth_to_rgb(gt_depth),
                    ],
                    dim=2,
                )

            vid_image = torch.cat([vid_image, depth_image], dim=1)
            # VideoWriter expects H,W,C array in BGR format
            vid_image_cv = (
                (vid_image * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            )  # C,H,W -> H,W,C
            vid_image_cv = cv2.cvtColor(vid_image_cv, cv2.COLOR_RGB2BGR)
            self.video_writer.write(vid_image_cv)

            # # Save images
            # print("Saving image " + "{0:0d}".format(idx))
            # render_path = os.path.join(self.cfg["outputdir"], "images")
            # os.makedirs(render_path, exist_ok=True)
            # torchvision.utils.save_image(
            #     vid_image,
            #     os.path.join(render_path, "{0:05d}".format(idx) + name + ".png"),
            # )

    def save_map(self, iteration):
        point_cloud_path = os.path.join(
            self.cfg["outputdir"], "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        print("Map saved to " + os.path.join(point_cloud_path, "point_cloud.ply"))

    def save_results(self, last_idx):
        results = {}

        self.estimate_pose_list = self.estimate_pose_list[:last_idx, :]
        self.gt_pose_list = self.gt_pose_list[:last_idx, :]

        # Save pose estimates and GT pose
        results["pose_est"] = self.estimate_pose_list.detach().cpu().numpy()
        results["pose_gt"] = self.gt_pose_list.detach().cpu().numpy()

        # Save video
        if self.cfg["debug"]["create_video"]:
            self.video_writer.release()
            self.video_writer_full.release()

        # Save keyframes
        results["keyframes"] = [
            {
                "idx": kf.idx,
                "gt_color": kf.gt_color,
                "est_pose": kf.pose,
                "gt_depth": kf.gt_depth,
                "est_depth": kf.est_depth,
            }
            for kf in self.mapper.keyframes
        ]

        # Calculate ATE RMSE
        cam_centers = torch.zeros_like(self.estimate_pose_list)
        for i_t in range(len(self.estimate_pose_list)):
            cam_centers[i_t, :] = get_tensor_from_camera(
                torch.inverse(get_camera_from_tensor(self.estimate_pose_list[i_t]))
            )
        gt_centers = torch.zeros_like(self.gt_pose_list)
        for i_t in range(len(self.gt_pose_list)):
            gt_centers[i_t, :] = get_tensor_from_camera(
                torch.inverse(get_camera_from_tensor(self.gt_pose_list[i_t]))
            )

        # ATE fixed around world frame
        _, ate_rmse_c2w = evaluate_ate_rmse(cam_centers, gt_centers, method="umeyama")
        # ATE fixed around camera frame
        _, ate_rmse_w2c = evaluate_ate_rmse(
            self.estimate_pose_list, self.gt_pose_list, method="umeyama"
        )

        results["ate_rmse"] = ate_rmse_w2c
        print(f"Average Trajectory Error RMSE: {ate_rmse_w2c} m")

        # Calculate PSNR/SSIM/LPIPS
        psnr_list, ssim_list, lpips_list = self.evaluate_images(last_idx)
        results["psnr_list"] = psnr_list
        results["ssim_list"] = ssim_list
        results["lpips_list"] = lpips_list
        print("  PSNR : {:>12.7f}".format(np.array(psnr_list).mean(), ".5"))
        print("  SSIM : {:>12.7f}".format(np.array(ssim_list).mean(), ".5"))
        print("  LPIPS: {:>12.7f}".format(np.array(lpips_list).mean(), ".5"))

        # Calculate runtime optimization stats
        if self.cfg["debug"]["get_runtime_stats"]:
            if self.tracker.tracking_iter_count == 0:
                self.tracker.tracking_iter_count = 1
            if self.mapper.mapping_iter_count == 0:
                self.mapper.mapping_iter_count = 1
            tracking_time_per_iter = (
                self.tracker.tracking_time_sum / self.tracker.tracking_iter_count
            )
            mapping_time_per_iter = (
                self.mapper.mapping_time_sum / self.mapper.mapping_iter_count
            )
            print(
                f"\nAverage Tracking/Iteration Time: {tracking_time_per_iter*1000} ms"
            )
            print(f"Average Mapping/Iteration Time: {mapping_time_per_iter*1000} ms")

            results["avg_tracking_it_time"] = tracking_time_per_iter * 1000
            results["avg_mapping_it_time"] = mapping_time_per_iter * 1000

        np.savez(os.path.join(self.cfg["outputdir"], "results"), **results)
        print("Results saved to " + os.path.join(self.cfg["outputdir"], "results.npz"))

    def run(self):
        """Run SLAM"""
        pbar = tqdm(range(len(self.dataset)))
        print("Method: " + self.cfg["method"])

        last_idx = 0
        try:
            for idx in pbar:
                # Preprocess data
                gt_color, gt_depth, _, gt_c2w, imu_meas = self.dataset[idx]
                gt_depth = gt_depth.squeeze()

                gt_w2c = torch.inverse(gt_c2w)
                gt_color = (
                    gt_color.permute(2, 0, 1) / 255
                )  # H,W,C -> C,H,W to match Gaussian renderer

                est_depth = None
                est_depth_scaled = None
                if not self.cfg["use_gt_depth"]:
                    # If GT depth not available, get the monocular depth estimate
                    est_depth = self.dpt.estimate_depth(gt_color)

                # Initialize first time step
                if idx == 0:
                    self.estimate_pose_list[idx] = get_tensor_from_camera(gt_w2c)
                else:
                    # Tracking
                    # For each frame after the first frame, run tracker
                    if self.cfg["tracking"]["use_gt_pose"]:
                        self.estimate_pose_list[idx] = get_tensor_from_camera(gt_w2c)
                    else:
                        self.tracker.run_frame(
                            idx, gt_color, gt_depth, est_depth, imu_meas
                        )

                # Scale the depth estimate to align with depth rendering
                if not self.cfg["use_gt_depth"]:
                    with torch.no_grad():
                        if idx == 0 and "iteration" not in self.cfg:
                            if self.cfg["dataset"].lower() == "utmm":
                                # Until visual-inertial initialization sequence is implemented,
                                # Scale first frame depth estimate to GT depth
                                mask = gt_depth > 0
                                scale, shift = get_scale_shift(
                                    est_depth, gt_depth, mask, method="LS"
                                )
                                est_depth_scaled = (
                                    1 / (scale * est_depth + shift).detach()
                                )
                            else:
                                # Arbitrarily scale the first frame depth estimate
                                est_depth_scaled = (
                                    1
                                    / (est_depth + 0.001)
                                    * self.cfg["cam"]["png_depth_scale"]
                                    / 10
                                )

                        else:
                            # Fit depth estimates to the current Gaussians via Least Squares
                            camera_pose = self.estimate_pose_list[idx]
                            result = self.renderer.render(
                                self.gaussians,
                                camera_pose=camera_pose,
                            )
                            render_depth = result["depth"][0, :, :]
                            silhouette = result["depth"][1, :, :]
                            mask = silhouette > 0.99
                            mask = mask & (est_depth > 1e-6)
                            scale, shift = get_scale_shift(
                                est_depth, render_depth, mask, method="LS"
                            )
                            est_depth_scaled = 1 / (scale * est_depth + shift).detach()

                if self.cfg["debug"]["create_video"] and idx > 0:
                    self.save_video_frame(
                        idx, gt_color, gt_depth, est_depth_scaled, "drack"
                    )

                # Mapping
                if idx == 0:
                    # Estimate the camera extent (scene radius). The best way to do this is with the depth measurement
                    if self.cfg["use_gt_depth"]:
                        self.mapper.camera_extent = self.get_scene_radius(gt_depth)
                    else:
                        self.mapper.camera_extent = self.get_scene_radius(
                            est_depth_scaled
                        )

                # Every n frames, run mapper
                new_gaussians_vis_mask = self.mapper.run_frame(
                    idx, gt_color, gt_depth, est_depth_scaled, imu_meas
                )

                # Logging
                with torch.no_grad():
                    self.gt_pose_list[idx] = get_tensor_from_camera(gt_w2c)
                    if self.cfg["debug"]["create_video"]:
                        if new_gaussians_vis_mask is not None:
                            self.save_video_frame(
                                idx,
                                gt_color,
                                gt_depth,
                                new_gaussians_vis_mask.float(),
                                "map",
                            )
                        else:
                            self.save_video_frame(
                                idx, gt_color, gt_depth, est_depth_scaled, "map"
                            )

                    # Save map as checkpoint
                    if (
                        "save_iterations" in self.cfg
                        and idx in self.cfg["save_iterations"]
                    ):
                        self.save_map(idx)
                    last_idx += 1
        except Exception as e:
            print(e)
            print("\nSLAM failed. Saving map and results.\n")
        finally:
            with torch.no_grad():
                # Save final map as checkpoint
                self.save_map(last_idx)
                # Save results
                self.save_results(last_idx)
