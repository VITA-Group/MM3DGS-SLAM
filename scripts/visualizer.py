# Note that currently, this code only supports viewing isotropic Gaussians

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

from configs.config import load_config
from slam.gaussian_model import GaussianModel
from slam.renderer import Renderer
from utils.pose_utils import get_camera_from_tensor, get_tensor_from_camera
from utils.sh_utils import eval_sh

W, H = 640, 330
near, far = 0.01, 100.0
VIEW_SCALE = 2
fps = 20
VISUALIZE_CAMS = True
RENDER_MODE = "color"  # ['color', 'depth' or 'centers']
OFFSET_FIRST_VIZ_CAM = True
SHOW_SIL = False


def make_lineset(all_pts, all_cols, num_lines):
    linesets = []
    for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(pts, np.float64)
        )
        lineset.colors = o3d.utility.Vector3dVector(
            np.ascontiguousarray(cols, np.float64)
        )
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(
            np.ascontiguousarray(line_indices, np.int32)
        )
        linesets.append(lineset)
    return linesets


def render(renderer, gaussians, camera_pose):
    with torch.no_grad():
        if not isinstance(camera_pose, torch.Tensor):
            camera_pose = torch.tensor(camera_pose).cuda()

        result = renderer.render(
            gaussians,
            camera_pose=camera_pose,
        )
        image = result["render"]
        depth = result["depth"][0, :, :]
        silhouette = result["depth"][1, :, :]

        return image, depth, silhouette


def rgbd2pcd(color, depth, camera_pose, cfg):
    width, height = color.shape[2], color.shape[1]
    CX = cfg["cam"]["cx"]
    CY = cfg["cam"]["cy"]
    FX = cfg["cam"]["fx"]
    FY = cfg["cam"]["fy"]

    w2c = get_camera_from_tensor(camera_pose)

    # Compute indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth.reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # Convert to Open3D format
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())

    # Colorize point cloud
    if RENDER_MODE == "depth":
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap("jet")
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array(
            [1.0, 1.0, 1.0]
        )
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


def visualize(cfg_path, model=None, iteration=None):
    # Load the 3DGS map
    cfg = load_config(cfg_path)
    output_path = model if model is not None else cfg["outputdir"]
    cfg["desired_width"] = W
    cfg["desired_height"] = H
    # Scale intrinsics to match the visualization resolution
    cfg["cam"]["cx"] *= cfg["desired_width"] / cfg["cam"]["image_width"]
    cfg["cam"]["cy"] *= cfg["desired_height"] / cfg["cam"]["image_height"]
    cfg["cam"]["fx"] *= cfg["desired_width"] / cfg["cam"]["image_width"]
    cfg["cam"]["fy"] *= cfg["desired_height"] / cfg["cam"]["image_height"]
    gaussians = GaussianModel(cfg)
    gaussians.load_ply(
        os.path.join(
            output_path,
            "point_cloud",
            "iteration_"
            + str(iteration if iteration is not None else cfg["iteration"]),
            "point_cloud.ply",
        )
    )
    if RENDER_MODE.lower() != "centers":
        cfg["white_background"] = True
    renderer = Renderer(cfg)

    # Initialize camera
    results = np.load(os.path.join(output_path, "results.npz"))

    k = np.eye(3)
    k[0][2] = cfg["cam"]["cx"]
    k[1][2] = cfg["cam"]["cy"]
    k[0][0] = cfg["cam"]["fx"]
    k[1][1] = cfg["cam"]["fy"]

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=int(W * VIEW_SCALE), height=int(H * VIEW_SCALE), visible=True
    )

    # Render very first frame
    est_poses = results["pose_est"]
    camera_pose = est_poses[0, :]
    im, depth, sil = render(renderer, gaussians, camera_pose)
    init_pts, init_cols = rgbd2pcd(im, depth, camera_pose, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    if VISUALIZE_CAMS:
        # Initialize Estimated Camera Frustums
        frustum_size = 0.045
        num_t = len(est_poses)
        cam_centers = []
        cam_colormap = plt.get_cmap("cool")
        norm_factor = 0.5
        for i_t in range(num_t):
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                W,
                H,
                k,
                get_camera_from_tensor(est_poses[i_t]).cpu().numpy(),
                frustum_size,
            )
            frustum.paint_uniform_color(
                np.array(cam_colormap(i_t * norm_factor / num_t)[:3])
            )
            vis.add_geometry(frustum)
            cam_centers.append(
                np.linalg.inv(get_camera_from_tensor(est_poses[i_t]).cpu())[:3, 3]
            )

        # Initialize Camera Trajectory
        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap("cool")
        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(
                np.array(
                    line_colormap(
                        (line_t * norm_factor / total_num_lines) + norm_factor
                    )[:3]
                )
            )
        cols = np.array(cols)
        all_cols = [cols]
        out_pts = [np.array(cam_centers)]
        linesets = make_lineset(out_pts, all_cols, num_lines)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    # Initialize View Control
    view_k = k * VIEW_SCALE
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    if OFFSET_FIRST_VIZ_CAM:
        view_w2c = get_camera_from_tensor(est_poses[0]).cpu().numpy()
        view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
    else:
        view_w2c = get_camera_from_tensor(est_poses[0]).cpu().numpy()
    cparams.extrinsic = view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(H * VIEW_SCALE)
    cparams.intrinsic.width = int(W * VIEW_SCALE)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = VIEW_SCALE
    render_options.light_on = False

    # Interactive Rendering
    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / VIEW_SCALE
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if RENDER_MODE == "centers":
            pts = o3d.utility.Vector3dVector(
                gaussians.get_xyz.clone().detach().contiguous().double().cpu().numpy()
            )

            shs_view = gaussians.get_features.transpose(1, 2).view(
                -1, 3, (gaussians.max_sh_degree + 1) ** 2
            )
            dir_pp = gaussians.get_xyz - torch.tensor(w2c, device="cuda").inverse()[
                3, :3
            ].repeat(gaussians.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

            cols = o3d.utility.Vector3dVector(
                colors_precomp.clone()
                .detach()
                .clone()
                .detach()
                .contiguous()
                .double()
                .cpu()
                .numpy()
            )
        else:
            im, depth, sil = render(renderer, gaussians, get_tensor_from_camera(w2c))
            if SHOW_SIL:
                im = (1 - sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, get_tensor_from_camera(w2c), cfg)

        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

    # Cleanup
    vis.destroy_window()
    del view_control
    del vis
    del render_options


def visualize_online(cfg_path, model=None, iteration=None):
    # Load the 3DGS map
    cfg = load_config(cfg_path)
    output_path = model if model is not None else cfg["outputdir"]
    cfg["desired_width"] = W
    cfg["desired_height"] = H
    # Scale intrinsics to match the visualization resolution
    cfg["cam"]["cx"] *= cfg["desired_width"] / cfg["cam"]["image_width"]
    cfg["cam"]["cy"] *= cfg["desired_height"] / cfg["cam"]["image_height"]
    cfg["cam"]["fx"] *= cfg["desired_width"] / cfg["cam"]["image_width"]
    cfg["cam"]["fy"] *= cfg["desired_height"] / cfg["cam"]["image_height"]
    gaussians = GaussianModel(cfg)
    gaussians.load_ply(
        os.path.join(
            output_path,
            "point_cloud",
            "iteration_"
            + str(iteration if iteration is not None else cfg["iteration"]),
            "point_cloud.ply",
        )
    )
    if RENDER_MODE.lower() != "centers":
        cfg["white_background"] = True
    renderer = Renderer(cfg)

    # Initialize camera
    results = np.load(os.path.join(output_path, "results.npz"))

    k = np.eye(3)
    k[0][2] = cfg["cam"]["cx"]
    k[1][2] = cfg["cam"]["cy"]
    k[0][0] = cfg["cam"]["fx"]
    k[1][1] = cfg["cam"]["fy"]

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=int(W * VIEW_SCALE), height=int(H * VIEW_SCALE), visible=True
    )

    # Render very first frame
    est_poses = results["pose_est"]
    camera_pose = est_poses[0, :]
    im, depth, sil = render(renderer, gaussians, camera_pose)
    init_pts, init_cols = rgbd2pcd(im, depth, camera_pose, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols
    vis.add_geometry(pcd)

    # Initialize Estimated Camera Frustums
    frustum_size = 0.045
    num_t = len(est_poses)
    all_w2cs = []
    for i in range(num_t):
        all_w2cs.append(get_camera_from_tensor(est_poses[i]).cpu().numpy())
    cam_centers = []
    cam_colormap = plt.get_cmap("cool")
    norm_factor = 0.5
    total_num_lines = num_t - 1
    line_colormap = plt.get_cmap("cool")

    # Initialize View Control
    view_k = k * VIEW_SCALE
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()
    if OFFSET_FIRST_VIZ_CAM:
        first_view_w2c = get_camera_from_tensor(est_poses[0]).cpu().numpy()
        first_view_w2c[:3, 3] = first_view_w2c[:3, 3] + np.array([0, 0, 0.5])
    else:
        first_view_w2c = get_camera_from_tensor(est_poses[0]).cpu().numpy()
    cparams.extrinsic = first_view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(H * VIEW_SCALE)
    cparams.intrinsic.width = int(W * VIEW_SCALE)
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = VIEW_SCALE
    render_options.light_on = False

    # Rendering of Online Reconstruction
    start_time = time.time()
    num_timesteps = num_t
    viz_start = True
    curr_timestep = 0
    while curr_timestep < (num_timesteps - 1):
        passed_time = time.time() - start_time
        passed_frames = passed_time * fps
        curr_timestep = int(passed_frames % num_timesteps)
        if not viz_start:
            if curr_timestep == prev_timestep:
                continue

        # Update Camera Frustum
        if curr_timestep == 0:
            cam_centers = []
            if not viz_start:
                vis.remove_geometry(prev_lines)
        if not viz_start:
            vis.remove_geometry(prev_frustum)
        new_frustum = o3d.geometry.LineSet.create_camera_visualization(
            W, H, k, all_w2cs[curr_timestep], frustum_size
        )
        new_frustum.paint_uniform_color(
            np.array(cam_colormap(curr_timestep * norm_factor / num_t)[:3])
        )
        vis.add_geometry(new_frustum)
        prev_frustum = new_frustum
        cam_centers.append(np.linalg.inv(all_w2cs[curr_timestep])[:3, 3])

        # Update Camera Trajectory
        if len(cam_centers) > 1 and curr_timestep > 0:
            num_lines = [1]
            cols = []
            for line_t in range(curr_timestep):
                cols.append(
                    np.array(
                        line_colormap(
                            (line_t * norm_factor / total_num_lines) + norm_factor
                        )[:3]
                    )
                )
            cols = np.array(cols)
            all_cols = [cols]
            out_pts = [np.array(cam_centers)]
            linesets = make_lineset(out_pts, all_cols, num_lines)
            lines = o3d.geometry.LineSet()
            lines.points = linesets[0].points
            lines.colors = linesets[0].colors
            lines.lines = linesets[0].lines
            vis.add_geometry(lines)
            prev_lines = lines
        elif not viz_start:
            vis.remove_geometry(prev_lines)

        # Get Current View Camera
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / VIEW_SCALE
        k[2, 2] = 1
        w2c = cam_params.extrinsic
        w2c = np.dot(first_view_w2c, all_w2cs[curr_timestep])
        cam_params.extrinsic = w2c
        view_control.convert_from_pinhole_camera_parameters(
            cam_params, allow_arbitrary=True
        )

        if RENDER_MODE == "centers":
            pts = o3d.utility.Vector3dVector(
                gaussians.get_xyz.clone().detach().contiguous().double().cpu().numpy()
            )

            shs_view = gaussians.get_features.transpose(1, 2).view(
                -1, 3, (gaussians.max_sh_degree + 1) ** 2
            )
            dir_pp = gaussians.get_xyz - torch.tensor(w2c, device="cuda").inverse()[
                3, :3
            ].repeat(gaussians.get_features.shape[0], 1)
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

            cols = o3d.utility.Vector3dVector(
                colors_precomp.clone()
                .detach()
                .clone()
                .detach()
                .contiguous()
                .double()
                .cpu()
                .numpy()
            )
        else:
            im, depth, sil = render(renderer, gaussians, get_tensor_from_camera(w2c))
            if SHOW_SIL:
                im = (1 - sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, get_tensor_from_camera(w2c), cfg)

        # Update Gaussians
        pcd.points = pts
        pcd.colors = cols
        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()
        prev_timestep = curr_timestep
        viz_start = False

    # Cleanup
    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for visualizing 3DGS")
    parser.add_argument("--config", "-c", type=str, help="Path to config file.")
    parser.add_argument(
        "--model", "-m", type=str, help="Path to map output", default=None
    )
    parser.add_argument("--iteration", "-i", type=int, help="Iteration", default=None)
    parser.add_argument("--online", action="store_true", default=False)

    args = parser.parse_args()

    # Visualize
    if args.online:
        visualize_online(args.config, args.model, args.iteration)
    else:
        visualize(args.config, args.model, args.iteration)
