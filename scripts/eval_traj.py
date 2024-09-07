import argparse
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D, proj3d

from utils.eval_utils import evaluate_ate_rmse
from utils.pose_utils import get_camera_from_tensor, get_tensor_from_camera

# plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13})
# plt.style.use('dark_background')


def multiply_quaternions(q1, q2):
    w1 = q1[:, 0]
    x1 = q1[:, 1]
    y1 = q1[:, 2]
    z1 = q1[:, 3]
    w2 = q2[:, 0]
    x2 = q2[:, 1]
    y2 = q2[:, 2]
    z2 = q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z]).transpose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to output directory.")
    parser.add_argument(
        "--video", action="store_true", default=False, help="Create animation"
    )
    args = parser.parse_args()

    path = args.path
    results = np.load(os.path.join(path, "results.npz"))

    # Read trajectory
    traj_track = results["pose_est"]
    traj_gt = results["pose_gt"]
    time = np.array(list(range(len(traj_gt))))

    start = 0
    N = len(time)
    # N = 15

    # Early cutoff trajectory poses
    traj_track = traj_track[start:N, :]
    traj_gt = traj_gt[start:N, :]
    time = time[start:N]

    cam_centers = np.zeros_like(traj_track)
    for i_t in range(len(traj_track)):
        cam_centers[i_t, :] = get_tensor_from_camera(
            np.linalg.inv(get_camera_from_tensor(traj_track[i_t]).cpu())
        ).cpu()
    gt_centers = np.zeros_like(traj_gt)
    for i_t in range(len(traj_gt)):
        gt_centers[i_t, :] = get_tensor_from_camera(
            np.linalg.inv(get_camera_from_tensor(traj_gt[i_t]).cpu())
        ).cpu()

    # Align the trajectories
    traj_track_aligned, ate_rmse_c2w = evaluate_ate_rmse(
        cam_centers, gt_centers, method="umeyama"
    )
    _, ate_rmse_w2c = evaluate_ate_rmse(traj_track, traj_gt, method="umeyama")
    traj_gt_aligned = gt_centers

    # Print RMSE
    # ATE fixed around camera frame
    print(f"ATE RMSE: {ate_rmse_w2c}")
    # ATE fixed around world frame
    print(f"ATE RMSE c2w: {ate_rmse_c2w}")

    fig1, ax = plt.subplots(nrows=1, ncols=3, figsize=(6.4 * 1.5, 4.8 * 1.5))
    ax1 = ax[0]
    ax2 = ax[1]
    ax3 = ax[2]
    ax3.axis("off")
    ax3 = fig1.add_subplot(1, 3, 3, projection="3d")

    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax2.set_xlabel("X [m]")
    ax2.set_ylabel("Z [m]")
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Y [m]")
    ax3.set_zlabel("Z [m]")

    fig1.set_facecolor("white")
    ax1.grid(False)
    ax2.grid(False)
    ax3.set_box_aspect([1, 1, 1])

    ax1.axis("square")
    ax2.axis("square")
    ax3.axis("square")
    x_gt = traj_gt_aligned[:, 4]
    y_gt = traj_gt_aligned[:, 6]
    z_gt = traj_gt_aligned[:, 5]
    x_track = traj_track_aligned[:, 4]
    y_track = traj_track_aligned[:, 6]
    z_track = traj_track_aligned[:, 5]

    if args.video:
        # Function to update the plot for each frame
        def update(frame):
            ax1.clear()  # Clear the previous plot
            ax2.clear()  # Clear the previous plot

            # Update the plot for the current frame
            ax1.plot(
                x_gt[: frame + 1],
                y_gt[: frame + 1],
                color="red",
                linestyle="dashed",
                label="Ground Truth Traj.",
            )
            ax1.plot(
                x_track[: frame + 1],
                y_track[: frame + 1],
                color="blue",
                label="Tracked Traj.",
            )
            ax1.set_xlabel("X [m]")
            ax1.set_ylabel("Y [m]")

            ax1.set_xlim(-0.5, 6)  # Set fixed x-axis limits
            ax1.set_ylim(-0.5, 6)  # Set fixed y-axis limits

            ax2.plot(
                x_gt[: frame + 1],
                z_gt[: frame + 1],
                color="red",
                linestyle="dashed",
                label="Ground Truth Traj.",
            )
            ax2.plot(
                x_track[: frame + 1],
                z_track[: frame + 1],
                color="blue",
                label="Tracked Traj.",
            )
            ax2.set_xlabel("X [m]")
            ax2.set_ylabel("Z [m]")

            ax2.set_xlim(-0.5, 6)  # Set fixed x-axis limits
            ax2.set_ylim(-0.5, 6)  # Set fixed y-axis limits

            ax3.plot(
                traj_gt_aligned[: frame + 1, 6],
                traj_gt_aligned[: frame + 1, 4],
                traj_gt_aligned[: frame + 1, 5],
                color="red",
                linestyle="dashed",
                label="Ground Truth Traj.",
            )
            ax3.plot(
                traj_track_aligned[: frame + 1, 6],
                traj_track_aligned[: frame + 1, 4],
                traj_track_aligned[: frame + 1, 5],
                color="blue",
                label="Tracked Traj.",
            )
            ax3.set_ylabel("X [m]")
            ax3.set_xlabel("Y [m]")
            ax3.set_zlabel("Z [m]")
            ax3.set_ylim(-0.5, 6)  # Set fixed x-axis limits
            ax3.set_xlim(-0.5, 6)  # Set fixed y-axis limits
            ax3.set_zlim(-1, 1)  # Set fixed y-axis limits

            ax2.legend(loc="upper right")  # Add legend to the second plot
            fig1.tight_layout()  # Adjust layout

            print(f"Frame {frame} updated.")  # Debug print to check frame updates

        # Create the animation
        ani = FuncAnimation(fig1, update, frames=N)

        plt.show()

        # Save the animation as a video file
        video_filename = os.path.join(".", "trajectory_plot.mp4")
        ani.save(video_filename, writer="ffmpeg", fps=15, dpi=150)
    else:
        ax1.set_xlim(-1, 6)
        ax1.set_ylim(-1, 6)
        ax2.set_xlim(-1, 6)
        ax2.set_ylim(-1, 6)

        ax1.plot(
            x_gt,
            y_gt,
            color="red",
            linestyle="dashed",
            label="Ground Truth Traj.",
        )
        ax1.plot(
            x_track,
            y_track,
            color="blue",
            label="Tracked Traj.",
        )
        ax2.plot(
            x_gt,
            z_gt,
            color="red",
            linestyle="dashed",
            label="Ground Truth Traj.",
        )
        ax2.plot(
            x_track,
            z_track,
            color="blue",
            label="Tracked Traj.",
        )
        ax3.plot(
            traj_gt_aligned[:, 6],
            traj_gt_aligned[:, 4],
            traj_gt_aligned[:, 5],
            color="red",
            linestyle="dashed",
            label="Ground Truth Traj.",
        )
        ax3.plot(
            traj_track_aligned[:, 6],
            traj_track_aligned[:, 4],
            traj_track_aligned[:, 5],
            color="blue",
            label="Tracked Traj.",
        )
        ax3.set_ylim(-0.5, 6)  # Set fixed x-axis limits
        ax3.set_xlim(-0.5, 6)  # Set fixed y-axis limits
        ax3.set_zlim(-1, 1)  # Set fixed y-axis limits

        ax2.legend(loc="upper right")  # Add legend to the second plot
        fig1.tight_layout()  # Adjust layout
        fig1.savefig(
            "trajectory_plot.png", bbox_inches="tight", pad_inches=0.01, dpi=150
        )

        plt.show()

    print("Done")
