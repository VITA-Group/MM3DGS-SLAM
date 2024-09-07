import argparse
import os
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rosbag
import rospy
import tf
from cv_bridge import CvBridge
from PIL import Image as PILImage
from rosbag import Bag
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, PointCloud2
from tf.transformations import (
    euler_matrix,
    quaternion_from_euler,
    quaternion_from_matrix,
    quaternion_matrix,
)


def read_bag(bag_file_path, rgb_path, depth_path):
    with rosbag.Bag(bag_file_path, "r") as bag:
        topics = bag.get_type_and_topic_info().topics.keys()
        print("Topics in the bag file:")
        for topic in topics:
            print(topic)

        os.makedirs(rgb_path, exist_ok=True)
        os.makedirs(depth_path, exist_ok=True)

        # Read camera intrinsics
        print("Reading camera intrinsics")
        intrinsics_file = os.path.join(bag_dir_path, scene, "intrinsics.txt")
        with open(intrinsics_file, "w") as f:
            f.write("# camera intrinsics\n")
            f.write(f"# file: {scene}.bag\n")
            f.write("# timestamp K\n")
            for topic, msg, t in bag.read_messages(
                topics=["/realsense/color/camera_info"]
            ):
                time = f"{msg.header.stamp.secs}.{msg.header.stamp.nsecs:09d}"
                f.write(f"{time} {msg.K}\n")

        # IMU to camera transform is static. In the meantime,
        # we can extract using `rosrun tf2_tools echo.py realsense_color_frame microstrain_link`
        # Read transformations
        print("Reading transformations")
        tf_file = os.path.join(bag_dir_path, scene, "tf.txt")
        tf_listener = tf.TransformListener()
        with open(tf_file, "w") as f:
            f.write("# transformations\n")
            f.write(f"# file: {scene}.bag\n")
            f.write("# tx ty tz qx qy qz qw\n")
            f.write("# microstrain_link to realsense_color_frame\n")
            (t, q) = tf_listener.lookupTransform(
                "microstrain_link", "realsense_color_frame"
            )
            f.write(f"{t.x} {t.y} {t.z} {q.x} {q.y} {q.z} {q.w}\n")

        # Create groundtruth.txt file
        print("Reading GT trajectory")
        gt_file = os.path.join(bag_dir_path, scene, "groundtruth.txt")
        with open(gt_file, "w") as f:
            f.write("# ground truth trajectory\n")
            f.write(f"# file: {scene}.bag\n")
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for topic, msg, t in bag.read_messages(
                topics=["/vrpn_client_node/Jackal_Latest/pose"]
            ):
                time = f"{msg.header.stamp.secs}.{msg.header.stamp.nsecs:09d}"
                t = msg.pose.position
                q = msg.pose.orientation
                f.write(f"{time} {t.x} {t.y} {t.z} {q.x} {q.y} {q.z} {q.w}\n")

        bridge = CvBridge()
        print("Reading image files")
        # Create rgb.txt file
        rgb_info_file = os.path.join(bag_dir_path, scene, "rgb.txt")
        with open(rgb_info_file, "w") as f:
            f.write("# color images\n")
            f.write(f"# file: {scene}.bag\n")
            f.write("# timestamp filename\n")
            i = 0
            for topic, msg, t in bag.read_messages(
                topics=["/realsense/color/image_raw/compressed"]
            ):
                # np_arr = np.frombuffer(msg.data, np.uint8)
                # image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                # Crop the bottom 60 pixels
                image = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
                image = image[:-60, :]
                time = f"{msg.header.stamp.secs}.{msg.header.stamp.nsecs:09d}"

                cv2.imwrite(os.path.join(rgb_path, f"{i:06d}.png"), image)

                f.write(f"{time} rgb/{i:06d}.png\n")
                i += 1

        bridge = CvBridge()
        print("Reading depth files")
        # Create depth.txt file
        depth_info_file = os.path.join(bag_dir_path, scene, "depth.txt")
        with open(depth_info_file, "w") as f:
            f.write("# depth images\n")
            f.write(f"# file: {scene}.bag\n")
            f.write("# timestamp filename\n")
            i = 0
            for topic, msg, t in bag.read_messages(
                topics=["/realsense/depth/image_rect_raw"]
            ):
                time = f"{msg.header.stamp.secs}.{msg.header.stamp.nsecs:09d}"
                image_data = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                    (msg.height, msg.width)
                )
                image_data = image_data[:-60, :]
                pil_image = PILImage.fromarray(image_data, mode="I;16")

                pil_image.save(os.path.join(depth_path, f"{i:06d}.png"))

                f.write(f"{time} depth/{i:06d}.png\n")
                i += 1

        print("Reading imu files")
        # Create imu.txt file
        imu_file = os.path.join(bag_dir_path, scene, "imu.txt")
        with open(imu_file, "w") as f:
            f.write("# imu measurements\n")
            f.write(f"# file: {scene}.bag\n")
            f.write(
                "# timestamp ori_x ori_y ori_z ori_w "
                + "ori_cov1 ori_cov2 ori_cov3 ori_cov4 ori_cov5 ori_cov6 ori_cov7 ori_cov8 ori_cov9 "
                + "ang_x ang_y ang_z "
                + "ang_cov1 ang_cov2 ang_cov3 ang_cov4 ang_cov5 ang_cov6 ang_cov7 ang_cov8 ang_cov9 "
                + "acc_x acc_y acc_z "
                + "acc_cov1 acc_cov2 acc_cov3 acc_cov4 acc_cov5 acc_cov6 acc_cov7 acc_cov8 acc_cov9\n"
            )
            i = 0
            for topic, msg, t in bag.read_messages(topics=["/microstrain/imu/data"]):
                time = f"{msg.header.stamp.secs}.{msg.header.stamp.nsecs:09d}"

                o = msg.orientation
                oc = msg.orientation_covariance
                a = msg.angular_velocity
                ac = msg.angular_velocity_covariance
                l = msg.linear_acceleration
                lc = msg.linear_acceleration_covariance
                f.write(
                    f"{time} {o.x} {o.y} {o.z} {o.w} "
                    + f"{oc[0]} {oc[1]} {oc[2]} {oc[3]} {oc[4]} {oc[5]} {oc[6]} {oc[7]} {oc[8]} "
                    + f"{a.x} {a.y} {a.z} "
                    + f"{ac[0]} {ac[1]} {ac[2]} {ac[3]} {ac[4]} {ac[5]} {ac[6]} {ac[7]} {ac[8]} "
                    + f"{l.x} {l.y} {l.z} "
                    + f"{lc[0]} {lc[1]} {lc[2]} {lc[3]} {lc[4]} {lc[5]} {lc[6]} {lc[7]} {lc[8]}\n"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to rosbag directory.")
    parser.add_argument("--scene", type=str, help="Name of scene.")

    args = parser.parse_args()

    bag_dir_path = args.path
    scene = args.scene

    bag_file_path = glob(os.path.join(bag_dir_path, scene, "*.bag"))[0]
    rgb_path = os.path.join(bag_dir_path, scene, "rgb")
    depth_path = os.path.join(bag_dir_path, scene, "depth")

    read_bag(bag_file_path, rgb_path, depth_path)
