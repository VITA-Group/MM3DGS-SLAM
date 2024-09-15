# MM3DGS-SLAM

The code for our paper for "MM3DGS SLAM: Multi-modal 3D Gaussian Splatting for SLAM Using
Vision, Depth, and Inertial Measurements".

[[Project Page]](https://vita-group.github.io/MM3DGS-SLAM/) | [[Video]](https://www.youtube.com/watch?v=drf6UxehChE) | [[Paper]](https://arxiv.org/pdf/2404.00923.pdf) | [[Arxiv]](https://arxiv.org/abs/2404.00923)

## Framework

![overview](./docs/static/images/framework.jpg)

As shown above, we present the framework for Multi-modal 3D Gaussian Splatting for SLAM. We utilize inertial measurements, RGB images, and depth measurements to create a SLAM method using 3D Gaussians. Through the effort of integrating multiple modalities, our SLAM method performs with high tracking and mapping accuracy.

## Cloning the Repository

The repository contains submodules, thus please check it out with

```shell
# SSH
git clone git@github.com:VITA-Group/MM3DGS-SLAM.git --recursive
```

## Environment setup

The simplest way to install dependencies is with
[anaconda](https://www.anaconda.com/)

```shell
conda create -n mm3dgs python=3.10
conda activate mm3dgs
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda env update -f environment.yml
```

## Run

To run MM3DGS, simply run the top level module with the path to the config
file.

```shell
python slam_top.py --config <config path>
```

Included in this repo are the configs for UT-MM and TUM-RGBD datasets. For
example,

```shell
python slam_top.py --config ./configs/UTMM.yml
```

Note that the directory to the dataset must first be added to the config
file before running, e.g.,

```yaml
inputdir: /datasets/UTMM/ # TODO: input dataset location
```

Outputs, including any enabled debug, will be put in the `./output` directory.

## Evaluation

After running the above script, trajectory and image evaluation will
automatically be recorded.

Trajectory error can be recalculated through `scripts/eval_traj.py`.
This script reads a `results.npz` file containing numpy arrays of estimate
and ground truth poses.
It also creates trajectory plots and animations from the paper.

```shell
python scripts/eval_traj.py --path <output path>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for scripts/eval_traj.py</span></summary>

#### --path

Path to the output directory containing a `results.npz` file

#### --video

Add this flag to animate the plot

</details>
<br>

Image evaluation results can be recalculated with `scripts/eval_image.py`.
This script loads a saved checkpoint and renders the map from estimated poses,
comparing them with ground truth images from the dataset.

```shell
python scripts/eval_image.py -c <config path>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for scripts/eval_image.py</span></summary>

#### --config / -c

Path to the config file

#### --output / -o

Optional. Path to the output directory if not defined in config file

#### --iteration / -i

Optional. Iteration checkpoint to evaluate if not defined in config file

</details>
<br>

Further, the output map can be visualized using `scripts/visualizer.py`

```shell
python scripts/visualizer.py -c <config path> -m <output path> -i <iteration>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for scripts/visualizer.py</span></summary>

#### --config / -c

Path to the config file

#### --model / -m

Path to the output directory containing a `results.npz` file and `point_cloud` directory

#### --iteration / -i

Iteration number of the output

#### --online

Add this flag to animate the visualizer along the trajectory path

</details>
<br>

## UT-MM Dataset

To access our dataset, visit [Hugging Face](https://huggingface.co/datasets/neel1302/UT-MM/tree/main). To see videos of the content, please visit our [[Project Page]](https://vita-group.github.io/MM3DGS-SLAM/).

A script to convert the UT-MM rosbag data to usable .txt files is included in
`scripts/bag2data.py`. Note that [ROS](https://wiki.ros.org/ROS/Installation)
is required to read the camera-to-imu transformation.

```shell
python scripts/bag2data.py --path <rosbag dir path> --scene <scene name>
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for scripts/bag2data.py</span></summary>

#### --path

Path to the directory containing rosbags

#### --scene

Name of the scene to read

</details>
<br>

### Rosbag topics

Combined, the datasets contain 8387 images, 2796 LiDAR scans, and 27971 IMU measurements. The following is a quick overview of the topics that can be found within each bag file:

```
/vrpn_client_node/Jackal_Latest/pose   : geometry_msgs/PoseStamped (Pose Coordinates, 100hz)
/realsense/color/image_raw/compressed  : sensor_msgs/CompressedImage (RGB Image, 30hz)
/realsense/depth/image_rect_raw        : sensor_msgs/Image (Depth Image, 30hz)
/microstrain/imu/data                  : sensor_msgs/Imu (Imu Measurements, 100hz)
/ouster/points                         : sensor_msgs/PointCloud2 (LiDAR Point Clouds, 10hz)
```

To visualize the dataset, ROS is needed. Some scripts are provided in the UT_MM_Scripts directory.

```
roscore

rosrun rviz rviz -d UT_MM_Scripts/configs/jackal.rviz

rqt --clear-config --perspective-file UT_MM_Scripts/configs/rqt_plots.perspective

rosbag play --clock *.bag --pause

python3 imu_preintegration.py
```

<br>

## Citation

If you find this work interesting and use it in your research, please consider citing our paper.

```
@misc{sun2024mm3dgs,
      title={MM3DGS SLAM: Multi-modal 3D Gaussian Splatting for SLAM Using Vision, Depth, and Inertial Measurements},
      author={Lisong C. Sun and Neel P. Bhatt and Jonathan C. Liu and Zhiwen Fan and Zhangyang Wang and Todd E. Humphreys and Ufuk Topcu},
      year={2024},
      eprint={2404.00923},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
