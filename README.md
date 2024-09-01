# MM3DGS-SLAM

The website of our paper for "MM3DGS SLAM: Multi-modal 3D Gaussian Splatting for SLAM Using
Vision, Depth, and Inertial Measurements".

[[Project Page]](https://vita-group.github.io/MM3DGS-SLAM/) | [[Video]](https://www.youtube.com/watch?v=drf6UxehChE) | [[Paper]](https://arxiv.org/pdf/2404.00923.pdf) | [[Arxiv]](https://arxiv.org/abs/2404.00923)

## Framework

![overview](./docs/static/images/framework.jpg)

As shown above, we present the framework for Multi-modal 3D Gaussian Splatting for SLAM. We utilize inertial measurements, RGB images, and depth measurements to create a SLAM method using 3D Gaussians. Through the effort of integrating multiple modalities, our SLAM method performs with high tracking and mapping accuracy.

## Dataset

To access our dataset, visit [Hugging Face](https://huggingface.co/datasets/neel1302/UT-MM/tree/main). To see videos of the content, please visit our [[Project Page]](https://vita-group.github.io/MM3DGS-SLAM/).

The following is a quick overview of the topics that can be found within each bag file:

```
/vrpn_client_node/Jackal_Latest/pose   : geometry_msgs/PoseStamped (Pose Coordinates, 100hz)
/realsense/color/image_raw/compressed  : sensor_msgs/CompressedImage (RGB Image, 30hz)
/realsense/depth/image_rect_raw        : sensor_msgs/Image (Depth Image, 30hz)
/microstrain/imu/data                  : sensor_msgs/Imu (Imu Measurements, 100hz)
/ouster/points                         : sensor_msgs/PointCloud2 (LiDAR Point Clouds, 10hz)
```


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
