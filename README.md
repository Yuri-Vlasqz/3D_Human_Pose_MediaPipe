# 3D Human Pose MediaPipe
**Multi-Camera Human Pose triangulation, with real-time 3D graph feedback.**

- Frame inference is multithreaded for performance increase in I/O bound operations, such as, concurrent image aquisition of all IP/RTSP cameras.


- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker), developed by Google, is used as the 2D Human Pose inference backbone.

  *MediaPipe Pose - landmark model*

  ![MediaPipe Pose - landmark model](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/cc737d53-a247-4f00-8c1a-7e4a673b8db2)


- The triangulation of all the detected poses are calculated using the DLT(Direct Linear Transform) method, making possible to minimize error with imperfect image captures.


## Datasets:
CMU Panoptic datasets used for video feed testing in a controlled environment and ground truth comparison. Please refer to [Panoptic-Toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox.git) for detailed instructions.
![CMU_Dataset](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/f706f27e-6f11-410b-952a-cdf9ff1f8587)


## Results:

| **Statistics (mean)**[^1] | **Measurement** |
| --------------------- |:-----------:|
| Inference time        |  40.8 ms    |
| Triangulation time    |  0.48 ms    |
| 3D graph time         |  76.5 ms    |
| Frametime             |  90.4 ms    |
| Frames per second     |  11.1 FPS   |
| MPJPE (4 cameras)[^2] |  50.8 mm    |
[^1]: Test Machine specification: Ryzen 7 3700X, 16 GB RAM 
[^2]: Mean Per Joint Position Error


- *Multiple perspective mosaic overlaid with inference pose*
![Multithread_multiview_2d_pose](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/f3533641-1146-4cf5-9a9d-ee9de5413e70)


- *Real-time triangulated pose feedback (with cameras pyramidal field of view)*
![3d pose and cameras pyramids](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/b0585099-2bab-4011-bae1-ac0be9fc9a6a)


- *Dataset ground truth comparison for MPJPE calculation (green: CMU Panoptic, red: 3D Human Pose MediaPipe)*
![mediapipe vs panoptic GT](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/ce239b2a-0c71-4ef6-859b-b081271c1084)


## Usage:

Download dataset: [dataset link](http://domedb.perception.cs.cmu.edu/171026_pose3.html)

Unpack hdVideos and calibration files in 'dataset/Panoptic/171026_pose3'

`pip install -r requirements.txt`

`python main_3d_human_pose.py`

Any basic modifications can be made in the [inicialization.yaml](inicialization.yaml) file
