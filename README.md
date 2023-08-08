# 3D Human Pose MediaPipe
**Multi-Camera Human Pose triangulation, with real-time 3D graph feedback.**

- Main program is multithreaded for performance increase in I/O bound operations, such as, concurrent image aquisition of all IP/RTSP cameras.

- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker), developed by Google, is used as the 2D Human Pose inference backbone.

  *MediaPipe Pose - landmark model*
  ![MediaPipe Pose - landmark model](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/cc737d53-a247-4f00-8c1a-7e4a673b8db2)

- The triangulation of all the detected poses are calculated using the DLT(Direct Linear Transform) method, making possible to minimize error with imperfect image captures.


## Datasets:
CMU Panoptic datasets used for video feed testing in a controlled environment and ground truth comparison. Please refer to [Panoptic-Toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox.git) for detailed instructions.
![CMU_Dataset](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/f706f27e-6f11-410b-952a-cdf9ff1f8587)


## Results:
- **Performance Statistics[^1]:**
  - mean inference time:&emsp;&emsp;&nbsp;53.2 ms
  - mean triangulation time:&ensp;1.2 ms
  - mean 3d draw time:&emsp;&emsp;&ensp;&nbsp;91.6 ms
  - mean frame time:&emsp;&emsp;&emsp;&ensp;&nbsp;98.8 ms
  - mean FPS:&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;10.1
  - MPJPE (4 cameras):&emsp;&emsp;&emsp;&nbsp;50.8 mm (Mean Per Joint Position Error)

[^1]: Test Machine specification: Ryzen 7 3700X, 16 GB RAM 


- *Multiple perspective mosaic overlaid with inference pose*
![Multithread_multiview_2d_pose](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/f3533641-1146-4cf5-9a9d-ee9de5413e70)


- *Triangulated Pose (with cameras pyramidal field of view)*
![3d pose and cameras pyramids](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/b0585099-2bab-4011-bae1-ac0be9fc9a6a)


- *Dataset ground truth comparison (green: CMU Panoptic, red: 3D Human Pose MediaPipe)*
![mediapipe vs panoptic GT](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/ce239b2a-0c71-4ef6-859b-b081271c1084)

