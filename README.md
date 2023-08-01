# 3D Human Pose MediaPipe
Multi-Camera Human Pose triangulation, with real-time 3D graph feedback.

- Main program is multithreaded for performance increase in I/O bound operations, such as, concurrent image aquisition of all IP/RTSP cameras.

- [MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker), developed by Google, is used as the 2D Human Pose inference backbone.

  *MediaPipe Pose - landmark model*
  ![MediaPipe Pose - landmark model](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/cc737d53-a247-4f00-8c1a-7e4a673b8db2)

- The triangulation of all the detected poses are calculated using the DLT(Direct Linear Transform) method, making possible to minimize error with imperfect image captures.


## Datasets:
CMU Panoptic datasets used for video feed testing in a controlled environment and ground truth comparison. Please refer to [Panoptic-Toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox.git) for detailed instructions.
![CMU_Dataset](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/f706f27e-6f11-410b-952a-cdf9ff1f8587)


## Results:
- Performance Statistics:
  - mean inference time:&emsp;&emsp;53.2 ms
  - mean triangulation time:&ensp;1.2 ms
  - mean 3d draw time:&emsp;&emsp;&ensp;91.6 ms
  - mean frame time:&emsp;&emsp;&emsp;&ensp;98.8 ms
  - mean FPS:&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;10.1

- *Multiple perspective mosaic overlaid with inference pose*
![Multithread_multiview_2d_pose](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/f3533641-1146-4cf5-9a9d-ee9de5413e70)


- *Triangulated Pose (with camera location as floating points )*
![3d_pose](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/726285c6-0b8c-49c0-9500-4cbb3f4ca68c)


