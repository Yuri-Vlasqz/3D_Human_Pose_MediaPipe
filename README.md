# 3D Human Pose MediaPipe
Multi-Camera Human Pose triangulation, with real-time 3D graph feedback.

Main program is multithreaded for performance increase in I/O bound operations, such as, concurrent image aquisition of all IP/RTSP cameras.
[MediaPipe Pose](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker), developed by Google, is used as the 2D Human Pose inference backbone.
The triangulation of all the detected poses are calculated using the DLT(Direct Linear Transform) method, making possible to minimize error with imperfect image captures.

- *Multiple perspective mosaic overlaid with inference pose*
![Multithread_multiview_2d_pose](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/c53f2afc-be0b-4c6f-bb18-3c136e48a49f)


- *Triangulated Pose (with camera location as floating points )*
![3d_pose](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/726285c6-0b8c-49c0-9500-4cbb3f4ca68c)

- *MediaPipe Pose - landmark model*
![MediaPipe Pose - landmark model](https://github.com/Yuri-Vlasqz/3D_Human_Pose_MediaPipe/assets/106136458/cc737d53-a247-4f00-8c1a-7e4a673b8db2)

## Datasets used:
CMU Panoptic datasets. Please refer to [Panoptic-Toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox.git) for detailed instructions.
