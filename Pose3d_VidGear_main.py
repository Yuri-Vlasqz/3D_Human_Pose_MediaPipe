from vidgear.gears import CamGear
import numpy as np
import cv2
import time
from Pose3dModule import MpPose2D, Pose3D
from Dataset.single_person.dataset_tests import get_all_camera_parameters
from Dataset.Panoptic.panoptic_camera_parameters import get_panoptic_parameters


def concat_tile(im_list_2d):
    """Concatenate a matrix of images into a mosaic"""
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


# Define the number of rows and columns in the mosaic
ROWS = 2
COLS = 2

# Define the dimensions of each video frame in mosaic
FRAME_WIDTH = 480
FRAME_HEIGHT = 480

# Define the font and color for the FPS display
font = cv2.FONT_HERSHEY_SIMPLEX
line_type = cv2.LINE_AA
size = 0.75
RED = (0, 0, 255)
WHITE = (255, 255, 255)

# Define the URLs for each camera feed
URLS = ['Dataset/single_person/1.mp4',
        'Dataset/single_person/5.mp4',
        'Dataset/single_person/13.mp4',
        'Dataset/single_person/21.mp4']

# Create a list of VideoGear objects for each camera feed =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
cameras_gears = [CamGear(source=url, logging=False, backend=cv2.CAP_FFMPEG).start() for url in URLS]

# List of MpPose2D objects for each capture =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
all_mp_poses = [MpPose2D() for _ in range(len(URLS))]
for mp_pose in all_mp_poses:
    print(mp_pose)

# Create 3d pose object =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# Single Person
rotational_matrices, translation_vectors, camera_matrices, camera_distortions = get_all_camera_parameters()
# Panoptic
# rotational_matrices, translation_vectors, camera_matrices, camera_distortions = get_panoptic_parameters([0, 1, 6, 13])

pose_3d = Pose3D(camera_mats=camera_matrices,
                 rot_mats=rotational_matrices,
                 trans_vecs=translation_vectors,
                 min_visibility=0.75)


# 3d space canvas
axes = pose_3d.create_3d_space()

# Create a blank canvas for the mosaic
frames_list = np.empty([ROWS, COLS], dtype=np.ndarray)
frame_count = [0] * len(URLS)
landmarks_list = [[] for _ in range(len(URLS))]

all_triangulantion_time_ms = []
all_3d_draw_time_ms = []

# Loop through the camera feeds and merge them into a mosaic
while True:
    prev_time = time.time()
    # Loop through the frames and add them to the mosaic
    for i, video in enumerate(cameras_gears):
        row = i // COLS
        col = i % COLS

        frame = video.read()
        frame_count[i] += 1

        if frame is None:
            print(f"Could not read frame from {video}.")
            # cv2.putText(mosaic, f"Feed not found", (10, 30), font, size, RED, 1, line_type)
            continue  # skip to the next feed

        # Process feed
        pose_frame, landmarks_list[i] = all_mp_poses[i].calc_2d_kpt(frame)
        # Resize the frame to match the size of the mosaic
        pose_frame = cv2.resize(pose_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
        cv2.putText(pose_frame, f"Frame: {frame_count[i]}", (10, 30), font, size, WHITE, 1, line_type)
        frames_list[row][col] = pose_frame

    img_tile = concat_tile(frames_list)

    # Calculate and display the FPS for Mosaic
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    cv2.putText(img_tile, f"FPS: {fps:.0f}", (FRAME_WIDTH-50, FRAME_HEIGHT-20), font, size, RED, 1, line_type)

    # Display the mosaic
    cv2.imshow('VidGear: Synced Mosaic', img_tile)

    # Triangulate poses
    start = time.time()
    # avg_3d_pose = pose_3d.calc_avg_3d_kpt(landmarks_list)
    best_3d_pose = pose_3d.calc_best_3d_kpt(landmarks_list)
    stop = time.time()
    # print(f"Triangulation time: {round((stop - start) * 1000, 4)} miliseconds")
    all_triangulantion_time_ms.append(round((stop - start) * 1000, 4))

    # Draw 3d pose
    start = time.time()
    # pose_3d.update_3d_pose(avg_3d_pose, axes)
    pose_3d.update_3d_pose(best_3d_pose, axes)
    stop = time.time()
    # print(f"3d draw time: {round((stop - start) * 1000, 4)} miliseconds")
    all_3d_draw_time_ms.append(round((stop - start) * 1000, 4))

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("mean triangulation time (ms):", round(sum(all_triangulantion_time_ms) / len(all_triangulantion_time_ms), 3))
print("mean 3d draw time (ms):", round(sum(all_3d_draw_time_ms) / len(all_3d_draw_time_ms), 3))

# Release the camera feeds and close the window
for camera in cameras_gears:
    camera.stop()
cv2.destroyAllWindows()
