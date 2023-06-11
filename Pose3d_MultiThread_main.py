from Pose3dModule import MpPose2D, Pose3D
import concurrent.futures
import numpy as np
import threading
import time
import cv2

from Dataset.single_person.dataset_tests import get_all_camera_parameters
from Dataset.Panoptic.panoptic_camera_parameters import get_panoptic_parameters
from Dataset.DELED_RTSP_Cameras.Intrinsic_parameters import get_deled_intrinsics


# print(cv2.getBuildInformation())

# Define RTSP camera URLs

# cam_urls = ['Dataset/single_person/1.mp4',
#             'Dataset/single_person/5.mp4',
#             'Dataset/single_person/13.mp4',
#             'Dataset/single_person/21.mp4']
cam_urls = ['Dataset/Panoptic/171026_pose3/hdVideos/hd_00_00.mp4',
            'Dataset/Panoptic/171026_pose3/hdVideos/hd_00_01.mp4',
            'Dataset/Panoptic/171026_pose3/hdVideos/hd_00_06.mp4',
            'Dataset/Panoptic/171026_pose3/hdVideos/hd_00_13.mp4']


# Define the number of rows and columns in the mosaic
num_rows = 2
num_cols = 2
frame_width = 480
frame_height = 270

# FPS display variables
font = cv2.FONT_HERSHEY_SIMPLEX
line_type = cv2.LINE_AA
size = 0.6


def create_mosaic_structure(n_feeds, n_rows, n_cols, frame_h, frame_w):
    empty_mosaic = np.zeros((n_rows * frame_h, n_cols * frame_w, 3), dtype=np.uint8)

    frame_positions = []
    for position in range(n_feeds):
        row = position // n_cols
        col = position % n_cols
        y1 = row * frame_h
        y2 = y1 + frame_h
        x1 = col * frame_w
        x2 = x1 + frame_w
        yc = (y2 + y1) // 2
        xc = (x2 + x1) // 2
        frame_positions.append([y1, y2, x1, x2, yc, xc])

    print(f"Mosaic: {n_rows}-rows X {n_cols}-columns "
          f"of {frame_w}x{frame_h} frames")

    return empty_mosaic, frame_positions


def process_feed(feed, pose_2d, frame_limits: list, index: int, distorted_feed: bool,
                 dist_cam_mat, dist_coeffs, undistorted_cam_mat, roi):
    """ Function to process each camera feed"""

    # Global variables
    global frame_width
    global frame_height
    global mosaic
    global pose_output_buffer
    global stop_event
    global thread_lock

    # Get the location in the mosaic to place the frame
    y1, y2, x1, x2, yc, xc = frame_limits

    timeout = False

    # Initialize the previous time for FPS calculation
    prev_time = time.time()

    # Loop over the frames from one camera =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    while not stop_event.is_set():

        # Read a frame from the camera
        ret, frame = feed.read()

        if not ret:

            if timeout:
                continue  # skip to the next frame

            # print(f"Could not read frame from {cap}.")
            cv2.putText(mosaic, "Feed not found:", (xc - 70, yc), font, size, (0, 0, 255), 1, line_type)
            cv2.putText(mosaic, f"{feed}", (x1 + 10, y2 - 35), font, size, (0, 0, 255), 1, line_type)
            timeout = True
            continue  # skip to the next frame

        timeout = False

        if distorted_feed:
            # Undistort the frame
            frame = cv2.undistort(frame, dist_cam_mat, dist_coeffs, None, undistorted_cam_mat)
            # Crop the undistorted image based on the region of interest (ROI) calculated (for alpha = 1)
            # x, y, w, h = roi
            # frame = frame[y:y + h, x:x + w]

        # Process MediaPipe-Pose backend
        pose_frame, landmarks_list = pose_2d.calc_2d_kpt(frame)

        # Resize the frame to match the size of the mosaic
        pose_frame = cv2.resize(pose_frame, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        with thread_lock:

            # update the output 2d pose list
            pose_output_buffer[index] = landmarks_list

            # Update the mosaic with the frame with 2d joints
            mosaic[y1:y2, x1:x2] = pose_frame

            # Calculate and display the FPS for this camera stream
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            cv2.putText(mosaic, f"FPS: {fps:.0f}", (x1 + 10, y2 - 10), font, size, (255, 255, 255), 1, line_type)

        prev_time = time.time()

    # Release the video capture object for this camera feed
    feed.release()


# Create video capture objects for each camera =-=-=-=-=-=-=-=-=-=-=-=
captures = [cv2.VideoCapture(url, cv2.CAP_FFMPEG) for url in cam_urls]
# for videos: cv2.CAP_MSMF
# todo: test if cv2.CAP_MSMF works for H.264 decoding rtsp stream???
#       https://github.com/cgohlke/vidsrc/issues/1#issue-805320891
# set the start time for recorded videos
# for cap in captures:
#    cap.set(cv2.CAP_PROP_POS_FRAMES, 1700)

# List of MpPose2D objects for each capture =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
all_mp_poses = [MpPose2D() for _ in range(len(captures))]

# get camera parameters =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# single person
# rotational_matrices, translation_vectors, camera_matrices, camera_distortions = get_all_camera_parameters()
# distorted = False

# Panoptic
rotational_matrices, translation_vectors, camera_matrices, camera_distortions = get_panoptic_parameters([0, 1, 6, 13])
distorted = False

# DELED RTSP
dist_cameras_mats, undist_cameras_mats, cameras_dists_coeff, cameras_roi = get_deled_intrinsics()
# distorted = True

# Create 3d pose object =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
pose_3d = Pose3D(camera_mats=camera_matrices,
                 rot_mats=rotational_matrices,
                 trans_vecs=translation_vectors,
                 min_visibility=0.5)

# Create an empty mosaic structure
mosaic, frames_limits = create_mosaic_structure(len(captures), num_rows, num_cols, frame_height, frame_width)

# 2d pose buffer
pose_output_buffer = [None] * len(captures)

# 3d space canvas
axes = pose_3d.create_3d_space()

# Create an event object to signal when the program should stop
stop_event = threading.Event()
# Thread synchronization primitive (to make mosaic writing section of code thread safe)
thread_lock = threading.Lock()


# Create a thread pool and submit a task for each camera feed
with concurrent.futures.ThreadPoolExecutor() as executor:
    print("Initializing threads for processing feeds:")
    for i, cap in enumerate(captures):
        if not distorted:
            dist_cameras_mats[i], cameras_dists_coeff[i], undist_cameras_mats[i], cameras_roi[i] = None, None, None, None
        # Submit task for each camera feed
        executor.submit(process_feed, cap, all_mp_poses[i], frames_limits[i], i, distorted,
                        dist_cameras_mats[i], cameras_dists_coeff[i], undist_cameras_mats[i], cameras_roi[i])

        # Wait for a moment to allow each thread to start
        print(f"\r- Thread {i} for {cap} with {all_mp_poses[i]}")
        # time.sleep(0.1)

    # Main loop =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    all_triangulantion_time_ms = []
    all_3d_draw_time_ms = []
    while True:
        # Display the mosaic
        cv2.imshow('Multi-Thread: MediaPipe Pose-3D', mosaic)

        # Checking if all frames poses are detected
        if all(pose_output_buffer):
            pose_output = pose_output_buffer

            # Reset output for new poses
            pose_output_buffer = [None] * len(captures)

            start = time.time()
            # avg_3d_pose = pose_3d.calc_avg_3d_kpt(pose_output)
            best_3d_pose = pose_3d.calc_best_3d_kpt(pose_output)
            stop = time.time()
            # print(f"Triangulation time: {round((stop - start) * 1000, 4)} miliseconds")
            all_triangulantion_time_ms.append(round((stop - start) * 1000, 4))

            start = time.time()
            # pose_3d.update_3d_pose(avg_3d_pose, axes)
            pose_3d.update_3d_pose(best_3d_pose, axes)

            stop = time.time()
            # print(f"3d draw time: {round((stop - start) * 1000, 4)} miliseconds")
            all_3d_draw_time_ms.append(round((stop - start) * 1000, 4))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

print("mean triangulation time (ms):", sum(all_triangulantion_time_ms) / len(all_triangulantion_time_ms))
print("mean 3d draw time (ms):", sum(all_3d_draw_time_ms) / len(all_3d_draw_time_ms))

# Release video capture objects and close window
cv2.destroyAllWindows()
