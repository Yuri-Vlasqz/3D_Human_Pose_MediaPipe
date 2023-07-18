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

# Global Variables =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Define RTSP camera URLs
cam_urls = ['Dataset/Panoptic/171026_pose3/hdVideos/hd_00_00.mp4',
            'Dataset/Panoptic/171026_pose3/hdVideos/hd_00_01.mp4',
            'Dataset/Panoptic/171026_pose3/hdVideos/hd_00_06.mp4',
            'Dataset/Panoptic/171026_pose3/hdVideos/hd_00_13.mp4',
            'Dataset/Panoptic/171026_pose3/hdVideos/hd_00_16.mp4',
            'Dataset/Panoptic/171026_pose3/hdVideos/hd_00_21.mp4']

# Define the number of rows and columns in the frame
num_rows = 2
num_cols = 3
frame_width = 480
frame_height = 270

# FPS display variables
font = cv2.FONT_HERSHEY_SIMPLEX
line_type = cv2.LINE_AA
size = 0.6

# Thread synchronization Barriers
sync_pose = threading.Barrier(len(cam_urls)+1)


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
                 dist_cam_mat, dist_coeffs, undistorted_cam_mat):
    """ Function to process each camera cam"""

    # Global variables
    global frame_width
    global frame_height
    global mosaic
    global pose_output_buffer
    global sync_pose
    global thread_lock
    global all_inference_times

    # Get the location in the frame to place the frame
    y1, y2, x1, x2, yc, xc = frame_limits

    timeout = False

    # Initialize the previous time for FPS calculation
    prev_time = time.time()

    # Loop over the frames from one camera =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    while True:
        # Read a frame from the camera
        ret, frame = feed.read()

        if not ret:

            if timeout:
                continue  # skip to the next frame

            # print(f"Could not read frame from {cap}.")
            cv2.putText(mosaic, f"Feed {index} not found:", (xc - 70, yc), font, size, (0, 0, 255), 1, line_type)
            cv2.putText(mosaic, f"{feed}", (x1 + 10, y2 - 35), font, size, (0, 0, 255), 1, line_type)
            timeout = True
            continue  # skip to the next frame

        timeout = False

        if distorted_feed:
            # Undistort the frame (for alpha = 0)
            frame = cv2.undistort(frame, dist_cam_mat, dist_coeffs, None, undistorted_cam_mat)

        inference_start = time.time()
        # Process MediaPipe-Pose backend
        pose_frame, landmarks_list = pose_2d.calc_2d_kpt(frame)
        inference_stop = time.time()

        # Resize the frame to match the size of the frame
        pose_frame = cv2.resize(pose_frame, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        with thread_lock:
            # update the output 2d pose list
            pose_output_buffer[index] = landmarks_list

            # Update the frame with the frame with 2d joints
            mosaic[y1:y2, x1:x2] = pose_frame

            all_inference_times.append(inference_stop - inference_start)

            # Calculate and display the FPS for this camera stream
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            cv2.putText(mosaic, f"FPS: {fps:.0f}", (x1 + 10, y2 - 10), font, size, (255, 255, 255), 1, line_type)

        prev_time = time.time()

        try:
            sync_pose.wait()
        except threading.BrokenBarrierError:  # Stop of thread
            break

    # Release the video capture object for this camera cam
    feed.release()


# Create video capture objects for each camera =-=-=-=-=-=-=-=-=-=-=-=
captures = [cv2.VideoCapture(url, cv2.CAP_FFMPEG) for url in cam_urls]
# for videos: cv2.CAP_MSMF

# set the start time for recorded videos
# for cap in captures:
#    cap.set(cv2.CAP_PROP_POS_FRAMES, 1700)

# List of MpPose2D objects for each capture
all_mp_poses = [MpPose2D() for _ in captures]

# Camera parameters =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Panoptic
rotational_matrices, translation_vectors, camera_matrices, camera_distortions = get_panoptic_parameters(
    [0, 1, 6, 13, 16, 21]
)
distorted = False

# Create 3d pose object =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
pose_3d = Pose3D(camera_mats=camera_matrices,
                 rot_mats=rotational_matrices,
                 trans_vecs=translation_vectors,
                 min_visibility=0.5)

# Create an empty frame structure
mosaic, frames_limits = create_mosaic_structure(len(captures), num_rows, num_cols, frame_height, frame_width)

# 2d pose buffer
pose_output_buffer = [None] * len(captures)

# 3d space canvas
axes = pose_3d.create_3d_space()

# Thread synchronization primitive (to make code thread safe)
thread_lock = threading.Lock()

# Cameras Thread pool
with concurrent.futures.ThreadPoolExecutor() as executor:
    print("\nInitializing threads for processing feeds:")
    for i, cap in enumerate(captures):
        # Submit task for each camera
        executor.submit(process_feed, cap, all_mp_poses[i], frames_limits[i], i, distorted,
                        None, None, None)
        print(f"- Thread {i} for {cap} with {all_mp_poses[i]}")

    # Main loop =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    all_triangulantion_time_ms = []
    all_3d_draw_time_ms = []
    all_mosaic_time = []
    all_inference_times = []

    start_mosaic_time = time.time()
    while True:

        sync_pose.wait()
        buffered_mosaic = mosaic.copy()

        # number of detected poses
        n_poses = sum(x is not None for x in pose_output_buffer)

        # Checking if at least 2 poses are detected
        if n_poses >= 2:
            # Copy and Reset output for new poses
            pose_output = pose_output_buffer.copy()
            pose_output_buffer = [None] * len(captures)

            start = time.time()
            best_3d_pose = pose_3d.dynamic_best_3d_kpt(pose_output)
            stop = time.time()
            all_triangulantion_time_ms.append(stop - start)

            start = time.time()
            pose_3d.update_3d_pose(best_3d_pose, axes)
            stop = time.time()
            all_3d_draw_time_ms.append(stop - start)

        stop_mosaic_time = time.time()
        mosaic_fps = 1 / (stop_mosaic_time - start_mosaic_time)
        cv2.putText(buffered_mosaic, f"FPS: {mosaic_fps:.1f}", (frame_width - 90, frame_height - 10),
                    font, size, (0, 0, 255), 1, line_type)
        all_mosaic_time.append(stop_mosaic_time - start_mosaic_time)
        start_mosaic_time = stop_mosaic_time

        # Display the frame
        cv2.imshow('Multi-Thread: MediaPipe Pose-3D', buffered_mosaic)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop waiting threads
    sync_pose.abort()

# Performance Report =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
print("\n--- Performance Statistics ---")

print("mean inference time (ms):\t\t",
      round((sum(all_inference_times) / len(all_inference_times))*1000, 3))

print("mean triangulation time (ms):\t",
      round((sum(all_triangulantion_time_ms) / len(all_triangulantion_time_ms))*1000, 3))

print("mean 3d draw time (ms):\t\t\t",
      round((sum(all_3d_draw_time_ms) / len(all_3d_draw_time_ms))*1000, 3))

print("mean frame time (ms):\t\t\t",
      round((sum(all_mosaic_time) / len(all_mosaic_time))*1000, 3))

# Close Mosaic window
cv2.destroyAllWindows()
