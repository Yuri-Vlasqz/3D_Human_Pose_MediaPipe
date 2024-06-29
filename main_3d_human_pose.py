import concurrent.futures as futures
import os
import threading
import time
from copy import deepcopy

import cv2
import numpy as np
from matplotlib import pyplot as plt

from models import MosaicStructure
from modules import PoseDetection
from utils import ThreadSpecificationLoader, execution_time_report


def process_feed(
        feed: cv2.VideoCapture,
        pose_detector: PoseDetection,
        poses_buffer: list[np.ndarray],
        mosaic_structure: MosaicStructure,
        index: int, camera_id: int,
        inference_times: list,
        thread_lock: threading.Lock,
        thread_barrier: threading.Barrier
) -> None:
    """ Function to detect and overlay pose from calibrated camera feed """

    tile_height, tile_width = mosaic_structure.tile_resolution
    inference_times_buffer: list[float] = []

    # | Detection thread Loop |
    while True:
        ret, frame = feed.read()
        if not ret:
            try:
                thread_barrier.wait()
                continue  # skip to next frame
            except threading.BrokenBarrierError:  # Stop of thread
                break
        # if distorted_feed:  # Undistort alpha = 0
        #     frame = cv2.undistort(
        #         frame, dist_cam_mat, dist_coeffs, None, undistorted_cam_mat
        #     )
        # Resize downsampling interpolation:
        # cv2.INTER_NEAREST -> fastest but less precise
        # cv2.INTER_LINEAR  -> fast & precise (default)
        # cv2.INTER_AREA    -> slow with best precision
        frame = cv2.resize(
            frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA
        )
        inference_start = time.perf_counter()
        landmarks = pose_detector.detect_landmarks(frame)
        inference_times_buffer.append(time.perf_counter() - inference_start)
        overlay = pose_detector.overlay_landmarks(frame)

        with thread_lock:
            poses_buffer[index] = landmarks
            mosaic_structure.insert_tile_content(
                index, overlay, f"Camera {camera_id}"
            )
        try:
            thread_barrier.wait()
        except threading.BrokenBarrierError:  # Stop of thread
            break

    feed.release()
    with thread_lock:
        print(f"- Feed {index + 1} stopped")
        inference_times.extend(inference_times_buffer)


def main() -> None:
    """
    The main function that runs the MultiThreaded 3D Human Pose Estimation application.

    This function loads specifications from "inicialization.yaml", starts the pose
    detection thread pool, performs triangulation and visualize the best 3D pose.

    **OBS**: All thread loops runs continuously until all feeds have ended or
    the user presses 'q'. A performance report is printed before exiting.
    """
    specs = ThreadSpecificationLoader("inicialization.yaml")
    print(specs)

    barrier_wait_times: list[float] = []
    inference_times: list[float] = []
    triangulation_times: list[float] = []
    draw_times: list[float] = []
    mosaic_times: list[float] = []
    all_3d_poses: list[np.ndarray] = []
    current_3d_pose: np.ndarray = np.zeros((33, 3))
    current_frame = 0
    max_frame = specs.frame_count

    # Pose detection thread pool
    with futures.ThreadPoolExecutor() as executor:
        print("Initializing inference threads:")
        for i, cap in enumerate(specs.video_captures):
            executor.submit(
                process_feed, cap,
                specs.pose_detectors[i],
                specs.pose_buffers,
                specs.mosaic_structure,
                i, specs.cameras_idx[i],
                inference_times,
                specs.thread_lock,
                specs.thread_barrier,
            )
            print(f"- Thread {i + 1}: {specs.pose_detectors[i]}")

        print("\nRunning")
        start_mosaic_time = time.perf_counter()

        # | Main thread loop |
        while True:
            try:
                start = time.perf_counter()
                specs.thread_barrier.wait()  # Wait all threads
                barrier_wait_times.append(time.perf_counter() - start)

                with specs.thread_lock:
                    poses_output = deepcopy(specs.pose_buffers)

            except threading.BrokenBarrierError:  # Stop of thread
                break

            num_poses = sum([1 for arr in poses_output if arr.size])
            if num_poses >= 2:
                start = time.perf_counter()
                current_3d_pose = specs.pose_triangulator.triangulate_landmarks(
                    poses_output, method=specs.triangulation_method
                )  # methods: 'mean', 'weighted', 'sqr_weighted', 'best'
                triangulation_times.append(time.perf_counter() - start)

                start = time.perf_counter()
                specs.pose_visualizer.update_3d_pose(current_3d_pose)
                draw_times.append(time.perf_counter() - start)

            all_3d_poses.append(current_3d_pose)

            stop_mosaic_time = time.perf_counter()
            mosaic_times.append(stop_mosaic_time - start_mosaic_time)
            if not current_frame % 5:
                mosaic_fps = 1 / (stop_mosaic_time - start_mosaic_time)
            start_mosaic_time = stop_mosaic_time

            cv2.putText(
                specs.mosaic_structure.mosaic_content, f"FPS: {mosaic_fps:.0f}",
                (5, 25), specs.text['font'], specs.text['size'], specs.text['color'],
                specs.text['thickness'], specs.text['line_type']
            )
            # Display Mosaic
            cv2.imshow(
                winname='MediaPipe Human Pose',
                mat=specs.mosaic_structure.mosaic_content
            )
            if cv2.waitKey(1) & 0xFF == ord('q') or current_frame == max_frame:
                break

            current_frame += 1

        # Stop threads
        specs.thread_barrier.abort()

    # Close Mosaic and 3D Pose window
    cv2.destroyAllWindows()
    plt.close('all')

    print("\n| Performance report |\n")
    execution_time_report(barrier_wait_times[1:], "Barrier wait time")
    execution_time_report(inference_times, "Inference time")
    execution_time_report(triangulation_times, "Triangulation time")
    execution_time_report(draw_times, "3D draw time")
    execution_time_report(mosaic_times[1:], "Mosaic time")

    hd_pose3d_path = 'dataset/Panoptic/'+specs.sequence_name+'/hdPose3d_stage1_coco19/'
    files_list = os.listdir(hd_pose3d_path)
    idx_stop = int(files_list[-1][12:20])  # 171026_pose3: 129-7309
    if len(all_3d_poses) > idx_stop:
        is_sure: bool = input(
            f'Save {len(all_3d_poses)} poses? (y/n): '
        ).lower().strip() == 'y'
        if is_sure:
            output_file_path = (
                f"pose_output/NEW_MP_poses_{specs.sequence_name}_{len(specs)}_cams_"
                f"model_{specs.pose_model}_method_{specs.triangulation_method}.npy"
            )
            np.save(output_file_path, np.array(all_3d_poses))
            print(f"All Poses Saved!\nShape:{np.load(output_file_path).shape}")


if __name__ == "__main__":
    main()
