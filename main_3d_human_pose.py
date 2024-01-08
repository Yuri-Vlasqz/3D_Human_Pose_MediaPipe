import numpy as np
import cv2

import concurrent.futures as futures
from copy import deepcopy
import threading
import time

from modules import PoseDetection, PoseTriangulation, StageSpace
from models import CamerasParameters, MosaicStructure
from utils import execution_time_report, yaml_parser, panoptic_parser

# Multithread global variables
pose_output_buffer = []
inference_times = []


def initialise(specs_path: str = "inicialization.yaml"):
    """ Function to inicialise from specifications file"""

    inicialization_specs = yaml_parser(specs_path)
    feeds, img_res = inicialization_specs['feeds'].values()
    tile_grid, tile_res = inicialization_specs['mosaic'].values()
    text_variables = list(inicialization_specs['text'].values())

    video_captures = [cv2.VideoCapture(feed) for feed in feeds]
    # alternative: cv2.CAP_MSMF (videos only)
    frame_count = video_captures[0].get(cv2.CAP_PROP_FRAME_COUNT)
    # set the start time for recorded videos
    # for cap in video_captures:
    #     cap.set(cv2.CAP_PROP_POS_FRAMES, 7980)

    mosaic_structure = MosaicStructure(len(feeds), tile_grid, tile_res)
    pose_detection = [PoseDetection(img_res, 1, 0.8) for _ in feeds]

    # Panoptic calibration parameters extraction
    rotational_matrices, translation_vectors, camera_matrices, camera_distortions = panoptic_parser(
        [0, 1, 6, 13]
    )  # 0, 1, 6, 13, 16, 21
    cameras_params = CamerasParameters(len(feeds),
                                       camera_matrices,
                                       rotational_matrices,
                                       translation_vectors,
                                       camera_distortions)
    pose_triangulation = PoseTriangulation(cameras_params)
    stage_space = StageSpace(cameras_params)

    global pose_output_buffer
    pose_output_buffer = [np.empty(0)] * len(feeds)

    thread_barrier = threading.Barrier(len(feeds) + 1)
    thread_lock = threading.Lock()

    return (video_captures, mosaic_structure, pose_detection, pose_triangulation,
            stage_space, thread_barrier, thread_lock, text_variables, frame_count)


def process_feed(feed: cv2.VideoCapture, pose_detector: PoseDetection, mosaic_structure: MosaicStructure,
                 index: int, thread_lock: threading.Lock, thread_barrier: threading.Barrier):
    """ Function to detect pose from calibrated camera feed """

    global pose_output_buffer
    global inference_times
    frame_height, frame_width = mosaic_structure.tile_resolution

    # | Detection thread Loop |
    while True:
        ret, frame = feed.read()

        if not ret:
            try:
                thread_barrier.wait()
            except threading.BrokenBarrierError:  # Stop of thread
                break
            continue  # skip to next frame

        # if distorted_feed:
        #     # Undistort alpha = 0
        #     frame = cv2.undistort(frame, dist_cam_mat, dist_coeffs, None, undistorted_cam_mat)

        inference_start = time.time()
        landmarks = pose_detector.detect_landmarks(frame)
        inference_stop = time.time()

        frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
        overlay = pose_detector.overlay_landmarks(frame)

        with thread_lock:
            pose_output_buffer[index] = landmarks
            inference_times.append(inference_stop - inference_start)
            mosaic_structure.insert_tile_content(index, overlay)

        try:
            thread_barrier.wait()
        except threading.BrokenBarrierError:  # Stop of thread
            break

    if not ret:  # first to end feed
        with thread_lock:
            print(f"Feed {index} ended")
        thread_barrier.abort()
    feed.release()


def main():
    (captures, mosaic,
     pose_detectors, pose_triangulator,
     stage_vizualiser,
     sync_barrier, sync_lock,
     TEXT, FRAME_COUNT) = initialise()

    global pose_output_buffer
    global inference_times

    # Pose detection thread pool
    with futures.ThreadPoolExecutor() as executor:
        print("\nInitializing feed processing threads:")
        for i, cap in enumerate(captures):
            executor.submit(process_feed, cap, pose_detectors[i], mosaic, i, sync_lock, sync_barrier)
            print(f"- Thread {i}: {pose_detectors[i]}")

        triangulation_times = []
        draw_times = []
        mosaic_times = []
        start_mosaic_time = time.time()

        # | Main thread loop |
        while True:

            try:
                # Wait all threads
                sync_barrier.wait()
                with sync_lock:
                    pose_output = deepcopy(pose_output_buffer)

            except threading.BrokenBarrierError:  # Stop of thread
                break

            num_poses = sum([1 for arr in pose_output if arr.size > 0])
            if num_poses >= 2:
                start = time.time()
                best_3d_pose = pose_triangulator.triangulate_best_landmarks(pose_output)
                stop = time.time()
                triangulation_times.append(stop - start)

                start = time.time()
                stage_vizualiser.update_3d_pose(best_3d_pose)
                stop = time.time()
                draw_times.append(stop - start)

            stop_mosaic_time = time.time()
            mosaic_fps = 1 / (stop_mosaic_time - start_mosaic_time)
            cv2.putText(mosaic.mosaic_content, f"FPS: {mosaic_fps:.0f}", (5, 20),
                        TEXT[0], TEXT[1], TEXT[2], TEXT[3], TEXT[4])
            mosaic_times.append(stop_mosaic_time - start_mosaic_time)
            start_mosaic_time = stop_mosaic_time

            # Display the frame
            cv2.imshow('MediaPipe Human Pose-3D', mosaic.mosaic_content)

            if captures[0].get(cv2.CAP_PROP_POS_FRAMES) == FRAME_COUNT:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        # Stop threads
        sync_barrier.abort()

    # Close Mosaic window
    cv2.destroyAllWindows()
    # inference_times, triangulation_times, draw_times, mosaic_times = main()

    print("\n| Performance report |\n")
    execution_time_report(inference_times, "Inference time", tails=False)
    execution_time_report(triangulation_times, "Triangulation time", tails=False)
    execution_time_report(draw_times, "3D draw time", tails=False)
    execution_time_report(mosaic_times, "Mosaic time", tails=False)


if __name__ == "__main__":
    main()
