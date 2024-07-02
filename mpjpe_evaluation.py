import time

import keyboard
import numpy as np

from modules import PoseVisualization
from utils import MATCHING_POSE_CONNECTIONS, compute_mpjpe, body3dscene_extractor


def mpjpe_evaluation(
        pose_directory: str = 'pose_output', sequence_name: str = '171026_pose3',
        camera_number: int = 9, model: int = 1, method: str = 'sqr_weighted',
        plot_mpjpe: bool = False
) -> None:

    gt_frames = np.load(f'{pose_directory}/Panoptic_GT_{sequence_name}.npz')['frames']
    gt_poses = np.load(f'{pose_directory}/Panoptic_GT_{sequence_name}.npz')['poses']

    pose_path = (f'{pose_directory}/NEW_MP_poses_{sequence_name}_{camera_number}_cams'
                 f'_model_{model}_method_{method}.npy')
    estimated_poses = np.load(pose_path)

    start = time.perf_counter()
    gt_matched_poses, estimated_matched_poses, mpjpe = compute_mpjpe(
        gt_poses, estimated_poses, gt_frames
    )
    mpjpe_time = time.perf_counter() - start
    print(f"{len(gt_frames)} poses compared in {mpjpe_time * 1000:.3f} ms\n"
          f"| Final MPJPE: {mpjpe * 10:.3f} mm |")

    if plot_mpjpe:
        pose_visualizer = PoseVisualization(
            xlim=(-85, 85), ylim=(0, -170), zlim=(-85, 85),
            **{'elev': 0, 'azim': 0, 'roll': 90}, title='3D MPJPE'
        )
        draw_times = []
        start = time.perf_counter()
        for gt_pose, est_pose in zip(gt_matched_poses, estimated_matched_poses):
            try:
                if keyboard.is_pressed('q'):
                    print('\nPoses comparison stopped!')
                    break
            finally:
                pose_visualizer.update_mpjpe_poses(
                    gt_pose, est_pose, MATCHING_POSE_CONNECTIONS
                )

            stop = time.perf_counter()
            draw_times.append(stop - start)
            start = stop

        print(f"Mean draw time: {np.mean(draw_times[1:]) * 1000:.3f} ms")


if __name__ == "__main__":
    # Sequences: '171204_pose1_sample' and '171026_pose3'
    body3dscene_extractor(sequence_name='171026_pose3')  # Extract sequence GT
    mpjpe_evaluation(
        sequence_name='171026_pose3',
        camera_number=9,
        model=1,
        method='sqr_weighted',
        plot_mpjpe=True
    )
    # mean mpjpe draw time: ~45 ms
    # mean FPS: ~22
