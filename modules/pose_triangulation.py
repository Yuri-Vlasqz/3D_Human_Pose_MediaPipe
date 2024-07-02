from itertools import combinations

import cv2
import numpy as np

from models import CamerasParameters


class PoseTriangulation:
    def __init__(self, cameras_parameters: CamerasParameters):
        """
        Class for triangulation of 2D poses using camera's projection matrices.

        :param cameras_parameters: Dataclass with all camera's parameters.
        """
        self.cameras_params = cameras_parameters
        self.pairs = list(
            combinations(iterable=range(cameras_parameters.camera_count), r=2)
        )

    def __repr__(self):
        return (f"Pose Triangulation ------\n"
                f"- Camera unique pairs: {len(self.pairs)}")

    def __dlt_landmarks(self, pair: list[int], poses: list[np.ndarray]) -> np.ndarray:
        """ Triangulation of 2D pose pair by Direct Linear Transform (DLT).
        Reference: https://docs.opencv.org/4.8.0/d0/dbd/group__triangulation.html

        :param pair: Pair of camera indices.
        :param poses: Pair of 2D pose landmarks.
        :return: Triangulated 3D pose landmarks.
        """
        i, j = pair
        homog_points_3d = cv2.triangulatePoints(self.cameras_params.projection_mats[i],
                                                self.cameras_params.projection_mats[j],
                                                poses[0].T, poses[1].T)
        euclid_points_3d = cv2.convertPointsFromHomogeneous(homog_points_3d.T)
        return euclid_points_3d[:, 0, :]

    def triangulate_landmarks(
            self, mp_2d_poses: list[np.ndarray], method: str = 'mean'
    ) -> np.ndarray:
        """
        Triangulates landmarks from 2D poses using a chosen combination method.

        - mean: Simple average of all 3D landmarks.
        - weighted: Weighted average using the visibility scores.
        - sqr_weighted: Weighted average using the square of the visibility scores.
        - best: Selects the 3D landmark with the best visibility score for each joint.

        :param mp_2d_poses: A list of 2D mediapipe poses, each pose is a numpy array of shape (33, 3).
        :param method: Optional: Method for final 3D pose computation.
        :return: Final 3D pose, Numpy array of shape (33, 3).
        """
        mp_3d_poses, min_vis = zip(
            *[(self.__dlt_landmarks(
                pair=[i, j], poses=[mp_2d_poses[i][:, :2], mp_2d_poses[j][:, :2]]
            ),
               np.minimum(mp_2d_poses[i][:, 2:], mp_2d_poses[j][:, 2:])[:, 0])
                for i, j in self.pairs if mp_2d_poses[i].size and mp_2d_poses[j].size]
        )
        mp_3d_poses = np.array(mp_3d_poses)
        min_vis = np.array(min_vis)

        if method == 'weighted':
            min_vis = min_vis.repeat(3, axis=1).reshape(mp_3d_poses.shape)
            return np.average(mp_3d_poses, axis=0, weights=min_vis)

        if method == 'sqr_weighted':
            min_vis = min_vis.repeat(3, axis=1).reshape(mp_3d_poses.shape)
            return np.average(mp_3d_poses, axis=0, weights=np.square(min_vis))

        if method == 'best':
            best_3d_pose_idxs = min_vis.argmax(axis=0)
            return np.array(
                [mp_3d_poses[pose_idx][landmark_idx]
                 for landmark_idx, pose_idx in enumerate(best_3d_pose_idxs)]
            )

        if method == 'mean':
            return mp_3d_poses.mean(axis=0)

        print(f"{method} Method not found")
        return np.zeros((33, 3))


if __name__ == "__main__":
    pass
