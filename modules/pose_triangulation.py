import numpy as np
import cv2

from itertools import combinations

from models import CamerasParameters


class PoseTriangulation:
    def __init__(self, cameras_parameters: CamerasParameters):
        self.cameras_params = cameras_parameters
        self.pairs = list(combinations(iterable=range(cameras_parameters.camera_count), r=2))

    def __repr__(self):
        return (f"Pose Triangulation ----------------\n"
                f"{self.cameras_params}\n"
                f"camera unique pairs: \t{len(self.pairs)}\n")

    def dlt_landmarks(self, pair: list[int, int], poses: list[np.ndarray, np.ndarray]) -> np.ndarray:
        """ Triangulation by DLT method:
        https://docs.opencv.org/4.8.0/d0/dbd/group__triangulation.html"""
        i, j = pair
        homog_points_3d = cv2.triangulatePoints(self.cameras_params.projection_mats[i],
                                                self.cameras_params.projection_mats[j],
                                                poses[0].T, poses[1].T)
        euclid_points_3d = cv2.convertPointsFromHomogeneous(homog_points_3d.T)
        return euclid_points_3d[:, 0, :]

    def triangulate_best_landmarks(self, all_2d_poses: list[np.ndarray]) -> np.ndarray:
        all_3d_poses = []
        all_min_vis = []

        for i, j in self.pairs:
            if (all_2d_poses[i].size == 0) or (all_2d_poses[j].size == 0):
                continue  # skip pair with empty element

            pose_3d_pair = self.dlt_landmarks(pair=[i, j],
                                              poses=[all_2d_poses[i][:, :2], all_2d_poses[j][:, :2]])
            min_visibility_pair = np.minimum(all_2d_poses[i][:, 2:], all_2d_poses[j][:, 2:])[:, 0]

            all_min_vis.append(min_visibility_pair)
            all_3d_poses.append(pose_3d_pair)

        # Best pose based on landmark visibility
        best_3d_pose_idxs = np.array(all_min_vis).argmax(axis=0)
        best_3d_pose = [all_3d_poses[pose_idx][landmark_idx]
                        for landmark_idx, pose_idx in enumerate(best_3d_pose_idxs)]
        return np.array(best_3d_pose)


if __name__ == "__main__":
    pass
