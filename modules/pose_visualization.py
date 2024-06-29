from typing import Optional

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d  # type: ignore
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore

from models import CamerasParameters
from modules import POSE_CONNECTIONS

# import matplotlib
# matplotlib.use('Qt5Agg')
# 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WX', 'WXAgg', 'WXCairo'


class PoseVisualization:
    def __init__(self, camera_params: Optional[CamerasParameters] = None,
                 camera_ids: Optional[list[int]] = None, unit='cm', title='3D Pose',
                 xlim=(-150, 150), ylim=(0, -300), zlim=(-150, 150),
                 elev=-30, azim=170, roll=-85,  # elev=-80, azim=90, roll=0,
                 color='g', focal_len_scaled=50, aspect_ratio=0.3):
        """
        Class for visualizing 3D space and updating 3D pose.
        """
        plt.ion()  # Interactive mode improves plotting speed
        fig = plt.figure(figsize=(8, 8))
        ax: axes3d.Axes3D = fig.add_subplot(111, projection='3d')
        # ax.set_proj_type('persp', focal_length=0.577)  # focal_length = 1/tan(FOV/2)

        # axes labels (Panoptic in cm)
        ax.set_xlabel(f' X {unit}')
        ax.set_ylabel(f' Y {unit}')
        ax.set_zlabel(f' Z {unit}')

        # axes limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        ax.text2D(0.5, 0.95, title, size=16, transform=ax.transAxes)
        # ax.text(0, 0, 0, '.', color='g')

        if camera_params is not None:
            # Cameras pyramid FOV
            vertex_std = np.array(
                [[0, 0, 0, 1],
                 [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                 [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                 [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                 [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]]
            )
            for t_vec, r_mat, camera_id in zip(
                    camera_params.translation_vecs, camera_params.rotation_mats, camera_ids
            ):
                rot = np.array(r_mat)
                pos = -rot.T @ t_vec
                x, y, z = pos[:, 0]
                ax.text(x, y, z, camera_id, size=12, color='red')

                extrinsic = np.concatenate(
                    [np.concatenate([rot.T, pos], axis=1), [[0, 0, 0, 1]]], axis=0)
                vertex_transformed = vertex_std @ extrinsic.T
                meshes = [
                    [vertex_transformed[0, :-1], vertex_transformed[1, :-1], vertex_transformed[2, :-1]],
                    [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                    [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                    [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                    [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1],
                     vertex_transformed[4, :-1]]
                ]
                ax.add_collection3d(Poly3DCollection(
                    meshes, facecolors=color, linewidths=0.3, edgecolors='black', alpha=0.4)
                )

        plt.tight_layout()

        self.angles = {'elev': elev, 'azim': azim, 'roll': roll}
        ax.view_init(**self.angles)
        self.axes: axes3d.Axes3D = ax
        self.pose_connections = POSE_CONNECTIONS

    def update_3d_pose(self, pose_3d: np.ndarray) -> None:
        """
        Updates the estimated 3D pose in the visualization.

        :param pose_3d: 3D mediapipe pose landmarks to be updated.
        """
        # Removing old pose
        for artist in self.axes.lines:
            artist.remove()

        # Plotting new pose
        for connection in self.pose_connections:
            xs, ys, zs = pose_3d[connection, :].T
            self.axes.plot(xs, ys, zs, c='r', linewidth=1.75,
                           marker='.', markerfacecolor='b', markersize=6)

    def update_mpjpe_poses(
            self, gt_pose: np.ndarray, estimated_pose: np.ndarray, matching_connections
    ) -> None:
        """
        Updates the 3D ground truth and estimated pose in the MPJPE comparison.

        :param gt_pose: Matching joints ground truth pose.
        :param estimated_pose: Matching joints estimated pose.
        :param matching_connections: Connections for the matching poses.
        """
        # Removing old poses
        for artist in self.axes.lines:
            artist.remove()

        # Plotting new poses
        for connection in matching_connections:
            xs, ys, zs = gt_pose[connection, :].T
            self.axes.plot(xs, ys, zs, color='g', linewidth=2.5, marker='d',
                           markerfacecolor='orange', markersize=2, alpha=0.6)

            xs, ys, zs = estimated_pose[connection, :].T
            self.axes.plot(xs, ys, zs, color='r', linewidth=1.5, marker='.',
                           markerfacecolor='b', markersize=4)

        self.angles['elev'] += 0.6
        if self.angles['elev'] >= 360:
            self.angles['elev'] = 1
        if 90 <= self.angles['elev'] <= 270:
            self.angles['azim'] = -1
        else:
            self.angles['azim'] = 1
        self.axes.view_init(**self.angles)

        plt.pause(0.001)


if __name__ == "__main__":
    pass
