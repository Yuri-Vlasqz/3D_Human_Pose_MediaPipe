from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import mpl_toolkits.mplot3d.axes3d
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np

from models import CamerasParameters

# todo test pytransform3d
#  [1]:https://dfki-ric.github.io/pytransform3d/index.html

# todo test camera representation (matplotlib & 3D)
#  [2]:https://dfki-ric.github.io/pytransform3d/_auto_examples/plots/plot_camera_3d.html#sphx-glr-auto-examples-plots-plot-camera-3d-py
#  [3]:https://dfki-ric.github.io/pytransform3d/_auto_examples/visualizations/vis_camera_3d.html#sphx-glr-auto-examples-visualizations-vis-camera-3d-py


class StageSpace:
    def __init__(self, camera_params: CamerasParameters, unit='cm',
                 xlim=(-200, 200), ylim=(0, -400), zlim=(-200, 200),
                 color='r', focal_len_scaled=50, aspect_ratio=0.3):
        # Create a new 3D plot
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=-30, azim=170, roll=-85)  # elev=90, azim=-90
        # ax.set_proj_type('persp', focal_length=0.2)  # FOV = 157.4 deg

        # Set labels for the axes (Panoptic in centimeters, DELED in milimeters)
        ax.set_xlabel(f'X {unit}')
        ax.set_ylabel(f'Y {unit}')
        ax.set_zlabel(f'Z {unit}')

        # Set limits for the x, y, and z axes
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)

        # set axes ticks
        ax.set_xticks(range(-200, 201, 100))
        ax.set_zticks(range(-200, 201, 100))
        ax.set_yticks(range(0, -401, -50))

        # Cameras pyramid FOV
        for t_vec, r_mat in zip(camera_params.translation_vecs, camera_params.rotation_mats):
            rot = np.array(r_mat)
            # print(r_mat.shape, type(r_mat))
            # print(t_vec.shape, type(t_vec))
            pos = np.array((-r_mat.transpose() * t_vec))
            extrinsic = np.concatenate([np.concatenate([rot.T, pos], axis=1), np.array([[0, 0, 0, 1]])], axis=0)

            vertex_std = np.array(
                [[0, 0, 0, 1],
                 [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                 [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                 [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                 [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1]]
            )
            vertex_transformed = vertex_std @ extrinsic.T

            meshes = [
                [vertex_transformed[0, :-1], vertex_transformed[1, :-1], vertex_transformed[2, :-1]],
                [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
                [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
                [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
                [vertex_transformed[1, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1],
                 vertex_transformed[4, :-1]]
            ]

            ax.add_collection3d(
                Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors='black', alpha=0.5)
            )

        plt.tight_layout()
        self.axes = ax

    def update_3d_pose(self, pose_3d: np.ndarray):
        # Removing old poses
        for artist in self.axes.lines:
            artist.remove()

        # Plotting new poses
        connections_frozenset = mp.solutions.pose.POSE_CONNECTIONS
        for i, connection in enumerate(connections_frozenset):
            x = [pose_3d[connection[0], 0], pose_3d[connection[1], 0]]
            y = [pose_3d[connection[0], 1], pose_3d[connection[1], 1]]
            z = [pose_3d[connection[0], 2], pose_3d[connection[1], 2]]
            color = plt.cm.get_cmap('hsv')(i / len(connections_frozenset))
            self.axes.plot(x, y, z, c=color, marker='.', markerfacecolor='black', markersize=5)

        plt.pause(0.0001)


if __name__ == "__main__":
    pass
