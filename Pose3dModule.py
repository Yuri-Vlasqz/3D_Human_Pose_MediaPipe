import numpy as np
import mediapipe as mp
import cv2
from itertools import combinations
import matplotlib.pyplot as plt


class MpPose2D:
    def __init__(self):
        # MediaPipe 2D Pose Model (https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/index#models)
        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawingStyles = mp.solutions.drawing_styles
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            model_complexity=1,  # 1 = Pose landmarker (Full)
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )

    @staticmethod
    def mp_landmarks_to_list(pose_landmarks, image_width, image_height):
        landmarks_list = []
        for norm_point in pose_landmarks.landmark:
            landmarks_list.append({
                'pixel_xy': [norm_point.x * image_width, norm_point.y * image_height],
                'visibility': norm_point.visibility
            })
        
        return landmarks_list

    def calc_2d_kpt(self, image, draw=True):
        # image resolution
        h, w = image.shape[:2]

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Pose Inference
        results = self.pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Pose Landmarks list in sub-pixels coordinates
            landmarks_list = self.mp_landmarks_to_list(results.pose_landmarks, image_width=w, image_height=h)
            
            if draw:
                # Draw the pose landmarks on the image.
                self.mpDraw.draw_landmarks(
                    image=image,
                    landmark_list=results.pose_landmarks,
                    connections=self.mpPose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mpDrawingStyles.get_default_pose_landmarks_style()
                )

            return image, landmarks_list

        print("\nNo landmarks found")

        return image, None


class Pose3D:
    def __init__(self, camera_mats, rot_mats, trans_vecs, min_visibility=0.6):
        # Camera Parameters
        self.C_mats = camera_mats
        self.R_mats = rot_mats
        self.T_vecs = trans_vecs
        self.P_mats = self.get_projection_matrices()

        # Cameras unique pairs combination
        self.pairs = list(combinations(range(len(camera_mats)), 2))
        print(f"{len(self.pairs)} camera pairs combinations")
        self.min_visibility = min_visibility
        print(f"Minimun visibility threshold: {self.min_visibility}")

    def get_projection_matrices(self):
        p_mats = []
        for cmtx, R, t in zip(self.C_mats, self.R_mats, self.T_vecs):
            # Essential matrix
            e_mat = np.zeros((4, 4))
            e_mat[:3, :3] = R
            e_mat[:3, 3] = t.reshape(3)
            e_mat[3, 3] = 1

            # Calculate projection matrix: K.[R|t]
            p = cmtx @ e_mat[:3, :]
            p_mats.append(p)

        return p_mats

    def cv2_3d_pose_pair(self, pair, poses):
        i, j = pair
        points1 = np.array([d['pixel_xy'] for d in poses[i]]).T
        points2 = np.array([d['pixel_xy'] for d in poses[j]]).T

        # Triangulate from a pair of pose joints
        homog_points_3d = cv2.triangulatePoints(self.P_mats[i], self.P_mats[j], points1, points2)
        # Internally it uses DLT method: https://docs.opencv.org/4.7.0/d0/dbd/group__triangulation.html
        # Homgeneous points are returned
        euclid_points_3d = cv2.convertPointsFromHomogeneous(np.transpose(homog_points_3d))

        return euclid_points_3d

    def calc_avg_3d_kpt(self, all_2d_poses):  # Not recomended, debugging required !!!
        
        all_3d_poses = []
        all_min_vis = []
        
        for i, j in self.pairs:
            # triangulate 3d points per camera pair
            pose_3d_pair = self.cv2_3d_pose_pair(pair=[i, j], poses=all_2d_poses)
            # List o minimum visibility of the pair in each joint
            min_visibility_pair = [min(joint1['visibility'], joint2['visibility'])
                                   for joint1, joint2 in zip(all_2d_poses[i], all_2d_poses[j])]
            all_min_vis.append(min_visibility_pair)
            all_3d_poses.append(pose_3d_pair)

        # empty average 3d pose
        avg_3d_pose = np.zeros((len(all_min_vis[0]), 3))
        # empty visibility inliers count list
        joints_inliers = np.zeros((len(all_min_vis[0]), 1))

        for pose_pair, pair_vis in enumerate(all_min_vis):
            for joint, visibility in enumerate(pair_vis):
                # Only consider joints above minimum visibility threshold
                if visibility >= self.min_visibility:
                    # Cumulative sum of 3d joint coordinates
                    avg_3d_pose[joint] += all_3d_poses[pose_pair][joint, 0]
                    # Cumulative Count of inliers
                    joints_inliers[joint] += 1

        if any(joints_inliers == 0):
            # replace any zero inliers count with ones (to prevent zero division error)
            np.place(joints_inliers[:, None], joints_inliers[:, None] == 0, [1])

        # Calculation of average pose with joints above visibility threshold
        avg_3d_pose = avg_3d_pose / joints_inliers[:, None]

        return avg_3d_pose[0]

    def calc_best_3d_kpt(self, all_2d_poses):
        all_3d_poses = []
        all_min_vis = []
        for i, j in self.pairs:
            # triangulate 3d points per camera pair
            pose_3d_pair = self.cv2_3d_pose_pair(pair=[i, j], poses=all_2d_poses)
            # list o minimum visibility of the pair in each joint
            min_visibility_pair = [min(joint1['visibility'], joint2['visibility'])
                                   for joint1, joint2 in zip(all_2d_poses[i], all_2d_poses[i])]
            all_min_vis.append(min_visibility_pair)
            all_3d_poses.append(pose_3d_pair)

        # Finding array/pose index with best visisibility in all all_min_vis
        best_pair_idx = np.array(all_min_vis).argmax(axis=0)
        # 3d pose with *best* visibility for each joint
        best_3d_pose = [all_3d_poses[pair_idx][joint_idx, 0] for joint_idx, pair_idx in enumerate(best_pair_idx)]

        return np.array(best_3d_pose)

    def create_3d_space(self):
        camera_positions = self.T_vecs
        camera_orientations = self.R_mats
        # Create a new 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=90, azim=-90)
        # Set labels for the axes
        ax.set_xlabel('Z')
        ax.set_ylabel('Y')
        ax.set_zlabel('X')
        
        # set axes ticks (Panoptic in centimeters)
        # Set limits for the x, y, and z axes
        ax.set_xlim((-200, 200))
        ax.set_zlim((-200, 200))
        ax.set_ylim((000, -400))

        # Draw cameras positions
        for i, camera in enumerate(camera_positions):
            # Add the points to the plot
            # panoptic
            cc = (-camera.transpose() * camera_orientations[i])
            ax.scatter(xs=cc[0, 0], ys=cc[0, 1], zs=cc[0, 2], marker='.', linewidths=6)

        return ax

    @staticmethod
    def update_3d_pose(pose_3d, axes):
        
        for artist in axes.lines:
            artist.remove()

        connections_frozenset = mp.solutions.pose.POSE_CONNECTIONS
        # Plotting lines
        for i, connection in enumerate(connections_frozenset):
            x = [pose_3d[connection[0], 0], pose_3d[connection[1], 0]]
            y = [pose_3d[connection[0], 1], pose_3d[connection[1], 1]]
            z = [pose_3d[connection[0], 2], pose_3d[connection[1], 2]]
            color = plt.cm.get_cmap('hsv')(i / len(connections_frozenset))
            axes.plot(x, y, z, c=color, marker='.', markerfacecolor='black', markersize=5)

        plt.pause(0.0001)

        
