"""
Panoptic - camera parameters extraction
"""
import numpy as np
import json


def get_panoptic_parameters(cam_list: list, directory: str = "Dataset/Panoptic/", seq_name: str = '171026_pose3'):
    """get a selected list of HD camera parameters (intrinsic and extrinsic)"""

    # Load camera calibration parameters (for visualizing cameras)
    with open(f'{directory}{seq_name}/calibration_{seq_name}.json') as cfile:
        calib = json.load(cfile)

    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}

    # Choose only selected list of HD cameras
    hd_cam_idx = zip([0] * 30, cam_list)
    hd_cameras = [cameras[cam].copy() for cam in hd_cam_idx]

    camera_matrices = []
    camera_distortions = []
    rotational_matrices = []
    translation_vectors = []

    print("Selected Panoptic cameras:")
    for cam in hd_cameras:
        # Convert data into numpy arrays for convenience
        camera_matrices.append(np.matrix(cam['K']))
        camera_distortions.append(np.array(cam['distCoef']))
        rotational_matrices.append(np.matrix(cam['R']))
        translation_vectors.append(np.array(cam['t']).reshape((3, 1)))

        print(f"- {cam['type']}_{cam['name']}")

    return rotational_matrices, translation_vectors, camera_matrices, camera_distortions


if __name__ == "__main__":
    # test
    get_panoptic_parameters([0, 10, 20, 30], directory="", seq_name='171026_pose3')
    print("--------------------------")
