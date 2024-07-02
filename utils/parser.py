import json
import os

import cv2
import numpy as np
import yaml

RED = (0, 0, 255)
WHITE = (255, 255, 255)
DEFAULT_SPECS = {
    "title": "Inicialization Specifications",
    "feeds": {
        "directory": "dataset/Panoptic/171026_pose3/hdVideos/",
        "videos": ['hd_00_00.mp4', 'hd_00_01.mp4', 'hd_00_03.mp4',
                   'hd_00_06.mp4', 'hd_00_13.mp4', 'hd_00_16.mp4',
                   'hd_00_21.mp4', 'hd_00_25.mp4', 'hd_00_30.mp4'],
        "resolution": [1080, 1920]
    },
    "pose_landmarker": {
        "model": 1,
        "visibility_threshold": 0.75,
        "triangulation_method": 'sqr_weighted'
    },
    "mosaic": {
        "tile_grid": [3, 3],
        "tile_resolution": [270, 480]
    },
    "text": {
        "font": cv2.FONT_HERSHEY_SIMPLEX,
        "size": 0.75,
        "color": WHITE,
        "thickness": 1,
        "line_type": cv2.LINE_AA,
    }
}


def yaml_parser(file_path: str) -> dict:
    """
    Parses a YAML file and returns its contents as a dictionary.
    If the file is not found, it creates a new file with default values.

    :param file_path: The path to the YAML file.
    :return: The contents of the YAML file as a dictionary.
    :raises YAMLError: If there is an error parsing the YAML file
    """
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        if data is None:
            print("Data in ", end="")
            raise FileNotFoundError  # No data, recreate file

    except FileNotFoundError:
        print(f"'{file_path}' not found.")
        data = DEFAULT_SPECS
        with open(file_path, 'w') as file:
            yaml.safe_dump(
                data, file, sort_keys=False, default_flow_style=False, indent=4
            )
        print(f"Default variables saved to '{file_path}'.")

    except yaml.YAMLError as e:
        exit(f"Error parsing YAML in {file_path}: {e}")

    return data


def panoptic_parser(
        camera_ids: list,
        dataset_path: str = "dataset/Panoptic",
        sequence_name: str = '171026_pose3'
) -> tuple:
    """
    Get the camera parameters for the specified list of camera IDs.
    (rotation matrices, translation vectors, camera matrices, camera distortions)
    :param camera_ids: List of camera IDs
    :param dataset_path: Path to the dataset
    :param sequence_name: Name of the sequence
    :return: Tuple containing the camera parameters
    """
    # Load camera calibration parameters
    calib_file_path = os.path.join(
        dataset_path, sequence_name, f"calibration_{sequence_name}.json"
    )
    with open(calib_file_path) as file:
        calib = json.load(file)

    # Filter HD cameras based on panel=0 and node# in camera_ids
    hd_cameras = [
        cam for cam in calib['cameras']
        if cam['panel'] == 0 and cam['node'] in camera_ids
    ]
    # Convert parameters into numpy arrays for convenience
    (camera_matrices, camera_distortions,
     rotational_matrices, translation_vectors) = zip(
        *[(np.matrix(cam['K']), np.array(cam['distCoef']),
           np.matrix(cam['R']), np.array(cam['t']).reshape((3, 1)))
          for cam in hd_cameras]
    )
    return rotational_matrices, translation_vectors, camera_matrices, camera_distortions


def body3dscene_extractor(
        data_path: str = 'dataset/Panoptic',
        sequence_name: str = '171026_pose3',
        output_path: str = 'pose_output'
) -> None:
    print(f"Extracting sequence {sequence_name} Ground Truth from: /{data_path}/")
    # Setup paths
    hd_skel_json_path = data_path + '/' + sequence_name + '/hdPose3d_stage1_coco19/'
    # 171026_pose3: 129-7309
    files_list = os.listdir(hd_skel_json_path)
    hd_idx_start = int(files_list[0][12:20])
    hd_idx_stop = int(files_list[-1][12:20])

    skels = []
    for file in files_list:
        # 'body3DScene_{0:08d}.json'.format(frame)
        skel_json_fname = hd_skel_json_path + file
        try:
            # Load frame's skeletons
            with open(skel_json_fname) as dfile:
                bframe = json.load(dfile)

            # Cycle through all detected bodies
            for body in bframe['bodies']:
                # 19 3D joints array: [x1,y1,z1,c1, x2,y2,z2,c2, ...]
                # c1 ... c19 : per-joint detection confidences
                skel = np.array(body['joints19']).reshape((-1, 4))
                skels.append(skel[:, :-1])

        except IOError as e:
            print('Error reading {0}\n'.format(skel_json_fname) + e.strerror)

    poses = np.array(skels)
    frames = np.array(range(hd_idx_start, hd_idx_stop + 1))
    np.savez(
        f'{output_path}/Panoptic_GT_{sequence_name}.npz',
        frames=frames,  poses=poses
    )
    print(f"body3dscene files extracted in: /{output_path}/\n"
          f"Poses: {poses.shape}\n"
          f"Frames: {frames.shape}\n")


if __name__ == "__main__":
    body3dscene_extractor(data_path='../dataset/Panoptic', output_path='../pose_output')
