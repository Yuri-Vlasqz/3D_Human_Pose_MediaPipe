import yaml
import json
import cv2
import numpy as np

RED = (0, 0, 255)
WHITE = (255, 255, 255)
default_specs = {
    "title": "Inicialization Specifications",
    "feeds": {
        "path": ['dataset/Panoptic/171026_pose3/hdVideos/hd_00_00.mp4',
                 'dataset/Panoptic/171026_pose3/hdVideos/hd_00_01.mp4',
                 'dataset/Panoptic/171026_pose3/hdVideos/hd_00_06.mp4',
                 'dataset/Panoptic/171026_pose3/hdVideos/hd_00_13.mp4', ],
        "resolution": [1080, 1920]},
    "mosaic": {
        "tile_grid": [2, 2],
        "tile_resolution": [270, 480]
    },
    "text": {
        "font": cv2.FONT_HERSHEY_SIMPLEX,
        "size": 0.6,
        "color": RED,
        "thickness": 1,
        "line_type": cv2.LINE_AA,
    }
}


def yaml_parser(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        if data is None:
            print("Data in ", end="")
            raise FileNotFoundError

    except FileNotFoundError:
        print(f"'{file_path}' not found.")
        data = default_specs
        with open(file_path, 'w') as file:
            yaml.safe_dump(data, file, sort_keys=False, default_flow_style=False, indent=4)
        print(f"Default variables saved to '{file_path}'.")

    except yaml.YAMLError as e:
        exit(f"Error parsing YAML in {file_path}: {e}")

    return data


def json_parser(file_path: str, default_data: dict) -> dict:
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)

    except FileNotFoundError or json.JSONDecodeError:
        print(f"File not found: {file_path}")
        print("Creating a new file with default data...")
        with open(file_path, 'w') as file:
            json.dump(default_data, file, indent=4)
        data = default_data

    except json.JSONDecodeError as e:
        exit(f"Error decoding JSON in file {file_path}: {e}")

    return data


# Panoptic - camera parameters extraction
def panoptic_parser(cam_list: list, directory: str = "dataset/Panoptic/", seq_name: str = '171026_pose3'):
    """get a selected list of HD camera parameters (intrinsic and extrinsic)"""

    # Load camera calibration parameters
    with open(f'{directory}{seq_name}/calibration_{seq_name}.json') as cfile:
        calib = json.load(cfile)

    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'], cam['node']): cam for cam in calib['cameras']}

    # Selected list of HD cameras
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
    yaml_data = yaml_parser("../inicialization.yaml")
    TEXT = list(yaml_data['text'].values())
    print(TEXT)
    dataset_dir = "../dataset/Panoptic/"
    R, t, K, d = panoptic_parser([0, 10, 20, 30], directory=dataset_dir, seq_name='171026_pose3')
    print(f"-----------------------------------\n"
          f"R: {R[0].shape} \t{type(R[0])}\n"
          f"t: {t[0].shape} \t{type(t[0])}\n"
          f"K: {K[0].shape} \t{type(K[0])}\n"
          f"d: {d[0].shape} \t{type(d[0])}")
