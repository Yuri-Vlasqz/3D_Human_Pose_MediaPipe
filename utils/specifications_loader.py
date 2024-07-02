import threading
from dataclasses import dataclass

import cv2
import numpy as np

from models import CamerasParameters, MosaicStructure
from modules import PoseDetection, PoseTriangulation, PoseVisualization
from utils import panoptic_parser, yaml_parser


@dataclass(slots=True)
class ThreadSpecificationLoader:
    sequence_name: str
    cameras_idx: list[int]
    pose_model: int
    triangulation_method: str
    video_captures: list[cv2.VideoCapture]
    frame_count: int
    mosaic_structure: MosaicStructure
    pose_detectors: list[PoseDetection]
    pose_buffers: list[np.ndarray]
    pose_triangulator: PoseTriangulation
    pose_visualizer: PoseVisualization
    thread_lock: threading.Lock
    thread_barrier: threading.Barrier
    text: dict

    def __init__(self, specs_path: str):
        """
        Data class for storing thread specifications and inicializing modules

        :param specs_path: inicialization yaml file location
        """
        yaml_specs = yaml_parser(specs_path)
        directory, videos, img_res = yaml_specs['feeds'].values()
        tile_grid, tile_res = yaml_specs['mosaic'].values()
        model, visibility, triangulation_method = yaml_specs['pose_landmarker'].values()

        sequence_name = directory.split(sep='/')[-3]
        number_of_feeds = len(videos)
        cameras_idx = [int(video[-6:-4]) for video in videos]  # hd_00_xx.mp4
        # new test -> 0, 1, 3, 6, 13, 16, 21, 25, 30
        
        (rotational_matrices,
         translation_vectors,
         camera_matrices,
         camera_distortions) = panoptic_parser(cameras_idx, sequence_name=sequence_name)
        cameras_params = CamerasParameters(number_of_feeds,
                                           camera_matrices,
                                           rotational_matrices,
                                           translation_vectors,
                                           camera_distortions)

        self.sequence_name = sequence_name
        self.cameras_idx = cameras_idx
        self.pose_model = model
        self.triangulation_method = triangulation_method
        self.video_captures = [cv2.VideoCapture(directory + video) for video in videos]
        self.frame_count = int(self.video_captures[0].get(cv2.CAP_PROP_FRAME_COUNT))
        self.mosaic_structure = MosaicStructure(number_of_feeds, tile_grid, tile_res)
        self.pose_detectors = [PoseDetection(img_res, model, visibility) for _ in videos]
        self.pose_buffers = [np.empty(0)] * number_of_feeds
        self.pose_triangulator = PoseTriangulation(cameras_params)
        self.pose_visualizer = PoseVisualization(cameras_params, cameras_idx)
        self.thread_lock = threading.Lock()
        self.thread_barrier = threading.Barrier(number_of_feeds + 1)
        self.text = yaml_specs['text']

    def __repr__(self):
        return (f"\n| Specifications |\n"
                f"- Sequence: {self.sequence_name}\n"
                f"- Cameras: {self.cameras_idx}\n\n"
                f"{self.mosaic_structure}\n\n"
                f"{self.pose_triangulator}\n")

    def __len__(self):
        return len(self.cameras_idx)


@dataclass(slots=True)
class ProcessSpecificationLoader:
    cameras_idx: list[int]
    video_paths: list[str]
    pose_params: tuple

    def __init__(self, specs_path: str):
        """
        Data class for storing process specifications and inicializing modules

        :param specs_path: inicialization yaml file location
        """
        yaml_specs = yaml_parser(specs_path)
        directory, videos, img_res = yaml_specs['feeds'].values()
        tile_grid, tile_res = yaml_specs['mosaic'].values()
        model, visibility = yaml_specs['pose_landmarker'].values()
        number_of_feeds = len(videos)
        cameras_idx = [int(video[-6:-4]) for video in videos]

        self.cameras_idx = cameras_idx
        self.video_paths = [directory + video for video in videos]
        self.pose_params = (img_res, model, visibility)

    def __len__(self):
        return len(self.cameras_idx)


if __name__ == "__main__":
    pass
