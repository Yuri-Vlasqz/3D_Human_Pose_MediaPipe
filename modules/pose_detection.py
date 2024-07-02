from collections import namedtuple

import cv2
import mediapipe as mp  # type: ignore
import numpy as np

# from mediapipe.framework.formats import landmark_pb2.NormalizedLandmarkList
poseLandmarkerResult = namedtuple('poseLandmarkerResult',
                                  ["pose_landmarks",
                                   "pose_world_landmarks",
                                   "segmentation_mask"])
BLUE = (255, 0, 0)
RED = (0, 0, 255)
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
draw_landmarks = mp.solutions.drawing_utils.draw_landmarks
drawing_spec = mp.solutions.drawing_styles.DrawingSpec


class PoseDetection:

    def __init__(self, image_resolution: list[int],
                 landmarker_model: int = 1, visibility_threshold: float = 0.75):
        """ Class for inference of pose landmarks using the Mediapipe Pose models:
        - 0: lite
        - 1: full
        - 2: heavy

        :param image_resolution: Height and width of the input image.
        :param landmarker_model: Pose estimation model complexity (defaults to 1).
        :param visibility_threshold: Confidence threshold for detection and tracking
        (default = 0.75).
        """
        self.image_resolution = image_resolution
        self.pose = mp.solutions.pose.Pose(
            model_complexity=landmarker_model,
            min_detection_confidence=visibility_threshold,
            min_tracking_confidence=visibility_threshold
        )
        self.results = poseLandmarkerResult(None, None, None)

    def __repr__(self):
        return f'<Mediapipe {self.pose.__repr__().split(".")[-1]}'

    def __landmarks_to_ndarray(self) -> np.ndarray:
        """ Converts Mediapipe NormalizedLandmarkList to a NumPy array. """
        return np.array([
            [point.x * self.image_resolution[1],  # x% × image width
             point.y * self.image_resolution[0],  # y% × image height
             point.visibility] for point in self.results.pose_landmarks.landmark
        ])

    def detect_landmarks(self, image: np.ndarray) -> np.ndarray:
        """ Detects pose landmarks in the input image.

        :param image: Input image in BGR format.
        :return: (33×3) NumPy array with 2D landmarks and visibility.
                 (empty array if pose not detected).
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(rgb_image)  # .pose_landmarks

        if not self.results.pose_landmarks:
            return np.empty(0)

        return self.__landmarks_to_ndarray()

    def overlay_landmarks(self, image: np.ndarray) -> np.ndarray:
        """ Overlays pose landmarks on the input image.

        OBS: works in any resolution preserving the aspect ratio
        :param image: Input image in BGR format.
        :return: Image with overlaid pose landmarks.
        """
        if self.results.pose_landmarks:
            draw_landmarks(
                image=image,
                landmark_list=self.results.pose_landmarks,
                connections=POSE_CONNECTIONS,
                landmark_drawing_spec=drawing_spec(color=BLUE, thickness=1, circle_radius=1),
                connection_drawing_spec=drawing_spec(color=RED, thickness=1),
            )
        return image


if __name__ == "__main__":
    test = PoseDetection([360, 360],
                         0, 0.6)
    print(test)
