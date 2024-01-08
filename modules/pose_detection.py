import mediapipe as mp
import numpy as np
import cv2

from timeit import repeat
# from mediapipe.framework.formats.landmark_pb2 import _globals
# print(_globals['_NORMALIZEDLANDMARKLIST'])
BLUE = (255, 0, 0)
RED = (0, 0, 255)


class PoseDetection:
    def __init__(self, image_resolution: list[int, int],
                 landmarker_model: int = 1, visibility_threshold: float = 0.75):

        self.image_resolution = image_resolution  # (height, width)
        self.draw_landmarks = mp.solutions.drawing_utils.draw_landmarks
        self.drawing_spec = mp.solutions.drawing_styles.DrawingSpec
        self.pose_connections = mp.solutions.pose.POSE_CONNECTIONS
        self.pose = mp.solutions.pose.Pose(
            model_complexity=landmarker_model,
            min_detection_confidence=visibility_threshold,
            min_tracking_confidence=visibility_threshold
        )
        self.results = None

    def __repr__(self):
        return f'<Mediapipe {self.pose.__repr__().split(".")[-1]}'

    def landmarks_to_ndarray(self, pose_landmarks) -> np.ndarray:
        return np.array([[norm_point.x * self.image_resolution[1],  # image width
                          norm_point.y * self.image_resolution[0],  # image height
                          norm_point.visibility] for norm_point in pose_landmarks.landmark])

    def detect_landmarks(self, image: np.ndarray) -> np.ndarray:
        mp_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(mp_image).pose_landmarks

        if not self.results:
            return np.empty(0)

        return self.landmarks_to_ndarray(self.results)

    def overlay_landmarks(self, image: np.ndarray) -> np.ndarray:  # works in any resolution preserving aspect ratio
        if self.results:
            self.draw_landmarks(
                image=image,
                landmark_list=self.results,
                connections=self.pose_connections,
                landmark_drawing_spec=self.drawing_spec(color=BLUE, thickness=1, circle_radius=1),
                connection_drawing_spec=self.drawing_spec(color=RED, thickness=2),
            )
        return image


if __name__ == "__main__":
    test = PoseDetection([360, 360],
                         0, 0.6)
    print(test)
    emp = np.empty(0)
    print(emp, type(emp), emp.size)

    # Define a simple function to test
    def example_function():
        for _ in range(1000000):
            np.empty(0)


    # measure the execution time of the function
    execution_time = repeat(example_function, repeat=10, number=10)
    # Print the result
    print(f"Execution time: {round(np.mean(execution_time), 3)} microseconds")
