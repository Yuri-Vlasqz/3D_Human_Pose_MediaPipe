from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class CamerasParameters:
    """
    Data class for storing camera parameters.

    :param camera_count: Number of cameras.
    :param camera_mats: List of camera matrices.
    :param rotation_mats: List of rotation matrices.
    :param translation_vecs: List of translation vectors.
    :param distortion_coefs: List of distortion coefficients vectors.
    :param projection_mats: List of projection matrices (initialized automatically).
    """
    camera_count: int
    camera_mats: list[np.matrix]
    rotation_mats: list[np.matrix]
    translation_vecs: list[np.ndarray]
    distortion_coefs: list[np.ndarray]
    projection_mats: list[np.matrix] = field(init=False)

    def __post_init__(self):
        """
        Calculates the projection matrices for each camera.
        """
        # Projection matrix:
        # ┌     ┐   ┌     ┐   ┌ ┌     ┐┌     ┐ ┐
        # │  P  │ = │  K  │ × │ │  R  ││  t  │ │
        # └(3×4)┘   └(3×3)┘   └ └(3×3)┘└(3×1)┘ ┘
        self.projection_mats = [
            K @ np.hstack((R, t)) for K, R, t in zip(
                self.camera_mats, self.rotation_mats, self.translation_vecs)
        ]

    def __repr__(self):
        return (
            f"Cameras Parameters:\n"
            f"K: {self.camera_mats[0].shape} \t{type(self.camera_mats[0])}\n"
            f"R: {self.rotation_mats[0].shape} \t{type(self.rotation_mats[0])}\n"
            f"t: {self.translation_vecs[0].shape} \t{type(self.translation_vecs[0])}\n"
            f"d: {self.distortion_coefs[0].shape} \t{type(self.distortion_coefs[0])}\n"
            f"P: {self.projection_mats[0].shape} \t{type(self.projection_mats[0])}\n"
            f"camera count: \t\t\t{self.camera_count}"
        )


if __name__ == "__main__":
    pass
