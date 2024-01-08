import numpy as np

from dataclasses import dataclass, field


@dataclass(slots=True)
class CamerasParameters:
    camera_count: int
    camera_mats: list[np.matrix]
    rotation_mats: list[np.matrix]
    translation_vecs: list[np.ndarray]
    distortion_coefs: list[np.ndarray]
    projection_mats: list[np.matrix] = field(init=False)

    def __post_init__(self):
        proj_mats = []
        for K, R, t in zip(self.camera_mats, self.rotation_mats, self.translation_vecs):
            # Essential matrix: [R|t]
            e_mat = np.empty((3, 4))
            e_mat[:3, :3] = R
            e_mat[:3, 3] = t.reshape(3)
            # Projection matrix:
            # ┌     ┐   ┌     ┐   ┌ ┌     ┐┌     ┐ ┐
            # │  P  │ = │  K  │ × │ │  R  ││  t  │ │
            # └(3×4)┘   └(3×3)┘   └ └(3×3)┘└(3×1)┘ ┘
            p = K @ e_mat
            proj_mats.append(p)

        self.projection_mats = proj_mats

    def __repr__(self):
        return (f"Cameras Parameters:\n"
                f"K: {self.camera_mats[0].shape} \t{type(self.camera_mats[0])}\n"
                f"R: {self.rotation_mats[0].shape} \t{type(self.rotation_mats[0])}\n"
                f"t: {self.translation_vecs[0].shape} \t{type(self.translation_vecs[0])}\n"
                f"d: {self.distortion_coefs[0].shape} \t{type(self.distortion_coefs[0])}\n"
                f"P: {self.projection_mats[0].shape} \t{type(self.projection_mats[0])}\n"
                f"camera count: \t\t\t{self.camera_count}")


if __name__ == "__main__":
    pass
