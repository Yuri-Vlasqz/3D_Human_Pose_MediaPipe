import numpy as np

# Panoptic-COCO19 to Mediapipe-Pose matching joints (17)
MATCHING_POSE_JOINTS = (
    (1, 0),     # 0 Nose
    (3, 11),    # 1 Left Shoulder
    (9, 12),    # 2 Right Shoulder
    (4, 13),    # 3 Left Elbow
    (10, 14),   # 4 Right Elbow
    (5, 15),    # 5 Left Wrist
    (11, 16),   # 6 Right Wrist
    (6, 23),    # 7 Left Hip
    (12, 24),   # 8 Right Hip
    (7, 25),    # 9 Left Knee
    (13, 26),   # 10 Right Knee
    (8, 27),    # 11 Left Ankle
    (14, 28),   # 12 Right Ankle
    (15, 2),    # 13 Left Eye
    (16, 7),    # 14 Left Ear
    (17, 5),    # 15 Right Eye
    (18, 8)     # 16 Right Ear
)

MATCHING_POSE_CONNECTIONS = (
    (0, 13),    # Nose <-> Left Eye
    (0, 15),    # Nose <-> Right Eye
    (13, 14),   # Left Eye <-> Left Ear
    (15, 16),   # Right Eye <-> Right Ear
    (1, 2),     # Left Shoulder <-> Right Shoulder
    (1, 3),     # Left Shoulder <-> Left Elbow
    (2, 4),     # Right Shoulder <-> Right Elbow
    (3, 5),     # Left Elbow <-> Left Wrist
    (4, 6),     # Right Elbow <-> Right Wrist
    (1, 7),     # Left Shoulder <-> Left Hip
    (2, 8),     # Right Shoulder <-> Right Hip
    (7, 8),     # Left Hip <-> Right Hip
    (7, 9),     # Left Hip <-> Left Knee
    (8, 10),    # Right Hip <-> Right Knee
    (9, 11),    # Left Knee <-> Left Ankle
    (10, 12)    # Right Knee <-> Right Ankle
)


def execution_time_report(times_list: list, title: str = "Execution time", unit: str = "ms"):
    match unit:
        case "ms" | "milliseconds":
            multiplier = 1_000
        case "us" | "microseconds":
            multiplier = 1_000_000
        case "ns" | "nanoseconds":
            multiplier = 1_000_000_000
        case _:
            unit = "s"
            multiplier = 1

    print(f"--- {title} ---\n"
          f" - Mean: {round(np.mean(times_list) * multiplier, 3)} {unit}\n"
          f" - Std: ±{round(np.std(times_list) * multiplier, 3)} {unit}\n")
    # f" - Max: {round(np.max(times_list) * multiplier, 3)} {unit}\n"
    # f" - Min: {round(np.min(times_list) * multiplier, 3)} {unit}\n"
    # f" - Top 1%: {round(np.percentile(times_list, 99) * multiplier, 3)} {unit}\n"
    # f" - Min 1%: {round(np.percentile(times_list, 1) * multiplier, 3)} {unit}\n"


def compute_mpjpe(
        ground_truth_poses: np.ndarray,
        estimated_poses: np.ndarray,
        frames: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes the Mean Per Joint Position Error (MPJPE) of all sequence frames, between
    ground truth and estimated pose.
    Both poses should be numpy arrays with 3D joints positions (x,y,z).
    """
    gt_joints, est_joints = np.transpose(MATCHING_POSE_JOINTS)
    ground_truth_poses = ground_truth_poses[:, gt_joints]     # 19 COCO -> 17
    estimated_poses = estimated_poses[frames][:, est_joints]  # 33 Mediapipe -> 17

    # Position Error = Euclidean distance
    mpjpe = np.mean(np.linalg.norm(estimated_poses - ground_truth_poses, axis=2))

    return ground_truth_poses, estimated_poses, mpjpe


if __name__ == "__main__":
    execution_time_report(list(range(11)), unit="us")
    execution_time_report(list(range(11)))
