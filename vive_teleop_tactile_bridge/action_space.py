from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


STATE_NAMES = [
    "current_x", "current_y", "current_z",
    "current_rotvec_x", "current_rotvec_y", "current_rotvec_z",
    "current_gripper_width",
]

ACTION_NAMES = [
    "delta_x", "delta_y", "delta_z",
    "delta_rotvec_x", "delta_rotvec_y", "delta_rotvec_z",
    "target_gripper_width",
]


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def pose7_to_state7(pose7: np.ndarray, gripper_width: float) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float32).reshape(7).copy()
    pose7[3:7] = normalize_quat_xyzw(pose7[3:7])
    rotvec = R.from_quat(pose7[3:7].astype(float)).as_rotvec().astype(np.float32)
    return np.concatenate(
        [pose7[:3], rotvec, np.asarray([gripper_width], dtype=np.float32)],
        axis=0,
    ).astype(np.float32)


def target_pose_to_relative_action7(
    *,
    target_pose7: np.ndarray,
    current_pose7: np.ndarray,
    target_gripper_width: float,
) -> np.ndarray:
    target_pose7 = np.asarray(target_pose7, dtype=np.float32).reshape(7).copy()
    current_pose7 = np.asarray(current_pose7, dtype=np.float32).reshape(7).copy()
    target_pose7[3:7] = normalize_quat_xyzw(target_pose7[3:7])
    current_pose7[3:7] = normalize_quat_xyzw(current_pose7[3:7])

    dpos = target_pose7[:3] - current_pose7[:3]
    r_target = R.from_quat(target_pose7[3:7].astype(float))
    r_current = R.from_quat(current_pose7[3:7].astype(float))
    drotvec = (r_target * r_current.inv()).as_rotvec().astype(np.float32)

    return np.concatenate(
        [dpos.astype(np.float32), drotvec, np.asarray([target_gripper_width], dtype=np.float32)],
        axis=0,
    ).astype(np.float32)


def relative_action7_to_target_pose7(
    *,
    current_pose7: np.ndarray,
    action7: np.ndarray,
) -> Tuple[np.ndarray, float]:
    current_pose7 = np.asarray(current_pose7, dtype=np.float32).reshape(7).copy()
    action7 = np.asarray(action7, dtype=np.float32).reshape(7)
    current_pose7[3:7] = normalize_quat_xyzw(current_pose7[3:7])

    target_xyz = current_pose7[:3] + action7[:3]
    r_current = R.from_quat(current_pose7[3:7].astype(float))
    r_delta = R.from_rotvec(action7[3:6].astype(float))
    r_target = r_delta * r_current

    target_pose7 = np.concatenate(
        [target_xyz.astype(np.float32), r_target.as_quat().astype(np.float32)],
        axis=0,
    )
    target_pose7[3:7] = normalize_quat_xyzw(target_pose7[3:7])
    return target_pose7.astype(np.float32), float(action7[6])
