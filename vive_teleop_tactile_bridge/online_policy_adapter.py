from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .image_utils import image_msg_to_rgb8
    from .observation_builder import ObservationFrame
except ImportError:
    from image_utils import image_msg_to_rgb8  # type: ignore
    from observation_builder import ObservationFrame  # type: ignore


FRONT_IMAGE_KEY = "observation.images.front"
WRIST_IMAGE_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"
TACTILE_LEFT_KEY = "observation.tactile_left"
TACTILE_RIGHT_KEY = "observation.tactile_right"


@dataclass(frozen=True)
class OnlineObservation:
    data: dict[str, Any]
    sync: dict[str, Any]


def to_lerobot_observation(frame: ObservationFrame) -> OnlineObservation:
    """Build the in-memory observation expected after decoding a LeRobot sample."""
    front_rgb = image_msg_to_rgb8(frame.image_item.payload)
    wrist_rgb = image_msg_to_rgb8(frame.wrist_image_item.payload)
    state = np.asarray(frame.observation_state7, dtype=np.float32).reshape(7)
    tactile_left = np.asarray(frame.tactile_left_payload["data"])
    tactile_right = np.asarray(frame.tactile_right_payload["data"])

    data = {
        FRONT_IMAGE_KEY: front_rgb,
        WRIST_IMAGE_KEY: wrist_rgb,
        STATE_KEY: state,
        TACTILE_LEFT_KEY: tactile_left,
        TACTILE_RIGHT_KEY: tactile_right,
    }
    sync = {
        "dt_image": frame.dt_image,
        "dt_wrist_image": frame.dt_wrist_image,
        "dt_action": frame.dt_action,
        "dt_state": frame.dt_state,
        "dt_vr": frame.dt_vr if frame.ok_vr else None,
        "reason_vr": frame.reason_vr,
        "dt_tactile_left": frame.dt_tactile_left,
        "dt_tactile_right": frame.dt_tactile_right,
    }
    return OnlineObservation(data=data, sync=sync)


def assert_canonical_observation(obs: dict[str, Any]) -> None:
    front = np.asarray(obs[FRONT_IMAGE_KEY])
    wrist = np.asarray(obs[WRIST_IMAGE_KEY])
    state = np.asarray(obs[STATE_KEY])
    tactile_left = np.asarray(obs[TACTILE_LEFT_KEY])
    tactile_right = np.asarray(obs[TACTILE_RIGHT_KEY])

    _assert_rgb_image(FRONT_IMAGE_KEY, front)
    _assert_rgb_image(WRIST_IMAGE_KEY, wrist)
    if state.shape != (7,) or state.dtype != np.float32:
        raise ValueError(f"{STATE_KEY} must be float32[7], got dtype={state.dtype} shape={state.shape}")
    if tactile_left.ndim != 1:
        raise ValueError(f"{TACTILE_LEFT_KEY} must be 1D, got shape={tactile_left.shape}")
    if tactile_right.ndim != 1:
        raise ValueError(f"{TACTILE_RIGHT_KEY} must be 1D, got shape={tactile_right.shape}")


def _assert_rgb_image(name: str, image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must be HWC RGB, got shape={image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got dtype={image.dtype}")

