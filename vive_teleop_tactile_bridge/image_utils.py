from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def image_msg_to_rgb8(msg: Any) -> np.ndarray:
    """Convert common ROS Image encodings to a contiguous HWC RGB uint8 image."""
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    step = int(msg.step)

    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")

    if encoding in ("bgr8", "rgb8"):
        channels = 3
        expected_row_bytes = width * channels
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        arr = arr[:, :expected_row_bytes].reshape(height, width, channels)
        if encoding == "bgr8":
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(arr)

    if encoding in ("bgra8", "rgba8"):
        channels = 4
        expected_row_bytes = width * channels
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        arr = arr[:, :expected_row_bytes].reshape(height, width, channels)
        if encoding == "rgba8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
        return np.ascontiguousarray(arr)

    if encoding in ("mono8", "8uc1"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        arr = arr[:, :width].reshape(height, width)
        return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))

    raise ValueError(f"unsupported image encoding for RGB conversion: {msg.encoding}")

