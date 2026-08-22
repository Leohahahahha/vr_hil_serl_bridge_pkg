from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import requests


@dataclass
class HttpConfig:
    server_url: str
    timeout_sec: float = 0.15


class FrankaHttpClient:
    """Small wrapper around the HIL-SERL / SERL Flask robot server."""

    def __init__(self, config: HttpConfig):
        self.config = config
        self.session = requests.Session()

    def _url(self, route: str) -> str:
        route = route if route.startswith('/') else f'/{route}'
        return f"{self.config.server_url.rstrip('/')}{route}"

    def post(self, route: str, payload: Optional[Dict[str, Any]] = None, timeout_sec: Optional[float] = None):
        timeout = self.config.timeout_sec if timeout_sec is None else timeout_sec
        if payload is None:
            response = self.session.post(self._url(route), timeout=timeout)
        else:
            response = self.session.post(self._url(route), json=payload, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        if 'application/json' in content_type:
            return response.json()
        return response.text

    def get_pose(self) -> np.ndarray:
        data = self.post('/getpos', timeout_sec=max(0.3, self.config.timeout_sec))
        pose = np.asarray(data['pose'], dtype=float)
        if pose.shape != (7,):
            raise RuntimeError(f'Expected pose with shape (7,), got {pose.shape}')
        return pose

    def command_pose(self, pose_xyz_xyzw: np.ndarray) -> None:
        pose = np.asarray(pose_xyz_xyzw, dtype=float).reshape(-1)
        if pose.shape != (7,):
            raise ValueError(f'Pose must be length 7, got {pose.shape}')
        self.post('/pose', {'arr': pose.tolist()})

    def clear_error(self) -> None:
        self.post('/clearerr')

    def joint_reset(self) -> None:
        self.post('/jointreset', timeout_sec=3.0)

    def activate_gripper(self) -> None:
        self.post('/activate_gripper', timeout_sec=0.5)

    def open_gripper(self) -> None:
        self.post('/open_gripper', timeout_sec=0.5)

    def close_gripper(self) -> None:
        self.post('/close_gripper', timeout_sec=0.5)

    def move_gripper(self, gripper_pos_0_255: int) -> None:
        pos = int(np.clip(gripper_pos_0_255, 0, 255))
        self.post('/move_gripper', {'gripper_pos': pos}, timeout_sec=0.5)

    def move_gripper_width(self, gripper_width_m: float) -> None:
        self.post('/move_gripper', {'gripper_width': float(gripper_width_m)}, timeout_sec=0.5)

    def get_gripper(self) -> float:
        data = self.post('/get_gripper', timeout_sec=0.5)
        if 'gripper_width' in data:
            return float(data['gripper_width'])
        if 'gripper' in data:
            return float(data['gripper'])
        return float(data['gripper_pos'])

    def get_gripper_width(self) -> float:
        return self.get_gripper()

    def get_state(self):
        return self.post('/getstate', timeout_sec=max(0.3, self.config.timeout_sec))
