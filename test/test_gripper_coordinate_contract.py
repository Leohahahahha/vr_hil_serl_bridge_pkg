from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np


if importlib.util.find_spec("scipy") is None:
    class _IdentityRotation:
        @classmethod
        def from_quat(cls, _value):
            return cls()

        @classmethod
        def from_rotvec(cls, _value):
            return cls()

        def as_rotvec(self):
            return np.zeros(3, dtype=float)

        def as_quat(self):
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)

        def inv(self):
            return self

        def __mul__(self, _other):
            return self

    scipy = types.ModuleType("scipy")
    scipy.__path__ = []
    spatial = types.ModuleType("scipy.spatial")
    spatial.__path__ = []
    transform = types.ModuleType("scipy.spatial.transform")
    transform.Rotation = _IdentityRotation
    scipy.spatial = spatial
    spatial.transform = transform
    sys.modules.update(
        {
            "scipy": scipy,
            "scipy.spatial": spatial,
            "scipy.spatial.transform": transform,
        }
    )

from vive_teleop_tactile_bridge.action_space import (
    finger_position_to_gripper_width,
    gripper_width_to_finger_position,
    pose7_to_state7,
    target_pose_to_relative_action7,
)
from vive_teleop_tactile_bridge.observation_builder import (
    ObservationSyncThresholds,
    build_observation_frame,
    extract_state_fields,
)


class GripperCoordinateContractTest(unittest.TestCase):
    def test_extracts_normalized_total_width_and_single_finger_separately(self) -> None:
        fields = extract_state_fields(
            {
                "state": {
                    "pose": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
                    "gripper_pos": 1.0,
                    "gripper_width": 0.085,
                }
            }
        )
        self.assertEqual(fields["robot.gripper_pos"], 1.0)
        self.assertEqual(fields["robot.gripper_pos_normalized"], 1.0)
        self.assertAlmostEqual(fields["robot.gripper_width"], 0.085)
        self.assertAlmostEqual(fields["robot.gripper_finger_position"], 0.0425)

    def test_does_not_treat_normalized_position_as_physical_width(self) -> None:
        fields = extract_state_fields(
            {
                "state": {
                    "pose": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
                    "gripper_pos": 0.5,
                }
            }
        )
        self.assertEqual(fields["robot.gripper_pos_normalized"], 0.5)
        self.assertIsNone(fields["robot.gripper_width"])
        self.assertIsNone(fields["robot.gripper_finger_position"])

    def test_state_and_action_slot_use_single_finger_meters(self) -> None:
        current_pose = np.asarray([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
        target_pose = np.asarray([0.41, 0.02, 0.31, 0.0, 0.0, 0.0, 1.0])
        state = pose7_to_state7(current_pose, gripper_width_to_finger_position(0.085))
        action = target_pose_to_relative_action7(
            target_pose7=target_pose,
            current_pose7=current_pose,
            target_gripper_finger_position_m=gripper_width_to_finger_position(0.04),
        )
        self.assertAlmostEqual(float(state[6]), 0.0425)
        self.assertAlmostEqual(float(action[6]), 0.02)
        self.assertAlmostEqual(finger_position_to_gripper_width(float(action[6])), 0.04)

    def test_negative_physical_gripper_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            gripper_width_to_finger_position(-0.001)
        with self.assertRaises(ValueError):
            finger_position_to_gripper_width(-0.001)

    def test_observation_builder_applies_the_contract_to_saved_slots(self) -> None:
        t_frame = 10.0
        item = lambda payload: SimpleNamespace(t=t_frame, payload=payload)
        result = build_observation_frame(
            bundle={
                "image": item(object()),
                "wrist_image": item(object()),
                "wrist_camera_info": None,
                "command": item(
                    {
                        "action8": [0.41, 0.02, 0.31, 0.0, 0.0, 0.0, 1.0, 0.04],
                        "target_gripper_width": 0.04,
                        "target_gripper_finger_position": 0.02,
                    }
                ),
                "state": item(
                    {
                        "state": {
                            "pose": [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
                            "gripper_pos": 1.0,
                            "gripper_width": 0.085,
                        }
                    }
                ),
                "vr_pose": None,
                "enabled": None,
                "joystick_y": None,
                "tactile_left": item({"data_len": 8}),
                "tactile_right": item({"data_len": 8}),
            },
            t_frame=t_frame,
            thresholds=ObservationSyncThresholds(
                max_image_dt_sec=0.1,
                max_wrist_image_dt_sec=0.1,
                max_action_dt_sec=0.1,
                max_state_dt_sec=0.1,
                max_vr_dt_sec=0.1,
                max_tactile_dt_sec=0.1,
            ),
            wait_for_motion_enable=False,
        )
        self.assertTrue(result.ok, result.reason)
        self.assertAlmostEqual(float(result.frame.observation_state7[6]), 0.0425)
        self.assertAlmostEqual(float(result.frame.relative_action7[6]), 0.02)
        self.assertAlmostEqual(result.frame.target_gripper_finger_position, 0.02)


if __name__ == "__main__":
    unittest.main()
