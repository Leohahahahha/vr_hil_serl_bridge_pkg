from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

from vive_teleop_tactile_bridge.stream_sampling import (
    has_usable_action_label,
    nearest_time_ordered,
    ros_message_stamp_ns,
    skip_reason_key,
)


@dataclass
class Item:
    t: float
    payload: object


def message(stamp_ns: int):
    sec, nanosec = divmod(stamp_ns, 1_000_000_000)
    return SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))
    )


class StreamSamplingTest(unittest.TestCase):
    def test_nearest_prefers_newer_item_on_tie(self) -> None:
        items = [Item(0.9, "old"), Item(1.1, "new")]
        self.assertEqual(nearest_time_ordered(items, 1.0).payload, "new")

    def test_nearest_can_require_a_strictly_newer_image_stamp(self) -> None:
        items = [
            Item(0.98, message(100)),
            Item(1.02, message(200)),
            Item(1.08, message(300)),
        ]
        selected = nearest_time_ordered(
            items,
            1.0,
            accept=lambda item: ros_message_stamp_ns(item.payload) > 200,
        )
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(ros_message_stamp_ns(selected.payload), 300)

    def test_action_readiness_preserves_pre_action_gate(self) -> None:
        self.assertFalse(has_usable_action_label(None, 1.0, 0.08))
        self.assertFalse(
            has_usable_action_label(Item(0.8, {"action8": [0] * 8}), 1.0, 0.08)
        )
        self.assertTrue(
            has_usable_action_label(Item(0.96, {"action8": [0] * 8}), 1.0, 0.08)
        )

    def test_skip_reason_key_removes_varying_numeric_offsets(self) -> None:
        reason = (
            "image=ok wrist_image=stale_wrist_image_dt_-0.123 "
            "action=ok state=ok tactile_left=missing_tactile_left tactile_right=ok"
        )
        self.assertEqual(
            skip_reason_key(reason),
            "wrist_image:stale_wrist_image+tactile_left:missing_tactile_left",
        )


if __name__ == "__main__":
    unittest.main()
