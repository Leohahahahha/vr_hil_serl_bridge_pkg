from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path

from vive_teleop_tactile_bridge.image_stream_health import (
    RemoteImageStampMonitor,
    normalize_image_message_type,
    normalize_qos_reliability,
)


@dataclass
class Stamp:
    sec: int
    nanosec: int

    @classmethod
    def from_ns(cls, value: int) -> "Stamp":
        sec, nanosec = divmod(int(value), 1_000_000_000)
        return cls(sec=sec, nanosec=nanosec)


class FakeLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []

    def info(self, message: str) -> None:
        self.info_messages.append(message)

    def warning(self, message: str) -> None:
        self.warning_messages.append(message)


class RemoteImageStampMonitorTest(unittest.TestCase):
    def make_monitor(self, *, reject_invalid: bool = True) -> RemoteImageStampMonitor:
        return RemoteImageStampMonitor(
            logger=FakeLogger(),
            stream_name="front_zed_remote",
            expected_hz=30.0,
            max_apparent_age_sec=0.20,
            max_future_sec=0.05,
            reject_invalid=reject_invalid,
            diagnostics_interval_sec=5.0,
        )

    def test_maps_valid_remote_wall_stamp_to_local_monotonic(self) -> None:
        monitor = self.make_monitor()
        receive_wall_ns = 1_700_000_000_000_000_000
        receive_monotonic_ns = 100_000_000_000
        stamp = Stamp.from_ns(receive_wall_ns - 20_000_000)
        mapped = monitor.map_stamp(
            stamp,
            receive_wall_ns=receive_wall_ns,
            receive_monotonic_ns=receive_monotonic_ns,
        )
        self.assertAlmostEqual(mapped or 0.0, 99.98, places=9)

    def test_rejects_duplicate_backwards_old_and_future_stamps(self) -> None:
        receive_wall_ns = 1_700_000_000_000_000_000
        receive_monotonic_ns = 100_000_000_000
        monitor = self.make_monitor()
        first_stamp_ns = receive_wall_ns - 20_000_000
        first = Stamp.from_ns(first_stamp_ns)
        self.assertIsNotNone(
            monitor.map_stamp(
                first,
                receive_wall_ns=receive_wall_ns,
                receive_monotonic_ns=receive_monotonic_ns,
            )
        )
        self.assertIsNone(
            monitor.map_stamp(
                first,
                receive_wall_ns=receive_wall_ns + 1_000_000,
                receive_monotonic_ns=receive_monotonic_ns + 1_000_000,
            )
        )
        self.assertIsNone(
            monitor.map_stamp(
                Stamp.from_ns(first_stamp_ns - 1),
                receive_wall_ns=receive_wall_ns + 2_000_000,
                receive_monotonic_ns=receive_monotonic_ns + 2_000_000,
            )
        )

        old_monitor = self.make_monitor()
        self.assertIsNone(
            old_monitor.map_stamp(
                Stamp.from_ns(receive_wall_ns - 250_000_000),
                receive_wall_ns=receive_wall_ns,
                receive_monotonic_ns=receive_monotonic_ns,
            )
        )
        future_monitor = self.make_monitor()
        self.assertIsNone(
            future_monitor.map_stamp(
                Stamp.from_ns(receive_wall_ns + 60_000_000),
                receive_wall_ns=receive_wall_ns,
                receive_monotonic_ns=receive_monotonic_ns,
            )
        )

    def test_can_fallback_to_receive_time_for_diagnostics(self) -> None:
        monitor = self.make_monitor(reject_invalid=False)
        mapped = monitor.map_stamp(
            Stamp(sec=0, nanosec=0),
            receive_wall_ns=1_700_000_000_000_000_000,
            receive_monotonic_ns=123_000_000_000,
        )
        self.assertAlmostEqual(mapped or 0.0, 123.0, places=9)

    def test_qos_reliability_normalisation(self) -> None:
        self.assertEqual(
            normalize_qos_reliability("BEST-EFFORT", field_name="image.qos"),
            "best_effort",
        )
        self.assertEqual(
            normalize_qos_reliability("reliable", field_name="image.qos"),
            "reliable",
        )
        with self.assertRaises(ValueError):
            normalize_qos_reliability("sometimes", field_name="image.qos")

    def test_image_message_type_normalisation(self) -> None:
        self.assertEqual(
            normalize_image_message_type("CompressedImage", field_name="image.message_type"),
            "compressed",
        )
        self.assertEqual(
            normalize_image_message_type("image", field_name="image.message_type"),
            "raw",
        )
        with self.assertRaises(ValueError):
            normalize_image_message_type("theora", field_name="image.message_type")


class RemoteZedConfigTest(unittest.TestCase):
    def test_recorder_uses_latest_only_remote_zed_qos(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        text = (repository_root / "config" / "record_hilserl_raw_tactile_dmtac_w.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn('expected_hz: 30.0', text)
        self.assertIn('message_type: "compressed"', text)
        self.assertIn('/zed/zed_node/rgb/color/rect/image/compressed', text)
        self.assertIn('qos_reliability: "best_effort"', text)
        self.assertIn('qos_depth: 1', text)
        self.assertIn('reject_unsynced_remote_stamp: true', text)

    def test_orin_profile_is_rgb_only_at_30_hz(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        text = (repository_root / "config" / "zed_orin_rgb_30hz.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("pub_frame_rate: 30.0", text)
        self.assertIn("depth_mode: 'NONE'", text)
        self.assertIn("publish_point_cloud: false", text)
        self.assertIn("pos_tracking_enabled: false", text)


if __name__ == "__main__":
    unittest.main()
