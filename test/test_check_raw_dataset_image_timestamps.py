from __future__ import annotations

import importlib
import sys
import types
import unittest

import pandas as pd


def _load_checker_without_zarr_dependency():
    module_name = "vive_teleop_tactile_bridge.check_raw_dataset"
    previous = sys.modules.get("zarr")
    sys.modules["zarr"] = types.ModuleType("zarr")
    try:
        return importlib.import_module(module_name)
    finally:
        if previous is None:
            sys.modules.pop("zarr", None)
        else:
            sys.modules["zarr"] = previous


checker = _load_checker_without_zarr_dependency()


class ImageTimestampCheckerTest(unittest.TestCase):
    def run_check(self, stamps: list[float]):
        frame = pd.DataFrame(
            {
                "episode_index": [0] * len(stamps),
                "image.ros_stamp_float": stamps,
            }
        )
        report = checker.Reporter()
        checker._check_image_timestamps(frame, "image", {"fps": 10.0}, report)
        return report

    def test_accepts_unique_monotonic_10_hz_saved_stamps(self) -> None:
        report = self.run_check([1000.0, 1000.1, 1000.2, 1000.3])
        self.assertEqual(report.errors, [])
        self.assertEqual(report.warnings, [])

    def test_rejects_duplicate_and_backwards_stamps(self) -> None:
        report = self.run_check([1000.0, 1000.1, 1000.1, 1000.05])
        self.assertTrue(any("reuse the same" in message for message in report.errors))
        self.assertTrue(any("move backwards" in message for message in report.errors))

    def test_warns_about_large_saved_image_gap(self) -> None:
        report = self.run_check([1000.0, 1000.1, 1000.5])
        self.assertEqual(report.errors, [])
        self.assertTrue(any("gaps exceed" in message for message in report.warnings))


class CandidateContinuityCheckerTest(unittest.TestCase):
    def run_check(self, candidates: list[int]):
        frame = pd.DataFrame(
            {
                "episode_index": [0] * len(candidates),
                "candidate_index": candidates,
            }
        )
        report = checker.Reporter()
        checker._check_candidate_continuity(frame, report)
        return report

    def test_ignores_candidates_before_first_saved_action(self) -> None:
        report = self.run_check([25, 26, 27, 28])
        self.assertEqual(report.errors, [])

    def test_rejects_only_internal_active_interval_holes(self) -> None:
        report = self.run_check([25, 26, 29, 30])
        self.assertTrue(
            any("2 active-interval candidate steps" in message for message in report.errors)
        )


if __name__ == "__main__":
    unittest.main()
