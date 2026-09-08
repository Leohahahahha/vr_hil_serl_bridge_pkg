from __future__ import annotations

import importlib
import sys
import types
import unittest

import pandas as pd


def _load_tabero_exporter_without_video_or_parquet_dependencies():
    module_name = (
        "vive_teleop_tactile_bridge.lerobot_export."
        "raw_dmtac_w_to_tabero_lerobot"
    )
    stub_names = ("cv2", "pyarrow", "pyarrow.parquet", "zarr")
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in stub_names}

    cv2 = types.ModuleType("cv2")
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.__path__ = []
    parquet = types.ModuleType("pyarrow.parquet")
    pyarrow.parquet = parquet
    zarr = types.ModuleType("zarr")
    try:
        sys.modules.update(
            {
                "cv2": cv2,
                "pyarrow": pyarrow,
                "pyarrow.parquet": parquet,
                "zarr": zarr,
            }
        )
        return importlib.import_module(module_name)
    finally:
        for name, old_value in previous.items():
            if old_value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


tabero_export = _load_tabero_exporter_without_video_or_parquet_dependencies()


class TaberoGripperContractLightweightTest(unittest.TestCase):
    def row(self, **updates) -> pd.Series:
        values = {
            "observation.state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.025],
        }
        values.update(updates)
        return pd.Series(values)

    def convert(self, row: pd.Series, *, state_unit="meter"):
        return tabero_export._tabero_state(
            row,
            state_gripper_unit=state_unit,
            state_gripper_coordinate="finger",
            gripper_open_width_m=0.085,
        )

    def test_writes_state_as_finger_meters(self) -> None:
        state = self.convert(self.row())
        self.assertAlmostEqual(float(state[6]), 0.025)

    def test_legacy_normalized_state(self) -> None:
        state = self.convert(
            self.row(
                **{
                    "observation.state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                }
            ),
            state_unit="normalized",
        )
        self.assertAlmostEqual(float(state[6]), 0.0425)

    def test_command_fields_are_irrelevant(self) -> None:
        state = self.convert(self.row(**{"action.pose7": "invalid"}))
        self.assertAlmostEqual(float(state[6]), 0.025)


if __name__ == "__main__":
    unittest.main()
