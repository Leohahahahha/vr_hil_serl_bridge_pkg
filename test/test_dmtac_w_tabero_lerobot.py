import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from vive_teleop_tactile_bridge.lerobot_export import raw_dmtac_w_to_tabero_lerobot as tabero_export
from vive_teleop_tactile_bridge.lerobot_export.raw_dmtac_w_to_n0vtla_lerobot import DMTacSource
from vive_teleop_tactile_bridge.lerobot_export.raw_dmtac_w_to_tabero_lerobot import (
    ACTION_KEY,
    DEPTH_KEY,
    MARKER_KEY,
    export_raw_dmtac_w_to_tabero_lerobot,
    validate_tabero_lerobot,
)


HEIGHT = 240
WIDTH = 320
PACKED_BYTES = 921_600


class _NpyDMTacSource(DMTacSource):
    """Test storage adapter; decoder behavior remains the production implementation."""

    def __init__(self, raw_root: Path, side: str):
        self.side = side
        side_root = raw_root / "tactile" / f"tactile_{side}"
        self.array = np.load(side_root / "data.npy", mmap_mode="r")
        self.meta_path = side_root / "meta.jsonl"
        self.meta = {
            int(row["zarr_index"]): row
            for row in (
                json.loads(line)
                for line in self.meta_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }


def _layout() -> dict[str, int | str]:
    return {
        "schema_version": 3,
        "output_mode": "shear_depth",
        "byte_order_little_endian": 1,
        "packed_frame_bytes": PACKED_BYTES,
        "shear_start": 0,
        "shear_len": HEIGHT * WIDTH * 2 * 4,
        "shear_height": HEIGHT,
        "shear_width": WIDTH,
        "shear_channels": 2,
        "shear_itemsize": 4,
        "depth_start": HEIGHT * WIDTH * 2 * 4,
        "depth_len": HEIGHT * WIDTH * 4,
        "depth_height": HEIGHT,
        "depth_width": WIDTH,
        "depth_channels": 1,
        "depth_itemsize": 4,
    }


def _packed(shear_x: float, shear_y: float, depth_value: float) -> np.ndarray:
    shear = np.empty((HEIGHT, WIDTH, 2), dtype="<f4")
    shear[..., 0] = shear_x
    shear[..., 1] = shear_y
    depth = np.full((HEIGHT, WIDTH, 1), depth_value, dtype="<f4")
    return np.concatenate(
        [shear.view(np.uint8).reshape(-1), depth.view(np.uint8).reshape(-1)]
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonlines(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _build_raw_dataset(root: Path) -> None:
    episode_lengths = (2, 3)
    total_frames = sum(episode_lengths)
    _write_json(
        root / "meta" / "info.json",
        {
            "dataset_type": "synthetic_dmtac_w_raw",
            "tactile_msg_package": "dmtac_tactile",
            "task_description": "touch the object",
            "fps": 10.0,
        },
    )
    _write_jsonlines(
        root / "meta" / "episodes.jsonl",
        [{"episode_index": index, "length": length} for index, length in enumerate(episode_lengths)],
    )

    for side in ("left", "right"):
        side_root = root / "tactile" / f"tactile_{side}"
        side_root.mkdir(parents=True, exist_ok=True)
        array = np.empty((total_frames, PACKED_BYTES), dtype=np.uint8)
        metadata = []
        for index in range(total_frames):
            offset = 0.0 if side == "left" else 10.0
            depth = 100.0 + index if side == "left" else 200.0 + index
            array[index] = _packed(offset + index, -(offset + index), depth)
            metadata.append(
                {
                    "zarr_index": index,
                    "frame_idx": 1000 + index,
                    "sync_dt": 0.0,
                    "msg_package": "dmtac_tactile",
                    "layout": _layout(),
                }
            )
        np.save(side_root / "data.npy", array)
        _write_jsonlines(side_root / "meta.jsonl", metadata)

    global_index = 0
    for episode_index, length in enumerate(episode_lengths):
        rows = []
        for frame_index in range(length):
            image_rel = Path("image") / "image" / f"episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
            wrist_rel = Path("image") / "wrist_image" / f"episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
            for rel, color in ((image_rel, (10, 20, 30)), (wrist_rel, (40, 50, 60))):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (16, 16), tuple(value + frame_index for value in color)).save(path)
            rows.append(
                {
                    "timestamp": 10.0 * episode_index + frame_index * 0.1,
                    "candidate_index": frame_index,
                    "frame_index": frame_index,
                    "observation.state": np.asarray(
                        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.025], dtype=np.float32
                    ),
                    "action.pose7": np.asarray(
                        [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
                    ),
                    "action.target_gripper_width": 0.0425,
                    "action.target_gripper_finger_position": 0.02125,
                    "robot.force": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                    "robot.torque": np.asarray([4.0, 5.0, 6.0], dtype=np.float32),
                    "image.path": str(image_rel),
                    "wrist_image.path": str(wrist_rel),
                    "tactile.left.zarr_index": global_index,
                    "tactile.right.zarr_index": global_index,
                }
            )
            global_index += 1
        episode_path = root / "data" / "episodes" / f"episode_{episode_index:06d}.parquet"
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(episode_path, index=False)


class DMTacTaberoLeRobotTest(unittest.TestCase):
    def test_end_to_end_v21_export_and_episode_history_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            raw_root = temp / "raw"
            out_root = temp / "tabero"
            _build_raw_dataset(raw_root)

            with mock.patch.object(tabero_export, "DMTacSource", _NpyDMTacSource):
                result = export_raw_dmtac_w_to_tabero_lerobot(
                    raw_root=raw_root,
                    out_root=out_root,
                    overwrite=False,
                )
            self.assertTrue(result["ok"])
            self.assertEqual(result["total_episodes"], 2)
            self.assertEqual(result["total_frames"], 5)
            self.assertEqual(validate_tabero_lerobot(out_root)["action_dim"], 7)

            info = json.loads((out_root / "meta" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["codebase_version"], "v2.1")
            self.assertEqual(info["features"][ACTION_KEY]["shape"], [7])
            self.assertEqual(info["features"][MARKER_KEY]["shape"], [9, 198, 2])
            self.assertEqual(info["features"][DEPTH_KEY]["shape"], [2, 240, 320])
            self.assertFalse(info["wrench_supervision"])
            self.assertEqual(info["gripper_unit"], "meter")
            self.assertEqual(info["gripper_coordinate"], "single_finger_absolute_position")
            self.assertAlmostEqual(info["gripper_max_finger_position_m"], 0.0425)

            episode0 = pq.read_table(out_root / "data" / "chunk-000" / "episode_000000.parquet")
            episode1 = pq.read_table(out_root / "data" / "chunk-000" / "episode_000001.parquet")
            marker0 = np.asarray(episode0[MARKER_KEY].to_pylist(), dtype=np.float32)
            marker1 = np.asarray(episode1[MARKER_KEY].to_pylist(), dtype=np.float32)
            depth0 = np.asarray(episode0[DEPTH_KEY].to_pylist(), dtype=np.float32)
            actions0 = np.asarray(episode0[ACTION_KEY].to_pylist(), dtype=np.float32)
            states0 = np.asarray(episode0["state"].to_pylist(), dtype=np.float32)

            self.assertEqual(marker0.shape, (2, 9, 198, 2))
            self.assertTrue(np.array_equal(marker0[0, 1], marker0[0, 8]))
            self.assertTrue(np.array_equal(marker1[0, 1], marker1[0, 8]))
            self.assertAlmostEqual(float(marker0[1, 8, 0, 0] - marker0[1, 0, 0, 0]), 1.0)
            self.assertAlmostEqual(float(marker1[0, 8, 0, 0] - marker1[0, 0, 0, 0]), 2.0)
            self.assertAlmostEqual(float(depth0[0, 0, 0, 0]), 100.0)
            self.assertAlmostEqual(float(depth0[0, 1, 0, 0]), 200.0)
            np.testing.assert_allclose(states0[0, 6], 0.025)
            np.testing.assert_allclose(
                actions0[0],
                [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.02125],
            )

            hf_meta = json.loads(episode0.schema.metadata[b"huggingface"])
            hf_features = hf_meta["info"]["features"]
            self.assertEqual(hf_features[MARKER_KEY]["_type"], "Array3D")
            self.assertEqual(hf_features[DEPTH_KEY]["_type"], "Array3D")

    def test_strict_timing_rejects_internal_candidate_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            raw_root = temp / "raw"
            _build_raw_dataset(raw_root)
            episode = raw_root / "data" / "episodes" / "episode_000000.parquet"
            frame = pd.read_parquet(episode)
            frame.loc[1, "candidate_index"] = 2
            frame.to_parquet(episode, index=False)
            with mock.patch.object(tabero_export, "DMTacSource", _NpyDMTacSource):
                with self.assertRaisesRegex(ValueError, "missing_candidate_steps=1"):
                    export_raw_dmtac_w_to_tabero_lerobot(
                        raw_root=raw_root,
                        out_root=temp / "out",
                    )

    def test_legacy_normalized_state_is_converted_to_finger_meters(self) -> None:
        row = pd.Series(
            {
                "observation.state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                "action.pose7": [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
                "action.target_gripper_width": 0.085,
            }
        )
        state, action = tabero_export._tabero_state_action(
            row,
            state_gripper_unit="normalized",
            state_gripper_coordinate="finger",
            action_gripper_unit="meter",
            gripper_open_width_m=0.085,
        )
        self.assertAlmostEqual(float(state[6]), 0.0425)
        self.assertAlmostEqual(float(action[6]), 0.0425)

    def test_rejects_inconsistent_explicit_target_finger_position(self) -> None:
        row = pd.Series(
            {
                "observation.state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.02],
                "action.pose7": [0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
                "action.target_gripper_width": 0.04,
                "action.target_gripper_finger_position": 0.03,
            }
        )
        with self.assertRaisesRegex(ValueError, "does not equal"):
            tabero_export._tabero_state_action(
                row,
                state_gripper_unit="meter",
                state_gripper_coordinate="finger",
                action_gripper_unit="meter",
                gripper_open_width_m=0.085,
            )


if __name__ == "__main__":
    unittest.main()
