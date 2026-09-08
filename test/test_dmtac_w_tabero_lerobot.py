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
                        [
                            0.1 + episode_index + 0.01 * frame_index,
                            0.2 + 0.02 * frame_index,
                            0.3 + 0.03 * frame_index,
                            0.0,
                            0.0,
                            0.1 * frame_index,
                            0.020 + 0.001 * (episode_index + frame_index),
                        ],
                        dtype=np.float32,
                    ),
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
            self.assertEqual(result["total_frames"], 3)
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
            self.assertEqual(
                info["action_supervision"],
                "actions[t] = state[t+1] within each episode",
            )

            episode0 = pq.read_table(out_root / "data" / "chunk-000" / "episode_000000.parquet")
            episode1 = pq.read_table(out_root / "data" / "chunk-000" / "episode_000001.parquet")
            marker0 = np.asarray(episode0[MARKER_KEY].to_pylist(), dtype=np.float32)
            marker1 = np.asarray(episode1[MARKER_KEY].to_pylist(), dtype=np.float32)
            depth0 = np.asarray(episode0[DEPTH_KEY].to_pylist(), dtype=np.float32)
            actions0 = np.asarray(episode0[ACTION_KEY].to_pylist(), dtype=np.float32)
            states0 = np.asarray(episode0["state"].to_pylist(), dtype=np.float32)
            actions1 = np.asarray(episode1[ACTION_KEY].to_pylist(), dtype=np.float32)
            states1 = np.asarray(episode1["state"].to_pylist(), dtype=np.float32)

            self.assertEqual(marker0.shape, (1, 9, 198, 2))
            self.assertEqual(marker1.shape, (2, 9, 198, 2))
            self.assertTrue(np.array_equal(marker0[0, 1], marker0[0, 8]))
            self.assertTrue(np.array_equal(marker1[0, 1], marker1[0, 8]))
            self.assertAlmostEqual(float(marker1[0, 8, 0, 0] - marker1[0, 0, 0, 0]), 2.0)
            self.assertAlmostEqual(float(depth0[0, 0, 0, 0]), 100.0)
            self.assertAlmostEqual(float(depth0[0, 1, 0, 0]), 200.0)
            np.testing.assert_allclose(states0[0, 6], 0.020)
            np.testing.assert_allclose(
                actions0[0],
                [0.11, 0.22, 0.33, 0.0, 0.0, 0.1, 0.021],
            )
            np.testing.assert_allclose(actions1[0], states1[1])
            np.testing.assert_allclose(
                actions1[-1],
                [1.12, 0.24, 0.36, 0.0, 0.0, 0.2, 0.023],
            )

            conversion = json.loads(
                (out_root / "meta" / "tabero_conversion.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(conversion["version"], 3)
            self.assertEqual(
                conversion["action_source"],
                "observation.state at the next source frame within the same episode",
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

    def test_compact_export_excludes_source_episode_and_reports_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            raw_root = temp / "raw"
            out_root = temp / "out"
            _build_raw_dataset(raw_root)
            episode = raw_root / "data" / "episodes" / "episode_000000.parquet"
            frame = pd.read_parquet(episode)
            frame.loc[1, "candidate_index"] = 2
            frame.to_parquet(episode, index=False)

            with mock.patch.object(tabero_export, "DMTacSource", _NpyDMTacSource):
                result = export_raw_dmtac_w_to_tabero_lerobot(
                    raw_root=raw_root,
                    out_root=out_root,
                    timing_policy="compact",
                    exclude_episodes=[1],
                )

            self.assertEqual(result["total_episodes"], 1)
            self.assertEqual(result["total_frames"], 1)
            self.assertEqual(result["excluded_source_episode_indices"], [1])
            self.assertEqual(result["compacted_source_episode_indices"], [0])
            conversion = json.loads(
                (out_root / "meta" / "tabero_conversion.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(conversion["version"], 3)
            self.assertEqual(conversion["excluded_source_episode_indices"], [1])
            self.assertEqual(conversion["converted_source_episode_indices"], [0])
            self.assertEqual(conversion["compacted_source_episode_indices"], [0])
            self.assertEqual(conversion["total_missing_candidate_steps"], 1)
            self.assertEqual(conversion["timing_reports"][0]["source_episode_index"], 0)
            self.assertEqual(conversion["timing_reports"][0]["output_episode_index"], 0)
            self.assertEqual(conversion["timing_reports"][0]["missing_candidate_steps"], 1)
            self.assertTrue(conversion["timing_reports"][0]["compacted"])

    def test_exclude_episode_cli_parsing(self) -> None:
        args = tabero_export.build_arg_parser().parse_args(
            [
                "--raw-root",
                "/tmp/raw",
                "--out-root",
                "/tmp/out",
                "--exclude-episodes",
                "2",
                "29",
            ]
        )
        self.assertEqual(args.exclude_episodes, [2, 29])
        self.assertEqual(
            tabero_export._source_episode_index(
                Path("data") / "chunk-000" / "file-000.parquet"
            ),
            0,
        )

    def test_legacy_normalized_state_is_converted_to_finger_meters(self) -> None:
        row = pd.Series(
            {
                "observation.state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
            }
        )
        state = tabero_export._tabero_state(
            row,
            state_gripper_unit="normalized",
            state_gripper_coordinate="finger",
            gripper_open_width_m=0.085,
        )
        self.assertAlmostEqual(float(state[6]), 0.0425)

    def test_state_conversion_does_not_read_command_columns(self) -> None:
        row = pd.Series(
            {
                "observation.state": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.02],
            }
        )
        state = tabero_export._tabero_state(
            row,
            state_gripper_unit="meter",
            state_gripper_coordinate="finger",
            gripper_open_width_m=0.085,
        )
        self.assertAlmostEqual(float(state[6]), 0.02)


if __name__ == "__main__":
    unittest.main()
