from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1] / "vive_teleop_tactile_bridge"
sys.path.insert(0, str(PACKAGE_DIR))

import dmtac_w_ipc as ipc  # noqa: E402
import dmtac_w_sdk_worker as sdk_worker  # noqa: E402


def _zero_modalities(output_mode: str) -> dict[str, np.ndarray]:
    return {
        spec.name: np.zeros(spec.shape, dtype=spec.dtype)
        for spec in ipc.get_modality_specs(output_mode)
    }


class DMTacWLayoutModeTest(unittest.TestCase):
    def test_legacy_defaults_still_mean_full_mode(self) -> None:
        expected_names = (
            "raw_image",
            "deformation2d",
            "normal",
            "shear",
            "depth",
        )
        self.assertEqual(ipc.get_payload_bytes(), 1_920_000)
        self.assertEqual(tuple(spec.name for spec in ipc.MODALITY_SPECS), expected_names)
        self.assertEqual(
            len(ipc.pack_modalities(_zero_modalities(ipc.OUTPUT_MODE_FULL))),
            1_920_000,
        )

    def test_repository_configs_select_the_same_compact_mode(self) -> None:
        repository_root = PACKAGE_DIR.parent
        bridge_text = (
            repository_root / "config" / "dmtac_w_bridge.params.yaml"
        ).read_text(encoding="utf-8")
        recorder_text = (
            repository_root
            / "config"
            / "record_hilserl_raw_tactile_dmtac_w.yaml"
        ).read_text(encoding="utf-8")

        mode_pattern = re.compile(
            r'^\s*output_mode:\s*["\']?([^"\'\s#]+)', re.MULTILINE
        )
        bridge_match = mode_pattern.search(bridge_text)
        recorder_match = mode_pattern.search(recorder_text)
        self.assertIsNotNone(bridge_match)
        self.assertIsNotNone(recorder_match)
        assert bridge_match is not None and recorder_match is not None
        bridge_mode = bridge_match.group(1)
        recorder_mode = recorder_match.group(1)
        self.assertEqual(bridge_mode, ipc.OUTPUT_MODE_SHEAR_DEPTH)
        self.assertEqual(recorder_mode, bridge_mode)

    def test_layout_sizes_order_and_roundtrip(self) -> None:
        expected = {
            ipc.OUTPUT_MODE_FULL: (
                1_920_000,
                ("raw_image", "deformation2d", "normal", "shear", "depth"),
            ),
            ipc.OUTPUT_MODE_SHEAR_DEPTH: (
                921_600,
                ("shear", "depth"),
            ),
        }
        for output_mode, (payload_bytes, names) in expected.items():
            with self.subTest(output_mode=output_mode):
                specs = ipc.get_modality_specs(output_mode)
                self.assertEqual(tuple(spec.name for spec in specs), names)
                self.assertEqual(ipc.get_payload_bytes(output_mode), payload_bytes)
                self.assertEqual(
                    tuple(spec.offset for spec in specs),
                    tuple(
                        sum(previous.nbytes for previous in specs[:index])
                        for index in range(len(specs))
                    ),
                )

                packed = ipc.pack_modalities(
                    _zero_modalities(output_mode), output_mode
                )
                self.assertEqual(len(packed), payload_bytes)
                unpacked = ipc.unpack_modalities(packed, output_mode)
                self.assertEqual(tuple(unpacked), names)

        compact_metadata = ipc.packed_layout_metadata(
            ipc.OUTPUT_MODE_SHEAR_DEPTH
        )
        self.assertEqual(compact_metadata["schema_version"], 3)
        self.assertTrue(all(isinstance(value, int) for value in compact_metadata.values()))

    def test_writer_reader_roundtrip_for_both_modes(self) -> None:
        for output_mode in ipc.OUTPUT_MODES:
            with self.subTest(output_mode=output_mode), tempfile.TemporaryDirectory() as tmp:
                payload = ipc.pack_modalities(
                    _zero_modalities(output_mode), output_mode
                )
                path = Path(tmp) / "sensor.frame"
                writer = ipc.MMapFrameWriter(
                    path, session_id=123, output_mode=output_mode
                )
                reader = ipc.MMapFrameReader(
                    path, session_id=123, output_mode=output_mode
                )
                try:
                    sequence = writer.write(
                        frame_idx=7,
                        capture_start_ns=1_000,
                        capture_end_ns=2_000,
                        payload=payload,
                    )
                    snapshot = reader.read_new(last_sequence=0)
                    self.assertIsNotNone(snapshot)
                    assert snapshot is not None
                    self.assertEqual(snapshot.sequence, sequence)
                    self.assertEqual(snapshot.frame_idx, 7)
                    self.assertEqual(snapshot.payload, payload)
                finally:
                    reader.close()
                    writer.close()

    def test_layout_mismatch_is_retryable_then_bounded(self) -> None:
        output_mode = ipc.OUTPUT_MODE_SHEAR_DEPTH
        payload = ipc.pack_modalities(_zero_modalities(output_mode), output_mode)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sensor.frame"
            writer = ipc.MMapFrameWriter(
                path, session_id=456, output_mode=output_mode
            )
            reader = ipc.MMapFrameReader(
                path,
                session_id=456,
                output_mode=output_mode,
                max_consecutive_layout_mismatches=2,
            )
            try:
                writer.write(
                    frame_idx=1,
                    capture_start_ns=10,
                    capture_end_ns=20,
                    payload=payload,
                )
                payload_size_offset = ipc.HEADER_BYTES - ipc.SEQUENCE_STRUCT.size
                ipc.SEQUENCE_STRUCT.pack_into(writer._mmap, payload_size_offset, 0)
                self.assertIsNone(reader.read_new(last_sequence=0))
                self.assertEqual(reader.consecutive_layout_mismatches, 1)

                ipc.SEQUENCE_STRUCT.pack_into(
                    writer._mmap, payload_size_offset, writer.payload_bytes
                )
                self.assertIsNotNone(reader.read_new(last_sequence=0))
                self.assertEqual(reader.consecutive_layout_mismatches, 0)

                ipc.SEQUENCE_STRUCT.pack_into(writer._mmap, payload_size_offset, 0)
                self.assertIsNone(reader.read_new(last_sequence=0))
                with self.assertRaisesRegex(RuntimeError, "stable layout mismatch"):
                    reader.read_new(last_sequence=0)
                ipc.SEQUENCE_STRUCT.pack_into(
                    writer._mmap, payload_size_offset, writer.payload_bytes
                )
            finally:
                reader.close()
                writer.close()


class _CompactModeSensor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def getShear(self) -> np.ndarray:
        self.calls.append("shear")
        return np.zeros((240, 320, 2), dtype=np.float32)

    def getDepth(self) -> np.ndarray:
        self.calls.append("depth")
        return np.zeros((240, 320), dtype=np.float32)

    def getRawImage(self) -> np.ndarray:
        raise AssertionError("compact mode must not call getRawImage")

    def getDeformation2D(self) -> np.ndarray:
        raise AssertionError("compact mode must not call getDeformation2D")

    def getNormal(self) -> np.ndarray:
        raise AssertionError("compact mode must not call getNormal")


class DMTacWSdkWorkerModeTest(unittest.TestCase):
    def test_compact_mode_calls_only_shear_and_depth(self) -> None:
        sensor = _CompactModeSensor()
        result = sdk_worker._read_modalities(
            sensor, ipc.OUTPUT_MODE_SHEAR_DEPTH
        )
        self.assertEqual(sensor.calls, ["shear", "depth"])
        self.assertEqual(tuple(result), ("shear", "depth"))


if __name__ == "__main__":
    unittest.main()
