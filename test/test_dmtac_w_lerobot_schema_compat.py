import importlib
import sys
import types
import unittest

import numpy as np


def _load_decoder_without_offline_export_dependencies():
    """Load the packed decoder without requiring video/parquet dependencies."""
    module_name = (
        "vive_teleop_tactile_bridge.lerobot_export."
        "raw_dmtac_w_to_n0vtla_lerobot"
    )
    stub_names = ("cv2", "pandas", "pyarrow", "pyarrow.parquet", "PIL", "PIL.Image")
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in stub_names}

    cv2 = types.ModuleType("cv2")
    pandas = types.ModuleType("pandas")
    pyarrow = types.ModuleType("pyarrow")
    pyarrow.__path__ = []
    parquet = types.ModuleType("pyarrow.parquet")
    pyarrow.parquet = parquet
    pil = types.ModuleType("PIL")
    pil.__path__ = []
    image = types.ModuleType("PIL.Image")
    pil.Image = image
    stubs = {
        "cv2": cv2,
        "pandas": pandas,
        "pyarrow": pyarrow,
        "pyarrow.parquet": parquet,
        "PIL": pil,
        "PIL.Image": image,
    }
    try:
        sys.modules.update(stubs)
        module = importlib.import_module(module_name)
        return module.DMTacSource
    finally:
        # Do not make later tests inherit our lightweight stand-ins. The
        # decoder class keeps the imported module globals it needs.
        sys.modules.pop(module_name, None)
        for name, old_value in previous.items():
            if old_value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value


DMTacSource = _load_decoder_without_offline_export_dependencies()


HEIGHT = 240
WIDTH = 320
SHEAR_BYTES = HEIGHT * WIDTH * 2 * 4
DEPTH_BYTES = HEIGHT * WIDTH * 4


def _layout(schema_version: int) -> dict[str, int | str]:
    if schema_version == 1:
        packed_bytes = 1_920_000
        shear_start = 998_400
        depth_start = 1_612_800
        output_mode = "full"
    elif schema_version == 3:
        packed_bytes = 921_600
        shear_start = 0
        depth_start = 614_400
        output_mode = "shear_depth"
    else:
        raise AssertionError(f"test has no layout for schema {schema_version}")
    return {
        "schema_version": schema_version,
        "output_mode": output_mode,
        "byte_order_little_endian": 1,
        "packed_frame_bytes": packed_bytes,
        "shear_start": shear_start,
        "shear_len": SHEAR_BYTES,
        "shear_height": HEIGHT,
        "shear_width": WIDTH,
        "shear_channels": 2,
        "shear_itemsize": 4,
        "depth_start": depth_start,
        "depth_len": DEPTH_BYTES,
        "depth_height": HEIGHT,
        "depth_width": WIDTH,
        "depth_channels": 1,
        "depth_itemsize": 4,
    }


def _source(schema_version: int, *, include_output_mode: bool = True) -> DMTacSource:
    layout = _layout(schema_version)
    if not include_output_mode:
        # Old schema-1 recordings predate this informational metadata key.
        layout.pop("output_mode")
    payload = np.zeros(int(layout["packed_frame_bytes"]), dtype=np.uint8)
    shear = np.empty((HEIGHT, WIDTH, 2), dtype="<f4")
    shear[..., 0] = 1.25
    shear[..., 1] = -2.5
    depth = np.full((HEIGHT, WIDTH, 1), 3.75, dtype="<f4")
    shear_start = int(layout["shear_start"])
    depth_start = int(layout["depth_start"])
    payload[shear_start : shear_start + SHEAR_BYTES] = shear.view(np.uint8).reshape(-1)
    payload[depth_start : depth_start + DEPTH_BYTES] = depth.view(np.uint8).reshape(-1)

    source = object.__new__(DMTacSource)
    source.side = "left"
    source.array = payload.reshape(1, -1)
    source.meta = {
        0: {
            "zarr_index": 0,
            "msg_package": "dmtac_tactile",
            "layout": layout,
        }
    }
    return source


class DMTacSchemaCompatibilityTest(unittest.TestCase):
    def test_legacy_schema_1_decodes_shear_depth_in_n0vtla_order(self) -> None:
        field = _source(1, include_output_mode=False).field(0)
        self.assertEqual(field.shape, (HEIGHT, WIDTH, 3))
        np.testing.assert_allclose(field[0, 0], [1.25, -2.5, 3.75])

    def test_schema_3_without_output_mode_decodes_in_n0vtla_order(self) -> None:
        # Current online layout metadata contains integers only; output_mode is
        # recorded at dataset level and is optional in each row's layout.
        source = _source(3, include_output_mode=False)
        self.assertEqual(source.schema_version(0), 3)
        field = source.field(0)
        self.assertEqual(field.shape, (HEIGHT, WIDTH, 3))
        np.testing.assert_allclose(field[-1, -1], [1.25, -2.5, 3.75])

    def test_schema_and_output_mode_mismatch_is_rejected(self) -> None:
        source = _source(3)
        source.meta[0]["layout"]["output_mode"] = "full"
        with self.assertRaisesRegex(ValueError, "requires output_mode='shear_depth'"):
            source.field(0)


if __name__ == "__main__":
    unittest.main()
