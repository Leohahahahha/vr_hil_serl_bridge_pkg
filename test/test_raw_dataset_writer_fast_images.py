from __future__ import annotations

from io import BytesIO
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest

from PIL import Image as PILImage


def _load_writer_with_ros_stubs():
    sensor_msgs = types.ModuleType("sensor_msgs")
    sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")

    class Image:
        pass

    class CompressedImage:
        pass

    sensor_msgs_msg.Image = Image
    sensor_msgs_msg.CompressedImage = CompressedImage
    sensor_msgs.msg = sensor_msgs_msg

    action_space = types.ModuleType("vive_teleop_tactile_bridge.action_space")
    action_space.ACTION_NAMES = [f"action_{index}" for index in range(7)]
    action_space.STATE_NAMES = [f"state_{index}" for index in range(7)]
    image_utils = types.ModuleType("vive_teleop_tactile_bridge.image_utils")
    image_utils.image_msg_to_rgb8 = lambda _msg: None
    observation_builder = types.ModuleType(
        "vive_teleop_tactile_bridge.observation_builder"
    )
    observation_builder.ros_stamp_to_float_sec = (
        lambda stamp: float(stamp.sec) + float(stamp.nanosec) * 1e-9
    )

    replacements = {
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "vive_teleop_tactile_bridge.action_space": action_space,
        "vive_teleop_tactile_bridge.image_utils": image_utils,
        "vive_teleop_tactile_bridge.observation_builder": observation_builder,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    previous_writer = sys.modules.pop(
        "vive_teleop_tactile_bridge.raw_dataset_writer", None
    )
    try:
        sys.modules.update(replacements)
        module = importlib.import_module(
            "vive_teleop_tactile_bridge.raw_dataset_writer"
        )
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
        if previous_writer is not None:
            sys.modules["vive_teleop_tactile_bridge.raw_dataset_writer"] = (
                previous_writer
            )
    return module, CompressedImage


writer_module, CompressedImage = _load_writer_with_ros_stubs()


class RawDatasetWriterFastImageTest(unittest.TestCase):
    def make_jpeg_message(self):
        stream = BytesIO()
        PILImage.new("RGB", (16, 12), color=(20, 40, 60)).save(
            stream, format="JPEG", quality=80
        )
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.data = stream.getvalue()
        return msg

    def make_writer(self):
        writer = object.__new__(writer_module.RawDatasetWriter)
        writer._preserve_compressed_images = True
        writer._png_compress_level = 1
        return writer

    def test_preserves_source_jpeg_bytes_without_reencoding(self) -> None:
        writer = self.make_writer()
        msg = self.make_jpeg_message()
        self.assertEqual(writer._image_extension(msg), ".jpg")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frame.jpg"
            writer._write_image(msg, output)
            self.assertEqual(output.read_bytes(), msg.data)
            with PILImage.open(output) as image:
                self.assertEqual(image.size, (16, 12))
                self.assertEqual(image.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
