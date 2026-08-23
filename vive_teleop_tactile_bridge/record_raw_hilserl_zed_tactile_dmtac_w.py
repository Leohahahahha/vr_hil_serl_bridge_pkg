#!/usr/bin/env python3
"""Record HIL-SERL data with two legacy DM-Tac W SDK 0.1.4 sensors.

The same-host bridge publishes one fixed-layout packed frame per sensor. The
worker reads the two sensors concurrently while keeping each sensor's five SDK
getters serial. The packed message carries the SDK capture midpoint; the
recorder maps it into the monotonic clock domain before nearest-frame matching.
The legacy SDK does not expose a hardware frame id or an atomic snapshot API.
"""
from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, String

try:
    from .dmtac_w_ipc import (
        PACKED_IMAGE_ENCODING,
        PAYLOAD_BYTES,
        capture_wall_ns_to_monotonic_sec,
        packed_layout_metadata,
    )
    from .raw_dataset_writer import DMTAC_W_RAW_WRITER_PROFILE, RawDatasetWriter
    from .record_raw_hilserl_zed_tactile_paxini import (
        RawCollectionNode as CommonRawCollectionNode,
        TimedBuffer,
        now_monotonic,
        record_episode,
        ros_stamp_to_float_sec,
        wait_for_topics_ready,
    )
except ImportError:
    from dmtac_w_ipc import (  # type: ignore
        PACKED_IMAGE_ENCODING,
        PAYLOAD_BYTES,
        capture_wall_ns_to_monotonic_sec,
        packed_layout_metadata,
    )
    from raw_dataset_writer import DMTAC_W_RAW_WRITER_PROFILE, RawDatasetWriter  # type: ignore
    from record_raw_hilserl_zed_tactile_paxini import (  # type: ignore
        RawCollectionNode as CommonRawCollectionNode,
        TimedBuffer,
        now_monotonic,
        record_episode,
        ros_stamp_to_float_sec,
        wait_for_topics_ready,
    )


DMTAC_SCHEMA_VERSION = 1
DMTAC_IMAGE_SHAPE = (240, 320)
DMTAC_PACKED_FRAME_BYTES = PAYLOAD_BYTES


@dataclass(frozen=True)
class ImageModalitySpec:
    encoding: str
    dtype: np.dtype[Any]
    channels: int
    fixed_shape: Optional[tuple[int, int]] = None


DMTAC_IMAGE_SPECS = {
    "raw_image": ImageModalitySpec("mono8", np.dtype(np.uint8), 1, DMTAC_IMAGE_SHAPE),
    "deformation2d": ImageModalitySpec("32FC2", np.dtype(np.float32), 2, DMTAC_IMAGE_SHAPE),
    # Verified on both physical sensors: getNormal() is one float32 channel.
    "normal": ImageModalitySpec("32FC1", np.dtype(np.float32), 1, DMTAC_IMAGE_SHAPE),
    "shear": ImageModalitySpec("32FC2", np.dtype(np.float32), 2, DMTAC_IMAGE_SHAPE),
    "depth": ImageModalitySpec("32FC1", np.dtype(np.float32), 1, DMTAC_IMAGE_SHAPE),
}
DMTAC_FRAME_PARTS = tuple(DMTAC_IMAGE_SPECS)


@dataclass
class RecordConfig:
    dataset_root: Path
    fps: float
    num_episodes: int
    max_episode_sec: float
    warmup_timeout_sec: float
    task_description: str
    task_index: int
    wait_for_motion_enable_to_record: bool
    debug: bool

    image_topic: str
    image_name: str
    wrist_image_topic: str
    wrist_camera_info_topic: str
    wrist_image_name: str

    raw_vr_target_pose_topic: str
    enabled_topic: str
    joystick_y_topic: str
    command_event_topic: str
    robot_state_topic: str
    tactile_msg_package: str
    tactile_left_packed_topic: str
    tactile_right_packed_topic: str

    max_image_dt_sec: float
    max_wrist_image_dt_sec: float
    max_action_dt_sec: float
    max_state_dt_sec: float
    max_vr_dt_sec: float
    max_tactile_dt_sec: float
    sync_lag_sec: float
    use_image_header_stamp: bool

    save_master_parquet_every_episode: bool
    save_debug_jsonl: bool
    write_queue_size: int
    tactile_batch_frames: int


def _side_packed_topic(tactile: dict[str, Any], side: str, base_topic: str) -> str:
    side_cfg = tactile.get(side, {})
    if not isinstance(side_cfg, dict):
        raise ValueError(f"tactile.{side} must be a mapping")
    return str(side_cfg.get("packed_topic", f"{base_topic}/{side}/packed_frame"))


def load_config(path: str | Path) -> RecordConfig:
    path = Path(path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    dataset = raw.get("dataset", {})
    record = raw.get("record", {})
    image = raw.get("image") or raw.get("zed_image") or {}
    wrist_image = raw.get("wrist_image") or raw.get("d405_image") or {}
    vr = raw.get("vr", {})
    command = raw.get("command", {})
    robot_state = raw.get("robot_state", {})
    tactile = raw.get("tactile", {})
    sync = raw.get("sync", {})
    output = raw.get("output", {})
    tactile_msg_package = str(tactile.get("msg_package", "dmtac_tactile"))
    if tactile_msg_package != "dmtac_tactile":
        raise ValueError(
            f"DM-Tac recorder requires tactile.msg_package=dmtac_tactile, got: {tactile_msg_package}"
        )
    base_topic = str(tactile.get("base_topic", "/dmtac")).rstrip("/")
    if not base_topic.startswith("/"):
        raise ValueError("tactile.base_topic must be an absolute ROS topic namespace")

    return RecordConfig(
        dataset_root=Path(dataset.get("root", "./fr3_dmtac_raw_dataset")).expanduser().resolve(),
        fps=float(record.get("fps", 10.0)),
        num_episodes=int(record.get("num_episodes", 50)),
        max_episode_sec=float(record.get("max_episode_sec", 300.0)),
        warmup_timeout_sec=float(record.get("warmup_timeout_sec", 30.0)),
        task_description=str(record.get("task_description", "fr3 with DM-Tac W raw teleop")),
        task_index=int(record.get("task_index", 0)),
        wait_for_motion_enable_to_record=bool(record.get("wait_for_motion_enable_to_record", False)),
        debug=bool(record.get("debug", False)),
        image_topic=str(image.get("topic", "/zed/zed_node/rgb/color/rect/image")),
        image_name=str(image.get("name", "front")),
        wrist_image_topic=str(wrist_image.get("topic", "/camera/d405/color/image_raw")),
        wrist_camera_info_topic=str(
            wrist_image.get("camera_info_topic", "/camera/d405/color/camera_info")
        ),
        wrist_image_name=str(wrist_image.get("name", "wrist")),
        raw_vr_target_pose_topic=str(vr.get("target_pose_topic", "/vr_bridge/target_pose")),
        enabled_topic=str(vr.get("enabled_topic", "/vr_bridge/enabled")),
        joystick_y_topic=str(vr.get("joystick_y_topic", "/vr/right_controller/joystick_y")),
        command_event_topic=str(command.get("command_event_topic", "/hilserl/command_event_json")),
        robot_state_topic=str(robot_state.get("state_topic", "/hilserl/robot_state_json")),
        tactile_msg_package=tactile_msg_package,
        tactile_left_packed_topic=_side_packed_topic(tactile, "left", base_topic),
        tactile_right_packed_topic=_side_packed_topic(tactile, "right", base_topic),
        max_image_dt_sec=float(sync.get("max_image_dt_sec", 0.10)),
        max_wrist_image_dt_sec=float(sync.get("max_wrist_image_dt_sec", 0.10)),
        max_action_dt_sec=float(sync.get("max_action_dt_sec", 0.08)),
        max_state_dt_sec=float(sync.get("max_state_dt_sec", 0.08)),
        max_vr_dt_sec=float(sync.get("max_vr_dt_sec", 0.10)),
        max_tactile_dt_sec=float(sync.get("max_tactile_dt_sec", 0.15)),
        sync_lag_sec=max(0.0, float(sync.get("sync_lag_sec", 0.25))),
        use_image_header_stamp=bool(sync.get("use_image_header_stamp", True)),
        save_master_parquet_every_episode=bool(
            output.get("save_master_parquet_every_episode", True)
        ),
        save_debug_jsonl=bool(output.get("save_debug_jsonl", True)),
        write_queue_size=max(4, int(output.get("write_queue_size", 48))),
        tactile_batch_frames=max(1, int(output.get("tactile_batch_frames", 8))),
    )


def _image_msg_to_array(msg: Image, modality: str) -> np.ndarray:
    spec = DMTAC_IMAGE_SPECS[modality]
    height = int(msg.height)
    width = int(msg.width)
    if height <= 0 or width <= 0:
        raise ValueError(f"{modality} has invalid image size: {width}x{height}")
    if spec.fixed_shape is not None and (height, width) != spec.fixed_shape:
        raise ValueError(
            f"{modality} image size mismatch: expected={spec.fixed_shape[1]}x"
            f"{spec.fixed_shape[0]}, actual={width}x{height}"
        )
    if str(msg.encoding) != spec.encoding:
        raise ValueError(
            f"{modality} encoding mismatch: expected={spec.encoding}, actual={msg.encoding}"
        )

    row_bytes = width * spec.channels * spec.dtype.itemsize
    if int(msg.step) < row_bytes:
        raise ValueError(f"{modality} step is too small: expected>={row_bytes}, actual={msg.step}")
    if len(msg.data) < int(msg.step) * height:
        raise ValueError(f"{modality} data buffer is shorter than height*step")
    if spec.dtype == np.dtype(np.float32) and bool(msg.is_bigendian):
        raise ValueError(f"{modality} big-endian float images are not supported")

    rows = np.frombuffer(msg.data, dtype=np.uint8, count=int(msg.step) * height)
    rows = rows.reshape(height, int(msg.step))[:, :row_bytes].copy()
    shape = (height, width) if spec.channels == 1 else (height, width, spec.channels)
    if spec.dtype == np.dtype(np.uint8):
        return rows.reshape(shape)
    return rows.reshape(-1).view("<f4").reshape(shape)


def _pack_modalities(
    arrays: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, int]]:
    chunks: list[np.ndarray] = []
    layout: dict[str, int] = {
        "schema_version": DMTAC_SCHEMA_VERSION,
        "byte_order_little_endian": 1,
    }
    offset = 0
    for modality, spec in DMTAC_IMAGE_SPECS.items():
        array = arrays[modality]
        if spec.dtype == np.dtype(np.uint8):
            encoded = np.ascontiguousarray(array, dtype=np.uint8).reshape(-1)
            itemsize = 1
        else:
            encoded = np.ascontiguousarray(array, dtype="<f4").view(np.uint8).reshape(-1)
            itemsize = 4
        layout[f"{modality}_start"] = offset
        layout[f"{modality}_len"] = int(encoded.size)
        layout[f"{modality}_height"] = int(array.shape[0])
        layout[f"{modality}_width"] = int(array.shape[1])
        layout[f"{modality}_channels"] = spec.channels
        layout[f"{modality}_itemsize"] = itemsize
        chunks.append(encoded)
        offset += int(encoded.size)

    packed = np.concatenate(chunks).astype(np.uint8, copy=False)
    layout["packed_frame_bytes"] = int(packed.size)
    if int(packed.size) != DMTAC_PACKED_FRAME_BYTES:
        raise RuntimeError(
            "DM-Tac W packed frame size mismatch: "
            f"expected={DMTAC_PACKED_FRAME_BYTES}, actual={packed.size}"
        )
    return packed, layout


def _parse_frame_id(frame_id: str) -> tuple[str, int]:
    marker = "|fid="
    if marker not in frame_id:
        raise ValueError(f"invalid DM-Tac frame_id (missing '{marker}'): {frame_id!r}")
    sensor_id, fid_text = frame_id.rsplit(marker, 1)
    if not sensor_id:
        raise ValueError("DM-Tac frame_id has an empty sensor identification code")
    try:
        sdk_fid = int(fid_text)
    except ValueError as exc:
        raise ValueError(f"invalid DM-Tac SDK frame id: {fid_text!r}") from exc
    if sdk_fid < 0:
        raise ValueError(f"DM-Tac SDK frame id must be non-negative: {sdk_fid}")
    return sensor_id, sdk_fid


def _packed_msg_to_payload(
    msg: Image,
    *,
    side: str,
    sensor_index: int,
    receive_monotonic_ns: int,
    receive_wall_ns: int,
) -> tuple[float, dict[str, Any]]:
    if int(msg.height) != 1 or int(msg.width) != DMTAC_PACKED_FRAME_BYTES:
        raise ValueError(
            f"{side} packed frame shape must be 1x{DMTAC_PACKED_FRAME_BYTES}, "
            f"got {msg.height}x{msg.width}"
        )
    if str(msg.encoding) != PACKED_IMAGE_ENCODING or int(msg.step) != DMTAC_PACKED_FRAME_BYTES:
        raise ValueError(
            f"{side} packed frame encoding/step mismatch: "
            f"encoding={msg.encoding!r}, step={msg.step}"
        )
    packed = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1).copy()
    if packed.size != DMTAC_PACKED_FRAME_BYTES:
        raise ValueError(
            f"{side} packed frame bytes={packed.size}, expected={DMTAC_PACKED_FRAME_BYTES}"
        )
    sensor_id, sdk_fid = _parse_frame_id(str(msg.header.frame_id))
    capture_wall_ns = (
        int(msg.header.stamp.sec) * 1_000_000_000
        + int(msg.header.stamp.nanosec)
    )
    capture_monotonic = capture_wall_ns_to_monotonic_sec(
        capture_wall_ns,
        receive_wall_ns=receive_wall_ns,
        receive_monotonic_ns=receive_monotonic_ns,
    )
    receive_monotonic = receive_monotonic_ns * 1e-9
    capture_wall_sec = capture_wall_ns * 1e-9
    payload = {
        "msg_package": "dmtac_tactile",
        "sensor_id": sensor_id,
        "sensor_index": sensor_index,
        "frame_idx": sdk_fid,
        "capture_time": capture_wall_sec,
        "capture_monotonic": capture_monotonic,
        "transport_delay_sec": receive_monotonic - capture_monotonic,
        "repeated": False,
        "data": packed,
        "data_len": int(packed.size),
        "data_dtype": "uint8",
        "source_dtype": (
            "raw_image:uint8;deformation2d,normal,shear,depth:float32_le"
        ),
        "layout": packed_layout_metadata(),
        "ros_stamp_sec": int(msg.header.stamp.sec),
        "ros_stamp_nanosec": int(msg.header.stamp.nanosec),
        "ros_stamp_float": capture_wall_sec,
        "frame_id": str(msg.header.frame_id),
        "recv_time": receive_monotonic,
    }
    return capture_monotonic, payload


class DMTacFrameAssembler:
    def __init__(self, side: str, sensor_index: int) -> None:
        self.side = side
        self.sensor_index = sensor_index
        self.pending: dict[int, dict[str, Any]] = {}
        self.last_completed_stamp_ns = -1

    def add(self, modality: str, msg: Any, recv_time: float) -> Optional[dict[str, Any]]:
        stamp = msg.header.stamp
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if stamp_ns <= self.last_completed_stamp_ns:
            return None
        frame = self.pending.setdefault(stamp_ns, {})
        frame[modality] = msg
        while len(self.pending) > 8:
            self.pending.pop(min(self.pending), None)
        if any(name not in frame for name in DMTAC_FRAME_PARTS):
            return None

        frame_ids = {str(frame[name].header.frame_id) for name in DMTAC_FRAME_PARTS}
        if len(frame_ids) != 1 or not next(iter(frame_ids)):
            raise ValueError(f"{self.side} DM-Tac frame has inconsistent or empty sensor IDs")
        frame_id = next(iter(frame_ids))
        sensor_id, sdk_fid = _parse_frame_id(frame_id)
        arrays = {
            name: _image_msg_to_array(frame[name], name)
            for name in DMTAC_IMAGE_SPECS
        }
        perception_shape = arrays["depth"].shape
        for name in ("deformation2d", "normal", "shear"):
            if arrays[name].shape[:2] != perception_shape:
                raise ValueError(
                    f"{self.side} {name} shape {arrays[name].shape[:2]} does not match "
                    f"depth {perception_shape}"
                )
        packed, layout = _pack_modalities(arrays)
        payload = {
            "msg_package": "dmtac_tactile",
            "sensor_id": sensor_id,
            "sensor_index": self.sensor_index,
            "frame_idx": sdk_fid,
            "capture_time": ros_stamp_to_float_sec(stamp),
            "repeated": False,
            "data": packed,
            "data_len": int(packed.size),
            "data_dtype": "uint8",
            "source_dtype": (
                "raw_image:uint8;deformation2d,normal,shear,depth:float32_le"
            ),
            "layout": layout,
            "ros_stamp_sec": int(stamp.sec),
            "ros_stamp_nanosec": int(stamp.nanosec),
            "ros_stamp_float": ros_stamp_to_float_sec(stamp),
            "frame_id": frame_id,
            "recv_time": float(recv_time),
        }
        self.last_completed_stamp_ns = stamp_ns
        for old_stamp in [key for key in self.pending if key <= stamp_ns]:
            self.pending.pop(old_stamp, None)
        return payload


class RawCollectionNode(CommonRawCollectionNode):
    def __init__(self, cfg: RecordConfig):
        Node.__init__(self, "dmtac_w_raw_hilserl_zed_recorder")
        self.cfg = cfg
        self._lock = threading.Lock()

        self.image_buffer = TimedBuffer(maxlen=96)
        self.wrist_image_buffer = TimedBuffer(maxlen=96)
        self.wrist_camera_info_buffer = TimedBuffer(maxlen=128)
        self.command_buffer = TimedBuffer(maxlen=4096)
        self.state_buffer = TimedBuffer(maxlen=4096)
        self.vr_pose_buffer = TimedBuffer(maxlen=2048)
        self.enabled_buffer = TimedBuffer(maxlen=2048)
        self.joystick_y_buffer = TimedBuffer(maxlen=2048)
        self.tactile_left_buffer = TimedBuffer(maxlen=64)
        self.tactile_right_buffer = TimedBuffer(maxlen=64)

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        fast_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        packed_tactile_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        log_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )

        self.create_subscription(Image, cfg.image_topic, self._image_cb, image_qos)
        self.create_subscription(Image, cfg.wrist_image_topic, self._wrist_image_cb, image_qos)
        self.create_subscription(
            CameraInfo, cfg.wrist_camera_info_topic, self._wrist_camera_info_cb, image_qos
        )
        self.create_subscription(String, cfg.command_event_topic, self._command_event_cb, log_qos)
        self.create_subscription(String, cfg.robot_state_topic, self._robot_state_cb, log_qos)
        self.create_subscription(PoseStamped, cfg.raw_vr_target_pose_topic, self._vr_pose_cb, fast_qos)
        self.create_subscription(Bool, cfg.enabled_topic, self._enabled_cb, fast_qos)
        self.create_subscription(Float32, cfg.joystick_y_topic, self._joystick_y_cb, fast_qos)

        for side, topic, sensor_index in (
            ("left", cfg.tactile_left_packed_topic, 0),
            ("right", cfg.tactile_right_packed_topic, 1),
        ):
            self.create_subscription(
                Image,
                topic,
                lambda msg, s=side, i=sensor_index: self._dmtac_packed_cb(s, i, msg),
                packed_tactile_qos,
            )
            self.get_logger().info(f"subscribe DM-Tac {side} packed frame: {topic}")

    def _dmtac_packed_cb(self, side: str, sensor_index: int, msg: Image) -> None:
        receive_monotonic_ns = time.monotonic_ns()
        receive_wall_ns = time.time_ns()
        try:
            capture_monotonic, payload = _packed_msg_to_payload(
                msg,
                side=side,
                sensor_index=sensor_index,
                receive_monotonic_ns=receive_monotonic_ns,
                receive_wall_ns=receive_wall_ns,
            )
        except Exception as exc:
            self.get_logger().error(f"invalid DM-Tac {side} packed frame: {exc!r}")
            return
        with self._lock:
            target = self.tactile_left_buffer if side == "left" else self.tactile_right_buffer
            target.append(capture_monotonic, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)
    cfg.dataset_root.mkdir(parents=True, exist_ok=True)

    print("====== [START] Raw HIL-SERL ZED legacy DM-Tac W recorder ======")
    print(f"[DATASET] root: {cfg.dataset_root}")
    print(f"[FPS] {cfg.fps}")
    print(f"[IMAGE] {cfg.image_topic}")
    print(f"[COMMAND] {cfg.command_event_topic}")
    print(f"[STATE] {cfg.robot_state_topic}")
    print(
        f"[SYNC] lag={cfg.sync_lag_sec:.3f}s image_header_stamp={cfg.use_image_header_stamp}"
    )
    print(
        f"[WRITER] queue={cfg.write_queue_size} tasks "
        f"tactile_batch={cfg.tactile_batch_frames} frames"
    )
    print(
        f"[DM-TAC W] SDK 0.1.4 packed schema version: {DMTAC_SCHEMA_VERSION} "
        f"({DMTAC_IMAGE_SHAPE[1]}x{DMTAC_IMAGE_SHAPE[0]}, "
        f"{DMTAC_PACKED_FRAME_BYTES} bytes/side)"
    )

    writer = RawDatasetWriter(cfg, config_path=cfg_path, profile=DMTAC_W_RAW_WRITER_PROFILE)
    writer.prepare()

    rclpy.init(args=None)
    node = RawCollectionNode(cfg)
    spin_running = True

    def spin_loop() -> None:
        while spin_running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)

    spin_thread = threading.Thread(target=spin_loop, daemon=True)
    spin_thread.start()

    try:
        wait_for_topics_ready(node, timeout_sec=cfg.warmup_timeout_sec)
        episode_index = 0
        while episode_index < cfg.num_episodes:
            saved, quit_after = record_episode(cfg, node, writer, episode_index)
            if quit_after:
                print("[QUIT] user requested quit")
                break
            if saved:
                episode_index += 1
                if episode_index < cfg.num_episodes:
                    input("\n====== [RESET] Reset scene/robot, then press Enter for next episode ======")
            else:
                print("[RETRY] re-record same episode index")
        writer.flush_master()
    except KeyboardInterrupt:
        print("\n[KeyboardInterrupt] stopping")
        writer.flush_master()
    finally:
        spin_running = False
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.shutdown()
        writer.close()

    print("====== [END] legacy DM-Tac W recording finished ======")


if __name__ == "__main__":
    main()
