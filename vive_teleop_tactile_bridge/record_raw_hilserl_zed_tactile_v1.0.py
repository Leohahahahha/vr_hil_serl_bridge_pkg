#!/usr/bin/env python3
"""
原始数据采集和保存脚本
数据包括

全局场景相机zed2i的彩色图像

机械臂腕部相机D405的彩色图像

实际发给机器人的动作命令
    action.sent_action8 =  action.pose7 + action.target_gripper
        [
        target_x,
        target_y,
        target_z,
        target_qx,
        target_qy,
        target_qz,
        target_qw,
        target_gripper
        ]
    action.pose7
    action.target_gripper

机械臂本体状态（位置、姿态、关节状态等）和夹爪宽度
    robot.ee_pose：7维数据
    robot.gripper_pos：夹爪宽度
    robot.q：7维数据，关节角度
    robot.dq：7维数据，关节速度
    robot.force：末端执行器受力
    robot.torque：末端执行器受力矩
    robot.raw_state_json：原始状态信息，包含所有关节角度、速度、力、力矩等数据
    
vive控制器动作命令（来自VR控制器的目标位置、姿态和夹爪开合度）

##To do 两个触觉传感器的原始数据

同步信息

This recorder does NOT send any robot command. It only subscribes and writes data.

Output layout:
  dataset_root/

    #zed相机 彩色图像 每条轨迹一个文件夹，每帧一张PNG图像
    image/front/episode_000000/frame_000000.png
    #zed相机 index.jsonl存储所有彩色图像的索引文件，记录每帧图像的相对路径和ROS时间戳等信息
    image/front/index.jsonl

    #d405相机 彩色图像 每条轨迹一个文件夹，每帧一张PNG图像
    image/wrist/episode_000000/frame_000000.png
    #d405相机 index.jsonl存储所有彩色图像的索引文件，记录每帧图像的相对路径和ROS时间戳等信息
    image/wrist/index.jsonl

    #机械臂本体状态（位置、姿态、关节状态等）和夹爪宽度
    raw/robot_state_events.jsonl

    #实际发给机器人的动作命令
    raw/command_events.jsonl

    #vive控制器动作命令
    raw/vr_controller_events.jsonl

    #同步后的信息
    #包含所有数据的同步信息，每个轨迹一个parquet文件，具体是同步后的数值数据本身和外部文件路径,如图像
    # 数据总表
    data/chunk-000/file-000.parquet
    # 按轨迹保存
    data/episodes/episode_000000.parquet

    # 数据集元信息
    meta/info.json    #数据集说明文件
    meta/config.yaml  #本次采集使用的配置文件副本
    meta/episodes.jsonl #每条轨迹的索引文件，记录每条轨迹的数据文件路径和基本信息
    

Run:
  source /opt/ros/jazzy/setup.bash
  python3 record_raw_hilserl_zed_tactile.py --config record_hilserl_raw.yaml
"""
from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Deque

import numpy as np
import pandas as pd
import yaml
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from paxini_tactile.msg import TactileFrame as PaxiniTactileFrame

try:
    from tashan_tactile.msg import TactileFrame as TashanTactileFrame
except ImportError:
    TashanTactileFrame = None
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, String


TACTILE_FRAME_BYTES = 234
TACTILE_FORCE_BYTES = 3

ACTION_NAMES = [
    "target_x", "target_y", "target_z",
    "target_qx", "target_qy", "target_qz", "target_qw",
    "target_gripper",
]


def now_monotonic() -> float:
    return float(time.monotonic())


def ros_stamp_to_float_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = float(np.linalg.norm(q))
    if not np.isfinite(n) or n < 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def parse_json_str(s: str) -> dict[str, Any]:
    try:
        out = json.loads(s)
        return out if isinstance(out, dict) else {"value": out}
    except Exception as e:
        return {"parse_error": repr(e), "raw": s}


@dataclass
class TimedItem:
    t: float
    payload: Any


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
    tactile_left_topic: str
    tactile_right_topic: str

    max_image_dt_sec: float
    max_wrist_image_dt_sec: float
    max_action_dt_sec: float
    max_state_dt_sec: float
    max_vr_dt_sec: float
    max_tactile_dt_sec: float

    save_master_parquet_every_episode: bool
    save_debug_jsonl: bool


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
    tactile_msg_package = str(tactile.get("msg_package", "paxini_tactile"))
    if tactile_msg_package not in ("paxini_tactile", "tashan_tactile"):
        raise ValueError(f"unsupported tactile.msg_package: {tactile_msg_package}")
    default_tactile_ns = f"/{tactile_msg_package}"

    return RecordConfig(
        dataset_root=Path(dataset.get("root", "./fr3_tactile_raw_dataset")).expanduser().resolve(),
        
        #采集频率：10Hz
        
        fps=float(record.get("fps", 10.0)),
        num_episodes=int(record.get("num_episodes", 50)),
        max_episode_sec=float(record.get("max_episode_sec", 300.0)),
        warmup_timeout_sec=float(record.get("warmup_timeout_sec", 10.0)),
        task_description=str(record.get("task_description", "fr3 with tactile raw teleop")),
        task_index=int(record.get("task_index", 0)),
        wait_for_motion_enable_to_record=bool(record.get("wait_for_motion_enable_to_record", True)),
        debug=bool(record.get("debug", False)),

        image_topic=str(image.get("topic", "/zed/zed_node/rgb/color/rect/image")),
        image_name=str(image.get("name", "front")),
        wrist_image_topic=str(wrist_image.get("topic", "/camera/d405/color/image_raw")),
        wrist_camera_info_topic=str(wrist_image.get("camera_info_topic", "/camera/d405/color/camera_info")),
        wrist_image_name=str(wrist_image.get("name", "wrist")),

        raw_vr_target_pose_topic=str(vr.get("target_pose_topic", "/vr_bridge/target_pose")),
        enabled_topic=str(vr.get("enabled_topic", "/vr_bridge/enabled")),
        joystick_y_topic=str(vr.get("joystick_y_topic", "/vr/right_controller/joystick_y")),
        command_event_topic=str(command.get("command_event_topic", "/hilserl/command_event_json")),
        robot_state_topic=str(robot_state.get("state_topic", "/hilserl/robot_state_json")),
        tactile_msg_package=tactile_msg_package,
        tactile_left_topic=str(tactile.get("left_topic", f"{default_tactile_ns}/left/data")),
        tactile_right_topic=str(tactile.get("right_topic", f"{default_tactile_ns}/right/data")),
        
        # 同步时间阈值
        
        max_image_dt_sec=float(sync.get("max_image_dt_sec", sync.get("max_cloud_dt_sec", 0.10))),
        max_wrist_image_dt_sec=float(sync.get("max_wrist_image_dt_sec", sync.get("max_image_dt_sec", 0.10))),
        max_action_dt_sec=float(sync.get("max_action_dt_sec", 0.08)),
        max_state_dt_sec=float(sync.get("max_state_dt_sec", 0.08)),
        max_vr_dt_sec=float(sync.get("max_vr_dt_sec", 0.10)),
        max_tactile_dt_sec=float(sync.get("max_tactile_dt_sec", 0.10)),

        save_master_parquet_every_episode=bool(output.get("save_master_parquet_every_episode", True)),
        save_debug_jsonl=bool(output.get("save_debug_jsonl", True)),
    )


# ============================================================
# Image decode
# ============================================================

def image_msg_to_bgr8(msg: Image) -> np.ndarray:
    """Convert common ROS Image encodings to an OpenCV BGR uint8 image."""
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    step = int(msg.step)

    if height <= 0 or width <= 0:
        raise ValueError(f"invalid image size: {width}x{height}")

    if encoding in ("bgr8", "rgb8"):
        channels = 3
        expected_row_bytes = width * channels
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        arr = arr[:, :expected_row_bytes].reshape(height, width, channels)
        if encoding == "rgb8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(arr)

    if encoding in ("bgra8", "rgba8"):
        channels = 4
        expected_row_bytes = width * channels
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        arr = arr[:, :expected_row_bytes].reshape(height, width, channels)
        if encoding == "rgba8":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return np.ascontiguousarray(arr)

    if encoding in ("mono8", "8uc1"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)
        arr = arr[:, :width].reshape(height, width)
        return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR))

    raise ValueError(f"unsupported image encoding for PNG recording: {msg.encoding}")


def tactile_msg_to_record(msg: Any, recv_time: float, msg_package: str) -> dict[str, Any]:
    if msg_package == "paxini_tactile":
        data = np.asarray(list(msg.data), dtype=np.uint8)
        layout = {
            "force_raw_start": 0,
            "force_raw_len": TACTILE_FORCE_BYTES,
            "distribution_raw_start": TACTILE_FORCE_BYTES,
            "distribution_raw_len": TACTILE_FRAME_BYTES - TACTILE_FORCE_BYTES,
        }
    elif msg_package == "tashan_tactile":
        data = np.asarray(list(msg.data), dtype=np.float32)
        layout = {
            "cap_len": int(msg.cap_len),
            "nf_len": int(msg.nf_len),
            "tf_len": int(msg.tf_len),
            "tf_dir_len": int(msg.tf_dir_len),
            "s_prox_len": int(msg.s_prox_len),
            "m_prox_len": int(msg.m_prox_len),
        }
    else:
        raise ValueError(f"unsupported tactile msg package: {msg_package}")
    stamp = msg.header.stamp
    return {
        "msg_package": msg_package,
        "sensor_id": str(msg.sensor_id),
        "sensor_index": int(msg.sensor_index),
        "frame_idx": int(msg.frame_idx),
        "capture_time": float(msg.capture_time),
        "repeated": bool(msg.repeated),
        "data": data,
        "data_len": int(data.size),
        "data_dtype": str(data.dtype),
        "source_dtype": str(data.dtype),
        "layout": layout,
        "ros_stamp_sec": int(stamp.sec),
        "ros_stamp_nanosec": int(stamp.nanosec),
        "ros_stamp_float": ros_stamp_to_float_sec(stamp),
        "frame_id": str(msg.header.frame_id),
        "recv_time": float(recv_time),
    }


# ============================================================
# Ring buffer helpers
# ============================================================

class TimedBuffer:
    def __init__(self, maxlen: int = 2048):
        self._buf: Deque[TimedItem] = deque(maxlen=maxlen)

    def append(self, t: float, payload: Any) -> None:
        self._buf.append(TimedItem(float(t), payload))

    def latest(self) -> Optional[TimedItem]:
        if not self._buf:
            return None
        return self._buf[-1]

    def nearest(self, t: float) -> Optional[TimedItem]:
        if not self._buf:
            return None
        # Buffer is short; linear scan is simple and robust.
        best = min(self._buf, key=lambda item: abs(item.t - t))
        return best


# ============================================================
# ROS node
# ============================================================

class RawCollectionNode(Node):
    def __init__(self, cfg: RecordConfig):
        super().__init__("tactile_raw_hilserl_zed_recorder")
        self.cfg = cfg
        self._lock = threading.Lock()

        self.image_buffer = TimedBuffer(maxlen=512)
        self.wrist_image_buffer = TimedBuffer(maxlen=512)
        self.wrist_camera_info_buffer = TimedBuffer(maxlen=512)
        self.command_buffer = TimedBuffer(maxlen=4096)
        self.state_buffer = TimedBuffer(maxlen=4096)
        self.vr_pose_buffer = TimedBuffer(maxlen=2048)
        self.enabled_buffer = TimedBuffer(maxlen=2048)
        self.joystick_y_buffer = TimedBuffer(maxlen=2048)
        self.tactile_left_buffer = TimedBuffer(maxlen=4096)
        self.tactile_right_buffer = TimedBuffer(maxlen=4096)

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
        log_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
        )
        tactile_msg_type = self._get_tactile_msg_type(cfg.tactile_msg_package)

        self.create_subscription(Image, cfg.image_topic, self._image_cb, image_qos)
        self.create_subscription(Image, cfg.wrist_image_topic, self._wrist_image_cb, image_qos)
        self.create_subscription(CameraInfo, cfg.wrist_camera_info_topic, self._wrist_camera_info_cb, image_qos)
        self.create_subscription(String, cfg.command_event_topic, self._command_event_cb, log_qos)
        self.create_subscription(String, cfg.robot_state_topic, self._robot_state_cb, log_qos)
        self.create_subscription(PoseStamped, cfg.raw_vr_target_pose_topic, self._vr_pose_cb, fast_qos)
        self.create_subscription(Bool, cfg.enabled_topic, self._enabled_cb, fast_qos)
        self.create_subscription(Float32, cfg.joystick_y_topic, self._joystick_y_cb, fast_qos)
        self.create_subscription(tactile_msg_type, cfg.tactile_left_topic, self._tactile_left_cb, fast_qos)
        self.create_subscription(tactile_msg_type, cfg.tactile_right_topic, self._tactile_right_cb, fast_qos)

        self.get_logger().info(f"subscribe image: {cfg.image_topic}")
        self.get_logger().info(f"subscribe wrist image: {cfg.wrist_image_topic}")
        self.get_logger().info(f"subscribe wrist camera info: {cfg.wrist_camera_info_topic}")
        self.get_logger().info(f"subscribe command events: {cfg.command_event_topic}")
        self.get_logger().info(f"subscribe robot state: {cfg.robot_state_topic}")
        self.get_logger().info(f"subscribe raw VR pose: {cfg.raw_vr_target_pose_topic}")
        self.get_logger().info(f"subscribe enabled: {cfg.enabled_topic}")
        self.get_logger().info(f"subscribe joystick y: {cfg.joystick_y_topic}")
        self.get_logger().info(f"subscribe tactile msg package: {cfg.tactile_msg_package}")
        self.get_logger().info(f"subscribe tactile left: {cfg.tactile_left_topic}")
        self.get_logger().info(f"subscribe tactile right: {cfg.tactile_right_topic}")

    def _get_tactile_msg_type(self, msg_package: str) -> Any:
        if msg_package == "paxini_tactile":
            return PaxiniTactileFrame
        if msg_package == "tashan_tactile" and TashanTactileFrame is not None:
            return TashanTactileFrame
        raise RuntimeError(
            f"tactile msg package '{msg_package}' is not available. "
            "Source/build the matching tactile package first."
        )

    def _image_cb(self, msg: Image) -> None:
        recv_t = now_monotonic()
        with self._lock:
            self.image_buffer.append(recv_t, msg)

    def _wrist_image_cb(self, msg: Image) -> None:
        recv_t = now_monotonic()
        with self._lock:
            self.wrist_image_buffer.append(recv_t, msg)

    def _wrist_camera_info_cb(self, msg: CameraInfo) -> None:
        recv_t = now_monotonic()
        with self._lock:
            self.wrist_camera_info_buffer.append(recv_t, msg)

    def _command_event_cb(self, msg: String) -> None:
        event = parse_json_str(msg.data)
        # Use pose-send start time as the action timestamp when available.
        t = float(event.get("http", {}).get("t_send_start", event.get("publish_time", now_monotonic())))
        with self._lock:
            self.command_buffer.append(t, event)

    def _robot_state_cb(self, msg: String) -> None:
        event = parse_json_str(msg.data)
        t = float(event.get("t_query_mid", event.get("publish_time", now_monotonic())))
        with self._lock:
            self.state_buffer.append(t, event)

    def _vr_pose_cb(self, msg: PoseStamped) -> None:
        pose7 = np.asarray(
            [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ],
            dtype=np.float32,
        )
        pose7[3:7] = normalize_quat_xyzw(pose7[3:7])
        payload = {
            "pose7": pose7.tolist(),
            "ros_stamp": ros_stamp_to_float_sec(msg.header.stamp),
            "frame_id": msg.header.frame_id,
            "recv_time": now_monotonic(),
        }
        with self._lock:
            self.vr_pose_buffer.append(payload["recv_time"], payload)

    def _enabled_cb(self, msg: Bool) -> None:
        t = now_monotonic()
        with self._lock:
            self.enabled_buffer.append(t, {"enabled": bool(msg.data), "recv_time": t})

    def _joystick_y_cb(self, msg: Float32) -> None:
        t = now_monotonic()
        with self._lock:
            self.joystick_y_buffer.append(t, {"joystick_y": float(msg.data), "recv_time": t})

    def _tactile_left_cb(self, msg: Any) -> None:
        recv_t = now_monotonic()
        payload = tactile_msg_to_record(msg, recv_t, self.cfg.tactile_msg_package)
        with self._lock:
            self.tactile_left_buffer.append(recv_t, payload)

    def _tactile_right_cb(self, msg: Any) -> None:
        recv_t = now_monotonic()
        payload = tactile_msg_to_record(msg, recv_t, self.cfg.tactile_msg_package)
        with self._lock:
            self.tactile_right_buffer.append(recv_t, payload)

    def get_nearest_bundle(self, t_frame: float) -> dict[str, Optional[TimedItem]]:
        with self._lock:
            return {
                "image": self.image_buffer.nearest(t_frame),
                "wrist_image": self.wrist_image_buffer.nearest(t_frame),
                "wrist_camera_info": self.wrist_camera_info_buffer.nearest(t_frame),
                "command": self.command_buffer.nearest(t_frame),
                "state": self.state_buffer.nearest(t_frame),
                "vr_pose": self.vr_pose_buffer.nearest(t_frame),
                "enabled": self.enabled_buffer.nearest(t_frame),
                "joystick_y": self.joystick_y_buffer.nearest(t_frame),
                "tactile_left": self.tactile_left_buffer.nearest(t_frame),
                "tactile_right": self.tactile_right_buffer.nearest(t_frame),
            }

    def has_minimum_topics(self) -> tuple[bool, dict[str, bool]]:
        with self._lock:
            status = {
                "image": self.image_buffer.latest() is not None,
                "wrist_image": self.wrist_image_buffer.latest() is not None,
                "command": self.command_buffer.latest() is not None,
                "state": self.state_buffer.latest() is not None,
                "enabled": self.enabled_buffer.latest() is not None,
                "tactile_left": self.tactile_left_buffer.latest() is not None,
                "tactile_right": self.tactile_right_buffer.latest() is not None,
            }
        return all(status.values()), status


# ============================================================
# Dataset writer
# ============================================================

class DatasetWriter:
    def __init__(self, cfg: RecordConfig, config_path: Path):
        self.cfg = cfg
        self.root = cfg.dataset_root
        self.config_path = Path(config_path).expanduser().resolve()

        self.data_dir = self.root / "data" / "chunk-000"
        self.episode_data_dir = self.root / "data" / "episodes"
        self.master_parquet = self.data_dir / "file-000.parquet"

        self.image_root = self.root / "image" / cfg.image_name
        self.image_index_path = self.image_root / "index.jsonl"
        self.wrist_image_root = self.root / "image" / cfg.wrist_image_name
        self.wrist_image_index_path = self.wrist_image_root / "index.jsonl"
        self.tactile_root = self.root / "tactile"
        self.tactile_left_root = self.tactile_root / "tactile_left"
        self.tactile_right_root = self.tactile_root / "tactile_right"
        self.tactile_left_zarr_path = self.tactile_left_root / "data.zarr"
        self.tactile_right_zarr_path = self.tactile_right_root / "data.zarr"
        self.tactile_left_meta_path = self.tactile_left_root / "meta.jsonl"
        self.tactile_right_meta_path = self.tactile_right_root / "meta.jsonl"

        self.meta_dir = self.root / "meta"
        self.info_path = self.meta_dir / "info.json"
        self.config_copy_path = self.meta_dir / "config.yaml"
        self.episodes_index_path = self.meta_dir / "episodes.jsonl"

        self.raw_dir = self.root / "raw"
        self.vr_log_path = self.raw_dir / "vr_controller_events.jsonl"
        self.command_log_path = self.raw_dir / "command_events.jsonl"
        self.state_log_path = self.raw_dir / "robot_state_events.jsonl"

        self.rows: list[dict[str, Any]] = []
        self.global_index = 0

        self.current_tmp_image_dir: Optional[Path] = None
        self.current_tmp_wrist_image_dir: Optional[Path] = None
        self.current_episode_rows: list[dict[str, Any]] = []
        self.current_image_records: list[dict[str, Any]] = []
        self.current_wrist_image_records: list[dict[str, Any]] = []
        self.current_tactile_left_records: list[dict[str, Any]] = []
        self.current_tactile_right_records: list[dict[str, Any]] = []

    def prepare(self) -> None:
        try:
            import zarr  # noqa: F401
        except ImportError as e:
            raise RuntimeError("zarr is required to save tactile data. Install python package 'zarr'.") from e

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.episode_data_dir.mkdir(parents=True, exist_ok=True)
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.wrist_image_root.mkdir(parents=True, exist_ok=True)
        self.tactile_left_root.mkdir(parents=True, exist_ok=True)
        self.tactile_right_root.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        if self.config_path.exists():
            shutil.copy2(self.config_path, self.config_copy_path)

        info = {
            "dataset_type": "fr3_zed_d405_rgb_hilserl_raw_sidecar",
            "description": "Raw ZED front RGB PNG + D405 wrist RGB PNG images + HIL-SERL sent command + franka_server robot state. Convert this raw dataset to SDP HDF5 before training.",
            "fps": self.cfg.fps,
            "task_description": self.cfg.task_description,
            "task_index": self.cfg.task_index,
            "tactile_msg_package": self.cfg.tactile_msg_package,
            "action_names": ACTION_NAMES,
            "action_dim": 8,
            "storage": {
                "master_parquet": "data/chunk-000/file-000.parquet",
                "per_episode_parquet_dir": "data/episodes",
                "image_dir": f"image/{self.cfg.image_name}",
                "wrist_image_dir": f"image/{self.cfg.wrist_image_name}",
                "tactile_left_zarr": "tactile/tactile_left/data.zarr",
                "tactile_left_meta": "tactile/tactile_left/meta.jsonl",
                "tactile_right_zarr": "tactile/tactile_right/data.zarr",
                "tactile_right_meta": "tactile/tactile_right/meta.jsonl",
                "raw_vr_controller_events": "raw/vr_controller_events.jsonl",
                "raw_command_events": "raw/command_events.jsonl",
                "raw_robot_state_events": "raw/robot_state_events.jsonl",
            },
            "features": {
                "action.sent_action8": "float32[8], final command actually sent through HTTP /pose + gripper target",
                "robot.ee_pose": "best-effort parsed from /getstate, if available",
                "image.path": "relative path to ZED front RGB PNG sidecar",
                "wrist_image.path": "relative path to D405 wrist RGB PNG sidecar",
                "tactile.*.zarr_index": "row index into tactile/tactile_left|right/data.zarr",
                "tactile.*.data": self._tactile_data_feature_description(),
                "timestamp": "seconds from episode start on fixed-FPS frame grid",
                "wall_time": "monotonic time at target frame",
                "sync.*": "diagnostics for nearest-neighbor alignment",
            },
        }
        with open(self.info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    def _tactile_data_feature_description(self) -> str:
        if self.cfg.tactile_msg_package == "paxini_tactile":
            return "uint8[] raw Paxini tactile frame in zarr; [0:3]=force, [3:234]=distributed force"
        if self.cfg.tactile_msg_package == "tashan_tactile":
            return "float32[] flattened Tashan tactile frame in zarr; layout lengths are stored in tactile metadata"
        return "tactile frame in zarr"

    def start_episode(self, episode_index: int) -> None:
        self.current_episode_rows = []
        self.current_image_records = []
        self.current_wrist_image_records = []
        self.current_tactile_left_records = []
        self.current_tactile_right_records = []
        tmp = self.image_root / f".tmp_episode_{episode_index:06d}"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True, exist_ok=True)
        self.current_tmp_image_dir = tmp
        wrist_tmp = self.wrist_image_root / f".tmp_episode_{episode_index:06d}"
        if wrist_tmp.exists():
            shutil.rmtree(wrist_tmp)
        wrist_tmp.mkdir(parents=True, exist_ok=True)
        self.current_tmp_wrist_image_dir = wrist_tmp

    def discard_episode(self) -> None:
        if self.current_tmp_image_dir is not None and self.current_tmp_image_dir.exists():
            shutil.rmtree(self.current_tmp_image_dir)
        if self.current_tmp_wrist_image_dir is not None and self.current_tmp_wrist_image_dir.exists():
            shutil.rmtree(self.current_tmp_wrist_image_dir)
        self.current_tmp_image_dir = None
        self.current_tmp_wrist_image_dir = None
        self.current_episode_rows = []
        self.current_image_records = []
        self.current_wrist_image_records = []
        self.current_tactile_left_records = []
        self.current_tactile_right_records = []

    def _final_image_rel_path(self, episode_index: int, frame_index: int) -> Path:
        return Path("image") / self.cfg.image_name / f"episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"

    def _final_wrist_image_rel_path(self, episode_index: int, frame_index: int) -> Path:
        return Path("image") / self.cfg.wrist_image_name / f"episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"

    def _save_image_to_dir(
        self,
        *,
        episode_index: int,
        frame_index: int,
        msg: Image,
        tmp_dir: Path,
        rel_path: Path,
        topic: str,
        records: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:

        bgr = image_msg_to_bgr8(msg)
        out_path = tmp_dir / f"frame_{frame_index:06d}.png"
        ok = cv2.imwrite(str(out_path), bgr)
        if not ok:
            raise RuntimeError(f"failed to write PNG: {out_path}")

        stamp = msg.header.stamp
        frame_id = str(msg.header.frame_id)
        rec = {
            "episode_index": int(episode_index),
            "record_frame_index": int(frame_index),
            "path": str(rel_path),
            "tmp_path": str(out_path.relative_to(self.root)),
            "height": int(msg.height),
            "width": int(msg.width),
            "encoding": str(msg.encoding),
            "step": int(msg.step),
            "ros_stamp_sec": int(stamp.sec),
            "ros_stamp_nanosec": int(stamp.nanosec),
            "ros_stamp_float": ros_stamp_to_float_sec(stamp),
            "frame_id": frame_id,
            "topic": topic,
        }
        records.append(rec)
        return str(rel_path), rec

    def save_image(self, episode_index: int, frame_index: int, msg: Image) -> tuple[str, dict[str, Any]]:
        if self.current_tmp_image_dir is None:
            raise RuntimeError("start_episode must be called before save_image")
        return self._save_image_to_dir(
            episode_index=episode_index,
            frame_index=frame_index,
            msg=msg,
            tmp_dir=self.current_tmp_image_dir,
            rel_path=self._final_image_rel_path(episode_index, frame_index),
            topic=self.cfg.image_topic,
            records=self.current_image_records,
        )

    def save_wrist_image(self, episode_index: int, frame_index: int, msg: Image) -> tuple[str, dict[str, Any]]:
        if self.current_tmp_wrist_image_dir is None:
            raise RuntimeError("start_episode must be called before save_wrist_image")
        return self._save_image_to_dir(
            episode_index=episode_index,
            frame_index=frame_index,
            msg=msg,
            tmp_dir=self.current_tmp_wrist_image_dir,
            rel_path=self._final_wrist_image_rel_path(episode_index, frame_index),
            topic=self.cfg.wrist_image_topic,
            records=self.current_wrist_image_records,
        )

    def append_debug_jsonl(self, command_event: dict[str, Any], state_event: dict[str, Any]) -> None:
        if not self.cfg.save_debug_jsonl:
            return
        with open(self.command_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(command_event, ensure_ascii=False) + "\n")
        with open(self.state_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(state_event, ensure_ascii=False) + "\n")

    def append_vr_jsonl(self, vr_event: dict[str, Any]) -> None:
        with open(self.vr_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(vr_event, ensure_ascii=False) + "\n")

    def append_row(self, row: dict[str, Any]) -> None:
        self.current_episode_rows.append(row)

    def append_tactile_pair(
        self,
        *,
        episode_index: int,
        frame_index: int,
        timestamp: float,
        wall_time: float,
        left_payload: dict[str, Any],
        left_dt: float,
        right_payload: dict[str, Any],
        right_dt: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        left_rec = self._make_tactile_record(
            episode_index=episode_index,
            frame_index=frame_index,
            timestamp=timestamp,
            wall_time=wall_time,
            payload=left_payload,
            sync_dt=left_dt,
            zarr_path=Path("tactile") / "tactile_left" / "data.zarr",
        )
        right_rec = self._make_tactile_record(
            episode_index=episode_index,
            frame_index=frame_index,
            timestamp=timestamp,
            wall_time=wall_time,
            payload=right_payload,
            sync_dt=right_dt,
            zarr_path=Path("tactile") / "tactile_right" / "data.zarr",
        )
        self.current_tactile_left_records.append(left_rec)
        self.current_tactile_right_records.append(right_rec)
        return left_rec, right_rec

    def _make_tactile_record(
        self,
        *,
        episode_index: int,
        frame_index: int,
        timestamp: float,
        wall_time: float,
        payload: dict[str, Any],
        sync_dt: float,
        zarr_path: Path,
    ) -> dict[str, Any]:
        return {
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
            "zarr_path": str(zarr_path),
            "zarr_index": None,
            "timestamp": float(timestamp),
            "wall_time": float(wall_time),
            "sensor_id": str(payload["sensor_id"]),
            "sensor_index": int(payload["sensor_index"]),
            "frame_idx": int(payload["frame_idx"]),
            "capture_time": float(payload["capture_time"]),
            "repeated": bool(payload["repeated"]),
            "ros_stamp_sec": int(payload["ros_stamp_sec"]),
            "ros_stamp_nanosec": int(payload["ros_stamp_nanosec"]),
            "ros_stamp_float": float(payload["ros_stamp_float"]),
            "frame_id": str(payload["frame_id"]),
            "recv_time": float(payload["recv_time"]),
            "sync_dt": float(sync_dt),
            "data_len": int(payload["data_len"]),
            "data_dtype": str(payload["data_dtype"]),
            "source_dtype": str(payload["source_dtype"]),
            "msg_package": str(payload["msg_package"]),
            "layout": dict(payload["layout"]),
            "data": np.asarray(payload["data"], dtype=np.dtype(str(payload["data_dtype"]))).copy(),
        }

    def _append_tactile_records_to_zarr(
        self,
        *,
        records: list[dict[str, Any]],
        zarr_path: Path,
        meta_path: Path,
    ) -> None:
        if not records:
            return
        try:
            import zarr
        except ImportError as e:
            raise RuntimeError("zarr is required to save tactile data. Install python package 'zarr'.") from e

        data_dtypes = {str(rec["data_dtype"]) for rec in records}
        if len(data_dtypes) != 1:
            raise RuntimeError(f"mixed tactile data dtypes in one batch: {sorted(data_dtypes)}")
        data_dtype = np.dtype(next(iter(data_dtypes)))
        data = np.stack([np.asarray(rec["data"], dtype=data_dtype) for rec in records], axis=0)
        if data.ndim != 2:
            raise RuntimeError(f"invalid tactile batch shape: {data.shape}")
        frame_len = int(data.shape[1])

        if zarr_path.exists():
            arr = zarr.open(str(zarr_path), mode="a")
            if int(arr.shape[1]) != frame_len:
                raise RuntimeError(f"tactile zarr frame length mismatch: existing={arr.shape[1]} new={frame_len}")
            if np.dtype(arr.dtype) != data_dtype:
                raise RuntimeError(f"tactile zarr dtype mismatch: existing={arr.dtype} new={data_dtype}")
            start = int(arr.shape[0])
            arr.resize((start + data.shape[0], frame_len))
            arr[start:start + data.shape[0], :] = data
        else:
            arr = zarr.open(
                str(zarr_path),
                mode="w",
                shape=(data.shape[0], frame_len),
                chunks=(min(1024, data.shape[0]), frame_len),
                dtype=data_dtype,
            )
            start = 0
            arr[:, :] = data

        with open(meta_path, "a", encoding="utf-8") as f:
            for offset, rec in enumerate(records):
                rec["zarr_index"] = int(start + offset)
                meta = dict(rec)
                meta.pop("data", None)
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def commit_episode(self, episode_index: int) -> None:
        if self.current_tmp_image_dir is None or self.current_tmp_wrist_image_dir is None:
            return
        if not self.current_episode_rows:
            self.discard_episode()
            return

        final_image_dir = self.image_root / f"episode_{episode_index:06d}"
        if final_image_dir.exists():
            shutil.rmtree(final_image_dir)
        self.current_tmp_image_dir.rename(final_image_dir)
        final_wrist_image_dir = self.wrist_image_root / f"episode_{episode_index:06d}"
        if final_wrist_image_dir.exists():
            shutil.rmtree(final_wrist_image_dir)
        self.current_tmp_wrist_image_dir.rename(final_wrist_image_dir)

        if len(self.current_tactile_left_records) != len(self.current_episode_rows):
            raise RuntimeError(
                f"left tactile record count mismatch: "
                f"{len(self.current_tactile_left_records)} != {len(self.current_episode_rows)}"
            )
        if len(self.current_tactile_right_records) != len(self.current_episode_rows):
            raise RuntimeError(
                f"right tactile record count mismatch: "
                f"{len(self.current_tactile_right_records)} != {len(self.current_episode_rows)}"
            )
        self._append_tactile_records_to_zarr(
            records=self.current_tactile_left_records,
            zarr_path=self.tactile_left_zarr_path,
            meta_path=self.tactile_left_meta_path,
        )
        self._append_tactile_records_to_zarr(
            records=self.current_tactile_right_records,
            zarr_path=self.tactile_right_zarr_path,
            meta_path=self.tactile_right_meta_path,
        )

        committed = []
        for row, left_rec, right_rec in zip(
            self.current_episode_rows,
            self.current_tactile_left_records,
            self.current_tactile_right_records,
        ):
            row = dict(row)
            row["tactile.left.zarr_index"] = int(left_rec["zarr_index"])
            row["tactile.right.zarr_index"] = int(right_rec["zarr_index"])
            row["index"] = int(self.global_index)
            self.global_index += 1
            committed.append(row)

        ep_path = self.episode_data_dir / f"episode_{episode_index:06d}.parquet"
        pd.DataFrame(committed).to_parquet(ep_path, index=False)

        with open(self.image_index_path, "a", encoding="utf-8") as f:
            for rec in self.current_image_records:
                rec = dict(rec)
                rec.pop("tmp_path", None)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(self.wrist_image_index_path, "a", encoding="utf-8") as f:
            for rec in self.current_wrist_image_records:
                rec = dict(rec)
                rec.pop("tmp_path", None)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        episode_info = {
            "episode_index": int(episode_index),
            "num_frames": int(len(committed)),
            "data_path": str(Path("data") / "episodes" / f"episode_{episode_index:06d}.parquet"),
            "image_dir": str(Path("image") / self.cfg.image_name / f"episode_{episode_index:06d}"),
            "wrist_image_dir": str(Path("image") / self.cfg.wrist_image_name / f"episode_{episode_index:06d}"),
            "tactile_left_zarr": str(Path("tactile") / "tactile_left" / "data.zarr"),
            "tactile_right_zarr": str(Path("tactile") / "tactile_right" / "data.zarr"),
            "task_index": int(self.cfg.task_index),
            "task_description": self.cfg.task_description,
        }
        with open(self.episodes_index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode_info, ensure_ascii=False) + "\n")

        self.rows.extend(committed)
        print(f"[SAVE] episode {episode_index}: {len(committed)} frames")
        print(f"[SAVE] episode parquet: {ep_path}")

        self.current_tmp_image_dir = None
        self.current_tmp_wrist_image_dir = None
        self.current_episode_rows = []
        self.current_image_records = []
        self.current_wrist_image_records = []
        self.current_tactile_left_records = []
        self.current_tactile_right_records = []

    def flush_master(self) -> None:
        if not self.rows:
            print("[WARN] no committed rows to write master parquet")
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_parquet(self.master_parquet, index=False)
        print(f"[SAVE] master parquet: {self.master_parquet}")
        print(f"[SAVE] total rows: {len(self.rows)}")


# ============================================================
# Flatten helpers
# ============================================================

def _as_float_list(x: Any, max_len: Optional[int] = None) -> Optional[list[float]]:
    if x is None:
        return None
    try:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if max_len is not None:
            arr = arr[:max_len]
        return [float(v) for v in arr.tolist()]
    except Exception:
        return None


def extract_state_fields(state_event: dict[str, Any]) -> dict[str, Any]:
    raw_state = state_event.get("state")
    if not isinstance(raw_state, dict):
        return {
            "robot.raw_state_json": json.dumps(raw_state, ensure_ascii=False),
            "robot.ee_pose": None,
            "robot.gripper_pos": None,
            "robot.q": None,
            "robot.dq": None,
            "robot.force": None,
            "robot.torque": None,
        }

    # Common franka_server keys include pose, vel, force, torque, q, dq, jacobian, gripper_pos.
    ee_pose = raw_state.get("pose", raw_state.get("ee_pose", raw_state.get("pos")))
    gripper_pos = raw_state.get("gripper_pos", raw_state.get("gripper", raw_state.get("gripper_width")))

    return {
        "robot.raw_state_json": json.dumps(raw_state, ensure_ascii=False),
        "robot.ee_pose": _as_float_list(ee_pose, 7),
        "robot.gripper_pos": None if gripper_pos is None else float(np.asarray(gripper_pos).reshape(-1)[0]),
        "robot.q": _as_float_list(raw_state.get("q"), 7),
        "robot.dq": _as_float_list(raw_state.get("dq"), 7),
        "robot.vel": _as_float_list(raw_state.get("vel"), None),
        "robot.force": _as_float_list(raw_state.get("force"), None),
        "robot.torque": _as_float_list(raw_state.get("torque"), None),
    }


def valid_pair(name: str, item: Optional[TimedItem], t_frame: float, max_dt: float) -> tuple[bool, Optional[float], str]:
    if item is None:
        return False, None, f"missing_{name}"
    dt = float(item.t - t_frame)
    if abs(dt) > max_dt:
        return False, dt, f"stale_{name}_dt_{dt:.3f}"
    return True, dt, "ok"


def camera_info_to_record(item: Optional[TimedItem], t_frame: float) -> dict[str, Any]:
    if item is None:
        return {
            "wrist_camera_info.available": False,
            "wrist_camera_info.dt": None,
            "wrist_camera_info.ros_stamp_float": None,
            "wrist_camera_info.frame_id": None,
            "wrist_camera_info.height": None,
            "wrist_camera_info.width": None,
            "wrist_camera_info.k": None,
            "wrist_camera_info.d": None,
            "wrist_camera_info.distortion_model": None,
        }

    msg = item.payload
    return {
        "wrist_camera_info.available": True,
        "wrist_camera_info.dt": float(item.t - t_frame),
        "wrist_camera_info.ros_stamp_float": ros_stamp_to_float_sec(msg.header.stamp),
        "wrist_camera_info.frame_id": str(msg.header.frame_id),
        "wrist_camera_info.height": int(msg.height),
        "wrist_camera_info.width": int(msg.width),
        "wrist_camera_info.k": [float(v) for v in msg.k],
        "wrist_camera_info.d": [float(v) for v in msg.d],
        "wrist_camera_info.distortion_model": str(msg.distortion_model),
    }


def tactile_row_fields(prefix: str, rec: dict[str, Any]) -> dict[str, Any]:
    fields = {
        f"{prefix}.zarr_path": str(rec["zarr_path"]),
        f"{prefix}.zarr_index": None,
        f"{prefix}.msg_package": str(rec["msg_package"]),
        f"{prefix}.sensor_id": str(rec["sensor_id"]),
        f"{prefix}.sensor_index": int(rec["sensor_index"]),
        f"{prefix}.frame_idx": int(rec["frame_idx"]),
        f"{prefix}.capture_time": float(rec["capture_time"]),
        f"{prefix}.repeated": bool(rec["repeated"]),
        f"{prefix}.ros_stamp_sec": int(rec["ros_stamp_sec"]),
        f"{prefix}.ros_stamp_nanosec": int(rec["ros_stamp_nanosec"]),
        f"{prefix}.ros_stamp_float": float(rec["ros_stamp_float"]),
        f"{prefix}.frame_id": str(rec["frame_id"]),
        f"{prefix}.recv_time": float(rec["recv_time"]),
        f"{prefix}.sync_dt": float(rec["sync_dt"]),
        f"{prefix}.data_len": int(rec["data_len"]),
        f"{prefix}.data_dtype": str(rec["data_dtype"]),
        f"{prefix}.source_dtype": str(rec["source_dtype"]),
    }
    for key, value in rec.get("layout", {}).items():
        fields[f"{prefix}.{key}"] = int(value)
    return fields


class ManualEpisodeController:
    def __init__(self):
        self.finish = False
        self.discard = False
        self.quit = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, episode_index: int) -> None:
        self.finish = False
        self.discard = False
        self.quit = False
        self._stop.clear()

        def _loop() -> None:
            print(
                f"\n====== [RECORDING] Episode {episode_index + 1} is running ======\n"
                "Press Enter to FINISH and SAVE this episode.\n"
                "Type r + Enter to DISCARD and RE-RECORD this episode.\n"
                "Type q + Enter to FINISH this episode and QUIT after saving.\n"
            )
            while not self._stop.is_set():
                try:
                    line = input().strip().lower()
                except EOFError:
                    time.sleep(0.1)
                    continue
                if line == "":
                    self.finish = True
                    break
                if line in ("r", "redo", "retry"):
                    self.discard = True
                    self.finish = True
                    break
                if line in ("q", "quit", "stop"):
                    self.quit = True
                    self.finish = True
                    break
                print("Unknown command. Use Enter=save, r=discard/re-record, q=save-and-quit.")

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# ============================================================
# Recording loop
# ============================================================

def wait_for_topics_ready(node: RawCollectionNode, timeout_sec: float = 10.0) -> None:
    print(
        f"[WARMUP] waiting for front image + wrist image + command + state + "
        f"enabled + tactile left/right, timeout={timeout_sec:.1f}s"
    )
    start = now_monotonic()
    last_log = 0.0
    while now_monotonic() - start < timeout_sec:
        ok, status = node.has_minimum_topics()
        if ok:
            print("[WARMUP] ready.")
            return
        now = now_monotonic()
        if now - last_log >= 1.0:
            missing = [name for name, ready in status.items() if not ready]
            print(f"[WARMUP] missing: {', '.join(missing)} status={status}", flush=True)
            last_log = now
        time.sleep(0.05)
    _, status = node.has_minimum_topics()
    raise RuntimeError(f"Warmup timeout. topic status={status}")


def record_episode(cfg: RecordConfig, node: RawCollectionNode, writer: DatasetWriter, episode_index: int) -> tuple[bool, bool]:
    input(
        f"\n====== [READY] Prepare task and press Enter to START episode "
        f"{episode_index + 1}/{cfg.num_episodes} ======"
    )

    writer.start_episode(episode_index)
    controller = ManualEpisodeController()
    controller.start(episode_index)

    period = 1.0 / cfg.fps
    ep_start = now_monotonic()
    next_t = ep_start
    candidate_count = 0
    saved_count = 0
    skip_count = 0

    print(f"[RECORD] episode={episode_index}, fps={cfg.fps}, max_episode_sec={cfg.max_episode_sec}")

    try:
        while True:
            now = now_monotonic()
            if controller.finish:
                break
            if now - ep_start > cfg.max_episode_sec:
                print("[RECORD] max_episode_sec reached, finishing episode.")
                break
            if now < next_t:
                time.sleep(min(0.002, next_t - now))
                continue

            t_frame = next_t
            timestamp = float(t_frame - ep_start)
            next_t += period
            candidate_count += 1

            bundle = node.get_nearest_bundle(t_frame)
            image_item = bundle["image"]
            wrist_image_item = bundle["wrist_image"]
            wrist_camera_info_item = bundle["wrist_camera_info"]
            command_item = bundle["command"]
            state_item = bundle["state"]
            vr_pose_item = bundle["vr_pose"]
            enabled_item = bundle["enabled"]
            joystick_y_item = bundle["joystick_y"]
            tactile_left_item = bundle["tactile_left"]
            tactile_right_item = bundle["tactile_right"]

            ok_image, dt_image, reason_image = valid_pair("image", image_item, t_frame, cfg.max_image_dt_sec)
            ok_wrist_image, dt_wrist_image, reason_wrist_image = valid_pair("wrist_image", wrist_image_item, t_frame, cfg.max_wrist_image_dt_sec)
            ok_action, dt_action, reason_action = valid_pair("action", command_item, t_frame, cfg.max_action_dt_sec)
            ok_state, dt_state, reason_state = valid_pair("state", state_item, t_frame, cfg.max_state_dt_sec)
            ok_vr, dt_vr, reason_vr = valid_pair("vr", vr_pose_item, t_frame, cfg.max_vr_dt_sec)
            ok_tactile_left, dt_tactile_left, reason_tactile_left = valid_pair(
                "tactile_left", tactile_left_item, t_frame, cfg.max_tactile_dt_sec
            )
            ok_tactile_right, dt_tactile_right, reason_tactile_right = valid_pair(
                "tactile_right", tactile_right_item, t_frame, cfg.max_tactile_dt_sec
            )

            enabled_val = None
            if enabled_item is not None:
                enabled_val = bool(enabled_item.payload.get("enabled", False))
            if cfg.wait_for_motion_enable_to_record and not enabled_val:
                skip_count += 1
                if cfg.debug and skip_count % 20 == 0:
                    print(f"[SKIP] enabled false or missing. saved={saved_count} skip={skip_count}")
                continue

            if not (ok_image and ok_wrist_image and ok_action and ok_state and ok_tactile_left and ok_tactile_right):
                skip_count += 1
                if cfg.debug and skip_count % 10 == 0:
                    print(
                        f"[SKIP] saved={saved_count} skip={skip_count} "
                        f"image={reason_image} wrist_image={reason_wrist_image} "
                        f"action={reason_action} state={reason_state} "
                        f"tactile_left={reason_tactile_left} tactile_right={reason_tactile_right}"
                    )
                continue

            tactile_left_payload = tactile_left_item.payload
            tactile_right_payload = tactile_right_item.payload
            if tactile_left_payload["data_len"] != tactile_right_payload["data_len"]:
                skip_count += 1
                print(
                    "[WARN] left/right tactile data length mismatch: "
                    f"{tactile_left_payload['data_len']} != {tactile_right_payload['data_len']}"
                )
                continue

            command_event = command_item.payload
            state_event = state_item.payload
            action8 = command_event.get("action8")
            if action8 is None:
                skip_count += 1
                if cfg.debug:
                    print("[SKIP] command_event has no action8")
                continue
            action8 = np.asarray(action8, dtype=np.float32).reshape(-1)
            if action8.shape[0] != 8:
                skip_count += 1
                if cfg.debug:
                    print(f"[SKIP] action8 has wrong shape: {action8.shape}")
                continue
            action8[3:7] = normalize_quat_xyzw(action8[3:7])

            try:
                image_rel_path, image_rec = writer.save_image(episode_index, saved_count, image_item.payload)
                wrist_image_rel_path, wrist_image_rec = writer.save_wrist_image(
                    episode_index, saved_count, wrist_image_item.payload
                )
            except Exception as e:
                skip_count += 1
                print(f"[WARN] image save failed: {repr(e)}")
                continue

            vr_payload = vr_pose_item.payload if vr_pose_item is not None else {}
            joystick_y_payload = joystick_y_item.payload if joystick_y_item is not None else {}
            state_fields = extract_state_fields(state_event)
            wrist_camera_info_fields = camera_info_to_record(wrist_camera_info_item, t_frame)
            tactile_left_rec, tactile_right_rec = writer.append_tactile_pair(
                episode_index=episode_index,
                frame_index=saved_count,
                timestamp=timestamp,
                wall_time=t_frame,
                left_payload=tactile_left_payload,
                left_dt=float(dt_tactile_left),
                right_payload=tactile_right_payload,
                right_dt=float(dt_tactile_right),
            )

            row: dict[str, Any] = {
                "timestamp": np.float32(timestamp),
                "wall_time": float(t_frame),
                "frame_index": int(saved_count),
                "candidate_index": int(candidate_count - 1),
                "episode_index": int(episode_index),
                "index": None,
                "task_index": int(cfg.task_index),

                "action": action8.astype(np.float32).tolist(),
                "action.sent_action8": action8.astype(np.float32).tolist(),
                "action.pose7": action8[:7].astype(np.float32).tolist(),
                "action.target_gripper": float(action8[7]),
                "action.http_ok": bool(command_event.get("http", {}).get("ok", False)),
                "action.http_latency_ms": float(command_event.get("http", {}).get("latency_ms", np.nan)),
                "action.t_send_start": float(command_event.get("http", {}).get("t_send_start", np.nan)),
                "action.t_send_end": float(command_event.get("http", {}).get("t_send_end", np.nan)),
                "action.workspace_clipped": bool(command_event.get("workspace_clipped", False)),

                "image.path": image_rel_path,
                "image.height": int(image_rec["height"]),
                "image.width": int(image_rec["width"]),
                "image.encoding": str(image_rec["encoding"]),
                "image.step": int(image_rec["step"]),
                "image.ros_stamp_sec": int(image_rec["ros_stamp_sec"]),
                "image.ros_stamp_nanosec": int(image_rec["ros_stamp_nanosec"]),
                "image.ros_stamp_float": float(image_rec["ros_stamp_float"]),
                "image.frame_id": str(image_rec["frame_id"]),

                "wrist_image.path": wrist_image_rel_path,
                "wrist_image.height": int(wrist_image_rec["height"]),
                "wrist_image.width": int(wrist_image_rec["width"]),
                "wrist_image.encoding": str(wrist_image_rec["encoding"]),
                "wrist_image.step": int(wrist_image_rec["step"]),
                "wrist_image.ros_stamp_sec": int(wrist_image_rec["ros_stamp_sec"]),
                "wrist_image.ros_stamp_nanosec": int(wrist_image_rec["ros_stamp_nanosec"]),
                "wrist_image.ros_stamp_float": float(wrist_image_rec["ros_stamp_float"]),
                "wrist_image.frame_id": str(wrist_image_rec["frame_id"]),

                "robot.state_ok": bool(state_event.get("ok", False)),
                "robot.t_query_start": float(state_event.get("t_query_start", np.nan)),
                "robot.t_query_end": float(state_event.get("t_query_end", np.nan)),
                "robot.t_query_mid": float(state_event.get("t_query_mid", np.nan)),
                "robot.latency_ms": float(state_event.get("latency_ms", np.nan)),

                "vr.enabled": None if enabled_val is None else bool(enabled_val),
                "vr.joystick_y": None if not joystick_y_payload else float(joystick_y_payload.get("joystick_y", np.nan)),
                "vr.raw_target_pose7": vr_payload.get("pose7"),
                "vr.raw_target_pose_ros_stamp": vr_payload.get("ros_stamp"),

                "sync.valid": True,
                "sync.dt_image": None if dt_image is None else float(dt_image),
                "sync.dt_wrist_image": None if dt_wrist_image is None else float(dt_wrist_image),
                "sync.dt_action": None if dt_action is None else float(dt_action),
                "sync.dt_state": None if dt_state is None else float(dt_state),
                "sync.dt_vr": None if not ok_vr else float(dt_vr),
                "sync.reason_vr": reason_vr,
                "sync.dt_tactile_left": float(dt_tactile_left),
                "sync.dt_tactile_right": float(dt_tactile_right),
            }
            row.update(state_fields)
            row.update(wrist_camera_info_fields)
            row.update(tactile_row_fields("tactile.left", tactile_left_rec))
            row.update(tactile_row_fields("tactile.right", tactile_right_rec))

            vr_event: dict[str, Any] = {
                "episode_index": int(episode_index),
                "frame_index": int(saved_count),
                "candidate_index": int(candidate_count - 1),
                "timestamp": float(timestamp),
                "wall_time": float(t_frame),
                "image_path": image_rel_path,
                "wrist_image_path": wrist_image_rel_path,
                "enabled": None if enabled_val is None else bool(enabled_val),
                "raw_target_pose7": vr_payload.get("pose7"),
                "raw_target_pose_ros_stamp": vr_payload.get("ros_stamp"),
                "raw_target_pose_frame_id": vr_payload.get("frame_id"),
                "joystick_y": None if not joystick_y_payload else float(joystick_y_payload.get("joystick_y", np.nan)),
                "sent_action8": action8.astype(np.float32).tolist(),
                "sent_pose7": action8[:7].astype(np.float32).tolist(),
                "target_gripper": float(action8[7]),
                "sync_dt_vr": None if not ok_vr else float(dt_vr),
                "sync_reason_vr": reason_vr,
            }

            writer.append_row(row)
            writer.append_vr_jsonl(vr_event)
            writer.append_debug_jsonl(command_event, state_event)
            saved_count += 1

            if cfg.debug and saved_count % 10 == 0:
                print(
                    f"[RECORD] ep={episode_index} saved={saved_count} skip={skip_count} "
                    f"front={image_rec['width']}x{image_rec['height']} dt_image={dt_image:+.3f} "
                    f"wrist={wrist_image_rec['width']}x{wrist_image_rec['height']} "
                    f"dt_wrist={dt_wrist_image:+.3f} "
                    f"dt_action={dt_action:+.3f} dt_state={dt_state:+.3f} "
                    f"dt_tactile_left={dt_tactile_left:+.3f} dt_tactile_right={dt_tactile_right:+.3f}"
                )

    finally:
        controller.stop()

    if controller.discard:
        print(f"[DISCARD] episode {episode_index}")
        writer.discard_episode()
        return False, False

    if saved_count == 0:
        print(f"[WARN] episode {episode_index} has zero frames. Discarding.")
        writer.discard_episode()
        return False, controller.quit

    writer.commit_episode(episode_index)
    if cfg.save_master_parquet_every_episode:
        writer.flush_master()
    print(f"[DONE] episode {episode_index} saved={saved_count} skipped={skip_count}")
    return True, controller.quit


def run_recorder(expected_tactile_msg_package: Optional[str] = None, recorder_label: str = "tactile") -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)
    if expected_tactile_msg_package is not None and cfg.tactile_msg_package != expected_tactile_msg_package:
        raise RuntimeError(
            f"{recorder_label} recorder requires tactile.msg_package={expected_tactile_msg_package}, "
            f"but config has {cfg.tactile_msg_package}: {cfg_path}"
        )
    cfg.dataset_root.mkdir(parents=True, exist_ok=True)

    print(f"====== [START] Raw HIL-SERL ZED {recorder_label} recorder ======")
    print(f"[DATASET] root: {cfg.dataset_root}")
    print(f"[TACTILE] msg_package: {cfg.tactile_msg_package}")
    print(f"[TACTILE] left: {cfg.tactile_left_topic}")
    print(f"[TACTILE] right: {cfg.tactile_right_topic}")
    print(f"[FPS] {cfg.fps}")
    print(f"[IMAGE] {cfg.image_topic}")
    print(f"[COMMAND] {cfg.command_event_topic}")
    print(f"[STATE] {cfg.robot_state_topic}")

    writer = DatasetWriter(cfg, config_path=cfg_path)
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

    print(f"====== [END] {recorder_label} recording finished ======")


def main() -> None:
    run_recorder()


if __name__ == "__main__":
    main()
