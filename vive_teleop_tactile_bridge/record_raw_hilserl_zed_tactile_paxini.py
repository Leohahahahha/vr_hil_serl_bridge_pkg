#!/usr/bin/env python3
"""
原始数据采集和保存脚本
数据包括

全局场景相机zed2i的彩色图像

机械臂腕部相机D405的彩色图像

训练动作命令
    action = 相对当前机械臂状态的目标增量 + 目标单指绝对位置（m）
        [
        target_x - current_x,
        target_y - current_y,
        target_z - current_z,
        delta_rotvec_x,
        delta_rotvec_y,
        delta_rotvec_z,
        target_gripper_width / 2
        ]
    action.sent_action8 仍保存实际发给机器人的原始绝对命令
    action.pose7 = [target_x, target_y, target_z, target_qx, target_qy, target_qz, target_qw]
    action.target_gripper_width
    action.target_gripper_finger_position

训练本体感知状态
    observation.state = 当前末端 xyz + 当前姿态 rotvec + 当前单指绝对位置（m）
        [
        current_x,
        current_y,
        current_z,
        current_rotvec_x,
        current_rotvec_y,
        current_rotvec_z,
        current_gripper_width / 2
        ]

原始机械臂本体状态（位置、姿态、关节状态等）和夹爪宽度
    robot.ee_pose：7维数据
    robot.gripper_pos：服务端原始归一化夹爪位置
    robot.gripper_width：实际总开口宽度（m）
    robot.gripper_finger_position：单指绝对位置（m）
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
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Deque

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
#from paxini_tactile.msg import TactileFrame as PaxiniTactileFrame
try:
    from paxini_tactile.msg import TactileFrame as PaxiniTactileFrame
except ImportError:
    PaxiniTactileFrame = None
    
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float32, String

try:
    from .action_space import (
        ACTION_NAMES,
        STATE_NAMES,
        normalize_quat_xyzw,
    )
    from .observation_builder import (
        ObservationSyncThresholds,
        build_observation_frame,
        camera_info_to_record as builder_camera_info_to_record,
        tactile_row_fields as builder_tactile_row_fields,
    )
    from .raw_dataset_writer import PAXINI_RAW_WRITER_PROFILE, RawDatasetWriter
    from .stream_sampling import (
        candidate_continuity_report,
        has_usable_action_label,
        nearest_time_ordered,
        ros_message_stamp_ns,
        skip_reason_key,
    )
except ImportError:
    from action_space import (  # type: ignore
        ACTION_NAMES,
        STATE_NAMES,
        normalize_quat_xyzw,
    )
    from observation_builder import (  # type: ignore
        ObservationSyncThresholds,
        build_observation_frame,
        camera_info_to_record as builder_camera_info_to_record,
        tactile_row_fields as builder_tactile_row_fields,
    )
    from raw_dataset_writer import PAXINI_RAW_WRITER_PROFILE, RawDatasetWriter  # type: ignore
    from stream_sampling import (  # type: ignore
        candidate_continuity_report,
        has_usable_action_label,
        nearest_time_ordered,
        ros_message_stamp_ns,
        skip_reason_key,
    )


TACTILE_FRAME_BYTES = 234
TACTILE_FORCE_BYTES = 3


def now_monotonic() -> float:
    return float(time.monotonic())


def ros_header_stamp_to_monotonic(
    stamp: Any,
    *,
    receive_wall_ns: int,
    receive_monotonic_ns: int,
) -> float:
    """Map a same-host ROS wall-clock stamp into the monotonic clock domain.

    Camera callbacks otherwise use message receipt time, which folds DDS and
    executor delay into the apparent capture time.  Invalid or clearly
    different-clock stamps fall back to receipt time instead of corrupting the
    synchronisation timeline.
    """
    stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    receive_monotonic = receive_monotonic_ns * 1e-9
    if stamp_ns <= 0:
        return receive_monotonic
    transport_ns = receive_wall_ns - stamp_ns
    if abs(transport_ns) > 60_000_000_000:
        return receive_monotonic
    return float(receive_monotonic_ns - transport_ns) * 1e-9


def ros_stamp_to_float_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


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
    sync_lag_sec: float
    use_image_header_stamp: bool

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
    if tactile_msg_package != "paxini_tactile":
        raise ValueError(f"paxini recorder requires tactile.msg_package=paxini_tactile, got: {tactile_msg_package}")
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
        wait_for_motion_enable_to_record=bool(record.get("wait_for_motion_enable_to_record", False)),
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
        sync_lag_sec=max(0.0, float(sync.get("sync_lag_sec", 0.0))),
        use_image_header_stamp=bool(sync.get("use_image_header_stamp", False)),

        save_master_parquet_every_episode=bool(output.get("save_master_parquet_every_episode", True)),
        save_debug_jsonl=bool(output.get("save_debug_jsonl", True)),
    )


def tactile_msg_to_record(msg: Any, recv_time: float) -> dict[str, Any]:
    data = np.asarray(list(msg.data), dtype=np.uint8)
    layout = {
        "force_raw_start": 0,
        "force_raw_len": TACTILE_FORCE_BYTES,
        "distribution_raw_start": TACTILE_FORCE_BYTES,
        "distribution_raw_len": TACTILE_FRAME_BYTES - TACTILE_FORCE_BYTES,
    }
    stamp = msg.header.stamp
    return {
        "msg_package": "paxini_tactile",
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

    @property
    def maxlen(self) -> int:
        return int(self._buf.maxlen or 0)

    def latest(self) -> Optional[TimedItem]:
        if not self._buf:
            return None
        return self._buf[-1]

    def nearest(
        self,
        t: float,
        *,
        accept: Optional[Callable[[TimedItem], bool]] = None,
    ) -> Optional[TimedItem]:
        if not self._buf:
            return None
        return nearest_time_ordered(self._buf, t, accept=accept)


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
        tactile_msg_type = PaxiniTactileFrame

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

    def _image_cb(self, msg: Image) -> None:
        receive_monotonic_ns = time.monotonic_ns()
        receive_wall_ns = time.time_ns()
        recv_t = receive_monotonic_ns * 1e-9
        sample_t = (
            ros_header_stamp_to_monotonic(
                msg.header.stamp,
                receive_wall_ns=receive_wall_ns,
                receive_monotonic_ns=receive_monotonic_ns,
            )
            if self.cfg.use_image_header_stamp
            else recv_t
        )
        with self._lock:
            self.image_buffer.append(sample_t, msg)

    def _wrist_image_cb(self, msg: Image) -> None:
        receive_monotonic_ns = time.monotonic_ns()
        receive_wall_ns = time.time_ns()
        recv_t = receive_monotonic_ns * 1e-9
        sample_t = (
            ros_header_stamp_to_monotonic(
                msg.header.stamp,
                receive_wall_ns=receive_wall_ns,
                receive_monotonic_ns=receive_monotonic_ns,
            )
            if self.cfg.use_image_header_stamp
            else recv_t
        )
        with self._lock:
            self.wrist_image_buffer.append(sample_t, msg)

    def _wrist_camera_info_cb(self, msg: CameraInfo) -> None:
        recv_t = now_monotonic()
        with self._lock:
            self.wrist_camera_info_buffer.append(recv_t, msg)

    def _command_event_cb(self, msg: String) -> None:
        event = parse_json_str(msg.data)
        http_info = event.get("http")
        if isinstance(http_info, dict) and "ok" in http_info and not bool(http_info["ok"]):
            return
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
        payload = tactile_msg_to_record(msg, recv_t)
        with self._lock:
            self.tactile_left_buffer.append(recv_t, payload)

    def _tactile_right_cb(self, msg: Any) -> None:
        recv_t = now_monotonic()
        payload = tactile_msg_to_record(msg, recv_t)
        with self._lock:
            self.tactile_right_buffer.append(recv_t, payload)

    def get_nearest_bundle(
        self,
        t_frame: float,
        *,
        min_image_stamp_ns: Optional[int] = None,
        min_wrist_image_stamp_ns: Optional[int] = None,
    ) -> dict[str, Optional[TimedItem]]:
        def newer_than(minimum: Optional[int]) -> Optional[Callable[[TimedItem], bool]]:
            if minimum is None or minimum <= 0:
                return None
            return lambda item: (
                ros_message_stamp_ns(item.payload) <= 0
                or ros_message_stamp_ns(item.payload) > minimum
            )

        with self._lock:
            return {
                "image": self.image_buffer.nearest(
                    t_frame, accept=newer_than(min_image_stamp_ns)
                ),
                "wrist_image": self.wrist_image_buffer.nearest(
                    t_frame, accept=newer_than(min_wrist_image_stamp_ns)
                ),
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
                "state": self.state_buffer.latest() is not None,
                "tactile_left": self.tactile_left_buffer.latest() is not None,
                "tactile_right": self.tactile_right_buffer.latest() is not None,
            }
        return all(status.values()), status


class ManualEpisodeController:
    def __init__(self):
        self.finish = False
        self.finish_time: Optional[float] = None
        self.discard = False
        self.quit = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self, episode_index: int) -> None:
        self.finish = False
        self.finish_time = None
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
                    self.finish_time = now_monotonic()
                    self.finish = True
                    break
                if line in ("r", "redo", "retry"):
                    self.finish_time = now_monotonic()
                    self.discard = True
                    self.finish = True
                    break
                if line in ("q", "quit", "stop"):
                    self.finish_time = now_monotonic()
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


def record_episode(
    cfg: RecordConfig,
    node: RawCollectionNode,
    writer: RawDatasetWriter,
    episode_index: int,
) -> tuple[bool, bool]:
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
    pre_action_skip_count = 0
    active_candidate_count = 0
    active_skip_count = 0
    active_started = False
    active_skip_reasons: Counter[str] = Counter()
    pre_action_skip_reasons: Counter[str] = Counter()
    require_unique_image_stamps = bool(
        getattr(cfg, "require_unique_image_stamps", False)
    )
    last_image_stamp_ns: Optional[int] = None
    last_wrist_image_stamp_ns: Optional[int] = None

    print(f"[RECORD] episode={episode_index}, fps={cfg.fps}, max_episode_sec={cfg.max_episode_sec}")
    sync_thresholds = ObservationSyncThresholds(
        max_image_dt_sec=cfg.max_image_dt_sec,
        max_wrist_image_dt_sec=cfg.max_wrist_image_dt_sec,
        max_action_dt_sec=cfg.max_action_dt_sec,
        max_state_dt_sec=cfg.max_state_dt_sec,
        max_vr_dt_sec=cfg.max_vr_dt_sec,
        max_tactile_dt_sec=cfg.max_tactile_dt_sec,
    )
    sync_lag_sec = max(0.0, float(cfg.sync_lag_sec))
    episode_deadline = ep_start + cfg.max_episode_sec
    max_loop_lag_sec = 0.0

    try:
        while True:
            now = now_monotonic()
            if controller.discard:
                break
            stop_time = controller.finish_time
            cutoff_time = episode_deadline if stop_time is None else min(stop_time, episode_deadline)
            if next_t > cutoff_time:
                if stop_time is None:
                    print("[RECORD] max_episode_sec reached, finishing episode.")
                else:
                    print(
                        f"[RECORD] finish requested; all candidates through the stop time "
                        f"were evaluated with sync_lag={sync_lag_sec:.2f}s"
                    )
                break
            evaluation_time = next_t + sync_lag_sec
            if now < evaluation_time:
                time.sleep(min(0.002, evaluation_time - now))
                continue

            t_frame = next_t
            loop_lag_sec = max(0.0, now - evaluation_time)
            max_loop_lag_sec = max(max_loop_lag_sec, loop_lag_sec)
            timestamp = float(t_frame - ep_start)
            next_t += period
            candidate_count += 1

            bundle = node.get_nearest_bundle(
                t_frame,
                min_image_stamp_ns=(
                    last_image_stamp_ns if require_unique_image_stamps else None
                ),
                min_wrist_image_stamp_ns=(
                    last_wrist_image_stamp_ns if require_unique_image_stamps else None
                ),
            )
            action_ready = has_usable_action_label(
                bundle.get("command"),
                t_frame,
                cfg.max_action_dt_sec,
            )
            if action_ready and not active_started:
                active_started = True
                print(
                    f"[ACTIVE] first valid action at candidate={candidate_count - 1} "
                    f"timestamp={timestamp:.3f}s"
                )
            if active_started:
                active_candidate_count += 1
            build_result = build_observation_frame(
                bundle=bundle,
                t_frame=t_frame,
                thresholds=sync_thresholds,
                wait_for_motion_enable=cfg.wait_for_motion_enable_to_record,
            )
            if not build_result.ok or build_result.frame is None:
                skip_count += 1
                reason_key = skip_reason_key(build_result.reason)
                if active_started:
                    active_skip_count += 1
                    active_skip_reasons[reason_key] += 1
                else:
                    pre_action_skip_count += 1
                    pre_action_skip_reasons[reason_key] += 1
                if build_result.reason.startswith("left/right tactile data length mismatch"):
                    print(f"[WARN] {build_result.reason}")
                elif active_started:
                    print(
                        f"[SKIP_ACTIVE] candidate={candidate_count - 1} "
                        f"timestamp={timestamp:.3f}s saved={saved_count} "
                        f"loop_lag_ms={loop_lag_sec * 1000.0:.1f} "
                        f"reason={build_result.reason}"
                    )
                elif cfg.debug and build_result.reason == "enabled false or missing" and skip_count % 20 == 0:
                    print(f"[SKIP] enabled false or missing. saved={saved_count} skip={skip_count}")
                elif cfg.debug and skip_count % 10 == 0:
                    print(
                        f"[SKIP] saved={saved_count} skip={skip_count} "
                        f"loop_lag_ms={loop_lag_sec * 1000.0:.1f} {build_result.reason}"
                    )
                continue

            obs = build_result.frame
            image_item = obs.image_item
            wrist_image_item = obs.wrist_image_item
            wrist_camera_info_item = obs.wrist_camera_info_item
            tactile_left_payload = obs.tactile_left_payload
            tactile_right_payload = obs.tactile_right_payload
            command_event = obs.command_event
            state_event = obs.state_event
            action8 = obs.action8
            vr_payload = obs.vr_payload
            joystick_y_payload = obs.joystick_y_payload
            state_fields = obs.state_fields
            target_gripper_width = obs.target_gripper_width
            target_gripper_finger_position = obs.target_gripper_finger_position
            observation_state7 = obs.observation_state7
            relative_action7 = obs.relative_action7
            enabled_val = obs.enabled_val
            ok_vr = obs.ok_vr
            dt_image = obs.dt_image
            dt_wrist_image = obs.dt_wrist_image
            dt_action = obs.dt_action
            dt_state = obs.dt_state
            dt_vr = obs.dt_vr
            dt_tactile_left = obs.dt_tactile_left
            dt_tactile_right = obs.dt_tactile_right
            reason_vr = obs.reason_vr

            try:
                (
                    image_rel_path,
                    image_rec,
                    wrist_image_rel_path,
                    wrist_image_rec,
                    tactile_left_rec,
                    tactile_right_rec,
                ) = writer.write_frame_assets(
                    episode_index=episode_index,
                    frame_index=saved_count,
                    timestamp=timestamp,
                    wall_time=t_frame,
                    image_msg=image_item.payload,
                    wrist_image_msg=wrist_image_item.payload,
                    left_payload=tactile_left_payload,
                    left_dt=float(dt_tactile_left),
                    right_payload=tactile_right_payload,
                    right_dt=float(dt_tactile_right),
                )
            except Exception as e:
                skip_count += 1
                reason_key = "frame_asset_enqueue_or_write_failed"
                if active_started:
                    active_skip_count += 1
                    active_skip_reasons[reason_key] += 1
                else:
                    pre_action_skip_count += 1
                    pre_action_skip_reasons[reason_key] += 1
                print(f"[WARN] frame asset enqueue/write failed: {repr(e)}")
                continue

            if require_unique_image_stamps:
                image_stamp_ns = ros_message_stamp_ns(image_item.payload)
                wrist_stamp_ns = ros_message_stamp_ns(wrist_image_item.payload)
                if image_stamp_ns > 0:
                    last_image_stamp_ns = image_stamp_ns
                if wrist_stamp_ns > 0:
                    last_wrist_image_stamp_ns = wrist_stamp_ns

            wrist_camera_info_fields = builder_camera_info_to_record(wrist_camera_info_item, t_frame)

            row: dict[str, Any] = {
                "timestamp": np.float32(timestamp),
                "wall_time": float(t_frame),
                "frame_index": int(saved_count),
                "candidate_index": int(candidate_count - 1),
                "episode_index": int(episode_index),
                "index": None,
                "task_index": int(cfg.task_index),

                "observation.state": observation_state7.astype(np.float32).tolist(),
                "observation.state_names": STATE_NAMES,

                "action": relative_action7.astype(np.float32).tolist(),
                "action.names": ACTION_NAMES,
                "action.sent_action8": action8.astype(np.float32).tolist(),
                "action.pose7": action8[:7].astype(np.float32).tolist(),
                "action.target_gripper": float(action8[7]),
                "action.target_gripper_width": float(target_gripper_width),
                "action.target_gripper_finger_position": float(
                    target_gripper_finger_position
                ),
                "action.command_type": str(command_event.get("command_type", command_event.get("type", ""))),
                "action.heartbeat": bool(command_event.get("heartbeat", False)),
                "action.repeated": bool(command_event.get("repeated", False)),
                "action.pose_commanded": bool(command_event.get("pose_commanded", True)),
                "action.gripper_commanded": bool(command_event.get("gripper_commanded", False)),
                "action.motion_enabled": bool(command_event.get("enabled", False)),
                "action.http_route": str(command_event.get("http", {}).get("route", "")),
                "action.http_ok": bool(command_event.get("http", {}).get("ok", False)),
                "action.http_latency_ms": float(command_event.get("http", {}).get("latency_ms", np.nan)),
                "action.t_send_start": float(command_event.get("http", {}).get("t_send_start", np.nan)),
                "action.t_send_end": float(command_event.get("http", {}).get("t_send_end", np.nan)),
                "action.workspace_clipped": bool(command_event.get("workspace_clipped", False)),

                "image.path": image_rel_path,
                "image.height": int(image_rec["height"]),
                "image.width": int(image_rec["width"]),
                "image.encoding": str(image_rec["encoding"]),
                "image.storage_encoding": str(image_rec.get("storage_encoding", "png")),
                "image.step": int(image_rec["step"]),
                "image.ros_stamp_sec": int(image_rec["ros_stamp_sec"]),
                "image.ros_stamp_nanosec": int(image_rec["ros_stamp_nanosec"]),
                "image.ros_stamp_float": float(image_rec["ros_stamp_float"]),
                "image.frame_id": str(image_rec["frame_id"]),

                "wrist_image.path": wrist_image_rel_path,
                "wrist_image.height": int(wrist_image_rec["height"]),
                "wrist_image.width": int(wrist_image_rec["width"]),
                "wrist_image.encoding": str(wrist_image_rec["encoding"]),
                "wrist_image.storage_encoding": str(
                    wrist_image_rec.get("storage_encoding", "png")
                ),
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
            row.update(builder_tactile_row_fields("tactile.left", tactile_left_rec))
            row.update(builder_tactile_row_fields("tactile.right", tactile_right_rec))

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
                "observation_state7": observation_state7.astype(np.float32).tolist(),
                "relative_action7": relative_action7.astype(np.float32).tolist(),
                "sent_action8": action8.astype(np.float32).tolist(),
                "sent_pose7": action8[:7].astype(np.float32).tolist(),
                "target_gripper": float(action8[7]),
                "target_gripper_width": float(target_gripper_width),
                "target_gripper_finger_position": float(
                    target_gripper_finger_position
                ),
                "command_type": str(command_event.get("command_type", command_event.get("type", ""))),
                "heartbeat": bool(command_event.get("heartbeat", False)),
                "repeated": bool(command_event.get("repeated", False)),
                "pose_commanded": bool(command_event.get("pose_commanded", True)),
                "gripper_commanded": bool(command_event.get("gripper_commanded", False)),
                "motion_enabled": bool(command_event.get("enabled", False)),
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
                    f"dt_tactile_left={dt_tactile_left:+.3f} dt_tactile_right={dt_tactile_right:+.3f} "
                    f"loop_lag_ms={loop_lag_sec * 1000.0:.1f} "
                    f"writer_queue={writer.pending_write_tasks()}"
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

    print(
        f"[SKIP_SUMMARY] total={skip_count} pre_action={pre_action_skip_count} "
        f"active_candidates={active_candidate_count} active_skipped={active_skip_count} "
        f"pre_action_reasons={dict(pre_action_skip_reasons)} "
        f"active_reasons={dict(active_skip_reasons)}"
    )

    continuity = candidate_continuity_report(
        row["candidate_index"] for row in writer.current_episode_rows
    )
    print(
        f"[EPISODE_QUALITY] saved={continuity.saved_frames} "
        f"first_candidate={continuity.first_candidate} "
        f"last_candidate={continuity.last_candidate} "
        f"internal_missing={continuity.internal_missing} "
        f"duplicates={continuity.duplicate_steps} "
        f"backwards={continuity.backwards_steps} "
        f"active_skipped={active_skip_count}"
    )
    reject_on_active_skip = bool(
        getattr(cfg, "discard_episode_on_active_skip", False)
    )
    quality_failed = active_skip_count > 0 or not continuity.continuous
    if reject_on_active_skip and quality_failed:
        print(
            f"[QUALITY_REJECT] episode {episode_index} was not committed: "
            f"active_skipped={active_skip_count}, "
            f"internal_missing={continuity.internal_missing}, "
            f"duplicates={continuity.duplicate_steps}, "
            f"backwards={continuity.backwards_steps}. "
            "Reset the scene and re-record the same episode index."
        )
        writer.discard_episode()
        return False, controller.quit

    writer.commit_episode(episode_index)
    if cfg.save_master_parquet_every_episode:
        writer.flush_master()
    print(
        f"[DONE] episode {episode_index} saved={saved_count} skipped={skip_count} "
        f"max_loop_lag_ms={max_loop_lag_sec * 1000.0:.1f}"
    )
    performance_text = getattr(writer, "performance_text", None)
    if callable(performance_text):
        print(f"[WRITER_SUMMARY] {performance_text()}")
    return True, controller.quit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = load_config(cfg_path)
    recorder_label = "Paxini tactile"
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

    writer = RawDatasetWriter(cfg, config_path=cfg_path, profile=PAXINI_RAW_WRITER_PROFILE)
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


if __name__ == "__main__":
    main()
