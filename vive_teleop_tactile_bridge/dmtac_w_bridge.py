#!/usr/bin/env python3
"""ROS 2 packed-frame publisher for the Python-3.11 DM-Tac W SDK worker.

The ROS process never imports ``dmrobotics``.  It starts the SDK worker with an
explicit virtual-environment interpreter, reads fixed frames from /dev/shm, and
publishes one packed sensor_msgs/Image per side. Legacy modality topics are
optional and disabled by default.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from .dmtac_w_ipc import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MODALITY_SPECS,
    PACKED_IMAGE_ENCODING,
    PAYLOAD_BYTES,
    FrameSnapshot,
    MMapFrameReader,
    payload_segment,
)


PACKAGE_NAME = "vive_teleop_tactile_bridge"


class DMTacWBridge(Node):
    def __init__(self) -> None:
        super().__init__("dmtac_w_bridge")
        self.declare_parameter("sdk_python", "")
        self.declare_parameter("worker_script", "")
        self.declare_parameter("left_serial", "CHANGE_ME_LEFT")
        self.declare_parameter("right_serial", "CHANGE_ME_RIGHT")
        self.declare_parameter("left_frame_path", "/dev/shm/dmtac_w_left.frame")
        self.declare_parameter("right_frame_path", "/dev/shm/dmtac_w_right.frame")
        self.declare_parameter("base_topic", "/dmtac")
        self.declare_parameter("max_fps", 30.0)
        self.declare_parameter("poll_hz", 200.0)
        self.declare_parameter("startup_timeout_sec", 60.0)
        self.declare_parameter("frame_timeout_sec", 10.0)
        self.declare_parameter("warmup_cycles", 3)
        self.declare_parameter("show_sdk_fps", False)
        self.declare_parameter("qos_depth", 1)
        self.declare_parameter("publish_legacy_modalities", False)
        self.declare_parameter("worker_shutdown_timeout_sec", 5.0)

        self._closing = False
        self._background_error: BaseException | None = None
        self._worker: subprocess.Popen[str] | None = None
        self._worker_log_thread: threading.Thread | None = None
        self._timer: Any = None

        self.sdk_python = Path(str(self.get_parameter("sdk_python").value)).expanduser()
        worker_script_value = str(self.get_parameter("worker_script").value).strip()
        if worker_script_value:
            self.worker_script = Path(worker_script_value).expanduser()
        else:
            self.worker_script = (
                Path(get_package_share_directory(PACKAGE_NAME))
                / "scripts"
                / "dmtac_w_sdk_worker.py"
            )

        self.serials = {
            "left": str(self.get_parameter("left_serial").value).strip(),
            "right": str(self.get_parameter("right_serial").value).strip(),
        }
        self.frame_paths = {
            "left": Path(str(self.get_parameter("left_frame_path").value)).expanduser(),
            "right": Path(str(self.get_parameter("right_frame_path").value)).expanduser(),
        }
        self.session_id = secrets.randbits(63) + 1
        self.base_topic = str(self.get_parameter("base_topic").value).strip().rstrip("/")
        self.max_fps = float(self.get_parameter("max_fps").value)
        self.poll_hz = float(self.get_parameter("poll_hz").value)
        self.startup_timeout_sec = float(self.get_parameter("startup_timeout_sec").value)
        self.frame_timeout_sec = float(self.get_parameter("frame_timeout_sec").value)
        self.warmup_cycles = int(self.get_parameter("warmup_cycles").value)
        self.show_sdk_fps = bool(self.get_parameter("show_sdk_fps").value)
        self.qos_depth = int(self.get_parameter("qos_depth").value)
        self.publish_legacy_modalities = bool(
            self.get_parameter("publish_legacy_modalities").value
        )
        self.worker_shutdown_timeout_sec = float(
            self.get_parameter("worker_shutdown_timeout_sec").value
        )

        self._validate_parameters()

        packed_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.qos_depth,
        )
        self._packed_publishers = {
            side: self.create_publisher(
                Image, f"{self.base_topic}/{side}/packed_frame", packed_qos
            )
            for side in ("left", "right")
        }
        self._image_publishers: dict[str, dict[str, Any]] = {}
        if self.publish_legacy_modalities:
            legacy_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=self.qos_depth,
            )
            for side in ("left", "right"):
                prefix = f"{self.base_topic}/{side}"
                self._image_publishers[side] = {
                    spec.name: self.create_publisher(Image, f"{prefix}/{spec.name}", legacy_qos)
                    for spec in MODALITY_SPECS
                }

        self.readers = {
            side: MMapFrameReader(path, session_id=self.session_id)
            for side, path in self.frame_paths.items()
        }
        self.last_sequences = {"left": 0, "right": 0}
        self.last_frame_indices = {"left": 0, "right": 0}
        self.last_frame_monotonic: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self.started_monotonic = time.monotonic()

        try:
            self._start_worker()
            self._timer = self.create_timer(1.0 / self.poll_hz, self._poll_shared_frames)
        except Exception:
            self.close()
            raise

        self.get_logger().warning(
            "DM-Tac W automatically resets on SDK startup. Keep both tactile surfaces "
            "completely unloaded until the worker reports both sensors OK."
        )
        self.get_logger().info(
            f"DM-Tac W bridge ready: left={self.serials['left']} "
            f"right={self.serials['right']} target_fps={self.max_fps} "
            f"packed_topics={self.base_topic}/<side>/packed_frame "
            f"legacy_modalities={self.publish_legacy_modalities}"
        )

    def _validate_parameters(self) -> None:
        if not self.sdk_python.is_file() or not os.access(self.sdk_python, os.X_OK):
            raise ValueError(f"sdk_python is not executable: {self.sdk_python}")
        if not self.worker_script.is_file():
            raise ValueError(f"worker_script does not exist: {self.worker_script}")
        for side, serial in self.serials.items():
            if not serial or serial.startswith("CHANGE_ME"):
                raise ValueError(f"set a real {side}_serial in dmtac_w_bridge.params.yaml")
        if self.serials["left"] == self.serials["right"]:
            raise ValueError("left_serial and right_serial must differ")
        if self.frame_paths["left"] == self.frame_paths["right"]:
            raise ValueError("left_frame_path and right_frame_path must differ")
        if not self.base_topic.startswith("/"):
            raise ValueError("base_topic must be an absolute ROS namespace")
        if self.max_fps <= 0:
            raise ValueError("max_fps must be positive")
        if self.poll_hz < self.max_fps:
            raise ValueError("poll_hz must be greater than or equal to max_fps")
        if self.startup_timeout_sec <= 0 or self.frame_timeout_sec <= 0:
            raise ValueError("startup_timeout_sec and frame_timeout_sec must be positive")
        if self.warmup_cycles < 0:
            raise ValueError("warmup_cycles must be non-negative")
        if self.qos_depth <= 0:
            raise ValueError("qos_depth must be positive")
        if self.worker_shutdown_timeout_sec <= 0:
            raise ValueError("worker_shutdown_timeout_sec must be positive")

    def _worker_command(self) -> list[str]:
        command = [
            str(self.sdk_python),
            "-I",
            "-B",
            str(self.worker_script),
            "--left-serial",
            self.serials["left"],
            "--right-serial",
            self.serials["right"],
            "--left-frame-path",
            str(self.frame_paths["left"]),
            "--right-frame-path",
            str(self.frame_paths["right"]),
            "--session-id",
            str(self.session_id),
            "--fps",
            str(self.max_fps),
            "--startup-timeout-sec",
            str(self.startup_timeout_sec),
            "--warmup-cycles",
            str(self.warmup_cycles),
        ]
        if self.show_sdk_fps:
            command.append("--show-sdk-fps")
        return command

    def _start_worker(self) -> None:
        environment = os.environ.copy()
        # The SDK interpreter must never import ROS Jazzy's Python-3.12 modules.
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        command = self._worker_command()
        self.get_logger().info(f"starting SDK worker with {self.sdk_python}")
        self._worker = subprocess.Popen(
            command,
            cwd=str(self.worker_script.parent),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._worker_log_thread = threading.Thread(
            target=self._forward_worker_logs,
            name="dmtac-w-sdk-log",
            daemon=True,
        )
        self._worker_log_thread.start()

    def _forward_worker_logs(self) -> None:
        worker = self._worker
        if worker is None or worker.stdout is None:
            return
        try:
            for line in worker.stdout:
                line = line.rstrip()
                if line:
                    self.get_logger().info(f"[sdk-worker] {line}")
        except Exception as exc:
            if not self._closing:
                self._set_background_error(
                    RuntimeError(f"failed to read SDK worker logs: {exc!r}")
                )

    def _set_background_error(self, error: BaseException) -> None:
        if self._background_error is None:
            self._background_error = error
            self.get_logger().error(str(error))

    @staticmethod
    def _stamp_image(message: Image, capture_ns: int) -> None:
        message.header.stamp.sec = int(capture_ns // 1_000_000_000)
        message.header.stamp.nanosec = int(capture_ns % 1_000_000_000)

    def _publish_snapshot(self, side: str, snapshot: FrameSnapshot) -> None:
        frame_id = f"{self.serials[side]}|fid={snapshot.frame_idx}"
        capture_ns = snapshot.capture_mid_ns
        packed = Image()
        self._stamp_image(packed, capture_ns)
        packed.header.frame_id = frame_id
        packed.height = 1
        packed.width = PAYLOAD_BYTES
        packed.encoding = PACKED_IMAGE_ENCODING
        packed.is_bigendian = 0
        packed.step = PAYLOAD_BYTES
        packed.data = snapshot.payload
        self._packed_publishers[side].publish(packed)

        if self.publish_legacy_modalities:
            for spec in MODALITY_SPECS:
                message = Image()
                self._stamp_image(message, capture_ns)
                message.header.frame_id = frame_id
                message.height = IMAGE_HEIGHT
                message.width = IMAGE_WIDTH
                message.encoding = spec.encoding
                message.is_bigendian = 0
                message.step = spec.step
                message.data = payload_segment(snapshot.payload, spec)
                self._image_publishers[side][spec.name].publish(message)

    def _poll_shared_frames(self) -> None:
        if self._closing or self._background_error is not None:
            return
        worker = self._worker
        if worker is None:
            self._set_background_error(RuntimeError("SDK worker was not started"))
            return
        return_code = worker.poll()
        if return_code is not None:
            self._set_background_error(
                RuntimeError(f"DM-Tac W SDK worker exited with code {return_code}")
            )
            return

        now = time.monotonic()
        for side, reader in self.readers.items():
            try:
                snapshot = reader.read_new(self.last_sequences[side])
            except Exception as exc:
                self._set_background_error(
                    RuntimeError(f"failed to read {side} shared frame: {exc!r}")
                )
                return
            if snapshot is None:
                continue
            if snapshot.frame_idx <= self.last_frame_indices[side]:
                self._set_background_error(
                    RuntimeError(
                        f"non-monotonic {side} SDK frame index: "
                        f"last={self.last_frame_indices[side]}, current={snapshot.frame_idx}"
                    )
                )
                return
            first_frame = self.last_frame_indices[side] == 0
            self._publish_snapshot(side, snapshot)
            self.last_sequences[side] = snapshot.sequence
            self.last_frame_indices[side] = snapshot.frame_idx
            self.last_frame_monotonic[side] = now
            if first_frame:
                capture_ms = (
                    snapshot.capture_end_ns - snapshot.capture_start_ns
                ) / 1_000_000.0
                self.get_logger().info(
                    f"received first {side} frame; SDK capture_ms={capture_ms:.2f}"
                )

        startup_limit = self.startup_timeout_sec + 15.0
        missing = [
            side for side, last_time in self.last_frame_monotonic.items() if last_time is None
        ]
        if missing and now - self.started_monotonic > startup_limit:
            self._set_background_error(
                TimeoutError(f"no DM-Tac W frames before startup timeout: {missing}")
            )
            return

        stale = [
            side
            for side, last_time in self.last_frame_monotonic.items()
            if last_time is not None and now - last_time > self.frame_timeout_sec
        ]
        if stale:
            self._set_background_error(
                TimeoutError(f"DM-Tac W frame stream became stale: {stale}")
            )

    def raise_background_error(self) -> None:
        if self._background_error is not None:
            raise RuntimeError("DM-Tac W bridge failed") from self._background_error

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._timer is not None:
            try:
                self._timer.cancel()
            except Exception:
                pass

        worker = self._worker
        if worker is not None and worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=self.worker_shutdown_timeout_sec)
            except subprocess.TimeoutExpired:
                self.get_logger().warning("SDK worker did not stop; sending SIGKILL")
                worker.kill()
                worker.wait(timeout=2.0)
        if worker is not None and worker.stdout is not None:
            worker.stdout.close()
        if self._worker_log_thread is not None:
            self._worker_log_thread.join(timeout=1.0)
        for reader in self.readers.values():
            reader.close()


def main() -> None:
    rclpy.init(args=None)
    node: DMTacWBridge | None = None
    try:
        node = DMTacWBridge()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.raise_background_error()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
