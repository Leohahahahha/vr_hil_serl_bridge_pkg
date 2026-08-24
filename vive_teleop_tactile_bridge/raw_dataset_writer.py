from __future__ import annotations

from io import BytesIO
import json
import queue
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from sensor_msgs.msg import CompressedImage, Image

try:
    from .action_space import ACTION_NAMES, STATE_NAMES
    from .image_utils import image_msg_to_rgb8
    from .observation_builder import ros_stamp_to_float_sec
except ImportError:
    from action_space import ACTION_NAMES, STATE_NAMES  # type: ignore
    from image_utils import image_msg_to_rgb8  # type: ignore
    from observation_builder import ros_stamp_to_float_sec  # type: ignore


class RawDatasetWriterConfig(Protocol):
    dataset_root: Path
    fps: float
    task_description: str
    task_index: int
    tactile_msg_package: str
    image_name: str
    image_topic: str
    wrist_image_name: str
    wrist_image_topic: str
    save_debug_jsonl: bool


@dataclass(frozen=True)
class RawDatasetWriterProfile:
    tactile_feature_description: str
    stream_tactile: bool = False


@dataclass
class _FrameWriteTask:
    image_msg: Image | CompressedImage
    image_out_path: Path
    wrist_image_msg: Image
    wrist_image_out_path: Path
    left_record: dict[str, Any]
    right_record: dict[str, Any]


@dataclass
class _FlushWriteTask:
    completed: threading.Event


_STOP_WRITER = object()


TASHAN_RAW_WRITER_PROFILE = RawDatasetWriterProfile(
    tactile_feature_description=(
        "float32[] flattened Tashan tactile frame in zarr; "
        "layout lengths are stored in tactile metadata"
    )
)

PAXINI_RAW_WRITER_PROFILE = RawDatasetWriterProfile(
    tactile_feature_description=(
        "uint8[] raw Paxini tactile frame in zarr; "
        "[0:3]=force, [3:234]=distributed force"
    )
)

DMTAC_RAW_WRITER_PROFILE = RawDatasetWriterProfile(
    tactile_feature_description=(
        "uint8[] lossless packed DM-Tac SDK 1.2.13.1 frame in zarr; contains raw and "
        "inference images, deformation2d, depth, shear, distributed force, six-axis "
        "wrench, and contact area; source dtypes, image shapes, byte offsets, and lengths "
        "are stored in tactile metadata"
    ),
    stream_tactile=True,
)

DMTAC_W_RAW_WRITER_PROFILE = RawDatasetWriterProfile(
    tactile_feature_description=(
        "uint8[] lossless packed DM-Tac W SDK 0.1.4 software sample in zarr; "
        "schema 1/full contains raw image, deformation2d, normal, shear, and "
        "depth; schema 3/shear_depth contains shear then depth; source dtypes, "
        "image shapes, byte offsets, and lengths are stored in tactile metadata; "
        "the legacy SDK does not expose an atomic hardware frame id"
    ),
    stream_tactile=True,
)


# Dataset schema 2 belongs to the newer DM-Tac SDK and intentionally is not
# included here. These constants guard only the legacy DM-Tac W recorder when
# its config exposes ``tactile_output_mode``.
_DMTAC_W_FORMATS = {
    "full": (1, 1_920_000),
    "shear_depth": (3, 921_600),
}


class RawDatasetWriter:
    def __init__(self, cfg: RawDatasetWriterConfig, config_path: Path, profile: RawDatasetWriterProfile):
        self.cfg = cfg
        self.profile = profile
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
        self.current_tactile_left_start: Optional[int] = None
        self.current_tactile_right_start: Optional[int] = None

        self._async_frame_writes = bool(profile.stream_tactile)
        queue_size = max(4, int(getattr(cfg, "write_queue_size", 48)))
        self._tactile_batch_frames = max(
            1, int(getattr(cfg, "tactile_batch_frames", 8))
        )
        self._write_queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_error: Optional[BaseException] = None
        self._expected_dmtac_w_format = self._resolve_expected_dmtac_w_format()

    def _resolve_expected_dmtac_w_format(self) -> Optional[tuple[str, int, int]]:
        """Return configured legacy DM-Tac W mode/schema/bytes, if applicable."""
        mode = getattr(self.cfg, "tactile_output_mode", None)
        if mode is None:
            return None
        if str(self.cfg.tactile_msg_package) != "dmtac_tactile":
            raise RuntimeError(
                "tactile_output_mode is only valid with tactile_msg_package=dmtac_tactile"
            )
        mode = str(mode).strip().lower()
        if mode not in _DMTAC_W_FORMATS:
            raise RuntimeError(
                f"unsupported DM-Tac W output mode {mode!r}; "
                f"expected one of {sorted(_DMTAC_W_FORMATS)}"
            )
        mapped_schema, mapped_bytes = _DMTAC_W_FORMATS[mode]
        schema = int(getattr(self.cfg, "tactile_schema_version", mapped_schema))
        packed_bytes = int(getattr(self.cfg, "tactile_packed_frame_bytes", mapped_bytes))
        if (schema, packed_bytes) != (mapped_schema, mapped_bytes):
            raise RuntimeError(
                "configured DM-Tac W mode/schema/size mismatch: "
                f"mode={mode!r} requires schema={mapped_schema}, bytes={mapped_bytes}; "
                f"got schema={schema}, bytes={packed_bytes}"
            )
        return mode, schema, packed_bytes

    def _validate_existing_dmtac_w_format(self) -> None:
        """Refuse to append a configured schema to a root containing another one."""
        expected = self._expected_dmtac_w_format
        if expected is None:
            return
        mode, schema, packed_bytes = expected

        if self.info_path.exists():
            try:
                info = json.loads(self.info_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(
                    f"cannot verify existing dataset metadata {self.info_path}: {exc}"
                ) from exc
            existing_package = info.get("tactile_msg_package")
            if existing_package not in (None, "dmtac_tactile"):
                raise RuntimeError(
                    "dataset root already contains another tactile package: "
                    f"existing={existing_package!r}, configured='dmtac_tactile'"
                )
            checks = (
                ("tactile_output_mode", mode),
                ("tactile_schema_version", schema),
                ("tactile_packed_frame_bytes", packed_bytes),
            )
            for key, expected_value in checks:
                existing_value = info.get(key)
                if existing_value is not None and existing_value != expected_value:
                    raise RuntimeError(
                        "refusing to mix DM-Tac W formats in one dataset root: "
                        f"meta/info.json {key}={existing_value!r}, "
                        f"configured={expected_value!r}. Use a new empty dataset root."
                    )

        # Old schema-1 datasets predate the explicit info.json format fields,
        # so the Zarr frame width is the authoritative compatibility check.
        try:
            import zarr
        except ImportError as exc:
            raise RuntimeError(
                "zarr is required to verify an existing tactile dataset"
            ) from exc
        for side, zarr_path in (
            ("left", self.tactile_left_zarr_path),
            ("right", self.tactile_right_zarr_path),
        ):
            if not zarr_path.exists():
                continue
            arr = zarr.open(str(zarr_path), mode="r")
            if len(arr.shape) != 2:
                raise RuntimeError(
                    f"existing DM-Tac W {side} zarr has invalid shape {tuple(arr.shape)}"
                )
            existing_bytes = int(arr.shape[1])
            if existing_bytes != packed_bytes:
                raise RuntimeError(
                    "refusing to mix DM-Tac W formats in one dataset root: "
                    f"existing {side} zarr frame bytes={existing_bytes}, "
                    f"configured output_mode={mode!r}/schema={schema}/bytes={packed_bytes}. "
                    "Use a new empty dataset root."
                )

    def _validate_dmtac_w_payload(self, payload: dict[str, Any]) -> None:
        expected = self._expected_dmtac_w_format
        if expected is None:
            return
        mode, schema, packed_bytes = expected
        layout = payload.get("layout")
        if not isinstance(layout, dict):
            raise RuntimeError("DM-Tac W payload layout must be a mapping")
        actual_schema = int(layout.get("schema_version", -1))
        actual_packed_bytes = int(layout.get("packed_frame_bytes", -1))
        actual_data_len = int(payload.get("data_len", -1))
        actual_data_size = int(np.asarray(payload.get("data", ())).size)
        if (
            actual_schema != schema
            or actual_packed_bytes != packed_bytes
            or actual_data_len != packed_bytes
            or actual_data_size != packed_bytes
        ):
            raise RuntimeError(
                "DM-Tac W payload does not match configured format: "
                f"configured mode={mode!r}/schema={schema}/bytes={packed_bytes}; "
                f"payload schema={actual_schema}, packed_frame_bytes={actual_packed_bytes}, "
                f"data_len={actual_data_len}, data_size={actual_data_size}"
            )

    def prepare(self) -> None:
        try:
            import zarr  # noqa: F401
        except ImportError as e:
            raise RuntimeError("zarr is required to save tactile data. Install python package 'zarr'.") from e

        self._validate_existing_dmtac_w_format()

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
            "description": (
                "Raw ZED front RGB PNG + D405 wrist RGB PNG images + HIL-SERL sent command "
                "+ franka_server robot state. Convert this raw dataset to SDP HDF5 before training."
            ),
            "fps": self.cfg.fps,
            "task_description": self.cfg.task_description,
            "task_index": self.cfg.task_index,
            "tactile_msg_package": self.cfg.tactile_msg_package,
            "state_names": STATE_NAMES,
            "state_dim": 7,
            "action_names": ACTION_NAMES,
            "action_dim": 7,
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
                "observation.state": "float32[7], current xyz + absolute rotation vector + current gripper width",
                "action": (
                    "float32[7], target pose relative to current pose as xyz delta "
                    "+ rotation-vector delta + target gripper width"
                ),
                "action.sent_action8": (
                    "float32[8], raw absolute robot command event pose7 + target gripper width"
                ),
                "action.pose7": "float32[7], raw absolute target/held pose used by the robot command event",
                "action.command_type": "string, command_event_json type; currently robot_command",
                "action.heartbeat": "bool, true when the label repeats the last successful held command",
                "action.repeated": "bool, true when no new HTTP command was sent for this event",
                "action.pose_commanded": "bool, true when this command event sent an HTTP /pose target",
                "action.gripper_commanded": "bool, true when this command event sent a gripper command",
                "action.motion_enabled": "bool, /vr_bridge/enabled value captured by the command event",
                "action.http_route": "string, HTTP route used by the command event, e.g. /pose or /move_gripper",
                "robot.ee_pose": "float32[7], absolute current end-effector pose parsed from /getstate",
                "robot.gripper_pos": "float, current gripper width parsed from /getstate",
                "image.path": "relative path to ZED front RGB PNG sidecar",
                "wrist_image.path": "relative path to D405 wrist RGB PNG sidecar",
                "tactile.*.zarr_index": "row index into tactile/tactile_left|right/data.zarr",
                "tactile.*.data": self.profile.tactile_feature_description,
                "timestamp": "seconds from episode start on fixed-FPS frame grid",
                "wall_time": "monotonic time at target frame",
                "sync.*": "diagnostics for nearest-neighbor alignment",
            },
        }
        if self._expected_dmtac_w_format is not None:
            mode, schema, packed_bytes = self._expected_dmtac_w_format
            info.update(
                {
                    "tactile_output_mode": mode,
                    "tactile_schema_version": schema,
                    "tactile_packed_frame_bytes": packed_bytes,
                }
            )
        with open(self.info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        if self._async_frame_writes:
            self._start_background_writer()

    def _start_background_writer(self) -> None:
        if self._writer_thread is not None:
            return
        self._writer_thread = threading.Thread(
            target=self._background_writer_loop,
            name="raw-dataset-writer",
            daemon=True,
        )
        self._writer_thread.start()

    def _raise_writer_error(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError("background dataset writer failed") from self._writer_error

    def _enqueue_write(self, task: object) -> None:
        self._raise_writer_error()
        while True:
            try:
                self._write_queue.put(task, timeout=0.1)
                return
            except queue.Full:
                self._raise_writer_error()

    def _background_writer_loop(self) -> None:
        tactile_batch: list[_FrameWriteTask] = []
        while True:
            task = self._write_queue.get()
            try:
                if self._writer_error is not None:
                    if isinstance(task, _FlushWriteTask):
                        task.completed.set()
                    if task is _STOP_WRITER:
                        return
                    continue
                if task is _STOP_WRITER:
                    self._flush_tactile_task_batch(tactile_batch)
                    return
                if isinstance(task, _FlushWriteTask):
                    self._flush_tactile_task_batch(tactile_batch)
                    task.completed.set()
                    continue
                if not isinstance(task, _FrameWriteTask):
                    raise TypeError(f"unsupported writer task: {type(task)!r}")

                self._write_png(task.image_msg, task.image_out_path)
                self._write_png(task.wrist_image_msg, task.wrist_image_out_path)
                tactile_batch.append(task)
                if len(tactile_batch) >= self._tactile_batch_frames:
                    self._flush_tactile_task_batch(tactile_batch)
            except BaseException as exc:
                self._writer_error = exc
                if isinstance(task, _FlushWriteTask):
                    task.completed.set()
            finally:
                self._write_queue.task_done()

    @staticmethod
    def _write_png(msg: Image | CompressedImage, out_path: Path) -> None:
        if isinstance(msg, CompressedImage):
            try:
                with PILImage.open(BytesIO(bytes(msg.data))) as encoded_image:
                    encoded_image.convert("RGB").save(str(out_path), format="PNG")
            except Exception as exc:
                raise RuntimeError(f"failed to decode/write compressed PNG: {out_path}") from exc
            return
        rgb = image_msg_to_rgb8(msg)
        try:
            PILImage.fromarray(rgb, mode="RGB").save(str(out_path))
        except Exception as exc:
            raise RuntimeError(f"failed to write PNG: {out_path}") from exc

    def _flush_tactile_task_batch(self, tasks: list[_FrameWriteTask]) -> None:
        if not tasks:
            return
        left_records = [task.left_record for task in tasks]
        right_records = [task.right_record for task in tasks]
        left_start = self._tactile_zarr_row_count(self.tactile_left_zarr_path)
        right_start = self._tactile_zarr_row_count(self.tactile_right_zarr_path)
        success = False
        try:
            self._append_tactile_records_to_zarr(
                records=left_records,
                zarr_path=self.tactile_left_zarr_path,
                meta_path=self.tactile_left_meta_path,
                write_metadata=False,
            )
            self._append_tactile_records_to_zarr(
                records=right_records,
                zarr_path=self.tactile_right_zarr_path,
                meta_path=self.tactile_right_meta_path,
                write_metadata=False,
            )
            success = True
        except BaseException:
            self._truncate_tactile_zarr(self.tactile_left_zarr_path, left_start)
            self._truncate_tactile_zarr(self.tactile_right_zarr_path, right_start)
            raise
        finally:
            if success:
                for record in left_records + right_records:
                    record.pop("data", None)
            tasks.clear()

    def pending_write_tasks(self) -> int:
        return int(self._write_queue.qsize()) if self._async_frame_writes else 0

    def flush_pending_writes(self) -> None:
        if not self._async_frame_writes:
            return
        completed = threading.Event()
        barrier = _FlushWriteTask(completed=completed)
        while True:
            try:
                self._write_queue.put(barrier, timeout=0.1)
                break
            except queue.Full:
                continue
        while not completed.wait(timeout=0.1):
            if self._writer_thread is not None and not self._writer_thread.is_alive():
                raise RuntimeError("background dataset writer stopped before flush")
        self._raise_writer_error()

    def close(self) -> None:
        if self._writer_thread is None:
            return
        flush_error: Optional[BaseException] = None
        try:
            self.flush_pending_writes()
        except BaseException as exc:
            flush_error = exc
        self._write_queue.put(_STOP_WRITER)
        self._writer_thread.join(timeout=10.0)
        if self._writer_thread.is_alive():
            raise TimeoutError("background dataset writer did not stop")
        self._writer_thread = None
        if flush_error is not None:
            raise RuntimeError("failed to flush background dataset writer") from flush_error
        self._raise_writer_error()

    def start_episode(self, episode_index: int) -> None:
        self.flush_pending_writes()
        self.current_episode_rows = []
        self.current_image_records = []
        self.current_wrist_image_records = []
        self.current_tactile_left_records = []
        self.current_tactile_right_records = []
        if self.profile.stream_tactile:
            self.current_tactile_left_start = self._tactile_zarr_row_count(self.tactile_left_zarr_path)
            self.current_tactile_right_start = self._tactile_zarr_row_count(self.tactile_right_zarr_path)
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
        self.flush_pending_writes()
        if self.profile.stream_tactile:
            self._truncate_tactile_zarr(self.tactile_left_zarr_path, self.current_tactile_left_start)
            self._truncate_tactile_zarr(self.tactile_right_zarr_path, self.current_tactile_right_start)
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
        self.current_tactile_left_start = None
        self.current_tactile_right_start = None

    @staticmethod
    def _tactile_zarr_row_count(zarr_path: Path) -> int:
        if not zarr_path.exists():
            return 0
        import zarr

        arr = zarr.open(str(zarr_path), mode="r")
        if len(arr.shape) != 2:
            raise RuntimeError(f"invalid tactile zarr shape: {arr.shape}")
        return int(arr.shape[0])

    @staticmethod
    def _truncate_tactile_zarr(zarr_path: Path, row_count: Optional[int]) -> None:
        if row_count is None or not zarr_path.exists():
            return
        import zarr

        arr = zarr.open(str(zarr_path), mode="a")
        if len(arr.shape) != 2:
            raise RuntimeError(f"invalid tactile zarr shape: {arr.shape}")
        if int(arr.shape[0]) < int(row_count):
            raise RuntimeError(
                f"cannot restore tactile zarr to {row_count} rows; current shape is {arr.shape}"
            )
        arr.resize((int(row_count), int(arr.shape[1])))

    def _final_image_rel_path(self, episode_index: int, frame_index: int) -> Path:
        return Path("image") / self.cfg.image_name / f"episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"

    def _final_wrist_image_rel_path(self, episode_index: int, frame_index: int) -> Path:
        return Path("image") / self.cfg.wrist_image_name / f"episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"

    def _make_image_record(
        self,
        *,
        episode_index: int,
        frame_index: int,
        msg: Image | CompressedImage,
        out_path: Path,
        rel_path: Path,
        topic: str,
    ) -> dict[str, Any]:
        stamp = msg.header.stamp
        if isinstance(msg, CompressedImage):
            try:
                with PILImage.open(BytesIO(bytes(msg.data))) as encoded_image:
                    width, height = encoded_image.size
            except Exception as exc:
                raise ValueError(
                    f"failed to read compressed image metadata: {msg.format!r}"
                ) from exc
            source_encoding = f"compressed:{msg.format}"
            step = int(width) * 3
        else:
            height = int(msg.height)
            width = int(msg.width)
            source_encoding = str(msg.encoding)
            step = int(msg.step)
        return {
            "episode_index": int(episode_index),
            "record_frame_index": int(frame_index),
            "path": str(rel_path),
            "tmp_path": str(out_path.relative_to(self.root)),
            "height": int(height),
            "width": int(width),
            "encoding": "rgb8",
            "source_encoding": source_encoding,
            "step": step,
            "ros_stamp_sec": int(stamp.sec),
            "ros_stamp_nanosec": int(stamp.nanosec),
            "ros_stamp_float": ros_stamp_to_float_sec(stamp),
            "frame_id": str(msg.header.frame_id),
            "topic": topic,
        }

    def _save_image_to_dir(
        self,
        *,
        episode_index: int,
        frame_index: int,
        msg: Image | CompressedImage,
        tmp_dir: Path,
        rel_path: Path,
        topic: str,
        records: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any]]:
        out_path = tmp_dir / f"frame_{frame_index:06d}.png"
        self._write_png(msg, out_path)
        rec = self._make_image_record(
            episode_index=episode_index,
            frame_index=frame_index,
            msg=msg,
            out_path=out_path,
            rel_path=rel_path,
            topic=topic,
        )
        records.append(rec)
        return str(rel_path), rec

    def save_image(
        self,
        episode_index: int,
        frame_index: int,
        msg: Image | CompressedImage,
    ) -> tuple[str, dict[str, Any]]:
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
        if self.profile.stream_tactile:
            self._append_tactile_records_to_zarr(
                records=[left_rec],
                zarr_path=self.tactile_left_zarr_path,
                meta_path=self.tactile_left_meta_path,
                write_metadata=False,
            )
            self._append_tactile_records_to_zarr(
                records=[right_rec],
                zarr_path=self.tactile_right_zarr_path,
                meta_path=self.tactile_right_meta_path,
                write_metadata=False,
            )
            left_rec.pop("data", None)
            right_rec.pop("data", None)
        self.current_tactile_left_records.append(left_rec)
        self.current_tactile_right_records.append(right_rec)
        return left_rec, right_rec

    def write_frame_assets(
        self,
        *,
        episode_index: int,
        frame_index: int,
        timestamp: float,
        wall_time: float,
        image_msg: Image | CompressedImage,
        wrist_image_msg: Image,
        left_payload: dict[str, Any],
        left_dt: float,
        right_payload: dict[str, Any],
        right_dt: float,
    ) -> tuple[str, dict[str, Any], str, dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Schedule all large assets for one aligned frame as one ordered task."""
        if not self._async_frame_writes:
            image_rel_path, image_rec = self.save_image(
                episode_index, frame_index, image_msg
            )
            wrist_rel_path, wrist_rec = self.save_wrist_image(
                episode_index, frame_index, wrist_image_msg
            )
            left_rec, right_rec = self.append_tactile_pair(
                episode_index=episode_index,
                frame_index=frame_index,
                timestamp=timestamp,
                wall_time=wall_time,
                left_payload=left_payload,
                left_dt=left_dt,
                right_payload=right_payload,
                right_dt=right_dt,
            )
            return (
                image_rel_path,
                image_rec,
                wrist_rel_path,
                wrist_rec,
                left_rec,
                right_rec,
            )

        if self.current_tmp_image_dir is None or self.current_tmp_wrist_image_dir is None:
            raise RuntimeError("start_episode must be called before write_frame_assets")

        image_rel = self._final_image_rel_path(episode_index, frame_index)
        wrist_rel = self._final_wrist_image_rel_path(episode_index, frame_index)
        image_out = self.current_tmp_image_dir / f"frame_{frame_index:06d}.png"
        wrist_out = self.current_tmp_wrist_image_dir / f"frame_{frame_index:06d}.png"
        image_rec = self._make_image_record(
            episode_index=episode_index,
            frame_index=frame_index,
            msg=image_msg,
            out_path=image_out,
            rel_path=image_rel,
            topic=self.cfg.image_topic,
        )
        wrist_rec = self._make_image_record(
            episode_index=episode_index,
            frame_index=frame_index,
            msg=wrist_image_msg,
            out_path=wrist_out,
            rel_path=wrist_rel,
            topic=self.cfg.wrist_image_topic,
        )
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

        self._enqueue_write(
            _FrameWriteTask(
                image_msg=image_msg,
                image_out_path=image_out,
                wrist_image_msg=wrist_image_msg,
                wrist_image_out_path=wrist_out,
                left_record=left_rec,
                right_record=right_rec,
            )
        )
        self.current_image_records.append(image_rec)
        self.current_wrist_image_records.append(wrist_rec)
        self.current_tactile_left_records.append(left_rec)
        self.current_tactile_right_records.append(right_rec)
        return (
            str(image_rel),
            image_rec,
            str(wrist_rel),
            wrist_rec,
            left_rec,
            right_rec,
        )

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
        self._validate_dmtac_w_payload(payload)
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
            "capture_monotonic": float(payload.get("capture_monotonic", np.nan)),
            "transport_delay_sec": float(payload.get("transport_delay_sec", np.nan)),
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
        write_metadata: bool = True,
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
            arr[start : start + data.shape[0], :] = data
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

        for offset, rec in enumerate(records):
            rec["zarr_index"] = int(start + offset)
        if write_metadata:
            self._append_tactile_metadata(records, meta_path)

    @staticmethod
    def _append_tactile_metadata(records: list[dict[str, Any]], meta_path: Path) -> None:
        with open(meta_path, "a", encoding="utf-8") as f:
            for rec in records:
                meta = dict(rec)
                meta.pop("data", None)
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    def commit_episode(self, episode_index: int) -> None:
        self.flush_pending_writes()
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
        if self.profile.stream_tactile:
            self._append_tactile_metadata(self.current_tactile_left_records, self.tactile_left_meta_path)
            self._append_tactile_metadata(self.current_tactile_right_records, self.tactile_right_meta_path)
        else:
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
        if self._expected_dmtac_w_format is not None:
            mode, schema, packed_bytes = self._expected_dmtac_w_format
            episode_info.update(
                {
                    "tactile_output_mode": mode,
                    "tactile_schema_version": schema,
                    "tactile_packed_frame_bytes": packed_bytes,
                }
            )
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
        self.current_tactile_left_start = None
        self.current_tactile_right_start = None

    def flush_master(self) -> None:
        self.flush_pending_writes()
        if not self.rows:
            print("[WARN] no committed rows to write master parquet")
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_parquet(self.master_parquet, index=False)
        print(f"[SAVE] master parquet: {self.master_parquet}")
        print(f"[SAVE] total rows: {len(self.rows)}")
