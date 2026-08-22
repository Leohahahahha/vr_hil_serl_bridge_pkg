from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    from .action_space import normalize_quat_xyzw, pose7_to_state7, target_pose_to_relative_action7
except ImportError:
    from action_space import normalize_quat_xyzw, pose7_to_state7, target_pose_to_relative_action7  # type: ignore


@dataclass(frozen=True)
class ObservationSyncThresholds:
    max_image_dt_sec: float
    max_wrist_image_dt_sec: float
    max_action_dt_sec: float
    max_state_dt_sec: float
    max_vr_dt_sec: float
    max_tactile_dt_sec: float


@dataclass
class ObservationFrame:
    image_item: Any
    wrist_image_item: Any
    wrist_camera_info_item: Any
    command_item: Any
    state_item: Any
    vr_pose_item: Any
    enabled_item: Any
    joystick_y_item: Any
    tactile_left_item: Any
    tactile_right_item: Any

    tactile_left_payload: dict[str, Any]
    tactile_right_payload: dict[str, Any]
    command_event: dict[str, Any]
    state_event: dict[str, Any]
    state_fields: dict[str, Any]
    action8: np.ndarray
    target_gripper_width: float
    observation_state7: np.ndarray
    relative_action7: np.ndarray

    vr_payload: dict[str, Any]
    joystick_y_payload: dict[str, Any]
    enabled_val: Optional[bool]

    ok_vr: bool
    dt_image: Optional[float]
    dt_wrist_image: Optional[float]
    dt_action: Optional[float]
    dt_state: Optional[float]
    dt_vr: Optional[float]
    dt_tactile_left: Optional[float]
    dt_tactile_right: Optional[float]
    reason_image: str
    reason_wrist_image: str
    reason_action: str
    reason_state: str
    reason_vr: str
    reason_tactile_left: str
    reason_tactile_right: str


@dataclass
class ObservationBuildResult:
    ok: bool
    reason: str
    frame: Optional[ObservationFrame] = None


def ros_stamp_to_float_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


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


def valid_pair(name: str, item: Any, t_frame: float, max_dt: float) -> tuple[bool, Optional[float], str]:
    if item is None:
        return False, None, f"missing_{name}"
    dt = float(item.t - t_frame)
    if abs(dt) > max_dt:
        return False, dt, f"stale_{name}_dt_{dt:.3f}"
    return True, dt, "ok"


def camera_info_to_record(item: Any, t_frame: float) -> dict[str, Any]:
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
        f"{prefix}.capture_monotonic": float(rec.get("capture_monotonic", np.nan)),
        f"{prefix}.transport_delay_sec": float(rec.get("transport_delay_sec", np.nan)),
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


def build_observation_frame(
    *,
    bundle: dict[str, Any],
    t_frame: float,
    thresholds: ObservationSyncThresholds,
    wait_for_motion_enable: bool,
) -> ObservationBuildResult:
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

    ok_image, dt_image, reason_image = valid_pair("image", image_item, t_frame, thresholds.max_image_dt_sec)
    ok_wrist_image, dt_wrist_image, reason_wrist_image = valid_pair(
        "wrist_image", wrist_image_item, t_frame, thresholds.max_wrist_image_dt_sec
    )
    ok_action, dt_action, reason_action = valid_pair("action", command_item, t_frame, thresholds.max_action_dt_sec)
    ok_state, dt_state, reason_state = valid_pair("state", state_item, t_frame, thresholds.max_state_dt_sec)
    ok_vr, dt_vr, reason_vr = valid_pair("vr", vr_pose_item, t_frame, thresholds.max_vr_dt_sec)
    ok_tactile_left, dt_tactile_left, reason_tactile_left = valid_pair(
        "tactile_left", tactile_left_item, t_frame, thresholds.max_tactile_dt_sec
    )
    ok_tactile_right, dt_tactile_right, reason_tactile_right = valid_pair(
        "tactile_right", tactile_right_item, t_frame, thresholds.max_tactile_dt_sec
    )
  
    enabled_val = None
    if enabled_item is not None:
        enabled_val = bool(enabled_item.payload.get("enabled", False))
    if wait_for_motion_enable and not enabled_val:
        return ObservationBuildResult(ok=False, reason="enabled false or missing")

    if not (ok_image and ok_wrist_image and ok_action and ok_state and ok_tactile_left and ok_tactile_right):
        return ObservationBuildResult(
            ok=False,
            reason=(
                f"image={reason_image} wrist_image={reason_wrist_image} "
                f"action={reason_action} state={reason_state} "
                f"tactile_left={reason_tactile_left} tactile_right={reason_tactile_right}"
            ),
        )

    tactile_left_payload = tactile_left_item.payload
    tactile_right_payload = tactile_right_item.payload
    if tactile_left_payload["data_len"] != tactile_right_payload["data_len"]:
        return ObservationBuildResult(
            ok=False,
            reason=(
                "left/right tactile data length mismatch: "
                f"{tactile_left_payload['data_len']} != {tactile_right_payload['data_len']}"
            ),
        )

    command_event = command_item.payload
    state_event = state_item.payload
    action8 = command_event.get("action8")
    if action8 is None:
        return ObservationBuildResult(ok=False, reason="command_event has no action8")
    action8 = np.asarray(action8, dtype=np.float32).reshape(-1)
    if action8.shape[0] != 8:
        return ObservationBuildResult(ok=False, reason=f"action8 has wrong shape: {action8.shape}")
    action8[3:7] = normalize_quat_xyzw(action8[3:7])

    state_fields = extract_state_fields(state_event)
    current_pose7 = state_fields.get("robot.ee_pose")
    current_gripper_width = state_fields.get("robot.gripper_pos")
    if current_pose7 is None or current_gripper_width is None:
        return ObservationBuildResult(ok=False, reason="robot state missing ee pose or gripper width")

    try:
        target_gripper_width = float(
            command_event.get("target_gripper_width", command_event.get("target_gripper", action8[7]))
        )
        observation_state7 = pose7_to_state7(
            np.asarray(current_pose7, dtype=np.float32),
            current_gripper_width,
        )
        relative_action7 = target_pose_to_relative_action7(
            target_pose7=action8[:7],
            current_pose7=np.asarray(current_pose7, dtype=np.float32),
            target_gripper_width=target_gripper_width,
        )
    except Exception as e:
        return ObservationBuildResult(ok=False, reason=f"state/action conversion failed: {repr(e)}")

    frame = ObservationFrame(
        image_item=image_item,
        wrist_image_item=wrist_image_item,
        wrist_camera_info_item=wrist_camera_info_item,
        command_item=command_item,
        state_item=state_item,
        vr_pose_item=vr_pose_item,
        enabled_item=enabled_item,
        joystick_y_item=joystick_y_item,
        tactile_left_item=tactile_left_item,
        tactile_right_item=tactile_right_item,
        tactile_left_payload=tactile_left_payload,
        tactile_right_payload=tactile_right_payload,
        command_event=command_event,
        state_event=state_event,
        state_fields=state_fields,
        action8=action8,
        target_gripper_width=target_gripper_width,
        observation_state7=observation_state7,
        relative_action7=relative_action7,
        vr_payload=vr_pose_item.payload if vr_pose_item is not None else {},
        joystick_y_payload=joystick_y_item.payload if joystick_y_item is not None else {},
        enabled_val=enabled_val,
        ok_vr=ok_vr,
        dt_image=dt_image,
        dt_wrist_image=dt_wrist_image,
        dt_action=dt_action,
        dt_state=dt_state,
        dt_vr=dt_vr,
        dt_tactile_left=dt_tactile_left,
        dt_tactile_right=dt_tactile_right,
        reason_image=reason_image,
        reason_wrist_image=reason_wrist_image,
        reason_action=reason_action,
        reason_state=reason_state,
        reason_vr=reason_vr,
        reason_tactile_left=reason_tactile_left,
        reason_tactile_right=reason_tactile_right,
    )
    return ObservationBuildResult(ok=True, reason="ok", frame=frame)
