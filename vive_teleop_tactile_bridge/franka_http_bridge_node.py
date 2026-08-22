#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import json
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from .http_client import FrankaHttpClient, HttpConfig


def pose_msg_to_numpy(msg: PoseStamped) -> np.ndarray:
    q = np.array([
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    ], dtype=float)
    return np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
        q[0], q[1], q[2], q[3],
    ], dtype=float)


def pose_msg_to_pr(msg: PoseStamped) -> Tuple[np.ndarray, R]:
    q = np.array([
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    ], dtype=float)
    p = np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
    ], dtype=float)
    return p, R.from_quat(q)


def pose7_to_pr(pose7: np.ndarray) -> Tuple[np.ndarray, R]:
    pose7 = np.asarray(pose7, dtype=float).reshape(7)
    return pose7[:3].copy(), R.from_quat(pose7[3:])


def pr_to_pose7(p: np.ndarray, r: R) -> np.ndarray:
    q = r.as_quat()
    return np.array([p[0], p[1], p[2], q[0], q[1], q[2], q[3]], dtype=float)


def normalize_pose7(pose7: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose7, dtype=float).reshape(7).copy()
    q = pose[3:7]
    q_norm = float(np.linalg.norm(q))
    if np.isfinite(q_norm) and q_norm > 1e-8:
        pose[3:7] = q / q_norm
    else:
        pose[3:7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=float)
    return pose


def state_to_pose7(state: Any) -> Optional[np.ndarray]:
    if not isinstance(state, dict):
        return None
    pose = state.get('pose', state.get('ee_pose', state.get('pos')))
    if pose is None:
        return None
    try:
        return normalize_pose7(np.asarray(pose, dtype=float).reshape(7))
    except Exception:
        return None


def lowpass_pose(prev_pose: np.ndarray, new_pose: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = new_pose.copy()
    out[:3] = (1.0 - alpha) * prev_pose[:3] + alpha * new_pose[:3]

    r_prev = R.from_quat(prev_pose[3:])
    r_new = R.from_quat(new_pose[3:])
    delta = (r_prev.inv() * r_new).as_rotvec()
    r_interp = r_prev * R.from_rotvec(alpha * delta)
    out[3:] = r_interp.as_quat()
    return out


def pose_delta(p_prev: np.ndarray, r_prev: R, p_now: np.ndarray, r_now: R) -> Tuple[np.ndarray, R]:
    """Same delta convention as the old MoveIt Servo teleop node."""
    dp = r_prev.inv().apply(p_now - p_prev)
    dr = r_prev.inv() * r_now
    return dp, dr


@dataclass
class BridgeParams:
    server_url: str = 'http://192.168.1.10:5000'
    request_timeout_sec: float = 0.12
    publish_rate: float = 90.0

    enable_robot_commands: bool = False

    input_pose_topic: str = '/vr/right_controller/pose_hmd'

    # Motion enable
    use_grip_button_for_motion_enable: bool = True
    trigger_enable_threshold: float = 0.60
    trigger_disable_threshold: float = 0.40

    # Toggle gripper
    trigger_toggle_threshold: float = 0.75
    gripper_control_mode: str = 'trigger_toggle'
    gripper_width_command_topic: str = '/hilserl/gripper_width_cmd'
    gripper_width_min: float = 0.0
    gripper_width_max: float = 0.085
    gripper_initial_width: float = 0.085
    joystick_gripper_deadzone: float = 0.05
    joystick_gripper_width_rate: float = 0.04
    joystick_gripper_send_hz: float = 20.0
    joystick_gripper_invert: bool = False

    # Incremental per-frame gains, tuned for HTTP pose integration rather than Twist velocities
    delta_linear_gain_xy: float = 0.20
    delta_linear_gain_z: float = 0.10
    delta_angular_gain: float = 0.15
    enable_rotation: bool = False
    pose_lowpass_alpha: float = 0.25
    translation_deadzone: float = 0.002
    rotation_deadzone: float = 0.02

    speed_scale_fast: float = 1.0
    speed_scale_slow: float = 0.35
    speed_mode_transition_time: float = 0.25

    vr_to_robot_rotation: list = None
    workspace_min: list = None
    workspace_max: list = None

    activate_gripper_on_start: bool = False
    clear_error_on_start: bool = False

    # A button is fixed as joint reset in this version
    a_button_reanchor: bool = False
    a_button_joint_reset: bool = True

    planning_frame: str = 'fr3_link0'
    debug_target_topic: str = '/vr_bridge/target_pose'
    debug_enabled_topic: str = '/vr_bridge/enabled'
    debug_speed_topic: str = '/vr_bridge/speed_scale'

    # Data-collection logging outputs. These are used by the raw recorder.
    sent_action_topic: str = '/hilserl/sent_action8'
    command_event_topic: str = '/hilserl/command_event_json'
    http_status_topic: str = '/hilserl/http_status_json'
    robot_state_topic: str = '/hilserl/robot_state_json'
    enable_command_logging: bool = True
    enable_state_poll: bool = True
    state_poll_hz: float = 30.0
    state_poll_timeout_sec: float = 0.12

    def __post_init__(self):
        if self.vr_to_robot_rotation is None:
            self.vr_to_robot_rotation = [
                1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0,
            ]
        if self.workspace_min is None:
            self.workspace_min = [0.30, -0.30, 0.08]
        if self.workspace_max is None:
            self.workspace_max = [0.75, 0.30, 0.45]


class FrankaHttpBridgeNode(Node):
    """Incremental VR teleop bridge with trigger-toggle gripper over HTTP.

    In this version:
    - trigger toggles gripper open/close
    - grip_button enables/disables robot motion
    - A button is fixed to joint reset
    """

    def __init__(self):
        super().__init__('tactile_franka_http_bridge')
        self.params = self._load_params()
        self.client = FrankaHttpClient(HttpConfig(
            server_url=self.params.server_url,
            timeout_sec=self.params.request_timeout_sec,
        ))

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(PoseStamped, self.params.input_pose_topic, self._pose_cb, qos)
        self.create_subscription(Float32, '/vr/right_controller/trigger', self._trigger_cb, qos)
        self.create_subscription(Float32, '/vr/right_controller/joystick_y', self._joystick_cb, qos)
        self.create_subscription(Bool, '/vr/right_controller/a_button', self._a_button_cb, qos)
        self.create_subscription(Bool, '/vr/right_controller/grip_button', self._grip_cb, qos)
        if self.params.gripper_width_command_topic:
            self.create_subscription(
                Float32,
                self.params.gripper_width_command_topic,
                self._gripper_width_cmd_cb,
                qos,
            )

        self.target_pub = self.create_publisher(PoseStamped, self.params.debug_target_topic, 10)
        self.enabled_pub = self.create_publisher(Bool, self.params.debug_enabled_topic, 10)
        self.speed_pub = self.create_publisher(Float32, self.params.debug_speed_topic, 10)

        # These publishers make the existing teleop node recordable.
        # The recorder should learn from sent_action8 / command_event_json,
        # not from the raw VR target pose.
        self.sent_action_pub = self.create_publisher(Float32MultiArray, self.params.sent_action_topic, 10)
        self.command_event_pub = self.create_publisher(String, self.params.command_event_topic, 10)
        self.http_status_pub = self.create_publisher(String, self.params.http_status_topic, 10)
        self.robot_state_pub = self.create_publisher(String, self.params.robot_state_topic, 10)

        self.latest_pose: Optional[PoseStamped] = None
        self.latest_trigger: float = 0.0
        self.latest_joystick_y: float = 0.0
        self.latest_a_button: bool = False
        self.latest_grip_button: bool = False

        self.last_a_button: bool = False
        self.last_trigger_toggle_pressed: bool = False
        self.gripper_closed: bool = False
        self.gripper_width: float = float(np.clip(
            self.params.gripper_initial_width,
            self.params.gripper_width_min,
            self.params.gripper_width_max,
        ))
        self.last_gripper_width_send_time: float = 0.0
        self._did_read_initial_gripper_width: bool = False

        self.enabled: bool = False
        self.prev_vr_p: Optional[np.ndarray] = None
        self.prev_vr_r: Optional[R] = None

        self.cmd_p: Optional[np.ndarray] = None
        self.cmd_r: Optional[R] = None
        self.current_pose_cmd: Optional[np.ndarray] = None
        self.last_robot_pose7: Optional[np.ndarray] = None

        self.speed_scale: float = float(self.params.speed_scale_fast)
        self.speed_scale_target: float = float(self.params.speed_scale_fast)
        self.speed_mode_ramp_rate: float = 1.0 / max(self.params.speed_mode_transition_time, 1e-3)

        self.R_map = np.asarray(self.params.vr_to_robot_rotation, dtype=float).reshape(3, 3)
        self.workspace_min = np.asarray(self.params.workspace_min, dtype=float).reshape(3)
        self.workspace_max = np.asarray(self.params.workspace_max, dtype=float).reshape(3)

        self._did_startup_actions = False
        self.last_publish_time = self.get_clock().now()

        self._state_poll_stop = threading.Event()
        self._state_poll_thread: Optional[threading.Thread] = None
        if self.params.enable_state_poll:
            self._state_poll_thread = threading.Thread(target=self._state_poll_loop, daemon=True)
            self._state_poll_thread.start()

        self.create_timer(1.0 / self.params.publish_rate, self._timer_cb)
        self.get_logger().info(
            f'Franka HTTP bridge started | server={self.params.server_url} | rate={self.params.publish_rate:.1f} Hz | '
            f'pose_topic={self.params.input_pose_topic}'
        )

    def _load_params(self) -> BridgeParams:
        defaults = BridgeParams()
        values = {}
        for field_name in defaults.__dataclass_fields__:
            default = getattr(defaults, field_name)
            self.declare_parameter(field_name, default)
            values[field_name] = self.get_parameter(field_name).value
        return BridgeParams(**values)

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.latest_pose = msg

    def _trigger_cb(self, msg: Float32) -> None:
        self.latest_trigger = float(msg.data)

    def _joystick_cb(self, msg: Float32) -> None:
        self.latest_joystick_y = float(msg.data)

    def _a_button_cb(self, msg: Bool) -> None:
        self.latest_a_button = bool(msg.data)

    def _grip_cb(self, msg: Bool) -> None:
        self.latest_grip_button = bool(msg.data)

    def _gripper_width_cmd_cb(self, msg: Float32) -> None:
        if not self.params.enable_robot_commands:
            self.get_logger().warn(
                "gripper command ignored: enable_robot_commands is false"
            )
            return
        width = float(np.clip(
            msg.data,
            self.params.gripper_width_min,
            self.params.gripper_width_max,
        ))
        self.gripper_width = width
        t_send_start = time.monotonic()
        http_ok = False
        http_error = ''
        try:
            self.client.move_gripper_width(width)
            http_ok = True
            self.last_gripper_width_send_time = time.monotonic()
        except Exception as exc:
            http_error = repr(exc)
            self.get_logger().warn(f'gripper_width_cmd failed: {exc}')
        t_send_end = time.monotonic()
        self._publish_command_event(
            pose7=self._command_event_pose7(),
            pose_commanded=False,
            gripper_commanded=True,
            http_route="/move_gripper",
            http_ok=http_ok,
            http_error=http_error,
            t_send_start=t_send_start,
            t_send_end=t_send_end,
        )

    def _run_startup_actions_once(self) -> None:
        if self._did_startup_actions:
            return
        self._did_startup_actions = True
        try:
            if self.params.clear_error_on_start:
                self.client.clear_error()
                self.get_logger().info('clearerr sent on startup')
            if self.params.activate_gripper_on_start:
                self.client.activate_gripper()
                self.get_logger().info('activate_gripper sent on startup')
            try:
                self.gripper_width = float(np.clip(
                    self.client.get_gripper_width(),
                    self.params.gripper_width_min,
                    self.params.gripper_width_max,
                ))
                self._did_read_initial_gripper_width = True
                self.get_logger().info(f'initial gripper_width={self.gripper_width:.4f} m')
            except Exception as exc:
                self.get_logger().warn(f'Could not read initial gripper_width, using configured value: {exc}')
        except Exception as exc:
            self.get_logger().warn(f'Startup action failed: {exc}')

    def _start_motion(self) -> None:
        if not self.params.enable_robot_commands:
            self.get_logger().warn(
                "motion enable ignored: enable_robot_commands is false"
            )
            return
        if self.latest_pose is None:
            self.get_logger().warn(f'Enable requested, but no {self.params.input_pose_topic} received yet')
            return
        try:
            robot_pose = self.client.get_pose()
            self.last_robot_pose7 = normalize_pose7(robot_pose)
            self.cmd_p, self.cmd_r = pose7_to_pr(robot_pose)
            self.current_pose_cmd = robot_pose.copy()
            self.prev_vr_p, self.prev_vr_r = pose_msg_to_pr(self.latest_pose)
            self.enabled = True
            self.get_logger().info('VR motion enabled (incremental mode)')
        except Exception as exc:
            self.enabled = False
            self.get_logger().error(f'Failed to read /getpos from Franka server: {exc}')

    def _stop_motion(self) -> None:
        if self.enabled:
            self.get_logger().info('VR motion disabled')
        self.enabled = False
        self.prev_vr_p = None
        self.prev_vr_r = None

    def _maybe_update_enable_state(self) -> None:
        if self.params.use_grip_button_for_motion_enable:
            if (not self.enabled) and self.latest_grip_button:
                self._start_motion()
            elif self.enabled and (not self.latest_grip_button):
                self._stop_motion()
        else:
            if (not self.enabled) and self.latest_trigger > self.params.trigger_enable_threshold:
                self._start_motion()
            elif self.enabled and self.latest_trigger < self.params.trigger_disable_threshold:
                self._stop_motion()

    def _maybe_handle_a_button(self) -> None:
        current = self.latest_a_button
        rising = current and not self.last_a_button
        self.last_a_button = current
        if not rising:
            return

        try:
            self.client.joint_reset()
            self.get_logger().warn('A button -> jointreset')
        except Exception as exc:
            self.get_logger().error(f'jointreset failed: {exc}')

    def _maybe_handle_gripper(self, dt: float) -> Optional[dict[str, Any]]:
        if self.params.gripper_control_mode == 'joystick_width':
            return self._maybe_handle_gripper_width(dt)

        pressed = self.latest_trigger > self.params.trigger_toggle_threshold

        if pressed and (not self.last_trigger_toggle_pressed):
            t_send_start = time.monotonic()
            http_ok = False
            http_error = ''
            http_route = "/open_gripper" if self.gripper_closed else "/close_gripper"
            try:
                if self.gripper_closed:
                    self.client.open_gripper()
                    self.gripper_closed = False
                    self.get_logger().info('open_gripper (toggle)')
                else:
                    self.client.close_gripper()
                    self.gripper_closed = True
                    self.get_logger().info('close_gripper (toggle)')
                http_ok = True
            except Exception as exc:
                http_error = repr(exc)
                self.get_logger().warn(f'Gripper toggle failed: {exc}')
            t_send_end = time.monotonic()
            self.last_trigger_toggle_pressed = pressed
            return {
                "http_route": http_route,
                "http_ok": http_ok,
                "http_error": http_error,
                "t_send_start": t_send_start,
                "t_send_end": t_send_end,
            }

        self.last_trigger_toggle_pressed = pressed
        return None

    def _maybe_handle_gripper_width(self, dt: float) -> Optional[dict[str, Any]]:
        joystick = float(self.latest_joystick_y)
        if abs(joystick) < self.params.joystick_gripper_deadzone:
            return None

        sign = -1.0 if self.params.joystick_gripper_invert else 1.0
        width = self.gripper_width + sign * joystick * self.params.joystick_gripper_width_rate * max(dt, 0.0)
        width = float(np.clip(width, self.params.gripper_width_min, self.params.gripper_width_max))
        if abs(width - self.gripper_width) < 1e-6:
            return None

        self.gripper_width = width
        now = time.monotonic()
        min_period = 1.0 / max(float(self.params.joystick_gripper_send_hz), 1e-3)
        if now - self.last_gripper_width_send_time < min_period:
            return None

        t_send_start = time.monotonic()
        http_ok = False
        http_error = ''
        try:
            self.client.move_gripper_width(width)
            http_ok = True
            self.last_gripper_width_send_time = time.monotonic()
        except Exception as exc:
            http_error = repr(exc)
            self.get_logger().warn(f'continuous gripper command failed: {exc}')
        t_send_end = time.monotonic()
        return {
            "http_route": "/move_gripper",
            "http_ok": http_ok,
            "http_error": http_error,
            "t_send_start": t_send_start,
            "t_send_end": t_send_end,
        }

    def _compute_incremental_command(self, dt: float) -> Optional[np.ndarray]:
        if (not self.enabled) or (self.latest_pose is None) or (dt < 1e-4):
            return None
        if self.prev_vr_p is None or self.prev_vr_r is None or self.cmd_p is None or self.cmd_r is None:
            return None

        p_now, r_now = pose_msg_to_pr(self.latest_pose)
        dp_vr, dr_vr = pose_delta(self.prev_vr_p, self.prev_vr_r, p_now, r_now)
        self.prev_vr_p, self.prev_vr_r = p_now.copy(), r_now

        rotvec_vr = dr_vr.as_rotvec()

        if np.linalg.norm(dp_vr) < self.params.translation_deadzone:
            dp_vr[:] = 0.0
        if np.linalg.norm(rotvec_vr) < self.params.rotation_deadzone:
            rotvec_vr[:] = 0.0

        dp_step_local = np.array([
            self.params.delta_linear_gain_xy * dp_vr[0],
            self.params.delta_linear_gain_xy * dp_vr[1],
            self.params.delta_linear_gain_z * dp_vr[2],
        ], dtype=float) * self.speed_scale

        dp_step_robot = self.R_map @ dp_step_local
        self.cmd_p = self.cmd_p + dp_step_robot
        self.cmd_p = np.clip(self.cmd_p, self.workspace_min, self.workspace_max)

        if self.params.enable_rotation:
            rotvec_step_robot = self.R_map @ (self.params.delta_angular_gain * rotvec_vr * self.speed_scale)
            self.cmd_r = self.cmd_r * R.from_rotvec(rotvec_step_robot)

        return pr_to_pose7(self.cmd_p, self.cmd_r)

    def _publish_debug(self) -> None:
        msg_enabled = Bool()
        msg_enabled.data = self.enabled
        self.enabled_pub.publish(msg_enabled)

        msg_speed = Float32()
        msg_speed.data = float(self.speed_scale)
        self.speed_pub.publish(msg_speed)

        if self.current_pose_cmd is None:
            return

        msg_pose = PoseStamped()
        msg_pose.header.stamp = self.get_clock().now().to_msg()
        msg_pose.header.frame_id = self.params.planning_frame
        msg_pose.pose.position.x = float(self.current_pose_cmd[0])
        msg_pose.pose.position.y = float(self.current_pose_cmd[1])
        msg_pose.pose.position.z = float(self.current_pose_cmd[2])
        msg_pose.pose.orientation.x = float(self.current_pose_cmd[3])
        msg_pose.pose.orientation.y = float(self.current_pose_cmd[4])
        msg_pose.pose.orientation.z = float(self.current_pose_cmd[5])
        msg_pose.pose.orientation.w = float(self.current_pose_cmd[6])
        self.target_pub.publish(msg_pose)

    def _publish_json(self, pub, payload: dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        pub.publish(msg)

    def _command_event_pose7(self) -> np.ndarray:
        if self.current_pose_cmd is not None:
            return normalize_pose7(self.current_pose_cmd)
        if self.last_robot_pose7 is not None:
            return normalize_pose7(self.last_robot_pose7)
        try:
            pose = self.client.get_pose()
            self.last_robot_pose7 = normalize_pose7(pose)
            return self.last_robot_pose7.copy()
        except Exception as exc:
            self.get_logger().warn(f'Could not read pose for command_event, using workspace center: {exc}')
            center = 0.5 * (self.workspace_min + self.workspace_max)
            return np.asarray([center[0], center[1], center[2], 0.0, 0.0, 0.0, 1.0], dtype=float)

    def _publish_command_event(
        self,
        *,
        pose7: np.ndarray,
        pose_commanded: bool,
        gripper_commanded: bool,
        http_route: str,
        http_ok: bool,
        http_error: str,
        t_send_start: float,
        t_send_end: float,
    ) -> None:
        """Publish the exact robot command attempted through HTTP.

        This is the action label used by the recorder. It is emitted from the same
        code path that sends /pose or /move_gripper, so action labels follow real
        robot commands rather than raw controller button states.
        """
        if not self.params.enable_command_logging:
            return

        pose7 = normalize_pose7(np.asarray(pose7, dtype=np.float32))

        if self.params.gripper_control_mode == 'joystick_width':
            target_gripper = np.float32(self.gripper_width)
        else:
            target_gripper = np.float32(1.0 if self.gripper_closed else -1.0)
        action8 = np.concatenate([pose7, np.asarray([target_gripper], dtype=np.float32)], axis=0)

        arr_msg = Float32MultiArray()
        arr_msg.data = action8.astype(np.float32).tolist()
        self.sent_action_pub.publish(arr_msg)

        event = {
            "type": "robot_command",
            "command_type": "robot_command",
            "pose_commanded": bool(pose_commanded),
            "gripper_commanded": bool(gripper_commanded),
            "t_event": float(t_send_start),
            "wall_time": float(time.time()),
            "action8": action8.astype(float).tolist(),
            "pose7": pose7.astype(float).tolist(),
            "target_gripper": float(target_gripper),
            "target_gripper_width": float(self.gripper_width),
            "gripper_control_mode": self.params.gripper_control_mode,
            "enabled": bool(self.enabled),
            "gripper_closed": bool(self.gripper_closed),
            "workspace_min": np.asarray(self.workspace_min, dtype=float).tolist(),
            "workspace_max": np.asarray(self.workspace_max, dtype=float).tolist(),
            "workspace_clipped": bool(
                np.any(pose7[:3] <= self.workspace_min + 1e-8) or
                np.any(pose7[:3] >= self.workspace_max - 1e-8)
            ),
            "http": {
                "route": str(http_route),
                "ok": bool(http_ok),
                "error": str(http_error),
                "t_send_start": float(t_send_start),
                "t_send_end": float(t_send_end),
                "latency_ms": float(1000.0 * (t_send_end - t_send_start)),
            },
            "source": {
                "node": self.get_name(),
                "input_pose_topic": self.params.input_pose_topic,
                "planning_frame": self.params.planning_frame,
            },
        }
        self._publish_json(self.command_event_pub, event)
        self._publish_json(self.http_status_pub, event["http"])

    def _state_poll_loop(self) -> None:
        """Poll franka_server /getstate in a background thread.

        Running this in the bridge process ensures the recorder sees robot
        state timestamps from the same host clock as command_event timestamps.
        """
        hz = max(float(self.params.state_poll_hz), 1e-3)
        period = 1.0 / hz
        while not self._state_poll_stop.is_set():
            t0 = time.monotonic()
            ok = False
            state = None
            error = ""
            try:
                state = self.client.post('/getstate', timeout_sec=float(self.params.state_poll_timeout_sec))
                ok = True
                pose7 = state_to_pose7(state)
                if pose7 is not None:
                    self.last_robot_pose7 = pose7
            except Exception as exc:
                error = repr(exc)

            t1 = time.monotonic()
            payload = {
                "type": "robot_state",
                "ok": bool(ok),
                "error": error,
                "t_query_start": float(t0),
                "t_query_end": float(t1),
                "t_query_mid": float(0.5 * (t0 + t1)),
                "wall_time": float(time.time()),
                "latency_ms": float(1000.0 * (t1 - t0)),
                "state": state if isinstance(state, dict) else {"raw": state},
            }
            self._publish_json(self.robot_state_pub, payload)

            sleep_s = period - (time.monotonic() - t0)
            if sleep_s > 0:
                self._state_poll_stop.wait(sleep_s)

    def stop_background_threads(self) -> None:
        self._state_poll_stop.set()
        if self._state_poll_thread is not None:
            self._state_poll_thread.join(timeout=1.0)

    def _timer_cb(self) -> None:
        if not self.params.enable_robot_commands:
            if self.enabled:
                self._stop_motion()
            self._publish_debug()
            return
        self._run_startup_actions_once()

        now = self.get_clock().now()
        dt = max((now - self.last_publish_time).nanoseconds * 1e-9, 1e-3)
        self.last_publish_time = now

        self._maybe_update_enable_state()
        self._maybe_handle_a_button()
        gripper_command = self._maybe_handle_gripper(dt)

        step = self.speed_mode_ramp_rate * dt
        delta = self.speed_scale_target - self.speed_scale
        self.speed_scale = float(np.clip(
            self.speed_scale + np.clip(delta, -step, step),
            min(self.params.speed_scale_slow, self.params.speed_scale_fast),
            max(self.params.speed_scale_slow, self.params.speed_scale_fast)
        ))

        target_pose = self._compute_incremental_command(dt)
        if target_pose is None:
            if gripper_command is not None:
                self._publish_command_event(
                    pose7=self._command_event_pose7(),
                    pose_commanded=False,
                    gripper_commanded=True,
                    http_route=str(gripper_command["http_route"]),
                    http_ok=bool(gripper_command["http_ok"]),
                    http_error=str(gripper_command["http_error"]),
                    t_send_start=float(gripper_command["t_send_start"]),
                    t_send_end=float(gripper_command["t_send_end"]),
                )
            self._publish_debug()
            return

        if self.current_pose_cmd is None:
            self.current_pose_cmd = target_pose.copy()
        else:
            self.current_pose_cmd = lowpass_pose(self.current_pose_cmd, target_pose, self.params.pose_lowpass_alpha)
            self.cmd_p, self.cmd_r = pose7_to_pr(self.current_pose_cmd)

        t_send_start = time.monotonic()
        http_ok = False
        http_error = ''
        try:
            self.client.command_pose(self.current_pose_cmd)
            http_ok = True
            self.last_robot_pose7 = normalize_pose7(self.current_pose_cmd)
        except Exception as exc:
            http_error = repr(exc)
            self.get_logger().error(f'/pose command failed: {exc}')
            self._stop_motion()
        t_send_end = time.monotonic()

        self._publish_command_event(
            pose7=self.current_pose_cmd,
            pose_commanded=True,
            gripper_commanded=gripper_command is not None,
            http_route="/pose",
            http_ok=http_ok,
            http_error=http_error,
            t_send_start=t_send_start,
            t_send_end=t_send_end,
        )

        self._publish_debug()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FrankaHttpBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop_background_threads()
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
