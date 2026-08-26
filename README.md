# vive_teleop_tactile_bridge

ROS 2 `ament_python` package for Ubuntu 24.04 that:

1. reads HTC/OpenVR pose and buttons,
2. reuses your existing topic names,
3. converts VR deltas into absolute end-effector poses,
4. sends those poses to the HIL-SERL / SERL `franka_server.py` over HTTP.

## Included nodes

- `tactile_vr_tracker_node`
  - publishes `/vr/right_controller/pose_hmd`
  - publishes `/vr/right_controller/trigger`
  - publishes `/vr/right_controller/joystick_y`
  - publishes `/vr/right_controller/a_button`
  - publishes `/vr/right_controller/grip_button`
- `tactile_vr_converter_node` / `tactile_franka_http_bridge_node`
  - subscribes to the same topics
  - sends `/getpos`, `/pose`, `/move_gripper`, `/clearerr`, `/jointreset`

## Control mapping

- Hold `grip_button`:
  - start clutch / enable motion
- Release `grip_button`:
  - stop motion
- `trigger`:
  - legacy open/close toggle if `gripper_control_mode: trigger_toggle`
- `joystick_y`:
  - continuous Robotiq width command if `gripper_control_mode: joystick_width`
- `/hilserl/gripper_width_cmd`:
  - direct continuous width command, `std_msgs/Float32`, units are meters
- `a_button` rising edge:
  - re-anchor VR pose to the current robot pose

## Build

```bash
cd ~/ros2_ws/src
cp -r /path/to/vr_hil_serl_bridge_pkg ./vive_teleop_tactile_bridge
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select vive_teleop_tactile_bridge
source install/setup.bash
```

## Run

Only bridge:

```bash
ros2 launch vive_teleop_tactile_bridge bridge_only.launch.py \
  params_file:=/absolute/path/to/bridge.params.yaml
```

Full stack (OpenVR + bridge), old launch style preserved:

```bash
ros2 launch vive_teleop_tactile_bridge vr_teleop.launch.py
```

or:

```bash
ros2 launch vive_teleop_tactile_bridge full_stack.launch.py
```

## DM-Tac W packed acquisition

The default `output_mode: shear_depth` requests only `getShear()` and
`getDepth()` from each sensor. It publishes packed schema 3 (921,600 bytes per
side), with channel data ordered as shear then depth. Set `output_mode: full`
in both DM-Tac W YAML files only when the five legacy SDK outputs are required;
that mode publishes schema 1 (1,920,000 bytes per side). The worker targets
30 Hz and reads the left/right sensors concurrently. Getters within one
physical sensor remain serial because SDK 0.1.4 does not expose an atomic
snapshot API.

The bridge publishes one latest-only packed message per side:

    /dmtac/left/packed_frame
    /dmtac/right/packed_frame

Each packed message is a latest-only `sensor_msgs/Image` (`8UC1`). Its header
timestamp is the SDK getter-group capture midpoint. The raw recorder maps that
timestamp into its monotonic clock domain before selecting the nearest tactile
frame for the 10 Hz trajectory.

After rebuilding and sourcing the ROS 2 workspace, start the tactile bridge and
recorder in separate terminals:

```bash
ros2 launch vive_teleop_tactile_bridge dmtac_w_bridge.launch.py \
  params_file:=/home/enine/ros2_hj/src/vr_hil_serl_bridge_pkg/config/dmtac_w_bridge.params.yaml

ros2 run vive_teleop_tactile_bridge record_raw_hilserl_zed_tactile_dmtac_w \
  --config /home/enine/ros2_hj/src/vr_hil_serl_bridge_pkg/config/record_hilserl_raw_tactile_dmtac_w.yaml
```

Use a new empty `dataset.root` after changing output mode. A schema-3 row must
never be appended to an existing schema-1 zarr array.

The worker reports actual_fps, per-side capture time, pair time, and
left/right capture skew every five seconds. A configured 30 Hz target is valid
only when the reported capture/pair time remains below the 33.3 ms period.
Set publish_legacy_modalities to true only when the five old modality topics
are explicitly needed; doing so restores their serialization/callback load.

The recording pipeline keeps the 10 Hz candidate grid but evaluates each
candidate after a configurable 250 ms synchronization lag. ZED and D405 image
header stamps are mapped into the recorder's monotonic clock domain, so DDS
callback delay is not mistaken for camera capture time. The Franka HTTP bridge
publishes a 20 Hz held-action heartbeat without sending extra HTTP requests;
real HTTP command events remain separately marked with `heartbeat=false`.

PNG encoding and tactile writes run on a bounded background queue. Left/right
zarr arrays are resized and appended in aligned batches (eight frames by
default), and episode commit/discard waits for a writer barrier before changing
files. Watch `loop_lag_ms` and `max_loop_lag_ms` in recorder output: a sustained
value near zero means disk work is no longer delaying the fixed-FPS
synchronization loop.

## DM-Tac W to Tabero LeRobot v2.1 (action-only baseline)

This separate exporter is for the first Tabero-VTLA reproduction. It writes
one LeRobot v2.1 Parquet file and two RGB videos per episode with these flat
Tabero keys:

- `image`, `wrist_image`: the ZED and D405 RGB streams.
- `state`: float32 `[7]`, xyz + axis-angle + gripper.
- `actions`: float32 `[7]`, absolute target xyz + axis-angle + gripper.
- `tactile_marker_motion`: float32 `[9,198,2]`. Slice 0 is the fixed 9x11
  marker reference grid from each finger; slices 1:9 are the last eight
  `reference + DM-Tac shear` frames, padded with episode frame 0 at startup.
- `wrist_wrench`: float32 `[6]`, retained for later work but not included in
  the first action target.
- `tactile_depth`: float32 `[2,240,320]`, retained but not read by the first
  Tabero loader.

The first Tabero loader/config must select only `image`, `wrist_image`,
`state`, `tactile_marker_motion`, and `actions`. Its action dimension is 7;
force prediction and force loss must be disabled. Do not concatenate
`wrist_wrench` onto `actions`.

Install the offline conversion dependencies in the Python environment used by
the command: `numpy`, `pandas`, `pyarrow`, `zarr`, `Pillow`, and OpenCV. After
rebuilding and sourcing the ROS 2 workspace, run:

```bash
ros2 run vive_teleop_tactile_bridge raw_dmtac_w_to_tabero_lerobot \
  --raw-root /absolute/path/to/raw_dataset \
  --out-root /absolute/path/to/tabero_lerobot_v21

ros2 run vive_teleop_tactile_bridge validate_tabero_lerobot \
  --root /absolute/path/to/tabero_lerobot_v21
```

The default `--timing-policy strict` refuses missing candidates or irregular
source timing. `--timing-policy compact` exists only for a file-format smoke
test because it changes the trajectory time scale. The exact grid indices,
shear scale, gripper units, and retained-field definitions are recorded in
`meta/tabero_conversion.json`.

`--shear-scale 1.0` assumes the DM-Tac shear channels are expressed in the
same pixel-coordinate convention as the sampled marker grid. Calibrate this
factor (and sensor-axis orientation in the training preprocessing, if needed)
before a final experiment; the converter deliberately does not estimate a
scale from the training set.

## DM-Tac W to N0-VTLA canonical LeRobot

The exporter accepts legacy schema 1 and shear/depth schema 3, decodes either
lossless layout to `[shear_x, shear_y, depth]`, maps it to two 3-channel tactile
videos, and writes the N0-VTLA single-arm canonical schema
(`observation.state`, `action`, and `action_mask` are all 32-dimensional;
dimensions `0:10` are active).

Install the offline conversion dependencies in the Python environment used by
the command: `numpy`, `pandas`, `pyarrow`, `zarr`, `Pillow`, and OpenCV.

After rebuilding and sourcing the ROS 2 workspace, run:

```bash
ros2 run vive_teleop_tactile_bridge raw_dmtac_w_to_n0vtla_lerobot \
  --raw-root /absolute/path/to/raw_dataset \
  --out-root /absolute/path/to/lerobot_dataset
```

The default `--timing-policy strict` rejects missing fixed-FPS candidates and
irregular timestamps instead of silently changing the trajectory time scale.
`--timing-policy compact` is only for a format smoke test and must not be used
to turn an irregular recording into training data.

Keep `meta/tactile_encoding.json` with the converted dataset and apply the same
channel order/scales during deployment. The first tactile frame of each
episode must be a clear, no-contact baseline because N0-VTLA forms tactile
inputs as `current - episode_frame_0`.

## Notes

- `server_url` must point to the Ubuntu 20.04 machine running `franka_server.py`, not the FR3 controller box.
- `vr_to_robot_rotation` is the main parameter to tune if your axes feel swapped or mirrored.
- First bring-up should use `enable_rotation: false`.
- `publish_gripper_tcp_tf` is optional and only meant for RViz/TF display compatibility with your old launch style.
- Continuous gripper control posts `{"gripper_width": value}` to `/move_gripper`; tune `gripper_width_min` and `gripper_width_max` after measuring your Robotiq range.
