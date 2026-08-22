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

## Notes

- `server_url` must point to the Ubuntu 20.04 machine running `franka_server.py`, not the FR3 controller box.
- `vr_to_robot_rotation` is the main parameter to tune if your axes feel swapped or mirrored.
- First bring-up should use `enable_rotation: false`.
- `publish_gripper_tcp_tf` is optional and only meant for RViz/TF display compatibility with your old launch style.
- Continuous gripper control posts `{"gripper_width": value}` to `/move_gripper`; tune `gripper_width_min` and `gripper_width_max` after measuring your Robotiq range.
