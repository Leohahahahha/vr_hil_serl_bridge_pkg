#!/usr/bin/env python3
"""
VR to HIL-SERL Franka HTTP Launch File
启动VR数据接收和HTTP桥接系统
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory('vive_teleop_tactile_bridge'),
        'config',
        'bridge.params.yaml',
    )

    declared_arguments = [
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='Path to the configuration file'
        ),
        DeclareLaunchArgument(
            'update_rate',
            default_value='90.0',
            description='VR tracking update rate (Hz)'
        ),
        DeclareLaunchArgument(
            'launch_tracker',
            default_value='true',
            description='Whether to launch the OpenVR tracker node'
        ),
        DeclareLaunchArgument(
            'publish_gripper_tcp_tf',
            default_value='false',
            description='Whether to publish an optional static TF from gripper base to TCP'
        ),
    ]

    config_file = LaunchConfiguration('config_file')

    vr_tracker_node = Node(
        condition=IfCondition(LaunchConfiguration('launch_tracker')),
        package='vive_teleop_tactile_bridge',
        executable='vr_tracker_node',
        name='tactile_vr_tracker_node',
        output='screen',
        parameters=[
            config_file,
            {'update_rate': LaunchConfiguration('update_rate')}
        ]
    )

    vr_converter_node = Node(
        package='vive_teleop_tactile_bridge',
        executable='vr_converter_node',
        name='tactile_vr_converter_node',
        output='screen',
        parameters=[config_file]
    )

    def _make_gripper_tcp_tf(context):
        if LaunchConfiguration('publish_gripper_tcp_tf').perform(context).lower() not in ('1', 'true', 'yes', 'on'):
            return []

        config_path = LaunchConfiguration('config_file').perform(context)
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            data = {}

        params = (
            data.get('tactile_vr_converter_node', {})
            .get('ros__parameters', {})
        )
        xyz = params.get('gripper_tcp_xyz', [0.0, 0.0, 0.0])
        rpy = params.get('gripper_tcp_rpy', [0.0, 0.0, 0.0])
        parent = str(params.get('gripper_parent_frame', 'robotiq_85_base_link'))
        child = str(params.get('gripper_child_frame', 'robotiq_85_tcp'))
        xyz = [float(v) for v in (xyz + [0.0, 0.0, 0.0])[:3]]
        rpy = [float(v) for v in (rpy + [0.0, 0.0, 0.0])[:3]]

        return [
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='tactile_gripper_tcp_static_tf',
                output='screen',
                arguments=[
                    '--frame-id', parent,
                    '--child-frame-id', child,
                    '--x', str(xyz[0]),
                    '--y', str(xyz[1]),
                    '--z', str(xyz[2]),
                    '--roll', str(rpy[0]),
                    '--pitch', str(rpy[1]),
                    '--yaw', str(rpy[2]),
                ],
            )
        ]

    return LaunchDescription(
        declared_arguments + [
            LogInfo(msg='=' * 60),
            LogInfo(msg='VR to HIL-SERL Franka HTTP System Starting...'),
            LogInfo(msg='=' * 60),
            GroupAction([
                vr_tracker_node,
                vr_converter_node,
                OpaqueFunction(function=_make_gripper_tcp_tf),
            ]),
            LogInfo(msg='VR tracking and HTTP bridge nodes launched.'),
            LogInfo(msg='Waiting for SteamVR and Franka server...'),
        ]
    )
