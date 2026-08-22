from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('vive_teleop_tactile_bridge'),
        'config',
        'bridge.params.yaml',
    )
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Path to ROS2 parameters file for tactile_vr_converter_node / tactile_franka_http_bridge.'
        ),
        Node(
            package='vive_teleop_tactile_bridge',
            executable='vr_converter_node',
            name='tactile_vr_converter_node',
            output='screen',
            parameters=[params_file],
        ),
    ])
