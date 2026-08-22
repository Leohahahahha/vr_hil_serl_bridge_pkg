from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    default_params = PathJoinSubstitution(
        [
            FindPackageShare("vive_teleop_tactile_bridge"),
            "config",
            "dmtac_w_bridge.params.yaml",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_params),
            Node(
                package="vive_teleop_tactile_bridge",
                executable="dmtac_w_bridge",
                name="dmtac_w_bridge",
                output="screen",
                emulate_tty=True,
                parameters=[params_file],
            ),
        ]
    )
