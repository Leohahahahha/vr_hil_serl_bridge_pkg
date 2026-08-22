from setuptools import find_packages, setup

package_name = 'vive_teleop_tactile_bridge'

setup(
    name=package_name,
    version='0.1.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', [
            'launch/bridge_only.launch.py',
            'launch/dmtac_w_bridge.launch.py',
            'launch/full_stack.launch.py',
            'launch/vr_teleop.launch.py',
        ]),
        (f'share/{package_name}/config', [
            'config/bridge.params.yaml',
            'config/dmtac_w_bridge.params.yaml',
            'config/record_hilserl_raw.yaml',
            'config/record_hilserl_raw_tactile_dmtac_w.yaml',
            'config/record_hilserl_raw_tactile_paxini.yaml',
            'config/record_hilserl_raw_tactile_tashan.yaml',
        ]),
        (f'share/{package_name}/scripts', [
            'vive_teleop_tactile_bridge/dmtac_w_ipc.py',
            'vive_teleop_tactile_bridge/dmtac_w_sdk_worker.py',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='1',
    maintainer_email='1@example.com',
    description='ROS 2 package for HTC VR to HIL-SERL Franka HTTP bridge on Ubuntu 24.04.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vr_tracker_node = vive_teleop_tactile_bridge.vr_ros2_node:main',
            'franka_http_bridge_node = vive_teleop_tactile_bridge.franka_http_bridge_node:main',
            'vr_converter_node = vive_teleop_tactile_bridge.franka_http_bridge_node:main',
            'record_raw_hilserl_zed = vive_teleop_tactile_bridge.record_raw_hilserl_zed:main',
            'record_raw_hilserl_zed_tactile_paxini = vive_teleop_tactile_bridge.record_raw_hilserl_zed_tactile_paxini:main',
            'record_raw_hilserl_zed_tactile_tashan = vive_teleop_tactile_bridge.record_raw_hilserl_zed_tactile_tashan:main',
            'record_raw_hilserl_zed_tactile_dmtac_w = vive_teleop_tactile_bridge.record_raw_hilserl_zed_tactile_dmtac_w:main',
            'dmtac_w_bridge = vive_teleop_tactile_bridge.dmtac_w_bridge:main',
            'check_raw_dataset = vive_teleop_tactile_bridge.check_raw_dataset:main',
            'merge_raw_tactile_datasets = vive_teleop_tactile_bridge.merge_raw_tactile_datasets:main',
            'raw_tactile_to_csv = vive_teleop_tactile_bridge.export_raw_tactile_csv:main',
            'raw_tashan_to_lerobot_v3 = vive_teleop_tactile_bridge.lerobot_export.raw_tashan_to_lerobot_v3:main',
            'raw_paxini_to_lerobot_v3 = vive_teleop_tactile_bridge.lerobot_export.raw_paxini_to_lerobot_v3:main',
            'raw_dmtac_w_to_n0vtla_lerobot = vive_teleop_tactile_bridge.lerobot_export.raw_dmtac_w_to_n0vtla_lerobot:main',
            'validate_lerobot_v3 = vive_teleop_tactile_bridge.lerobot_export.validate_lerobot_v3:main',
            'inspect_lerobot_v3_sample = vive_teleop_tactile_bridge.lerobot_export.inspect_lerobot_v3_sample:main',
            'lerobot_tactile_to_csv = vive_teleop_tactile_bridge.lerobot_export.export_lerobot_tactile_csv:main',
        ],
    },
)
