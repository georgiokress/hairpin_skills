#!/usr/bin/env python3

"""
Bring-up launch file for the UR3e used in the hairpin experiments.

This launch file starts only the Universal Robots driver.
It does not start a skill, force mode, or an automatic robot motion.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Create the launch description for the UR3e driver."""
    robot_ip = LaunchConfiguration("robot_ip")
    reverse_ip = LaunchConfiguration("reverse_ip")
    ur_type = LaunchConfiguration("ur_type")
    headless_mode = LaunchConfiguration("headless_mode")
    launch_rviz = LaunchConfiguration("launch_rviz")
    initial_joint_controller = LaunchConfiguration(
        "initial_joint_controller"
    )
    kinematics_params_file = LaunchConfiguration(
        "kinematics_params_file"
    )

    ur_driver_launch_file = PathJoinSubstitution(
        [
            FindPackageShare("ur_robot_driver"),
            "launch",
            "ur_control.launch.py",
        ]
    )

    default_calibration_file = PathJoinSubstitution(
        [
            FindPackageShare("hairpin_skills"),
            "config",
            "ur3e_calibration.yaml",
        ]
    )

    declared_arguments = [
        DeclareLaunchArgument(
            "robot_ip",
            default_value="192.168.214.203",
            description="IPv4 address of the UR3e controller.",
        ),
        DeclareLaunchArgument(
            "reverse_ip",
            default_value="192.168.214.204",
            description="IPv4 address of the ROS control computer.",
        ),
        DeclareLaunchArgument(
            "ur_type",
            default_value="ur3e",
            choices=["ur3e"],
            description="Universal Robots model.",
        ),
        DeclareLaunchArgument(
            "headless_mode",
            default_value="false",
            choices=["true", "false"],
            description=(
                "Start the external-control program automatically "
                "without requiring manual program start."
            ),
        ),
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="false",
            choices=["true", "false"],
            description="Start the UR driver RViz configuration.",
        ),
        DeclareLaunchArgument(
            "initial_joint_controller",
            default_value="scaled_joint_trajectory_controller",
            description="Controller activated when the driver starts.",
        ),
        DeclareLaunchArgument(
            "kinematics_params_file",
            default_value=default_calibration_file,
            description="Robot-specific kinematic calibration YAML file.",
        ),
    ]

    ur_driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ur_driver_launch_file),
        launch_arguments={
            "ur_type": ur_type,
            "robot_ip": robot_ip,
            "reverse_ip": reverse_ip,
            "kinematics_params_file": kinematics_params_file,
            "headless_mode": headless_mode,
            "launch_rviz": launch_rviz,
            "initial_joint_controller": initial_joint_controller,
        }.items(),
    )
    ati_sensor_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="ati_sensor_static_tf",
        output="screen",
        arguments=[
            "--x", "0.0",
            "--y", "0.0",
            "--z", "0.076",
            "--roll", "3.141592654",
            "--pitch", "0.0",
            "--yaw", "-1.178097245",
            "--frame-id", "tool0",
            "--child-frame-id", "ati_sensor",
        ],
    )
    ati_ur_aligned_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="ati_ur_aligned_static_tf",
        output="screen",
        arguments=[
            "--x", "0.0",
            "--y", "0.0",
            "--z", "0.076",
            "--roll", "0.0",
            "--pitch", "0.0",
            "--yaw", "0.0",
            "--frame-id", "tool0",
            "--child-frame-id", "ati_ur_aligned",
        ],
    )
    sliding_frame_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="setznest2_sliding_frame_static_tf",
        output="screen",
        arguments=[
            "--x", "-0.22768",
            "--y", "-0.39961",
            "--z", "0.03753",
            "--qx", "0.002069",
            "--qy", "0.000319",
            "--qz", "-0.970849",
            "--qw", "-0.239684",
            "--frame-id", "base",
            "--child-frame-id", "sliding_frame",
        ],
    )
    return LaunchDescription(
        declared_arguments
        + [
            ur_driver,
            ati_sensor_tf,
            ati_ur_aligned_tf,
            sliding_frame_tf,
        ]
    )
