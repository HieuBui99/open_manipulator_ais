from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
    TimerAction
)
from launch.conditions import IfCondition
from launch.event_handlers import (
    OnExecutionComplete,
    OnProcessExit,
    OnProcessIO,
    OnProcessStart,
    OnShutdown
)
from launch.events import Shutdown
from launch.substitutions import (
    EnvironmentVariable,
    FindExecutable,
    LaunchConfiguration,
    LocalSubstitution,
    PythonExpression
)
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution

from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command


serial_no_to_image_topic = {
    "_218622274409": "/camera/camera_wrist/color/image_raw/compressed",
    "_826212070364": "/camera/camera_scene1/color/image_raw/compressed",
    "_941322072865": "/camera/camera_scene2/color/image_raw/compressed"
}

serial_no_to_node = {
    "_218622274409": "/camera/camera_wrist",
    "_826212070364": "/camera/camera_scene1",
    "_941322072865": "/camera/camera_scene2"
}


def generate_launch_description():
    # Step 3: Start leader launch file
    camera_realsense = ExecuteProcess(
        cmd=[
            'ros2',
            'launch',
            'camera_calibrate',
            'camera_realsense_launch.py',
            'serial_no1:=_218622274409',
            'serial_no2:=_826212070364',
            'serial_no3:=_941322072865',
        ],
        output='screen',
        shell=True,
    )

    camera_pose = IncludeLaunchDescription(PathJoinSubstitution([
        FindPackageShare('camera_calibrate'),
        'launch',
        'camera_pose_launch.py'
    ]),
    )

    return LaunchDescription([
        camera_realsense,
        camera_pose,
        # tf_node
        # RegisterEventHandler(
        #     OnProcessStart(
        #         target_action=camera_realsense,
        #         on_start=[tf_node]
        #     )
        # ),
    ]
        )

