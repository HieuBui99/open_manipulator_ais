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

    publish_tf_node = Node(
        package='open_manipulator_bringup',
        executable='pose_publish_tf',
        name='eef_to_base',  # Optional: node name in rqt or `ros2 node list`
        output='screen'     # Print stdout/stderr to terminal
    )


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
    # camera_realsense = IncludeLaunchDescription(PathJoinSubstitution([
    #     FindPackageShare('camera_calibrate'),
    #     'launch',
    #     'camera_realsense_launch.py'
    # ]),
    #     launch_arguments={
    #         "serial_no1": "_218622274409",
    #         "serial_no2": "_826212070364",
    #         "serial_no3": "_941322072865",
    #     }.items()
    # )
    camera_pose = IncludeLaunchDescription(PathJoinSubstitution([
        FindPackageShare('camera_calibrate'),
        'launch',
        'camera_pose_launch.py'
    ]),
    )
    visualize_chessboard = Node(
        package='camera_calibrate',
        executable='visualize_chessboard',
        name='visualizeChessboard',
        parameters=[PathJoinSubstitution([
            FindPackageShare('camera_calibrate'), 'config', 'extrinsics_estimate.yaml']),
        ],
        output='screen'
    )
    prefix = LaunchConfiguration('prefix')
    urdf_file = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        ' ',
        PathJoinSubstitution([
            FindPackageShare('open_manipulator_description'),
            'urdf',
            'omy_f3m',
            'omy_f3m.urdf.xacro',
        ]),
        ' ',
        'prefix:=',
        prefix,
        ' ',
        'use_fake_hardware:=',
        'True',
    ])


    return LaunchDescription([
        DeclareLaunchArgument(
            'serial_no',
            default_value='_826212070364',
            description='Serial number of the camera to calibrate'
        ),
        DeclareLaunchArgument(
            'prefix',
            default_value='""',
            description='prefix of the joint and link names',
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': urdf_file}],
            output='screen',
        ),
        camera_realsense,
        camera_pose,
        publish_tf_node,
        RegisterEventHandler(
            OnProcessStart(
                target_action=camera_realsense,
                on_start=[visualize_chessboard]
            )
        ),
        RegisterEventHandler(
            OnProcessStart(
                target_action=visualize_chessboard,
                on_start=[
                    LogInfo(msg='Visualizing chessboard...'),
                    Node(
                        package='rviz2',
                        executable='rviz2',
                        name='rviz2',
                        arguments=['-d', PathJoinSubstitution([
                            FindPackageShare('camera_calibrate'),
                            'rviz_config',
                            'chessboard_vis.rviz'
                        ])],
                        output='screen'
                    )
                ]
            )
        ),
        # visualize_chessboard,
        # TimerAction(
        #     period=5.0,
        #     actions=[visualize_chessboard]
        # ),

        # TimerAction(
        #     period=8.0,
        #     actions = [Node(
        #         package='rviz2',
        #         executable='rviz2',
        #         name='rviz2',
        #         arguments=['-d', PathJoinSubstitution([
        #             FindPackageShare('camera_calibrate'),
        #             'rviz_config',
        #             'chessboard_vis.rviz'
        #         ])],
        #         output='screen'
        #     )]),
    ])

