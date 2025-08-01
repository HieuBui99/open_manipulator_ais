from launch import LaunchDescription
from launch_ros.actions import Node
import yaml
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

def get_camera_tf():
    ''' Get the camera extrinsics from the YAML file and convert to a pose dictionary.
    '''
    camera_file_pth = os.path.join(
        get_package_share_directory('camera_calibrate'),
        'config',
        'cameras.yaml'
    )
    # Load the YAML
    with open(camera_file_pth, 'r') as f:
        config = yaml.safe_load(f)

    serial_nos = {
        # "_218622274409"
        "_826212070364": "camera_scene1_color_frame",
        "_941322072865": "camera_scene2_color_frame"}
    nodes = []
    for serial_number in serial_nos.keys():
        if serial_number not in config:
            raise ValueError(f"Serial number {serial_number} not found in cameras.yaml")

        extrinsics = config[serial_number]['extrinsics']
        R_matrix = np.array(extrinsics['rotation'])
        T_vector = np.array(extrinsics['translation']).squeeze()/1000  # Convert mm to m
        quaternion = R.from_matrix(R_matrix).as_quat()
        pose = {
            'position': {
                'x': T_vector[0],
                'y': T_vector[1],
                'z': T_vector[2]
            },
            'orientation': {
                'x': quaternion[0],
                'y': quaternion[1],
                'z': quaternion[2],
                'w': quaternion[3]
            }
        }
        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            # name=f'camera{serial_number}',
            arguments=[
                '--x', str(pose['position']['x']),
                '--y', str(pose['position']['y']),
                '--z', str(pose['position']['z']),
                '--qx', str(pose['orientation']['x']),
                '--qy', str(pose['orientation']['y']),
                '--qz', str(pose['orientation']['z']),
                '--qw', str(pose['orientation']['w']),
                '--frame-id', 'world',  # Base frame
                '--child-frame-id', serial_nos[serial_number]  # Child frame
            ]
        ))
    return nodes

def generate_launch_description():
    nodes = get_camera_tf()
    return LaunchDescription(nodes)

# get_camera_tf()