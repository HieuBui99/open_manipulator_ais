#!/usr/bin/env python3
#
# Copyright 2024 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Sungho Woo

import select
import sys
import termios
import threading
import time
import tty

from control_msgs.action import GripperCommand
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from roboticstoolbox import Robot
from roboticstoolbox import Robot
from spatialmath import SE3
from roboticstoolbox.tools.data import rtb_path_to_datafile
import numpy as np
import pdb
class OMY(Robot):
    def __init__(self, boudary:list=None):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "/root/ros2_ws/src/open_manipulator/custom_xacro/omy_description/robots/omy_f3m.urdf.xacro",
            # tld=xacro_path,
        )
        super().__init__(
            links,
            name="OMY",
            manufacturer="N/A",
            gripper_links=links[9],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )
        if boudary is not None:
            self.boudary = boudary  # [xmin, xmax, ymin, ymax, zmin, zmax]      
        else:
            self.boudary = [[-0.3, 0.3],
                            [-0.3, 0.3],
                            [0.1, 0.6]]#onion

    def IK(self, eef_pose):
        sol = self.ik_LM(eef_pose) 
        # Placeholder for IK implementation
        # This should return the joint positions for the given end-effector position
        return sol
    
    def clipper(self, transform):
        def clip(value, Range):
            if value < Range[0]:
                return Range[0]
            elif value > Range[1]:
                return Range[1]
            else:
                return value
        tVec = transform.t
        tVec_clipped = list(map(clip, tVec, self.boudary))
        transform_new = SE3(tVec_clipped)
        rotation = transform.R
        T = np.eye(4)
        T[:3,:3] = rotation
        return np.array(transform_new).dot(T)
    
    def cal_follower_joint_angles(self, target_pose, q0):
        transform_new = self.clipper(target_pose)[0]

        # joint_angles = self.IK(target_pose)[0]
        joint_angles = self.ik_LM(transform_new, q0=q0)[0] 
        # breakpoint()
        return joint_angles

class KeyboardController(Node):
    def __init__(self):
        super().__init__('keyboard_controller')
        self.robot = OMY()

        # Publisher for arm joint control
        self.arm_publisher = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10
        )

        # Subscriber for joint states
        self.subscription = self.create_subscription(
            JointState, '/leader/joint_states', self.joint_state_callback, 10
        )

        self.arm_joint_positions = [0.0] * 7
        self.arm_joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'rh_r1_joint'
        ]

        self.gripper_position = 0.0
        self.gripper_max = 1.1
        self.gripper_min = 0.0

        self.joint_received = False

        self.max_delta = 0.02
        self.gripper_delta = 0.1
        self.last_command_time = time.time()
        self.command_interval = 0.02

        self.running = True  # for thread loop control

        self.get_logger().info('Waiting for /joint_states...')
        self.rate = self.create_rate(10)

    def joint_state_callback(self, msg):
        # print(msg.name)
        if set(self.arm_joint_names).issubset(set(msg.name)):
            for i, joint in enumerate(self.arm_joint_names):
                # print(i, joint)
                index = msg.name.index(joint)
                self.arm_joint_positions[i] = msg.position[index]
        # breakpoint()
        joint_positions_new = self.robot.cal_follower_joint_angles(self._eef_pose, self.arm_joint_positions).tolist()
        # breakpoint()
        joint_positions_new.append(self.arm_joint_positions[-1])  # Keep the gripper joint position
        #we may need mod function
        self.joint_received = True
        self.get_logger().info(
            f'Received joint states: {self.arm_joint_positions}, '
            # f'Gripper: {self.gripper_position}'
        )

        arm_msg = JointTrajectory()
        arm_msg.joint_names = self.arm_joint_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = joint_positions_new
        arm_point.time_from_start.sec = 0
        arm_msg.points.append(arm_point)
        self.arm_publisher.publish(arm_msg)
        self.get_logger().info(f'Arm command sent: {joint_positions_new}')
    
    @property
    def _eef_pose(self):
        return self.robot.fkine(self.arm_joint_positions)

def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = KeyboardController()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

