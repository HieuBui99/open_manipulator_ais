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
from scipy.spatial.transform import Rotation as R
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
            # gripper_links=links[10],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )
        # self.R0 = np.array([[0,0,1],
        #                     [1,0,0],
        #                     [0,1,0]])
        self.Rz = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0],
                                          [np.sin(theta), np.cos(theta), 0],
                                          [0, 0, 1]])   
        self.Rx = lambda theta: np.array([[1, 0, 0],
                                          [0, np.cos(theta), -np.sin(theta)],
                                          [0, np.sin(theta), np.cos(theta)]])
        self.Ry = lambda theta: np.array([[np.cos(theta), 0, np.sin(theta)],
                                          [0, 1, 0],
                                          [-np.sin(theta), 0, np.cos(theta)]])
        if boudary is not None:
            self.boudary = boudary  # [xmin, xmax, ymin, ymax, zmin, zmax]      
        else:
            self.boudary = [[0.1, 0.4],
                            [-0.3, 0.3],
                            [0.1, 0.2]]#onion
    

    def IK(self, eef_pose, q0=None):
        joint_angles, success, _, _, _ = self.ik_LM(eef_pose, q0=q0) 
        # breakpoint()
        if not success:
            print("IK failed, using q0 as fallback")
            if q0 is not None:
                joint_angles = np.array(q0)[:-1]
        else:
            difference = np.abs(joint_angles - q0[:-1])
            invalid_indices = np.where(difference > np.pi/10)[0]
            print(invalid_indices)
            if len(invalid_indices) > 0:
                joint_angles = np.array(q0)[:-1]#keep static if the solution is dramatically different
        return joint_angles

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
        # rotation = self.R0
        rotation = np.array(transform.R)  # eef pose of the leader
        R_rel = rotation @ self.R0.T # compute the relative rotation between  the initial leader pose and the current leader pose
        rot_rel = R.from_matrix(R_rel) # LIe bracket --> get the twist vector
        axis_angle = rot_rel.as_rotvec() # convert to axis angle representation, angle * rotation axis
        print("axis_angle", axis_angle/np.linalg.norm(axis_angle))
        # breakpoint()
        # x_init = self.R0[:,0] # RANDOM GUESS, [0,1,0]
        angle_about_z = np.dot(axis_angle, [0,0,1]) # compute the angle about the z axis of the initial pose (scalar)
        # print("angle_about_z", angle_about_x)
        rotation_new = self.Rz(angle_about_z) @ self.R0 #Ry RANDOM GUESS 
        # breakpoint()
        # print("rotation_new", rotation_new)
        # print(rotation)
        T = np.eye(4)
        # T[:3,:3] = np.array(transform.R)
        T[:3,:3] = rotation_new
        # res = np.array(transform_new).dot(T)
        # print("res", transform)
        return np.array(transform_new).dot(T)

    def cal_follower_joint_angles(self, target_pose, q0, q_follower):
        transform_follower = self.clipper(target_pose)[0]
        # lower_bound = np.array(q0) - band
        # upper_bound = np.array(q0) + band
        # joint_limits = np.stack([lower_bound, 
        #                          upper_bound]).T
        # breakpoint()
        # joint_limits_clipped = np.clip(joint_limits, -np.pi, np.pi)
        # joint_angles = self.ik_LM(
        #                         transform_new, 
        #                         q0=q0, 
        #                         joint_limits=True,
        #                         )[0] 
        joint_angles, success, _, _, _ = self.ik_LM(
                                transform_follower, 
                                q0=q_follower,
                                # q0=q0, 
                                )
        if not success:
            print("IK failed, using q0 as fallback")
            joint_angles = np.array(q_follower)[:-1]

        difference = np.abs(joint_angles - q_follower[:-1])
        invalid_indices = np.where(difference > np.pi)[0]
        print(invalid_indices)
        if len(invalid_indices) > 0:
            joint_angles[invalid_indices] = np.array(q_follower)[invalid_indices]
        # if np.abs(np.abs(joint_angles) - np.abs(q_follower[:-1])).max() > 0.5:
        #     joint_angles = q0
        #     print("To faset!")
        #     breakpoint()
        # joint_angles = np.clip(joint_angles, -np.pi, np.pi)
        # if np.abs(joint_angles - q0[:-1]).max() > np.pi/2:
        #     breakpoint()
        return np.array(joint_angles)


class KeyboardController(Node):

    def __init__(self):
        super().__init__('keyboard_controller')
        self.robot = OMY()

        # Publisher for arm joint control
        self.arm_publisher = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10
        )

        # Action client for GripperCommand
        # self.gripper_client = ActionClient(
        #     self, GripperCommand, '/gripper_controller/gripper_cmd'
        # )

        # Subscriber for joint states
        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        self.arm_joint_positions = [0.0] * 7
        self.arm_joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'rh_r1_joint'  # Gripper joint
        ]

        self.gripper_position = 0.0
        self.gripper_max = 1.1
        self.gripper_min = 0.0

        self.joint_received = False

        self.max_delta = 0.002
        self.max_delta_radians = 0.01  # radians for rotation
        self.gripper_delta = 0.1
        self.last_command_time = time.time()
        self.command_interval = 0.02

        self.running = True  # for thread loop control

        self.get_logger().info('Waiting for /joint_states...')
        self.rate = self.create_rate(10)

    @property
    def _eef_pose(self):
        return self.robot.fkine(self.arm_joint_positions)

    def joint_state_callback(self, msg):
        if set(self.arm_joint_names).issubset(set(msg.name)):
            # follower's joint positions
            for i, joint in enumerate(self.arm_joint_names):
                index = msg.name.index(joint)
                self.arm_joint_positions[i] = msg.position[index]
                self.get_logger().info(
                    f'Joint {joint} position: {self.arm_joint_positions[i]}')
        # if 'rh_r1_joint' in msg.name:
        #     # index = msg.name.index('rh_r1_joint')
        #     self.gripper_position = msg.position[index]

        self.joint_received = True
        # self.get_logger().info(
        #     f'Received joint states: {self.arm_joint_positions}, '
        #     f'Gripper: {self.gripper_position}'
        # )

    def get_key(self, timeout=0.01):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def send_arm_command(self, joint_position):
        arm_msg = JointTrajectory()
        arm_msg.joint_names = self.arm_joint_names
        arm_point = JointTrajectoryPoint()
        self.arm_joint_positions[:-1] = joint_position  # Update the arm joint positions
        arm_point.positions = joint_position + [self.arm_joint_positions[-1]]  # Add gripper joint position
        arm_point.time_from_start.sec = 0
        arm_msg.points.append(arm_point)
        self.arm_publisher.publish(arm_msg)
        self.get_logger().info(f'Arm command sent: {joint_position}')

    def send_gripper_command(self):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = self.gripper_position
        goal_msg.command.max_effort = 10.0

        self.get_logger().info(f'Sending gripper command: {goal_msg.command.position}')
        self.gripper_client.wait_for_server()
        send_goal_future = self.gripper_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

    def run(self):
        while not self.joint_received and rclpy.ok() and self.running:
            self.get_logger().info('Waiting for initial joint states...')
            rclpy.spin_once(self, timeout_sec=1.0)

        self.get_logger().info('Ready to receive keyboard input!')
        self.get_logger().info(
            'Use 1/q, 2/w, 3/e, 4/r, 5/t, 6/y for joints 1-6, o/p for gripper. '
            'Press ESC to exit.'
        )

        try:
            while rclpy.ok() and self.running:
                key = self.get_key()
                current_time = time.time()

                if key is None:
                    continue

                if current_time - self.last_command_time >= self.command_interval:
                    eef_curr = self._eef_pose
                    eef_curr_new = eef_curr.copy()
                    if key == '\x1b':  # ESC
                        self.running = False
                        break

                    elif key == '1':
                        eef_curr_new.t[0] += self.max_delta  # Move in x direction
                        joint_positions_new = self.robot.IK(eef_curr_new, self.arm_joint_positions) # 6-dim 
 
                    elif key == 'q':
                        eef_curr_new.t[0] -= self.max_delta  # Move in x direction
                        joint_positions_new = self.robot.IK(eef_curr_new, self.arm_joint_positions) # 6-dim 
 

                    elif key == '2':
                        eef_curr_new.t[1] += self.max_delta  # Move in x direction
                        joint_positions_new = self.robot.IK(eef_curr_new, self.arm_joint_positions) # 6-dim 
 
                    elif key == 'w':
                        eef_curr_new.t[1] -= self.max_delta  # Move in x direction
                        joint_positions_new = self.robot.IK(eef_curr_new, self.arm_joint_positions) # 6-dim 
 
                    elif key == '3':
                        eef_curr_new.t[2] += self.max_delta  # Move in x direction
                        joint_positions_new = self.robot.IK(eef_curr_new, self.arm_joint_positions) # 6-dim 
 
                    elif key == 'e':
                        eef_curr_new.t[2] -= self.max_delta  # Move in x direction
                        joint_positions_new = self.robot.IK(eef_curr_new, self.arm_joint_positions) # 6-dim 
 
                    elif key == '4':
                        #rotation around x-axis
                        new_rotation = self.robot.Rx(self.max_delta_radians)
                        eef_curr_new.R = new_rotation @ eef_curr.R
                        joint_positions_new = self.robot.IK(eef_curr_new, self.arm_joint_positions)

                    elif key == 'r':
                        #rotation around x-axis
                        new_rotation = self.robot.Rx(-self.max_delta_radians)
                        eef_curr_new.R = new_rotation @ eef_curr.R
                        joint_positions_new = self.robot.IK(eef_curr_new, self.arm_joint_positions)

                    elif key == '5':
                        #rotation around y-axis
                        new_rotation = self.robot.Ry(self.max_delta_radians)
                        eef_curr_new.R = new_rotation @ eef_curr.R
                        joint_positions_new = self.robot.IK(eef_curr_new, self.arm_joint_positions)

                    elif key == 't':
                        #rotation around y-axis
                        new_rotation = self.robot.Ry(-self.max_delta_radians)
                        eef_curr_new.R = new_rotation @ eef_curr.R
                        joint_positions_new = self.robot.IK(eef_curr_new, self.arm_joint_positions)

                    elif key == '6':
                        #rotation around z-axis
                        new_rotation = self.robot.Rz(self.max_delta_radians)
                        eef_curr_new.R = new_rotation @ eef_curr.R
                        joint_positions_new = self.robot.IK(eef_curr_new, self.arm_joint_positions)

                    elif key == 'y':
                        #rotation around z-axis
                        new_rotation = self.robot.Rz(-self.max_delta_radians)
                        eef_curr_new.R = new_rotation @ eef_curr.R
                        joint_positions_new = self.robot.IK(eef_curr_new, self.arm_joint_positions)

                    elif key == 'o':  # Open gripper
                        new_pos = max(
                            self.gripper_position - self.gripper_delta, self.gripper_min
                        )
                        self.gripper_position = new_pos
                        self.send_gripper_command()
                    elif key == 'p':  # Close gripper
                        new_pos = min(
                            self.gripper_position + self.gripper_delta, self.gripper_max
                        )
                        self.gripper_position = new_pos
                        self.send_gripper_command()

                    self.send_arm_command(joint_positions_new.tolist())
                    self.last_command_time = current_time

        except Exception as e:
            self.get_logger().error(f'Exception in run loop: {e}')


def main():
    rclpy.init()
    node = KeyboardController()
    # node_thread = threading.Thread(target=rclpy.spin, args=(node,))
    thread = threading.Thread(target=node.run)
    thread.start()
    # node_thread.start()
    try:
        while thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print('\nCtrl+C detected. Shutting down...')
        node.running = False
        thread.join()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
