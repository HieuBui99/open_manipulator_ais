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
# Author: AIS LAB

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from roboticstoolbox import Robot
from spatialmath import SE3
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml
from ament_index_python.packages import get_package_share_directory
import os

 
class OMYKinematics(Robot):
    def __init__(self, boundary: dict = None, constrain_on_orientation: bool = True, mirror_y_axis: bool = False):
        links, _, urdf_string, urdf_filepath = self.URDF_read(
            "/root/ros2_ws/src/open_manipulator/custom_xacro/omy_description/robots/omy_f3m.urdf.xacro",
            # tld=xacro_path,
        )
        super().__init__(
            links,
            name="OMYKinematics",
            manufacturer="N/A",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )
        self.constrain_on_orientation = constrain_on_orientation
        self.R0 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
        self.Rz = lambda theta: np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        self.Rx = lambda theta: np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )
        self.Ry = lambda theta: np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

        if boundary is not None:
            self.boundary = [
                [boundary["x_min"], boundary["x_max"]],
                [boundary["y_min"], boundary["y_max"]],
                [boundary["z_min"], boundary["z_max"]],
            ]  # [xmin, xmax, ymin, ymax, zmin, zmax]
        else:
            self.boundary = [[0.1, 0.4], [-0.3, 0.3], [0.1, 0.2]]  # onion
        self.mirror_y_axis = mirror_y_axis

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
        if self.mirror_y_axis:
            tVec[1] = -tVec[1]
        tVec_clipped = list(map(clip, tVec, self.boundary))
        transform_new = SE3(tVec_clipped)
        # rotation = self.R0
        if self.constrain_on_orientation:
            # Compute the relative rotation
            rotation = np.array(transform.R)  # eef pose of the leader
            R_rel = (
                rotation @ self.R0.T
            )  # compute the relative rotation between  the initial leader pose and the current leader pose
            rot_rel = R.from_matrix(R_rel)  # LIe bracket --> get the twist vector
            axis_angle = (
                rot_rel.as_rotvec()
            )  # convert to axis angle representation, angle * rotation axis
            # print("axis_angle", axis_angle/np.linalg.norm(axis_angle))
            # breakpoint()
            # x_init = self.R0[:,0] # RANDOM GUESS, [0,1,0]
            angle_about_z = np.dot(
                axis_angle, [0, 0, 1]
            )*1.5  # compute the angle about the z axis of the initial pose (scalar), amplified by 1.5.

            if abs(angle_about_z) > np.pi:
                angle_about_z = np.sign(angle_about_z) * np.pi
            # print("angle_about_z", angle_about_x)
            rotation_new = self.Rz(angle_about_z) @ self.R0  # Ry RANDOM GUESS
            # breakpoint()
            # print("rotation_new", rotation_new)
            # print(rotation)
            T = np.eye(4)
            # T[:3,:3] = np.array(transform.R)
            T[:3, :3] = rotation_new
            # res = np.array(transform_new).dot(T)
            # print("res", transform)
            return np.array(transform_new).dot(T)
        else:
            transform_new.R = transform.R
            return np.array(transform_new)

    def cal_follower_joint_angles(self, target_pose, q0, q_follower):
            # print("target_pose", target_pose)
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

class ConstrainedTeleOperation(Node):
    def __init__(self):
        super().__init__('keyboard_controller')

        config_file_pth = os.path.join(
            get_package_share_directory("open_manipulator_bringup"),
            "config",
            "constrained_tele_op.yaml",
        )
        with open(config_file_pth, "r") as f:
            config = yaml.safe_load(f)

        self.Kp = config["Teleoperation"]["Kp"]
        self.Kd = config["Teleoperation"]["Kd"]
        boundary = config["Boundary"]
        constrain_on_orientation = config.get("constrain_on_orientation", True)
        mirror_y_axis = config.get("mirror_y_axis", False)
        self.robot = OMYKinematics(boundary=boundary,
                                   constrain_on_orientation=constrain_on_orientation,
                                   mirror_y_axis=mirror_y_axis)

        # Publisher for arm joint control
        self.arm_publisher = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 30
        )

        # Subscriber for joint states
        self.subscription = self.create_subscription(
            JointTrajectory, '/leader/joint_trajectory', self.joint_state_callback, 30
        )

        self.subscription_follower = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback_follower, 30
        )
        self.leader_rev = False
        self.follower_rev = False

        self.arm_joint_positions = [0.0] * 7
        self.arm_joint_positions_follower = [0.0] * 7
        self.arm_joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'rh_r1_joint'
        ]
        self.error_t_1 = [0.0] * 6

        self.gripper_position = 0.0
        self.gripper_max = 1
        self.gripper_min = 0.0

        self.joint_received = False

        self.last_command_time = time.time()

        self.get_logger().info('Waiting for /joint_states...')
        self.rate = self.create_rate(10)

    def joint_state_callback_follower(self, msg):
        # print(msg.name)
        if set(self.arm_joint_names).issubset(set(msg.name)):
            for i, joint in enumerate(self.arm_joint_names):
                # print(i, joint)
                index = msg.name.index(joint)
                self.arm_joint_positions_follower[i] = msg.position[index]
        self.follower_rev = True

    def joint_state_callback(self, msg):
        # print(msg.name)
        if set(self.arm_joint_names).issubset(set(msg.joint_names)):
            for i, joint in enumerate(self.arm_joint_names):
                # print(i, joint)
                index = msg.joint_names.index(joint)
                # self.get_logger().info(msg.points)
                # self.arm_joint_positions[i] = msg.points[index].position
                self.arm_joint_positions[i] = msg.points[0].positions[index]
        self.leader_rev = True
        if not (self.leader_rev and self.follower_rev):
            return
        # breakpoint()
        # print(self._eef_pose)
        joint_positions_new = self.robot.cal_follower_joint_angles(self._eef_pose, 
                                                                   self.arm_joint_positions,
                                                                   self.arm_joint_positions_follower).tolist()
        # breakpoint()
        joint_positions_new.append(self.arm_joint_positions[-1])  # Keep the gripper joint position
        joint_positions_new[-1] = np.clip(joint_positions_new[-1], self.gripper_min, self.gripper_max)
        # self.get_logger().info("Gripper position: {}".format(joint_positions_new[-1]))
        # self.get_logger().info("Kp: {}, Kd: {}".format(self.Kp, self.Kd))
        #we may need mod function
        self.joint_received = True
        # self.get_logger().info(
        #     f'Received joint states: {self.arm_joint_positions}, '
        #     # f'Gripper: {self.gripper_position}'
        # )

        target_joint_positions = joint_positions_new
        current_joint_positions = self.arm_joint_positions_follower[:-1]
        e_t = np.array(target_joint_positions)[:-1] - np.array(current_joint_positions)
        # self.get_logger().info("current joint angles:{}, et:{}".format(current_joint_positions, e_t))
        current_joint_positions = (
            np.array(current_joint_positions)
            + self.Kp * e_t
            + self.Kd * (np.array(e_t) - np.array(self.error_t_1))
        ).tolist()
        current_joint_positions.append(self.arm_joint_positions[-1])
        self.error_t_1 = e_t.tolist()

        arm_msg = JointTrajectory()
        arm_msg.joint_names = self.arm_joint_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = current_joint_positions
        arm_point.time_from_start.sec = 0
        arm_msg.points.append(arm_point)
        self.arm_publisher.publish(arm_msg)
        # self.get_logger().info(f'Arm command sent: {joint_positions_new}')
    
    @property
    def _eef_pose(self):
        return self.robot.fkine(self.arm_joint_positions)

def main(args=None):
    rclpy.init(args=args)

    tele_op_node = ConstrainedTeleOperation()

    rclpy.spin(tele_op_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    tele_op_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

