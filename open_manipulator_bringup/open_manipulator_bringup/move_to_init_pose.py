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
        self.R0 = np.array([[0,0,1],
                            [1,0,0],
                            [0,1,0]])
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


    
    # def test(self):
    #     curent_pose
    #     target_pose
    #     while current_pose != target_p:


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
        # rotation = self.R0
        rotation = np.array(transform.R)  # eef pose of the leader
        R_rel = rotation @ self.R0.T # compute the relative rotation between  the initial leader pose and the current leader pose
        rot_rel = R.from_matrix(R_rel) # LIe bracket --> get the twist vector
        axis_angle = rot_rel.as_rotvec() # convert to axis angle representation, angle * rotation axis
        # print("axis_angle", axis_angle/np.linalg.norm(axis_angle))
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
            joint_angles[invalid_indices] = np.array(q_follower[:-1])[invalid_indices]
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
            JointTrajectory, '/arm_controller/joint_trajectory', 30
        )

        # Subscriber for joint states
        self.subscription = self.create_subscription(
            JointState, '/leader/joint_states', self.joint_state_callback, 30
        )

        self.subscription_follower = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback_follower, 30
        )
        self.leader_rev = False
        self.follower_rev = False
        self.arm_joint_positions = [0.0] * 7
        self.arm_joint_positions_follower = [0.0] * 7
        self.error_t_1 = [0.0] * 6
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

        self.traj = JointTrajectory()
        self.running = True  # for thread loop control
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
        if set(self.arm_joint_names).issubset(set(msg.name)):
            for i, joint in enumerate(self.arm_joint_names):
                # print(i, joint)
                index = msg.name.index(joint)
                self.arm_joint_positions[i] = msg.position[index]
        self.leader_rev = True
        if not (self.leader_rev and self.follower_rev):
            return
        # breakpoint()
        # print(self._eef_pose)
        joint_positions_new = self.robot.cal_follower_joint_angles(self._eef_pose, 
                                                                   self.arm_joint_positions,
                                                                   self.arm_joint_positions_follower).tolist()
        # breakpoint()
        # joint_positions_new.append(self.arm_joint_positions[-1])  # Keep the gripper joint position
        #we may need mod function
        self.joint_received = True
        

        target_joint_positions = joint_positions_new
        current_joint_positiosn = self.arm_joint_positions_follower[:-1]
        e_t = np.array(target_joint_positions) - np.array(current_joint_positiosn)
        increment = 0.006
        Kp = 0.03
        Kd = 0.01
        # interpolate init_pose and target_pose
        print("target_joint_positions", target_joint_positions)
        print("current_joint_positiosn", current_joint_positiosn)
        if  np.linalg.norm(np.array(target_joint_positions) - np.array(current_joint_positiosn)) < increment:
            print("Target reached, shutting down...")
            self.destroy_node()
            rclpy.shutdown()
            sys.exit(0)
        current_joint_positiosn = np.array(current_joint_positiosn) +\
         Kp * (np.array(target_joint_positions) - np.array(current_joint_positiosn)) +\
            Kd * (np.array(e_t) - np.array(self.error_t_1))
        self.error_t_1 = e_t.tolist()
        # current_joint_positiosn = np.array(current_joint_positiosn) + increment * (np.array(target_joint_positions) - np.array(current_joint_positiosn)) / np.linalg.norm(np.array(target_joint_positions) - np.array(current_joint_positiosn))
        current_joint_positiosn = current_joint_positiosn.tolist()
        # joint_positions_new = self.robot.cal_follower_joint_angles(self._eef_pose, 
        #                                                         current_pose,
        #                                                         self.arm_joint_positions_follower).tolist()

        # self.get_logger().info(
        #     f'Received joint states: {self.arm_joint_positions}, '
        #     # f'Gripper: {self.gripper_position}'
        # )
        # joint_positions_new.append(self.arm_joint_positions[-1]) 
        current_joint_positiosn.append(self.arm_joint_positions[-1])
        arm_msg = JointTrajectory()
        arm_msg.joint_names = self.arm_joint_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = current_joint_positiosn
        arm_point.time_from_start.sec = 0
        arm_msg.points.append(arm_point)
        self.arm_publisher.publish(arm_msg)
        # self.follower_rev = False
        # self.leader_rev = False
        # # self.get_logger().info(f'Arm command sent: {joint_positions_new}')
    
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

