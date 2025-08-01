import yaml

import cv2
from cv_bridge import CvBridge
from cv2 import calibrateHandEye, CALIB_HAND_EYE_TSAI, Rodrigues
import numpy as np
import os
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
import select
import sys
import termios
import threading
import time
import tty
import uuid
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile
from sensor_msgs.msg import CompressedImage, Image
from scipy.spatial.transform import Rotation as R
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from tf2_ros import Buffer, TransformListener
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import dataclasses
import pdb

import yaml
from ament_index_python.packages import get_package_share_directory

def invert_homogeneous(H):
    R = H[:3, :3]          # rotation part (3×3)
    t = H[:3, 3]           # translation (3,)
    R_T = R.T              # since R is orthonormal, R.T == R⁻¹
    H_inv = np.eye(4)
    H_inv[:3, :3] = R_T
    H_inv[:3, 3]  = -R_T @ t
    return H_inv

def pose_to_homogeneous(pose):
    """
    Convert a Pose message to a homogeneous transformation matrix.
    """
    pose = pose.transform
    rot = R.from_quat([pose.rotation.x, pose.rotation.y, 
                     pose.rotation.z, pose.rotation.w]).as_matrix()
    t = np.array([pose.translation.x, pose.translation.y, pose.translation.z])
    H = np.eye(4)
    H[:3, :3] = rot
    H[:3, 3] = t
    return H

serial_no_to_image_topic = {
    "_218622274409": "/camera/camera_wrist/color/image_rect_raw",
    "_826212070364": "/camera/camera_scene1/color/image_raw/compressed",
    "_941322072865": "/camera/camera_scene2/color/image_raw/compressed"
}


class ExtrinsicsEstimator:
    def __init__(self, config_pth, serial_number, square_size_mm, chessboard_size=(8, 5)):

        with open(config_pth, 'r') as file:
            self.data = yaml.safe_load(file)
        self.config_pth = config_pth
        self.serial_number = serial_number
        self.camera_matrix = np.array(self.data[serial_number]['intrinsics']['camera_matrix'])
        self.dist_coeffs = np.array(self.data[serial_number]['intrinsics']['dist_coeffs'])
        self.square_size_mm = square_size_mm  # Size of a square in millimeters
        self.chessboard_size = chessboard_size  # Number of internal corners per chessboard
        self._cache = []
        self.R_hand_eye = np.eye(3)
        self.T_hand_eye = np.zeros(3)

    def add(self, image, pose):
        self._cache.append([image, pose])

    def calibrate(self):
        """
        Returns:
        - ret: RMS re-projection error.
        - camera_matrix: Camera intrinsic matrix.
        - dist_coeffs: Distortion coefficients.
        - rvecs: Rotation vectors.
        - tvecs: Translation vectors.
        """
        print("square_size_mm: {}, chessboard_size: {}".format(self.square_size_mm, self.chessboard_size))
        print("serial_number: {}".format(self.serial_number))
        if len(self._cache) < 3:
            return
        # breakpoint()
        # Step 3: Extract and Convert Robot Poses
        imgs, Hs = zip(*self._cache)
        Hs = np.array(Hs)
        R_gripper2base = Hs[:, :3, :3]  # Extract rotation matrices
        t_gripper2base = Hs[:, :3, 3]*1000  # Extract
        # Termination criteria for cornerSubPix
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)

        # Prepare object points based on the actual chessboard dimensions
        objp = np.zeros((self.chessboard_size[1]*self.chessboard_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                            0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size_mm  # Scale by square size to get real-world coordinates

        # Arrays to store object points and image points from all images
        objpoints = []  # 3D points in real-world space
        imgpoints = []  # 2D points in image plane
        # Process each image
        for idx, img in enumerate(imgs):
            # img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            # breakpoint()
            try:
                ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
                # Refine corner locations to sub-pixel accuracy
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                objpoints.append(objp)
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue


        R_target2cam = []
        T_target2cam = []
        for i in range(len(objpoints)):
            objp_current = objpoints[i]
            imgp_current = imgpoints[i]
            ret, rvec, tvec = cv2.solvePnP(objp_current, imgp_current, self.camera_matrix, self.dist_coeffs)
            # print("ret: {}, rvec: {}, tvec: {}".format(ret, rvec, tvec))
            if ret:
                R_cam, _ = Rodrigues(rvec)  # Convert rotation vector to rotation matrix
                T_cam = tvec.flatten()
                R_target2cam.append(R_cam)
                T_target2cam.append(T_cam)

        # Perform calibration
        R_hand_eye, T_hand_eye = calibrateHandEye(
            R_gripper2base,#eef in base frame
            t_gripper2base,
            R_target2cam,#target in camera frame
            T_target2cam,
            method=CALIB_HAND_EYE_TSAI
        )

        print("Hand-Eye Calibration Results:")
        print("Rotation Matrix (R_hand_eye):\n", R_hand_eye)
        print("Translation Vector (T_hand_eye):\n", T_hand_eye)
        self.R_hand_eye = R_hand_eye
        self.T_hand_eye = T_hand_eye

    def save_calibration(self):
        """
        Save the camera calibration results to a file.

        Parameters:
        - camera_matrix: Camera intrinsic matrix.
        - dist_coeffs: Distortion coefficients.
        - save_path: Directory to save the calibration file.
        """

        # calibration_file = os.path.join('extrinsics.npz')
        # np.savez(calibration_file, R=self.R_hand_eye, T=self.T_hand_eye)
        # print(f"\nCalibration results saved to {calibration_file}")
        self.data[self.serial_number]['extrinsics'] = {
            'rotation': self.R_hand_eye.tolist(),
            'translation': self.T_hand_eye.tolist()
        }
        with open(self.config_pth, 'w') as file:
            yaml.dump(self.data, file)

# import yaml

# import yaml

# # A list of dictionaries
# data = {serial_wrist:{'intrinsics': {"camera_matrix": intr2['camera_matrix'].tolist(), "dist_coeffs":intr2['dist_coeffs'].tolist()}, 'extrinsics': {"rotation": extr1['R'].tolist(), "translation": extr1["T"].tolist()}},

#     serial_cam1: {'intrinsics': {"camera_matrix": intr2['camera_matrix'].tolist(), "dist_coeffs":intr2['dist_coeffs'].tolist()}, 'extrinsics': {"rotation": extr2['R'].tolist(), "translation": extr2["T"].tolist()}},
#     serial_cam2: {'intrinsics': {"camera_matrix": intr1['camera_matrix'].tolist(), "dist_coeffs":intr1['dist_coeffs'].tolist()}, 'extrinsics': {"rotation": extr3['R'].tolist(), "translation": extr3["T"].tolist()}}}



class ExtrinsicsEstimatorNode(Node):
    def __init__(self, stop_event: threading.Event):
        super().__init__('extrinsics_estimator_node')
        # breakpoint()
        # self.declare_parameter('square_size_mm', 18.0, 
        #     ParameterDescriptor(
        #         type=ParameterType.PARAMETER_DOUBLE,
        #         description='Size of a square in millimeters'
        #     )
        # )

        # self.declare_parameter('chessboard_size', [9, 6],
        #     ParameterDescriptor(
        #         type=ParameterType.PARAMETER_INTEGER_ARRAY,
        #         description='Number of internal corners per chessboard'
        #     )
        # )

        # self.declare_parameter('serial_number', '_826212070364',
        #     ParameterDescriptor(
        #         type=ParameterType.PARAMETER_STRING,
        #         description='Serial number of the camera'
        #     )
        # )

        # # self.declare_parameter('image_topic', '/camera/camera_scene1/color/image_raw',
        # #     ParameterDescriptor(
        # #         type=ParameterType.PARAMETER_STRING,
        # #         description='Topic to subscribe to for images'
        # #     )
        # # )

        # self.declare_parameter('eye_out_of_hand', False,
        #     ParameterDescriptor(
        #         type=ParameterType.PARAMETER_BOOL,
        #         description='If True, the camera is mounted on the end-effector'
        #     )
        # )
        # Get absolute path to the config file
        config_file_pth = os.path.join(
            get_package_share_directory('camera_calibrate'),
            'config',
            'extrinsics_estimate.yaml'
        )

        camera_file_pth = os.path.join(
            get_package_share_directory('camera_calibrate'),
            'config',
            'cameras.yaml'
        )

        serial_no_to_frame = {
        "_218622274409": "camera_wrist_color_frame_estimate",
        "_826212070364": "camera_scene1_color_frame_estimate",
        "_941322072865": "camera_scene2_color_frame_estimate"}

        # Load the YAML
        with open(config_file_pth, 'r') as f:
            config = yaml.safe_load(f)['/**']['ros__parameters']
        # breakpoint()
        self.eye_out_of_hand = config['eye_out_of_hand']
        self.extrinsics_estimator = ExtrinsicsEstimator(
            config_pth=camera_file_pth,
            serial_number=config['serial_no'],
            square_size_mm=config['square_size_mm'],
            chessboard_size=tuple(config['chessboard_size'])
        )
        self.frame = serial_no_to_frame[config['serial_no']]
        # breakpoint()
        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        # while not self.tf_buffer.can_transform('link6', 'end_effector_link', rclpy.time.Time()):
        #     self.get_logger().info('Waiting for transform from link6 to end_effector_link...')
        #     rclpy.spin_once(self, timeout_sec=1.0)
        # self.timer = self.create_timer(0.1, self.lookup_eef_to_base)        
        # self.T_static = None

        self.bridge = CvBridge()
        self.stop_event = stop_event
        # self.msg_type = CompressedImage
        self.msg_type = Image

        self.subscription = self.create_subscription(
            self.msg_type,
            serial_no_to_image_topic[config['serial_no']],
            self.listener_image_callback,
            QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST),
        )

        self.subscription_eef = self.create_subscription(
            Pose,
            "/eef_pose",
            self.listener_pose_callback,
            QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST),
        )

        self.latest_img = None
        self.latest_img_received = False
        self.latest_pose = np.eye(4)
        self.latest_pose_received = False
        
        #keyboard control variables
        self.last_command_time = time.time()
        self.command_interval = 0.02
        self.running = True  # for thread loop control
        self.rate = self.create_rate(10)

    def listener_image_callback(self, msg):
        if self.stop_event.is_set():
            rclpy.shutdown()
            return

        # Directly process the image in the callback
        try:
            # print(msg)
            if self.msg_type == Image:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            elif self.msg_type == CompressedImage:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                raise ValueError(f"Unsupported message type: {self.msg_type}")
            self.latest_img = cv_image
            self.latest_img_received = True
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def listener_pose_callback(self, msg):
        tVec = np.array([msg.position.x, msg.position.y, msg.position.z])
        rMat = R.from_quat([msg.orientation.x, msg.orientation.y, 
                            msg.orientation.z, msg.orientation.w]).as_matrix()
        self.latest_pose = np.eye(4)
        self.latest_pose[:3, :3] = rMat
        self.latest_pose[:3, 3] = tVec
        self.latest_pose_received = True

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

    def run(self):
        while  not self.latest_img_received and not self.latest_pose_received and rclpy.ok() and self.running:
            self.get_logger().info('Waiting for initial image...')
            rclpy.spin_once(self, timeout_sec=1.0)

        while not self.latest_pose_received and rclpy.ok() and self.running:
            self.get_logger().info('Waiting for initial pose...')
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
                    if key == '\x1b':  # ESC
                        self.running = False
                        break

                    elif key == '1':
                        self.latest_img_received = False
                        self.latest_pose_received = False
                        while not self.latest_img_received and rclpy.ok():
                            self.get_logger().info('Waiting for image... Spinning once.')
                            rclpy.spin_once(self, timeout_sec=1.0)

                        while not self.latest_pose_received and rclpy.ok():
                            self.get_logger().info('Waiting for pose... Spinning once.')
                            rclpy.spin_once(self, timeout_sec=1.0)
                        self.get_logger().info('latest pose: {}'.format(self.latest_pose))
                        if self.eye_out_of_hand:
                            self.extrinsics_estimator.add(self.latest_img, invert_homogeneous(self.latest_pose))
                        else:
                            self.extrinsics_estimator.add(self.latest_img, self.latest_pose)
                        
                        self.extrinsics_estimator.calibrate()
                        rotation_tmp = R.from_matrix(self.extrinsics_estimator.R_hand_eye).as_quat()
                        translation_tmp = self.extrinsics_estimator.T_hand_eye.squeeze()/1000.  # Convert to meters
                        # breakpoint()
                        t = TransformStamped()
                        t.header.stamp = self.get_clock().now().to_msg()
                        t.header.frame_id = 'world'  # Base frame
                        t.child_frame_id = self.frame  # Child frame
                        t.transform.translation.x = translation_tmp[0]
                        t.transform.translation.y = translation_tmp[1]
                        t.transform.translation.z = translation_tmp[2]
                        t.transform.rotation.x = rotation_tmp[0]
                        t.transform.rotation.y = rotation_tmp[1]
                        t.transform.rotation.z = rotation_tmp[2]
                        t.transform.rotation.w = rotation_tmp[3]
                        self.tf_broadcaster.sendTransform(t)
                        self.get_logger().info('Image and pose added for calibration.')                     
                        self.get_logger().info('Image added for calibration.')
                    elif key == '2':
                        self.extrinsics_estimator._cache.pop()
                        self.get_logger().info('Calibration completed.')
                    elif key == '3':
                        self.extrinsics_estimator.save_calibration()
                        self.get_logger().info('Calibration saved.')
                    self.last_command_time = current_time

        except Exception as e:
            self.get_logger().error(f'Exception in run loop: {e}')


def main():
    rclpy.init()
    node = ExtrinsicsEstimatorNode(stop_event=threading.Event())
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
