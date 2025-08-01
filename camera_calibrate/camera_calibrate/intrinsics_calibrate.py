import cv2
import numpy as np
import glob
import os


import rclpy
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf_transformations
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

import select
import sys
import termios
import threading
import time
import tty
import uuid
from control_msgs.action import GripperCommand
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile
from sensor_msgs.msg import CompressedImage, Image

import numpy as np
import pdb
from scipy.spatial.transform import Rotation as R
import pdb
import dataclasses
import pdb

@dataclasses.dataclass
class RosCameraConfig:
    width: int = 640
    height: int = 480
    channels: int = 3
    fps: float = None
    topic_name: str = "/camera/camera/color/image_rect_raw"
    msg_type: type = CompressedImage

class IntrinsicsEstimator:
    def __init__(self, square_size_mm, chessboard_size=(9, 6)):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.square_size_mm = square_size_mm  # Size of a square in millimeters
        self.chessboard_size = chessboard_size  # Number of internal corners per chessboard
        self._imgs = []

    def add(self, image):
        self._imgs.append(image)

    def get_camera_matrix(self):
        return self.camera_matrix

    def get_dist_coeffs(self):
        return self.dist_coeffs

    def visualize_chessboard(self, img):
        # Termination criteria for cornerSubPix
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        # img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        # breakpoint()
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        # If found, add object points and image points
        if ret:
            # Refine corner locations to sub-pixel accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners (optional)
            img_drawn = cv2.drawChessboardCorners(img.copy(), self.chessboard_size, corners2, ret)
            return img_drawn
        else:
            return img

    def calibrate(self):
        """
        Returns:
        - ret: RMS re-projection error.
        - camera_matrix: Camera intrinsic matrix.
        - dist_coeffs: Distortion coefficients.
        - rvecs: Rotation vectors.
        - tvecs: Translation vectors.
        """
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
        for idx, img in enumerate(self._imgs):
            # img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            # breakpoint()
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            if not ret:
                self._imgs.remove(img)
                continue 
            # If found, add object points and image points
            if ret:
                objpoints.append(objp)
                # Refine corner locations to sub-pixel accuracy
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners (optional)
                # img_drawn = cv2.drawChessboardCorners(img.copy(), self.chessboard_size, corners2, ret)
                # cv2.imshow('Chessboard Detection', img_drawn)
                # cv2.waitKey(5000)  # Display each image for 100ms
            else:
                print(f"Chessboard corners not found in image. Skipping.")

        # cv2.destroyAllWindows()

        if not objpoints or not imgpoints:
            print("No chessboard corners were detected in any image. Calibration cannot proceed.")
            return

        # Perform camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)

        # Calculate re-projection error
        mean_error = 0
        total_points = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
            mean_error += error**2
            total_points += len(objpoints[i])

        mean_error = np.sqrt(mean_error / total_points)
        print(f"Calibration RMS Re-projection Error: {ret}")
        print(f"Mean Re-projection Error: {mean_error}")

        print("\nCamera Matrix:")
        print(camera_matrix)

        print("\nDistortion Coefficients:")
        print(dist_coeffs.ravel())

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def save_calibration(self, save_path):
        """
        Save the camera calibration results to a file.

        Parameters:
        - camera_matrix: Camera intrinsic matrix.
        - dist_coeffs: Distortion coefficients.
        - save_path: Directory to save the calibration file.
        """
        calibration_file = 'calibration_result.npz'
        np.savez(calibration_file, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs)
        print(f"\nCalibration results saved to {calibration_file}")

class KeyboardController(Node):
    def __init__(self, stop_event: threading.Event, 
                 topic_name="/camera_scene2/camera_scene2/color/image_raw", 
                 fps=None, 
                 node_name=None, 
                 msg_type=Image,
                 square_size_mm=28.0,
                 chessboard_size=(9, 6)):
        if node_name is None:
            node_name = f"ros_image_subscriber_{uuid.uuid4().hex[:8]}_keyboard_controller"
        super().__init__(node_name)
        # breakpoint()
        self.bridge = CvBridge()
        self.stop_event = stop_event
        self.fps = fps
        self.msg_type = msg_type

        self.latest_msg = None
        self.lock = threading.Lock()

        self.subscription = self.create_subscription(
            self.msg_type,
            topic_name,
            self.listener_callback,
            QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST),
        )
        

        self.latest_img = None
        self.latest_img_received = False
        # Only create a timer if fps is not None
        if self.fps is not None:
            self.timer = self.create_timer(1.0 / self.fps, self.process_image)
        
        #keyboard control variables
        self.last_command_time = time.time()
        self.command_interval = 0.02
        self.running = True  # for thread loop control
        self.rate = self.create_rate(10)

        #instantiate intrinsics estimator
        self.intrinsics_estimator = IntrinsicsEstimator(square_size_mm=square_size_mm, chessboard_size=chessboard_size)

    def listener_callback(self, msg):
        if self.stop_event.is_set():
            rclpy.shutdown()
            return

        if self.fps is None:
            # Directly process the image in the callback
            try:
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
        else:
            with self.lock:
                self.latest_msg = msg

    def process_image(self):
        if self.stop_event.is_set():
            rclpy.shutdown()
            return

        with self.lock:
            if self.latest_msg is None:
                return
            msg = self.latest_msg
            self.latest_msg = None

        try:
            if self.msg_type == Image:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            elif self.msg_type == CompressedImage:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="rgb8")
            else:
                raise ValueError(f"Unsupported message type: {self.msg_type}")
            self.image_callback(cv_image)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

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
        while  not self.latest_img_received and rclpy.ok() and self.running:
            self.get_logger().info('Waiting for initial image...')
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
                        # breakpoint()
                        self.latest_img_received = False
                        while not self.latest_img_received and rclpy.ok():
                            self.get_logger().info('Waiting for image...')
                            rclpy.spin_once(self, timeout_sec=1.0)

                        self.intrinsics_estimator.add(self.latest_img)
                        self.intrinsics_estimator.calibrate()
                        self.get_logger().info('Image added for calibration.')
                        # cv2.imshow('latest image', self.latest_img)
                        # cv2.waitKey(0)
                        # self.intrinsics_estimator.calibrate()
                    elif key == '2':
                        self.intrinsics_estimator._imgs.pop()
                        # self.intrinsics_estimator.calibrate()
                        self.get_logger().info('Calibration completed.')
                        # self.intrinsics_estimator.save_calibration('calibration_results.npz')
                    elif key == '3':
                        # self.intrinsics_estimator._imgs.pop()
                        # self.intrinsics_estimator.calibrate()
                        # self.get_logger().info('Calibration completed.')
                        self.intrinsics_estimator.save_calibration('calibration_results.npz')
                    self.last_command_time = current_time

        except Exception as e:
            self.get_logger().error(f'Exception in run loop: {e}')


def main():
    rclpy.init()
    node = KeyboardController(stop_event=threading.Event(), chessboard_size=(8, 5), square_size_mm=28.0)
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
