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

class ExtrinsicsEstimator:
    def __init__(self, square_size_mm, chessboard_size=(8, 5), intrinsics_file='intrinsics_cam_scene1.npz'):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.square_size_mm = square_size_mm  # Size of a square in millimeters
        self.chessboard_size = chessboard_size  # Number of internal corners per chessboard
        intrinsics = np.load(intrinsics_file)
        self.camera_matrix = intrinsics['camera_matrix']
        self.dist_coeffs = intrinsics['dist_coeffs']
        self._cache = []

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
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            # Refine corner locations to sub-pixel accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            objpoints.append(objp)


        R_target2cam = []
        T_target2cam = []

        for i in range(len(objpoints)):
            objp_current = objpoints[i]
            imgp_current = imgpoints[i]
            ret, rvec, tvec = cv2.solvePnP(objp_current, imgp_current, self.camera_matrix, self.dist_coeffs)
            print("ret: {}, rvec: {}, tvec: {}".format(ret, rvec, tvec))
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
        calibration_file = os.path.join('extrinsics.npz')
        np.savez(calibration_file, R=self.R_handeye, T=self.T_handeye)
        print(f"\nCalibration results saved to {calibration_file}")

class KeyboardController(Node):
    def __init__(self, stop_event: threading.Event, 
                 topic_name="/camera/camera_scene2/color/image_raw", 
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

        self.subscription_eef = self.create_subscription(
            Pose,
            "/follower/pose",
            self.listener_pose_callback,
            QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST),
        )

        self.latest_img = None
        self.latest_img_received = False
        self.latest_pose = np.eye(4)
        self.latest_pose_received = False
        # Only create a timer if fps is not None
        if self.fps is not None:
            self.timer = self.create_timer(1.0 / self.fps, self.process_image)
        
        #keyboard control variables
        self.last_command_time = time.time()
        self.command_interval = 0.02
        self.running = True  # for thread loop control
        self.rate = self.create_rate(10)

        #instantiate intrinsics estimator
        self.extrinsics_estimator = ExtrinsicsEstimator(square_size_mm=square_size_mm, chessboard_size=chessboard_size)

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
                        self.latest_pose_received = False
                        while not self.latest_img_received and rclpy.ok():
                            self.get_logger().info('Waiting for image...')
                            rclpy.spin_once(self, timeout_sec=1.0)

                        while not self.latest_pose_received and rclpy.ok():
                            self.get_logger().info('Waiting for pose...')
                            rclpy.spin_once(self, timeout_sec=1.0)

                        self.extrinsics_estimator.add(self.latest_img, self.latest_pose)
                        self.extrinsics_estimator.calibrate()
                        self.get_logger().info('Image added for calibration.')
                        # cv2.imshow('latest image', self.latest_img)
                        # cv2.waitKey(0)
                        # self.intrinsics_estimator.calibrate()
                    elif key == '2':
                        self.extrinsics_estimator._cache.pop()
                        # self.intrinsics_estimator.calibrate()
                        self.get_logger().info('Calibration completed.')
                        # self.intrinsics_estimator.save_calibration('calibration_results.npz')
                    elif key == '3':
                        # self.intrinsics_estimator._imgs.pop()
                        # self.intrinsics_estimator.calibrate()
                        # self.get_logger().info('Calibration completed.')
                        self.extrinsics_estimator.save_calibration('extrinsics_calibration_results.npz')
                    self.last_command_time = current_time

        except Exception as e:
            self.get_logger().error(f'Exception in run loop: {e}')


def main():
    rclpy.init()
    node = KeyboardController(stop_event=threading.Event(), chessboard_size=(8, 5), square_size_mm=18.0)
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
