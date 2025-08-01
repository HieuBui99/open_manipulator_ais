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

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

class ChessboardVisualizer:
    def __init__(self, square_size_mm, chessboard_size=(9, 6), intrinsics_file='intrinsics_cam_scene1.npz'):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.square_size_mm = square_size_mm  # Size of a square in millimeters
        self.chessboard_size = chessboard_size  # Number of internal corners per chessboard
        self._imgs = []
        self.intrinsics = np.load(intrinsics_file)

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
        # Prepare object points based on the actual chessboard dimensions
        objp = np.zeros((self.chessboard_size[1]*self.chessboard_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                            0:self.chessboard_size[1]].T.reshape(-1, 2)
        
        axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
        # breakpoint()
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        # If found, add object points and image points
        if ret:
            # Refine corner locations to sub-pixel accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners (optional)
            img_drawn = cv2.drawChessboardCorners(img.copy(), self.chessboard_size, corners2, ret)

            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, self.intrinsics['camera_matrix'], self.intrinsics['dist_coeffs'])
    
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, self.intrinsics['camera_matrix'], self.intrinsics['dist_coeffs'])
    
            img_drawn = draw(img_drawn,corners2,imgpts)
            return img_drawn
        else:
            return img


class KeyboardController(Node):
    def __init__(self, stop_event: threading.Event, 
                 topic_name="/camera/camera_scene2/color/image_raw", 
                 fps=None, 
                 node_name=None, 
                 msg_type=Image,
                 square_size_mm=18.0,
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

        self.publisher = self.create_publisher(
            self.msg_type,
            topic_name + '_with_chessboard',
            10,
        )
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
        self.visualizer = ChessboardVisualizer(square_size_mm=square_size_mm, chessboard_size=chessboard_size)

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
                img_with_chessboard = self.visualizer.visualize_chessboard(cv_image)
                self.publisher.publish(self.bridge.cv2_to_imgmsg(img_with_chessboard, encoding="bgr8"))
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
            self.image_callback(cv_image)#????????????????????
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")




def main():
    rclpy.init()
    node = KeyboardController(stop_event=threading.Event(), chessboard_size=(8, 5), square_size_mm=18.0)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
