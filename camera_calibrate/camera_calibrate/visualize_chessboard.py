from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile
from sensor_msgs.msg import CompressedImage, Image
from scipy.spatial.transform import Rotation as R

import numpy as np
import pdb
import yaml
from tf2_ros import Buffer, TransformListener

serial_no_to_image_topic = {
    "_218622274409": "/camera/camera_wrist/color/image_rect_raw",
    "_826212070364": "/camera/camera_scene1/color/image_raw/compressed",
    "_941322072865": "/camera/camera_scene2/color/image_raw/compressed"
}


# def pose_to_homogeneous(pose):
#     """
#     Convert a Pose message to a homogeneous transformation matrix.
#     """
#     pose = pose.transform
#     rot = R.from_quat([pose.rotation.x, pose.rotation.y, 
#                      pose.rotation.z, pose.rotation.w]).as_matrix()
#     t = np.array([pose.translation.x, pose.translation.y, pose.translation.z])
#     H = np.eye(4)
#     H[:3, :3] = rot
#     H[:3, 3] = t
#     return H


class ChessboardVisualizer:
    def __init__(self, square_size_mm, chessboard_size=(9, 6), intrinsics=None):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.square_size_mm = square_size_mm  # Size of a square in millimeters
        self.chessboard_size = chessboard_size  # Number of internal corners per chessboard
        self._imgs = []
        self.intrinsics = intrinsics

    def add(self, image):
        self._imgs.append(image)

    def get_camera_matrix(self):
        return self.camera_matrix

    def get_dist_coeffs(self):
        return self.dist_coeffs

    @staticmethod
    def draw(img, corners, imgpts):
        corner = tuple(corners[0].ravel().astype("int32"))
        imgpts = imgpts.astype("int32")
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img

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
            ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, np.array(self.intrinsics['camera_matrix']), np.array(self.intrinsics['dist_coeffs']))
    
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, np.array(self.intrinsics['camera_matrix']), np.array(self.intrinsics['dist_coeffs']))

            img_drawn = self.draw(img_drawn,corners2,imgpts)
            # print(np.array(self.intrinsics['camera_matrix']))
            return img_drawn
        else:
            return img


class ChessboardVisualizerNode(Node):

    def __init__(self):
        super().__init__('chessboard_visualizer')

        self.declare_parameter('square_size_mm', 18.0, 
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Size of a square in millimeters'
            )
        )

        self.declare_parameter('chessboard_size', [9, 6],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER_ARRAY,
                description='Number of internal corners per chessboard'
            )
        )

        self.declare_parameter('serial_no', '_941322072865',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='camera serial numbers to use'
            )
        )   
        
        serial_number = self.get_parameter('serial_no').get_parameter_value().string_value
        with open('/root/ros2_ws/src/open_manipulator/camera_calibrate/config/cameras.yaml', 'r') as file:
            data = yaml.safe_load(file)
            intrinsics = data[serial_number]['intrinsics']

        image_topic = serial_no_to_image_topic[self.get_parameter('serial_no').get_parameter_value().string_value]

        self.square_size_mm = self.get_parameter('square_size_mm').get_parameter_value().double_value
        self.chessboard_size = list(self.get_parameter('chessboard_size').get_parameter_value().integer_array_value)

        self.visualizer = ChessboardVisualizer(
            square_size_mm=self.square_size_mm,
            chessboard_size=self.chessboard_size,
            intrinsics=intrinsics
        )
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            # CompressedImage,
            Image,
            image_topic,
            self.listener_callback,
            QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST),
        )
        self.publisher = self.create_publisher(
            Image,
            'image_with_chessboard',
            QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST)
        )

        # self.tf_buffer = Buffer()
        # self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.timer = self.create_timer(0.1, self.lookup_eef_to_base)   
        # self.T_static = None
        # self.latest_pose = None
        # self.latest_pose_received = False
        # while not self.tf_buffer.can_transform('link6', 'end_effector_link', rclpy.time.Time()):
        #     self.get_logger().info('Waiting for transform from link6 to end_effector_link...')
        #     rclpy.spin_once(self, timeout_sec=1.0)

    def listener_callback(self, msg):
        # Directly process the image in the callback
        try:
            # if self.msg_type == Image:
            #     cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # elif self.msg_type == CompressedImage:
            #     cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # else:
            #     raise ValueError(f"Unsupported message type: {self.msg_type}")
            # cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            img_with_chessboard = self.visualizer.visualize_chessboard(cv_image)
            self.publisher.publish(self.bridge.cv2_to_imgmsg(img_with_chessboard, encoding="bgr8"))
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    # def lookup_eef_to_base(self):
    #     if self.T_static is None:
    #         self.T_static = pose_to_homogeneous(self.tf_buffer.lookup_transform(
    #             target_frame='link6',
    #             source_frame='end_effector_link',  # ⬅️ end-effector
    #             time=rclpy.time.Time())
    #     )  # 0 = latest available

    #     try:
    #         transform = self.tf_buffer.lookup_transform(
    #             target_frame='link0',       # ⬅️ base frame
    #             source_frame='link6',         # ⬅️ end-effector
    #             time=rclpy.time.Time())     # 0 = latest available
    #         self.get_logger().info("Timestamp: {}".format(transform.header.stamp))
    #         self.get_logger().info(
    #             f"Transform from link0 to link6:\n{transform.transform}"
    #         )
    #         T_transform = pose_to_homogeneous(transform).dot(self.T_static)
    #         self.get_logger().info(
    #             f"Transform from link0 to end_effector_link:\n{T_transform}")
    #         self.latest_pose = T_transform
    #         # tVec = np.array([transform.translation.x, transform.translation.y, transform.translation.z])
    #         # rMat = R.from_quat([transform.rotation.x, transform.rotation.y, 
    #         #                     transform.rotation.z, transform.rotation.w]).as_matrix()
    #         # self.latest_pose = np.eye(4)
    #         # self.latest_pose[:3, :3] = rMat
    #         # self.latest_pose[:3, 3] = tVec
    #         self.latest_pose_received = True

        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {e}")



def main():
    rclpy.init()
    node = ChessboardVisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
