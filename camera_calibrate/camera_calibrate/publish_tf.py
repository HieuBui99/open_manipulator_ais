import math

from geometry_msgs.msg import TransformStamped

import numpy as np

from tf2_ros.buffer import Buffer


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from rclpy.qos import HistoryPolicy, QoSProfile
from cv_bridge import CvBridge
import threading
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
from marker_detect import detect_apriltag_from_images

#get the extrinsics of two cameras
extrinsics1 = np.load('/root/ros2_ws/src/open_manipulator/camera_calibrate/camera_calibrate/extrinsics_cam_scene1.npz')
extrinsics2 = np.load('/root/ros2_ws/src/open_manipulator/camera_calibrate/camera_calibrate/extrinsics_cam_scene2.npz')
T_cam1 = np.eye(4)
T_cam1[:3, :3] = extrinsics1['R']
T_cam1[:3, 3] = extrinsics1['T'].squeeze()/1000
T_cam2 = np.eye(4)
T_cam2[:3, :3] = extrinsics2['R']
T_cam2[:3, 3] = extrinsics2['T'].squeeze()/1000


#get the intrinsics of two cameras
toParam = lambda x: tuple([x[0,0], x[1,1], x[0,2], x[1,2]])
intrinsics1 = np.load('/root/ros2_ws/src/open_manipulator/camera_calibrate/camera_calibrate/intrinsics_cam_scene1.npz')
intrinsics2 = np.load('/root/ros2_ws/src/open_manipulator/camera_calibrate/camera_calibrate/intrinsics_cam_scene2.npz')
M_cam1 = intrinsics1['camera_matrix']
M_cam2 = intrinsics2['camera_matrix']


extrinsics = {
    "941322072865": T_cam1,
    "826212070364": T_cam2,
}

intrinsics = {
    "941322072865": toParam(M_cam1),
    "826212070364": toParam(M_cam2),
}


camera_serials = {
    "cam1": "941322072865",
    "cam2": "826212070364",
}

def generate_tag_transformations():

    T_tag0_com = np.eye(4)
    T_tag0_com[3,-1] = -0.02

    T_tag1_com = np.eye(4)
    T_tag1_com[:3, :3] = R.from_euler('x', -np.pi/2).as_matrix()
    T_tag1_com[3,-1] = -0.02

    T_tag2_com = np.eye(4)
    T_tag2_com[:3, :3] = R.from_euler('y', -np.pi/2).as_matrix().dot(T_tag1_com[:3, :3])
    T_tag2_com[3,-1] = -0.06

    T_tag3_com = np.eye(4)
    T_tag3_com[:3, :3] = R.from_euler('y', -np.pi).as_matrix().dot(T_tag1_com[:3, :3])
    T_tag3_com[3,-1] = -0.015

    T_tag4_com = np.eye(4)
    T_tag4_com[:3, :3] = R.from_euler('y', np.pi/2).as_matrix().dot(T_tag1_com[:3, :3])
    T_tag4_com[3,-1] = -0.06

    tag_transforms = dict(tag0_com=T_tag0_com, 
                          tag1_com=T_tag1_com, 
                          tag2_com=T_tag2_com, 
                          tag3_com=T_tag3_com, 
                          tag4_com=T_tag4_com)

    return tag_transforms



class ObjectPosePublisher(Node):

    def __init__(self):
        super().__init__('turtle_tf2_frame_publisher')

        self.bridge = CvBridge()
        self.stop_event = threading.Event()
        self.fps = None
        self.msg_type = CompressedImage

        self.latest_msg = None
        self.lock = threading.Lock()

        self.subsciption_cam1 = self.create_subscription(
            CompressedImage,
            '/camera/camera_scene1/color/image_raw/compressed',
            self.handle_image_cam1,
            QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST)
        )
        self.subsciption_cam2 = self.create_subscription(
            CompressedImage,
            '/camera/camera_scene2/color/image_raw/compressed',
            self.handle_image_cam2,
            QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST)
        )
        self.latest_images = {
            camera_serials["cam1"]: None,
            camera_serials["cam2"]: None,
        }
        # Only create a timer if fps is not None
        if self.fps is not None:
            self.timer = self.create_timer(1.0 / self.fps, self.process_image)
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tag_transforms = generate_tag_transformations()

        self.object_poses = {
            "object0": None,
            "object1": None,
            "object2": None,
            "object3": None,
            "object4": None,
            "object5": None
        }

    def handle_image_cam1(self, msg):
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
                self.latest_images[camera_serials["cam1"]] = cv_image
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
        else:
            with self.lock:
                self.latest_msg = msg

        if self.latest_images[camera_serials["cam2"]] is not None and self.latest_images[camera_serials["cam1"]] is not None:
            images = [
                self.latest_images[camera_serials["cam1"]],
                self.latest_images[camera_serials["cam2"]]
            ]
            serials = [camera_serials["cam1"], camera_serials["cam2"]]
            
            # change this to real camera matrices
            # camera_matrices = {
            #     camera_serials["cam1"]: intrinsics[camera_serials["cam1"]],
            #     camera_serials["cam2"]: np.eye(3)
            # }

            results = detect_apriltag_from_images(images, serials, intrinsics, tag_size=0.026)
            
            # 6 ojbects, 5 tags
            object_poses = {
                "object0": [],
                "object1": [],
                "object2": [],
                "object3": [],
                "object4": [],
                "object5": []
            }
            for res in results:
                world_pose = extrinsics[res['serial']] * res['pose']
                tag_id = res['detection'].tag_id % 5
                tag_transform = self.tag_transforms[f'tag{tag_id}_com']

                com_pose = np.dot(tag_transform, world_pose)
                object_id = res['detection'].tag_id // 5
                object_poses[f'object{object_id}'].append((com_pose, res['final_error']))
            
            for object_id, poses in object_poses.items():
                # breakpoint()
                #average the poses based
                if poses:
                    error = np.array([e for _, e in poses])
                    all_poses = [p for p, _ in poses]
                    weight = np.exp(error) / np.sum(np.exp(error))
                    avg_pose = np.average(all_poses, weights=weight, axis=0)

                    self.object_poses[object_id] = avg_pose
            print("Object poses:", self.object_poses)

            


    def handle_image_cam2(self, msg):
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
                self.latest_images[camera_serials["cam2"]] = cv_image
            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")
        else:
            with self.lock:
                self.latest_msg = msg




    # def handle_turtle_pose(self, msg):
    #     t = TransformStamped()

    #     # Read message content and assign it to
    #     # corresponding tf variables
    #     t.header.stamp = self.get_clock().now().to_msg()
    #     t.header.frame_id = 'world'
    #     t.child_frame_id = self.turtlename

    #     # Turtle only exists in 2D, thus we get x and y translation
    #     # coordinates from the message and set the z coordinate to 0
    #     t.transform.translation.x = msg.x
    #     t.transform.translation.y = msg.y
    #     t.transform.translation.z = 0.0

    #     # For the same reason, turtle can only rotate around one axis
    #     # and this why we set rotation in x and y to 0 and obtain
    #     # rotation in z axis from the message
    #     q = quaternion_from_euler(0, 0, msg.theta)
    #     t.transform.rotation.x = q[0]
    #     t.transform.rotation.y = q[1]
    #     t.transform.rotation.z = q[2]
    #     t.transform.rotation.w = q[3]

    #     # Send the transformation
    #     self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = ObjectPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()