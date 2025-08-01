

from geometry_msgs.msg import TransformStamped

import numpy as np

from rcl_interfaces.msg import ParameterDescriptor, ParameterType


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from rclpy.qos import HistoryPolicy, QoSProfile
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation as R
from ament_index_python.packages import get_package_share_directory
import os
import yaml
from . import apriltag
from functools import partial
from argparse import ArgumentParser
import math
from visualization_msgs.msg import Marker

def weighted_circular_mean(angles, weights):
    W = sum(weights)
    C = sum(w * math.cos(a) for a, w in zip(angles, weights)) / W
    S = sum(w * math.sin(a) for a, w in zip(angles, weights)) / W
    return math.atan2(S, C)

def project_onto_plane(v, n):
    n = n / np.linalg.norm(n)            # 1) unit normal
    projected = v - np.dot(v, n) * n          # 2)+3)
    return projected/np.linalg.norm(projected)


def rotz(theta):
    """Rotation matrix around Z axis by angle theta (radians)."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def get_camera_parameters():
    ''' Get the camera extrinsics from the YAML file and convert to a pose dictionary.
    '''
    camera_file_pth = os.path.join(
        get_package_share_directory('camera_calibrate'),
        'config',
        'cameras.yaml'
    )
    # Load the YAML
    with open(camera_file_pth, 'r') as f:
        config = yaml.safe_load(f)

    serial_nos = {
        # "_218622274409"
        "_826212070364": "camera_scene1_color_frame",
        "_941322072865": "camera_scene2_color_frame"}
    cam_tf = {}
    cam_intrinsics = {}
    for serial_number in serial_nos.keys():
        if serial_number not in config:
            raise ValueError(f"Serial number {serial_number} not found in cameras.yaml")

        extrinsics = config[serial_number]['extrinsics']
        R_matrix = np.array(extrinsics['rotation'])
        T_vector = np.array(extrinsics['translation']).squeeze()/1000  # Convert mm to m
        H = np.eye(4)
        H[:3, :3] = R_matrix
        H[:3, 3] = T_vector
        cam_tf[serial_number[1:]] = H
        intrinsics = np.array(config[serial_number]['intrinsics']['camera_matrix'])
        # x[0,0], x[1,1], x[0,2], x[1,2]
        cam_intrinsics[serial_number[1:]] = [
            intrinsics[0,0],
            intrinsics[1,1],
            intrinsics[0,2],
            intrinsics[1,2]
        ]
    return cam_tf, cam_intrinsics

def generate_tag_transformations():

    T_tag0_com = np.eye(4)
    T_tag0_com[2,-1] = -0.02#?????????????????

    T_tag1_com = np.eye(4)
    T_tag1_com[:3, :3] = R.from_euler('x', np.pi/2).as_matrix()
    T_tag1_com[2,-1] = -0.015

    T_tag2_com = np.eye(4)
    T_tag2_com[:3, :3] = R.from_euler('y', np.pi/2).as_matrix().dot(T_tag1_com[:3, :3])
    T_tag2_com[2,-1] = -0.06

    T_tag3_com = np.eye(4)
    T_tag3_com[:3, :3] = R.from_euler('y', np.pi).as_matrix().dot(T_tag1_com[:3, :3])
    T_tag3_com[2,-1] = -0.015

    T_tag4_com = np.eye(4)
    T_tag4_com[:3, :3] = R.from_euler('y', -np.pi/2).as_matrix().dot(T_tag1_com[:3, :3])
    T_tag4_com[2,-1] = -0.06

    tag_transforms = dict(tag0_com=T_tag0_com, 
                          tag1_com=T_tag1_com, 
                          tag2_com=T_tag2_com, 
                          tag3_com=T_tag3_com, 
                          tag4_com=T_tag4_com)

    return tag_transforms

class ObjectPoseTF(Node):

    def __init__(self):
        super().__init__('object_pose_tf_publisher')


        self.declare_parameter('marker_size_mm', 23.0, 
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Size of a square in millimeters'
            )
        )
        self.tag_transforms = generate_tag_transformations()
        self.extrinsics, self.intrinsics = get_camera_parameters()
        self.marker_size_mm = self.get_parameter('marker_size_mm').get_parameter_value().double_value
        parser = ArgumentParser(description='Detect AprilTags from static images.')
        apriltag.add_arguments(parser)
        options = parser.parse_args()
        options.families='tag25h9'
        options.refine_pose = True
        self.detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())
        self.bridge = CvBridge()
        self.msg_type = CompressedImage

        self.subsciption_cam1 = self.create_subscription(
            CompressedImage,
            '/camera/camera_scene1/color/image_raw/compressed',
            partial(self.camera_callback, camera_serial="826212070364"),
            QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST)
        )
        self.subsciption_cam2 = self.create_subscription(
            CompressedImage,
            '/camera/camera_scene2/color/image_raw/compressed',
            partial(self.camera_callback, camera_serial="941322072865"),
            QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST)
        )
        self.latest_images = {key:None for key in self.extrinsics.keys()}
        self.object_poses = {
            "object0": None,
            "object1": None,
            "object2": None,
            "object3": None,
            "object4": None,
            "object5": None
        }
        #################################################3
        # '''test for object0'''
        # self.kf = KalmanFilter(dim_x=4, dim_z=4)
        # self.kf.F = np.eye(4)           # random‑walk model
        # self.kf.H = np.eye(4)
        # self.kf.R = np.diag([0.02, 0.02, 0.02, 0.0004])
        # self.kf.Q = np.diag([1e-4, 1e-4, 1e-4, 1e-5])
        # self.kf.x = np.zeros(4)
        #################################################
        self.tf_broadcaster = TransformBroadcaster(self)
        self.markers = {f"object{i}": Marker() for i in range(6)}
        self.marker_publisher = self.create_publisher(Marker, 'visualization_marker', 12)
        self.timer = self.create_timer(0.05, self.broadcasting_object_poses)
        self.marker_timer = self.create_timer(0.05, self.visualizer_update)
        self.publisher = self.create_publisher(
            Image,
            'image_with_overlays',
            QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST)
        )
        self.marker_colors = {
            "object0": (0.2, 0.8, 0.3, 1.0),
            "object1": (0.8, 0.2, 0.3, 1.0),
            "object2": (0.3, 0.2, 0.8, 1.0),
            "object3": (0.8, 0.8, 0.2, 1.0),
            "object4": (0.2, 0.3, 0.8, 1.0),
            "object5": (0.8, 0.3, 0.2, 1.0)
        }
        self.t = None

    def update(self, z):
        self.kf.predict()
        self.kf.update(z)

    def camera_callback(self, msg, camera_serial=None):
        try:
            if self.msg_type == Image:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            elif self.msg_type == CompressedImage:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                raise ValueError(f"Unsupported message type: {self.msg_type}")
            self.latest_images[camera_serial] = cv_image
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")


    def update_object_poses(self):
        overlays = []
        object_poses = {
                "object0": [],
                "object1": [],
                "object2": [],
                "object3": [],
                "object4": [],
                "object5": []
            }
        for s, img in self.latest_images.items():


            # print('Reading {}...\n'.format(os.path.split(image)[1]))

            result, overlay = apriltag.detect_tags_filterred(img,
                                                self.detector,
                                                camera_params=self.intrinsics[s],
                                                tag_size=self.marker_size_mm / 1000,  # Convert mm to m
                                                vizualization=3,
                                                verbose=0,
                                                annotation=True
                                                )
            if not result:
                self.get_logger().warn(f"No tags detected in camera {s}")
                overlays.append(overlay)
                continue
            rotM = np.array([[1., 0.,  0., 0],
                            [0., -1.,  0., 0],
                            [0.,  0., -1., 0],
                            [0., 0.,  0., 1.]]) # to convert from
            poses_in_cam = np.stack([r['pose'].dot(rotM) for r in result])
            # print(poses_in_cam)
            tag_pose_in_base = np.einsum('ij, bjk -> bik', self.extrinsics[s], poses_in_cam) # [B, 4, 4]
            # poses.append(pose)
            ids = [r['detection'].tag_id for r in result] #detected tag ids
            T_tag_to_com = np.stack([self.tag_transforms['tag'+str(id%5)+'_com'] for id in ids]) # [B, 4, 4]
            pose_in_base = np.einsum('bij,bjk->bik', tag_pose_in_base, T_tag_to_com) # [B, 4, 4], com to base frame
            # print(pose_in_base)
            # breakpoint()
            for i, id in enumerate(ids):
                # print(ids)
                # print("id:{}, z axis:{}".format(id, tag_pose_in_base[i][:3,2]))
                # output[str(id)].append(dict(pose=pose_in_base[i], error=result[i]['final_error']))
                object_id = id // 5
                # breakpoint()
                # print(np.linalg.norm(pose_in_base[i][:3,-1]-self.object_poses[f'object{object_id}'][:3,-1]))
                # if (tag_pose_in_base[i][2,2] * self.extrinsics[s][2,2]) > 0:
                #=================================================================
                # if  id%5==0:
                #     # print(tag_pose_in_base[i][:3, 2])
                #     if tag_pose_in_base[i][2, 2] < 0:
                #         continue
                #     else:
                #         object_poses[f'object{object_id}'].append((pose_in_base[i], result[i]['final_error']))
                # else:
                #     tag_pose_proj = project_onto_plane(tag_pose_in_base[i][:3,2], [0,0,1])
                #     extrinsics_proj = project_onto_plane(self.extrinsics[s][:3,2], [0,0,1])
                #     if abs(tag_pose_in_base[i][2,2]) > 0.02:
                #         self.get_logger().warn(f"Tag {id} in camera {s} has a large z value: {tag_pose_in_base[i][2,2]}")
                #         continue
                #     #The codes below are wrong!!!!!!!!!!!!!!!!!
                #     elif np.arccos(abs(tag_pose_proj.dot(extrinsics_proj))) > np.pi/6\
                #     or tag_pose_proj.dot(extrinsics_proj) > 0:
                #         print(tag_pose_in_base[i][:3,2])
                #         # or tag_pose_proj.dot(extrinsics_proj) > 0:
                #         # or tag_pose_proj.dot(extrinsics_proj) > 0:

                #         continue
                #     else:
                #         # print(tag_pose_in_base[i][:3,2])
                #         object_poses[f'object{object_id}'].append((pose_in_base[i], result[i]['final_error']))
                #=================================================================
                # breakpoint()
                if  id%5==0:
                    if tag_pose_in_base[i][2, 2] < 0.9:
                        continue
                else:
                    if abs(tag_pose_in_base[i][2,2]) > 0.02:
                        self.get_logger().warn(f"Tag {id} in camera {s} has a large z value: {tag_pose_in_base[i][2,2]}")
                        continue
                tag_to_cam = self.extrinsics[s][:3, 3] - tag_pose_in_base[i][:3, 3]
                tag_to_cam = tag_to_cam / np.linalg.norm(tag_to_cam)
                if np.arccos(tag_to_cam.dot(tag_pose_in_base[i][:3, 2])) < np.pi/3.5:
                    self.get_logger().info("(ID, error): ({}, {})".format(id, result[i]['final_error']))
                    object_poses[f'object{object_id}'].append((pose_in_base[i], result[i]['final_error']))
                #=========================================================
                # print(project_onto_plane(tag_pose_in_base[i][:3,2], [0,0,1]),
                #     project_onto_plane(self.extrinsics[s][:3,2], [0,0,1]))
                # z_tag = tag_pose_in_base[i][:3,2]
                # z_extrinsics = self.extrinsics[s][:3,2]
                # # breakpoint()
                # # print(np.arccos(abs(tag_pose_proj.dot(extrinsics_proj))))
                # if np.arccos(abs(z_tag.dot(z_extrinsics))) > np.pi/4\
                #     or z_tag.dot(z_extrinsics) > 0:

                #     continue
                # else:
                #     object_poses[f'object{object_id}'].append((pose_in_base[i], result[i]['final_error']))
            # object_poses[f'object{object_id}'].append((pose_in_base[i], result[i]['final_error']))
            overlays.append(overlay)
        # average    the object poses based on the error
        for object_id, poses in object_poses.items():
            # breakpoint()
            #average the poses based
            if poses:
                error = np.array([-e for _, e in poses])
                weight = np.exp(error) / np.sum(np.exp(error))
                #one hot
                # error = np.array([e for _, e in poses])
                # weight = np.zeros_like(error)
                # weight[np.argmin(error)] = 1.0
                all_poses = [p for p, _ in poses]
                rotation_angles = np.array([R.from_matrix(p[:3, :3]).as_rotvec()[-1] for p in all_poses])
                # rotation_matrices = np.array([rotz(theta) for theta in rotation_angles])
                avg_theta = weighted_circular_mean(rotation_angles, weight)
                avg_pos = np.average([p[:3, 3] for p in all_poses], weights=weight, axis=0)
                # self.update(np.array([avg_pos[0], avg_pos[1], avg_pos[2], avg_theta]))
                # avg_pose = np.eye(4)
                # avg_pose[:3, :3] = rotz(self.kf.x[3]) # R.from_euler('z', avg_theta).as_matrix()    
                # avg_pose[:3, 3] = self.kf.x[:3]
                avg_pose = np.eye(4)
                avg_pose[:3, :3] = rotz(avg_theta)
                # breakpoint()
                avg_pose[:3, 3] = avg_pos
                self.object_poses[object_id] = avg_pose
        self.overlays = np.concatenate(overlays, axis=1)
        # print(self.overlays.shape)

    def visualizer_update(self):
        # --- Publish Marker ---
        # marker = Marker()
        t = self.t
        if t is None:
            return
        marker = self.markers[t.child_frame_id]
        marker.header.stamp = t.header.stamp
        marker.header.frame_id = 'world'  # Tie marker to the TF frame
        marker.ns = 'box'
        marker.id = int(t.child_frame_id[-1])  # Use the last character of the child frame ID as the marker ID
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = t.transform.translation.x
        marker.pose.position.y = t.transform.translation.y
        marker.pose.position.z = t.transform.translation.z
        marker.pose.orientation.x = t.transform.rotation.x
        marker.pose.orientation.y = t.transform.rotation.y
        marker.pose.orientation.z = t.transform.rotation.z  
        marker.pose.orientation.w = t.transform.rotation.w

        # Set box size
        marker.scale.y = 0.030  # Length
        marker.scale.x = 0.120  # Width
        marker.scale.z = 0.032  # Height

        # Set color (RGBA)
        color = self.marker_colors[t.child_frame_id]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        # marker.lifetime.sec = 1  # Optional: refresh every second
        self.marker_publisher.publish(marker)

    def broadcasting_object_poses(self):
        if any(list(map(lambda x: x is None, self.latest_images.values()))):
            self.get_logger().warn("Waiting for images to be received...")
            return
        # breakpoint()
        self.update_object_poses()
        self.publisher.publish(
            self.bridge.cv2_to_imgmsg(self.overlays, encoding="bgr8")
        )

        t = TransformStamped()
        # Read message content and assign it to
        # corresponding tf variables
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        for object_id, pose in self.object_poses.items():
            if pose is not None:
                t.child_frame_id = object_id
                t.transform.translation.x = pose[0, 3]
                t.transform.translation.y = pose[1, 3]
                t.transform.translation.z = pose[2, 3]

                q = R.from_matrix(pose[:3, :3]).as_quat()
                t.transform.rotation.x = q[0]
                t.transform.rotation.y = q[1]
                t.transform.rotation.z = q[2]
                t.transform.rotation.w = q[3]
                self.t = t
                self.visualizer_update()
                # Send the transformation
                self.tf_broadcaster.sendTransform(t)
                self.get_logger().info(f"Published transform for {object_id} at {t.header.stamp.sec}.{t.header.stamp.nanosec}")


def main():
    rclpy.init()
    node = ObjectPoseTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()