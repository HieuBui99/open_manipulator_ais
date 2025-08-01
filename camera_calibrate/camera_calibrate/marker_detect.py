import pyrealsense2 as rs
import numpy as np
import cv2   # only if you want to display the images

from argparse import ArgumentParser
import apriltag
import yaml
import os
from ament_index_python.packages import get_package_share_directory



def get_camera_tf():
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
    for serial_number in serial_nos.keys():
        if serial_number not in config:
            raise ValueError(f"Serial number {serial_number} not found in cameras.yaml")

        extrinsics = config[serial_number]['extrinsics']
        R_matrix = np.array(extrinsics['rotation'])
        T_vector = np.array(extrinsics['translation']).squeeze()/1000  # Convert mm to m
        H = np.eye(4)
        H[:3, :3] = R_matrix
        H[:3, 3] = T_vector
        cam_tf[serial_nos[serial_number]] = H
    return cam_tf

def detect_apriltag_from_images(images,
                   serials,
                   camera_matrices,
                   tag_size=0.02,
                  ):

    '''
    Detect AprilTags from static images.

    Args:   input_images [list(str)]: List of images to run detection algorithm on
            output_images [bool]: Boolean flag to save/not images annotated with detections
            display_images [bool]: Boolean flag to display/not images annotated with detections
            detection_window_name [str]: Title of displayed (output) tag detection window
    '''

    parser = ArgumentParser(description='Detect AprilTags from static images.')
    apriltag.add_arguments(parser)
    options = parser.parse_args()
    options.families='tag25h9'

    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())
    for s, img in zip(serials, images):
        # print('Reading {}...\n'.format(os.path.split(image)[1]))

        result, _ = apriltag.detect_tags_filterred(img,
                                               detector,
                                               camera_params=camera_matrices[s],
                                               tag_size=tag_size,
                                               vizualization=3,
                                               verbose=0,
                                               annotation=True
                                              )
        for r in result:
            r['serial'] = s

    return result


def apriltag_image(images,
                   serials,
                   camera_matrices,
                   tag_size=0.02,
                   display_images=True,
                   detection_window_name='AprilTag',
                  ):

    '''
    Detect AprilTags from static images.

    Args:   input_images [list(str)]: List of images to run detection algorithm on
            output_images [bool]: Boolean flag to save/not images annotated with detections
            display_images [bool]: Boolean flag to display/not images annotated with detections
            detection_window_name [str]: Title of displayed (output) tag detection window
    '''

    parser = ArgumentParser(description='Detect AprilTags from static images.')
    apriltag.add_arguments(parser)
    options = parser.parse_args()
    options.families='tag25h9'
    '''
    Set up a reasonable search path for the apriltag DLL.
    Either install the DLL in the appropriate system-wide
    location, or specify your own search paths as needed.
    '''

    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())
    overlays = []
    results = []
    for s, img in zip(serials, images):


        # print('Reading {}...\n'.format(os.path.split(image)[1]))

        result, overlay = apriltag.detect_tags_filterred(img,
                                               detector,
                                               camera_params=camera_matrices[s],
                                               tag_size=0.035,
                                               vizualization=3,
                                               verbose=3,
                                               annotation=True
                                              )
        for r in result:
            r['serial'] = s
        results.extend(result)
        overlays.append(overlay)
    
    if display_images:
        cv2.imshow(detection_window_name, np.concatenate(overlays, axis=1))
        cv2.waitKey(1)
        
        # while cv2.waitKey(5) < 0:   # Press any key to load subsequent image
            # pass
    return results


if __name__ == "__main__":
    #get the intrinsics of two cameras
    intrinsics1 = np.load('/root/ros2_ws/src/open_manipulator/camera_calibrate/camera_calibrate/intrinsics_cam_scene1.npz')
    intrinsics2 = np.load('/root/ros2_ws/src/open_manipulator/camera_calibrate/camera_calibrate/intrinsics_cam_scene2.npz')
    M_cam1 = intrinsics1['camera_matrix']
    M_cam2 = intrinsics2['camera_matrix']

    #get the extrinsics of two cameras
    extrinsics1 = np.load('/root/ros2_ws/src/open_manipulator/camera_calibrate/camera_calibrate/extrinsics_cam_scene1.npz')
    extrinsics2 = np.load('/root/ros2_ws/src/open_manipulator/camera_calibrate/camera_calibrate/extrinsics_cam_scene2.npz')
    T_cam1 = np.eye(4)
    T_cam1[:3, :3] = extrinsics1['R']
    T_cam1[:3, 3] = extrinsics1['T'].squeeze()/1000
    T_cam2 = np.eye(4)
    T_cam2[:3, :3] = extrinsics2['R']
    T_cam2[:3, 3] = extrinsics2['T'].squeeze()/1000


    toParam = lambda x: tuple([x[0,0], x[1,1], x[0,2], x[1,2]])

    camera_matrices ={
        "941322072865": toParam(M_cam1),
        "826212070364": toParam(M_cam2),
    }

    extrinsics_matrices = {
        "941322072865": T_cam1,
        "826212070364": T_cam2,
    }
    ########################################################################
    # 1.  Enumerate connected devices and grab the two you care about
    ########################################################################
    ctx       = rs.context()
    devices   = ctx.query_devices()
    if len(devices) < 2:
        raise RuntimeError("Need at least two cameras connected")

    serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices[:2]]
    print("Using cameras:", serials)

    ########################################################################
    # 2.  Build one pipeline+config per camera
    ########################################################################
    pipelines = []
    for s in serials:
        pipe = rs.pipeline()
        cfg  = rs.config()
        cfg.enable_device(s)             # bind to that specific camera
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # add depth stream too if you need it
        # cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = pipe.start(cfg)
        # pipelines.append(pipe)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        # depth_sensor = profile.get_device().first_depth_sensor()
        # depth_scale = depth_sensor.get_depth_scale()
        # print("Depth Scale is: " , depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        # clipping_distance_in_meters = 1 #1 meter
        # clipping_distance = clipping_distance_in_meters / depth_scale

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)
        pipelines.append([pipe, align])

    ########################################################################
    # 3.  Main loop – fetch a frameset from EACH pipeline every iteration
    ########################################################################
    try:
        while True:
            images = []                      # (color_image, serial) tuples
            for s, [pipe,align] in zip(serials, pipelines):
                frames   = pipe.wait_for_frames()           # blocks on that cam
                aligned_frames = align.process(frames)
                color = aligned_frames.get_color_frame()
                images.append(np.asanyarray(color.get_data()))
            results = apriltag_image(images,
                        serials,
                        camera_matrices,
                        tag_size=0.026,
                        display_images=True,
                        detection_window_name='AprilTag')
            
            



    except Exception as e:
        print("An error occurred:", e)