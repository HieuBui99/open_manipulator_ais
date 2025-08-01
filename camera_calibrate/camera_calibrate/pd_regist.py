import pyrealsense2 as rs
import numpy as np
import cv2   # only if you want to display the images
import open3d as o3d


#get the extrinsics of two cameras
extrinsics1 = np.load('/root/ros2_ws/src/open_manipulator/camera_calibrate/camera_calibrate/extrinsics_cam_scene1.npz')
extrinsics2 = np.load('/root/ros2_ws/src/open_manipulator/camera_calibrate/camera_calibrate/extrinsics_cam_scene2.npz')
T_cam1 = np.eye(4)
T_cam1[:3, :3] = extrinsics1['R']
T_cam1[:3, 3] = extrinsics1['T'].squeeze()/1000
T_cam2 = np.eye(4)
T_cam2[:3, :3] = extrinsics2['R']
T_cam2[:3, 3] = extrinsics2['T'].squeeze()/1000

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
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # add depth stream too if you need it
    # cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    # pipelines.append(pipe)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 1 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

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
        points_list = []                # pointclouds for each camera
        for s, [pipe,align] in zip(serials, pipelines):
            frames   = pipe.wait_for_frames()           # blocks on that cam
            aligned_frames = align.process(frames)
            color    = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            # img      = np.asanyarray(color.get_data())
            # images.append((img, s))
            pc  = rs.pointcloud()
            points = pc.calculate(aligned_depth_frame)           # geometry:contentReference[oaicite:1]{index=1}
            vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)            # pc.map_to(color)  # map the pointcloud to the color image
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            points_list.append(pcd)

        # 3.  Bring both point clouds into the same (world) frame
        pcd_world1 = points_list[0].transform(T_cam1.copy())        # transform() operates in-place
        pcd_world2 = points_list[1].transform(T_cam2.copy())

        # 4.  (Optional) Rough→fine alignment if the extrinsics aren’t perfect
        #     Use ICP on down-sampled clouds so it’s fast.
        voxel_size = 0.003                                     # pick ~1 % of scene extent
        pcd1_down = pcd_world1.voxel_down_sample(voxel_size)
        pcd2_down = pcd_world2.voxel_down_sample(voxel_size)
        pcd1_down.estimate_normals()
        pcd2_down.estimate_normals()

        result_icp = o3d.pipelines.registration.registration_icp(
            pcd2_down, pcd1_down,
            max_correspondence_distance= voxel_size*2,
            init     = np.eye(4),                              # start from extrinsic guess
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )

        # Apply the refined transform to the high-res second cloud
        pcd_world2.transform(result_icp.transformation)

        # 5.  Merge & clean up
        merged = pcd_world1 + pcd_world2
        merged = merged.voxel_down_sample(voxel_size/2)       # remove duplicates / noise
        merged.estimate_normals()

        # 6.  Visualise / save
        o3d.visualization.draw_geometries([merged])
except Exception as e:
    print("An error occurred:", e)