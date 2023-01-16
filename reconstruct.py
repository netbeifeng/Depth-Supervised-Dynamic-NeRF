import cv2
import numpy as np
import os
import open3d as o3d
# Read RGB and Depth images to reconstruct 3D point cloud

# RGB_PATH = "./logs/new_w_depth_MSE_DGS_0.001_0.01/renderonly_newb_test_399999/rgb_gt.npy"
# RGB_PATH = "./logs/new_w_depth_MSE_DGS_0.001_0.01/renderonly_newb_test_399999/rgb.npy"
# RGB_PATH = "./logs/new_w_depth_MSE__noDGS_0.001/renderonly_newb_test_399999/rgb.npy"
# RGB_PATH = "./logs/new_w_depth_GNLL_DGS_0.001_0.01/renderonly_newb_test_399999/rgb.npy"
# RGB_PATH = "./logs/new_wo_depth/renderonly_newb_test_399999/rgb.npy"


# RGB_PATH = "./logs/toss_w_depth_MSE_DGS_0.001_0.01/renderonly_newb_test_399999/rgb_gt.npy"
RGB_PATH = "./logs/toss_w_depth_MSE_DGS_0.001_0.01/renderonly_newb_test_399999/rgb.npy"
# RGB_PATH = "./logs/toss_w_depth_MSE__noDGS_0.001/renderonly_newb_test_399999/rgb.npy"
# RGB_PATH = "./logs/toss_w_depth_GNLL_DGS_0.001_0.01/renderonly_newb_test_399999/rgb.npy"
# RGB_PATH = "./logs/toss_wo_depth/renderonly_newb_test_399999/rgb.npy"

# DEPTH_PATH = "./logs/new_w_depth_MSE_DGS_0.001_0.01/renderonly_newb_test_399999/depth_gt.npy"
DEPTH_PATH = "./logs/new_w_depth_MSE_DGS_0.001_0.01/renderonly_newb_test_399999/depth.npy"
# DEPTH_PATH = "./logs/new_w_depth_MSE__noDGS_0.001/renderonly_newb_test_399999/depth.npy"
# DEPTH_PATH = "./logs/new_w_depth_GNLL_DGS_0.001_0.01/renderonly_newb_test_399999/depth.npy"
# DEPTH_PATH = "./logs/new_wo_depth/renderonly_newb_test_399999/depth.npy"

# DEPTH_PATH = "./logs/toss_w_depth_MSE_DGS_0.001_0.01/renderonly_newb_test_399999/depth_gt.npy"
# DEPTH_PATH = "./logs/toss_w_depth_MSE_DGS_0.001_0.01/renderonly_newb_test_399999/depth.npy"
# DEPTH_PATH = "./logs/toss_w_depth_MSE__noDGS_0.001/renderonly_newb_test_399999/depth.npy"
# DEPTH_PATH = "./logs/toss_w_depth_GNLL_DGS_0.001_0.01/renderonly_newb_test_399999/depth.npy"
# DEPTH_PATH = "./logs/toss_wo_depth/renderonly_newb_test_399999/depth.npy"

# POSE_PATH = "./logs/toss_wo_depth/renderall_test_399999/poses.npy"
POSE_PATH = "./data/dino/cams/extrinsics.npy"
INTRINSICS_PATH = "./data/dino/cams/intrinsics.npy"

intrinsics = np.load(INTRINSICS_PATH)
print (intrinsics.shape) # [4, 4]
# Read poses
poses = np.load(POSE_PATH)
# Get 17th pose
# pose = poses[17]
pose = poses[26]

# Get the intrinsic matrix
# fx = 544.2582548211519 # focal length x
# fy = 546.0878823951958 # focal length y
# cx = 326.8604521819424 # optical center x
# cy = 236.1210149172594 # optical center y
fx = intrinsics[0,0]
fy = intrinsics[1,1]
cx = intrinsics[0,2] 
cy = intrinsics[1,2]

intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Read RGB and Depth images
rgb = np.load(RGB_PATH) # [1, 480, 640, 3]
depth = np.load(DEPTH_PATH) # [1, 480, 640, 1]

# Remove the batch dimension
rgb = rgb[0]
depth = depth[0]

# Check mean depth and min,max depth values
print("Mean depth: ", np.mean(depth))
print("Min depth: ", np.min(depth))
print("Max depth: ", np.max(depth))
print("Var depth: ", np.var(depth))

# Get indices of all pixels
height, width, _ = depth.shape
u, v = np.meshgrid(np.arange(width), np.arange(height))
u = u.flatten()
v = v.flatten()
 
# Get 3D points in camera coordinate
z = depth.flatten()
x = (u - cx) * z / fx
y = (v - cy) * z / fy
points = np.vstack((x, y, z)).T
 

# # Open3D: Visualize 3D point cloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

# pcdMap = o3d.geometry.PointCloud()
# pcdMap.points = o3d.utility.Vector3dVector(points + np.array([2,0,0]))

# # Assign color to each point
# pcd.colors = o3d.utility.Vector3dVector(rgb.reshape(-1, 3))
# o3d.visualization.draw_geometries([pcd, pcdMap ])
 