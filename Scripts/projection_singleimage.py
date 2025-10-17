import os
import cv2
import numpy as np
import pandas as pd
import ast

# CONFIGURATION
DATA_DIR = "/home/orangepi/AV_Project/av_project/data"
PROJECTION_CSV = os.path.join(DATA_DIR, "projection.csv")
# Using an image where the trajectory is more visible
SELECTED_IMAGE = "images/image_1751541578.0106304.png"
DEBUG_LIMIT = 5 # Set how many points to debug

# CAMERA INTRINSICS
fx = 384.0262451171875
fy = 382.99444580078125
cx = 327.16033935546875
cy = 241.75653076171875

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])

# EXTRINSICS: Transformation from robot's base_link frame to camera's optical frame
# This defines the camera's position and orientation relative to the robot's center.

# ROTATION: base_link → optical_frame
R_base_to_optical = np.array([
    [0, -1, 0],
    [0, 0, -1], # Flipped the Z-axis to follow standard camera conventions (Z forward)
    [1, 0, 0]
])

# TRANSLATION: base_link → optical_frame (as a vector in the base_link frame)
t_in_base = np.array([0.08, 0.0, 0.12])

# Transform the translation vector into the optical frame's coordinate system
# NOTE: The original code had a potential issue here. If t_in_base is the location of the
# camera's origin *in the base_link frame*, then the transformation to the camera frame is
# t_in_optical = -R_base_to_optical @ t_in_base. However, for projection, we need the
# translation component of the extrinsic matrix, which represents the position of the
# world origin in camera coordinates. Let's stick to the original code's approach for now
# as it might be specific to the setup, but this is a common point of confusion.
# For this code, we'll assume the provided R and t correctly form the extrinsic matrix.
t_in_optical = -R_base_to_optical @ t_in_base

# Extrinsic matrix [R|t] from robot base_link to camera optical frame
RT_base_to_cam = np.hstack((R_base_to_optical, t_in_optical.reshape(-1, 1)))

# Full projection matrix from robot base_link to image plane
P = K @ RT_base_to_cam

# --- DATA LOADING ---
df = pd.read_csv(PROJECTION_CSV)
df["x_window"] = df["x_window"].apply(ast.literal_eval)
df["y_window"] = df["y_window"].apply(ast.literal_eval)
df["z_window"] = df["z_window"].apply(ast.literal_eval)
df["theta_window"] = df["theta_window"].apply(ast.literal_eval)

row = df[df["image_path"] == SELECTED_IMAGE]
if row.empty:
    print(f"❌ Image '{SELECTED_IMAGE}' not found.")
    exit()

row = row.iloc[0]
x_window = np.array(row["x_window"])
y_window = np.array(row["y_window"])
z_window = np.array(row["z_window"])
theta_window = np.array(row["theta_window"])

# --- IMAGE LOADING ---
img_path = os.path.join(DATA_DIR, row["image_path"])
image = cv2.imread(img_path)
if image is None:
    print(f"❌ Could not load image: {img_path}")
    exit()

# The pose of the robot that has the camera at the moment the picture was taken.
# All world coordinates will be transformed relative to this pose.
cam_robot_x = x_window[0]
cam_robot_y = y_window[0]
cam_robot_z = z_window[0]
cam_robot_theta = theta_window[0]

# Pre-calculate the inverse rotation of the camera-robot for efficiency
cos_inv_t = np.cos(-cam_robot_theta)
sin_inv_t = np.sin(-cam_robot_theta)
R_world_to_cam_robot_base = np.array([
    [cos_inv_t, -sin_inv_t, 0],
    [sin_inv_t,  cos_inv_t, 0],
    [0,          0,         1]
])
cam_robot_pos = np.array([cam_robot_x, cam_robot_y, cam_robot_z])

# Robot Footprint Dimensions (in robot's local base frame)
L, W = 0.32, 0.24
half_L, half_W = L/2, W/2
robot_corners_local = np.array([
    # Corners in order: Front-Right, Front-Left, Rear-Left, Rear-Right
    [ half_L, -half_W, 0], # Swapped order to be CCW for polylines
    [ half_L,  half_W, 0],
    [-half_L,  half_W, 0],
    [-half_L, -half_W, 0]
])

print("--- Starting Projection and Debugging ---")
np.set_printoptions(precision=4, suppress=True)

# --- MAIN PROJECTION LOOP ---
# Iterate through each pose in the trajectory window
for i in range(len(x_window)):
    # Current robot pose in the WORLD frame
    current_pose_x = x_window[i]
    current_pose_y = y_window[i]
    current_pose_z = z_window[i]
    current_pose_theta = theta_window[i]
    current_pos_world = np.array([current_pose_x, current_pose_y, current_pose_z])

    # --- 1. PROJECT THE CENTER POINT ---

    # Transform the center point from WORLD frame to the CAMERA-ROBOT's BASE frame
    # Step A: Translate
    center_point_rel = current_pos_world - cam_robot_pos
    # Step B: Rotate
    center_point_cam_base = R_world_to_cam_robot_base @ center_point_rel

    # Project to image plane
    center_homog = np.append(center_point_cam_base, 1)
    center_cam_coords = P @ center_homog
    center_proj = center_cam_coords[:2] / center_cam_coords[2] if center_cam_coords[2] > 0 else None

    # --- 2. PROJECT THE ROBOT BASE ---

    # Get the rotation matrix for the current pose to transform its corners into the WORLD frame
    cos_t = np.cos(current_pose_theta)
    sin_t = np.sin(current_pose_theta)
    R_local_to_world = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ])

    # Transform local robot corners to WORLD frame for the current pose
    corners_world = (R_local_to_world @ robot_corners_local.T).T + current_pos_world

    # Transform WORLD frame corners to the CAMERA-ROBOT's BASE frame
    # Step A: Translate
    corners_rel = corners_world - cam_robot_pos
    # Step B: Rotate
    corners_cam_base = (R_world_to_cam_robot_base @ corners_rel.T).T

    # Project corners to image plane
    corners_homog = np.hstack((corners_cam_base, np.ones((corners_cam_base.shape[0], 1))))
    corners_cam_coords = (P @ corners_homog.T).T

    # Normalize by z-coordinate to get pixel values
    corners_proj = corners_cam_coords.copy()
    valid_mask = corners_cam_coords[:, 2] > 0
    corners_proj[valid_mask, :2] /= corners_cam_coords[valid_mask, 2:3]


    # --- 3. DEBUGGING OUTPUT for the first few points ---
    if i < DEBUG_LIMIT:
        print(f"\n--- DEBUGGING POINT {i} ---")
        print(f"  Center Point World Coords (X, Y, Z, Theta): ({current_pose_x:.4f}, {current_pose_y:.4f}, {current_pose_z:.4f}, {current_pose_theta:.4f})")
        if center_proj is not None:
            print(f"  Center Point Projected Coords (u, v): ({center_proj[0]:.2f}, {center_proj[1]:.2f})")
        else:
            print("  Center Point is behind the camera.")
        print("  Corner World Coordinates:")
        print(corners_world)
        print("  Corner Projected Coordinates (u, v, z_cam):")
        print(corners_proj)

    # --- 4. DRAWING ---

    # Draw the footprint polygon
    corner_pixels = []
    if all(valid_mask): # Only draw if all corners are in front of the camera
        for pt in corners_proj:
            corner_pixels.append((int(round(pt[0])), int(round(pt[1]))))

        if len(corner_pixels) == 4:
            # Use a color that fades with distance for better visualization
            # Let's use green for close and blue for far
            depth = np.mean(corners_cam_base[:, 2]) # Use depth in camera frame for color
            # Simple linear interpolation for color from green to blue
            lerp = min(max((depth - 0.5) / 5.0, 0.0), 1.0) # Normalize depth between ~0.5m and 5.5m
            color = (0, int(255 * (1-lerp)), int(255 * lerp)) # BGR
            cv2.polylines(image, [np.array(corner_pixels)], isClosed=True, color=color, thickness=2)

    # Draw the center point
    if center_proj is not None:
        u_int, v_int = int(round(center_proj[0])), int(round(center_proj[1]))
        if 0 <= u_int < image.shape[1] and 0 <= v_int < image.shape[0]:
            cv2.circle(image, (u_int, v_int), 4, (0, 255, 0), -1)

# Show result
cv2.imshow("Projection with Robot Footprint", image)
cv2.waitKey(0)
cv2.destroyAllWindows()