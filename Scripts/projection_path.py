import os
import cv2
import numpy as np
import pandas as pd
import ast

# CONFIGURATION
DATA_DIR = "/home/orangepi/AV_Project/av_project/data"
PROJECTION_CSV = os.path.join(DATA_DIR, "projection.csv")
# Removed SELECTED_IMAGE as we are processing all images
DEBUG_LIMIT = 0 # Set how many points to debug for each image's trajectory

# Output directory for processed path images
PATH_OUTPUT_DIR = os.path.join(DATA_DIR, "path")
os.makedirs(PATH_OUTPUT_DIR, exist_ok=True) # Create the directory if it doesn't exist

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
    [0, 0, -1],
    [1, 0, 0]
])

# TRANSLATION: base_link → optical_frame (as a vector in the base_link frame)
t_in_base = np.array([0.08, 0.0, 0.12])

# Transform the translation vector into the optical frame's coordinate system
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

print("--- Starting Batch Projection and Saving ---")
np.set_printoptions(precision=4, suppress=True)

# Robot Footprint Dimensions (in robot's local base frame)
L, W = 0.32, 0.24
half_L, half_W = L/2, W/2
robot_corners_local = np.array([
    # Corners in order: Front-Right, Front-Left, Rear-Left, Rear-Right (CCW for polylines)
    [ half_L, -half_W, 0],
    [ half_L,  half_W, 0],
    [-half_L,  half_W, 0],
    [-half_L, -half_W, 0]
])

# Iterate through each row (each image) in the DataFrame
for index, row in df.iterrows():
    img_relative_path = row["image_path"]
    img_full_path = os.path.join(DATA_DIR, img_relative_path)
    output_filename = os.path.basename(img_relative_path)
    output_path = os.path.join(PATH_OUTPUT_DIR, output_filename)

    original_image = cv2.imread(img_full_path)
    if original_image is None:
        print(f"❌ Could not load image: {img_full_path}. Skipping.")
        continue

    # Create a new black image for the path projection, with the same dimensions as the original
    path_image = np.zeros_like(original_image)

    x_window = np.array(row["x_window"])
    y_window = np.array(row["y_window"])
    z_window = np.array(row["z_window"])
    theta_window = np.array(row["theta_window"])

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

    # --- MAIN PROJECTION LOOP for the current image's trajectory ---
    # Iterate through each pose in the trajectory window
    for i in range(len(x_window)):
        # Current robot pose in the WORLD frame
        current_pose_x = x_window[i]
        current_pose_y = y_window[i]
        current_pose_z = z_window[i]
        current_pose_theta = theta_window[i]
        current_pos_world = np.array([current_pose_x, current_pose_y, current_pose_z])

        # --- 1. PROJECT THE CENTER POINT (for debugging, not drawing) ---
        center_point_rel = current_pos_world - cam_robot_pos
        center_point_cam_base = R_world_to_cam_robot_base @ center_point_rel
        center_homog = np.append(center_point_cam_base, 1)
        center_cam_coords = P @ center_homog
        center_proj = center_cam_coords[:2] / center_cam_coords[2] if center_cam_coords[2] > 0 else None

        # --- 2. PROJECT THE ROBOT BASE (Footprint Corners) ---
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
        corners_rel = corners_world - cam_robot_pos
        corners_cam_base = (R_world_to_cam_robot_base @ corners_rel.T).T

        # Project corners to image plane
        corners_homog = np.hstack((corners_cam_base, np.ones((corners_cam_base.shape[0], 1))))
        corners_cam_coords = (P @ corners_homog.T).T

        # Normalize by z-coordinate to get pixel values
        corners_proj = corners_cam_coords.copy()
        valid_mask = corners_cam_coords[:, 2] > 0 # Check if corners are in front of the camera
        corners_proj[valid_mask, :2] /= corners_cam_coords[valid_mask, 2:3]

        # --- 3. DEBUGGING OUTPUT for the first few points of this trajectory ---
        if i < DEBUG_LIMIT:
            print(f"\n--- DEBUGGING IMAGE: {output_filename} - POINT {i} ---")
            print(f"   Center Point World Coords (X, Y, Z, Theta): ({current_pose_x:.4f}, {current_pose_y:.4f}, {current_pose_z:.4f}, {current_pose_theta:.4f})")
            if center_proj is not None:
                print(f"   Center Point Projected Coords (u, v): ({center_proj[0]:.2f}, {center_proj[1]:.2f})")
            else:
                print("   Center Point is behind the camera.")
            print("   Corner World Coordinates:")
            print(corners_world)
            print("   Corner Projected Coordinates (u, v, z_cam):")
            print(corners_proj)

        # --- 4. DRAWING ON THE NEW BLACK IMAGE ---
        corner_pixels = []
        if all(valid_mask): # Only draw if all corners are in front of the camera
            # Convert projected float coordinates to integer pixel coordinates
            for pt in corners_proj:
                # Ensure points are within image boundaries before adding
                u_int, v_int = int(round(pt[0])), int(round(pt[1]))
                if 0 <= u_int < original_image.shape[1] and 0 <= v_int < original_image.shape[0]:
                    corner_pixels.append((u_int, v_int))
                else:
                    # If any corner is out of bounds, do not draw this polygon
                    corner_pixels = []
                    break

            # Ensure we have 4 valid corners to form a polygon
            if len(corner_pixels) == 4:
                # Draw the footprint polygon filled in white
                cv2.fillPoly(path_image, [np.array(corner_pixels)], color=(255, 255, 255)) # White color in BGR

    # --- SAVE THE PROCESSED IMAGE ---
    cv2.imwrite(output_path, path_image)
    print(f"✅ Processed and saved: {output_filename}")

print("\n--- All images processed and saved to the 'path' folder. ---")