import os
import csv
import glob
import numpy as np
import pandas as pd
import math

# Configuration
DATA_DIR = "/home/orangepi/AV_Project/av_project/data"
IMG_DIR = os.path.join(DATA_DIR, "images")
ODOM_PATH = os.path.join(DATA_DIR, "odom", "odom.csv")
PROJECTION_CSV = os.path.join(DATA_DIR, "projection.csv")
N = 50  # Number of future timesteps

# Load odometry data
odom_df = pd.read_csv(ODOM_PATH)
odom_df = odom_df.sort_values("timestamp").reset_index(drop=True)

# Get images
image_files = sorted(glob.glob(os.path.join(IMG_DIR, "image_*.png")))
image_data = []
for img_path in image_files:
    try:
        basename = os.path.basename(img_path)
        ts_str = basename.split("image_")[1].split(".png")[0]
        timestamp = float(ts_str)
        image_data.append((timestamp, img_path))
    except Exception:
        continue

# Helper to extract yaw from quaternion
def yaw_from_quaternion(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

# Generate projections
rows = []
odom_timestamps = odom_df["timestamp"].values

for img_ts, img_path in image_data:
    idx = np.searchsorted(odom_timestamps, img_ts)
    if idx + N > len(odom_df):
        continue

    window = odom_df.iloc[idx:idx + N]
    x_window = window["x"].tolist()
    y_window = window["y"].tolist()
    z_window = window["z"].tolist()

    theta_window = []
    for _, row_odom in window.iterrows():
        qx, qy, qz, qw = row_odom["qx"], row_odom["qy"], row_odom["qz"], row_odom["qw"]
        theta = yaw_from_quaternion(qx, qy, qz, qw)
        theta_window.append(theta)

    rows.append({
        "timestamp": img_ts,
        "image_path": os.path.relpath(img_path, DATA_DIR),
        "x_window": str(x_window),
        "y_window": str(y_window),
        "z_window": str(z_window),
        "theta_window": str(theta_window),
    })

# Save CSV
with open(PROJECTION_CSV, "w", newline="") as csvfile:
    fieldnames = ["timestamp", "image_path", "x_window", "y_window", "z_window", "theta_window"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"✅ Saved {len(rows)} projections to {PROJECTION_CSV}")
