# Indoor Traversability Prediction with Self-Supervised Learning

This project adapts the core ideas of **WayFAST** to indoor mobile robotics: estimating traversability and surface difficulty by comparing **commanded** vs **estimated** motion. It combines **state estimation**, **geometric projection**, and **learning-ready dataset construction** in **ROS 2** on a **DJI RoboMaster EP Core** with **Intel RealSense D455**.

The system integrates perception, estimation, and learning modules within a unified traversability prediction pipeline, validated through full data acquisition, state estimation, and geometric projection workflows.

---

## Highlights

- ROS 2 teleop + logging on RoboMaster EP Core (D455 RGB-D).
- EKF-based state estimation fusing odometry and IMU; per-frame traversability coefficients.
- Camera–base extrinsics + multi-timestep pose projection onto image plane for pixel-level labels.
- Learning-ready dataset (RGB-D, path masks, µ/ν coefficient maps, kinematics) and a ResNet-Depth-UNet training scaffold.

---

## System Overview

**Hardware**
- DJI RoboMaster EP Core
- Intel RealSense D455 (RGB-D)
- Jetson Nano (on-board), host workstation for development

**Software**
- ROS 2 (Humble recommended), `rclpy`, `robot_localization`
- OpenCV, NumPy, Matplotlib
- PyTorch (for the training scaffold)

**Pipeline**
1. **Collect** RGB-D, `/cmd_vel`, `/odom`, IMU via ROS 2 bags.
2. **Fuse** odometry + IMU with EKF for velocity/orientation estimation.
3. **Project** future odometry windows into the current image via camera extrinsics to create **path masks**.
4. **Assemble** synchronized training rows with RGB-D, path masks, µ/ν maps, and kinematics.
5. **Train (scaffold)** ResNet-Depth-UNet for pixel-wise traversability (setup complete; training not run due to coefficient bottleneck).

---

## Mathematical Framework

The traversability estimation framework builds upon the physical relationship between commanded and measured motion to infer surface interaction.  
Let \( v_c \) denote the commanded velocity and \( v_m \) the estimated velocity (from odometry or EKF).  
The **traversability coefficient** is defined as:

\[
\mu = \frac{v_m}{v_c}
\]

Under ideal traction, \( \mu \approx 1 \); lower values indicate slippage or reduced mobility.  
This coefficient serves as a **self-supervisory signal**, enabling learning without manual labels.

---

### 1. State Estimation (EKF)

An Extended Kalman Filter fuses odometry and IMU data to estimate the robot’s state:

\[
\mathbf{x} = [x, y, \theta, v, \omega, b_{a_x}, b_{a_y}, b_\omega]^T
\]

The EKF prediction and update follow:

\[
\mathbf{x}_{k|k-1} = f(\mathbf{x}_{k-1}, \mathbf{u}_{k-1}) , \quad
\mathbf{P}_{k|k-1} = \mathbf{F}\mathbf{P}_{k-1}\mathbf{F}^T + \mathbf{Q}
\]

\[
\mathbf{K} = \mathbf{P}\mathbf{H}^T(\mathbf{H}\mathbf{P}\mathbf{H}^T+\mathbf{R})^{-1}, \quad
\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K}(\mathbf{z}_k - h(\mathbf{x}_{k|k-1}))
\]

This provides smoothed velocity estimates \( v \) and \( \omega \), robust to IMU noise or wheel slip.

---

### 2. Geometric Projection

Future odometry poses are reprojected into the image plane using known camera intrinsics and extrinsics:

\[
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
= \frac{1}{Z_{cam}} \, \mathbf{K} [\mathbf{R}|\mathbf{t}]
\begin{bmatrix} X_{base} \\ Y_{base} \\ Z_{base} \\ 1 \end{bmatrix}
\]

This maps multi-timestep trajectories to their corresponding RGB-D pixels, creating **path masks** for self-supervision.

---

### 3. Self-Supervised Learning

A ResNet–Depth–UNet model predicts per-pixel traversability \( \hat{\mu}(u,v) \) from RGB-D input, optimized by:

\[
\mathcal{L} = \| \hat{\mu} - \mu_{EKF} \|_2^2
\]

The robot thus learns traversability directly from its own motion feedback—bridging **state estimation**, **geometry**, and **learning** for adaptive navigation.

---

## Repository Structure

```
JDSRobo/
│
├─ Sample Data/
│  ├─ camera_info/
│  ├─ cmd_vel/
│  ├─ odom/
│  ├─ final_training_data.csv
│  ├─ projection.csv
│  └─ Path images sample (future path without traversability coeff. scaling).mp4
│
├─ Scripts/
│  ├─ teleop.py                      # RoboMaster SDK teleop + IMU publish + /cmd_vel
│  ├─ record_rosbag.py               # ROS-native teleop; print odom; bag recording (template)
│  ├─ analyze_coefficients.py        # Teleop + odom sampling; histograms of v and µ = v_odom / v_cmd
│  ├─ teleop_EKF.py                  # 8-state EKF (x,y,θ,v,ω,biases); slip heuristic; plots + TF + /ekf_odom
│  ├─ extract_rosbag_to_data.py      # rosbag2 → images/, odom.csv, cmd_vel.csv, camera_intrinsics.yaml
│  ├─ generate_projection_csv.py     # For each image: future N-pose window (x,y,z,θ) → projection.csv
│  ├─ projection_path.py             # Batch projection: future robot footprints → path/*.png (black canvas)
│  └─ projection_singleimage.py     # Single-frame projection with debug prints and colored footprints
│
└─ Training/
   ├─ dataset.py
   ├─ train.py
   ├─ infer.py
   ├─ params.py
   ├─ resnet_depth_unet.py
   └─ utils.py
```

---

## Getting Started

### 1) Environment

- Ubuntu 22.04 + ROS 2 Humble
- Python 3.10+
- Recommended Python deps:
  ```bash
  pip install opencv-python numpy matplotlib pandas pyyaml scipy torch torchvision
  ```
- RealSense SDK (for device-level tools) if capturing new data.

### 2) Launch + Teleop (collect data)

- RoboMaster ROS 2 driver (example is spawned by the scripts; otherwise launch your own).
- Manual teleop options:
  - **RoboMaster SDK teleop:**
    ```bash
    ros2 run your_pkg teleop.py
    ```
  - **ROS-native teleop (console):**
    ```bash
    ros2 run your_pkg record_rosbag.py
    ```
    (Use this as a template; `ros2 bag record ...` lines are provided in-script as comments.)

- For quick coefficient sampling with odom histograms:
  ```bash
  ros2 run your_pkg analyze_coefficients.py
  ```
  Press `q` to quit and auto-plot histograms.

### 3) Convert ROS 2 bag → dataset folders

Edit paths in `extract_rosbag_to_data.py` to your bag URI, then:
```bash
python Scripts/extract_rosbag_to_data.py
```
Outputs:
- `Sample Data/images/image_<timestamp>.png`
- `Sample Data/odom/odom.csv`
- `Sample Data/cmd_vel/cmd_vel.csv`
- `Sample Data/camera_info/camera_intrinsics.yaml`

### 4) Build projection windows (future poses)

```bash
python Scripts/generate_projection_csv.py
```
Creates `Sample Data/projection.csv` with `x_window, y_window, z_window, theta_window` (length N per image).

### 5) Project trajectories into images (path masks)

- Batch:
  ```bash
  python Scripts/projection_path.py
  ```
  Saves masks to `Sample Data/path/` (white footprints on black backgrounds).
- Single-frame debug:
  ```bash
  python "Scripts/projection_single image.py"
  ```

### 6) EKF + traversability analysis (optional)

```bash
python Scripts/teleop_EKF.py
```
Publishes `/ekf_odom`, broadcasts TF, and at shutdown saves:
- `velocity_and_traversability_comparison_plots.png`
- `odom_and_coeff_histograms.png`

---

## Data Formats

### `Sample Data/projection.csv`
Per-image future trajectory windows for projection.

- `timestamp` *(float)* — image timestamp
- `image_path` *(str)* — relative path (e.g., `images/image_...png`)
- `x_window`, `y_window`, `z_window` *(list[float])* — future positions (length N)
- `theta_window` *(list[float])* — future yaw angles (length N)

### `Sample Data/final_training_data.csv`
Learning-ready table (one row per example).

- `rgb_img`, `depth_img` — image paths
- `path` — projected path mask image
- `mu`, `nu` — traversability coefficient map image paths
- `lin_vel`, `ang_vel` — sampled velocity arrays (stringified lists)
- `x`, `y` — odometry arrays (stringified lists)
- `traversability` — label/summary scalar or map (depending on your variant)
- `timestamps` — base timestamp

> The training pipeline and data are complete and ready for execution once all relevant image data is collected and organized under the directories `/mu`, `/nu`, `/path`, `/color`, and `/depth`.

---

## Key Components

- **EKF state estimation (teleop_EKF.py):**
  8-state filter `[x, y, θ, v, ω, b_ax, b_ay, b_ω]` with IMU bias states, dynamic R tuning for simple slip heuristics. Publishes `/ekf_odom` and TF.

- **Traversability coefficient:**
  Simplified proxy
  \[
    \mu = \frac{v_\text{measured}}{v_\text{commanded}}
  \]
  computed from EKF or odom estimates; used for self-supervision.

- **3D projection:**
  Known intrinsics K and extrinsics [R|t] from `base_link → optical` project future robot footprints into the current image for pixel-level path supervision.

- **Training scaffold (Training/):**
  ResNet-Depth-UNet backbone, dataset loader for synchronized RGB-D + kinematics + masks.

---

## Limitations and Next Steps

- The closed-loop controller in RoboMaster EP held commanded and measured velocities close, yielding inconsistent µ.
- Possible remedies:
  - Reduce or bypass internal velocity control (open-loop tests).
  - Collect on varied surfaces with controlled disturbances (e.g., low-friction patches, ramps).
  - Incorporate auxiliary cues (wheel current, contact microphones, tactile modules).
  - Extend labels using visual slip estimators or re-projection consistency.

---

## For Reviewers

This repository documents a complete, training-ready pipeline for self-supervised traversability in mobile robotics, integrating **state estimation**, **geometric calibration**, and **dataset engineering**. It is intended as a reproducible foundation for research on terrain-aware navigation in resource-constrained robots.

---

## Acknowledgments

- WayFAST: Navigation with Predictive Traversability in the Field (methodological inspiration)
- ROS 2 `robot_localization` documentation
- Intel RealSense D455 documentation

---

## Author

Muhammad Musab Ali Chaudhry — Lahore University of Management Sciences (LUMS)

Contact: 25100190@lums.edu.pk
