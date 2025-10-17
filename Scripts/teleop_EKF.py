#!/usr/bin/env python3
import sys, select, termios, tty, math, subprocess, time, os, signal
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy # For quaternion to Euler conversion
import csv
import matplotlib.pyplot as plt # Import matplotlib

# Constants for Teleop (remain the same)
MAX_SPEED = 0.7
MAX_TURN = 1.5
ACCEL = 0.1
DECEL = 0.1

# Constants for Odometry Sample Collection (re-added)
TARGET_VEL = 0.3
VEL_TOL = 0.02

# EKF Parameters
# State vector: [x, y, theta, v, omega, b_ax, b_ay, b_omega]
STATE_DIM = 8

# Process noise covariance (Q)
# Represents uncertainty in state propagation from prediction model.
# These values will likely need careful tuning through experimentation.
# If IMU is noisy, Q values related to accel/gyro (and thus v/omega) might need to be larger.
Q = np.diag([
    0.01**2,     # x (m^2)
    0.01**2,     # y (m^2)
    0.01**2,     # theta (rad^2)
    0.05**2,     # v (m/s)^2 - Uncertainty in how IMU accel translates to state v
    0.05**2,     # omega (rad/s)^2 - Uncertainty in how IMU gyro translates to state omega
    0.0001**2,   # b_ax (m/s^2)^2 - very small for bias random walk
    0.0001**2,   # b_ay (m/s^2)^2 - very small for bias random walk
    0.0001**2    # b_omega (rad/s)^2 - very small for bias random walk
])

# Measurement noise covariance (R)
# Will be populated dynamically from sensor covariances where available.
# Otherwise, set nominal values.
# /odom measurements correct the IMU-driven prediction.
# R_ODOM values here represent the *trust* in the /odom reading.
# Higher values mean less trust.
R_ODOM = np.diag([
    0.1**2,  # x_odom (m^2)
    0.1**2,  # y_odom (m^2)
    0.1**2,  # theta_odom (rad^2)
    0.05**2, # v_odom (m/s)^2
    0.05**2  # omega_odom (rad/s)^2
])

# Gravitational acceleration (m/s^2)
GRAVITY = 9.81

class ExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, Q):
        self.x = initial_state      # State vector [x, y, theta, v, omega, b_ax, b_ay, b_omega]
        self.P = initial_covariance # Covariance matrix
        self.Q = Q                  # Process noise covariance

        self.last_timestamp = None
        self.cmd_vel_input = np.array([0.0, 0.0]) # [v_cmd, w_cmd] (used in the prediction if IMU input is unavailable)

        # Store latest IMU inputs for prediction step
        self.imu_accel_x_input = 0.0
        self.imu_accel_y_input = 0.0 # Keeping for consistency, though user ignored linear.y
        self.imu_angular_z_input = 0.0

    def set_cmd_vel(self, v_cmd, w_cmd):
        self.cmd_vel_input = np.array([v_cmd, w_cmd])

    # New method to set IMU inputs for prediction
    def set_imu_inputs(self, accel_x, accel_y, angular_z):
        self.imu_accel_x_input = accel_x
        self.imu_accel_y_input = accel_y
        self.imu_angular_z_input = angular_z

    def predict(self, current_time):
        if self.last_timestamp is None:
            self.last_timestamp = current_time
            return # Cannot predict without a previous timestamp

        dt = current_time - self.last_timestamp
        if dt <= 0: return # No time elapsed or time went backward

        # State elements for convenience
        x, y, theta, v, omega, b_ax, b_ay, b_omega = self.x.flatten()

        # 1. State Transition Function (f)
        # WayFAST-like prediction: drive state with IMU measurements, compensating for biases.
        # This assumes IMU accel_x is along robot's forward axis and accel_y is sideways.
        # This is the "u_k" part from WayFAST where IMU measurements compose the input for f.

        # Position and orientation prediction based on current velocity
        x_pred = x + (v * math.cos(theta)) * dt
        y_pred = y + (v * math.sin(theta)) * dt
        theta_pred = theta + (omega) * dt

        # Velocity prediction using IMU accelerations corrected for bias
        # Assuming v corresponds to linear.x
        v_pred = v + (self.imu_accel_x_input - b_ax) * dt
        omega_pred = omega + (self.imu_angular_z_input - b_omega) * dt

        # Biases are modeled as random walks (they don't change based on control input in prediction)
        b_ax_pred = b_ax
        b_ay_pred = b_ay
        b_omega_pred = b_omega

        self.x = np.array([
            [x_pred],
            [y_pred],
            [theta_pred],
            [v_pred],
            [omega_pred],
            [b_ax_pred],
            [b_ay_pred],
            [b_omega_pred]
        ])

        # 2. Jacobian of the State Transition Function (F)
        # F = df/dx evaluated at x_hat_(k-1|k-1)
        F = np.eye(STATE_DIM)
        F[0, 2] = -v * math.sin(theta) * dt      # dx/d_theta
        F[0, 3] = math.cos(theta) * dt           # dx/dv
        F[1, 2] = v * math.cos(theta) * dt       # dy/d_theta
        F[1, 3] = math.sin(theta) * dt           # dy/dv
        F[2, 4] = dt                             # d_theta/d_omega

        # For v_pred = v + (imu_accel_x - b_ax) * dt:
        F[3, 3] = 1.0                            # dv_pred/dv
        F[3, 5] = -dt                            # dv_pred/d_b_ax (negative because we subtract bias)

        # For omega_pred = omega + (imu_angular_z - b_omega) * dt:
        F[4, 4] = 1.0                            # d_omega_pred/d_omega
        F[4, 7] = -dt                            # d_omega_pred/d_b_omega

        # 3. Project the error covariance forward (P_k_k_minus_1)
        self.P = F @ self.P @ F.T + self.Q
        self.last_timestamp = current_time

        # Debugging: check for NaN in state or covariance
        if np.isnan(self.x).any() or np.isnan(self.P).any():
            print("EKF NaN detected in predict step! Resetting.")
            self.x = np.zeros((STATE_DIM, 1))
            self.P = np.eye(STATE_DIM) * 0.1 # Re-initialize with some uncertainty

    def update(self, measurement_type, z, R):
        # State elements for convenience (predicted state)
        x, y, theta, v, omega, b_ax, b_ay, b_omega = self.x.flatten()

        # 1. Predicted measurement (z_hat_k_k_minus_1) and Jacobian of measurement model (H)
        H = np.zeros((z.shape[0], STATE_DIM)) # H matrix depends on measurement type

        if measurement_type == "odom":
            # h_odom(x_k) = [x_k, y_k, theta_k, v_k, omega_k]
            z_hat = np.array([[x], [y], [theta], [v], [omega]])
            H[0, 0] = 1.0 # dx_odom/dx
            H[1, 1] = 1.0 # dy_odom/dy
            H[2, 2] = 1.0 # d_theta_odom/d_theta
            H[3, 3] = 1.0 # dv_odom/dv
            H[4, 4] = 1.0 # d_omega_odom/d_omega
        else:
            print(f"Unknown measurement type: {measurement_type}")
            return

        # 2. Innovation (measurement residual)
        y_res = z - z_hat

        # Normalize angle residual (theta) if it's part of the measurement
        if measurement_type == "odom": # Only odom measurement includes theta
            y_res[2, 0] = math.atan2(math.sin(y_res[2, 0]), math.cos(y_res[2, 0]))

        # 3. Innovation (Measurement Residual) Covariance (S)
        S = H @ self.P @ H.T + R

        # 4. Kalman Gain (K)
        K = self.P @ H.T @ np.linalg.inv(S)

        # 5. Update the state estimate
        self.x = self.x + K @ y_res

        # Normalize theta in state
        self.x[2, 0] = math.atan2(math.sin(self.x[2, 0]), math.cos(self.x[2, 0]))

        # 6. Update the error covariance
        I = np.eye(STATE_DIM)
        # Joseph form for numerical stability
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T

        # Debugging: check for NaN in state or covariance
        if np.isnan(self.x).any() or np.isnan(self.P).any():
            print("EKF NaN detected in update step! Resetting.")
            self.x = np.zeros((STATE_DIM, 1))
            self.P = np.eye(STATE_DIM) * 0.1 # Re-initialize with some uncertainty


class TeleopWithEKF(Node):
    def __init__(self):
        super().__init__('teleop_with_ekf')
        self.settings = termios.tcgetattr(sys.stdin)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.ekf_odom_pub = self.create_publisher(Odometry, '/ekf_odom', 10) # Publish EKF estimated odometry
        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        # Create EKF instance
        initial_state = np.zeros((STATE_DIM, 1))
        initial_covariance = np.eye(STATE_DIM) * 0.1 # Small initial uncertainty
        # Initial position and orientation can be known (0,0,0) with high certainty
        initial_covariance[0,0] = 0.001 # x
        initial_covariance[1,1] = 0.001 # y
        initial_covariance[2,2] = 0.001 # theta

        self.ekf = ExtendedKalmanFilter(initial_state, initial_covariance, Q)

        self.v, self.w = 0.0, 0.0
        self.v_target, self.w_target = 0.0, 0.0
        self.last_odom_msg = None
        self.last_imu_msg = None
        self.last_ekf_update_time = self.get_clock().now().nanoseconds / 1e9
        self.last_print_time = time.time()

        # Store latest data for EKF (will be processed on a timer)
        self.latest_odom = None
        self.latest_imu = None

        # --- Data collection for plotting ---
        self.timestamps = []
        self.odom_v_data = []
        self.odom_w_data = []
        self.ekf_v_data = []
        self.ekf_w_data = []
        self.commanded_v_data = [] # To plot commanded velocity
        self.traversability_coeffs = [] # To store calculated traversability coefficients
        self.start_time = time.time() # To normalize timestamps

        # --- Data collection for histograms (re-added) ---
        self.odom_samples_hist = []
        self.coeff_samples_hist = []

        # Timer for EKF prediction/update at a fixed rate
        self.ekf_timer = self.create_timer(1.0/30.0, self.ekf_loop) # 30 Hz for EKF

        self.launch_robomaster()
        self.print_help()
        self.loop()

    def launch_robomaster(self):
        print("[✓] Launching RoboMaster driver...")
        launch_cmd = [
            'ros2', 'launch', 'robomaster_ros', 'main.launch',
            'model:=ep',
            'conn_type:=rndis',
            'chassis.enabled:=true',
            'chassis.rate:=30', # Increase rate for better EKF performance
            'chassis.imu_includes_orientation:=true', # RoboMaster IMU publishes orientation (quaternion)
            'chassis.force_level:=true'
        ]
        self.robomaster_proc = subprocess.Popen(launch_cmd, preexec_fn=os.setsid) # Added preexec_fn for proper termination
        time.sleep(5)  # Optional: give it time to start up

    def print_help(self):
        print("\n--- TELEOP (ROS-native) + EKF State Estimation ---")
        print("Controls:")
        print("    W/S: accelerate / brake")
        print("    A/D: turn left / right")
        print("    X  : stop")
        print("    Q  : quit and plot velocities\n")
        print(f"Collecting odom.linear.x where cmd_vel.linear.x ≈ {TARGET_VEL} AND angular.z = 0 for histograms\n")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def odom_callback(self, msg):
        self.latest_odom = msg

        # Collect samples for histogram when criteria are met
        if abs(self.v - TARGET_VEL) < VEL_TOL and abs(self.w) < 1e-3:
            v_odom = msg.twist.twist.linear.x
            coeff = v_odom / TARGET_VEL
            self.odom_samples_hist.append(v_odom)
            self.coeff_samples_hist.append(coeff)

    def imu_callback(self, msg):
        self.latest_imu = msg

    def ekf_loop(self):
        current_time_ns = self.get_clock().now().nanoseconds
        current_time_sec = current_time_ns / 1e9

        # Set IMU inputs for prediction
        if self.latest_imu:
            # Compensate for gravity in linear acceleration
            # Linear acceleration from IMU is in body frame.
            # Gravity vector in world frame is [0, 0, -GRAVITY].
            # Rotate orientation from IMU (world to body)
            quat = self.latest_imu.orientation
            r_imu = R_scipy.from_quat([quat.x, quat.y, quat.z, quat.w])

            # Roll, Pitch, Yaw from IMU orientation
            roll, pitch, yaw = r_imu.as_euler('xyz')

            # Gravity component along body's X-axis
            gravity_x = -GRAVITY * math.sin(pitch)

            # Compensated linear acceleration for prediction
            compensated_accel_x = self.latest_imu.linear_acceleration.x - gravity_x

            self.ekf.set_imu_inputs(compensated_accel_x, self.latest_imu.linear_acceleration.y, self.latest_imu.angular_velocity.z)
        else:
            # If no IMU data, use zero accelerations or fallback to previous (less accurate) prediction.
            self.ekf.set_imu_inputs(0.0, 0.0, 0.0)

        # EKF Prediction Step
        self.ekf.predict(current_time_sec)

        # EKF Update Step - process latest available Odom data
        if self.latest_odom:
            # Convert odometry quaternion to yaw
            quat = self.latest_odom.pose.pose.orientation
            r_odom = R_scipy.from_quat([quat.x, quat.y, quat.z, quat.w])
            _, _, yaw_odom = r_odom.as_euler('xyz') # Roll, Pitch, Yaw

            z_odom = np.array([
                [self.latest_odom.pose.pose.position.x],
                [self.latest_odom.pose.pose.position.y],
                [yaw_odom],
                [self.latest_odom.twist.twist.linear.x],
                [self.latest_odom.twist.twist.angular.z]
            ])
            # Use covariance from /odom topic if available, otherwise R_ODOM
            R_odom_actual = np.zeros((5,5))
            R_odom_actual[0,0] = self.latest_odom.pose.covariance[0]   # x variance
            R_odom_actual[1,1] = self.latest_odom.pose.covariance[7]   # y variance
            R_odom_actual[2,2] = self.latest_odom.pose.covariance[35]  # yaw variance (idx 5 in 6x6 for rot_z)
            R_odom_actual[3,3] = self.latest_odom.twist.covariance[0]  # vx variance
            R_odom_actual[4,4] = self.latest_odom.twist.covariance[35] # omega_z variance

            # --- Dynamic Adjustment of R (Simplified Slip Detection) ---
            ekf_v_est = self.ekf.x.flatten()[3] # EKF's estimated linear velocity
            odom_v = self.latest_odom.twist.twist.linear.x # Odometry reported linear velocity

            # Simple slip detection: If Odom reports much higher velocity than EKF's estimate
            # And robot is commanded to move
            if abs(self.v) > 0.1 and odom_v > ekf_v_est + 0.1: # Thresholds will need tuning
                # Temporarily increase the linear velocity variance in R_odom_actual
                # This tells the EKF to give less weight to the potentially slipping wheel encoder data
                R_odom_actual[3,3] = max(R_odom_actual[3,3] * 5.0, R_ODOM[3,3] * 10.0) # Multiply by a factor, cap at a max
                self.get_logger().warn(f"Slip heuristic: Odom_v {odom_v:.2f} > EKF_v_est {ekf_v_est:.2f}. Increasing R[vx]!")
            elif abs(self.v) > 0.1 and odom_v < ekf_v_est - 0.1: # If Odom is much lower than EKF est (e.g. stuck, but EKF still predicts motion from IMU)
                   # This might indicate that the IMU driven prediction is too optimistic, or odom is truly stuck.
                   # Decrease the linear velocity variance in R_odom_actual slightly, to trust odom more
                   R_odom_actual[3,3] = min(R_odom_actual[3,3] / 2.0, R_ODOM[3,3] / 2.0)
                   self.get_logger().warn(f"Odom_v {odom_v:.2f} < EKF_v_est {ekf_v_est:.2f}. Decreasing R[vx]!")
            else:
                # If no significant slip detected, revert to nominal or message covariance
                pass # Already initialized R_odom_actual, just let it be

            # Fallback to predefined R_ODOM if message covariance is all zeros (unknown)
            if np.all(R_odom_actual == 0):
                R_odom_actual = R_ODOM

            self.ekf.update("odom", z_odom, R_odom_actual)

        # Publish EKF estimated odometry
        self.publish_ekf_odom(current_time_ns)

        # --- Collect data for time-series plotting ---
        # Only collect if we have fresh odom data to ensure synchronized timestamps
        if self.latest_odom:
            self.timestamps.append(time.time() - self.start_time) # Relative time from start
            self.odom_v_data.append(self.latest_odom.twist.twist.linear.x)
            self.odom_w_data.append(self.latest_odom.twist.twist.angular.z)

            # EKF state is [x, y, theta, v, omega, b_ax, b_ay, b_omega]
            v_ekf, w_ekf = self.ekf.x.flatten()[3:5] # Extract v and omega
            self.ekf_v_data.append(v_ekf)
            self.ekf_w_data.append(w_ekf)
            self.commanded_v_data.append(self.v) # Store commanded velocity

            # Calculate and store traversability coefficient
            # This is a simplified proxy for WayFAST's mu
            if abs(self.v) > 0.05: # Avoid division by zero, only calculate when commanded to move
                coeff = abs(v_ekf) / abs(self.v)
                self.traversability_coeffs.append(min(coeff, 1.5)) # Coefficient shouldn't exceed 1.0 normally
            else:
                self.traversability_coeffs.append(0.0) # Or NaN, depending on how you want to handle no motion

            self.latest_odom = None # Now safe to clear after data collection
            self.latest_imu = None # Clear IMU as well once combined data processed

    def publish_ekf_odom(self, timestamp_ns):
        ekf_odom_msg = Odometry()
        ekf_odom_msg.header.stamp = self.get_clock().now().to_msg() # Use current ROS time
        ekf_odom_msg.header.frame_id = "odom" # Global frame
        ekf_odom_msg.child_frame_id = "base_link_ekf" # EKF estimated robot frame

        # EKF state is [x, y, theta, v, omega, b_ax, b_ay, b_omega]
        x_est, y_est, theta_est, v_est, omega_est, _, _, _ = self.ekf.x.flatten()

        ekf_odom_msg.pose.pose.position.x = x_est
        ekf_odom_msg.pose.pose.position.y = y_est
        ekf_odom_msg.pose.pose.position.z = 0.0 # 2D

        # Convert yaw (theta_est) to quaternion
        q_est = R_scipy.from_euler('xyz', [0, 0, theta_est]).as_quat()
        ekf_odom_msg.pose.pose.orientation.x = q_est[0]
        ekf_odom_msg.pose.pose.orientation.y = q_est[1]
        ekf_odom_msg.pose.pose.orientation.z = q_est[2]
        ekf_odom_msg.pose.pose.orientation.w = q_est[3]

        ekf_odom_msg.twist.twist.linear.x = v_est
        ekf_odom_msg.twist.twist.linear.y = 0.0 # 2D
        ekf_odom_msg.twist.twist.linear.z = 0.0
        ekf_odom_msg.twist.twist.angular.x = 0.0
        ekf_odom_msg.twist.twist.angular.y = 0.0
        ekf_odom_msg.twist.twist.angular.z = omega_est

        # Populate covariance from EKF's P matrix
        ekf_odom_msg.pose.covariance[0] = self.ekf.P[0,0] # x
        ekf_odom_msg.pose.covariance[7] = self.ekf.P[1,1] # y
        ekf_odom_msg.pose.covariance[35] = self.ekf.P[2,2] # theta (yaw)
        ekf_odom_msg.pose.covariance[1] = self.ekf.P[0,1]
        ekf_odom_msg.pose.covariance[5] = self.ekf.P[0,2] # x-theta
        ekf_odom_msg.pose.covariance[6] = self.ekf.P[1,0]
        ekf_odom_msg.pose.covariance[11] = self.ekf.P[1,2] # y-theta
        ekf_odom_msg.pose.covariance[30] = self.ekf.P[2,0] # theta-x
        ekf_odom_msg.pose.covariance[31] = self.ekf.P[2,1] # theta-y

        ekf_odom_msg.twist.covariance[0] = self.ekf.P[3,3] # v
        ekf_odom_msg.twist.covariance[35] = self.ekf.P[4,4] # omega
        ekf_odom_msg.twist.covariance[1] = self.ekf.P[3,4] # v-omega
        ekf_odom_msg.twist.covariance[6] = self.ekf.P[4,3] # omega-v

        self.ekf_odom_pub.publish(ekf_odom_msg)

        # Broadcast TF transform for visualization and other nodes
        t = TransformStamped()
        t.header.stamp = ekf_odom_msg.header.stamp
        t.header.frame_id = "odom" # Parent frame
        t.child_frame_id = "base_link_ekf" # Child frame (your EKF estimated robot base)
        t.transform.translation.x = x_est
        t.transform.translation.y = y_est
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q_est[0]
        t.transform.rotation.y = q_est[1]
        t.transform.rotation.z = q_est[2]
        t.transform.rotation.w = q_est[3]
        self.tf_broadcaster.sendTransform(t)


    def loop(self):
        try:
            while rclpy.ok():
                key = self.get_key()

                if key == 'w':
                    self.v_target = min(self.v_target + ACCEL, MAX_SPEED)
                elif key == 's':
                    self.v_target = max(self.v_target - ACCEL, -MAX_SPEED)
                elif key == 'a':
                    self.w_target = min(self.w_target + 0.05, MAX_TURN)
                elif key == 'd':
                    self.w_target = max(self.w_target - 0.05, -MAX_TURN)
                elif key == 'x':
                    self.v_target = 0.0
                    self.w_target = 0.0
                elif key == 'q':
                    break

                self.v += max(min(self.v_target - self.v, DECEL), -DECEL)
                self.w += max(min(self.w_target - self.w, DECEL), -DECEL)

                twist = Twist()
                twist.linear.x = self.v
                twist.angular.z = self.w
                self.cmd_pub.publish(twist)

                now = time.time()
                # Print status
                if self.latest_odom and now - self.last_print_time >= 1.0:
                    odom_msg = self.latest_odom
                    pos_odom = odom_msg.pose.pose.position
                    ori_odom = odom_msg.pose.pose.orientation
                    v_odom = odom_msg.twist.twist.linear.x
                    w_odom = odom_msg.twist.twist.angular.z

                    x_ekf, y_ekf, theta_ekf, v_ekf, w_ekf, b_ax_ekf, b_ay_ekf, b_omega_ekf = self.ekf.x.flatten()

                    # Safely access traversability_coeffs
                    current_trav_coeff = self.traversability_coeffs[-1] if self.traversability_coeffs else 0.0

                    print(f"\n[cmd_vel] v={self.v:.2f}, w={self.w:.2f} | "
                          f"[odom] x={pos_odom.x:.2f}, y={pos_odom.y:.2f}, "
                          f"v={v_odom:.2f}, w={w_odom:.2f} | "
                          f"[EKF Est] x={x_ekf:.2f}, y={y_ekf:.2f}, theta={math.degrees(theta_ekf):.1f} deg, "
                          f"v={v_ekf:.2f}, w={w_ekf:.2f} | "
                          f"Bias: B_ax={b_ax_ekf:.4f}, B_ay={b_ay_ekf:.4f}, B_w={b_omega_ekf:.4f} | "
                          f"Trav Coeff: {current_trav_coeff:.2f}")
                    self.last_print_time = now

                rclpy.spin_once(self, timeout_sec=0)

        finally:
            self.shutdown()

    def shutdown(self):
        print("\n--- SHUTDOWN ---")
        self.cmd_pub.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

        if self.robomaster_proc:
            os.killpg(os.getpgid(self.robomaster_proc.pid), signal.SIGTERM) # Use os.killpg for group termination
            self.robomaster_proc.wait()
            print("[✓] RoboMaster driver terminated.")

        # --- Plotting logic using matplotlib (time series) ---
        if self.timestamps:
            print("\n[✓] Generating velocity and traversability comparison plots...")

            # Create a figure with two subplots side-by-side
            fig, axs = plt.subplots(3, 1, figsize=(10, 12)) # 3 rows for linear, angular, and traversability

            # Plot Linear Velocities
            axs[0].plot(self.timestamps, self.commanded_v_data, label='Commanded v', color='black', linestyle=':')
            axs[0].plot(self.timestamps, self.odom_v_data, label='Odometry v', color='blue')
            axs[0].plot(self.timestamps, self.ekf_v_data, label='EKF Estimated v', color='red', linestyle='--')
            axs[0].set_title('Linear Velocity Comparison (Commanded vs. Odometry vs. EKF)')
            axs[0].set_ylabel('Velocity (m/s)')
            axs[0].legend()
            axs[0].grid(True)

            # Plot Angular Velocities
            axs[1].plot(self.timestamps, self.odom_w_data, label='Odometry w', color='green')
            axs[1].plot(self.timestamps, self.ekf_w_data, label='EKF Estimated w', color='purple', linestyle='--')
            axs[1].set_title('Angular Velocity Comparison (Odometry vs. EKF)')
            axs[1].set_ylabel('Angular Velocity (rad/s)')
            axs[1].legend()
            axs[1].grid(True)

            # Plot Traversability Coefficient
            axs[2].plot(self.timestamps, self.traversability_coeffs, label='Traversability Coeff (EKF_v / Cmd_v)', color='orange')
            axs[2].set_title('Traversability Coefficient')
            axs[2].set_xlabel('Time (s)')
            axs[2].set_ylabel('Coefficient (0-1)')
            axs[2].set_ylim(0, 1.1) # Ensure y-axis is sensible for coefficient
            axs[2].legend()
            axs[2].grid(True)

            plt.tight_layout() # Adjust subplot parameters for a tight layout
            plt.savefig("velocity_and_traversability_comparison_plots.png")
            print(f"[✓] Velocity and traversability comparison plot saved to velocity_and_traversability_comparison_plots.png")

        else:
            print("No velocity data recorded for time-series plotting.")

        # --- Plotting logic for histograms (new section) ---
        if self.odom_samples_hist:
            print("\n[✓] Generating odometry and coefficient histograms...")
            fig_hist, axs_hist = plt.subplots(1, 2, figsize=(14, 5))

            axs_hist[0].hist(self.odom_samples_hist, bins=30, color='lightblue', edgecolor='black')
            axs_hist[0].set_title(f"Odom linear.x Histogram (v ≈ {TARGET_VEL}, w = 0)")
            axs_hist[0].set_xlabel("Odom linear.x (m/s)")
            axs_hist[0].set_ylabel("Frequency")
            axs_hist[0].grid(True)

            axs_hist[1].hist(self.coeff_samples_hist, bins=np.linspace(0, 1.5, 45), color='lightgreen', edgecolor='black')
            axs_hist[1].set_title("Traversability Coefficient Histogram")
            axs_hist[1].set_xlabel("Coefficient (0–1)")
            axs_hist[1].set_ylabel("Frequency")
            # axs_hist[1].set_ylim(0, 1.1) # Set y-axis limit for coefficient
            axs_hist[1].grid(True)

            plt.tight_layout()
            plt.savefig("odom_and_coeff_histograms.png")
            print(f"[✓] Odometry and coefficient histograms saved to odom_and_coeff_histograms.png")
        else:
            print("No odometry samples recorded for histogram plotting matching criteria.")


def main(args=None):
    rclpy.init(args=args)
    teleop = TeleopWithEKF()
    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()