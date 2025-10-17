#!/usr/bin/env python3
import sys, select, termios, tty, math, subprocess, time, os, signal
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import matplotlib.pyplot as plt
import numpy as np

MAX_SPEED = 0.7
MAX_TURN = 1.5
ACCEL = 0.05
DECEL = 0.05
TARGET_VEL = 0.5  # used for computing traversability coefficient

class TeleopWithEKFAnalysis(Node):
    def __init__(self):
        super().__init__('teleop_with_ekf_analysis')
        self.settings = termios.tcgetattr(sys.stdin)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/ekf/odometry', self.ekf_callback, 10)

        self.v, self.w = 0.0, 0.0
        self.v_target, self.w_target = 0.0, 0.0
        self.last_ekf_odom = None
        self.last_print_time = time.time()

        self.ekf_velocity_samples = []
        self.coeff_samples = []

        self.launch_sensors_and_ekf()
        self.print_help()
        self.loop()

    def launch_sensors_and_ekf(self):
        print("[✓] Launching RoboMaster driver...")
        self.robomaster_proc = subprocess.Popen([
            'ros2', 'launch', 'robomaster_ros', 'main.launch',
            'model:=ep',
            'conn_type:=rndis',
            'chassis.enabled:=true',
            'chassis.rate:=15',
            'chassis.imu_includes_orientation:=true',
            'chassis.force_level:=true'
        ], preexec_fn=os.setsid)

        print("[✓] Launching Razor 9DOF IMU driver...")
        self.imu_proc = subprocess.Popen([
            'ros2', 'run', 'razor_imu_9dof', 'razor_imu_9dof',
            '--ros-args', '--remap', 'imu:=razor_imu'
        ], preexec_fn=os.setsid)

        print("[✓] Launching EKF node...")
        self.ekf_proc = subprocess.Popen([
            'ros2', 'launch', 'robot_localization', 'ekf.launch.py'
        ], preexec_fn=os.setsid)

        time.sleep(10)  # Allow time for all nodes to start

    def print_help(self):
        print("\n--- TELEOP + EKF VELOCITY ANALYSIS ---")
        print("Controls:")
        print("  W/S: accelerate / brake")
        print("  A/D: turn left / right")
        print("  X  : stop")
        print("  Q  : quit and plot histograms")
        print("\nDisplaying fused velocity from EKF...\n")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def ekf_callback(self, msg):
        self.last_ekf_odom = msg
        v_ekf = msg.twist.twist.linear.x
        if abs(self.v - TARGET_VEL) < 0.02 and abs(self.w) < 1e-3:
            coeff = v_ekf / TARGET_VEL  # Not clipped
            self.ekf_velocity_samples.append(v_ekf)
            self.coeff_samples.append(coeff)

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
                if self.last_ekf_odom and now - self.last_print_time >= 1.0:
                    v_ekf = self.last_ekf_odom.twist.twist.linear.x
                    w_ekf = self.last_ekf_odom.twist.twist.angular.z
                    print(f"\n[cmd_vel] v={self.v:.2f}, w={self.w:.2f}  |  "
                          f"[EKF_twist] v={v_ekf:.2f}, w={w_ekf:.2f}")
                    self.last_print_time = now

                rclpy.spin_once(self, timeout_sec=0)
        finally:
            self.shutdown()

    def shutdown(self):
        print("\n--- SHUTDOWN ---")
        self.cmd_pub.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

        if hasattr(self, 'robomaster_proc') and self.robomaster_proc:
            os.killpg(os.getpgid(self.robomaster_proc.pid), signal.SIGTERM)
            self.robomaster_proc.wait()
            print("[✓] RoboMaster driver terminated.")

        if hasattr(self, 'imu_proc') and self.imu_proc:
            os.killpg(os.getpgid(self.imu_proc.pid), signal.SIGTERM)
            self.imu_proc.wait()
            print("[✓] Razor IMU driver terminated.")

        if hasattr(self, 'ekf_proc') and self.ekf_proc:
            os.killpg(os.getpgid(self.ekf_proc.pid), signal.SIGTERM)
            self.ekf_proc.wait()
            print("[✓] EKF node terminated.")

        if self.ekf_velocity_samples and self.coeff_samples:
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))

            axs[0].hist(self.ekf_velocity_samples, bins=30, color='skyblue', edgecolor='black')
            axs[0].set_title(f"EKF Velocity Histogram (v ≈ {TARGET_VEL}, w ≈ 0)")
            axs[0].set_xlabel("EKF Linear.x (m/s)")
            axs[0].set_ylabel("Frequency")
            axs[0].grid(True)

            axs[1].hist(self.coeff_samples,
                        bins=np.linspace(min(self.coeff_samples), max(self.coeff_samples), 30),
                        color='lightgreen', edgecolor='black')
            axs[1].set_title("Traversability Coefficient Histogram")
            axs[1].set_xlabel("Coefficient (EKF_v / cmd_v)")
            axs[1].set_ylabel("Frequency")
            axs[1].grid(True)

            plt.tight_layout()
            plt.show()
        else:
            print("No valid EKF velocity or coefficient samples recorded.")

def main(args=None):
    rclpy.init(args=args)
    teleop = TeleopWithEKFAnalysis()
    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
