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
ACCEL = 0.1
DECEL = 0.1

TARGET_VEL = 0.4
VEL_TOL = 0.02

class TeleopWithOdomAnalysis(Node):
    def __init__(self):
        super().__init__('teleop_with_odom_analysis')
        self.settings = termios.tcgetattr(sys.stdin)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.v, self.w = 0.0, 0.0
        self.v_target, self.w_target = 0.0, 0.0
        self.last_odom = None
        self.last_print_time = time.time()
        self.odom_samples = []
        self.coeff_samples = []
        self.robomaster_proc = None # Initialize to None

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
            'chassis.rate:=15',
            'chassis.imu_includes_orientation:=true',
            'chassis.force_level:=true',
            #'chassis.odom_twist_from_pose_diff:=true'
        ]
        # Use preexec_fn=os.setsid to create a new process group for clean termination
        self.robomaster_proc = subprocess.Popen(launch_cmd, preexec_fn=os.setsid)
        time.sleep(5)

    def print_help(self):
        print("\n--- TELEOP + ODOM ANALYSIS ---")
        print("Controls:")
        print("   W/S: accelerate / brake")
        print("   A/D: turn left / right")
        print("   X  : stop")
        print("   Q  : quit and plot histograms (will also attempt to restart Ethernet)\n")
        print(f"Collecting odom.linear.x where cmd_vel.linear.x ≈ {TARGET_VEL} AND angular.z = 0\n")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def odom_callback(self, msg):
        if abs(self.v - TARGET_VEL) < VEL_TOL and abs(self.w) < 1e-3:
            v_odom = msg.twist.twist.linear.x
            coeff = v_odom / TARGET_VEL
            self.odom_samples.append(v_odom)
            self.coeff_samples.append(coeff)

        self.last_odom = msg

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
                if self.last_odom and now - self.last_print_time >= 1.0:
                    pos = self.last_odom.pose.pose.position
                    ori = self.last_odom.pose.pose.orientation
                    v_odom = self.last_odom.twist.twist.linear.x
                    w_odom = self.last_odom.twist.twist.angular.z

                    print(f"\n[cmd_vel] v={self.v:.2f}, w={self.w:.2f}  |  "
                          f"[odom] x={pos.x:.2f}, y={pos.y:.2f}, z={pos.z:.2f}, "
                          f"orientation=[{ori.x:.2f}, {ori.y:.2f}, {ori.z:.2f}, {ori.w:.2f}]  |  "
                          f"[odom_twist] v={v_odom:.2f}, w={w_odom:.2f}")
                    self.last_print_time = now

                rclpy.spin_once(self, timeout_sec=0)

        finally:
            self.shutdown()

    def restart_ethernet_connection(self, interface_name="usb0"): # Set default to 'usb0'
        """
        Restarts the specified Ethernet connection.
        NOTE: This typically requires root/sudo privileges.
        You might be prompted for a password or need to run the script with sudo.
        """
        print(f"\n[!] Attempting to restart Ethernet interface: {interface_name}...")
        try:
            # Command to bring the interface down
            subprocess.run(['sudo', 'ip', 'link', 'set', interface_name, 'down'], check=True)
            time.sleep(1) # Give it a moment to go down

            # Command to bring the interface up
            subprocess.run(['sudo', 'ip', 'link', 'set', interface_name, 'up'], check=True)
            print(f"[✓] Ethernet interface {interface_name} restarted successfully.")
            print("[i] Please wait a few seconds for network services to reinitialize and Robomaster to reconnect.")
        except subprocess.CalledProcessError as e:
            print(f"[X] Failed to restart Ethernet interface {interface_name}: {e}")
            print("    Please ensure you have the correct interface name and necessary permissions (e.g., run with sudo).")
        except FileNotFoundError:
            print("[X] 'ip' command not found. Ensure iproute2 is installed or use equivalent commands for your system (e.g., 'ifconfig').")
        except Exception as e:
            print(f"[X] An unexpected error occurred during Ethernet restart: {e}")


    def shutdown(self):
        print("\n--- SHUTDOWN ---")
        self.cmd_pub.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

        if self.robomaster_proc:
            # Terminate the entire process group
            os.killpg(os.getpgid(self.robomaster_proc.pid), signal.SIGTERM)
            self.robomaster_proc.wait()
            print("[✓] RoboMaster driver terminated.")

        # Call the function to restart Ethernet connection, using 'usb0'
        self.restart_ethernet_connection(interface_name="usb0")


        if self.odom_samples:
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))

            axs[0].hist(self.odom_samples, bins=30, color='lightblue', edgecolor='black')
            axs[0].set_title(f"Odom linear.x Histogram (v ≈ {TARGET_VEL}, w = 0)")
            axs[0].set_xlabel("Odom linear.x (m/s)")
            axs[0].set_ylabel("Frequency")
            axs[0].grid(True)

            axs[1].hist(self.coeff_samples, bins=np.linspace(0, 2, 30), color='lightgreen', edgecolor='black')
            axs[1].set_title("Traversability Coefficient Histogram")
            axs[1].set_xlabel("Coefficient (0–1)")
            axs[1].set_ylabel("Frequency")
            axs[1].grid(True)

            plt.tight_layout()
            plt.show()
        else:
            print("No odometry samples recorded matching criteria.")


def main(args=None):
    rclpy.init(args=args)
    teleop = TeleopWithOdomAnalysis()
    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()