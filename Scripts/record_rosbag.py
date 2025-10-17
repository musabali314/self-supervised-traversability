#!/usr/bin/env python3
import sys, select, termios, tty, math, subprocess, time, os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

MAX_SPEED = 0.7
MAX_TURN = 1.5
ACCEL = 0.05
DECEL = 0.05

class TeleopWithRosbag(Node):
    def __init__(self):
        super().__init__('teleop_with_rosbag')
        self.settings = termios.tcgetattr(sys.stdin)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.v, self.w = 0.0, 0.0
        self.v_target, self.w_target = 0.0, 0.0
        self.last_odom = None
        self.last_print_time = time.time()

        self.launch_robomaster()
        # self.start_rosbag_record()
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
            'chassis.force_level:=true'
        ]
        self.robomaster_proc = subprocess.Popen(launch_cmd)
        time.sleep(5)  # Optional: give it time to start up


    def print_help(self):
        print("\n--- TELEOP (ROS-native) + /cmd_vel -> /odom + ROSBAG ---")
        print("Controls:")
        print("  W/S: accelerate / brake")
        print("  A/D: turn left / right")
        print("  X  : stop")
        print("  Q  : quit\n")

    # def start_rosbag_record(self):
    #     timestamp = time.strftime("%Y%m%d_%H%M%S")
    #     bag_dir = os.path.expanduser(f"~/rosbags/teleop_{timestamp}")
    #     # os.makedirs(bag_dir, exist_ok=True)

    #     topics = [
    #         '/cmd_vel',
    #         '/odom',
    #         '/tf',
    #         '/tf_static',
    #         '/camera/color/image_raw',
    #         '/camera/color/camera_info'
    #     ]
    #     cmd = ['ros2', 'bag', 'record', '-o', bag_dir] + topics
    #     self.rosbag_proc = subprocess.Popen(cmd)
    #     print(f"\n[✓] Recording rosbag to: {bag_dir}\n")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def odom_callback(self, msg):
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

    def shutdown(self):
        print("\n--- SHUTDOWN ---")
        self.cmd_pub.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

        # if self.rosbag_proc:
        #     self.rosbag_proc.terminate()
        #     print("[✓] Rosbag recording stopped.")

        if self.robomaster_proc:
            self.robomaster_proc.terminate()
            print("[✓] RoboMaster shutdown complete.")


def main(args=None):
    rclpy.init(args=args)
    teleop = TeleopWithRosbag()
    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()