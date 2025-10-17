#!/usr/bin/env python3
import sys, select, termios, tty, math, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3, Quaternion
from sensor_msgs.msg import Imu
from robomaster import robot

MAX_SPEED = 0.7
MAX_TURN = 1.5
ACCEL = 0.03
DECEL = 0.05

class SmoothTeleop(Node):
    def __init__(self):
        super().__init__('robomaster_smooth_teleop')
        self.settings = termios.tcgetattr(sys.stdin)

        # ROS publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)

        # Initialize RoboMaster
        self.ep_robot = robot.Robot()
        self.ep_robot.initialize(conn_type="rndis")
        self.chassis = self.ep_robot.chassis

        # Subscribe to IMU
        self.chassis.sub_imu(freq=10, callback=self.imu_callback)

        # Control state
        self.v, self.w = 0.0, 0.0
        self.v_target, self.w_target = 0.0, 0.0

        self.print_help()
        self.loop()

    def print_help(self):
        print("\n--- SMOOTH TELEOP ---")
        print("Controls:")
        print("  W/S: accelerate / brake")
        print("  A/D: turn left / right")
        print("  X  : stop")
        print("  Q  : quit\n")

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.01)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def imu_callback(self, data):
        acc_x, acc_y, acc_z, gx, gy, gz = data

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = "imu"

        imu_msg.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        imu_msg.angular_velocity = Vector3(x=gx, y=gy, z=gz)
        imu_msg.linear_acceleration = Vector3(x=acc_x, y=acc_y, z=acc_z)

        self.imu_pub.publish(imu_msg)

    def loop(self):
        rate = self.create_rate(20)

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

                # Smooth control
                self.v += max(min(self.v_target - self.v, DECEL), -DECEL)
                self.w += max(min(self.w_target - self.w, DECEL), -DECEL)

                # ROS cmd_vel
                twist = Twist()
                twist.linear.x = self.v
                twist.angular.z = self.w
                self.cmd_pub.publish(twist)

                # RoboMaster SDK control
                self.chassis.drive_speed(x=self.v, y=0, z=math.degrees(self.w))

                print(f"[v={self.v:.2f}, w={self.w:.2f} rad/s]      ", end="\r")
                rclpy.spin_once(self, timeout_sec=0)

        finally:
            self.shutdown()

    def shutdown(self):
        msg = Twist()
        self.cmd_pub.publish(msg)
        self.chassis.drive_speed(x=0, y=0, z=0)
        self.ep_robot.close()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        print("\n--- TELEOP SHUTDOWN ---")

def main(args=None):
    rclpy.init(args=args)
    teleop = SmoothTeleop()
    teleop.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
