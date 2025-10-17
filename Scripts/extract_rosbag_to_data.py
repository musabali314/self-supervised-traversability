import rclpy
import os
import cv2
import yaml
import numpy as np
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import rosbag2_py

DATA_DIR = "/home/orangepi/AV_Project/av_project/data"
IMG_DIR = os.path.join(DATA_DIR, "images")
CAMERA_INFO_PATH = os.path.join(DATA_DIR, "camera_info", "camera_intrinsics.yaml")
ODOM_PATH = os.path.join(DATA_DIR, "odom", "odom.csv")
CMD_VEL_PATH = os.path.join(DATA_DIR, "cmd_vel", "cmd_vel.csv")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CAMERA_INFO_PATH), exist_ok=True)
os.makedirs(os.path.dirname(ODOM_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CMD_VEL_PATH), exist_ok=True)

bridge = CvBridge()

def write_camera_info(msg):
    intrinsics = {
        'height': msg.height,
        'width': msg.width,
        'K': list(msg.k),
        'D': list(msg.d),
        'R': list(msg.r),
        'P': list(msg.p),
        'distortion_model': msg.distortion_model
    }
    with open(CAMERA_INFO_PATH, 'w') as f:
        yaml.dump(intrinsics, f)

def main():
    rclpy.init()

    storage_options = rosbag2_py.StorageOptions(
        uri='/home/orangepi/rosbags/teleop_20250703_071901/teleop_20250703_071901_0.db3',  # Use full path if needed
        storage_id='sqlite3'
    )

    converter_options = rosbag2_py.ConverterOptions('', '')
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    odom_file = open(ODOM_PATH, 'w')
    odom_file.write("timestamp,x,y,z,qx,qy,qz,qw\n")

    cmd_file = open(CMD_VEL_PATH, 'w')
    cmd_file.write("timestamp,linear_x,angular_z\n")

    type_map = {}
    written_camera_info = False

    while reader.has_next():
        topic, data, t = reader.read_next()
        msg_type = type_map.get(topic)

        if not msg_type:
            msg_type_str = reader.get_all_topics_and_types()
            for entry in msg_type_str:
                if entry.name == topic:
                    msg_type = get_message(entry.type)
                    type_map[topic] = msg_type
                    break

        if not msg_type:
            continue

        msg = deserialize_message(data, msg_type)
        timestamp = t * 1e-9  # nanoseconds to seconds

        if topic.endswith("image_raw"):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            filename = os.path.join(IMG_DIR, f"image_{timestamp}.png")
            cv2.imwrite(filename, cv_img)

        elif topic.endswith("camera_info") and not written_camera_info:
            write_camera_info(msg)
            written_camera_info = True

        elif topic == "/odom":
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            odom_file.write(f"{timestamp},{pos.x},{pos.y},{pos.z},{ori.x},{ori.y},{ori.z},{ori.w}\n")

        elif topic == "/cmd_vel":
            lin = msg.linear
            ang = msg.angular
            cmd_file.write(f"{timestamp},{lin.x},{ang.z}\n")

    odom_file.close()
    cmd_file.close()
    print("✅ Extraction complete.")

if __name__ == '__main__':
    main()
