# rock_detector/nodes/camera_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher = self.create_publisher(Image, 'video_raw', 10)
        self.bridge = CvBridge()
        
        # Use your video file path
        self.video_source = '/home/roopi66/mars_rocks/synthetic_video.mp4'
        self.cap = cv2.VideoCapture(self.video_source)
        
        # Create timer for publishing frames
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('Camera node initialized')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to ROS message
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.publisher.publish(msg)
        else:
            # Reset video when it ends
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
