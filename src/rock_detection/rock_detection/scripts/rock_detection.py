#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import os

class RockDetectionNode(Node):
    def __init__(self):
        super().__init__('rock_detection_node')
        
        # Initialize YOLO model
        model_path = os.path.expanduser('~/mars_rocks_project/data.yaml')
        self.model = YOLO('yolov8n.pt')
        self.model.train(data=model_path, epochs=1)  # Train on Mars rocks dataset
        self.bridge = CvBridge()
        
        # Create QoS profile
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to camera feed
        self.camera_subscription = self.create_subscription(
            Image,
            '/video_frames',
            self.camera_callback,
            self.qos_profile
        )
        
        # Publisher for processed images
        self.detection_publisher = self.create_publisher(
            Image,
            '/rock_detections',
            10
        )
        
        # Publisher for turtle control
        self.turtle_publisher = self.create_publisher(
            Twist,
            '/turtle1/cmd_vel',
            10
        )
        
        self.get_logger().info('Rock Detection Node has been initialized')

    def camera_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Run YOLOv8 detection
            results = self.model(cv_image)
            
            # Process detections
            largest_rock = None
            max_area = 0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    area = (x2 - x1) * (y2 - y1)
                    
                    if area > max_area:
                        max_area = area
                        largest_rock = (x1, y1, x2, y2)
                    
                    # Draw detection box
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'Rock {confidence:.2f}'
                    cv2.putText(cv_image, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Move turtle towards largest rock
            if largest_rock is not None:
                self.move_turtle_towards_rock(largest_rock, cv_image.shape[1])
            
            # Display image
            cv2.imshow('Rock Detection', cv_image)
            cv2.waitKey(1)
            
            # Publish processed image
            processed_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.detection_publisher.publish(processed_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in camera callback: {str(e)}')

    def move_turtle_towards_rock(self, rock_coords, image_width):
        x1, _, x2, _ = rock_coords
        rock_center_x = (x1 + x2) / 2
        
        # Calculate relative position (-1 to 1)
        relative_position = (rock_center_x - image_width/2) / (image_width/2)
        
        # Create and publish turtle movement command
        twist = Twist()
        twist.linear.x = 0.5  # Forward velocity
        twist.angular.z = -relative_position  # Turn based on rock position
        self.turtle_publisher.publish(twist)

class VideoPlayerNode(Node):
    def __init__(self):
        super().__init__('video_player_node')
        
        # Create publisher
        self.publisher = self.create_publisher(
            Image,
            '/video_frames',
            10
        )
        
        # Path to your Mars rocks dataset train images
        self.image_dir = os.path.expanduser('~/mars_rocks_project/train/images')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.current_image_index = 0
        
        # Timer for publishing frames
        self.timer = self.create_timer(1.0, self.timer_callback)  # 1 second per image
        
        self.bridge = CvBridge()
        self.get_logger().info('Video Player Node has been initialized')

    def timer_callback(self):
        if self.image_files:
            # Get current image path
            image_path = os.path.join(self.image_dir, self.image_files[self.current_image_index])
            
            # Read image
            frame = cv2.imread(image_path)
            if frame is not None:
                # Convert and publish
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.publisher.publish(msg)
                
                # Move to next image
                self.current_image_index = (self.current_image_index + 1) % len(self.image_files)

def main():
    rclpy.init()
    
    video_player = VideoPlayerNode()
    rock_detector = RockDetectionNode()
    
    try:
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(video_player)
        executor.add_node(rock_detector)
        executor.spin()
    finally:
        cv2.destroyAllWindows()  # Clean up OpenCV windows
        video_player.destroy_node()
        rock_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
