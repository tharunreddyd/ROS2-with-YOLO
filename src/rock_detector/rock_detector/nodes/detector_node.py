import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')
        
        # Load YOLO model correctly
        self.model = YOLO('/home/roopi66/mars_rocks/train/runs/detect/train/weights/best.pt')
        
        # Create publishers
        self.detection_pub = self.create_publisher(Image, 'detections', 10)
        self.position_pub = self.create_publisher(Float32MultiArray, 'rock_positions', 10)
        
        # Subscribe to camera node
        self.subscription = self.create_subscription(
            Image,
            'video_raw',
            self.detection_callback,
            10)
        
        self.bridge = CvBridge()
        
        # Create window for visualization
        cv2.namedWindow('Rock Detection', cv2.WINDOW_NORMAL)
        
        # Create video writer for saving
        self.out = None  # Will be initialized with first frame
        
        self.get_logger().info('Detector node initialized')

    def detection_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Initialize video writer if not done yet
            if self.out is None:
                height, width = cv_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter('rock_detection_output.mp4', fourcc, 30.0, (width, height))
            
            # Run YOLO detection
            results = self.model(cv_image)[0]
            
            # Process detections
            positions = []
            annotated_frame = cv_image.copy()
            
            if results.boxes is not None:
                for box in results.boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf)
                    
                    # Calculate center
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Store position and confidence
                    positions.extend([center_x, center_y, confidence])
                    
                    # Draw detection
                    cv2.rectangle(annotated_frame, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                (0, 255, 0), 2)
                    cv2.putText(annotated_frame, 
                              f'Rock {confidence:.2f}', 
                              (int(x1), int(y1)-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Rock Detection', annotated_frame)
            cv2.waitKey(1)
            
            # Save frame
            self.out.write(annotated_frame)
            
            # Publish annotated image
            detection_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
            self.detection_pub.publish(detection_msg)
            
            # Publish rock positions
            pos_msg = Float32MultiArray(data=positions)
            self.position_pub.publish(pos_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error in detection_callback: {str(e)}')

    def __del__(self):
        # Clean up
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
