# rock_detector/nodes/turtle_controller.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Twist

class TurtleController(Node):
    def __init__(self):
        super().__init__('turtle_controller')
        
        # Subscribe to rock positions
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'rock_positions',
            self.position_callback,
            10)
            
        # Publish turtle commands
        self.publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.get_logger().info('Turtle controller initialized')

    def position_callback(self, msg):
        if not msg.data:
            return
            
        # Find largest rock (highest confidence)
        positions = msg.data
        max_conf = 0
        target_x = 0
        
        # Each position is [x, y, confidence]
        for i in range(0, len(positions), 3):
            x, y, conf = positions[i:i+3]
            if conf > max_conf:
                max_conf = conf
                target_x = x
        
        # Create movement command
        twist = Twist()
        
        # Move forward at constant speed
        twist.linear.x = 1.0
        
        # Turn based on rock position
        image_center = 320  # Half of 640
        error = target_x - image_center
        twist.angular.z = -0.005 * error
        
        self.publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = TurtleController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
