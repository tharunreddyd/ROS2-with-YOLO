# rock_detector/launch/rock_detector.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Start turtlesim
        Node(
            package='turtlesim',
            executable='turtlesim_node',
            name='turtlesim'
        ),
        # Start camera node
        Node(
            package='rock_detector',
            executable='camera_node',
            name='camera_node'
        ),
        # Start detector node
        Node(
            package='rock_detector',
            executable='detector_node',
            name='detector_node'
        ),
        # Start turtle controller
        Node(
            package='rock_detector',
            executable='turtle_controller',
            name='turtle_controller'
        )
    ])

