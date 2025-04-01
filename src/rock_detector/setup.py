from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'rock_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Rock detection package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_node = rock_detector.nodes.camera_node:main',
            'detector_node = rock_detector.nodes.detector_node:main',
            'turtle_controller = rock_detector.nodes.turtle_controller:main',
        ],
    },
)
