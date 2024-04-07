from sensor_msgs.msg import LaserScan

import numpy as np

from rclpy.node import Node
import rclpy

assert rclpy

class LaserTransform(Node):

    def __init__(self):
        super().__init__("laser_transform")

        self.laser_sub = self.create_subscription(LaserScan, "/scan",
                                                  self.laser_callback,
                                                  10)
        
        self.transformed_pub = self.create_publisher(LaserScan, "/map_scan", 10)
    
    def laser_callback(self, scan):
        '''
        Whenever you get sensor data use the sensor model to compute the particle probabilities. 
        Then resample the particles based on these probabilities
        '''
        transformed = scan
        transformed.header.frame_id = "base_link"
        self.transformed_pub.publish(transformed)

        
def main(args=None):
    rclpy.init(args=args)

    lt = LaserTransform()

    rclpy.spin(lt)
    rclpy.shutdown()
