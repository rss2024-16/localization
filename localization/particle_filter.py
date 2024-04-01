from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

import numpy as np
import math

from rclpy.node import Node
import rclpy

assert rclpy

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        self.declare_parameter('num_particles', "default")
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        self.particles = []

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.avg_index = 0
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.
    
    def laser_callback(self, scan):
        '''
        Whenever you get sensor data use the sensor model to compute the particle probabilities. 
        Then resample the particles based on these probabilities
        '''
        weights = self.sensor_model.evaluate(self.particles, scan.ranges)

        # Publish the new drifted "average"
        self.avg_index = np.argmax(weights)
        avg_pose = Odometry()
        avg = self.particles[self.avg_index]
        avg_pose.pose.pose.position.x = avg[0]
        avg_pose.pose.pose.position.y = avg[1]
        avg_pose.pose.pose.orientation.w = np.cos(1/2 * avg[2])
        self.odom_pub.publish(avg_pose)
        self.odom_pub.publish(avg_pose)

        # Resample
        indices = np.arange(len(self.particles))
        indices = np.random.choice(indices, size=self.num_particles, p=weights)
        self.particles = [self.particles[i] for i in indices]

    def odom_callback(self, odom_data):
        '''
        Whenever you get odometry data use the motion model to update the particle positions
        '''
        # Let the particles drift
        x = odom_data.pose.pose.position.x
        y = odom_data.pose.pose.position.y
        theta = 2*np.arccos(odom_data.pose.pose.orientation.w)
        self.particles = self.motion_model.evaluate(self.particles, np.array([x, y, theta]))

        # Publish the new drifted "average"
        avg_pose = Odometry()
        avg = self.particles[self.avg_index]
        avg_pose.pose.pose.position.x = avg[0]
        avg_pose.pose.pose.position.y = avg[1]
        avg_pose.pose.pose.orientation.w = np.cos(1/2 * avg[2])
        self.odom_pub.publish(avg_pose)

    def pose_callback(self, pose_data):
        '''
        Initializes all of the particles
        '''
        # self.get_logger().info(str(pose_data.pose.))
        x = pose_data.pose.pose.position.x
        y = pose_data.pose.pose.position.y
        theta = 2*np.arccos(pose_data.pose.pose.orientation.w)
        self.get_logger().info(f'Real x: {x}\nReal y: {y}\nReal theta: {theta}')
        # self.get_logger().info(f'x: {x}\ny: {y}\n theta: {theta}\n')
        xs = x + np.random.default_rng().uniform(low=-1.0,high=1.0,size=self.num_particles)
        ys = y + np.random.default_rng().uniform(low=-1.0,high=1.0,size=self.num_particles)
        #wraps the angles to 2*pi
        thetas = np.angle(np.exp(1j * (theta + np.random.default_rng().uniform(low=0.0,high=2*np.pi,size=self.num_particles) ) ))
        self.particles = [np.array([x,y,theta]) for x,y,theta in zip(xs,ys,thetas)]
        
        

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
