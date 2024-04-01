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
    
    def laser_callback(self,scan):
        '''
        Whenever you get sensor data use the sensor model to compute the particle probabilities. 
        Then resample the particles based on these probabilities
        '''
        if len(self.particles) != 0:
            sample_size = 100

            #sample by closest ranges
            # ranges = sorted(enumerate(scan.ranges), key = lambda x: x[1])[:sample_size]
            #sample randomly
            observation = np.random.choice(scan.ranges,size=sample_size)

            weights = self.sensor_model.evaluate(self.particles,observation)
            if weights is not None:

                # self.get_logger().info(f'{probabilities}')

                PROB_THRESHOLD = 0.1

                resampled_particles = np.random.choice(self.particles,p=weights)

                # resampled_particles = np.array([self.particles[x] for x in range(len(normalized_probs))\
                #                                 if normalized_probs[x] > PROB_THRESHOLD])
                
                average_x = np.mean([x[0] for x in resampled_particles])
                average_y = np.mean([x[1] for x in resampled_particles])
                average_theta = math.atan2(sum([np.sin(x[2]) for x in resampled_particles]),\
                                           sum([np.cos(x[2]) for x in resampled_particles]))
                

                self.get_logger().info(f'Sampled x: {average_x}\nSampled y: {average_y}\nSample theta: {average_theta}')



    def odom_callback(self,odom_data):
        '''
        Whenever you get odometry data use the motion model to update the particle positions
        '''
        # print(odom_data.pose)
        dx = odom_data.pose
        # updated_particles = self.motion_model.evaluate(self.particles,dx)


    def pose_callback(self,pose_data):
        '''
        Initializes all of the particles
        '''
        # self.get_logger().info(str(pose_data.pose.))
        x = pose_data.pose.pose.position.x
        y = pose_data.pose.pose.position.y
        theta = 2*np.arccos(pose_data.pose.pose.orientation.w)
        self.get_logger().info(f'Real x: {x}\nReal y: {y}\nReal theta: {theta}')
        # self.get_logger().info(f'x: {x}\ny: {y}\n theta: {theta}\n')
        xs = x + np.random.default_rng().uniform(low=-1.0,high=1.0,size=200)
        ys = y + np.random.default_rng().uniform(low=-1.0,high=1.0,size=200)
        #wraps the angles to 2*pi
        thetas = np.angle(np.exp(1j * (theta + np.random.default_rng().uniform(low=0.0,high=2*np.pi,size=200) ) ))
        self.particles = [(x,y,theta) for x,y,theta in zip(xs,ys,thetas)]
        
        

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
