from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

import numpy as np

from rclpy.node import Node
import rclpy

import threading
import os

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

        self.previous_pose = None
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

        self.test_pub = self.create_publisher(Odometry, odom_topic, 10)

        self.timer_callback = self.create_timer(10,self.timer_cb)

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

        self.reals = []
        self.x_diff = []
        self.y_diff = []
        self.theta_diff = []
    
    def laser_callback(self, scan):
        '''
        Whenever you get sensor data use the sensor model to compute the particle probabilities. 
        Then resample the particles based on these probabilities
        '''
        with lock:
            if len(self.particles) > 0:
                ranges_sample = np.random.choice(scan.ranges,size=self.num_particles)
                weights = self.sensor_model.evaluate(self.particles, ranges_sample)

                # Publish the new drifted "average"
                self.avg_index = np.argmax(weights)
                avg_pose = Odometry()
                avg = self.particles[self.avg_index]
                avg_pose.pose.pose.position.x = avg[0]
                avg_pose.pose.pose.position.y = avg[1]
                avg_pose.pose.pose.orientation.w = np.cos(1/2 * avg[2])
                self.get_logger().info(f'\n-----\nSensor x: {avg[0]}\nSensor y: {avg[1]}\nSensor theta: {np.cos(avg[2]/2)}\n-----')
                self.x_diff.append(self.reals[0] - avg[0])
                self.y_diff.append(self.reals[1] - avg[1])
                self.theta_diff.append(self.reals[2] - np.cos(1/2 * avg[2]))
                self.odom_pub.publish(avg_pose)
                #do we mean to publish twice?
                # self.odom_pub.publish(avg_pose)

                # Resample
                indices = np.arange(len(self.particles))
                indices = np.random.choice(indices, size=self.num_particles, p=weights)
                # I think this may have bad time complexity, chatGPT says O(nlog(m))
                self.particles = np.array([self.particles[i] for i in indices])
                # self.particles = np.random.choice(self.particles,size=self.num_particles,p=weights)

    def odom_callback(self, odom_data):
        '''
        Whenever you get odometry data use the motion model to update the particle positions
        '''
        with lock:
            if len(self.particles) > 0:
                # Let the particles drift
                x = odom_data.pose.pose.position.x
                y = odom_data.pose.pose.position.y
                theta = 2*np.arccos(odom_data.pose.pose.orientation.w)

                if self.previous_pose is None:
                    self.previous_pose = np.array([x,y,theta])
                else:
                    dx = np.array([x,y,theta]) - self.previous_pose
                    self.particles = self.motion_model.evaluate(self.particles, dx)

                    # Publish the new drifted "average"
                    avg_pose = Odometry()
                    avg = self.particles[self.avg_index]
                    avg_pose.pose.pose.position.x = avg[0]
                    avg_pose.pose.pose.position.y = avg[1]
                    avg_pose.pose.pose.orientation.w = np.cos(1/2 * avg[2])
                    self.get_logger().info(f'\n-----\nMotion x: {avg[0]}\nMotion y: {avg[1]}\nMotion theta: {np.cos(avg[2]/2)}\n-----')
                    self.x_diff.append(self.reals[0] - avg[0])
                    self.y_diff.append(self.reals[1] - avg[1])
                    self.theta_diff.append(self.reals[2] - np.cos(1/2 * avg[2]))
                    self.odom_pub.publish(avg_pose)

    def pose_callback(self, pose_data):
        '''
        Initializes all of the particles
        '''
        with lock:
            x = pose_data.pose.pose.position.x
            y = pose_data.pose.pose.position.y
            theta = 2*np.arccos(pose_data.pose.pose.orientation.w)
            self.reals = [x,y,theta]

            self.get_logger().info(f'\n-----\nReal x: {x}\nReal y: {y}\nReal theta: {theta}\n-----')
            # self.get_logger().info(f'x: {x}\ny: {y}\n theta: {theta}\n')
            xs = x + np.random.default_rng().uniform(low=-1.0,high=1.0,size=self.num_particles)
            ys = y + np.random.default_rng().uniform(low=-1.0,high=1.0,size=self.num_particles)
            #wraps the angles between -pi and +pi
            thetas = np.angle(np.exp(1j * (theta + np.random.default_rng().uniform(low=0.0,high=2*np.pi,size=self.num_particles) ) ))
            self.particles = np.array([np.array([x,y,theta]) for x,y,theta in zip(xs,ys,thetas)])


    def timer_cb(self):
        avg_pose = Odometry()
        avg_pose.pose.pose.position.x = 0.0
        avg_pose.pose.pose.position.y = 0.0
        avg_pose.pose.pose.orientation.w = 0.0
        self.test_pub.publish(avg_pose)
        
        

def main(args=None):
    rclpy.init(args=args)

    global lock
    #thread locking so sensor and motion do not update list at the same time
    lock = threading.Lock()

    pf = ParticleFilter()

    path = os.getcwd()
    try:
        rclpy.spin(pf)
    except KeyboardInterrupt:
        np.save(path+'/x_diff',pf.x_diff)
        np.save(path+'/y_diff',pf.y_diff)
        np.save(path+'/theta_diff',pf.theta_diff)
        np.save(path+'/real',pf.reals)
    rclpy.shutdown()
