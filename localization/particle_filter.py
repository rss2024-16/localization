from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PoseArray, TransformStamped
from sensor_msgs.msg import LaserScan

from ackermann_msgs.msg import AckermannDriveStamped # Test

import numpy as np
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros
import time

from rclpy.node import Node
import rclpy

import threading

assert rclpy

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        self.declare_parameter('num_particles', "default")
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value

        # For some reason no matter what I do it either tells me this parameter was already declared or is never declared
        self.num_beams_per_particle = 99
        # if self.has_parameter('num_beams_per_particle'):
        #     self.num_beams_per_particle = self.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        # else:
        #     self.declare_parameter('num_beams_per_particle', "default")
        #     self.num_beams_per_particle = self.get_parameter('num_beams_per_particle').get_parameter_value().integer_value

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

        self.weighted_avg = np.array([0.0, 0.0, 0.0])
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
        self.br = tf2_ros.TransformBroadcaster(self)

        self.poses_pub = self.create_publisher(PoseArray, "mcl", 1)

        self.viz_timer = self.create_timer(1, self.timer_cb)
        self.pose_timer = self.create_timer(1.0/20.0, self.pose_cb)

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

        # Test
        self.cmd_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

        self.t1 = float(self.get_clock().now().nanoseconds)/1e9

    def part_to_pose(self, particle):
        '''
        Converts a particle [x, y, theta] to a pose message and returns it
        '''
        pose_msg = Pose()
        pose_msg.position.x = particle[0]
        pose_msg.position.y = particle[1]
        pose_msg.position.z = 0.0
        pose_msg.orientation.x = 0.0
        pose_msg.orientation.y = 0.0
        pose_msg.orientation.z = np.sin(1/2 * particle[2])
        pose_msg.orientation.w = np.cos(1/2 * particle[2])
        return pose_msg

    def part_to_odom(self, particle):
        '''
        Converts a particle [x, y, theta] to an odometry message and returns it
        '''
        odom_msg = Odometry()
        odom_msg.header.frame_id = "/map"
        odom_msg.pose.pose = self.part_to_pose(particle)
        return odom_msg
    
    def laser_callback(self, scan):
        '''
        Whenever you get sensor data use the sensor model to compute the particle probabilities. 
        Then resample the particles based on these probabilities
        '''
        with lock:
            if len(self.particles) > 0 and len(scan.ranges) > 0:
                ranges = scan.ranges
                new_ranges = ranges[: : 11]
                weights = self.sensor_model.evaluate(self.particles, new_ranges)
                # self.get_logger().info(f'{weights}')

                # Update the new drifted average
                weighted = np.average(self.particles[:, :2], axis=0, weights=weights)
                theta_mean = np.arctan2(np.mean(np.sin(self.particles[:, -1])), np.mean(np.cos(self.particles[:, -1])))
                self.weighted_avg = np.array([weighted[0], weighted[1], theta_mean])

                # Resample
                indices = np.arange(self.num_particles)
                indices = np.random.choice(indices, size=self.num_particles, p=weights)
                self.particles = np.array([self.particles[i] for i in indices])

    def odom_callback(self, odom_data):
        '''
        Whenever you get odometry data use the motion model to update the particle positions
        '''
        # self.get_logger().info('odom')
        do = True
        if do:
            with lock:
                if len(self.particles) > 0:
                    # Let the particles drift
                    x = odom_data.twist.twist.linear.x
                    y = odom_data.twist.twist.linear.y
                    # theta = 2*np.arccos(odom_data.pose.pose.orientation.w)
                    # theta = np.arctan2(odom_data.twist.twist.linear.y,odom_data.twist.twist.linear.x)
                    theta = odom_data.twist.twist.angular.z

                    # if self.previous_pose is None:
                    #     self.previous_pose = -np.array([x,y,theta])
                    # else:

                    self.get_logger().info(str(theta))
                    dv = -np.array([x,y,theta])# - self.previous_pose
                    #this might be the first thing we want to check
                    dt = time.time()-self.t1

                    dx = dv*dt
                    # self.get_logger().info(f'-----\nX: {dx[0]}\nY: {dx[1]}\nTheta: {dx[2]}\n-----\n')
                    self.particles = self.motion_model.evaluate(self.particles, dx)

                    # Let the average drift
                    self.weighted_avg = self.motion_model.evaluate_noiseless(self.weighted_avg, dx)
                    # self.previous_pose = np.array([x,y,theta])

                    self.t1 = float(self.get_clock().now().nanoseconds)/1e9

    def pose_callback(self, pose_data):
        '''
        Initializes all of the particles
        '''
        with lock:
            x = pose_data.pose.pose.position.x
            y = pose_data.pose.pose.position.y

            orientation = euler_from_quaternion((
            pose_data.pose.pose.orientation.x,
            pose_data.pose.pose.orientation.y,
            pose_data.pose.pose.orientation.z,
            pose_data.pose.pose.orientation.w))
            theta = orientation[2]

            # self.get_logger().info(f'Theta: {theta}')

            self.weighted_avg = np.array([x, y, theta])

            #self.get_logger().info(f'\n-----\nReal x: {x}\nReal y: {y}\nReal theta: {theta}\n-----')
            # self.get_logger().info(str(pose_data.pose.pose))

            xs = x + np.random.default_rng().uniform(low=-1.0, high=1.0, size=self.num_particles)
            ys = y + np.random.default_rng().uniform(low=-1.0, high=1.0, size=self.num_particles)

            # Wrap the angles between -pi and +pi
            thetas = np.angle(np.exp(1j * (theta + np.random.default_rng().uniform(low=0.0, high=2*np.pi, size=self.num_particles) ) ))
            self.particles = np.array([np.array([x,y,theta]) for x,y,theta in zip(xs,ys,thetas)])

    def timer_cb(self):
        if len(self.particles) > 0:
            poses_msg = PoseArray()
            poses_msg.header.frame_id = "/map"

            poses = []
            for particle in self.particles:
                poses.append(self.part_to_pose(particle))

            poses_msg.poses = poses
            # if self.poses_pub.get_num_connections() > 0:
            self.poses_pub.publish(poses_msg)

            # drive_cmd = AckermannDriveStamped()
            # drive_cmd.drive.speed = -0.5
            # drive_cmd.drive.steering_angle = 0.0
            # self.cmd_pub.publish(drive_cmd)

    def pose_cb(self):
        avg_pose = self.part_to_odom(self.weighted_avg)
        self.odom_pub.publish(avg_pose)

        # Also publish a transform 
        obj = TransformStamped()
        obj.header.stamp = self.get_clock().now().to_msg()
        obj.header.frame_id = "/map"
        obj.child_frame_id = "/base_link"
        obj.transform.translation.x = self.weighted_avg[0]
        obj.transform.translation.y = self.weighted_avg[1]
        obj.transform.translation.z = 0.0

        obj.transform.rotation.x = 0.0
        obj.transform.rotation.y = 0.0
        obj.transform.rotation.z = np.sin(1/2 * self.weighted_avg[2])
        obj.transform.rotation.w = np.cos(1/2 * self.weighted_avg[2])
        self.br.sendTransform(obj)

def main(args=None):
    rclpy.init(args=args)

    global lock

    # Thread lock so sensor and motion do not update list at the same time
    lock = threading.Lock()

    pf = ParticleFilter()

    rclpy.spin(pf)
    rclpy.shutdown()
