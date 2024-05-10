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
        self.declare_parameter('num_particles', "default")
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value   

        self.get_logger().info("Localization %s" % odom_topic)
        self.get_logger().info("Localization %s" % scan_topic)   
        self.get_logger().info(f"Localization num particles {self.num_particles}")  

        # Class variables
        self.weights = np.ones(int(self.num_particles)) / self.num_particles  # start with uniform weight for each particle
        self.particles = np.zeros((int(self.num_particles), 3))
        self.initialized = False
        self.lock = threading.Lock()

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        # Subscribers and publishers
        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  5)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 5)

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        self.particles_pub = self.create_publisher(PoseArray, "/debug", 1)

        # Fixed odom publishing
        odom_rate = 25.0 # Hz
        self.pose_timer = self.create_timer(1.0/odom_rate, self.estimate_odom)

        self.previous_pose = None
        self.prev_time = self.T0 = self.prev_log_time = self.get_clock().now()

        self.get_logger().info("=============+READY+=============")

    def estimate_odom(self):
        """
        Uses self.particles to get pose prediction. Puts prediction in Odometry message.

        If debug, then also publish the particles as PoseArray msg. 
        """
        if not self.initialized:
            return

        now = self.get_clock().now()

        x_avg, y_avg, theta_avg = self.previous_pose

        # Create the message
        msg = Odometry()
        msg.pose.pose.position.x, msg.pose.pose.position.y = x_avg, y_avg
        msg.pose.pose.orientation.x = np.cos(theta_avg / 2)
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = np.sin(theta_avg / 2)
        msg.header.frame_id = '/map'
        msg.header.stamp = now.to_msg()
        self.odom_pub.publish(msg)

        # Debug code, comment this out if running on bot
        array = PoseArray()
        array.header.frame_id = '/map'
        array.poses = []
        for particle in self.particles:
            part_pose = Pose()
            part_pose.position.x, part_pose.position.y = particle[:2]
            part_pose.orientation.x, part_pose.orientation.y, part_pose.orientation.z, part_pose.orientation.w = quaternion_from_euler(
                0, 0, particle[2])
            array.poses.append(part_pose)

        self.particles_pub.publish(array)

        return msg

    def laser_callback(self, scan):
        """
        update + resample
        """
        if not self.sensor_model.map_set:
            return

        now = self.get_clock().now()
        self.lock.acquire()

        # Particle weights from current scan
        unnormed_weights = self.sensor_model.evaluate(self.particles, scan.ranges)  # P(z|x) 
        best_p = max(unnormed_weights)
        self.weights = unnormed_weights / np.sum(unnormed_weights)

        # Resampling
        idxs = np.random.choice(int(self.num_particles), int(self.num_particles), p=self.weights)
        self.particles = self.particles[idxs]

        # Average the pose
        x_avg, y_avg = np.mean(self.particles[:, :2], axis=0)
        theta_avg = np.arctan2(np.mean(np.sin(self.particles[:, -1])), np.mean(np.cos(self.particles[:, -1])))
        self.previous_pose = (x_avg, y_avg, theta_avg)

        self.prev_time = now
        self.lock.release()

    def odom_callback(self, odom):
        """
        Prediction Step
        """
        if not self.sensor_model.map_set:
            return

        self.lock.acquire()
        now = self.get_clock().now()

        dt = (now - self.prev_time).nanoseconds / 1e9

        # Calculate our change in pose
        v = [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.angular.z]
        dx, dy, dtheta = v[0] * dt, v[1] * dt, v[2] * dt
        delta_x = [dx, dy, dtheta]

        # Let the particles drift
        self.particles = self.motion_model.evaluate(self.particles, delta_x, v)

        # Average the pose
        x_avg, y_avg = np.mean(self.particles[:, :2], axis=0)
        theta_avg = np.arctan2(np.mean(np.sin(self.particles[:, -1])), np.mean(np.cos(self.particles[:, -1])))
        self.previous_pose = (x_avg, y_avg, theta_avg)

        self.prev_time = now
        self.lock.release()

    def pose_callback(self, pose):
        """
        This is done whenever the green arrow is placed down in RVIZ. 

        pose: guess from YOU (?)

        Sample around the pose (x+eps_x, y+eps_y, theta+eps_theta) eps_? ~  N(0,sigma)
        """
        if not self.sensor_model.map_set:
            return

        self.lock.acquire()

        std_trans = 1.
        std_theta = np.pi / 4

        # Unpack the pose estimate
        x, y = pose.pose.pose.position.x, pose.pose.pose.position.y
        theta = euler_from_quaternion(
            [pose.pose.pose.orientation.x, pose.pose.pose.orientation.y, pose.pose.pose.orientation.z,
             pose.pose.pose.orientation.w])[-1]

        # Random generation of particles around ours
        x_samples = (std_trans * np.random.randn(int(self.num_particles)) + x)[:, None]
        y_samples = (std_trans * np.random.randn(int(self.num_particles)) + y)[:, None]
        theta_samples = (std_theta * np.random.randn(int(self.num_particles)) + theta)[:, None]

        # Organize the samples
        self.particles = np.hstack((x_samples, y_samples, theta_samples))

        # Average the pose
        x_avg, y_avg = np.mean(self.particles[:, :2], axis=0)
        theta_avg = np.arctan2(np.mean(np.sin(self.particles[:, -1])), np.mean(np.cos(self.particles[:, -1])))
        self.previous_pose = (x_avg, y_avg, theta_avg)

        self.initialized = True
        self.lock.release()

def main(args=None):
    rclpy.init(args=args)

    global lock

    # Thread lock so sensor and motion do not update list at the same time
    lock = threading.Lock()

    pf = ParticleFilter()
    try:
        rclpy.spin(pf)
    except KeyboardInterrupt:
        np.save('sensortimes',pf.sensor_times)
        np.save('motiontimes',pf.motion_times)
    rclpy.shutdown()
