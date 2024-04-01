import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', "default")
        node.declare_parameter('scan_theta_discretization', "default")
        node.declare_parameter('scan_field_of_view', "default")
        node.declare_parameter('lidar_scale_to_map_scale', 1)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters
        self.alpha_hit = 0
        self.alpha_short = 0
        self.alpha_max = 0
        self.alpha_rand = 0
        self.sigma_hit = 0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """

        zmax = self.table_width

        phits = np.empty((self.table_width, self.table_width))
        pothers = np.empty((self.table_width, self.table_width))

        #iterating left to right across columns
        for d in range(self.table_width): #cols
            for zk in range(self.table_width): #rows

                phit = 1/zmax * ( 1/np.sqrt(2*np.pi*self.sigma_hit**2) * np.exp(-(zk-d)**2/(2*self.sigma_hit**2)))
                
                pshort = 2/d * (1-zk/d) if zk <= d and d!= 0 else 0
                pmax = 1 if zk == (zmax-1) else 0
                prand = 1/zmax

                phits[zk][d] = phit
                pothers[zk][d] = self.alpha_short*pshort + self.alpha_rand*prand + self.alpha_max*pmax

        
        for row in range(phits.shape[0]):
            #normalize across increasing d values to sum phits to 1
            #across columns
            phits[row,:] = phits[row,:] / sum(phits[row,:])

        for d in range(self.table_width):
            for zk in range(self.table_width):
                #calculate total probabilities and create table
                ptotal = self.alpha_hit*phits[zk][d] + pothers[zk][d]
                self.sensor_model_table[zk][d] = ptotal

        for col in range(self.sensor_model_table.shape[1]):
            #normalize columns to sum probabilities to 1
            #columns represent a singular d value i.e. column 0 : d=0 column 1: d=1 etc
            self.sensor_model_table[:,col] = self.sensor_model_table[:,col] / sum(self.sensor_model_table[:,col])

        save = False
        if save:
            np.save('precomputed_table',self.sensor_model_table)
            np.save('phits',phits)
            np.save('pothers',pothers)


    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        scans = self.scan_sim.scan(particles) #this is an N by num_beams_per_particle matrix
        #each row of the matrix is num_beams_per_particle row of lidar scans from the map

        step = self.resolution * self.lidar_scale_to_map_scale
        scans = scans/ step
        observation = observation / step

        zmax = (self.table_width-1)*step
        scans = np.clip(scans,0,zmax)
        observation = np.clip(observation,0,zmax) 

        #TODO downsample the observations irl (sim is fine though)


        probabilities = []
        for particle_scan in scans:
            probability = self.sensor_model_table[particle_scan, observation] #should be a vector
            probabilities.append(np.product(probability))
        
        probabilities/=sum(probabilities)
        #matrix of probabilities

        #multiple across the rows
        #and then normalize

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")o