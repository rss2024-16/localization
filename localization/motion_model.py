import numpy as np

class MotionModel:

    def __init__(self, node):
        """
        idk what node is 
        """
        self.deterministic = False

        self.transform = lambda theta: np.array([ [np.cos(theta), -np.sin(theta), 0],
                                             [np.sin(theta), np.cos(theta), 0],
                                             [0, 0, 1]
                                            ])

        self.n = node

    def evaluate_noiseless(self, particle, odometry, prev_time):
        """
        Update the particles to reflect probable 
        future states given the odometry data.

        Unlike evaluate, does not incorporate any noise, 
        can be interpreted as ground truth odometry.
        """
        # particle is 1x3
        # odometry is 3x1
        # future_particle should be 3x1
        dt = float(self.n.get_clock().now().nanoseconds)/1e9 - prev_time
        dx = odometry * dt
        future_particle = particle.T + self.transform(particle[-1]) @ dx.T
        future_particle = future_particle.T

        return np.array(future_particle)


    def evaluate(self, particles, odometry, prev_time):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y1 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """
        particles_updated = []

        for particle in particles:


            # Standard deviation for the random noise, in meters (?)

            # particle is 1x3
            # odometry is 3x1
            # future_particle should be 3x1
            dt = float(self.n.get_clock().now().nanoseconds)/1e9 - prev_time
            dx = odometry * dt
            future_particle = particle.T + self.transform(particle[-1]) @ dx.T
            future_particle = future_particle.T

            # self.deterministic = True
            if not self.deterministic:
                # Standard deviation for the random noise, in meters (?)
                linear_noise = .05
                angular_noise = .05

                x_eps = np.random.normal(scale=linear_noise)
                y_eps = np.random.normal(scale=linear_noise)
                theta_eps = np.random.normal(scale=angular_noise)

                future_particle += np.array([x_eps, y_eps, theta_eps])

            particles_updated.append(future_particle)

        return np.array(particles_updated)
