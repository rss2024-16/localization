import numpy as np

class MotionModel:

    def __init__(self, node):
        self.transform = lambda theta: np.array([ [np.cos(theta), -np.sin(theta), 0],
                                             [np.sin(theta), np.cos(theta), 0],
                                             [0, 0, 1]
                                            ])

        self.node = node

    def evaluate_noiseless(self, particle, odometry):
        # particle is 1x3
        # odometry is 3x1
        # future_particle should be 3x1
        future_particle = particle.T + self.transform(particle[-1]) @ odometry.T
        future_particle = future_particle.T

        return np.array(future_particle)


    def evaluate(self, particles: np.array, dx, prev_t):
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

            # particle is 1x3
            # odometry is 3x1
            # future_particle should be 3x1
            dt = (self.node.get_clock().now().nanoseconds - prev_t) * 1e-9
            odometry = dx * dt
            future_particle = particle.T + self.transform(particle[-1]) @ odometry.T
            future_particle = future_particle.T

            # Standard deviation for the random noise, in meters (?)
            std = 0.1

            x_eps = np.random.normal(scale=std)
            y_eps = np.random.normal(scale=std)
            theta_eps = np.random.normal(scale=std)

            #future_particle += np.array([x_eps, y_eps, theta_eps])

            particles_updated.append(future_particle)

        return np.array(particles_updated)
    