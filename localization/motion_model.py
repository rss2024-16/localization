import numpy as np

class MotionModel:

    def __init__(self, node):
        self.trans_std = .05 * 2
        self.rot_std = np.pi / 30 * 2
        self.k_vx = .025
        self.k_vtheta = np.pi / 120

    def evaluate_noiseless(self, pose, odometry):
        """
        Updates a single pose based on odometry data,
        does not add any noise to the motion.

        args:
            pose: A 3-vector [x y theta] 

            odometry: A 3-vector [dx dy dtheta]

        returns:
            pose: an updated position based on the odometry
        """
        dx, dy, dtheta = odometry

        # Transformation matrix calculations
        cos_theta = np.cos(pose[2])
        sin_theta = np.sin(pose[2])

        # Compute the transformation for dx, dy considering the new orientation
        dx_prime = cos_theta * dx - sin_theta * dy
        dy_prime = sin_theta * dx + cos_theta * dy

        # Update particle positions
        pose[0] += dx_prime
        pose[1] += dy_prime
        pose[2] -= dtheta

        return pose

    def evaluate(self, particles, odometry, velocity):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y1 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]
            
            velocity: A 3-vector [vx vy vtheta]

        returns:
            particles: An updated matrix of the
                same size
        """
        n_particles = particles.shape[0]
        dx, dy, dtheta = odometry
        vx, vy, vtheta = velocity

        # Generate noise for all particles at once
        x_noise = (self.trans_std + self.k_vx * vx) * np.random.randn(n_particles)
        y_noise = (self.trans_std) * np.random.randn(n_particles)
        theta_noise = (self.rot_std + self.k_vtheta * vtheta) * np.random.randn(n_particles)

        # Transformation matrix calculations
        cos_theta = np.cos(particles[:, 2])
        sin_theta = np.sin(particles[:, 2])

        # Compute the transformation for dx, dy considering the new orientation
        dx_prime = cos_theta * (dx + x_noise) - sin_theta * (dy + y_noise)
        dy_prime = sin_theta * (dx + x_noise) + cos_theta * (dy + y_noise)

        # Update particle positions
        particles[:, 0] += dx_prime
        particles[:, 1] += dy_prime
        particles[:, 2] = particles[:, 2] - dtheta + theta_noise

        return particles
