import numpy as np

class MotionModel:

    def __init__(self, node):
        self.trans_std = .01 * 2
        self.rot_std = np.pi / 30 * 2
        self.k_vx = .02
        self.k_vy = .005
        self.k_vtheta = np.pi / 60

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
        y_noise = (self.trans_std + self.k_vy * vy) * np.random.randn(n_particles)
        theta_noise = (self.rot_std + self.k_vtheta * vtheta) * np.random.randn(n_particles)

        # Update particle orientations
        new_thetas = particles[:, 2] + dtheta + theta_noise
        cos_theta = np.cos(new_thetas)
        sin_theta = np.sin(new_thetas)

        # Compute the transformation for dx, dy considering the new orientation
        dx_prime = cos_theta * dx - sin_theta * dy + x_noise
        dy_prime = sin_theta * dx + cos_theta * dy + y_noise

        # Update particle positions
        particles[:, 0] += dx_prime
        particles[:, 1] += dy_prime
        particles[:, 2] = new_thetas  # Update thetas

        return particles
