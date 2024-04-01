import numpy as np

import random
import math

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        pass

        ####################################

    def evaluate(self, particles, odometry):
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

        transform = lambda theta: np.array([ [np.cos(theta), -np.sin(theta), 0],
                                             [np.sin(theta), np.cos(theta), 0],
                                             [0, 0, 1]
                                            ])

        particles_updated = []

        for particle in particles:
            x_eps = np.random.normal()
            y_eps = np.random.normal()
            theta_eps = np.random.normal()

            # particle is 1x3
            # odometry is 3x1
            # future_particle should be 3x1
            future_particle = particle.T + transform(particle[-1]) @ odometry.T
            future_particle = future_particle.T

            future_particle += np.array([x_eps, y_eps, theta_eps])

            particles_updated.append(future_particle)

            ###################### BOOK ALGORITHM ********************************
            # drot1 = math.atan2(dy,dx) - theta #change in rotation on car hemisphere
            # dtrans = ( dx**2 + dy**2 ) ** (1/2) #change in translation
            # drot2 = dtheta - drot1 #change in rotation across from car

            # drot1_hat = drot1 - normal_sample(a1*drot1 + a2*dtrans)
            # dtrans_hat = dtrans - normal_sample(a3*dtrans + a4* (drot1 + drot2))
            # drot2_hat = drot2 - normal_sample(a1*drot2 + a2*dtrans)

            # xprime = x + dtrans_hat * np.cos(theta + drot1_hat)
            # yprime = y + dtrans_hat * np.sin(theta + drot1_hat)
            # thetaprime = theta + drot1_hat + drot2_hat

            # xt = np.array([xprime,yprime,thetaprime])
            # particles_updated.append(xt)
            #**************************************************************************



        return np.array(particles_updated)
