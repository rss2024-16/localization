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
    
    def sample_normal_distribution(val):
        return val/6 * sum([random.random() for i in range(1,13)])

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
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
        
        rand_mod = lambda x: -1 + 2*random.random() #modifies random.random() to return in interval [-1,1]
        
        # normal_sample = lambda val: val/6 * sum([rand_mod(None) for i in range(1,13)])
        # normal_sample = np.random.normal

        a1 = 1
        a2 = 1
        a3 = 1

        particles_updated = []

        for particle in particles:
            x,y,theta = particle

            xp = np.random.normal(x)
            yp = np.random.normal(y)
            thetap = np.random.normal(theta)
            
            new_particle = [xp,yp,thetap]

            future_particles = new_particle + transform(new_particle[-1]) * odometry
            # xprime = a1*np.random.normal(future_particles[0])
            # yprime = a2*np.random.normal(future_particles[1])
            # thetaprime = a3*np.random.normal(future_particles[2])

            particles_updated.append(future_particles)

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



        return particles_updated
