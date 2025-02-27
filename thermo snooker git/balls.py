
"""
Created on Fri Nov 25 18:28:49 2022

@author: sophia
"""

import numpy as np
import pylab as pl

import matplotlib.pyplot as plt
bohr = 5.29e-11


r = 1000*bohr  # largerst radius used to set R_c, radius of container
w = 200
R_c = w*r


class Ball:
    '''
    2D ball object with m-mass, R-radius,p-position vector, v-velocity
    '''

    def __init__(self, m=0, R=0, p=[0, 0], v=[0, 0], ball=True):
        self.R = R
        self._p = np.asarray(p)
        self.p = []
        self._v = np.asarray(v)

        self.ball = ball
        if ball:
            self.m = m

            self.patch = pl.Circle(
                self._p, self.R, ec='b', fill=True, ls='solid')
        else:
            self.m = 1e40
            self.patch = pl.Circle([0, 0], R_c, ec='r', fill=False, ls='solid')

    def pos(self):
        return self._p

    def vel(self):
        return self._v

    def fix(r):
        r = np.fix(100000000*r)/100000000
        # self._v=np.fix(10000*self._v)/10000
        return r
# can be used to avoid floating point errors inpython by adding fix to all values thus only certain numbers of sf is considered.

    def move(self, dt):
        self._p = self._p+self._v*dt
        self.patch.center = self._p

        return self._p

    def time_to_collision(self, other):  # ball ball might be different
        R1 = self.R-other.R
        R2 = other.R+self.R
        if self.ball == True and other.ball == True:
            R = R2
        else:
            R = R1

        v = self._v-other._v
        p = self._p-other._p
        vp = np.dot(v, p)
        vv = np.dot(v, v)
        pp = np.dot(p, p)

        dic = (vp**2)-vv*(pp-R**2)
        if dic < 0:
            # print('dic')

            return 1000000
        else:

            delt1 = (-vp+np.sqrt(dic))/vv
            delt2 = (-vp-np.sqrt(dic))/vv  # solving qudratic equation
            if self.ball == True and other.ball == True:
                if delt2 >= 0:
                    return delt2  # only delt2 can be choose for ball balll collision, due to the sides involved in collision
                else:
                    # print('delt2') used to check if delt 2 or delt 1 is used as expected.

                    return 1000000
            else:
                if delt1 >= 0:

                    return delt1  # only delt1 can be choose for ball container collision, due to the sides involved in collision

                else:
                    # print('delt1')
                    return 1000000  # possibility of collide=0 moving away , make time almost infinite large

    def collide(self, other):
        """
        Finds the new velocity of two balls after a collision,using Momentum 
        and Energy Conservation
        """
        m1 = self.m
        m2 = other.m
        v1 = np.array(self._v)
        v2 = np.array(other._v)
        x1 = self._p
        x2 = other._p
        #print(x1, x2)
        v1_new = v1 - ((2*m2) * np.dot(v1-v2, x1-x2) * (x1-x2)
                       ) / ((m1+m2) * (x1-x2).dot(x1-x2))
        v2_new = v2 - ((2*m1) * np.dot(v2-v1, x2-x1) * (x2-x1)
                       ) / ((m1+m2) * (x1-x2).dot(x1-x2))
        # print('v1,v2')
        #print(v1_new, v2_new)
        self._v = v1_new
        other._v = v2_new

        return self

    def get_patch(self):

        self.patch.center = self._p
        return self.patch

    def KE(self):
        m = self.m

        KE = 0.5*m*(self._v[0]**2+self._v[1]**2)

        return KE

    # unit newton/meter 2piR assume contianer thick=2 radius of the balls

    def distance_balls(self, other):
        a = self._p-other._p
        dis_b = np.sqrt(a[0]**2+a[1]**2)
        return dis_b
# return the inter particle separation

    def distance_central(self):
        a = self._p
        dis_c = np.sqrt(a[0]**2+a[1]**2)
        return dis_c

# return the distance from central

    def change_in_momentum_C(self):
        vper = np.dot(self._v, self._p)/(R_c-r)
# return the change in momentum in magnitude when a ball collide withthe container
        return 2*self.m*abs(vper)

    def momentum(self):
        # return momentum of ball object
        self.p = self._v*self.m
        return self.p
