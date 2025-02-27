# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 18:22:41 2022

@author: yz6621
"""
import numpy as np
import pylab as pl
ttot = []

delv_1 = []
delv_2 = []


class Ball:
    def __init__(self, m=0, R=0, p=[0, 0], v=[0, 0], ball=True):
        self.R = R
        self._p = np.asarray(p)
        self._v = np.asarray(v)

        self.ball = ball
        if ball:
            self.m = m

            self.patch = pl.Circle(
                self._p, self.R, ec='b', fill=True, ls='solid')
        else:
            self.m = 10000000000
            self.patch = pl.Circle([0, 0], 10, ec='r', fill=False, ls='solid')

    def pos(self):
        return self._p

    def vel(self):
        return self._v

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
        # print(vv,pp,vp)
        # if self.container==False and other.container==False:#2 balls

        dic = (vp**2)-vv*(pp-R**2)

        if dic < 0:
            print('dic')
            return None

        delt1 = (-vp+np.sqrt(dic))/vv
        delt2 = (-vp-np.sqrt(dic))/vv
        if delt1 < 0:
            print('delt1')
            return None  # possibility of collide=0 moving away
        if delt1 > 0 and delt2 > 0:
            return delt2
        else:
            return delt1

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

        v1_new = v1 - ((2*m2) * np.dot(v1-v2, x1-x2) * (x1-x2)) / (
            (m1+m2) * (x1-x2).dot(x1-x2))
        v2_new = v2 - ((2*m1) * np.dot(v2-v1, x2-x1) * (x2-x1)) / (
            (m1+m2) * (x1-x2).dot(x1-x2))

        self._v = v1_new
        other._v = v2_new
        delv1 = abs(v1_new-v1)
        delv2 = abs(v2_new-v2)
        delv_1.append(delv1)
        delv_2.append(delv2)
        return self

    def get_patch(self):

        self.patch.center = self._p
        return self.patch

    def KE(self):
        KE = 0.5*self.m*self._v**2

        return KE

    def p(m, v):
        p = m*v
        return p

# %%


class Simulation:
    '''
    parameters
    m:
    t:
        R_c
        R_b
    '''

    def __init__(self, pb=[0, 0], R_c=10, R_b=1, v=[1, 0]):
        self.R_c = R_c
        self._container = Ball(
            p=[0, 0], v=[0, 0], R=self.R_c, ball=False, m=100000)
        self.vb = v
        self._ball = Ball(p=pb, v=self.vb, R=1, m=1)

        Ball.__init__(self)

    def KE(m, v):
        KE = 0.5*m*v**2
        return KE

    def next_collision(self, other):

        #        u1=self._v
        #        u2=other._v
        #        p1=self._p+u1*self.time_to_collision(other)
        #        p2=other._p+u2*self.time_to_collision(other)
        #        self._p=p1
        #        other._p=p2
        #        t=self.time_to_collision(other)
        #        self.move(t)
        #        other.move(t)
        #        #self.patch.center = self._p
        #        #other.patch.center= other._p
        #        self.collide(other)
        print(self._ball.KE)
        t = self._ball.time_to_collision(self._container)
        self._ball.move(t)
        self._container.move(t)
        self._ball.collide(self._container)
        ttot.append(t)
        KE1 = self._ball.KE(self._ball.m, self._ball._v)
        print(KE1)
        return self and ttot

    def pressure(self):

        pchange1 = self.m*(np.sum(del_v1))
        P = pchange1/np.sum(ttot)
        return P

    def run(self, num_frames, animate=False):
        if animate:
            f = pl.figure()
            ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
            ax.add_artist(self._container.get_patch())
            s = self._ball.get_patch()
            ax.add_patch(self._ball.get_patch())

        for frame in range(num_frames):
            self.next_collision(self._container)

            s.center = self._ball._p
            if animate:
                pl.pause(0.5)
        if animate:
            pl.show()
   # test  for conservation task 6
   # Ball container
   # KE_bef=[]
   # KE_aft=[]
   #energy= KE()


# %%
l = Simulation()


l.run(10, True)

# %%
for i in range(9):
    rng = np.random.randint
    rints = rng(low=-50, high=50, size=1)
    r1 = rints[0]*0.1

    gints = rng(low=-50, high=50, size=1)
    g = gints[0]*0.1
