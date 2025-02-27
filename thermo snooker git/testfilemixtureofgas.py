# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 22:02:45 2022

@author: yz6621
"""

import numpy as np
import pylab as pl
import generateballsauto as gb
import matplotlib.pyplot as plt
import balls as balls
from scipy.optimize import curve_fit
a = [0, 1]
b = [1, 0]
k = 1.38e-23
u = 1.66e-27
mass = u
bohr = 5.29e-11
r = 1000*bohr  # initialised radius of the ball
n = 100
w = 200  # times of the container radius to the ball radius


def distance_check(a, b):

    dp = np.array(a)-np.array(b)
    distance = np.sqrt((dp[0])**2+(dp[1])**2)
    if distance < 2:  # 2radius
        return False
    else:
        return True


class Simulation:
    '''
    parameters
    m:

    t:
        R_c radius of container
        R_b radius of balls
    '''
    """
    logic for multiple balls, find the min time to collisde for the ball between each ohter"
    and bewteen container, this is tf, use this tf to move the system, perform collision for the pair corresponds to this time to collision tf
    and repeat the process, plot animation.
    movce the system by time tf, generate a set of new patch center, perform collision, then

    """

    def __init__(self, pb=[], R_c=w*r, R_b=[800*bohr, 1000*bohr, 500*bohr], vlist=[], v=[], dp=[], pball=[], Number_of_balls=n, pressure=[], time=[], temperature=298, Mom_change=[]):

        self.vb = []
        self._ball = []
        self.R_c = R_c
        self.pb = []
        self.R_b = R_b
        self.Number_of_balls = Number_of_balls
        self.pressure = pressure
        self.time = gb.time
        self.temperature = temperature
        self.Mom_chan = Mom_change
        self.dp = dp
        self.pball = []
        self.xpball = []
        self.ypball = []
        self.vlist = []
        sigma = np.sqrt(3*k*self.temperature/(mass))
        vxx = []
        vyy = []
        for i in range(self.Number_of_balls):
            vx = np.random.uniform(-sigma, sigma)
            vy = ((-1)**i)*np.sqrt(sigma**2-vx**2)
            vxx.append(vx)
            vyy.append(vy)
        np.random.shuffle(vxx)
        np.random.shuffle(vyy)
        for i in range(Number_of_balls):
            self.vb.append([vxx[i], vyy[i]])
        r = self.R_b[1]
        n = self.Number_of_balls
        a = np.linspace(-(w/1.5)*r, (w/1.5)*r, num=round(np.sqrt(n)))

        for i in a:
            for j in a:
                posbi = [j, i]
                self.pb.append(posbi)

        self._container = balls.Ball(
            p=[0, 0], v=[0, 0], R=self.R_c, ball=False, m=1e40)

        for i in range(self.Number_of_balls):
            # print(len(self._ball))
            # self._ball[i]=Ball(p=pb[i],v=self.vb[i],R=1,m=1)
            self._ball.append(balls.Ball(
                p=self.pb[i], v=self.vb[i], R=self.R_b[np.random.choice([0, 1, 2])], m=mass))

        self._v = self.vb
        KE_list = []
        self.KE_list = KE_list
        tlist = []
        self.tlist = tlist
        pressurelist = []
        self.pressurelist = pressurelist
        balls.Ball.__init__(self)

    def next_collision(self):

        tb = []
        tc = []
        a = []

        for i in range(self.Number_of_balls):

            tc.append(self._ball[i].time_to_collision(self._container))
            # print(tc)
            # print(self.Number_of_balls-i)
            for j in range(self.Number_of_balls):
                if i > j:

                    tb.append(self._ball[i].time_to_collision(self._ball[j]))
                    a.append([i, j])
        tf = np.min(tb)
        tfc = np.min(tc)
        if tfc <= tf:
            b = tc.index(tfc)
            self.time.append(tfc)
            # print("%s=%s" % ('tfc=', tfc))
            self._container.move(tfc)
            for i in range(self.Number_of_balls):
                self._ball[i].move(tfc)

            self._ball[b].collide(self._container)
            dp1 = self._ball[b].change_in_momentum_C()
            self.dp.append(abs(dp1))

            # if tf==self._ball[i].time_to_collision(self._container):

        else:
            s = tb.index(tf)
            self.time.append(tf)
            f = a[s]  # index of the two balls i=f[0] j=f[1]
            self._container.move(tf)
            # print("%s=%s" % ('tf=', tf))
            pi = self._ball[f[0]].momentum()+self._ball[f[1]].momentum()

            for i in range(self.Number_of_balls):
                self._ball[i].move(tf)

            self._ball[f[0]].collide(self._ball[f[1]])
            dp2 = 0
            self.dp.append(dp2)
            pf = self._ball[f[0]].momentum()+self._ball[f[1]].momentum()
            self.Mom_chan.append(pf-pi)

            # if tf==self._ball[i].time_to_collision(self._container):

            # ttot.append(t)
            # KE1=self._ball.KE(self._ball.m,self._ball._v)
            # print(KE1)
        return self

    def total_pressure(self):
        dptot = sum(self.dp)
        dt = sum(self.time)
        pressure = dptot/dt/(2*np.pi*self.R_c)
        return pressure
#    def total_pressure(self):
#        vb=np.asarray(self.vb)
#        dptot=vb[:,0]**2+vb[:,1]**2
#
#        V=np.pi*(self.R_c)**2
#        pressure=dptot*self.m/3*V
#        return pressure

    def KE_all(self):
        KEall = []
        for i in range(self.Number_of_balls):

            KEi = self._ball[i].KE()
            KEall.append(KEi)
        return KEall

    def Temperature(self):
        # k = 1.38*10 ^ -23
        k = 1.38e-23
        temp = 2*sum(self.KE_all())/(3*k*n)
        return temp

    def run(self, num_frames, animate=False):
        s = []
        self.dp = []
        self.time = []
        self.pball = []
        pballs = []

        for i in range(self.Number_of_balls):

            s.append(self._ball[i].get_patch())

        if animate:
            f = pl.figure()
            ax = pl.axes(xlim=(-self.R_c, self.R_c),
                         ylim=(-self.R_c, self.R_c))
            ax.add_artist(self._container.get_patch())
            for i in range(self.Number_of_balls):

                # s.append(self._ball[i].get_patch())
                ax.add_patch(self._ball[i].get_patch())

        for frame in range(num_frames):
            self.KE_list.append(sum(self.KE_all())+self._container.KE())
            # print(frame)
            # momoemtum

            pballs = []
            pballsx = []
            pballsy = []
            for i in range(self.Number_of_balls):
                #                #sqrt(2*self._ball[i].KE()/self._ball[i].m) != np.linalg.norm(self._ball[i]._v):
                #                    raise Exception(np.sqrt(2*self._ball[i].KE()/self._ball[i].m),np.linalg.norm(self._ball[i]._v))
                pballi = np.sqrt(
                    self._ball[i]._v[0]**2+self._ball[i]._v[1]**2)*self._ball[i].m
                pballx = self._ball[i]._v[0]*self._ball[i].m
                pbally = self._ball[i]._v[1]*self._ball[i].m
                vx = self._ball[i]._v[0]
                vy = self._ball[i]._v[1]
                v = [vx, vy]
#                if round(self._ball[i].KE(),24)!=round(pballt**2/2*self._ball[i].m,24):
#                    raise Exception(round(self._ball[i].KE(),27),round(pballt**2/2*self._ball[i].m,27))
                self.vlist.append(v)
                pballs.append(pballi)
                pballsx.append(pballx)
                pballsy.append(pbally)
            pc = self._container.momentum()
            pcc = np.sqrt(pc[0]**2+pc[1]**2)
            pcx = pc[0]
            pcy = pc[1]
            # print(pcc)

            self.pball.append(np.sum(pballs)+pcc)
            # print(len(self.pball))
            self.xpball.append(np.sum(pballsx)+pcx)
            self.ypball.append(np.sum(pballsy)+pcy)
            # momentum
            self.next_collision()

            for i in range(self.Number_of_balls):

                s[i].center = self._ball[i]._p
            if frame % 100 == 0:
                print(frame)

            if animate:
                pl.pause(0.00001)
#

        if animate:
            # self.KE_con_checkplot()
            pl.show()

#            dis_balls = []
#            dis_central = []
#            n = self.Number_of_balls
#            for i in range(n):
#                for j in range(n):
#                    if i > j:
#
#                        dis_balls1 = self._ball[i].distance_balls(
#                            self._ball[j])
#                        dis_balls.append(dis_balls1)
#            for i in range(n):
#                dis_central1 = self._ball[i].distance_central()
#                dis_central.append(dis_central1)
#            print(dis_balls, dis_central)
#            fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
#            ax[0].hist(dis_central, bins=10)
#            ax[1].hist(dis_balls, bins=10)
#            plt.show()
            # self.KE_con_checkplot()

    def KE_con_checkplot(self):
        for i in range(len(self.KE_list)):
            self.KE_list[i] = round(self.KE_list[i], 23)
        y = self.KE_list
        x = self.time
        f = plt.plot(x, y, 'x')
        for i in range(len(self.KE_list)):
            for j in range(len(self.KE_list)):
                if i > j:
                    a = y[i]-y[j]
                    if a > 0.1E-7:
                        raise Exception("KE not conserved")
                        print('balls', i, j)

        return f

    def distance_plots(self):
        dis_balls = []
        dis_central = []
        n = self.Number_of_balls
        for i in range(n):
            for j in range(n):
                if i > j:

                    dis_balls1 = self._ball[i].distance_balls(
                        self._ball[j])
                    dis_balls.append(dis_balls1)
        for i in range(n):
            dis_central1 = self._ball[i].distance_central()
            dis_central.append(dis_central1)
        print(dis_balls, dis_central)
        fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
        ax[0].hist(dis_central, bins=10)
        ax[1].hist(dis_balls, bins=10)
        plt.show()

    def pressure_over_time(self):
        # pressurelist=[]
        # tlist=[]
        for i in range(1, 100):
            t = sum(self.time[:round((len(self.time)/100)*i)])
            p = sum(self.dp[:round((len(self.dp)/100)*i)])
            pressurelist1 = t/p
            self.pressurelist.append(pressurelist1)
            self.tlist.append(t)
        plt.plot(self.tlist, self.pressurelist, 'x')
        plt.show()

    def __repr__(self):

        return "%s(%g,%g)" % ('ball_pos', self._ball[0]._p, self._ball[1]._p)

   # test  for conservation task 6
   # Ball container
   # KE_bef=[]
   # KE_aft=[]
   # energy= KE()
# %%


N = 100

init_guess = [0, 5.5e-12]


def ideal_gas(V, N, k, T):
    return N*k*T/V


V = np.pi*(1.58e-5)**2  # radius of container


def VDW(T, a, b):
    return (N*k/(V-N*b))*T-a*N**2/V**2


xrr = np.array(x_r)

plt.plot(xrr, y_r, 'x')
# a1, b1 = np.polyfit(xrr, y_r1, 1)
popt10, pcov10 = curve_fit(VDW, xrr, y_r, init_guess)
plt.plot(xrr, VDW(xrr, popt10[0], popt10[1]), label='VDW for mixture of gas ')
plt.plot(xrr, ideal_gas(V, N, k, xrr), label='ideal gas')
plt.title(
    'Pressure-Temperature graph for mixture of gas(rad=500,800,1000bohr)', fontsize=8.5)
plt.xlabel('Temperature/K')
plt.ylabel('Pressure/pa')
plt.legend()
plt.show()
'''
array([-1.55854514e-33,  5.49911806e-12])
array([[1.99529766e-66, 1.40191447e-47],
       [1.40191447e-47, 1.40420850e-28]])
'''


# %%
# %%
# test
l = Simulation(temperature=298)

numrun = 6000
l.run(numrun, False)
# %%
'task13'
m = u
T = 1100
# probability P(x) probabliyty is frequency which is number of balls


def Max_boltz(v, A=1):
    '''
    equation used is from
    1.
Physics 1
Voronkov Vladimir Vasilyevich
https://ppt-online.org/519344
'''
    return (A)*(v)*np.exp(-(m*(v**2))/(k*T*2))


v1 = np.asarray(l.vlist)
vx = v1[:, 0:1]
y2 = Max_boltz(vx)
x2 = vx
plt.title('vx ')
plt.hist(x2, bins=10)
# plt.show()
plt.title('vy ')
plt.show()
# plt.plot(x2,y2)
plt.show()

# %%
'task13'

m = mass
T = 596
# probability P(x) probabliyty is frequency which is number of balls


def Max_boltz(v, A):
    '''
    equation used is from
    1.
Physics 1
Voronkov Vladimir Vasilyevich
https://ppt-online.org/519344
'''

    return (A)*(v)*np.exp(-(m*(v**2))/(k*T*2))


v1 = np.asarray(l.vlist)

vx = abs(v1[:, 0])


vy = abs(v1[:, 1])
x2 = vx
x3 = vy
x4 = np.sqrt(x2**2+x3**2)
bin_height, bin_edge = np.histogram(x4, bins=30)
# bin_mid=bin_edge[:(len(bin_edge)-1)]+np.diff(bin_edge)/2
bin_mid = bin_edge[1:]-np.diff(bin_edge)/2
xfit = bin_mid
yfit = bin_height/sum(bin_height)
initial_guess = []
popt, pcov = curve_fit(Max_boltz, xfit, yfit)
xcurve = np.linspace(0, 11800)
plt.plot(xcurve, Max_boltz(xcurve, popt[0]))
plt.title('v magnitude ')
plt.hist(bin_mid, bins=30, weights=yfit)


plt.show()
# %%
np.var(yfit)
'''
1424042.7946376898
'''

# %%

a = 5

yr = []
radius = np.array([10000, 50000, 100000, 500000, 1000000])*bohr
for j in range(5):
    #    R_b=10**j*bohr
    press = []
    temperature = []
    for i in range(a):
        temperaturei = 50+298*i
        l = Simulation(R_b=radius[j], temperature=temperaturei)
        l.run(1000, False)
        # pressi= round(l.total_pressure(),20)
        pressi = l.total_pressure()
        temperature.append(temperaturei)
        # print(sum(l.time))
        press.append(pressi)

        print('runtime', i)
    x_r = temperature
    y_r = press

    yr.append(y_r)
    print('j runtime', j)
#    xr=np.array(x_r)
#    plt.plot(xr,y_r1,'x')
#    a1, b1 = np.polyfit(xr, y_r1, 1)
#    plt.plot(xr1,a1*(xr1)+b1)
