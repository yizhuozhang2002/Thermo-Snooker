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
a = [0, 1]
b = [1, 0]
k=1.38e-23
u=1.66e-27
mass=u
bohr=5.29e-11
r=1000*bohr
n=100
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

    def __init__(self, pb=[], R_c=50*r, R_b=r, vlist=[],v=[[1, 1], [2, 0], [-1, 3]],dp=[], pball=[], Number_of_balls=n, pressure=[], time=[],temperature=298, Mom_change=[]):
        #posb =gb.posbb
        self.vb = []
        self._ball = []
        self.R_c = R_c
        self.pb=pb
        self.R_b = R_b
        self.Number_of_balls = Number_of_balls
        self.pressure = pressure
        self.time = gb.time
        self.temperature=temperature
        self.Mom_chan = Mom_change
        self.dp=dp
        self.pball=[]
        self.vlist=[]
        sigma=np.sqrt(3*k*self.temperature/(mass))
        vxx=[]
        vyy=[]
        
        r=self.R_b
        n=self.Number_of_balls
        a = np.linspace(-34*r, 34*r, num=round(np.sqrt(n)))
        posbb = []
        for i in a:
            for j in a:
                posbi = [j, i]
                posbb.append(posbi)
                
        self.pb=posbb
        for i in range(self.Number_of_balls):
            vx=np.random.uniform(-sigma,sigma)
            vy=((-1)**i)*np.sqrt(sigma**2-vx**2)
            vxx.append(vx)
            vyy.append(vy)
        np.random.shuffle(vxx)
        np.random.shuffle(vyy)
        for i in range(Number_of_balls):
            self.vb.append([vxx[i],vyy[i]])    
        self._container = balls.Ball(
            p=[0, 0], v=[0, 0], R=self.R_c, ball=False, m=1e40)

        for i in range(self.Number_of_balls):
            #print(len(self._ball))
            # self._ball[i]=Ball(p=pb[i],v=self.vb[i],R=1,m=1)
            self._ball.append(balls.Ball(p=self.pb[i], v=self.vb[i], R=r, m=mass))

        self._v = self.vb
        KE_list=[]
        self.KE_list=KE_list
        tlist=[]
        self.tlist=tlist
        pressurelist=[]
        self.pressurelist=pressurelist
        balls.Ball.__init__(self)
    
    def next_collision(self):
         
         
         
        tb = []
        tc = []
        a = []

        for i in range(self.Number_of_balls):

            tc.append(self._ball[i].time_to_collision(self._container))
            #print(tc)
            #print(self.Number_of_balls-i)
            for j in range(self.Number_of_balls):
                if i > j:

                    tb.append(self._ball[i].time_to_collision(self._ball[j]))
                    a.append([i, j])
        tf = np.min(tb)
        tfc = np.min(tc)
        if tfc <= tf:
            b = tc.index(tfc)
            self.time.append(tfc)
            #print("%s=%s" % ('tfc=', tfc))
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
            #print("%s=%s" % ('tf=', tf))
            pi = self._ball[f[0]].momentum()+self._ball[f[1]].momentum()

            for i in range(self.Number_of_balls):
                self._ball[i].move(tf)

            self._ball[f[0]].collide(self._ball[f[1]])
            dp2=0
            self.dp.append(dp2)
            pf = self._ball[f[0]].momentum()+self._ball[f[1]].momentum()
            self.Mom_chan.append(pf-pi)
            

            # if tf==self._ball[i].time_to_collision(self._container):

            # ttot.append(t)
            # KE1=self._ball.KE(self._ball.m,self._ball._v)
            # print(KE1)
        return self
    def total_pressure(self):
        dptot=sum(self.dp[200:])
        dt=sum(self.time[200:])
        pressure=dptot/dt/(2*np.pi*self.R_c)
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
        #k = 1.38*10 ^ -23
        k = 1.38e-23
        temp = 2*sum(self.KE_all())/(3*k*n)
        return temp

    def run(self, num_frames, animate=False):
        s = []
        self.dp=[]
        self.time=[]
        self.pball=[]
        pballs=[]
       
        for i in range(self.Number_of_balls):

            s.append(self._ball[i].get_patch())

        if animate:
            f = pl.figure()
            ax = pl.axes(xlim=(-self.R_c, self.R_c), ylim=(-self.R_c, self.R_c))
            ax.add_artist(self._container.get_patch())
            for i in range(self.Number_of_balls):

                # s.append(self._ball[i].get_patch())
                ax.add_patch(self._ball[i].get_patch())

        for frame in range(num_frames):
            self.KE_list.append(sum(self.KE_all())+self._container.KE())
            #print(frame)
            #momoemtum
            
            pballs=[]        
            for i in range(self.Number_of_balls):
#                #sqrt(2*self._ball[i].KE()/self._ball[i].m) != np.linalg.norm(self._ball[i]._v):
#                    raise Exception(np.sqrt(2*self._ball[i].KE()/self._ball[i].m),np.linalg.norm(self._ball[i]._v))
                pballi=np.sqrt(self._ball[i]._v[0]**2+self._ball[i]._v[1]**2)*self._ball[i].m
                vx=self._ball[i]._v[0]
                vy=self._ball[i]._v[1]
                v=[vx,vy]
#                if round(self._ball[i].KE(),24)!=round(pballt**2/2*self._ball[i].m,24):
#                    raise Exception(round(self._ball[i].KE(),27),round(pballt**2/2*self._ball[i].m,27))
                self.vlist.append(v)
                pballs.append(pballi)
            pc=self._container.momentum()    
            pcc=np.sqrt(pc[0]**2+pc[1]**2)
            #print(pcc)
            
            
            self.pball.append(np.sum(pballs)+pcc/2)
            #print(len(self.pball))
            
            #momentum
            self.next_collision()
            
            for i in range(self.Number_of_balls):

                s[i].center = self._ball[i]._p
            if frame%100==0:
                    print(frame)
                    
            if animate:
                pl.pause(0.00001)
#            
            
        if animate:
            #self.KE_con_checkplot()
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
            #self.KE_con_checkplot()
        
    def KE_con_checkplot(self):
         for i in range(len(self.KE_list)):
             self.KE_list[i]=round(self.KE_list[i],23)
         y = self.KE_list
         x = self.time
         f = plt.plot(x, y, 'x')
         for i in range(len(self.KE_list)):
             for j in range(len(self.KE_list)):
                 if i>j:
                     a=y[i]-y[j]
                     if a>0.1E-7:
                         raise Exception("KE not conserved")
                         print('balls',i,j)
                     
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
        #pressurelist=[]
        #tlist=[]
        for i in range (1,100):
            t=sum(self.time[:round((len(self.time)/100)*i)])
            p=sum(self.dp[:round((len(self.dp)/100)*i)])
            pressurelist1=t/p
            self.pressurelist.append(pressurelist1)
            self.tlist.append(t)
        plt.plot(self.tlist,self.pressurelist,'x')
        plt.show()
            
        
        
        
        
        
        
        
        
    def __repr__(self):

        return "%s(%g,%g)" % ('ball_pos', self._ball[0]._p, self._ball[1]._p)

   # test  for conservation task 6
   # Ball container
   # KE_bef=[]
   # KE_aft=[]
   # energy= KE()
   #%%
   '''
   a loop which runs from 298 to 298*a with 298 per interval to show PT relation 
   at certian V,R,r
   '''
a=5
press=[]
temperature=[]
#for j in range(1,5):
#    R_b=10**j*bohr
for i in range(a):
    temperaturei=50+298*i
    l = Simulation(R_b=balls.r,temperature=temperaturei)
    l.run(1000, False)
    pressi= round(l.total_pressure(),20)
    temperature.append(temperaturei)
    #print(sum(l.time))
    press.append(pressi)
    print('runtime',i)
x_r=temperature
y_r5=press

#%%
'''
plot system momentum with respect to time
ypt: total time spent
'''
#p=self.m*self._v
#self.time as x
#pball=[]
#for first 100 run, calculate pball
#pballi=self._ball[i].momentum()
#pball.append(pball)
yp=[]
yp1=l.pball
yp1=np.array(yp1)



for i in range(numrun):
    ypp=round(yp1[i],22)
    yp.append(ypp)
xp=[]
for i in range (numrun):
    
    xp1=l.time[:i]
    xpt=sum(xp1)
    xp.append(xpt)
plt.plot(xp,yp1)
x=yp1.mean()
plt.axhline (x) 
plt.ylim(0,2*x)
plt.title('moemntum versus time')
plt.xlabel('Time/s')
plt.ylabel('Momentum')
#%%
xk=[]
for i in range (numrun):
    xk1=l.time[:i]
    xkt=sum(xk1)
    xk.append(xkt)
#yk=[]
#for i in range(numrun):
#    ykk=round(l.KE_list[i],23)
#    yk.append(ykk)
yk=l.KE_list
plt.title('KE vs time')
plt.xlabel('KE/J')
plt.ylabel('Time/s')
plt.plot(xk,yk)
#%%
# test
l = Simulation(temperature=596)

numrun=100
l.run(numrun, True)


#pressure is 3.900000000000141e-08 at 293K
#pressure is 8.000000000000183e-08 at T =586K
'''

pressure change from 8 to 1.620000000000002e-07
Vto 4V
container to 2R
kinetic energy between pair of balls which collide with each other is conserved
'''     
#%%
'task13'
m=u
T=1100
#probability P(x) probabliyty is frequency which is number of balls
def Max_boltz(v,A=1):
    '''
    equation used is from
    1.
Physics 1
Voronkov Vladimir Vasilyevich
https://ppt-online.org/519344
'''
    return (A)*(v)*np.exp(-(m*(v**2))/(k*T*2))

v1=np.asarray(l.vlist)
vx=v1[:,0:1]
y2 = Max_boltz(vx)
x2=vx
plt.title('vx ')
plt.hist(x2, bins=10)
#plt.show()
plt.title('vy ')
plt.show()
#plt.plot(x2,y2)
plt.show()

#%%
plt.show()
x=np.array(temperature)
plt.plot(x,y,'x')

x1=np.linspace(0,3000,100)
y1=((100*(k))/(np.pi*(50*r)**2))*np.array(x)
plt.plot(x,y1)
a, b = np.polyfit(x, y, 1)
print(a,b)
plt.plot(x,a*x+b)
plt.title('Presure temperature reltionship')
plt.show()    
#%%
'task13'
m=mass
T=596
#probability P(x) probabliyty is frequency which is number of balls
def Max_boltz(v,A=1):
    '''
    equation used is from
    1.
Physics 1
Voronkov Vladimir Vasilyevich
https://ppt-online.org/519344
'''

    return (A)*(v)*np.exp(-(m*(v**2))/(k*T*2))
x5=np.linspace(0,1e8,10000)
v1=np.asarray(l.vlist)
x4=(x2**2+x3**2)
vx=abs(v1[:,0])
y2 = Max_boltz(x5,A=8000)
x2=vx

vy=abs(v1[:,1])
x3=vy
plt.title('v magnitude ')
plt.hist((x2**2+x3**2), bins=100)

plt.plot(x5,y2)

plt.show()
#%%
'''
task 12
'''
x_r1=[50, 348, 646, 944, 1242]
x_r2=[50, 348, 646, 944, 1242]
x_r3=[50, 348, 646, 944, 1242]
y_r1=[5.2554748904e-09, 3.763928227291e-08, 7.214701115632e-08, 9.401830200093e-08, 1.2612020079883e-07]
y_r2=[4.801559184235e-08, 3.3199339891012e-07, 6.1353167734307e-07, 9.1310960931694e-07, 1.16634846370408e-06]
y_r3=[4.7843771744862e-07, 3.36502457183983e-06, 6.22907584389921e-06, 9.05944248897433e-06, 1.18955855525761e-05]
y_r4=[2.88494636667e-09, 1.450167281264e-08, 2.963217809612e-08, 3.623531075838e-08, 6.180918688438e-08]
y_r5=[2.18205454813e-09, 2.429257285743e-08, 2.91461584438e-08, 4.657787224941e-08, 5.471884236857e-08]

plt.show()
xr=np.array(x_r1)
plt.plot(xr,y_r1,'x')
a1, b1 = np.polyfit(xr, y_r1, 1)
plt.plot(xr1,a1*(xr1)+b1)
#add label
#add theoritical line
#need another radius line between 10000r and 1000r

plt.plot(xr,y_r2,'x')
a2, b2 = np.polyfit(xr1, y_r2, 1)
plt.plot(xr2,a2*xr1+b2)



plt.plot(xr,y_r3,'x')
a3, b3 = np.polyfit(xr1, y_r3, 1)
plt.plot(xr1,a3*xr1+b3)



plt.plot(xr,y_r4,'x')
a4, b4 = np.polyfit(xr1, y_r4, 1)
plt.plot(xr1,a4*xr1+b4)



plt.plot(xr,y_r5,'x')
a5, b5 = np.polyfit(xr1, y_r5, 1)
plt.plot(xr1,a5*xr1+b5)




plt.title('Presure temperature reltionship')
plt.show()   
#x1=np.linspace(0,3000,100)
#y1=((100*(k))/(np.pi*(50*r)**2))*np.array(x)
#plt.plot(x,y1)
#%%
import balls as balls
balls.n=50
#x=N/V
