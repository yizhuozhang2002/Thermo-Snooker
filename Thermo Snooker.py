# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:33:34 2022

@author: yunfa
"""

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Kb=1.38E-23
u=1.66E-24
a0=5.29E-11
A=6.02E23


class Ball:
    def __init__(self, position, velocity,mass=u, radius=a0):
        self._m = mass
        self._R = radius
        self._r = np.array(position)
        self._v = np.array(velocity)
    
    def pos(self):
        return self._r
    def vel(self):
        return self._v
    def rad(self):
        return self._R
    def mass(self):
        return self._m
    
    def setvel(self,velocity):
        self._v = velocity
    
    def move(self, dt):
        self._r = self._r+self._v*dt

    def time_to_collision(self, other):
        
        if other.mass()==0:
            r = self._r
            v = self._v
            R = other.rad()-self._R
            v_square = (np.dot(self._v,self._v))
            

        else:
            dt=0
            r = self._r-other.pos()
            v = self._v-other.vel()
            R = self._R+other.rad()
            v_square = np.dot(v,v)
        dt = -np.dot(r,v)/v_square+R/np.sqrt(v_square)
            
        if not np.isreal(dt):
            return 0
        if  dt>0:
            return dt
        else:
            return 0

            
        
        

    def collide(self, other):
        m1 = self._m
        m2 = other.mass()
        if m2==0:#meaning the other is the container
            v1=self._v
            contact_dir= self._r#the normal line of collision
            v_nor=-sum(v1*contact_dir)*contact_dir/(sum(contact_dir**2))
            v_tan=v1+v_nor
            self._v=v_nor+v_tan
            return m1*np.sqrt(np.dot(self._v-v1,self._v-v1))
            
        else:#colliding with a ball
            M=m1+m2
            v1 = self._v
            v2 = other._v
            r1,r2=self._r,other.pos()
            d = np.linalg.norm(r1 - r2)**2
            u1 = v1 - 2*m2 / M * np.dot(v1-v2, r1-r2) / d * (r1 - r2)
            u2 = v2 - 2*m1 / M * np.dot(v2-v1, r2-r1) / d * (r2 - r1)

            self._v=v1
            other.setvel(v2)
            return 0
        
    def get_patch(self):
        return(pl.Circle(self._r,self._R,fc='r'))
        
class Container(Ball):
    def __init__(self,rad=10*a0):
        Ball.__init__(self,[0,0],0,0,radius=rad)
    def get_patch(self):
        return(pl.Circle(self._r,self._R,ec='b',fill=False))
        
class Simulation():
    def __init__(self, container,No_Of_Balls,temperature=273,radius=a0,Mass=u):
        self._container=container
        self._Balls=[]
        count=0
        pos=[]
        num_of_rs=int((container.rad()-radius)/radius)
        rs=np.array([i for i in range(num_of_rs)])#generate positions systematically
        for i in range (No_Of_Balls):
            r=np.random.choice(rs,p=rs/sum(rs))*radius
            theta=np.random.uniform(-np.pi,np.pi)
            x=r*np.cos(theta)
            y=r*np.sin(theta)
            pos.append([x,y])

        vxs=[]#generate velocity randomly but keep average as 0
        vys=[]
        vmax=np.sqrt(2*Kb*temperature/(Mass))
        for i in range(No_Of_Balls):
            vx=np.random.uniform(-vmax,vmax)
            vy=np.sqrt(vmax**2-vx**2)
            vxs.append(vx)
            vys.append(vy)
        np.random.shuffle(vxs)
        np.random.shuffle(vys)
        for i in range(No_Of_Balls):#set up the balls
            self._Balls.append(Ball(position=pos[i],velocity=[vxs[i],vys[i]],mass=Mass))
        self._This_Collision=[-1,-1]
        self._Shortest_Times=[]
        self._Collision_Partners=[]
        
    def getBalls(self):
        return self._Balls
    
    def next_collision(self):
        object2s=[]
        if self._This_Collision==[-1,-1]:
            for i in range(len(self._Balls)):
                self._Collision_Partners.append(-1)
                self._Shortest_Times.append(self._Balls[i].time_to_collision(self._container))
                for j in range(i+1,len(self._Balls)):
                    time=self._Balls[i].time_to_collision(self._Balls[j])
                    if time < self._Shortest_Times[i] and time != 0:
                        self._Collision_Partners[i] = j
                        
                        self._Shortest_Times[i] = time
            shortest_time=10000
            for i in range(len(self._Balls)):
                if self._Shortest_Times[i]<shortest_time:
                    self._This_Collision=[i,self._Collision_Partners[i]]
                    shortest_time=self._Shortest_Times[i]
        
            
        else:
            obj1=self._This_Collision[0]#change the collision partner for object one in this collision
            obj2=self._This_Collision[1]
            self._Collision_Partners[obj1]=-1
            container_time=self._Balls[obj1].time_to_collision(self._container)
            if container_time!=0:
                self._Shortest_Times[obj1]=container_time
            else:
                self._Shortest_Times[obj1]=10000
            for i in range(obj1+1,len(self._Balls)):
                if i != self._This_Collision[1]:
                    time=self._Balls[obj1].time_to_collision(self._Balls[i])
                    if time < self._Shortest_Times[obj1] and time != 0:
                        self._Collision_Partners[obj1] = i
                        self._Shortest_Times[obj1] = time
            rang=max([obj1,obj2])
            for i in range(rang):#only those before the objects has possibility of having it as collision partner
                if self._Collision_Partners[i] in self._This_Collision:
                    self._Collision_Partners[i]=-1
                    container_time=self._Balls[i].time_to_collision(self._container)
                    
                    if container_time!=0:
                        self._Shortest_Times[i]=container_time
                    else:
                        self._Shortest_Times[i]=10000
                        
                    
                    for j in range(i+1,len(self._Balls)):
                        time=self._Balls[i].time_to_collision(self._Balls[j])
                        if time < self._Shortest_Times[i] and time!=0:
                            self._Collision_Partners[i] = j
                            self._Shortest_Times[i] = time
            shortest_time=10000        
            for i in range(len(self._Balls)):
                if self._Shortest_Times[i]<shortest_time:
                    self._This_Collision=[i,self._Collision_Partners[i]]
                    shortest_time=self._Shortest_Times[i]

        for i in range (len(self._Balls)):
            if self._Balls[i] not in self._This_Collision:
                self._Shortest_Times[i]=self._Shortest_Times[i]-shortest_time
        for i in range (len(self._Balls)):
            self._Balls[i].move(shortest_time)
        P_to_wall=0
        if self._This_Collision[1] == -1:           
            P_to_wall=self._Balls[self._This_Collision[0]].collide(self._container)
        else:
            P_to_wall=self._Balls[self._This_Collision[0]].collide(self._Balls[self._This_Collision[1]])
        return shortest_time,P_to_wall    
    def run(self, num_frames, animate=False):
        if animate:
            f = pl.figure()
            ax = pl.axes(xlim=(-self._container.rad(), self._container.rad()), ylim=(-self._container.rad(), self._container.rad()))
            ax.add_artist(self._container.get_patch())
            patches=[]
            for i in range(len(self._Balls)):
                patches.append(self._Balls[i].get_patch())
                ax.add_patch(patches[i])
        else:
            patches=[]
            for i in range(len(self._Balls)):
                patches.append(self._Balls[i].get_patch())
        KE=[]    
        Px=[]
        Py=[]
        time=0
        times=[]
        totalPressure=0
        
        for frame in range(num_frames):
            #obtain kinetic energy,pressure and momentum against time
            t,p=self.next_collision()
            if frame>0.4*num_frames:
                time+=t
                totalPressure+=p/(np.pi*(a0**2))
            totalKE=0
            totalP=np.array([0,0])
            
            for i in range(len(self._Balls)):
                patches[i].center=self._Balls[i].pos()
                ke=0.5*self._Balls[i].mass()*(np.dot(self._Balls[i].vel(),self._Balls[i].vel()))
                totalKE+=ke
                momentum=self._Balls[i].mass()*self._Balls[i].vel()
                totalP = totalP+momentum
            KE.append(totalKE)
            Px.append(totalP[0])
            Py.append(totalP[1])
            times.append(time)
            
            
            
            if animate:
                pl.pause(0.1)
        
            
                
        if animate:
                pl.show()
        return totalPressure/time,KE,np.array(Px),np.array(Py),times
    

                
task=2
while task!=0:
    task=int(input('Which task would you like to see(9,11,12,13,14, 1 to see animation, 0 to exit):'))
    while task not in [0,1,9,11,12,13,14]:
        task = int(input('Which task would you like to see(9,11,12,13,14, 1 to see animation, 0 to exit):'))

    if task == 1:
        container=Container()
        No_Of_Particles=10
        system=Simulation(container,No_Of_Particles)
        Pressure,KE,Px,Py,times=system.run(1000,True)
    elif task == 9:
        container=Container(rad=100*a0)
        No_Of_Particles=100
        system=Simulation(container,No_Of_Particles)
        Pressure,KE,Px,Py,times=system.run(1000,False)
        # plot histograms of ball distance and seperation
        Balls=system.getBalls()
        ball_distances=[np.sqrt(np.dot(Balls[i].pos(),Balls[i].pos())) for i in range(No_Of_Particles)]
        plt.hist(ball_distances,bins=9)
        plt.title('Ball Distance From Container Centre')
        plt.xlabel('Ball Distance')
        plt.ylabel('Count')
        plt.show()
        inter_ds=[]
        for i in range(No_Of_Particles):
            Ball1=Balls[i]
            for j in range(i+1,No_Of_Particles):
                 Ball2=Balls[j]
                 d=Ball1.pos()-Ball2.pos()
                 inter_d=np.sqrt(np.dot(d,d))
                 inter_ds.append(inter_d)
        plt.hist(inter_ds,bins=18)
        plt.title('Inter-Ball Separation')
        plt.xlabel('Seperation')
        plt.ylabel('Count')
        plt.show()
                    
    elif task == 11:
        def linear(x,m,c):
            return(m*x+c)
        #Plot kinetic energy and momentum vs time
        
        container=Container()
        No_Of_Particles=10
        system=Simulation(container,No_Of_Particles)
        Pressure,KE,Px,Py,times=system.run(1000,False)
        plt.plot(times,KE)
        plt.title('System kinetic energy versus time')
        plt.xlabel('time')
        plt.ylabel('kinetic energy')
        plt.show()
        
        plt.plot(times,Px,label='Px')
        plt.plot(times,Py,label='Py')
        plt.title('Momentum versus time')
        plt.xlabel('time')
        plt.ylabel('Momentum')
        plt.legend()
        plt.show()

        #plot pressure versus temperature
        Pressure=[]
        temperature=[]
        V=(4/3)*np.pi*(container.rad()**3)

        for i in range (20):
            system=Simulation(container,20,temperature=(i+1)*273)
            pressure,KE,Px,Py,times=system.run(5000,False)
            Pressure.append(pressure)
            temperature.append((i+1)*273)


        plt.plot(temperature,Pressure)
        plt.title('Pressure Versus Temperature')
        plt.xlabel('Temperature(K)')
        plt.ylabel('Pressure(Pa)')
        plt.show()
        
        #change volume
        TPratio=[]
        Volumes=[]
       

        for i in range (30):
            container=Container(rad=10*(i+1)*a0)
            system=Simulation(container,10)
            pressure,KE,Px,Py,times=system.run(3000,False)
            TPratio.append(273/pressure)
            Volumes.append((4/3)*np.pi*(container.rad()**3))
        
        Volumes=np.array(Volumes)
        TPratio=np.array(TPratio)
        po,po_cov=curve_fit(linear,Volumes[20:]*(10**23),TPratio[20:]*(10**(-13)))
        
        a="{:.2e}".format(po[0]*(10**36))
        plt.plot(Volumes,TPratio,'rx')
        plt.plot(Volumes[20:],linear(Volumes[20:],po[0]*(10**36),po[1]*(10**13)),'b-',label='T/PV='+a)
        plt.title('P T Relationship when changing Volume')
        plt.xlabel('Volume($m^{3}$)')
        plt.ylabel('T/P(K/Pa)')
        plt.legend()
        plt.show()
        
        No_Of_Particles=[]
        TPratio=[]
        for i in range (10):
            
            container=Container()
            system=Simulation(container,5*(i+1))
            pressure,KE,Px,Py,times=system.run(5000,False)
            TPratio.append(273/pressure)
            No_Of_Particles.append(5*(i+1))


        plt.plot(No_Of_Particles,TPratio)
        plt.title('P T Relationship when changing Number of Particles')
        plt.xlabel('Number of Particles')
        plt.ylabel('T/P(K/Pa)')
        plt.show()
        
            
    elif task == 12:    
        
        #get temperature, pressure and radius relationship
    
        def linear(x,m,c):
            return(m*x+c)
        container=Container(rad=100*a0)
        Pressure=[]
        temperature=[]
        V=(4/3)*np.pi*(container.rad()**3)
        ratio=[]
        rads=[]
        for i in range (20):
            system=Simulation(container,20,273,radius=(i+1)*a0)
            pressure,KE,Px,Py,times=system.run(8000,False)
            Pressure.append(pressure)
            rads.append((i+1)*a0)
            ratio.append(Pressure[i]*V/273)
        rads=np.array(rads)
        po,po_cov=curve_fit(linear,rads,ratio)
        
        plt.plot(rads,linear(rads,po[0],po[1]),'b-')
        plt.plot(rads,ratio,'rx')
        plt.title('Ball Radius and Equation of State')
        plt.xlabel('Radius')
        plt.ylabel('PV/T')
        plt.legend()
        plt.show()
    
    elif task==13:
        #Maxwell Boltzmann Equation
        
        def Max_Boltz(v,c):
            return c*(v)*np.exp(-(u*(v**2))/(Kb*T*2))
        container=Container(rad=5000*a0)
        No_Of_Particles=500
        T=100
        system=Simulation(container,No_Of_Particles,T)
        Pressure,KE,Px,Py,times=system.run(10000,False)
        Balls=system.getBalls()
        
        vs=[]
        
        for i in range(len(Balls)):
            velocity=(Balls[i].vel())
            vs.append(np.sqrt(np.dot(velocity,velocity)))
        vs=np.array(vs)
        
        bin_height,bin_edge=np.histogram(vs,bins=20)
        bin_mid=np.array([(bin_edge[i]+bin_edge[i+1])/2 for i in range(20)])
        bin_height=bin_height/sum(bin_height)
        po,po_cov=curve_fit(Max_Boltz,bin_mid,bin_height)
        
        plt.hist(bin_mid,bins=20,weights=bin_height)
        plt.plot(bin_mid,Max_Boltz(bin_mid, po[0]),label='Maxwell-Boltzmann Distribution')
        plt.title('Ball Speed and Theoretical Model')
        plt.xlabel('Speed')
        plt.ylabel('Probability')
        plt.legend()
        plt.show()
    
    elif task == 14:
        def VDW(T,c1,c2):
            return c1*T-c2
        
        container=Container(rad=10*a0)
        Pressures=[]
        temperature=[]
        V=np.pi*(container.rad()**2)
        N=10
        
        
        for i in range (10):
            system=Simulation(container,N,(i+1)*273)
            Pressure,KE,Px,Py,times=system.run(10000,False)
            Pressures.append(Pressure)
            temperature.append((i+1)*273)
        temperature=np.array(temperature)
        
        po,po_cov=curve_fit(VDW, temperature, Pressures)
        b="{:.2e}".format((V-N*Kb/po[0])/N)
        a="{:.2e}".format(po[1]*(V**2)/(N**2))
        plt.plot(temperature,Pressures,'x')
        plt.plot(temperature,VDW(temperature,po[0],po[1]),label='a='+a+', b='+b)
        plt.title('Van der Waal')
        plt.xlabel('Temperature(K)')
        plt.ylabel('Pressure(Pa)')
        plt.legend()
        plt.show()
    else:
        break



        