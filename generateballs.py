# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:19:12 2022

@author: sophia
"""
import numpy as np
a=[0,1]
b=[1,0]
def distance_check(a,b):
    
    dp=np.array(a)-np.array(b)
    distance=np.sqrt((dp[0])**2+(dp[1])**2)
    if distance<2:#2radius
        return False
    else:
        return True


posb=[]
rbs=np.array([i for i in range(10)])#generate positions systematically
for i in range (10):
    rb=np.random.choice(rbs,p=rbs/sum(rbs))*1    
    the=np.random.uniform(-np.pi,np.pi)
    #if the=0 and the=np.pi and the=-np.pi and the=np.pi/2 and the=-np.pi/2:
        #return the=1
        
    x=np.fix(10*rb*np.cos(the))/10
    y=np.fix(rb*np.sin(the)*10)/10
    
    posb.append([x,y])
    
for i in range (9):
    distance_check(posb[i],posb[i+1])
    if distance_check(posb[i],posb[i+1])==False:
        rb=np.random.choice(rbs,p=rbs/sum(rbs))*1    
        the=np.random.uniform(-np.pi,np.pi)
        #if the=0 and the=np.pi and the=-np.pi and the=np.pi/2 and the=-np.pi/2:
            #return the=1
            
        x_new=np.fix(10*rb*np.cos(the))/10
        y_new=np.fix(rb*np.sin(the)*10)/10
        posb[i]=[x_new,y_new]
print(posb)