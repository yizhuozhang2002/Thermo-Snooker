# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 09:18:17 2022

@author: yz6621
"""
dis_balls=[]
dis_central=[]
class distance(ball):
    def __init__(self):
        Ball.__init__(self)
        
        
        
    def distance_balls(self,other):
        a=self._p-other._p
        dis_b=np.sqrt(a[0]**2+a[1]**2)
        return dis_b
    def distance_central(self):
        a=self._p
        dis_c=np.sqrt(a[0]**2+a[1]**2)
        return dis_b
    for i in range(n-1):
        dis_balls1=ball[i]._p.distance_balls(ball[i+1]._p)
        dis_balls.append(dis_balls1)
    for i in range(n):
        dis_central1=ball[i].distance_central
        dis_central.append(dis_central1)
        