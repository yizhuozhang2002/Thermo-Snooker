# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:40:16 2022

@author: yz6621
"""
def change_in_momentum_C(self):
   vper=np.dot(self._v,self._p)/(10-self.R_b)
   return 2*self.m*vper

def pressure(self):
    dp=change_in_momentum_C(self._ball[i])
    dt=sum(time)
    dp/dt/(2*np.pi*10)