<<<<<<< HEAD
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
<<<<<<< HEAD:pressure.py
=======
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
>>>>>>> 8fc9227 (Initial commit of Thermo-Snooker project)
    dp/dt/(2*np.pi*10)
=======
    dp/dt/(2*np.pi*10)
>>>>>>> fdbda5df5f56fe87017a42a6d2769c99f185cc8e:pressure11.py
