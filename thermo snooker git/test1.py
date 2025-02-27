# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 23:57:07 2022

@author: sophia
"""

import balls as balls
import simulation as sim
"""
task 10

KE=3/2 kT
conservation of energy and momentum

since v = np.sqrt(2RT/m)
v to 2v, ke to 4ke, t to np.sqrt(2)T,
pressure PV=nRT, p to 2p
"""
'''
task11

v to 2v,
pressure at v is
pressure at 2v is 18.108957597496378



'''
KEall = []
KEi = self._ball[i].KE(self.m, self.vb[i])
KEall.append(KEi)
# update KE after each collision
# for each run, update KEall0 to KEall run number
# calculate difference.

pi = self._ball[i].momentum(self.m, self.vb[i])


def Temperature():
    k = 1.38*10 ^ -23
    temp = 2*self.KE_all/(3*k)
    return temp
#%%
    
vx=np.random.uniform(-maxv,maxv)
#original code
#https://stackoverflow.com/questions/49855569/how-to-generate-random-numbers-to-satisfy-a-specific-mean-and-median-in-python
def gen_random(): 
    arr1 = np.random.randint(2, 7, 99).astype(np.float)
    arr2 = np.random.randint(7, 40, 99).astype(np.float)
    mid = [6, 7]
    i = ((np.sum(arr1 + arr2) + 13) - (12 * 200)) / 40
    args = np.argsort(arr2)
    arr2[args[-40:]] -= i
    return np.concatenate((arr1, mid, arr2))
#%%
def gen_random(): 
    arr1 = np.random.randint(2, 7, 99).astype(np.float)
    arr2 = np.random.randint(7, 40, 99).astype(np.float)
    mid = [3121, 3122]
    i = ((np.sum(arr1 + arr2) + 6243) - (3121 * 200)) / 40
    args = np.argsort(arr2)
    arr2[args[-40:]] -= i
    return np.concatenate((arr1, mid, arr2))
a=gen_random()
