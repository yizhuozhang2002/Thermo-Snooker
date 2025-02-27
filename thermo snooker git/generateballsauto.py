# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:19:12 2022

@author: sophia
"""
import numpy as np
import balls as balls

# systematically

n2 = n1*2
n = 100
time = []
# a = np.linspace(-10, 10, num=round(np.sqrt(n))
#a = np.linspace(-10, 10, num=100)
bohr = 5.29e-11
r = 1000*bohr
# r=balls.r
# initialisation of balls
a = np.linspace(-2000*r/1.5, 2000*r/1.5, num=round(np.sqrt(n)))
positions = []
for j in a:
    for i in a:
        positioni = [j, i]
        positions.append(positioni)
 # n is number of balls
posbb = []
for i in a:
    for j in a:
        posbi = [j, i]
        posbb.append(posbi)
for i in n:

    d = np.random.choice(posbb)
    posb.append(posbb[d])
    posbb.remove(d)
