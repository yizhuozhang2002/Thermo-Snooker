<<<<<<< HEAD
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
<<<<<<< HEAD:generateballs.py
print(posb)
=======
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:19:12 2022

@author: sophia
"""
import numpy as np
import balls as balls
import simulationauto as sim
# random generate
a = [0, 1]
b = [1, 0]
n = 25  # number of balls


def distance_check(a, b):

    distance = abs(np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2))
    if distance < 2:  # 2*radius
        return False
    else:
        return True


posb = []
rbs = np.array([i for i in range(10)])
for i in range(n):
    rb = np.random.choice(rbs, p=rbs/sum(rbs))*1
    the = np.random.uniform(-np.pi, np.pi)
    # if the=0 and the=np.pi and the=-np.pi and the=np.pi/2 and the=-np.pi/2:
    # return the=1

    x = np.fix(10*rb*np.cos(the))/10
    y = np.fix(rb*np.sin(the)*10)/10

    posb.append([x, y])
    print(posb)
    for j in range(n):
        if i > j:

#            distance_check(posb[i], posb[j])
#            print(i, j)
#            print(distance_check(posb[i], posb[j]))
            while distance_check(posb[i], posb[j]) is False:
                rb = np.random.choice(rbs, p=rbs/sum(rbs))*1
                the = np.random.uniform(-np.pi, np.pi)
        # if the=0 and the=np.pi and the=-np.pi and the=np.pi/2 and the=-np.pi/2:
        # return the=1

                x_new = np.fix(10*rb*np.cos(the))/10
                y_new = np.fix(rb*np.sin(the)*10)/10
                # then rerun the loop until distance check is true for all pairs
                posb[i] = [x_new, y_new]
                # recheck the new generated one with all pair until it returns true
                print(posb[i])
                for a in range(i):
                    for b in range(j):
                        if a > b:
                            distance_check(posb[a], posb[b])
                            if distance_check(posb[a], posb[b]) is True:
                                posb[i] = [x_new, y_new]
                                break

                # for j in range(n):
                #     if i > j:

                #         distance_check(posb[i], posb[j])
                #         if False:

                #         if True:
                #             break
                # if distance_check(posb[i], posb[j]) is False:
                    # break


print(posb)

# %%
# systematically
n = 30
# a = np.linspace(-10, 10, num=round(np.sqrt(n))
#a = np.linspace(-10, 10, num=100)

a = np.linspace(-6, 6, num=round(np.sqrt(30)))  # initialisation of balls
#        positions = []
#        for j in a:
#            for i in a:
#                positioni = [j, i]
#                positions.append(positioni)
# n is number of balls
posbb = []
for i in a:
    for j in a:
        posbi = [j, i]
        posbb.append(posbi)
>>>>>>> 8fc9227 (Initial commit of Thermo-Snooker project)
=======
print(posb)
>>>>>>> fdbda5df5f56fe87017a42a6d2769c99f185cc8e:gener11ateballs.py
