# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 23:15:21 2022

@author: yz6621
"""

'''
test2
'''
import simulation as sims

a = 5
press = []
temperature = []
for j in range(1, 5):
    R_b = 10**j*sims.bohr
    for i in range(a):
        temperaturei = 50+298*i
        l = sims.Simulation(R_b=R_b, posbb=gb.posbb, temperature=temperaturei)
        l.run(1000, False)
        pressi = round(l.total_pressure(), 20)
        temperature.append(temperaturei)
        # print(sum(l.time))
        press.append(pressi)
        print('runtime', i)
x_r1 = temperature
y_r1 = press
# %%
'''
test if the simulation worked as expected, change T, oberserv P
change Volumn of container, observe PT relation
Change Number of particles inside of the container, observe PT relation
'''
l = Simulation(temperature=298, R_b=3000*bohr)
l.total_pressure()
#%%
l = Simulation(temperature=298, R_c=3000*bohr)
# %%
# tes volume of container and pressure
l = Simulation(temperature=298, R_b=3000*bohr)

numrun = 3000
l.run(numrun, True)
