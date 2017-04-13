#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import pandas as pd

df = pd.read_csv("../data/20101.csv")
print("head = ",df.head())
valve_pressure1 = df['valve_pressure1']
vmax = valve_pressure1.max()
vmin = valve_pressure1.min()
valve_pressure1 = (valve_pressure1 - vmin)/(vmax - vmin)
# valve_pressure1 = list(valve_pressure1)

from matplotlib import pyplot as plt
plt.plot(valve_pressure1[0:200],'ro')
plt.plot(valve_pressure1[0:200],'r-')
plt.grid(True)
plt.show()