#!usr/bin/env/python 
# -*- coding: utf-8 -*

import numpy as np


x = np.random.random((100,1))

from matplotlib import pyplot as plt
plt.plot(x,'ro')
plt.plot(x,'r-')
plt.grid(True)
plt.show()