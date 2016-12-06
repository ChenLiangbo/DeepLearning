#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np

x1 = np.load('x_train32.npy')
y1 = np.load('y_train32.npy')

print "x1.shape = ",(x1.shape,y1.shape)

x2 = np.load('x_batch_1_32.npy')
y2 = np.load('y_batch_1_32.npy')
print "x2.shape = ",(x2.shape,y2.shape)

outdir = './tensordata/'

x = np.vstack([x1,x2])
y = np.vstack([y1,y2])
print "x.shape = ",(x.shape,y.shape)
np.save(outdir + 'x_train',x)
np.save(outdir + 'y_train',y)