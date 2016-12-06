#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np

x_train = np.load('../dataset/SampleDataSingleFiles/1.0train_images.npy')
y_train = np.load('../dataset/SampleDataSingleFiles/1.0train_labels.npy')
print "x_train.shape = ",x_train.shape
print "y_train.shape = ",y_train.shape
x_test = np.load('../dataset/SampleDataSingleFiles/1.0test_images.npy')
y_test = np.load('../dataset/SampleDataSingleFiles/1.0test_labels.npy')

print "x_test.shape = ",x_test.shape
print "y_test.shape = ",y_test.shape

print "y_test[0:10] = " ,y_test[10000:100010]

alist = y_train.tolist()
print "alist = ",len(alist)
aset = set(alist)
print "aset = ",len(aset)