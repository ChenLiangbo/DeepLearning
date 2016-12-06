#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np


'''
x_train = np.load('../dataset/kerasSet/x_train.npy')
y_train = np.load('../dataset/kerasSet/y_train.npy')
print "train shape = ",(x_train.shape,y_train.shape)

trainShape = x_train.shape
y_train512 = np.zeros((trainShape[0],512))
for i in xrange(trainShape[0]):
    y_train512[i,y_train[i]] = 1

np.save('../dataset/kerasSet/y_train512.npy',y_train512)

'''
x_test = np.load('../dataset/kerasSet/x_test.npy')
y_test = np.load('../dataset/kerasSet/y_test.npy')
print "test shape = ",(x_test.shape,y_test.shape)

testShape = x_test.shape
y_test512 = np.zeros((testShape[0],512))
for i in xrange(testShape[0]):
    y_test512[i,y_test[i]] = 1

np.save('../dataset/kerasSet/y_test512.npy',y_test512)



y_train512 = np.load('../dataset/kerasSet/y_train512.npy')
print "y_train512.shape = ",y_train512.shape
print y_train512[0,0:20]


y_test512 = np.load('../dataset/kerasSet/y_test512.npy')
print "y_test512.shape = ",y_test512.shape
print y_test512[0,0:20]