#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import input_data
from CNNPaperModel import MyCNNmodel

trainNumber = 55000
testNumber  = 200
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
print "x_train.shape = ",x_train.shape
x_train = x_train[0:trainNumber]
y_train = y_train[0:trainNumber]
x_test = x_test[0:testNumber]
y_test = y_test[0:testNumber]
x_train = x_train.reshape(-1, 28, 28, 1)
x_test  = x_test.reshape(-1, 28, 28, 1)
print "x_train.shape = ",x_train.shape

myModel = MyCNNmodel()
myModel.iterTimes = 200
myModel.batchSize = 178
myModel.learningrate  = 0.001
myModel.train(x_train,y_train)
print "CNN model trained successfully! "

import os
os.system('python predict.py')
