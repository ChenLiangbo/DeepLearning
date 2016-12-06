#!usr/bin/env/python 
# -*- coding: utf-8 -*-
import numpy as np
import input_data
from cnnModel import MyCNNmodel

trainNumber = 55000
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
print "x_train.shape = ",x_train.shape
x_train = x_train[0:trainNumber]
y_train = y_train[0:trainNumber]
x_train = x_train.reshape(-1, 28, 28, 1)

print "x_train.shape = ",(x_train.shape,y_train.shape)


myModel = MyCNNmodel()
myModel.iterTimes = 200
myModel.batchSize = 178
myModel.train(x_train,y_train)
import os
import json
content = {"iterTimes":myModel.iterTimes,"batchSize":myModel.batchSize,"trainNumber":trainNumber}
command = 'python predict.py'
os.system(command)
fp = open('./file/mnist_result.txt','ab')
content = json.dumps(content)
fp.write(content)
fp.write('\r\n')
fp.close()
