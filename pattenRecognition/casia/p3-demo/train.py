#!usr/bin/env/python 
# -*- coding: utf-8 -*-

from CNNPaperModel import MyCNNmodel
datadir = './tensordata/'

x_train = np.load(datadir + 'x_train.npy')
y_train = np.load(datadir + 'y_train.npy')
print "x_train.shape = ",(x_train.shape,y_train.shape)

myModel = MyCNNmodel()
myModel.iterTimes = 200
myModel.batchSize = 128
myModel.learningrate  = 0.001
myModel.train(x_train,y_train)
print "CNN model trained successfully! "

import os
os.system('python predict.py')
