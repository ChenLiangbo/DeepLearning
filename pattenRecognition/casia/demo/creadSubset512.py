#!usr/bin/env/python 
# -*- coding: utf-8 -*-

import numpy as np
from keras.utils import np_utils

sublength = 32
tfsize = (32,32,1)
thsize = (1,32,32)

x_test = np.load('../dataset/SampleDataSingleFiles/1.0test_images.npy')
y_test = np.load('../dataset/SampleDataSingleFiles/1.0test_labels.npy')

shape = x_test.shape
image_list = []
label_list = []
for i in xrange(shape[0]):
    if y_test[i] < sublength:
        x = x_test[i,:].reshape(1,32,32)
        image_list.append(x)
        label_list.append(y_test[i])

imageSet = np.array(image_list)
labelSet = np.array(label_list)
labelSet = np_utils.to_categorical(labelSet, sublength)
print "sub data set shape = ",(imageSet.shape,labelSet.shape)

np.save('../dataset/kerasSet/x_test' + str(sublength),imageSet)
np.save('../dataset/kerasSet/y_test' + str(sublength),labelSet)


x_train = np.load('../dataset/SampleDataSingleFiles/1.0train_images.npy')
y_train = np.load('../dataset/SampleDataSingleFiles/1.0train_labels.npy')

shape = x_train.shape
image_list = []
label_list = []
for i in xrange(shape[0]):
    if y_train[i] < sublength:
        x = x_train[i,:].reshape(1,32,32)
        image_list.append(x)
        label_list.append(y_train[i])

imageSet = np.array(image_list)
labelSet = np.array(label_list)
labelSet = np_utils.to_categorical(labelSet, sublength)
print "sub data set shape = ",(imageSet.shape,labelSet.shape)

np.save('../dataset/kerasSet/x_train' + str(sublength),imageSet)
np.save('../dataset/kerasSet/y_train' + str(sublength),labelSet)

x = np.load('../dataset/SampleData/HWDB1.1tst_images.npy')
y = np.load('../dataset/SampleData/HWDB1.1tst_labels.npy')

x_batch_1 = []
y_batch_1 = []
for i in xrange(x.shape[0]):
    if y[i] < sublength:
        xbatch = x[i,:].reshape(1,32,32)
        x_batch_1.append(xbatch)
        y_batch_1.append(y[i])

x_batch_1 = np.array(x_batch_1)
y_batch_1 = np.array(y_batch_1)
y_batch_1 = np_utils.to_categorical(y_batch_1,sublength)
print "x_batch shape = ",(x_batch_1.shape,y_batch_1.shape)
np.save('../dataset/kerasSet/x_batch_1_' + str(sublength),x_batch_1)
np.save('../dataset/kerasSet/y_batch_1_' + str(sublength),y_batch_1)
